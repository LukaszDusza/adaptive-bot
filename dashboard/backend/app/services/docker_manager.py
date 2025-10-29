"""
Docker container management service
"""
import docker
from docker.errors import DockerException, NotFound, APIError
from typing import List, Dict, Optional
import logging
import re
import os
from pathlib import Path

from app.config import settings
from app.models import ContainerInfo, ContainerStatus

logger = logging.getLogger(__name__)


class DockerManager:
    """Manages Docker containers for trading bots"""

    def __init__(self):
        """
        Initialize Docker client.

        Tries multiple socket locations:
        1. DOCKER_HOST environment variable
        2. macOS Docker Desktop: ~/.docker/run/docker.sock
        3. Standard Linux: /var/run/docker.sock
        """
        self.client = None

        # Try Docker Desktop socket on macOS
        macos_socket = Path.home() / ".docker" / "run" / "docker.sock"
        if macos_socket.exists():
            try:
                self.client = docker.DockerClient(base_url=f"unix://{macos_socket}")
                # Test connection
                self.client.ping()
                logger.info(f"✓ Docker client connected via {macos_socket}")
                return
            except DockerException as e:
                logger.warning(f"Failed to connect via {macos_socket}: {e}")

        # Try standard from_env (uses DOCKER_HOST or /var/run/docker.sock)
        try:
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
            logger.info("✓ Docker client connected via docker.from_env()")
            return
        except DockerException as e:
            logger.error(f"Failed to connect to Docker: {e}")
            self.client = None

    def get_all_containers(self) -> List[ContainerInfo]:
        """Get all bot containers (running and stopped)"""
        if not self.client:
            return []

        containers = []
        try:
            # Get all containers with names starting with 'bot-' or in adaptive-bot project
            all_containers = self.client.containers.list(all=True)

            for container in all_containers:
                # Filter bot containers
                if self._is_bot_container(container):
                    info = self._parse_container_info(container)
                    if info:
                        containers.append(info)

        except Exception as e:
            logger.error(f"Error listing containers: {e}")

        return containers

    def get_container(self, name: str) -> Optional[ContainerInfo]:
        """Get single container by name"""
        if not self.client:
            return None

        try:
            container = self.client.containers.get(name)
            return self._parse_container_info(container)
        except NotFound:
            logger.warning(f"Container not found: {name}")
            return None
        except Exception as e:
            logger.error(f"Error getting container {name}: {e}")
            return None

    def start_container(self, name: str) -> bool:
        """Start a stopped container"""
        if not self.client:
            return False

        try:
            container = self.client.containers.get(name)
            container.start()
            logger.info(f"✓ Started container: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start container {name}: {e}")
            return False

    def stop_container(self, name: str, timeout: int = 10) -> bool:
        """Stop a running container"""
        if not self.client:
            return False

        try:
            container = self.client.containers.get(name)
            container.stop(timeout=timeout)
            logger.info(f"✓ Stopped container: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop container {name}: {e}")
            return False

    def restart_container(self, name: str, timeout: int = 10) -> bool:
        """Restart a container"""
        if not self.client:
            return False

        try:
            container = self.client.containers.get(name)
            container.restart(timeout=timeout)
            logger.info(f"✓ Restarted container: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to restart container {name}: {e}")
            return False

    def get_container_logs(self, name: str, tail: int = 100) -> str:
        """Get recent logs from container"""
        if not self.client:
            return ""

        try:
            container = self.client.containers.get(name)
            logs = container.logs(tail=tail, timestamps=True)
            return logs.decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to get logs for {name}: {e}")
            return ""

    def _is_bot_container(self, container) -> bool:
        """Check if container is a bot container"""
        name = container.name
        labels = container.labels

        # Check if it's from our docker-compose project
        if labels.get("com.docker.compose.project") == "price_action":
            return True

        # Or if name starts with bot- or contains bot
        if name.startswith("bot-") or name.startswith("syl-") or name.startswith("luk-"):
            return True

        return False

    def _parse_container_info(self, container) -> Optional[ContainerInfo]:
        """Parse Docker container object into ContainerInfo model"""
        try:
            name = container.name
            status = self._map_container_status(container.status)

            # Extract ticker, account, strategy from name
            # Expected formats:
            #   bot-syl-sol-dca -> account=syl, ticker=SOL, strategy=dca
            #   syl-sol-dca -> account=syl, ticker=SOL, strategy=dca
            ticker, account, strategy = self._parse_container_name(name)

            # Get uptime
            uptime = None
            if status == ContainerStatus.RUNNING:
                started_at = container.attrs.get("State", {}).get("StartedAt")
                if started_at:
                    from dateutil import parser as date_parser
                    from datetime import datetime
                    start_time = date_parser.parse(started_at)
                    uptime = (datetime.now(start_time.tzinfo) - start_time).total_seconds()

            # Get health status
            health_status = container.attrs.get("State", {}).get("Health", {}).get("Status")

            # Get restart count
            restart_count = container.attrs.get("RestartCount", 0)

            return ContainerInfo(
                name=name,
                container_id=container.id[:12],  # Short ID
                status=status,
                ticker=ticker,
                account=account,
                strategy=strategy,
                uptime_seconds=uptime,
                health_status=health_status,
                restart_count=restart_count
            )

        except Exception as e:
            logger.error(f"Error parsing container info: {e}")
            return None

    def _map_container_status(self, docker_status: str) -> ContainerStatus:
        """Map Docker status string to our enum"""
        status_map = {
            "running": ContainerStatus.RUNNING,
            "exited": ContainerStatus.EXITED,
            "paused": ContainerStatus.PAUSED,
            "restarting": ContainerStatus.RESTARTING,
            "dead": ContainerStatus.DEAD,
            "created": ContainerStatus.STOPPED,
        }
        return status_map.get(docker_status.lower(), ContainerStatus.STOPPED)

    def _parse_container_name(self, name: str) -> tuple[str, str, str]:
        """
        Parse container name to extract ticker, account, strategy.

        Examples:
            bot-syl-sol-dca -> (SOLUSDT, syl, dca)
            syl-doge-limit -> (DOGEUSDT, syl, limit)
            bot-luk-pepe-partial-tp -> (PEPEUSDT, luk, partial-tp)
        """
        # Remove 'bot-' prefix if present
        clean_name = name.replace("bot-", "")

        # Split by hyphen
        parts = clean_name.split("-")

        if len(parts) >= 3:
            account = parts[0]  # syl, luk
            ticker_short = parts[1].upper()  # SOL, DOGE, PEPE
            strategy = "-".join(parts[2:])  # dca, limit, partial-tp

            # Map short ticker to full symbol
            ticker_map = {
                "SOL": "SOLUSDT",
                "DOGE": "DOGEUSDT",
                "PEPE": "PEPEUSDT",
                "LINK": "LINKUSDT",
                "BTC": "BTCUSDT",
                "ETH": "ETHUSDT",
            }
            ticker = ticker_map.get(ticker_short, f"{ticker_short}USDT")

            return ticker, account, strategy

        return "UNKNOWN", "UNKNOWN", "UNKNOWN"
