"""
Docker container management endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import List

from app.models import ContainerInfo, BotAction
from app.services.docker_manager import DockerManager

router = APIRouter(prefix="/containers", tags=["Containers"])

# Initialize service
docker_manager = DockerManager()


@router.get("/", response_model=List[ContainerInfo])
async def get_all_containers():
    """
    Get all bot containers (running and stopped).

    Returns:
        List of ContainerInfo with status, uptime, health, etc.
    """
    containers = docker_manager.get_all_containers()
    return containers


@router.get("/{name}", response_model=ContainerInfo)
async def get_container(name: str):
    """
    Get single container by name.

    Args:
        name: Container name (e.g., "bot-syl-sol-dca")

    Raises:
        404: Container not found
    """
    container = docker_manager.get_container(name)

    if not container:
        raise HTTPException(status_code=404, detail=f"Container not found: {name}")

    return container


@router.post("/{name}/start", response_model=dict)
async def start_container(name: str):
    """
    Start a stopped container.

    Args:
        name: Container name

    Returns:
        Success message

    Raises:
        500: Failed to start container
    """
    success = docker_manager.start_container(name)

    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to start container: {name}")

    return {"message": f"Container started successfully", "container": name}


@router.post("/{name}/stop", response_model=dict)
async def stop_container(name: str, timeout: int = 10):
    """
    Stop a running container.

    Args:
        name: Container name
        timeout: Graceful shutdown timeout in seconds (default: 10)

    Returns:
        Success message

    Raises:
        500: Failed to stop container
    """
    success = docker_manager.stop_container(name, timeout=timeout)

    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to stop container: {name}")

    return {"message": f"Container stopped successfully", "container": name}


@router.post("/{name}/restart", response_model=dict)
async def restart_container(name: str, timeout: int = 10):
    """
    Restart a container.

    Args:
        name: Container name
        timeout: Graceful shutdown timeout in seconds (default: 10)

    Returns:
        Success message

    Raises:
        500: Failed to restart container
    """
    success = docker_manager.restart_container(name, timeout=timeout)

    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to restart container: {name}")

    return {"message": f"Container restarted successfully", "container": name}


@router.get("/{name}/logs")
async def get_container_logs(name: str, tail: int = 100):
    """
    Get recent logs from container.

    Args:
        name: Container name
        tail: Number of recent log lines (default: 100)

    Returns:
        Log text as string

    Raises:
        404: Container not found
    """
    # First check if container exists
    container = docker_manager.get_container(name)
    if not container:
        raise HTTPException(status_code=404, detail=f"Container not found: {name}")

    logs = docker_manager.get_container_logs(name, tail=tail)

    return {"container": name, "logs": logs}
