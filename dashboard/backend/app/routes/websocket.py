"""
WebSocket endpoint for real-time updates
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict
import asyncio
import json
import logging
from datetime import datetime

from app.services.trade_parser import TradeParser
from app.services.docker_manager import DockerManager
from app.models import WSMessage

router = APIRouter(tags=["WebSocket"])
logger = logging.getLogger(__name__)

# Active WebSocket connections
active_connections: List[WebSocket] = []

# Services
trade_parser = TradeParser()
docker_manager = DockerManager()


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and register new connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove connection"""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict):
        """Send message to all connected clients"""
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

    async def send_personal(self, websocket: WebSocket, message: Dict):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates.

    Sends periodic updates:
    - Container status changes
    - Active position updates (unrealized PnL)
    - New trade notifications
    - Price updates (if available)

    Message format:
    {
        "event": "container_status" | "position_update" | "new_trade" | "price_update",
        "data": {...},
        "timestamp": "2023-10-28T12:00:00"
    }
    """
    await manager.connect(websocket)

    try:
        # Send initial welcome message
        await manager.send_personal(websocket, {
            "event": "connected",
            "data": {"message": "Connected to dashboard WebSocket"},
            "timestamp": datetime.now().isoformat()
        })

        # Start background task to send periodic updates
        update_task = asyncio.create_task(send_periodic_updates(websocket))

        # Listen for client messages (ping/pong, subscriptions, etc.)
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle client messages
            if message.get("type") == "ping":
                await manager.send_personal(websocket, {
                    "event": "pong",
                    "data": {},
                    "timestamp": datetime.now().isoformat()
                })

            elif message.get("type") == "subscribe":
                # Handle subscription to specific events
                logger.info(f"Client subscribed to: {message.get('events')}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
    finally:
        # Cancel background task
        if 'update_task' in locals():
            update_task.cancel()


async def send_periodic_updates(websocket: WebSocket):
    """
    Background task to send periodic updates to client.

    Sends updates every 5 seconds:
    - Container status
    - Active positions (if any)
    """
    while True:
        try:
            # Send container status update
            containers = docker_manager.get_all_containers()
            await manager.send_personal(websocket, {
                "event": "container_status_update",
                "data": {
                    "containers": [c.model_dump() for c in containers],
                    "count": len(containers),
                    "running": len([c for c in containers if c.status.value == "running"])
                },
                "timestamp": datetime.now().isoformat()
            })

            # Send active trades update
            active_trades = trade_parser.get_active_trades()
            if active_trades:
                await manager.send_personal(websocket, {
                    "event": "active_trades_update",
                    "data": {
                        "trades": [t.model_dump(mode='json') for t in active_trades],
                        "count": len(active_trades)
                    },
                    "timestamp": datetime.now().isoformat()
                })

            # Wait 5 seconds before next update
            await asyncio.sleep(5)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic updates: {e}")
            await asyncio.sleep(5)


# Utility function to broadcast events from other parts of the app
async def broadcast_event(event: str, data: Dict):
    """
    Broadcast event to all connected clients.

    Can be called from anywhere in the app to push updates.

    Args:
        event: Event type (e.g., "new_trade", "container_restart")
        data: Event payload
    """
    message = {
        "event": event,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast(message)
