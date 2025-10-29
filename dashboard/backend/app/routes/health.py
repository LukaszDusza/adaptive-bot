"""
Health check endpoint
"""
from fastapi import APIRouter
from datetime import datetime
import time

from app.models import HealthResponse

router = APIRouter(tags=["Health"])

# Track server start time
SERVER_START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    API health check endpoint.

    Returns server status and uptime.
    """
    uptime = time.time() - SERVER_START_TIME

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime_seconds=uptime
    )
