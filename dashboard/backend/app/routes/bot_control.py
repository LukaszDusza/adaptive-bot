"""
Bot control endpoints - pause/resume, config updates, emergency actions
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Optional
import logging

from app.models import (
    BotConfigUpdate, TradingPauseRequest, EmergencyCloseRequest
)
from app.services.bot_state_manager import BotStateManager
from app.services.docker_manager import DockerManager

router = APIRouter(prefix="/bot-control", tags=["Bot Control"])
logger = logging.getLogger(__name__)

# Initialize services
bot_state_manager = BotStateManager()
docker_manager = DockerManager()

# Initialize Bybit service for position control
try:
    from app.services.bybit_service import BybitService
    bybit_service = BybitService()
except Exception as e:
    bybit_service = None
    logger.warning(f"BybitService not available: {e}")


@router.get("/pause-state", response_model=Dict[str, bool])
async def get_pause_state():
    """
    Get current pause state for all tickers.

    Returns:
        Dict mapping ticker -> is_paused (bool)
    """
    pause_state = bot_state_manager.get_pause_state()
    return pause_state


@router.post("/pause", response_model=dict)
async def pause_trading(request: TradingPauseRequest):
    """
    Pause or resume trading.

    This sets a flag that bots should check before taking new positions.
    NOTE: Existing positions are NOT closed.

    Args:
        request:
            - paused: True to pause, False to resume
            - tickers: List of tickers to affect, or None for all
            - reason: Optional reason for logging

    Returns:
        Success message with affected tickers
    """
    success = bot_state_manager.set_pause_state(
        tickers=request.tickers,
        paused=request.paused,
        reason=request.reason or ""
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update pause state")

    action = "paused" if request.paused else "resumed"
    affected = "all tickers" if request.tickers is None else ", ".join(request.tickers)

    return {
        "message": f"Trading {action} for {affected}",
        "paused": request.paused,
        "tickers": request.tickers,
        "reason": request.reason
    }


@router.post("/config/{strategy_id}", response_model=dict)
async def update_bot_config(strategy_id: str, config: BotConfigUpdate):
    """
    Update bot configuration on the fly.

    IMPORTANT: Bot must be restarted to pick up changes!

    Args:
        strategy_id: e.g., "SOLUSDT_15m_plus_1h_4h_long"
        config: Configuration updates (prob_threshold, tp_pct, etc.)

    Returns:
        Success message

    Example:
        POST /api/bot-control/config/SOLUSDT_15m_plus_1h_4h_long
        {
            "prob_threshold": 0.75,
            "tp_pct": 0.04
        }
    """
    # Convert to dict, excluding None values
    updates = config.model_dump(exclude_none=True)

    if not updates:
        raise HTTPException(status_code=400, detail="No config updates provided")

    success = bot_state_manager.update_bot_config(strategy_id, updates)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update bot config")

    return {
        "message": f"Config updated for {strategy_id}. Restart bot to apply changes.",
        "strategy_id": strategy_id,
        "updates": updates,
        "restart_required": True
    }


@router.get("/config/{strategy_id}", response_model=dict)
async def get_bot_config(strategy_id: str):
    """
    Get current bot configuration.

    Args:
        strategy_id: e.g., "SOLUSDT_15m_plus_1h_4h_long"

    Returns:
        Bot config dict

    Raises:
        404: Config not found
    """
    config = bot_state_manager.get_bot_config(strategy_id)

    if config is None:
        raise HTTPException(status_code=404, detail=f"Config not found for {strategy_id}")

    return config


@router.post("/emergency-close", response_model=dict)
async def emergency_close_all(request: EmergencyCloseRequest):
    """
    🚨 EMERGENCY: Close all positions immediately.

    This is a PANIC BUTTON - use with caution!

    Process:
    1. Validates confirmation flag
    2. Closes all positions via Bybit API (market orders)
    3. Pauses trading to prevent new entries
    4. Logs emergency action

    Args:
        request:
            - confirmation: MUST be true to execute
            - tickers: List of tickers to close, or None for all
            - reason: Reason for emergency close (for logs)

    Returns:
        Summary of closed positions

    Raises:
        400: Confirmation not provided
        500: Failed to close positions
    """
    if not request.confirmation:
        raise HTTPException(
            status_code=400,
            detail="Emergency close requires confirmation=true"
        )

    logger.warning(f"🚨 EMERGENCY CLOSE REQUESTED: {request.reason}")

    try:
        # TODO: Implement Bybit API integration to close positions
        # For now, this is a placeholder that pauses trading

        # Step 1: Pause trading to prevent new entries
        bot_state_manager.set_pause_state(
            tickers=request.tickers,
            paused=True,
            reason=f"EMERGENCY CLOSE: {request.reason}"
        )

        # Step 2: Would close positions via Bybit API here
        # This requires pybit integration with API keys from .env
        closed_positions = []

        # Placeholder response
        logger.warning("⚠️ Emergency close executed (trading paused). Position closing not yet implemented.")

        return {
            "message": "Emergency action executed",
            "trading_paused": True,
            "positions_closed": len(closed_positions),
            "tickers_affected": request.tickers or "all",
            "reason": request.reason,
            "warning": "⚠️ Position closing via API not yet implemented. Please close manually in Bybit."
        }

    except Exception as e:
        logger.error(f"Emergency close failed: {e}")
        raise HTTPException(status_code=500, detail=f"Emergency close failed: {str(e)}")


@router.post("/restart-bot/{container_name}", response_model=dict)
async def restart_bot_with_new_config(container_name: str):
    """
    Convenience endpoint: Restart bot container after config update.

    Args:
        container_name: Docker container name (e.g., "bot-syl-sol-dca")

    Returns:
        Success message
    """
    success = docker_manager.restart_container(container_name)

    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to restart {container_name}")

    return {
        "message": f"Bot restarted successfully",
        "container": container_name,
        "note": "New configuration should now be active"
    }


@router.post("/close-position/{symbol}", response_model=dict)
async def close_position(symbol: str):
    """
    Close a specific position via Bybit API.

    Args:
        symbol: Trading pair (e.g., SOLUSDT)

    Returns:
        Success message with position details

    Raises:
        503: Bybit service not available
        500: Failed to close position
    """
    if not bybit_service:
        raise HTTPException(
            status_code=503,
            detail="Bybit API not available - check API credentials"
        )

    try:
        logger.info(f"📍 Close position request for {symbol}")
        result = bybit_service.close_position_by_symbol(symbol)
        return result
    except Exception as e:
        logger.error(f"Failed to close position for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to close position: {str(e)}"
        )


@router.post("/set-sl-breakeven/{symbol}", response_model=dict)
async def set_sl_breakeven(symbol: str, side: str, entry_price: float):
    """
    Set stop loss to breakeven (entry price) for a position.

    Args:
        symbol: Trading pair (e.g., SOLUSDT)
        side: Position side ("Long" or "Short")
        entry_price: Entry price to set as stop loss

    Returns:
        Success message with updated SL

    Raises:
        503: Bybit service not available
        500: Failed to update SL
    """
    if not bybit_service:
        raise HTTPException(
            status_code=503,
            detail="Bybit API not available - check API credentials"
        )

    try:
        logger.info(f"📍 Set SL to BE request for {symbol} {side} @ {entry_price}")
        result = bybit_service.set_stop_loss_to_breakeven(symbol, side, entry_price)
        return result
    except Exception as e:
        logger.error(f"Failed to set SL for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update stop loss: {str(e)}"
        )
