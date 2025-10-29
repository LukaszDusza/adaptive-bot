"""
Trade data endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from app.models import Trade, ActivePosition, TradeNoteUpdate
from app.services.trade_parser import TradeParser

router = APIRouter(prefix="/trades", tags=["Trades"])

# Initialize service
trade_parser = TradeParser()


@router.get("/", response_model=List[Trade])
async def get_all_trades(
    limit: Optional[int] = Query(100, description="Max number of trades to return"),
    ticker: Optional[str] = Query(None, description="Filter by ticker")
):
    """
    Get all trades with optional filtering.

    Args:
        limit: Maximum number of trades (default: 100)
        ticker: Filter by specific ticker (optional)

    Returns:
        List of Trade objects sorted by start_time descending
    """
    if ticker:
        trades = trade_parser.get_trades_by_ticker(ticker)
    else:
        trades = trade_parser.get_all_trades(limit=limit)

    return trades


@router.get("/active", response_model=List[Trade])
async def get_active_trades():
    """
    Get all currently active trades (positions).

    Returns only trades that are still open (no end_time).
    """
    active_trades = trade_parser.get_active_trades()
    return active_trades


@router.get("/{trade_id}", response_model=Trade)
async def get_trade_by_id(trade_id: str):
    """
    Get single trade by ID.

    Args:
        trade_id: Trade identifier (e.g., "20231028_120000_SOLUSDT_Long")

    Raises:
        404: Trade not found
    """
    trade = trade_parser.get_trade_by_id(trade_id)

    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade not found: {trade_id}")

    return trade


@router.patch("/{trade_id}/note", response_model=dict)
async def update_trade_note(trade_id: str, note_update: TradeNoteUpdate):
    """
    Add or update note for a trade.

    Useful for manual annotations like:
    - "Test nowych DCA levels"
    - "Market news impact"
    - "Manual intervention"

    Args:
        trade_id: Trade identifier
        note_update: Note text (max 500 chars)

    Returns:
        Success message
    """
    success = trade_parser.update_trade_note(trade_id, note_update.note)

    if not success:
        raise HTTPException(status_code=404, detail=f"Trade not found: {trade_id}")

    return {"message": "Note updated successfully", "trade_id": trade_id}


@router.get("/recent/{count}", response_model=List[Trade])
async def get_recent_trades(count: int = 10):
    """
    Get N most recent trades.

    Args:
        count: Number of recent trades (default: 10)

    Returns:
        List of most recent trades
    """
    trades = trade_parser.get_all_trades(limit=count)
    return trades
