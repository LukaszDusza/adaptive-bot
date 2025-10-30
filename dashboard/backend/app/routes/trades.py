"""
Trade data endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import httpx
import logging
from datetime import datetime

from app.models import Trade, ActivePosition, TradeNoteUpdate, PendingOrder, PendingOrdersResponse, TradeSide
from app.services.trade_parser import TradeParser
from app.services.bybit_service import BybitService
from app.config import settings

router = APIRouter(prefix="/trades", tags=["Trades"])
logger = logging.getLogger(__name__)

# Initialize Bybit service first (will be None if credentials not configured)
try:
    bybit_service = BybitService()
except Exception as e:
    logger.warning(f"Bybit service not available: {e}")
    bybit_service = None

# Initialize services
# Use JSON-only mode to ensure latest trades are visible
# (SQLite database sync is not implemented yet)
# Pass bybit_service for Bybit verification of state files
trade_parser = TradeParser(use_database=False, bybit_service=bybit_service)


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


@router.get("/pending-orders", response_model=PendingOrdersResponse)
async def get_pending_orders(tickers: Optional[str] = Query(None, description="Comma-separated list of tickers (e.g., 'SOLUSDT,DOGEUSDT')")):
    """
    Get all pending (unfilled) limit orders from Bybit.

    This endpoint fetches open orders directly from Bybit API for the specified tickers.
    Useful for monitoring limit orders placed by bots that are waiting for execution.

    Args:
        tickers: Optional comma-separated list of tickers. If None, tries to get from active trades.

    Returns:
        PendingOrdersResponse with list of pending orders

    Note:
        Requires BYBIT_API_KEY and BYBIT_API_SECRET to be configured in .env file.
    """
    # Check if Bybit service is available
    if not bybit_service or not bybit_service.is_available():
        logger.warning("Bybit service not configured - returning empty response")
        return PendingOrdersResponse(
            orders=[],
            total_count=0,
            by_ticker={},
            last_updated=datetime.now()
        )

    # Determine which tickers to check
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
    else:
        # Get tickers from:
        # 1. Active trades (actual positions)
        # 2. All bot_state files (bots monitoring tickers, even without active positions)
        active_trades = trade_parser.get_active_trades()
        ticker_set = set([trade.ticker for trade in active_trades])

        # Also include tickers from all state files (even those without active positions)
        # This ensures we show pending orders for tickers being monitored by bots
        import re
        from pathlib import Path
        state_dir = Path(settings.BOT_STATE_DIR)
        if state_dir.exists():
            for state_file in state_dir.glob("*_state.json"):
                # Extract ticker from filename: TICKER_15m_plus_..._state.json
                match = re.match(r'(.+?)_\d+[mhd]_plus_', state_file.stem)
                if match:
                    ticker_set.add(match.group(1))

        ticker_list = list(ticker_set)

        if not ticker_list:
            # No tickers to monitor, return empty response
            return PendingOrdersResponse(
                orders=[],
                total_count=0,
                by_ticker={},
                last_updated=datetime.now()
            )

    logger.info(f"Fetching pending orders for tickers: {ticker_list}")

    all_orders = []
    by_ticker = {}

    # Get active positions to filter out orphaned conditional orders
    # Orphaned orders = Untriggered SL/TP orders left after position closed
    active_positions_tickers = set()
    try:
        active_trades = trade_parser.get_active_trades()
        active_positions_tickers = {trade.ticker for trade in active_trades}
        logger.info(f"Active position tickers: {active_positions_tickers}")
    except Exception as e:
        logger.warning(f"Failed to get active trades for filtering: {e}")

    # Fetch orders for each ticker using BybitService
    orders_by_ticker = bybit_service.get_multiple_open_orders(ticker_list)

    # Process orders
    for ticker, order_list in orders_by_ticker.items():
        # Filter criteria:
        # - Always include: New, PartiallyFilled (actual limit orders)
        # - Include Untriggered ONLY if ticker has active position (valid SL/TP)
        # - Exclude Untriggered without position (orphaned SL/TP from closed trades)
        has_active_position = ticker in active_positions_tickers

        for order in order_list:
            order_status = order.get("orderStatus", "")

            # Filter logic:
            # 1. Always show "New" and "PartiallyFilled" - these are real pending limit orders
            # 2. Show "Untriggered" ONLY if position exists - these are active SL/TP
            # 3. Skip "Untriggered" without position - these are orphaned SL/TP from closed trades
            if order_status in {"New", "PartiallyFilled"}:
                should_include = True
            elif order_status == "Untriggered" and has_active_position:
                should_include = True
            else:
                should_include = False
                if order_status == "Untriggered" and not has_active_position:
                    logger.debug(f"Filtering orphaned Untriggered order for {ticker} (no active position)")

            if should_include:
                try:
                    # Parse stop loss and take profit
                    stop_loss_str = order.get("stopLoss", "")
                    take_profit_str = order.get("takeProfit", "")

                    stop_loss = None
                    if stop_loss_str and stop_loss_str != "" and stop_loss_str != "0":
                        try:
                            stop_loss = float(stop_loss_str)
                        except (ValueError, TypeError):
                            pass

                    take_profit = None
                    if take_profit_str and take_profit_str != "" and take_profit_str != "0":
                        try:
                            take_profit = float(take_profit_str)
                        except (ValueError, TypeError):
                            pass

                    pending_order = PendingOrder(
                        order_id=order.get("orderId", ""),
                        ticker=order.get("symbol", ticker),
                        side=TradeSide.LONG if order.get("side") == "Buy" else TradeSide.SHORT,
                        order_type=order.get("orderType", "Limit"),
                        price=float(order.get("price", 0)),
                        quantity=float(order.get("qty", 0)),
                        filled_quantity=float(order.get("cumExecQty", 0)),
                        status=order.get("orderStatus", "Unknown"),
                        created_at=datetime.fromtimestamp(int(order.get("createdTime", 0)) / 1000),
                        time_in_force=order.get("timeInForce", "GTC"),
                        reduce_only=order.get("reduceOnly", False),
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    all_orders.append(pending_order)
                except Exception as e:
                    logger.warning(f"Failed to parse order for {ticker}: {e}")
                    continue

        by_ticker[ticker] = len([o for o in all_orders if o.ticker == ticker])
        logger.info(f"✓ Found {by_ticker[ticker]} pending orders for {ticker}")

    logger.info(f"Total pending orders found: {len(all_orders)}")

    return PendingOrdersResponse(
        orders=all_orders,
        total_count=len(all_orders),
        by_ticker=by_ticker,
        last_updated=datetime.now()
    )


@router.delete("/pending-orders/{ticker}/{order_id}")
async def cancel_pending_order(ticker: str, order_id: str):
    """
    Cancel a specific pending order.

    Args:
        ticker: Trading pair (e.g., SOLUSDT)
        order_id: Order ID to cancel

    Returns:
        Success message

    Note:
        Requires BYBIT_API_KEY and BYBIT_API_SECRET to be configured in .env file.
    """
    if not bybit_service or not bybit_service.is_available():
        raise HTTPException(
            status_code=503,
            detail="Bybit service not configured"
        )

    success = bybit_service.cancel_order(ticker.upper(), order_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to cancel order {order_id}"
        )

    return {
        "message": "Order cancelled successfully",
        "ticker": ticker,
        "order_id": order_id
    }


@router.delete("/pending-orders/{ticker}")
async def cancel_all_pending_orders(ticker: str):
    """
    Cancel all pending orders for a ticker.

    Args:
        ticker: Trading pair (e.g., SOLUSDT)

    Returns:
        Success message

    Note:
        Requires BYBIT_API_KEY and BYBIT_API_SECRET to be configured in .env file.
    """
    if not bybit_service or not bybit_service.is_available():
        raise HTTPException(
            status_code=503,
            detail="Bybit service not configured"
        )

    success = bybit_service.cancel_all_orders(ticker.upper())

    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to cancel orders for {ticker}"
        )

    return {
        "message": "All orders cancelled successfully",
        "ticker": ticker
    }


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
