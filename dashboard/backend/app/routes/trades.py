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
    Get all currently active trades (positions) LIVE from BYBIT API.

    SOURCE OF TRUTH: Real-time data from Bybit exchange.
    Returns actual open positions with current prices and PnL.
    """
    if not bybit_service or not bybit_service.is_available():
        logger.warning("⚠️  Bybit service not available - falling back to local state files")
        return trade_parser.get_active_trades()

    try:
        # Get LIVE positions from Bybit API
        positions = bybit_service.get_active_positions_from_bybit()

        if not positions:
            logger.info("No active positions found on Bybit")
            return []

        # Convert Bybit position format to Trade format
        trades = []
        for pos in positions:
            # Calculate current PnL
            entry_price = pos.get('entry_price', 0)
            quantity = pos.get('quantity', 0)
            unrealized_pnl = pos.get('unrealized_pnl', 0)

            # Calculate PnL percentage
            pnl_percent = 0.0
            if entry_price > 0 and quantity > 0:
                position_value = entry_price * quantity
                pnl_percent = (unrealized_pnl / position_value) * 100

            # Map side: 'Buy' -> 'Long', 'Sell' -> 'Short'
            side = 'Long' if pos.get('side') == 'Buy' else 'Short'

            trade = Trade(
                trade_id=f"{pos.get('ticker')}_{pos.get('side')}",
                ticker=pos.get('ticker', ''),
                side=TradeSide(side),
                start_time=pos.get('created_time', datetime.now().isoformat()),
                end_time=None,
                entry_price=entry_price,
                exit_price=None,
                quantity=quantity,
                leverage=pos.get('leverage', 1),
                initial_sl=pos.get('stop_loss'),
                initial_tp=pos.get('take_profit'),
                current_sl=pos.get('stop_loss'),
                current_tp=pos.get('take_profit'),
                summary={
                    'pnl': unrealized_pnl,
                    'pnl_percent': pnl_percent,
                    'exit_reason': '',
                    'duration_seconds': 0,
                    'max_favorable_excursion': None,
                    'max_adverse_excursion': None,
                    'fees_paid': 0
                },
                events=[],
                indicators=None,
                is_active=True,
                notes=f"LIVE from Bybit | Created: {pos.get('created_time', 'N/A')}"
            )
            trades.append(trade)

        logger.info(f"✓ Retrieved {len(trades)} LIVE positions from Bybit")
        return trades

    except Exception as e:
        logger.error(f"Failed to get active positions from Bybit: {e}")
        logger.warning("Falling back to local state files")
        return trade_parser.get_active_trades()


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
        # Get ALL open orders from Bybit (no ticker filter)
        # This will show ALL pending orders including HBAR and any other ticker
        logger.info("No tickers specified - fetching ALL open orders from Bybit")
        ticker_list = None  # None = fetch all open orders

    all_orders = []
    by_ticker = {}

    # Get active positions from Bybit to filter out orphaned conditional orders
    # Orphaned orders = Untriggered SL/TP orders left after position closed
    # Map: ticker -> side (e.g., {'SOLUSDT': 'Buy', 'ETHUSDT': 'Sell'})
    active_positions_map = {}
    try:
        active_positions = bybit_service.get_active_positions_from_bybit()
        active_positions_map = {pos.get('ticker'): pos.get('side') for pos in active_positions}
        logger.info(f"Active positions from Bybit: {active_positions_map}")
    except Exception as e:
        logger.warning(f"Failed to get active positions for filtering: {e}")

    # Fetch orders from Bybit
    if ticker_list is None:
        # Get ALL open orders from Bybit (no filter)
        logger.info("Fetching ALL open orders from Bybit")
        orders_by_ticker = bybit_service.get_all_open_orders()
    else:
        # Get orders for specific tickers
        logger.info(f"Fetching pending orders for tickers: {ticker_list}")
        orders_by_ticker = bybit_service.get_multiple_open_orders(ticker_list)

    # DEBUG: Log raw response from Bybit
    total_raw_orders = sum(len(orders) for orders in orders_by_ticker.values())
    logger.info(f"📊 RAW DATA FROM BYBIT: {len(orders_by_ticker)} tickers, {total_raw_orders} total orders")
    logger.info(f"📊 Tickers in raw response: {list(orders_by_ticker.keys())}")
    for ticker, orders in orders_by_ticker.items():
        logger.info(f"📊 {ticker}: {len(orders)} orders")
        for order in orders:
            logger.info(f"   - Order {order.get('orderId')}: status={order.get('orderStatus')}, type={order.get('orderType')}, side={order.get('side')}")

    # Process orders
    for ticker, order_list in orders_by_ticker.items():
        # Filter criteria:
        # - Always include: New, PartiallyFilled (actual limit orders)
        # - Include Untriggered ONLY if ticker+side match active position (valid SL/TP)
        # - Exclude Untriggered with wrong side or no position (orphaned SL/TP from closed trades)

        for order in order_list:
            order_status = order.get("orderStatus", "")
            order_side = order.get("side", "")  # "Buy" or "Sell"

            # Check if ticker+side match active position
            active_position_side = active_positions_map.get(ticker)
            has_matching_position = (active_position_side == order_side)

            # Filter logic:
            # 1. Always show "New" and "PartiallyFilled" - these are real pending limit orders
            # 2. Show "Untriggered" ONLY if position exists AND side matches - these are active SL/TP
            # 3. Skip "Untriggered" with wrong side or no position - these are orphaned SL/TP from closed trades
            if order_status in {"New", "PartiallyFilled"}:
                should_include = True
                logger.info(f"✅ Including {ticker} {order_side} order (status={order_status})")
            elif order_status == "Untriggered" and has_matching_position:
                should_include = True
                logger.info(f"✅ Including {ticker} {order_side} Untriggered order (matching active position)")
            else:
                should_include = False
                if order_status == "Untriggered":
                    if not active_position_side:
                        logger.info(f"❌ FILTERING {ticker} {order_side} Untriggered: no active position")
                    else:
                        logger.info(f"❌ FILTERING {ticker} {order_side} Untriggered: position side is {active_position_side}, order side is {order_side}")
                else:
                    logger.info(f"❌ FILTERING {ticker} {order_side} order: status={order_status}")

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


@router.get("/closed", response_model=List[Trade])
async def get_closed_trades_from_bybit(limit: Optional[int] = Query(50, description="Max number of trades to return")):
    """
    Get closed trades LIVE from BYBIT API (closed P&L history).

    SOURCE OF TRUTH: Real closed trades from Bybit exchange.
    Returns actual historical trades with real PnL, exit prices, and timestamps.

    Args:
        limit: Maximum number of trades (default: 50, max: 100)

    Returns:
        List of Trade objects from Bybit closed P&L history
    """
    if not bybit_service or not bybit_service.is_available():
        logger.warning("⚠️  Bybit service not available - returning empty list")
        return []

    try:
        # Get closed P&L records from Bybit
        pnl_records = bybit_service.get_closed_pnl_history(limit=min(limit, 100))

        if not pnl_records:
            logger.info("No closed trades found on Bybit")
            return []

        # Convert Bybit closed PnL format to Trade format
        trades = []
        for record in pnl_records:
            # Parse timestamps
            created_time = record.get('createdTime', 0)
            updated_time = record.get('updatedTime', 0)

            # Convert milliseconds to ISO format
            start_time = datetime.fromtimestamp(int(created_time) / 1000).isoformat() if created_time else None
            end_time = datetime.fromtimestamp(int(updated_time) / 1000).isoformat() if updated_time else None

            # Parse PnL and prices
            closed_pnl = float(record.get('closedPnl', 0))
            avg_entry_price = float(record.get('avgEntryPrice', 0))
            avg_exit_price = float(record.get('avgExitPrice', 0))
            closed_size = float(record.get('closedSize', 0))

            # Calculate PnL percentage
            pnl_percent = 0.0
            if avg_entry_price > 0 and closed_size > 0:
                position_value = avg_entry_price * closed_size
                pnl_percent = (closed_pnl / position_value) * 100

            # Determine exit reason from PnL
            exit_reason = ""
            if closed_pnl > 0:
                exit_reason = "TP"  # Take Profit
            elif closed_pnl < 0:
                exit_reason = "SL"  # Stop Loss
            else:
                exit_reason = "Manual"  # Breakeven

            # Map side: In Bybit closed PnL, 'side' refers to the CLOSING transaction
            # Buy = closed by buying = original position was SHORT
            # Sell = closed by selling = original position was LONG
            # So we need to REVERSE the mapping
            side_raw = record.get('side', 'Buy')
            side = 'Short' if side_raw == 'Buy' else 'Long'  # REVERSED!

            # Calculate duration
            duration_seconds = 0
            if created_time and updated_time:
                duration_seconds = (int(updated_time) - int(created_time)) // 1000

            trade = Trade(
                trade_id=f"{record.get('symbol')}_{created_time}",
                ticker=record.get('symbol', ''),
                side=TradeSide(side),
                start_time=start_time,
                end_time=end_time,
                entry_price=avg_entry_price,
                exit_price=avg_exit_price,
                quantity=closed_size,
                leverage=int(record.get('leverage', 1)),
                initial_sl=None,
                initial_tp=None,
                current_sl=None,
                current_tp=None,
                summary={
                    'pnl': closed_pnl,
                    'pnl_percent': pnl_percent,
                    'exit_reason': exit_reason,
                    'duration_seconds': duration_seconds,
                    'max_favorable_excursion': None,
                    'max_adverse_excursion': None,
                    'fees_paid': float(record.get('cumEntryValue', 0)) * 0.00055  # Estimate
                },
                events=[],
                indicators=None,
                is_active=False,
                notes=f"LIVE from Bybit | Closed: {end_time}"
            )
            trades.append(trade)

        logger.info(f"✓ Retrieved {len(trades)} closed trades from Bybit")
        return trades

    except Exception as e:
        logger.error(f"Failed to get closed trades from Bybit: {e}")
        return []


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
