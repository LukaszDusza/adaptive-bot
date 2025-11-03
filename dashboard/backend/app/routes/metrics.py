"""
Metrics and analytics endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import List

from app.models import (
    MetricsResponse, TickerMetrics, EquityCurvePoint,
    ExitReasonStats, DrawdownPoint, TradeDurationBin
)
from app.services.trade_parser import TradeParser
from app.services.analytics import AnalyticsService

router = APIRouter(prefix="/metrics", tags=["Metrics"])

# Initialize Bybit service for position verification
try:
    from app.services.bybit_service import BybitService
    bybit_service = BybitService()
except Exception as e:
    bybit_service = None

# Initialize services
# Use JSON-only mode to ensure latest trades are visible (same as trades.py)
trade_parser = TradeParser(use_database=False, bybit_service=bybit_service)
analytics = AnalyticsService()


def _get_all_trades_without_duplicates() -> List:
    """
    Helper: Get all trades (completed + active) without duplicates.

    Combines:
    - All trades from logs (completed + active)
    - Active trades from state files (only those not in logs)

    Returns:
        List of Trade objects without duplicates
    """
    # Get all trades from logs (completed + active)
    all_trades = trade_parser.get_all_trades()

    # Get active trades (includes logs + state files)
    active_trades_all = trade_parser.get_active_trades()

    # Remove duplicates - get trade_ids from all_trades
    existing_trade_ids = {t.trade_id for t in all_trades}

    # Add only active trades that are NOT already in all_trades (from state files)
    unique_active_trades = [
        t for t in active_trades_all
        if t.trade_id not in existing_trade_ids
    ]

    # Combine without duplicates
    return all_trades + unique_active_trades


@router.get("/overall", response_model=MetricsResponse)
async def get_overall_metrics():
    """
    Get overall portfolio metrics.

    Returns:
        - Total PnL, Win Rate, Sharpe Ratio
        - Max Drawdown, Profit Factor
        - Average win/loss, trade duration
        - Total fees paid
    """
    trades = _get_all_trades_without_duplicates()
    metrics = analytics.calculate_overall_metrics(trades)
    return metrics


@router.get("/tickers", response_model=List[TickerMetrics])
async def get_ticker_metrics():
    """
    Get per-ticker performance breakdown.

    Returns list of TickerMetrics sorted by total PnL descending.
    """
    trades = _get_all_trades_without_duplicates()
    ticker_metrics = analytics.calculate_ticker_metrics(trades)
    return ticker_metrics


@router.get("/ticker/{ticker}", response_model=TickerMetrics)
async def get_single_ticker_metrics(ticker: str):
    """
    Get metrics for a specific ticker.

    Args:
        ticker: Trading pair (e.g., SOLUSDT)
    """
    # Get all trades without duplicates
    all_trades = _get_all_trades_without_duplicates()

    # Filter for this ticker only
    trades = [t for t in all_trades if t.ticker == ticker]

    if not trades:
        raise HTTPException(status_code=404, detail=f"No trades found for {ticker}")

    ticker_metrics = analytics.calculate_ticker_metrics(trades)

    # Find the specific ticker
    for tm in ticker_metrics:
        if tm.ticker == ticker:
            return tm

    raise HTTPException(status_code=404, detail=f"No metrics for {ticker}")


@router.get("/equity-curve", response_model=List[EquityCurvePoint])
async def get_equity_curve():
    """
    Get cumulative equity curve (P&L over time) from BYBIT API.

    SOURCE OF TRUTH: Uses real data from Bybit exchange (closed P&L history).
    Falls back to logs if Bybit API is not available.

    Returns list of EquityCurvePoint sorted by timestamp.
    """
    # TRY BYBIT API FIRST (SOURCE OF TRUTH)
    if bybit_service and bybit_service.is_available():
        try:
            # Get equity curve from Bybit API (last 100 trades)
            bybit_equity_data = bybit_service.build_equity_curve_from_bybit(limit=100)

            if bybit_equity_data:
                # Convert to EquityCurvePoint format
                # Calculate running max equity for drawdown calculation
                running_max_equity = 0.0
                equity_curve = []

                for i, point in enumerate(bybit_equity_data):
                    equity = point['equity']
                    running_max_equity = max(running_max_equity, equity)

                    # Calculate drawdown from peak
                    if running_max_equity > 0:
                        drawdown = ((running_max_equity - equity) / running_max_equity) * 100
                    else:
                        drawdown = 0.0

                    equity_curve.append(EquityCurvePoint(
                        timestamp=point['timestamp'],
                        cumulative_pnl=equity,  # Use actual equity from Bybit
                        trade_count=i + 1,  # Sequential trade count
                        drawdown=drawdown  # Drawdown percentage from peak
                    ))

                return equity_curve
        except Exception as e:
            print(f"⚠️  Failed to get equity curve from Bybit API: {e}")
            print("   Falling back to logs...")

    # FALLBACK: Use logs if Bybit API unavailable
    trades = trade_parser.get_all_trades()
    equity_curve = analytics.calculate_equity_curve(trades)
    return equity_curve


@router.get("/drawdown-curve", response_model=List[DrawdownPoint])
async def get_drawdown_curve():
    """
    Get underwater equity (drawdown chart).

    Shows how deep the drawdown was at each point in time.
    """
    trades = trade_parser.get_all_trades()
    drawdown_curve = analytics.calculate_drawdown_curve(trades)
    return drawdown_curve


@router.get("/exit-reasons", response_model=List[ExitReasonStats])
async def get_exit_reason_stats():
    """
    Get statistics grouped by exit reason.

    Returns counts and total PnL for:
    - TP (Take Profit)
    - SL (Stop Loss)
    - TSL (Trailing Stop Loss)
    - Partial_TP, Dynamic_TP
    - Timeout, Manual, Emergency
    """
    trades = trade_parser.get_all_trades()
    exit_stats = analytics.calculate_exit_reason_stats(trades)
    return exit_stats


@router.get("/duration-histogram", response_model=List[TradeDurationBin])
async def get_duration_histogram():
    """
    Get trade duration distribution.

    Bins: <1h, 1-4h, 4-12h, 12-24h, >24h
    """
    trades = trade_parser.get_all_trades()
    duration_hist = analytics.calculate_duration_histogram(trades)
    return duration_hist


@router.get("/wallet-balance")
async def get_wallet_balance():
    """
    Get current wallet balance (equity) from Bybit API.

    SOURCE OF TRUTH: Real balance from exchange.

    Returns:
        {
            "equity": 1234.56,
            "available": true,
            "timestamp": "2025-11-02T21:30:00"
        }
    """
    from datetime import datetime

    if not bybit_service or not bybit_service.is_available():
        return {
            "equity": 0.0,
            "available": False,
            "error": "Bybit API not configured",
            "timestamp": datetime.now().isoformat()
        }

    try:
        equity = bybit_service.get_wallet_balance()
        return {
            "equity": equity,
            "available": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "equity": 0.0,
            "available": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
