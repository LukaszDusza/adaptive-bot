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

# Initialize services
trade_parser = TradeParser()
analytics = AnalyticsService()


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
    # Get all trades (including active positions from bot_state)
    all_trades = trade_parser.get_all_trades()
    active_trades = trade_parser.get_active_trades()

    # Combine for metrics calculation
    trades = all_trades + active_trades

    metrics = analytics.calculate_overall_metrics(trades)
    return metrics


@router.get("/tickers", response_model=List[TickerMetrics])
async def get_ticker_metrics():
    """
    Get per-ticker performance breakdown.

    Returns list of TickerMetrics sorted by total PnL descending.
    """
    trades = trade_parser.get_all_trades()
    ticker_metrics = analytics.calculate_ticker_metrics(trades)
    return ticker_metrics


@router.get("/ticker/{ticker}", response_model=TickerMetrics)
async def get_single_ticker_metrics(ticker: str):
    """
    Get metrics for a specific ticker.

    Args:
        ticker: Trading pair (e.g., SOLUSDT)
    """
    trades = trade_parser.get_trades_by_ticker(ticker)

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
    Get cumulative equity curve (P&L over time).

    Returns list of EquityCurvePoint sorted by timestamp.
    """
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
