"""
Metrics and analytics endpoints - BYBIT API ONLY (Source of Truth)

All data comes from Bybit exchange API, not from local JSON logs.
This ensures real-time accuracy and consistency with actual trading results.
"""
from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime
import numpy as np

from app.models import (
    MetricsResponse, TickerMetrics, EquityCurvePoint,
    ExitReasonStats, DrawdownPoint, TradeDurationBin, FeeAnalysis, StrategyComparison,
    ExecutionQuality, FundingCosts, SLTPEffectiveness
)
from app.services.cache_manager import cached
from app.services.alert_manager import get_alert_manager, Alert

router = APIRouter(prefix="/metrics", tags=["Metrics"])

# Initialize Bybit service - SINGLE SOURCE OF TRUTH
try:
    from app.services.bybit_service import BybitService
    bybit_service = BybitService()

    if not bybit_service.is_available():
        raise RuntimeError("Bybit API credentials not configured")

except Exception as e:
    bybit_service = None
    import logging
    logging.error(f"⚠️  BYBIT SERVICE NOT AVAILABLE: {e}")
    logging.error("   Dashboard will have limited functionality without Bybit API")


def _check_bybit_available():
    """Check if Bybit service is available, raise HTTP error if not"""
    if not bybit_service or not bybit_service.is_available():
        raise HTTPException(
            status_code=503,
            detail="Bybit API not available - please check API credentials in .env file"
        )


@router.get("/overall", response_model=MetricsResponse)
@cached(ttl=10, key_prefix="overall_metrics_bybit")
async def get_overall_metrics():
    """
    Get overall portfolio metrics from BYBIT API (CACHED - 10s TTL).

    SOURCE OF TRUTH: All data from Bybit exchange API.

    Returns:
        - Total PnL, Win Rate, Sharpe Ratio
        - Max Drawdown, Profit Factor
        - Average win/loss
        - Total fees paid
        - Active positions count

    Performance: ~5ms (cached) vs ~300ms (uncached)
    """
    _check_bybit_available()

    # Get stats from Bybit
    stats = bybit_service.get_trade_history_stats_from_bybit(limit=500)

    # Get active positions
    active_positions = bybit_service.get_active_positions_from_bybit()

    # Get current equity
    current_equity = bybit_service.get_wallet_balance()

    # Calculate Sharpe ratio (simplified - using daily returns approximation)
    sharpe_ratio = None  # TODO: Requires more historical data

    # Calculate max drawdown from equity curve
    try:
        equity_curve_data = bybit_service.build_equity_curve_from_bybit(limit=100)
        if equity_curve_data:
            equities = [point['equity'] for point in equity_curve_data]
            running_max = np.maximum.accumulate(equities)
            drawdowns = running_max - equities
            max_dd = float(np.max(drawdowns))
            peak_value = running_max[np.argmax(drawdowns)]
            max_dd_pct = (max_dd / peak_value * 100) if peak_value > 0 else 0.0
        else:
            max_dd = 0.0
            max_dd_pct = 0.0
    except Exception:
        max_dd = 0.0
        max_dd_pct = 0.0

    return MetricsResponse(
        total_pnl=stats.get('total_pnl', 0.0),
        total_pnl_percent=0.0,  # Not directly available from Bybit
        win_rate=stats.get('win_rate', 0.0),
        total_trades=stats.get('total_trades', 0),
        active_trades=len(active_positions),
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_dd,
        max_drawdown_percent=max_dd_pct,
        profit_factor=stats.get('profit_factor'),
        avg_win=stats.get('avg_win', 0.0),
        avg_loss=stats.get('avg_loss', 0.0),
        avg_trade_duration_hours=0.0,  # Not available from Bybit closed PnL
        total_fees_paid=stats.get('total_fees', 0.0),
        last_updated=datetime.now()
    )


@router.get("/tickers", response_model=List[TickerMetrics])
async def get_ticker_metrics():
    """
    Get per-ticker performance breakdown from BYBIT API.

    NOTE: Currently returns empty list - requires per-symbol PnL data from Bybit.
    Bybit API doesn't provide per-ticker breakdown in closed PnL history.
    """
    _check_bybit_available()

    # TODO: Implement per-ticker breakdown when Bybit API provides it
    # For now, return empty list
    return []


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
@cached(ttl=15, key_prefix="equity_curve_bybit")
async def get_equity_curve():
    """
    Get cumulative equity curve (P&L over time) from BYBIT API (CACHED - 15s TTL).

    SOURCE OF TRUTH: Uses real data from Bybit exchange (closed P&L history).

    Returns list of EquityCurvePoint sorted by timestamp.
    """
    _check_bybit_available()

    # Get equity curve from Bybit API (last 100 trades)
    bybit_equity_data = bybit_service.build_equity_curve_from_bybit(limit=100)

    if not bybit_equity_data:
        return []

    # Convert to EquityCurvePoint format
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
            cumulative_pnl=equity,
            trade_count=i + 1,
            drawdown=drawdown
        ))

    return equity_curve


@router.get("/drawdown-curve", response_model=List[DrawdownPoint])
async def get_drawdown_curve():
    """
    Get underwater equity (drawdown chart) from BYBIT API.

    Shows how deep the drawdown was at each point in time.
    """
    _check_bybit_available()

    # Calculate from equity curve
    equity_curve = await get_equity_curve()

    if not equity_curve:
        return []

    drawdown_curve = []
    for point in equity_curve:
        # Calculate percentage drawdown
        peak = point.cumulative_pnl + point.drawdown
        dd_percent = (point.drawdown / peak * 100) if peak > 0 else 0.0

        drawdown_curve.append(DrawdownPoint(
            timestamp=point.timestamp,
            drawdown=point.drawdown,
            drawdown_percent=dd_percent
        ))

    return drawdown_curve


@router.get("/exit-reasons", response_model=List[ExitReasonStats])
async def get_exit_reason_stats():
    """
    Get statistics grouped by exit reason.

    NOTE: Not available from Bybit API - requires local trade logs.
    Returns empty list.
    """
    return []


@router.get("/duration-histogram", response_model=List[TradeDurationBin])
async def get_duration_histogram():
    """
    Get trade duration distribution.

    NOTE: Not available from Bybit API - requires local trade logs.
    Returns empty list.
    """
    return []


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


@router.get("/alerts", response_model=List[Alert])
@cached(ttl=10, key_prefix="risk_alerts_bybit")
async def get_risk_alerts():
    """
    Get active risk alerts based on BYBIT metrics (CACHED - 10s TTL).

    Checks for:
    - High drawdown (>15%, >25%)
    - Low win rate (<40%, <30%)
    - Negative profit factor (<1.0, <1.5)

    NOTE: Losing streak detection disabled (requires full trade history).

    Returns:
        List of Alert objects with severity, message, and recommended action
    """
    _check_bybit_available()

    # Get current metrics from Bybit
    metrics = await get_overall_metrics()

    # Check alert conditions (without streak analysis)
    alert_manager = get_alert_manager()
    alerts = alert_manager.check_alerts(metrics, [])  # Empty trades list - no streak detection

    return alerts


@router.get("/fee-analysis", response_model=FeeAnalysis)
@cached(ttl=15, key_prefix="fee_analysis_bybit")
async def get_fee_analysis():
    """
    Get fee impact analysis from BYBIT API (CACHED - 15s TTL).

    Breaks down trading fees and shows:
    - Total fees paid
    - Average fee per trade
    - Gross vs Net PnL

    SOURCE OF TRUTH: Real fee data from Bybit exchange.
    """
    _check_bybit_available()

    stats = bybit_service.get_trade_history_stats_from_bybit(limit=500)

    total_fees = stats.get('total_fees', 0.0)
    total_pnl = stats.get('total_pnl', 0.0)
    gross_pnl = total_pnl + total_fees

    fee_impact_pct = (total_fees / gross_pnl * 100) if gross_pnl > 0 else 0.0

    return FeeAnalysis(
        total_fees=total_fees,
        fee_impact_pct=fee_impact_pct,
        fees_by_ticker={},  # Not available from Bybit per-symbol
        avg_fee_per_trade=stats.get('avg_fee_per_trade', 0.0),
        gross_pnl=gross_pnl,
        net_pnl=total_pnl,
        total_trades=stats.get('total_trades', 0)
    )


@router.get("/compare", response_model=List[StrategyComparison])
async def get_strategy_comparison(group_by: str = "side"):
    """
    Compare strategies grouped by 'side' or 'ticker'.

    NOTE: Not available from Bybit API - requires detailed trade logs.
    Returns empty list.
    """
    return []


@router.get("/execution-quality", response_model=ExecutionQuality)
@cached(ttl=30, key_prefix="execution_quality")
async def get_execution_quality():
    """
    Get execution quality analysis from BYBIT API (CACHED - 30s TTL).

    Analyzes recent executions (fills) to show:
    - Average slippage (order price vs execution price)
    - Maker vs Taker ratio (affects fees)
    - Best and worst execution slippage
    - Total fees paid on executions

    Useful for:
    - Optimizing order types (limit vs market)
    - Reducing fees by increasing maker orders
    - Identifying slippage patterns
    """
    _check_bybit_available()

    quality = bybit_service.analyze_execution_quality(limit=200)

    return ExecutionQuality(**quality)


@router.get("/funding-costs", response_model=FundingCosts)
@cached(ttl=60, key_prefix="funding_costs")
async def get_funding_costs(days: int = 30):
    """
    Get funding costs analysis from BYBIT API (CACHED - 60s TTL).

    Analyzes funding fees paid for holding positions over time.

    Args:
        days: Number of days to analyze (default 30)

    Returns:
        - Total funding fees paid
        - Daily average funding cost
        - Monthly projected cost
        - Breakdown by symbol

    Useful for:
    - Understanding true cost of holding positions
    - Deciding when to close positions (before funding time)
    - Comparing funding costs across different symbols
    """
    _check_bybit_available()

    costs = bybit_service.analyze_funding_costs(days=days)

    return FundingCosts(**costs)


@router.get("/sl-tp-effectiveness", response_model=SLTPEffectiveness)
@cached(ttl=30, key_prefix="sl_tp_effectiveness")
async def get_sl_tp_effectiveness():
    """
    Get SL/TP effectiveness analysis from BYBIT API (CACHED - 30s TTL).

    Analyzes order history to show:
    - TP hit rate (how often you hit take profit)
    - SL hit rate (how often you hit stop loss)
    - Average TP/SL distances
    - Risk/Reward ratio

    Useful for:
    - Evaluating strategy effectiveness
    - Optimizing TP/SL placement
    - Understanding win/loss patterns
    """
    _check_bybit_available()

    effectiveness = bybit_service.analyze_sl_tp_effectiveness(limit=100)

    return SLTPEffectiveness(**effectiveness)
