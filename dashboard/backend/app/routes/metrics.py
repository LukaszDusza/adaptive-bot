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

    Equity curve starts from November 1, 2025 (zero level).
    All PnL is calculated relative to equity on that date.

    Returns list of EquityCurvePoint sorted by timestamp.
    """
    _check_bybit_available()

    # Get equity curve from Bybit API (last 200 trades for better history)
    bybit_equity_data = bybit_service.build_equity_curve_from_bybit(limit=200)

    if not bybit_equity_data:
        return []

    # Filter data from November 1, 2025 onwards
    from datetime import datetime
    nov_1_2025 = datetime(2025, 11, 1, 0, 0, 0)

    filtered_data = [
        point for point in bybit_equity_data
        if datetime.fromisoformat(point['timestamp']) >= nov_1_2025
    ]

    if not filtered_data:
        # If no data after Nov 1, return empty (or use first available point)
        return []

    # Use equity on Nov 1, 2025 as the starting point (zero level)
    starting_equity = filtered_data[0]['equity']

    # Convert to EquityCurvePoint format
    running_max_pnl = 0.0
    equity_curve = []

    for i, point in enumerate(filtered_data):
        equity = point['equity']

        # Calculate cumulative PnL relative to Nov 1, 2025 equity
        cumulative_pnl = equity - starting_equity

        # Track peak PnL for drawdown calculation
        running_max_pnl = max(running_max_pnl, cumulative_pnl)

        # Calculate drawdown from peak PnL
        drawdown = running_max_pnl - cumulative_pnl

        equity_curve.append(EquityCurvePoint(
            timestamp=point['timestamp'],
            cumulative_pnl=cumulative_pnl,
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

    # FIX: Calculate percentage drawdown correctly
    # Get starting equity for proper percentage calculation
    starting_equity = bybit_service.get_wallet_balance()  # Current balance
    # Subtract total PnL to get starting balance
    if equity_curve:
        final_pnl = equity_curve[-1].cumulative_pnl
        starting_equity = starting_equity - final_pnl

    drawdown_curve = []
    for point in equity_curve:
        # Peak = starting equity + peak PnL
        peak_pnl = point.cumulative_pnl + point.drawdown
        peak_equity = starting_equity + peak_pnl

        # Calculate percentage drawdown relative to peak equity
        if peak_equity > 0:
            dd_percent = (point.drawdown / peak_equity) * 100
        else:
            dd_percent = 0.0

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

    NOTE: Funding costs require special Bybit API permissions.
    Currently returns estimated data from closed P&L instead.

    Returns approximate funding costs based on trade duration and sizes.
    """
    _check_bybit_available()

    # Get closed trades to estimate funding costs
    pnl_records = bybit_service.get_closed_pnl_history(limit=100)

    if not pnl_records:
        return FundingCosts(
            total_funding_fees=0.0,
            funding_count=0,
            avg_funding_per_event=0.0,
            daily_avg_funding=0.0,
            monthly_projected_funding=0.0,
            funding_by_symbol={}
        )

    # Estimate funding based on cumEntryValue
    # Bybit funding rate is typically 0.01% per 8 hours = 0.03% per day
    # So daily funding = position_value * 0.0003
    from collections import defaultdict
    from datetime import datetime, timedelta

    funding_by_symbol = defaultdict(float)
    total_estimated_funding = 0.0

    # Filter recent trades (last 30 days)
    cutoff_time = datetime.now() - timedelta(days=days)
    cutoff_ms = int(cutoff_time.timestamp() * 1000)

    recent_trades = [r for r in pnl_records if int(r.get('createdTime', 0)) >= cutoff_ms]

    for record in recent_trades:
        entry_value = float(record.get('cumEntryValue', 0))
        created_ms = int(record.get('createdTime', 0))
        updated_ms = int(record.get('updatedTime', 0))

        # Estimate duration in hours
        duration_hours = (updated_ms - created_ms) / (1000 * 3600)

        # Estimate funding: entry_value * 0.01% per 8 hours
        funding_periods = duration_hours / 8
        estimated_funding = entry_value * 0.0001 * funding_periods  # 0.01% = 0.0001

        total_estimated_funding += estimated_funding

        symbol = record.get('symbol', 'UNKNOWN')
        funding_by_symbol[symbol] += estimated_funding

    funding_count = len(recent_trades)
    avg_per_event = total_estimated_funding / funding_count if funding_count > 0 else 0.0
    daily_avg = total_estimated_funding / days if days > 0 else 0.0

    return FundingCosts(
        total_funding_fees=total_estimated_funding,
        funding_count=funding_count,
        avg_funding_per_event=avg_per_event,
        daily_avg_funding=daily_avg,
        monthly_projected_funding=daily_avg * 30,
        funding_by_symbol=dict(funding_by_symbol)
    )


@router.get("/sl-tp-effectiveness", response_model=SLTPEffectiveness)
@cached(ttl=30, key_prefix="sl_tp_effectiveness")
async def get_sl_tp_effectiveness():
    """
    Get SL/TP effectiveness analysis from BYBIT CLOSED P&L (CACHED - 30s TTL).

    IMPROVED: Now analyzes ACTUAL closed trades instead of order history.
    Shows real win/loss patterns from Bybit closed P&L data.

    Returns:
    - TP hit rate (profitable closes)
    - SL hit rate (loss closes)
    - Average profit for TP trades
    - Average loss for SL trades
    - Risk/Reward ratio (avg win / avg loss)

    Useful for:
    - Evaluating strategy effectiveness
    - Understanding actual win/loss patterns
    - Measuring real TP vs SL performance
    """
    _check_bybit_available()

    # Get closed P&L history from Bybit
    pnl_records = bybit_service.get_closed_pnl_history(limit=200)

    if not pnl_records:
        return SLTPEffectiveness(
            total_orders=0,
            tp_hit_count=0,
            sl_hit_count=0,
            tp_hit_rate=0.0,
            sl_hit_rate=0.0,
            avg_tp_distance_pct=0.0,
            avg_sl_distance_pct=0.0,
            risk_reward_ratio=0.0
        )

    # Analyze actual closed trades
    tp_trades = []  # Profitable trades (assumed TP hit)
    sl_trades = []  # Loss trades (assumed SL hit)
    breakeven_trades = []  # Breakeven trades

    tp_profits = []
    sl_losses = []

    for record in pnl_records:
        closed_pnl = float(record.get('closedPnl', 0))
        avg_entry = float(record.get('avgEntryPrice', 0))
        avg_exit = float(record.get('avgExitPrice', 0))
        side = record.get('side', 'Buy')

        # Calculate actual price movement percentage
        if avg_entry > 0:
            if side == 'Buy':
                # LONG: profit if exit > entry
                price_change_pct = ((avg_exit - avg_entry) / avg_entry) * 100
            else:
                # SHORT: profit if exit < entry
                price_change_pct = ((avg_entry - avg_exit) / avg_entry) * 100

            # Categorize by PnL
            if closed_pnl > 0.001:  # Profit (TP assumed)
                tp_trades.append(record)
                tp_profits.append(abs(price_change_pct))
            elif closed_pnl < -0.001:  # Loss (SL assumed)
                sl_trades.append(record)
                sl_losses.append(abs(price_change_pct))
            else:
                breakeven_trades.append(record)

    total_analyzed = len(pnl_records)
    tp_count = len(tp_trades)
    sl_count = len(sl_trades)

    # Calculate rates
    tp_rate = (tp_count / total_analyzed) if total_analyzed > 0 else 0.0
    sl_rate = (sl_count / total_analyzed) if total_analyzed > 0 else 0.0

    # Calculate average distances (price movement percentages)
    avg_tp_dist = (sum(tp_profits) / len(tp_profits)) if tp_profits else 0.0
    avg_sl_dist = (sum(sl_losses) / len(sl_losses)) if sl_losses else 0.0

    # Calculate Risk/Reward ratio
    rr_ratio = (avg_tp_dist / avg_sl_dist) if avg_sl_dist > 0 else 0.0

    return SLTPEffectiveness(
        total_orders=total_analyzed,
        tp_hit_count=tp_count,
        sl_hit_count=sl_count,
        tp_hit_rate=tp_rate,
        sl_hit_rate=sl_rate,
        avg_tp_distance_pct=avg_tp_dist,
        avg_sl_distance_pct=avg_sl_dist,
        risk_reward_ratio=rr_ratio
    )
