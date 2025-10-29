"""
Analytics service - calculates metrics, equity curves, statistics
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from app.models import (
    Trade, MetricsResponse, TickerMetrics, EquityCurvePoint,
    ExitReasonStats, DrawdownPoint, TradeDurationBin
)

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Calculates trading performance metrics and analytics"""

    def calculate_overall_metrics(self, trades: List[Trade]) -> MetricsResponse:
        """
        Calculate overall portfolio metrics.

        Returns:
            MetricsResponse with all key metrics
        """
        # Count active trades first (before any early returns)
        active_trades_count = len([t for t in trades if t.is_active])

        if not trades:
            return self._empty_metrics(active_trades=active_trades_count)

        # Filter completed trades
        completed_trades = [t for t in trades if t.summary is not None]

        if not completed_trades:
            return self._empty_metrics(active_trades=active_trades_count)

        # Extract PnL data
        pnls = [t.summary.pnl for t in completed_trades]
        pnl_pcts = [t.summary.pnl_percent for t in completed_trades]

        total_pnl = sum(pnls)
        total_pnl_percent = np.mean(pnl_pcts) if pnl_pcts else 0.0

        # Win rate
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        win_rate = len(wins) / len(pnls) if pnls else 0.0

        # Average win/loss
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        # Profit factor
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else None

        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe(pnls)

        # Max drawdown
        max_dd, max_dd_pct = self._calculate_max_drawdown(pnls)

        # Average trade duration
        durations = [t.summary.duration_seconds / 3600 for t in completed_trades]  # hours
        avg_duration = np.mean(durations) if durations else 0.0

        # Total fees
        total_fees = sum([t.summary.fees_paid for t in completed_trades])

        # Active trades
        active_trades = [t for t in trades if t.is_active]

        return MetricsResponse(
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            win_rate=win_rate,
            total_trades=len(completed_trades),
            active_trades=len(active_trades),
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd,
            max_drawdown_percent=max_dd_pct,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration_hours=avg_duration,
            total_fees_paid=total_fees,
            last_updated=datetime.now()
        )

    def calculate_ticker_metrics(self, trades: List[Trade]) -> List[TickerMetrics]:
        """Calculate per-ticker performance breakdown"""
        # Group trades by ticker
        ticker_groups = defaultdict(list)
        for trade in trades:
            if trade.summary:
                ticker_groups[trade.ticker].append(trade)

        ticker_metrics = []
        for ticker, ticker_trades in ticker_groups.items():
            pnls = [t.summary.pnl for t in ticker_trades]

            wins = [p for p in pnls if p > 0]
            win_rate = len(wins) / len(pnls) if pnls else 0.0

            total_pnl = sum(pnls)
            avg_pnl = np.mean(pnls) if pnls else 0.0
            sharpe = self._calculate_sharpe(pnls)
            max_dd, _ = self._calculate_max_drawdown(pnls)

            ticker_metrics.append(TickerMetrics(
                ticker=ticker,
                total_pnl=total_pnl,
                win_rate=win_rate,
                total_trades=len(ticker_trades),
                avg_pnl=avg_pnl,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                best_trade=max(pnls) if pnls else 0.0,
                worst_trade=min(pnls) if pnls else 0.0
            ))

        return sorted(ticker_metrics, key=lambda x: x.total_pnl, reverse=True)

    def calculate_equity_curve(self, trades: List[Trade]) -> List[EquityCurvePoint]:
        """
        Calculate cumulative equity curve.

        Returns:
            List of EquityCurvePoint sorted by timestamp
        """
        # Filter and sort completed trades by end time
        completed_trades = [t for t in trades if t.summary and t.end_time]
        completed_trades.sort(key=lambda t: t.end_time)

        if not completed_trades:
            return []

        equity_curve = []
        cumulative_pnl = 0.0
        peak = 0.0

        for i, trade in enumerate(completed_trades):
            cumulative_pnl += trade.summary.pnl

            # Update peak for drawdown calculation
            if cumulative_pnl > peak:
                peak = cumulative_pnl

            # Calculate current drawdown
            drawdown = peak - cumulative_pnl

            equity_curve.append(EquityCurvePoint(
                timestamp=trade.end_time,
                cumulative_pnl=cumulative_pnl,
                trade_count=i + 1,
                drawdown=drawdown
            ))

        return equity_curve

    def calculate_drawdown_curve(self, trades: List[Trade]) -> List[DrawdownPoint]:
        """
        Calculate underwater equity (drawdown chart).

        Returns:
            List of DrawdownPoint showing depth of drawdown over time
        """
        equity_curve = self.calculate_equity_curve(trades)

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

    def calculate_exit_reason_stats(self, trades: List[Trade]) -> List[ExitReasonStats]:
        """
        Calculate statistics grouped by exit reason.

        Returns:
            List of ExitReasonStats sorted by count descending
        """
        completed_trades = [t for t in trades if t.summary]

        if not completed_trades:
            return []

        # Group by exit reason
        exit_groups = defaultdict(list)
        for trade in completed_trades:
            exit_reason = trade.summary.exit_reason
            exit_groups[exit_reason].append(trade.summary.pnl)

        total_trades = len(completed_trades)

        stats = []
        for exit_reason, pnls in exit_groups.items():
            stats.append(ExitReasonStats(
                exit_reason=exit_reason,
                count=len(pnls),
                percentage=len(pnls) / total_trades * 100,
                total_pnl=sum(pnls)
            ))

        return sorted(stats, key=lambda x: x.count, reverse=True)

    def calculate_duration_histogram(self, trades: List[Trade]) -> List[TradeDurationBin]:
        """
        Calculate trade duration distribution.

        Bins: <1h, 1-4h, 4-12h, 12-24h, >24h

        Returns:
            List of TradeDurationBin with counts and avg PnL
        """
        completed_trades = [t for t in trades if t.summary]

        if not completed_trades:
            return []

        # Define bins (in hours)
        bins = [
            ("<1h", 0, 1),
            ("1-4h", 1, 4),
            ("4-12h", 4, 12),
            ("12-24h", 12, 24),
            (">24h", 24, float('inf'))
        ]

        bin_stats = []
        for label, min_h, max_h in bins:
            trades_in_bin = [
                t for t in completed_trades
                if min_h <= (t.summary.duration_seconds / 3600) < max_h
            ]

            if trades_in_bin:
                pnls = [t.summary.pnl for t in trades_in_bin]
                bin_stats.append(TradeDurationBin(
                    bin_label=label,
                    count=len(trades_in_bin),
                    avg_pnl=np.mean(pnls)
                ))
            else:
                bin_stats.append(TradeDurationBin(
                    bin_label=label,
                    count=0,
                    avg_pnl=0.0
                ))

        return bin_stats

    def _calculate_sharpe(self, pnls: List[float], risk_free_rate: float = 0.0) -> Optional[float]:
        """Calculate Sharpe ratio"""
        if len(pnls) < 2:
            return None

        returns = np.array(pnls)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return None

        # Annualize (assuming daily trades, adjust as needed)
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(365)
        return sharpe

    def _calculate_max_drawdown(self, pnls: List[float]) -> tuple[float, float]:
        """
        Calculate maximum drawdown.

        Returns:
            (max_drawdown_absolute, max_drawdown_percent)
        """
        if not pnls:
            return 0.0, 0.0

        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative

        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        # Calculate percentage
        peak_idx = np.argmax(running_max)
        peak_value = running_max[peak_idx]
        max_dd_pct = (max_dd / peak_value * 100) if peak_value > 0 else 0.0

        return float(max_dd), float(max_dd_pct)

    def _empty_metrics(self, active_trades: int = 0) -> MetricsResponse:
        """Return empty metrics response"""
        return MetricsResponse(
            total_pnl=0.0,
            total_pnl_percent=0.0,
            win_rate=0.0,
            total_trades=0,
            active_trades=active_trades,
            sharpe_ratio=None,
            max_drawdown=0.0,
            max_drawdown_percent=0.0,
            profit_factor=None,
            avg_win=0.0,
            avg_loss=0.0,
            avg_trade_duration_hours=0.0,
            total_fees_paid=0.0,
            last_updated=datetime.now()
        )
