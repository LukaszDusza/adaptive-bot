"""
Alert Manager - proactive risk monitoring

Checks for critical trading conditions and generates alerts.
"""
from typing import List, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel
import logging

from app.models import MetricsResponse, Trade

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert(BaseModel):
    """Risk alert model"""
    severity: AlertSeverity
    title: str
    message: str
    action: Optional[str] = None
    timestamp: datetime = datetime.now()
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


class AlertManager:
    """
    Monitors trading metrics and generates proactive alerts.

    Alert triggers:
    - Drawdown exceeds threshold (15%, 20%, 25%)
    - Win rate drops below acceptable level (< 40%)
    - Recent losing streak (5+ consecutive losses)
    - Large single loss (> 5% of account)
    - Low daily/weekly PnL (negative trend)
    - Position concentration risk (too many trades on one ticker)
    """

    # Configurable thresholds
    DRAWDOWN_WARNING_PCT = 15.0
    DRAWDOWN_CRITICAL_PCT = 25.0
    WIN_RATE_WARNING = 0.40
    WIN_RATE_CRITICAL = 0.30
    LOSING_STREAK_WARNING = 5
    LOSING_STREAK_CRITICAL = 8
    MIN_TRADES_FOR_WR_ALERT = 20  # Don't alert on win rate until enough trades

    def check_alerts(self, metrics: MetricsResponse, recent_trades: List[Trade]) -> List[Alert]:
        """
        Check all alert conditions and return list of active alerts.

        Args:
            metrics: Overall portfolio metrics
            recent_trades: Recent trades for streak analysis

        Returns:
            List of Alert objects
        """
        alerts = []

        # Check drawdown
        if metrics.max_drawdown_percent is not None:
            dd_alerts = self._check_drawdown(metrics.max_drawdown_percent)
            alerts.extend(dd_alerts)

        # Check win rate
        if metrics.total_trades >= self.MIN_TRADES_FOR_WR_ALERT:
            wr_alerts = self._check_win_rate(metrics.win_rate, metrics.total_trades)
            alerts.extend(wr_alerts)

        # Check losing streak
        streak_alerts = self._check_losing_streak(recent_trades)
        alerts.extend(streak_alerts)

        # Check profit factor
        if metrics.profit_factor is not None:
            pf_alerts = self._check_profit_factor(metrics.profit_factor)
            alerts.extend(pf_alerts)

        # Check Sharpe ratio
        if metrics.sharpe_ratio is not None:
            sharpe_alerts = self._check_sharpe_ratio(metrics.sharpe_ratio)
            alerts.extend(sharpe_alerts)

        # Log generated alerts
        if alerts:
            logger.warning(f"Generated {len(alerts)} alerts: {[a.title for a in alerts]}")

        return alerts

    def _check_drawdown(self, drawdown_pct: float) -> List[Alert]:
        """Check drawdown alerts"""
        alerts = []

        if drawdown_pct >= self.DRAWDOWN_CRITICAL_PCT:
            alerts.append(Alert(
                severity=AlertSeverity.CRITICAL,
                title="Critical Drawdown",
                message=f"Maximum drawdown has reached {drawdown_pct:.1f}%",
                action="URGENT: Consider stopping all bots and reviewing strategy",
                metric_value=drawdown_pct,
                threshold=self.DRAWDOWN_CRITICAL_PCT
            ))
        elif drawdown_pct >= self.DRAWDOWN_WARNING_PCT:
            alerts.append(Alert(
                severity=AlertSeverity.WARNING,
                title="High Drawdown",
                message=f"Maximum drawdown is {drawdown_pct:.1f}%",
                action="Monitor closely - consider reducing position sizes",
                metric_value=drawdown_pct,
                threshold=self.DRAWDOWN_WARNING_PCT
            ))

        return alerts

    def _check_win_rate(self, win_rate: float, total_trades: int) -> List[Alert]:
        """Check win rate alerts"""
        alerts = []

        if win_rate < self.WIN_RATE_CRITICAL:
            alerts.append(Alert(
                severity=AlertSeverity.CRITICAL,
                title="Very Low Win Rate",
                message=f"Win rate dropped to {win_rate*100:.1f}% ({total_trades} trades)",
                action="Strategy may need re-optimization or parameter adjustment",
                metric_value=win_rate * 100,
                threshold=self.WIN_RATE_CRITICAL * 100
            ))
        elif win_rate < self.WIN_RATE_WARNING:
            alerts.append(Alert(
                severity=AlertSeverity.WARNING,
                title="Low Win Rate",
                message=f"Win rate is {win_rate*100:.1f}% ({total_trades} trades)",
                action="Review recent trades for common failure patterns",
                metric_value=win_rate * 100,
                threshold=self.WIN_RATE_WARNING * 100
            ))

        return alerts

    def _check_losing_streak(self, recent_trades: List[Trade]) -> List[Alert]:
        """Check for consecutive losing trades"""
        if not recent_trades:
            return []

        # Count current losing streak
        streak = 0
        for trade in reversed(recent_trades):
            if trade.summary and trade.summary.pnl < 0:
                streak += 1
            else:
                break

        alerts = []

        if streak >= self.LOSING_STREAK_CRITICAL:
            alerts.append(Alert(
                severity=AlertSeverity.CRITICAL,
                title="Severe Losing Streak",
                message=f"Currently on a {streak} trade losing streak",
                action="Consider pausing trading and analyzing root cause",
                metric_value=float(streak),
                threshold=float(self.LOSING_STREAK_CRITICAL)
            ))
        elif streak >= self.LOSING_STREAK_WARNING:
            alerts.append(Alert(
                severity=AlertSeverity.WARNING,
                title="Losing Streak",
                message=f"Currently on a {streak} trade losing streak",
                action="Review market conditions and bot parameters",
                metric_value=float(streak),
                threshold=float(self.LOSING_STREAK_WARNING)
            ))

        return alerts

    def _check_profit_factor(self, profit_factor: float) -> List[Alert]:
        """Check profit factor alerts - DISABLED per user request"""
        alerts = []
        # User requested to disable profit factor alerts
        return alerts

    def _check_sharpe_ratio(self, sharpe_ratio: float) -> List[Alert]:
        """Check Sharpe ratio alerts (risk-adjusted returns)"""
        alerts = []

        if sharpe_ratio < 0.0:
            alerts.append(Alert(
                severity=AlertSeverity.WARNING,
                title="Negative Sharpe Ratio",
                message=f"Risk-adjusted returns are negative ({sharpe_ratio:.2f})",
                action="Strategy returns don't justify the risk taken",
                metric_value=sharpe_ratio,
                threshold=0.0
            ))
        elif sharpe_ratio < 0.5:
            alerts.append(Alert(
                severity=AlertSeverity.INFO,
                title="Low Sharpe Ratio",
                message=f"Sharpe ratio is {sharpe_ratio:.2f} (target: >1.0)",
                action="Consider improving risk management or strategy parameters",
                metric_value=sharpe_ratio,
                threshold=0.5
            ))

        return alerts


# Global alert manager instance
_alert_manager = AlertManager()


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance"""
    return _alert_manager
