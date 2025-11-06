"""
Pydantic models for API request/response schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    COMBINED = "COMBINED"  # For aggregated ticker stats (LONG + SHORT)

    # Also accept capitalized version from bot logs
    @classmethod
    def _missing_(cls, value):
        """Handle case-insensitive matching"""
        if isinstance(value, str):
            value_upper = value.upper()
            for member in cls:
                if member.value.upper() == value_upper:
                    return member
        return None


class ExitReason(str, Enum):
    TP = "TP"
    PARTIAL_TP = "Partial_TP"
    DYNAMIC_TP = "Dynamic_TP"
    SL = "SL"
    TSL = "TSL"
    TIMEOUT = "Timeout"
    MANUAL = "Manual"
    EMERGENCY = "Emergency"


class ContainerStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    RESTARTING = "restarting"
    PAUSED = "paused"
    EXITED = "exited"
    DEAD = "dead"


class BotAction(str, Enum):
    START = "start"
    STOP = "stop"
    RESTART = "restart"


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class HealthResponse(BaseModel):
    """API health check response"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Overall bot performance metrics"""
    total_pnl: float
    total_pnl_percent: float
    win_rate: float
    total_trades: int
    active_trades: int
    sharpe_ratio: Optional[float] = None
    max_drawdown: float
    max_drawdown_percent: float
    profit_factor: Optional[float] = None
    avg_win: float
    avg_loss: float
    avg_trade_duration_hours: float
    total_fees_paid: float
    trades_per_day: float = 0.0
    last_updated: datetime


class TickerMetrics(BaseModel):
    """Per-ticker performance breakdown"""
    ticker: str
    total_pnl: float
    win_rate: float
    total_trades: int
    avg_pnl: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: float
    best_trade: float
    worst_trade: float


class EquityCurvePoint(BaseModel):
    """Single point in equity curve"""
    timestamp: datetime
    cumulative_pnl: float
    trade_count: int
    drawdown: float  # Current drawdown at this point


class TradeEvent(BaseModel):
    """Single trade event (TSL update, partial TP, etc.)"""
    timestamp: datetime
    type: str
    data: Dict[str, Any]


class TradeIndicators(BaseModel):
    """Indicator snapshots at entry/exit"""
    entry: Optional[Dict[str, Any]] = None
    exit: Optional[Dict[str, Any]] = None


class TradeSummary(BaseModel):
    """Trade result summary"""
    pnl: float
    pnl_percent: float
    exit_reason: str
    duration_seconds: float
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    fees_paid: float


class Trade(BaseModel):
    """Complete trade object"""
    trade_id: str
    ticker: str
    side: TradeSide
    start_time: datetime
    end_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    quantity: Optional[float] = None
    leverage: Optional[int] = None

    # Risk management
    initial_sl: Optional[float] = None
    initial_tp: Optional[float] = None
    current_sl: Optional[float] = None
    current_tp: Optional[float] = None

    # Performance
    summary: Optional[TradeSummary] = None
    events: List[TradeEvent] = []
    indicators: Optional[TradeIndicators] = None

    # Status
    is_active: bool = True
    notes: Optional[str] = None


class ActivePosition(BaseModel):
    """Currently active trading position"""
    ticker: str
    side: TradeSide
    entry_price: float
    current_price: float
    quantity: float
    leverage: int
    unrealized_pnl: float
    unrealized_pnl_percent: float
    current_sl: float
    current_tp: float
    duration_hours: float
    entry_time: datetime


class ContainerInfo(BaseModel):
    """Docker container information"""
    name: str
    container_id: str
    status: ContainerStatus
    ticker: str
    account: str
    strategy: str
    uptime_seconds: Optional[float] = None
    health_status: Optional[str] = None
    restart_count: int = 0


class ExitReasonStats(BaseModel):
    """Statistics for exit reasons"""
    exit_reason: str
    count: int
    percentage: float
    total_pnl: float


class DrawdownPoint(BaseModel):
    """Single point in drawdown chart"""
    timestamp: datetime
    drawdown: float
    drawdown_percent: float


class TradeDurationBin(BaseModel):
    """Trade duration histogram bin"""
    bin_label: str  # e.g., "<1h", "1-4h"
    count: int
    avg_pnl: float


class FeeAnalysis(BaseModel):
    """Fee impact analysis"""
    total_fees: float
    fee_impact_pct: float  # Percentage of gross profit consumed by fees
    fees_by_ticker: Dict[str, float]
    avg_fee_per_trade: float
    gross_pnl: float  # PnL before fees
    net_pnl: float  # PnL after fees
    total_trades: int


class StrategyComparison(BaseModel):
    """Comparison of different strategies (side, ticker, etc.)"""
    name: str
    total_pnl: float
    win_rate: float
    total_trades: int
    sharpe_ratio: Optional[float] = None
    max_drawdown_percent: float
    profit_factor: Optional[float] = None
    avg_trade_duration_hours: float


class ExecutionQuality(BaseModel):
    """Execution quality analysis (slippage, maker/taker)"""
    total_executions: int
    avg_slippage_pct: float
    maker_count: int
    taker_count: int
    maker_ratio: float
    taker_ratio: float
    best_slippage_pct: float
    worst_slippage_pct: float
    total_fees: float
    avg_fee_per_execution: float


class FundingCosts(BaseModel):
    """Funding costs analysis"""
    total_funding_fees: float
    funding_count: int
    avg_funding_per_event: float
    daily_avg_funding: float
    monthly_projected_funding: float
    funding_by_symbol: Dict[str, float]


class SLTPEffectiveness(BaseModel):
    """SL/TP effectiveness analysis"""
    total_orders: int
    tp_hit_count: int
    sl_hit_count: int
    tp_hit_rate: float
    sl_hit_rate: float
    avg_tp_distance_pct: float
    avg_sl_distance_pct: float
    risk_reward_ratio: float


class SLTPTrendPoint(BaseModel):
    """Single point in SL/TP effectiveness trend"""
    date: str  # ISO date string (YYYY-MM-DD)
    tp_hit_rate: float
    sl_hit_rate: float
    risk_reward_ratio: float
    avg_tp_distance_pct: float
    avg_sl_distance_pct: float
    trade_count: int


# ============================================================================
# REQUEST MODELS
# ============================================================================

class BotConfigUpdate(BaseModel):
    """Update bot configuration on the fly"""
    prob_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_proba_diff: Optional[float] = Field(None, ge=0.0, le=1.0)
    tp_pct: Optional[float] = Field(None, ge=0.0, le=1.0)
    tsl_pct: Optional[float] = Field(None, ge=0.0, le=1.0)
    trade_size: Optional[float] = Field(None, gt=0)
    leverage: Optional[int] = Field(None, ge=1, le=100)


class TradingPauseRequest(BaseModel):
    """Pause/resume trading"""
    paused: bool
    tickers: Optional[List[str]] = None  # If None, affects all
    reason: Optional[str] = None


class EmergencyCloseRequest(BaseModel):
    """Emergency close all positions"""
    confirmation: bool = Field(..., description="Must be true to execute")
    tickers: Optional[List[str]] = None  # If None, closes all
    reason: str = "Emergency close via dashboard"


class TradeNoteUpdate(BaseModel):
    """Add/update note for a trade"""
    note: str = Field(..., max_length=500)


# ============================================================================
# WEBSOCKET MESSAGES
# ============================================================================

class WSMessage(BaseModel):
    """WebSocket message structure"""
    event: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class WSPriceUpdate(BaseModel):
    """Real-time price update"""
    ticker: str
    price: float
    change_24h: float
    volume_24h: float
    timestamp: datetime


class WSPositionUpdate(BaseModel):
    """Real-time position update"""
    ticker: str
    unrealized_pnl: float
    unrealized_pnl_percent: float
    current_price: float
    timestamp: datetime


class WSContainerStatusUpdate(BaseModel):
    """Container status change notification"""
    container_name: str
    old_status: str
    new_status: str
    timestamp: datetime


class PendingOrder(BaseModel):
    """Pending limit order waiting for execution"""
    order_id: str
    ticker: str
    side: TradeSide
    order_type: str  # "Limit", "Market", etc.
    price: float
    quantity: float
    filled_quantity: float = 0.0
    status: str  # "New", "PartiallyFilled", "Untriggered", etc.
    created_at: datetime
    time_in_force: str  # "GTC", "IOC", "FOK"
    reduce_only: bool = False
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class PendingOrdersResponse(BaseModel):
    """Response with pending orders grouped by ticker"""
    orders: List[PendingOrder]
    total_count: int
    by_ticker: Dict[str, int]  # Count per ticker
    last_updated: datetime


class ModelRanking(BaseModel):
    """Ticker performance ranking (aggregated LONG + SHORT, rolling 1 month window)"""
    model_id: str  # Just ticker e.g., "SOLUSDT" (aggregates all sides)
    ticker: str
    side: TradeSide  # Will be "COMBINED" for aggregated stats
    total_trades: int
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: Optional[float] = None
    risk_reward_ratio: Optional[float] = None  # RRR = avg_win / avg_loss (None if 100% win rate)
    sharpe_ratio: Optional[float] = None
    max_drawdown: float
    trades_per_day: float
    last_trade_time: Optional[datetime] = None