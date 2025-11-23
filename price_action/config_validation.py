# config_validation.py
"""
Configuration validation using Pydantic.

Provides validated configuration models to ensure:
- Type safety
- Value constraints
- Relationship validation (e.g., TP > SL)
- Clear error messages for invalid configurations

Usage:
    from config_validation import BotConfigValidated

    try:
        config = BotConfigValidated(
            ticker="SOLUSDT",
            timeframe="15m",
            leverage=10,
            trade_size_usd=100.0,
            tp_pct=0.03,
            tsl_pct=0.01
        )
    except ValidationError as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class TimeframeEnum(str, Enum):
    """Valid timeframe values."""
    m1 = "1m"
    m3 = "3m"
    m5 = "5m"
    m15 = "15m"
    m30 = "30m"
    h1 = "1h"
    h2 = "2h"
    h4 = "4h"
    h6 = "6h"
    h12 = "12h"
    d1 = "1d"
    w1 = "1w"


class SideEnum(str, Enum):
    """Trading side."""
    LONG = "long"
    SHORT = "short"


class BotConfigValidated(BaseModel):
    """
    Validated bot configuration with type safety and constraints.

    All fields are validated at initialization to prevent invalid configurations
    from reaching production.
    """

    # Trading parameters
    ticker: str = Field(..., min_length=6, max_length=20, description="Trading pair (e.g., SOLUSDT)")
    timeframe: str = Field(..., description="Main timeframe (e.g., 15m, 1h, 4h)")
    helper_timeframes: List[str] = Field(default_factory=list, description="Helper timeframes for multi-TF analysis")

    # Position sizing
    trade_size_usd: float = Field(..., gt=0, lt=1000000, description="Trade size in USD (must be positive)")
    trade_size_pct: float = Field(default=0.0, ge=0, le=1.0, description="Trade size as % of balance (0-100%)")
    leverage: int = Field(..., ge=1, le=125, description="Leverage (1-125x)")
    max_positions: int = Field(default=0, ge=0, le=10, description="Max concurrent positions (0 = unlimited)")

    # Risk management
    tp_pct: float = Field(..., gt=0, lt=1.0, description="Take profit percentage (0-100%)")
    tsl_pct: float = Field(..., gt=0, lt=1.0, description="Trailing stop loss percentage (0-100%)")
    sl_pct: Optional[float] = Field(default=None, gt=0, lt=1.0, description="Initial stop loss percentage")

    # ML thresholds
    probability_threshold: float = Field(..., ge=0.5, le=1.0, description="Min probability to open position (0.5-1.0)")
    min_confidence_ratio: float = Field(default=1.5, ge=1.0, le=10.0, description="Min confidence ratio (stronger/weaker signal)")

    # TP mechanisms (mutually exclusive)
    partial_tp_enabled: bool = Field(default=True, description="Enable partial TP (50% at halfway)")
    dynamic_tp_enabled: bool = Field(default=False, description="Enable dynamic TP (25% at 4 levels)")

    # Order types
    limit_order_mode: bool = Field(default=False, description="Use limit orders instead of market orders")
    limit_offset_pct: float = Field(default=0.005, ge=0, le=0.05, description="Limit order price offset (0-5%)")
    max_waiting_limit_order: int = Field(default=300, ge=10, le=3600, description="Max seconds to wait for limit order")
    limit_order_candles: int = Field(default=5, ge=1, le=20, description="Candles to wait for limit order")

    # DCA mode
    dca_enabled: bool = Field(default=False, description="Enable DCA mode")
    dca_atr_multiplier: float = Field(default=1.5, ge=0.5, le=5.0, description="ATR multiplier for DCA distance")

    # Advanced features
    hedge_mode: bool = Field(default=False, description="Enable hedge mode (LONG+SHORT simultaneously)")
    protect_profit_enabled: bool = Field(default=False, description="Move SL to BE if profit declines")
    cooldown_after_loss_minutes: int = Field(default=30, ge=0, le=1440, description="Cooldown after closed position (0-1440 min)")

    # ICT Context Mode
    ict_context_mode: bool = Field(default=False, description="Enable ICT context-aware decisions")
    ict_min_strength: float = Field(default=0.3, ge=0, le=1.0, description="Min ICT strength to apply (0-1)")
    ict_max_threshold_reduction: float = Field(default=0.20, ge=0, le=0.5, description="Max threshold reduction (0-50%)")

    # RSI Reversal Filter
    rsi_reversal_filter_enabled: bool = Field(default=True, description="Enable RSI reversal filter")
    rsi_high_threshold: float = Field(default=65.0, ge=50, le=80, description="Block LONG above this RSI")
    rsi_low_threshold: float = Field(default=35.0, ge=20, le=50, description="Block SHORT below this RSI")

    # Training parameters (optional, for --train mode)
    side: Optional[SideEnum] = Field(default=None, description="Training side: long or short")
    label_trials: int = Field(default=50, ge=10, le=1000, description="Optuna label optimization trials")
    model_trials: int = Field(default=100, ge=10, le=1000, description="Optuna model optimization trials")
    top_n_features: Optional[int] = Field(default=None, ge=10, le=500, description="Top N features to keep")

    # Backtest parameters
    limit: int = Field(default=3000, ge=100, le=200000, description="Number of candles to fetch")
    date_from: Optional[str] = Field(default=None, description="Training end date (YYYY-MM-DD)")
    fetch_max_history: bool = Field(default=False, description="Fetch max available history")

    class Config:
        str_strip_whitespace = True
        validate_assignment = True
        use_enum_values = True
        protected_namespaces = ()  # Allow 'model_' prefix in field names

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate ticker format (must end with USDT)."""
        v = v.upper().strip()
        if not v.endswith('USDT'):
            raise ValueError('Only USDT pairs are supported (e.g., SOLUSDT, BTCUSDT)')
        if len(v) < 6:
            raise ValueError('Ticker too short (min 6 characters)')
        return v

    @field_validator('timeframe')
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe format."""
        v = v.lower().strip()
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']
        if v not in valid_timeframes:
            raise ValueError(f'Invalid timeframe: {v}. Must be one of {valid_timeframes}')
        return v

    @field_validator('helper_timeframes')
    @classmethod
    def validate_helper_timeframes(cls, v: List[str]) -> List[str]:
        """Validate helper timeframes."""
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']
        for tf in v:
            tf_lower = tf.lower().strip()
            if tf_lower not in valid_timeframes:
                raise ValueError(f'Invalid helper timeframe: {tf}. Must be one of {valid_timeframes}')
        return [tf.lower() for tf in v]

    @model_validator(mode='after')
    def validate_tp_greater_than_tsl(self):
        """Ensure TP > TSL for profitability."""
        if self.tp_pct <= self.tsl_pct:
            raise ValueError(
                f'TP ({self.tp_pct:.2%}) must be greater than TSL ({self.tsl_pct:.2%}) '
                f'for profitable risk/reward ratio'
            )
        return self

    @model_validator(mode='after')
    def validate_tp_mechanisms_mutually_exclusive(self):
        """Ensure only one TP mechanism is enabled."""
        if self.partial_tp_enabled and self.dynamic_tp_enabled:
            raise ValueError('Cannot enable both partial_tp and dynamic_tp - choose one')
        return self

    @model_validator(mode='after')
    def validate_dca_requires_limit_order(self):
        """DCA mode requires limit order mode."""
        if self.dca_enabled and not self.limit_order_mode:
            raise ValueError('DCA mode requires limit_order_mode=True')
        return self

    @model_validator(mode='after')
    def validate_leverage_appropriate_for_size(self):
        """Warn if leverage very high with large trade size."""
        if self.leverage > 50 and self.trade_size_usd > 1000:
            # This is just a warning - could use logging instead of raising
            pass  # Could add logger.warning() here
        return self

    @model_validator(mode='after')
    def validate_rsi_thresholds(self):
        """Ensure RSI thresholds make sense."""
        if self.rsi_reversal_filter_enabled:
            if self.rsi_low_threshold >= self.rsi_high_threshold:
                raise ValueError(
                    f'RSI low threshold ({self.rsi_low_threshold}) must be '
                    f'< high threshold ({self.rsi_high_threshold})'
                )
        return self


class TrainingConfigValidated(BaseModel):
    """Validated configuration specifically for training mode."""

    ticker: str = Field(..., min_length=6, max_length=20)
    timeframe: str = Field(...)
    helper_timeframes: List[str] = Field(default_factory=list)
    side: SideEnum = Field(..., description="Required for training: long or short")
    version: str = Field(..., min_length=2, max_length=20, description="Model version (e.g., v1.0)")
    label_trials: int = Field(default=50, ge=10, le=1000)
    model_trials: int = Field(default=100, ge=10, le=1000)
    top_n_features: Optional[int] = Field(default=None, ge=10, le=500)
    limit: int = Field(default=3000, ge=100, le=200000)
    date_from: Optional[str] = Field(default=None)
    fetch_max_history: bool = Field(default=False)

    class Config:
        protected_namespaces = ()  # Allow 'model_' prefix in field names

    @field_validator('version')
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version format (e.g., v1.0, v2.1)."""
        import re
        if not re.match(r'^v\d+\.\d+$', v):
            raise ValueError('Version must be in format vX.Y (e.g., v1.0, v2.1)')
        return v


class BacktestConfigValidated(BaseModel):
    """Validated configuration specifically for backtest mode."""

    ticker: str = Field(..., min_length=6, max_length=20)
    timeframe: str = Field(...)
    helper_timeframes: List[str] = Field(default_factory=list)
    version: str = Field(..., min_length=2, max_length=20)
    probability_threshold: float = Field(..., ge=0.5, le=1.0)
    min_confidence_ratio: float = Field(default=1.5, ge=1.0, le=10.0)
    tp_pct: float = Field(..., gt=0, lt=1.0)
    tsl_pct: float = Field(..., gt=0, lt=1.0)
    trade_size_usd: float = Field(..., gt=0, lt=1000000)
    leverage: int = Field(..., ge=1, le=125)
    partial_tp_enabled: bool = Field(default=True)
    dynamic_tp_enabled: bool = Field(default=False)
    limit: int = Field(default=3000, ge=100, le=50000)

    @model_validator(mode='after')
    def validate_tp_greater_than_tsl(self):
        """Ensure TP > TSL."""
        if self.tp_pct <= self.tsl_pct:
            raise ValueError(f'TP ({self.tp_pct}) must be > TSL ({self.tsl_pct})')
        return self
