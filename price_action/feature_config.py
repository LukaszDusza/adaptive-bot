"""
Feature Engineering Configuration

ENHANCEMENT #8: Centralized configuration for feature parameters.
Instead of hardcoded magic numbers, all tunable parameters are here.

Usage:
    from feature_config import FEATURE_CONFIG
    rsi_period = FEATURE_CONFIG['indicators']['rsi_period']
"""

FEATURE_CONFIG = {
    # ========================================================================
    # TIER 1: BASIC TECHNICAL INDICATORS
    # ========================================================================
    'indicators': {
        # Momentum indicators
        'rsi_period': 14,               # RSI period (standard)
        'rsi_period_fast': 7,           # Fast RSI for short-term
        'rsi_period_slow': 21,          # Slow RSI for long-term

        # Moving averages
        'sma_short': 20,                # Short-term SMA
        'sma_medium': 50,               # Medium-term SMA
        'sma_long': 100,                # Long-term SMA (for regime detection)

        # Bollinger Bands
        'bbands_period': 20,            # BB period
        'bbands_std': 2.0,              # BB standard deviation multiplier

        # Volatility
        'atr_period': 14,               # ATR period (standard)

        # Volume
        'volume_ma_window': 20,         # Volume moving average window
        'volume_ma_long': 50,           # Long-term volume MA
    },

    # ========================================================================
    # TIER 2: PRICE ACTION & STRUCTURE
    # ========================================================================
    'price_action': {
        # Swing detection
        'swing_window': 50,             # Window for swing high/low detection

        # Support/Resistance
        'sr_lookback': 100,             # Lookback for S/R detection
        'sr_proximity_pct': 0.02,       # 2% distance threshold for "near S/R"

        # Candlestick patterns
        'doji_body_pct': 0.1,           # Max body size for doji (10% of range)
        'hammer_wick_ratio': 2.0,       # Min wick-to-body ratio for hammer
    },

    # ========================================================================
    # TIER 3: ADVANCED FEATURES
    # ========================================================================
    'advanced': {
        # Regime detection
        'adx_period': 14,               # ADX period for trend strength
        'adx_trend_threshold': 25,      # ADX > 25 = trending market

        # Volume analysis
        'high_volume_threshold': 1.5,   # 1.5x average = high volume
        'extreme_volume_threshold': 2.0, # 2x average = extreme volume

        # Volatility regimes
        'volatility_high': 1.0,         # 100% annualized vol = high
        'volatility_low': 0.5,          # 50% annualized vol = low
    },

    # ========================================================================
    # TIER 4: ICT & SMART MONEY
    # ========================================================================
    'ict': {
        # Fair Value Gaps
        'fvg_min_size_pct': 0.001,      # 0.1% minimum FVG size (to filter noise)

        # Liquidity Sweeps
        'sweep_lookback': 20,           # Lookback period for sweep detection

        # Order Blocks
        'ob_impulse_threshold': 0.015,  # 1.5% move to qualify as impulse
        'ob_lookback': 20,              # Order block lookback period

        # Breaker Blocks
        'breaker_lookback': 10,         # Breaker block detection window

        # Market Structure
        'mss_swing_period': 10,         # Market structure shift detection period

        # Institutional Candles
        'inst_candle_size_multiplier': 1.5, # Min 1.5x average candle size
        'inst_candle_volume_multiplier': 1.5, # Min 1.5x average volume

        # Liquidity Voids
        'liquidity_void_threshold': 0.5, # Volume < 50% average = void

        # Composite score weights
        'ict_weights': {
            'fvg': 0.15,                # Fair Value Gap weight
            'liquidity_sweep': 0.25,    # Sweep weight (highest - strongest signal)
            'order_block': 0.20,        # Order block weight
            'breaker_block': 0.15,      # Breaker weight
            'market_structure_shift': 0.20, # MSS weight
            'institutional_candle': 0.05    # Inst candle weight
        },

        # Rolling features
        'rolling_window': 10,           # Rolling window for ICT activity
        'bars_since_cap': 50,           # Cap for "bars since last event" features
    },

    # ========================================================================
    # PERFORMANCE & CACHING
    # ========================================================================
    'performance': {
        'parallel_ict': True,           # Enable parallel ICT feature computation
        'cache_features': True,         # Enable persistent disk cache
        'cache_freshness_multiplier': 1.5, # Cache freshness threshold (1.5x timeframe)
    },
}


# Convenience accessors for commonly used values
def get_indicator_param(name: str, default=None):
    """Get indicator parameter by name"""
    return FEATURE_CONFIG['indicators'].get(name, default)


def get_ict_param(name: str, default=None):
    """Get ICT parameter by name"""
    return FEATURE_CONFIG['ict'].get(name, default)


def get_ict_weight(name: str, default=0.0):
    """Get ICT composite score weight by name"""
    return FEATURE_CONFIG['ict']['ict_weights'].get(name, default)
