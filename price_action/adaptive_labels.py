"""
ADAPTIVE LABEL PARAMETERS MODULE
=================================

Phase 1 Improvement #3: Adaptive barriers based on local market volatility.

Key improvements:
- Barriers adjust to LOCAL ATR (not static)
- High volatility → wider barriers (avoid noise)
- Low volatility → tighter barriers (catch small moves)
- Time limits adapt to trend strength

Traditional approach (PROBLEM):
- PT = 2%, SL = 1.5% dla WSZYSTKICH candles
- W high vol (ATR=3%): 2% = noise
- W low vol (ATR=0.5%): 2% = miss opportunities

Adaptive approach (SOLUTION):
- PT = base_pct * ATR_multiplier * local_ATR
- High vol (ATR=3%): PT = 1.5% * 1.5 * 3.0 = 6.75%
- Low vol (ATR=0.5%): PT = 1.5% * 1.5 * 0.5 = 1.125%

Author: Łukasz + Claude
Created: 2024-11-23
Version: 2.0
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_atr_normalized(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate normalized ATR (ATR as % of price).

    Args:
        df: DataFrame with 'high', 'low', 'close'
        period: ATR period (default: 14)

    Returns:
        Series of ATR normalized by close price (0-1 range)
    """
    import pandas_ta as ta

    # Calculate standard ATR
    atr = ta.atr(df['high'], df['low'], df['close'], length=period)

    # Normalize by close price
    atr_normalized = atr / df['close']

    return atr_normalized.bfill()  # Fixed: use bfill() instead of fillna(method='bfill')


def calculate_trend_strength(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate trend strength indicator (-1 to 1).

    -1: Strong downtrend
     0: No trend (ranging)
    +1: Strong uptrend

    Args:
        df: DataFrame with 'close'
        period: Lookback period

    Returns:
        Series of trend strength
    """
    close = df['close']

    # Linear regression slope
    def rolling_slope(series):
        x = np.arange(len(series))
        y = series.values
        slope = np.polyfit(x, y, 1)[0]
        # Normalize by mean price
        return slope / series.mean()

    trend = close.rolling(period).apply(rolling_slope, raw=False)

    # Clip to -1, 1 range (normalize by std)
    trend_normalized = np.clip(trend / (trend.std() + 1e-8) * 0.5, -1, 1)

    return trend_normalized.fillna(0)


@njit
def _compute_adaptive_labels_fast(prices: np.ndarray,
                                  event_indices: np.ndarray,
                                  atr_values: np.ndarray,
                                  trend_values: np.ndarray,
                                  base_barrier_pct: float,
                                  atr_multiplier: float,
                                  base_time_limit: int,
                                  min_barrier_pct: float,
                                  max_barrier_pct: float) -> np.ndarray:
    """
    Numba-optimized adaptive triple-barrier labeling.

    For each event:
    1. Calculate adaptive barrier = base * atr_multiplier * local_ATR
    2. Calculate adaptive time_limit based on trend strength
    3. Apply triple-barrier logic

    Returns:
        Array of labels (0=timeout, 1=profit, 2=loss)
    """
    n_events = len(event_indices)
    outcomes = np.zeros(n_events, dtype=np.int64)

    for i in range(n_events):
        event_idx = event_indices[i]

        # Get local market conditions
        local_atr = atr_values[event_idx]
        local_trend = trend_values[event_idx]

        # Adaptive barrier based on volatility
        adaptive_barrier = base_barrier_pct * atr_multiplier * local_atr

        # Clamp to reasonable range
        adaptive_barrier = max(min_barrier_pct, min(max_barrier_pct, adaptive_barrier))

        # Symmetric barriers (PT = SL)
        upper_barrier_pct = adaptive_barrier
        lower_barrier_pct = adaptive_barrier

        # Adaptive time limit based on trend strength
        # Strong trend: shorter time (signals resolve faster)
        # Weak trend: longer time (need patience)
        if abs(local_trend) > 0.5:  # Strong trend
            time_limit = int(base_time_limit * 0.7)  # 30% shorter
        elif abs(local_trend) < 0.2:  # Weak trend (ranging)
            time_limit = int(base_time_limit * 1.3)  # 30% longer
        else:
            time_limit = base_time_limit

        # Apply triple-barrier logic
        start_price = prices[event_idx]
        upper_barrier = start_price * (1 + upper_barrier_pct)
        lower_barrier = start_price * (1 - lower_barrier_pct)

        for j in range(1, time_limit + 1):
            current_idx = event_idx + j
            if current_idx >= len(prices):
                outcomes[i] = 0  # Timeout (reached end of data)
                break

            price = prices[current_idx]

            if price >= upper_barrier:
                outcomes[i] = 1  # Profit target hit
                break
            elif price <= lower_barrier:
                outcomes[i] = 2  # Stop loss hit
                break
        else:
            outcomes[i] = 0  # Timeout (no barrier hit)

    return outcomes


def get_adaptive_triple_barrier_labels(df: pd.DataFrame,
                                       t_events: pd.Index,
                                       base_barrier_pct: float = 0.015,
                                       atr_multiplier: float = 1.5,
                                       base_time_limit: int = 24,
                                       min_barrier_pct: float = 0.005,
                                       max_barrier_pct: float = 0.05,
                                       atr_period: int = 14,
                                       trend_period: int = 20,
                                       verbose: bool = True) -> pd.Series:
    """
    Generate adaptive triple-barrier labels.

    Key improvements over static barriers:
    1. Barriers scale with local volatility (ATR)
    2. Time limits adapt to trend strength
    3. Prevents overfitting to specific volatility regimes

    Args:
        df: DataFrame with OHLCV data
        t_events: Index of events to label (typically all timestamps)
        base_barrier_pct: Base barrier size (default: 1.5%)
        atr_multiplier: Multiplier for ATR adaptation (default: 1.5x)
        base_time_limit: Base time limit in candles (default: 24)
        min_barrier_pct: Minimum barrier (prevents too tight, default: 0.5%)
        max_barrier_pct: Maximum barrier (prevents too wide, default: 5%)
        atr_period: ATR calculation period (default: 14)
        trend_period: Trend strength period (default: 20)
        verbose: Enable logging

    Returns:
        Series of labels (0=timeout, 1=profit, 2=loss)

    Example:
        # High volatility candle (ATR = 2.5%)
        adaptive_barrier = 1.5% * 1.5 * 2.5% = 5.625%
        clamped = min(5%, max(0.5%, 5.625%)) = 5%

        # Low volatility candle (ATR = 0.4%)
        adaptive_barrier = 1.5% * 1.5 * 0.4% = 0.9%
        clamped = min(5%, max(0.5%, 0.9%)) = 0.9%
    """
    if verbose:
        logger.info("\n" + "="*60)
        logger.info("ADAPTIVE TRIPLE-BARRIER LABELING")
        logger.info("="*60)
        logger.info(f"Base barrier: {base_barrier_pct*100:.2f}%")
        logger.info(f"ATR multiplier: {atr_multiplier}x")
        logger.info(f"Base time limit: {base_time_limit} candles")
        logger.info(f"Barrier range: {min_barrier_pct*100:.2f}% - {max_barrier_pct*100:.2f}%")

    # Calculate market condition indicators
    if 'atr_normalized' not in df.columns:
        if verbose:
            logger.info("Calculating ATR (not in dataframe)...")
        atr_normalized = calculate_atr_normalized(df, period=atr_period)
    else:
        atr_normalized = df['atr_normalized']

    # Calculate trend strength
    if verbose:
        logger.info("Calculating trend strength...")
    trend_strength = calculate_trend_strength(df, period=trend_period)

    # Handle duplicate index
    if not df.index.is_unique:
        if verbose:
            logger.warning(f"⚠️  Duplicate index detected: {df.index.duplicated().sum()} duplicates")
        df = df[~df.index.duplicated(keep='first')]
        atr_normalized = atr_normalized[~atr_normalized.index.duplicated(keep='first')]
        trend_strength = trend_strength[~trend_strength.index.duplicated(keep='first')]

    # Prepare arrays for Numba
    prices_arr = df['close'].values
    atr_arr = atr_normalized.values
    trend_arr = trend_strength.values

    # Get event indices
    event_indices = df.index.get_indexer(t_events)

    # Filter out invalid indices (-1 = not found)
    valid_mask = (event_indices >= 0)
    event_indices = event_indices[valid_mask]
    t_events_valid = t_events[valid_mask]

    if verbose:
        logger.info(f"Labeling {len(event_indices):,} events...")

    # Call Numba-optimized function
    outcomes = _compute_adaptive_labels_fast(
        prices_arr,
        event_indices,
        atr_arr,
        trend_arr,
        base_barrier_pct,
        atr_multiplier,
        base_time_limit,
        min_barrier_pct,
        max_barrier_pct
    )

    # Create Series
    labels = pd.Series(outcomes, index=t_events_valid)

    if verbose:
        label_dist = labels.value_counts(normalize=True).sort_index()
        logger.info(f"\nLabel distribution:")
        logger.info(f"  0 (Timeout): {label_dist.get(0, 0)*100:.1f}%")
        logger.info(f"  1 (Profit):  {label_dist.get(1, 0)*100:.1f}%")
        logger.info(f"  2 (Loss):    {label_dist.get(2, 0)*100:.1f}%")

        # Calculate average barriers applied
        avg_atr = atr_normalized.loc[t_events_valid].mean()
        avg_barrier = base_barrier_pct * atr_multiplier * avg_atr
        avg_barrier_clamped = np.clip(avg_barrier, min_barrier_pct, max_barrier_pct)

        logger.info(f"\nAdaptive barrier statistics:")
        logger.info(f"  Average ATR: {avg_atr*100:.2f}%")
        logger.info(f"  Average barrier: {avg_barrier_clamped*100:.2f}%")
        logger.info(f"  Min barrier: {min_barrier_pct*100:.2f}%")
        logger.info(f"  Max barrier: {max_barrier_pct*100:.2f}%")
        logger.info("="*60)

    return labels


def analyze_barrier_distribution(df: pd.DataFrame,
                                 labels: pd.Series,
                                 base_barrier_pct: float = 0.015,
                                 atr_multiplier: float = 1.5,
                                 min_barrier_pct: float = 0.005,
                                 max_barrier_pct: float = 0.05) -> pd.DataFrame:
    """
    Analyze distribution of adaptive barriers across dataset.

    Returns DataFrame with barrier stats per decile of ATR.
    Useful for understanding how adaptation works.
    """
    # Calculate ATR if not present
    if 'atr_normalized' not in df.columns:
        atr_normalized = calculate_atr_normalized(df)
    else:
        atr_normalized = df['atr_normalized']

    # Align with labels
    atr_labeled = atr_normalized.loc[labels.index]

    # Calculate adaptive barriers
    adaptive_barriers = base_barrier_pct * atr_multiplier * atr_labeled
    adaptive_barriers_clamped = np.clip(adaptive_barriers, min_barrier_pct, max_barrier_pct)

    # Create analysis DataFrame
    analysis_df = pd.DataFrame({
        'atr': atr_labeled,
        'barrier': adaptive_barriers_clamped,
        'label': labels
    })

    # Group by ATR deciles
    analysis_df['atr_decile'] = pd.qcut(analysis_df['atr'], q=10, labels=False, duplicates='drop')

    # Aggregate by decile
    summary = analysis_df.groupby('atr_decile').agg({
        'atr': ['mean', 'min', 'max'],
        'barrier': ['mean', 'min', 'max'],
        'label': lambda x: (x == 1).mean()  # Win rate
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'label_<lambda>': 'win_rate'})

    logger.info("\n" + "="*60)
    logger.info("ADAPTIVE BARRIER ANALYSIS (by ATR decile)")
    logger.info("="*60)
    logger.info(summary.to_string())
    logger.info("="*60)

    return summary
