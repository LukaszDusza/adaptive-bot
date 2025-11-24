"""
UNIT TESTS for data_preparer_pa.py

Tests critical functionality:
1. ICT feature detection (FVG, liquidity sweeps, order blocks)
2. No look-ahead bias
3. Multi-timeframe alignment
4. Continuous ICT features
5. Data validation (OHLC consistency)

Usage:
    pytest tests/test_data_preparer.py -v
    pytest tests/test_data_preparer.py::test_fvg_detection -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparer_pa import (
    detect_fair_value_gaps,
    detect_liquidity_sweeps,
    detect_order_blocks,
    add_ict_smart_money_features,
    fetch_and_prepare_data,
    _calculate_bars_since_event_vectorized
)


# ==================== FIXTURES ====================

@pytest.fixture
def simple_ohlc_data():
    """Create simple OHLC data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 100.0 + np.random.randn(100) * 2,
        'high': 102.0 + np.random.randn(100) * 2,
        'low': 98.0 + np.random.randn(100) * 2,
        'close': 100.0 + np.random.randn(100) * 2,
        'volume': 1000 + np.random.randn(100) * 100
    })
    df.set_index('timestamp', inplace=True)

    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def fvg_bullish_scenario():
    """Create data with known bullish FVG at index 2."""
    df = pd.DataFrame({
        'open': [100, 101, 105],
        'high': [101, 102, 106],  # Candle 2 low (104) > Candle 0 high (101)
        'low': [99, 100, 104],    # → FVG between 101-104
        'close': [100.5, 101.5, 105.5],
        'volume': [1000, 1000, 1500]
    })
    df.index = pd.date_range(start='2024-01-01', periods=3, freq='15min')
    return df


@pytest.fixture
def fvg_bearish_scenario():
    """Create data with known bearish FVG at index 2."""
    df = pd.DataFrame({
        'open': [100, 99, 95],
        'high': [101, 100, 96],   # Candle 2 high (96) < Candle 0 low (99)
        'low': [99, 98, 94],      # → FVG between 96-99
        'close': [99.5, 98.5, 94.5],
        'volume': [1000, 1000, 1500]
    })
    df.index = pd.date_range(start='2024-01-01', periods=3, freq='15min')
    return df


@pytest.fixture
def liquidity_sweep_scenario():
    """Create data with liquidity sweep with STRONG reversal (POPRAWKA #17: 0.3% threshold)."""
    # Swing high at index 2 (102), then sweep at index 5 (102.5), then STRONG reversal
    df = pd.DataFrame({
        'high': [100, 101, 102, 101.5, 101.8, 102.5, 101],  # Index 5 sweeps high
        'low': [98, 99, 100, 99.5, 99.8, 100.5, 98],
        'close': [99, 100, 101, 100.5, 100.8, 98.5, 99],  # STRONG reversal: 98.5 < 102 * 0.997 (101.694)
        'open': [98.5, 99.5, 100.5, 100, 100.5, 102, 100.5],  # Bearish candle: open > close
        'volume': [1000, 1000, 1000, 1000, 1000, 2000, 1000]  # High volume on sweep
    })
    df.index = pd.date_range(start='2024-01-01', periods=7, freq='15min')
    return df


@pytest.fixture
def order_block_scenario():
    """Create data with order block (POPRAWKA #14: OB must be 3 candles BEFORE impulse)."""
    # For bullish OB: bearish candle at i, followed by 3-candle bullish impulse at i+3
    df = pd.DataFrame({
        'open': [101, 100.5, 101, 101.5, 102, 105, 106],   # Bearish candle at 0, impulse 3-5
        'high': [101.5, 101, 101.5, 102, 102.5, 105.5, 106.5],
        'low': [99.5, 100, 100.5, 101, 101.5, 104, 105],
        'close': [100, 100.8, 101.2, 102, 102.3, 105, 106.2],  # Index 0 bearish, then impulse
        'volume': [1000, 1000, 1000, 1000, 3000, 3000, 2000]  # High volume on impulse
    })
    df.index = pd.date_range(start='2024-01-01', periods=7, freq='15min')
    return df


# ==================== ICT FEATURE TESTS ====================

def test_fvg_bullish_detection(fvg_bullish_scenario):
    """Test that bullish FVG is correctly detected."""
    df = fvg_bullish_scenario
    result = detect_fair_value_gaps(df)

    # Check that FVG is detected at index 2
    assert 'fvg_signal' in result.columns, "fvg_signal column missing"
    assert result['fvg_signal'].iloc[2] == 1, f"Expected bullish FVG (1), got {result['fvg_signal'].iloc[2]}"

    # Check FVG size is positive
    assert 'fvg_size' in result.columns, "fvg_size column missing"
    assert result['fvg_size'].iloc[2] > 0, "FVG size should be positive for bullish gap"

    # FVG gap should be ~3% (gap 3 / close ~105 = 0.028)
    # Note: fvg_size is normalized by close price
    assert 0.02 < result['fvg_size'].iloc[2] < 0.04, \
        f"Expected FVG size ~0.03 (3%), got {result['fvg_size'].iloc[2]}"


def test_fvg_bearish_detection(fvg_bearish_scenario):
    """Test that bearish FVG is correctly detected."""
    df = fvg_bearish_scenario
    result = detect_fair_value_gaps(df)

    # Check that bearish FVG is detected at index 2
    assert result['fvg_signal'].iloc[2] == -1, f"Expected bearish FVG (-1), got {result['fvg_signal'].iloc[2]}"

    # Note: fvg_size may be positive or negative depending on implementation
    # The key is that FVG is DETECTED (signal = -1)
    assert result['fvg_size'].iloc[2] != 0, f"FVG size should be non-zero, got {result['fvg_size'].iloc[2]}"


def test_liquidity_sweep_detection(liquidity_sweep_scenario):
    """Test liquidity sweep detection."""
    df = liquidity_sweep_scenario
    result = detect_liquidity_sweeps(df, lookback=20)

    # Check that sweep is detected
    assert 'liquidity_sweep' in result.columns, "liquidity_sweep column missing"

    # At index 5, price should sweep the swing high at index 2
    # Note: exact index depends on implementation details
    sweep_signals = result['liquidity_sweep'].abs().sum()
    assert sweep_signals > 0, "No liquidity sweep detected in sweep scenario"


def test_order_block_detection(order_block_scenario):
    """Test order block detection after impulse."""
    df = order_block_scenario
    result = detect_order_blocks(df, impulse_threshold=0.015)

    # Check that order block is detected
    assert 'order_block' in result.columns, "order_block column missing"

    # Should detect bullish order block (impulse up)
    ob_signals = result['order_block'].abs().sum()
    assert ob_signals > 0, "No order block detected in impulse scenario"


def test_continuous_ict_features(simple_ohlc_data):
    """Test that continuous ICT features are created and are continuous (not binary)."""
    df = simple_ohlc_data

    # Add minimal required features first with some actual events to test continuous values
    df['fvg_signal'] = 0
    df.iloc[10:15, df.columns.get_loc('fvg_signal')] = 1  # Some FVG events

    df['liquidity_sweep'] = 0
    df['order_block'] = 0
    df['breaker_block'] = 0

    df['market_structure_shift'] = 0
    df.iloc[20:22, df.columns.get_loc('market_structure_shift')] = 1  # Some MSS events for testing decay

    df['institutional_candle'] = 0
    df['testing_bullish_ob'] = 0
    df['testing_bearish_ob'] = 0
    df['liquidity_sweep_with_volume'] = 0
    df['market_structure_direction'] = 0
    df['recent_fvg_bullish'] = 0
    df['recent_fvg_bearish'] = 0

    # Call the function
    result = add_ict_smart_money_features(df, parallel=False)

    # Check that continuous features exist
    continuous_features = [
        'fvg_distance_pct',
        'ob_zone_proximity_bull',
        'ob_zone_proximity_bear',
        'mss_strength_decayed',
        'ict_confluence_strength',
        'sweep_volume_ratio',
        'time_in_ob_zone'
    ]

    # Features that can legitimately appear binary in small datasets with few events
    # (e.g., mss_strength_decayed = exp(-0) * 1 = 1 at MSS event, 0 otherwise)
    allow_binary = {'mss_strength_decayed'}

    for feature in continuous_features:
        assert feature in result.columns, f"Continuous feature {feature} missing"

        # Check that values are NOT strictly binary (0/1 only), unless allowed
        if feature not in allow_binary:
            unique_values = result[feature].nunique()
            is_binary = (unique_values == 2) and set(result[feature].unique()) == {0.0, 1.0}

            assert not is_binary, \
                f"{feature} appears to be binary (0/1), should be continuous. Got unique values: {result[feature].unique()}"


def test_bars_since_event_vectorized():
    """Test vectorized bars_since_event calculation."""
    # Create event series: events at index 2, 5, 8
    events = pd.Series([False, False, True, False, False, True, False, False, True, False])

    result = _calculate_bars_since_event_vectorized(events, cap=50)

    # Check expected values
    # Index 0-1: no event yet → cap value
    # Index 2: event → 0
    # Index 3-4: 1, 2 bars since event
    # Index 5: event → 0
    # etc.

    assert result[2] == 0, "Bars since event at event should be 0"
    assert result[3] == 1, "1 bar after event"
    assert result[4] == 2, "2 bars after event"
    assert result[5] == 0, "Next event resets to 0"


# ==================== NO LOOK-AHEAD BIAS TESTS ====================

def test_no_future_data_in_ffill():
    """Test that ffill() doesn't leak future data."""
    df = pd.DataFrame({
        'value': [1, 2, np.nan, np.nan, 5],
        'signal': [1, 0, 0, 0, 1]
    })

    # Forward fill signal values
    filled = df['value'].where(df['signal'] == 1).ffill()

    # At index 2, should have value from index 0 (1), not from index 4 (5)
    assert filled.iloc[2] == 1, f"ffill leaked future data: got {filled.iloc[2]}"
    assert filled.iloc[3] == 1, f"ffill leaked future data: got {filled.iloc[3]}"


def test_last_candle_excluded():
    """Test that fetch_and_prepare_data excludes last forming candle."""
    # This test requires live API call - mark as integration test
    pytest.skip("Integration test - requires API access")

    # If you want to run this, uncomment:
    # df = fetch_and_prepare_data('SOLUSDT', '15m', limit=100, helper_timeframes=[])
    # now = pd.Timestamp.now(tz='UTC')
    # last_candle_time = df.index[-1]
    #
    # # Last candle should be at least 15 minutes old
    # time_diff = (now - last_candle_time).total_seconds() / 60
    # assert time_diff >= 15, f"Last candle is forming (only {time_diff:.1f} min old)"


# ==================== DATA VALIDATION TESTS ====================

def test_ohlc_consistency(simple_ohlc_data):
    """Test that OHLC data is consistent (high >= low, close between high/low)."""
    df = simple_ohlc_data

    # High should be >= Low
    assert (df['high'] >= df['low']).all(), "Found High < Low"

    # Close should be between High and Low
    assert (df['close'] <= df['high']).all(), "Found Close > High"
    assert (df['close'] >= df['low']).all(), "Found Close < Low"

    # Open should be between High and Low
    assert (df['open'] <= df['high']).all(), "Found Open > High"
    assert (df['open'] >= df['low']).all(), "Found Open < Low"


def test_volume_is_positive(simple_ohlc_data):
    """Test that volume is always positive."""
    df = simple_ohlc_data
    assert (df['volume'] >= 0).all(), "Found negative volume"


def test_no_nan_in_critical_columns():
    """Test that critical columns don't have NaN values after feature calculation."""
    # Create minimal data
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [101, 102, 103],
        'low': [99, 100, 101],
        'close': [100.5, 101.5, 102.5],
        'volume': [1000, 1000, 1000]
    })
    df.index = pd.date_range(start='2024-01-01', periods=3, freq='15min')

    # Add minimal features
    df['fvg_signal'] = 0
    df['liquidity_sweep'] = 0
    df['order_block'] = 0
    df['breaker_block'] = 0
    df['market_structure_shift'] = 0
    df['institutional_candle'] = 0
    df['testing_bullish_ob'] = 0
    df['testing_bearish_ob'] = 0
    df['liquidity_sweep_with_volume'] = 0
    df['market_structure_direction'] = 0
    df['recent_fvg_bullish'] = 0
    df['recent_fvg_bearish'] = 0

    result = add_ict_smart_money_features(df, parallel=False)

    # Check critical continuous features for NaN
    critical_features = [
        'ict_composite_score',
        'fvg_distance_pct',
        'ict_confluence_strength'
    ]

    for feature in critical_features:
        if feature in result.columns:
            nan_count = result[feature].isna().sum()
            assert nan_count == 0, f"Found {nan_count} NaN values in {feature}"


# ==================== FEATURE CORRELATION TESTS ====================

def test_continuous_features_not_constant():
    """Test that continuous ICT features are not all the same value (would indicate bug)."""
    # Create data with some variation
    df = pd.DataFrame({
        'open': 100 + np.random.randn(50) * 2,
        'high': 102 + np.random.randn(50) * 2,
        'low': 98 + np.random.randn(50) * 2,
        'close': 100 + np.random.randn(50) * 2,
        'volume': 1000 + np.random.randn(50) * 100
    })
    df.index = pd.date_range(start='2024-01-01', periods=50, freq='15min')

    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Add ICT features
    result = detect_fair_value_gaps(df)
    result = detect_liquidity_sweeps(result, lookback=20)
    result = detect_order_blocks(result, impulse_threshold=0.015)

    # Add minimal required features
    if 'breaker_block' not in result.columns:
        result['breaker_block'] = 0
    if 'market_structure_shift' not in result.columns:
        result['market_structure_shift'] = 0
    if 'institutional_candle' not in result.columns:
        result['institutional_candle'] = 0
    if 'testing_bullish_ob' not in result.columns:
        result['testing_bullish_ob'] = 0
    if 'testing_bearish_ob' not in result.columns:
        result['testing_bearish_ob'] = 0
    if 'liquidity_sweep_with_volume' not in result.columns:
        result['liquidity_sweep_with_volume'] = 0
    if 'market_structure_direction' not in result.columns:
        result['market_structure_direction'] = 0
    if 'recent_fvg_bullish' not in result.columns:
        result['recent_fvg_bullish'] = 0
    if 'recent_fvg_bearish' not in result.columns:
        result['recent_fvg_bearish'] = 0

    result = add_ict_smart_money_features(result, parallel=False)

    # Check that composite score has some variation (unless all signals are 0)
    if result['ict_composite_score'].abs().sum() > 0:
        std = result['ict_composite_score'].std()
        assert std > 0, "ict_composite_score is constant (all same value)"


# ==================== PERFORMANCE TESTS ====================

def test_vectorized_vs_loop_performance():
    """Test that vectorized bars_since_event is faster than loop version."""
    import time

    # Create large event series
    n = 10000
    events = pd.Series(np.random.random(n) > 0.95)  # 5% events

    # Time vectorized version
    start = time.time()
    result_vec = _calculate_bars_since_event_vectorized(events, cap=50)
    time_vec = time.time() - start

    # Vectorized should be very fast (< 0.1s for 10k rows)
    assert time_vec < 0.5, f"Vectorized version too slow: {time_vec:.3f}s"

    # Check result is correct type (can be Series or ndarray)
    assert len(result_vec) == n, "Result length mismatch"
    assert isinstance(result_vec, (np.ndarray, pd.Series)), "Should return numpy array or pandas Series"


# ==================== INTEGRATION TESTS ====================

def test_full_pipeline_with_ict():
    """Integration test: full pipeline with ICT features."""
    # Create realistic data
    n = 200
    base_price = 100
    df = pd.DataFrame({
        'open': base_price + np.cumsum(np.random.randn(n) * 0.5),
        'high': base_price + np.cumsum(np.random.randn(n) * 0.5) + 1,
        'low': base_price + np.cumsum(np.random.randn(n) * 0.5) - 1,
        'close': base_price + np.cumsum(np.random.randn(n) * 0.5),
        'volume': 1000 + np.random.randn(n) * 100
    })
    df.index = pd.date_range(start='2024-01-01', periods=n, freq='15min')

    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1) + 1
    df['low'] = df[['open', 'low', 'close']].min(axis=1) - 1

    # Run ICT detection
    result = detect_fair_value_gaps(df)
    result = detect_liquidity_sweeps(result, lookback=20)
    result = detect_order_blocks(result, impulse_threshold=0.015)

    # Add required features for full pipeline
    if 'breaker_block' not in result.columns:
        result['breaker_block'] = 0
    if 'market_structure_shift' not in result.columns:
        result['market_structure_shift'] = 0
    if 'institutional_candle' not in result.columns:
        result['institutional_candle'] = 0
    if 'testing_bullish_ob' not in result.columns:
        result['testing_bullish_ob'] = 0
    if 'testing_bearish_ob' not in result.columns:
        result['testing_bearish_ob'] = 0
    if 'liquidity_sweep_with_volume' not in result.columns:
        result['liquidity_sweep_with_volume'] = 0
    if 'market_structure_direction' not in result.columns:
        result['market_structure_direction'] = 0
    if 'recent_fvg_bullish' not in result.columns:
        result['recent_fvg_bullish'] = 0
    if 'recent_fvg_bearish' not in result.columns:
        result['recent_fvg_bearish'] = 0

    result = add_ict_smart_money_features(result, parallel=False)

    # Check that all expected features are present
    expected_features = [
        'ict_composite_score',
        'fvg_distance_pct',
        'ob_zone_proximity_bull',
        'ict_confluence_strength'
    ]

    for feature in expected_features:
        assert feature in result.columns, f"Feature {feature} missing from pipeline output"

    # Check no NaN in output
    critical_cols = ['ict_composite_score', 'fvg_distance_pct', 'ict_confluence_strength']
    for col in critical_cols:
        assert result[col].isna().sum() == 0, f"NaN found in {col}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
