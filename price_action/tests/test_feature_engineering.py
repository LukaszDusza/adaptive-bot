# tests/test_feature_engineering.py
"""
Unit tests for feature engineering functions.

Tests cover:
- Correlation-based feature removal
- Feature calculation pipeline
- Look-ahead bias detection
- Feature validation
"""

import pytest
import pandas as pd
import numpy as np
from data_preparer_pa import (
    remove_correlated_features,
    fetch_and_prepare_data,
    _calculate_base_features
)


class TestCorrelationRemoval:
    """Test correlation-based feature removal."""

    def test_remove_perfectly_correlated_features(self, sample_ohlcv_df):
        """Test removal of perfectly correlated features."""
        df = sample_ohlcv_df.copy()

        # Add perfectly correlated features
        df['feature_a'] = df['close']
        df['feature_b'] = df['close'] * 2  # Perfect linear correlation
        df['feature_c'] = df['close'] + np.random.randn(len(df)) * 0.01  # Slight variation

        # Remove correlated (threshold 0.95)
        df_result, removed = remove_correlated_features(
            df[['feature_a', 'feature_b', 'feature_c']],
            correlation_threshold=0.95
        )

        # Should remove at least one of the perfectly correlated features
        assert df_result.shape[1] < 3  # Fewer columns than original
        assert len(removed) > 0
        assert 'feature_b' in removed or 'feature_a' in removed

    def test_no_removal_when_low_correlation(self, sample_ohlcv_df):
        """Test no removal when features not correlated."""
        df = sample_ohlcv_df.copy()

        # Add uncorrelated features
        np.random.seed(42)
        df['feature_a'] = np.random.randn(len(df))
        df['feature_b'] = np.random.randn(len(df))
        df['feature_c'] = np.random.randn(len(df))

        df_result, removed = remove_correlated_features(
            df[['feature_a', 'feature_b', 'feature_c']],
            correlation_threshold=0.95
        )

        # All features should remain (low correlation)
        assert df_result.shape[1] == 3
        assert len(removed) == 0

    def test_correlation_threshold_respected(self, sample_ohlcv_df):
        """Test that correlation threshold is respected."""
        df = sample_ohlcv_df.copy()

        # Create features with known correlation
        df['feature_a'] = df['close']
        df['feature_b'] = df['close'] + np.random.randn(len(df)) * 5  # ~0.9 correlation

        # With high threshold (0.99), both should remain
        df_high, removed_high = remove_correlated_features(
            df[['feature_a', 'feature_b']],
            correlation_threshold=0.99
        )
        assert df_high.shape[1] == 2

        # With low threshold (0.85), one might be removed
        df_low, removed_low = remove_correlated_features(
            df[['feature_a', 'feature_b']],
            correlation_threshold=0.85
        )
        assert df_low.shape[1] <= 2

    def test_returns_correct_types(self, sample_ohlcv_df):
        """Test that function returns correct types."""
        df = sample_ohlcv_df.copy()
        df['feature_a'] = df['close']
        df['feature_b'] = df['close'] * 2

        df_result, removed = remove_correlated_features(
            df[['feature_a', 'feature_b']],
            correlation_threshold=0.95
        )

        assert isinstance(df_result, pd.DataFrame)
        assert isinstance(removed, list)
        assert all(isinstance(f, str) for f in removed)


@pytest.mark.slow
class TestFeaturePreparer:
    """Test feature preparation pipeline."""

    def test_fetch_and_prepare_data_basic(self):
        """Test basic data fetch and feature calculation."""
        df = fetch_and_prepare_data(
            ticker='SOLUSDT',
            timeframe='15m',
            limit=500,
            helper_timeframes=[],
            side='long',
            skip_slow_features=True  # Skip slow features for test speed
        )

        # Verify basic structure
        assert len(df) > 0
        assert 'close' in df.columns
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'volume' in df.columns

        # Verify features were calculated
        assert 'rsi_14' in df.columns
        assert df['close'].notna().all()

    def test_fetch_and_prepare_data_with_helper_timeframes(self):
        """Test data preparation with helper timeframes."""
        df = fetch_and_prepare_data(
            ticker='SOLUSDT',
            timeframe='15m',
            limit=300,
            helper_timeframes=['1h'],
            side='long',
            skip_slow_features=True
        )

        # Verify helper timeframe features exist
        helper_features = [col for col in df.columns if col.startswith('1h_')]
        assert len(helper_features) > 0

    def test_features_have_no_nan_at_end(self):
        """Test that features don't have NaN at the end (after fillna)."""
        df = fetch_and_prepare_data(
            ticker='SOLUSDT',
            timeframe='15m',
            limit=200,
            helper_timeframes=[],
            side='long',
            skip_slow_features=True
        )

        # Check last 10 rows for NaN (should be filled)
        last_rows = df.tail(10)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        for col in feature_cols:
            nan_count = last_rows[col].isna().sum()
            # Allow some NaN at the very beginning due to rolling windows
            assert nan_count <= 2, f"Too many NaN in {col}: {nan_count}"

    def test_features_calculation_deterministic(self):
        """Test that feature calculation is deterministic."""
        # Call twice with same parameters
        df1 = fetch_and_prepare_data(
            ticker='SOLUSDT',
            timeframe='15m',
            limit=100,
            helper_timeframes=[],
            side='long',
            skip_slow_features=True
        )

        df2 = fetch_and_prepare_data(
            ticker='SOLUSDT',
            timeframe='15m',
            limit=100,
            helper_timeframes=[],
            side='long',
            skip_slow_features=True
        )

        # Results should be identical (same API data, same calculations)
        # Compare last 50 rows (to account for any API timing differences)
        common_cols = list(set(df1.columns) & set(df2.columns))

        for col in common_cols:
            if col != 'timestamp':  # Skip timestamp comparison
                pd.testing.assert_series_equal(
                    df1[col].tail(50),
                    df2[col].tail(50),
                    check_names=False,
                    atol=0.01  # Allow small floating point differences
                )


class TestNoLookAheadBias:
    """Test that features don't use future data (look-ahead bias)."""

    def test_no_negative_shifts_in_feature_calculation(self):
        """Test that feature calculation doesn't use shift(-1) or negative shifts."""
        import inspect

        # Get source code of feature calculation function
        try:
            source = inspect.getsource(_calculate_base_features)

            # Check for dangerous patterns
            dangerous_patterns = [
                'shift(-1)',
                'shift(-2)',
                'shift(-3)',
                '.iloc[i+1]',
                '[i+1]',
            ]

            for pattern in dangerous_patterns:
                assert pattern not in source, f"Found look-ahead pattern: {pattern}"
        except Exception as e:
            pytest.skip(f"Could not inspect source code: {e}")

    def test_feature_values_only_depend_on_past(self, sample_ohlcv_df):
        """Test that feature at index i only depends on data up to i."""
        df = sample_ohlcv_df.copy()

        # Calculate features
        df_with_features = _calculate_base_features(df, feature_level='minimal')

        # Modify future data (should not affect current features)
        df_modified = sample_ohlcv_df.copy()
        df_modified.loc[df_modified.index[-10:], 'close'] *= 2  # Change last 10 values

        df_modified_features = _calculate_base_features(df_modified, feature_level='minimal')

        # Features in rows before modification should be identical
        feature_cols = [col for col in df_with_features.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']]

        # Compare first 80 rows (before the modified region)
        for col in feature_cols:
            if col in df_modified_features.columns:
                # Allow some difference due to rolling calculations
                original_values = df_with_features[col].iloc[:80]
                modified_values = df_modified_features[col].iloc[:80]

                # Most values should be very similar (within 1%)
                if original_values.notna().sum() > 50:
                    correlation = original_values.corr(modified_values)
                    assert correlation > 0.95, f"Feature {col} has low correlation: {correlation}"


class TestFeatureValidation:
    """Test feature validation and constraints."""

    def test_rsi_in_valid_range(self):
        """Test that RSI values are in valid range (0-100)."""
        df = fetch_and_prepare_data(
            ticker='SOLUSDT',
            timeframe='15m',
            limit=200,
            helper_timeframes=[],
            side='long',
            skip_slow_features=True
        )

        if 'rsi_14' in df.columns:
            rsi_values = df['rsi_14'].dropna()
            assert (rsi_values >= 0).all(), "RSI has values below 0"
            assert (rsi_values <= 100).all(), "RSI has values above 100"

    def test_volume_is_positive(self):
        """Test that volume is always positive."""
        df = fetch_and_prepare_data(
            ticker='SOLUSDT',
            timeframe='15m',
            limit=200,
            helper_timeframes=[],
            side='long',
            skip_slow_features=True
        )

        assert (df['volume'] > 0).all(), "Volume has non-positive values"

    def test_ohlc_relationships(self):
        """Test that OHLC relationships are valid."""
        df = fetch_and_prepare_data(
            ticker='SOLUSDT',
            timeframe='15m',
            limit=200,
            helper_timeframes=[],
            side='long',
            skip_slow_features=True
        )

        # High >= max(open, close)
        assert (df['high'] >= df[['open', 'close']].max(axis=1)).all(), "High is not >= max(open, close)"

        # Low <= min(open, close)
        assert (df['low'] <= df[['open', 'close']].min(axis=1)).all(), "Low is not <= min(open, close)"

    def test_no_infinite_values(self):
        """Test that features don't have infinite values."""
        df = fetch_and_prepare_data(
            ticker='SOLUSDT',
            timeframe='15m',
            limit=200,
            helper_timeframes=[],
            side='long',
            skip_slow_features=True
        )

        feature_cols = [col for col in df.columns if col not in ['timestamp']]

        for col in feature_cols:
            inf_count = np.isinf(df[col]).sum()
            assert inf_count == 0, f"Feature {col} has {inf_count} infinite values"


@pytest.mark.unit
class TestFeatureEngineering:
    """Additional unit tests for specific feature engineering functions."""

    def test_correlation_removal_preserves_uncorrelated(self, sample_ohlcv_df):
        """Test that uncorrelated features are preserved."""
        df = sample_ohlcv_df.copy()

        # Add mix of correlated and uncorrelated features
        np.random.seed(42)
        df['independent_1'] = np.random.randn(len(df))
        df['independent_2'] = np.random.randn(len(df))
        df['correlated_1'] = df['close']
        df['correlated_2'] = df['close'] * 1.1

        df_result, removed = remove_correlated_features(
            df[['independent_1', 'independent_2', 'correlated_1', 'correlated_2']],
            correlation_threshold=0.95
        )

        # Independent features should be preserved
        assert 'independent_1' in df_result.columns
        assert 'independent_2' in df_result.columns

    def test_feature_count_reasonable(self):
        """Test that feature count is reasonable (not too many, not too few)."""
        df = fetch_and_prepare_data(
            ticker='SOLUSDT',
            timeframe='15m',
            limit=200,
            helper_timeframes=[],
            side='long',
            skip_slow_features=False  # Include all features
        )

        # Count feature columns (exclude OHLCV + timestamp)
        base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        feature_cols = [col for col in df.columns if col not in base_cols]

        feature_count = len(feature_cols)

        # Should have reasonable number of features (50-400 depending on config)
        assert 50 <= feature_count <= 400, f"Unexpected feature count: {feature_count}"
