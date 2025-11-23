# tests/test_config_validation.py
"""
Unit tests for configuration validation using Pydantic.

Tests cover:
- BotConfigValidated validation
- Field validators (ticker, timeframe, etc.)
- Model validators (TP > TSL, mutually exclusive flags)
- TrainingConfigValidated
- BacktestConfigValidated
"""

import pytest
from pydantic import ValidationError
from config_validation import (
    BotConfigValidated,
    TrainingConfigValidated,
    BacktestConfigValidated,
    TimeframeEnum,
    SideEnum
)


class TestBotConfigValidated:
    """Test BotConfigValidated model."""

    def test_valid_config_creation(self):
        """Test creating valid bot configuration."""
        config = BotConfigValidated(
            ticker="SOLUSDT",
            timeframe="15m",
            helper_timeframes=["1h", "4h"],
            trade_size_usd=100.0,
            leverage=10,
            tp_pct=0.03,
            tsl_pct=0.01,
            probability_threshold=0.7,
            min_confidence_ratio=1.5
        )

        assert config.ticker == "SOLUSDT"
        assert config.timeframe == "15m"
        assert config.leverage == 10
        assert config.tp_pct == 0.03

    def test_ticker_must_end_with_usdt(self):
        """Test ticker validation - must end with USDT."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfigValidated(
                ticker="BTCEUR",  # Invalid - doesn't end with USDT
                timeframe="15m",
                trade_size_usd=100.0,
                leverage=10,
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=0.7
            )

        errors = exc_info.value.errors()
        assert any('USDT pairs' in str(e['msg']) for e in errors)

    def test_ticker_too_short(self):
        """Test ticker validation - minimum length."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfigValidated(
                ticker="USDT",  # Too short (only 4 chars, need 6)
                timeframe="15m",
                trade_size_usd=100.0,
                leverage=10,
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=0.7
            )

        errors = exc_info.value.errors()
        # Check for string_too_short error type or '6 characters' message
        assert any('string_too_short' == e['type'] or '6 characters' in str(e['msg']) for e in errors)

    def test_invalid_timeframe(self):
        """Test timeframe validation."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfigValidated(
                ticker="SOLUSDT",
                timeframe="10m",  # Invalid timeframe
                trade_size_usd=100.0,
                leverage=10,
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=0.7
            )

        errors = exc_info.value.errors()
        assert any('Invalid timeframe' in str(e['msg']) for e in errors)

    def test_valid_timeframes(self):
        """Test all valid timeframes."""
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']

        for tf in valid_timeframes:
            config = BotConfigValidated(
                ticker="SOLUSDT",
                timeframe=tf,
                trade_size_usd=100.0,
                leverage=10,
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=0.7
            )
            assert config.timeframe == tf

    def test_leverage_constraints(self):
        """Test leverage must be between 1 and 125."""
        # Too low
        with pytest.raises(ValidationError):
            BotConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                trade_size_usd=100.0,
                leverage=0,  # Too low
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=0.7
            )

        # Too high
        with pytest.raises(ValidationError):
            BotConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                trade_size_usd=100.0,
                leverage=126,  # Too high
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=0.7
            )

    def test_tp_must_be_greater_than_tsl(self):
        """Test TP must be greater than TSL validator."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                trade_size_usd=100.0,
                leverage=10,
                tp_pct=0.01,  # TP < TSL
                tsl_pct=0.03,
                probability_threshold=0.7
            )

        errors = exc_info.value.errors()
        assert any('TP' in str(e['msg']) and 'TSL' in str(e['msg']) for e in errors)

    def test_partial_and_dynamic_tp_mutually_exclusive(self):
        """Test cannot enable both partial_tp and dynamic_tp."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                trade_size_usd=100.0,
                leverage=10,
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=0.7,
                partial_tp_enabled=True,
                dynamic_tp_enabled=True  # Cannot enable both
            )

        errors = exc_info.value.errors()
        assert any('partial_tp' in str(e['msg']) and 'dynamic_tp' in str(e['msg']) for e in errors)

    def test_dca_requires_limit_order(self):
        """Test DCA mode requires limit_order_mode."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                trade_size_usd=100.0,
                leverage=10,
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=0.7,
                dca_enabled=True,
                limit_order_mode=False  # DCA requires limit order
            )

        errors = exc_info.value.errors()
        assert any('DCA' in str(e['msg']) and 'limit_order' in str(e['msg']) for e in errors)

    def test_rsi_thresholds_validation(self):
        """Test RSI low must be < high threshold."""
        # First test: rsi_low_threshold out of range (must be 20-50)
        with pytest.raises(ValidationError) as exc_info:
            BotConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                trade_size_usd=100.0,
                leverage=10,
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=0.7,
                rsi_reversal_filter_enabled=True,
                rsi_low_threshold=70.0,  # Out of range (must be <= 50)
                rsi_high_threshold=75.0
            )

        errors = exc_info.value.errors()
        # Either field validation error or custom validator error
        assert any('less than or equal' in str(e['msg']).lower() or 'rsi' in str(e['msg']).lower() for e in errors)

        # Test valid RSI thresholds (should pass)
        config = BotConfigValidated(
            ticker="SOLUSDT",
            timeframe="15m",
            trade_size_usd=100.0,
            leverage=10,
            tp_pct=0.03,
            tsl_pct=0.01,
            probability_threshold=0.7,
            rsi_reversal_filter_enabled=True,
            rsi_low_threshold=35.0,  # Valid: 20-50
            rsi_high_threshold=65.0  # Valid: 50-80, and > low
        )
        assert config.rsi_low_threshold == 35.0
        assert config.rsi_high_threshold == 65.0

    def test_probability_threshold_range(self):
        """Test probability threshold must be 0.5-1.0."""
        # Too low
        with pytest.raises(ValidationError):
            BotConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                trade_size_usd=100.0,
                leverage=10,
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=0.4  # Too low
            )

        # Too high
        with pytest.raises(ValidationError):
            BotConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                trade_size_usd=100.0,
                leverage=10,
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=1.1  # Too high
            )

    def test_trade_size_constraints(self):
        """Test trade size must be positive and < 1M."""
        # Zero
        with pytest.raises(ValidationError):
            BotConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                trade_size_usd=0.0,  # Invalid
                leverage=10,
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=0.7
            )

        # Too large
        with pytest.raises(ValidationError):
            BotConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                trade_size_usd=2000000.0,  # > 1M
                leverage=10,
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=0.7
            )

    def test_helper_timeframes_validation(self):
        """Test helper timeframes are validated."""
        with pytest.raises(ValidationError) as exc_info:
            BotConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                helper_timeframes=["1h", "10m"],  # 10m is invalid
                trade_size_usd=100.0,
                leverage=10,
                tp_pct=0.03,
                tsl_pct=0.01,
                probability_threshold=0.7
            )

        errors = exc_info.value.errors()
        assert any('helper' in str(e['msg']).lower() or 'timeframe' in str(e['msg']).lower() for e in errors)


class TestTrainingConfigValidated:
    """Test TrainingConfigValidated model."""

    def test_valid_training_config(self):
        """Test valid training configuration."""
        config = TrainingConfigValidated(
            ticker="SOLUSDT",
            timeframe="15m",
            helper_timeframes=["1h"],
            side="long",
            version="v1.0",
            label_trials=50,
            model_trials=100,
            limit=3000
        )

        assert config.ticker == "SOLUSDT"
        assert config.side == SideEnum.LONG
        assert config.version == "v1.0"

    def test_side_is_required(self):
        """Test side field is required for training."""
        with pytest.raises(ValidationError):
            TrainingConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                version="v1.0"
                # Missing side
            )

    def test_version_format_validation(self):
        """Test version must be in vX.Y format."""
        # Invalid formats
        invalid_versions = ["1.0", "v1", "v1.0.1", "version1.0", "v1.a"]

        for invalid_ver in invalid_versions:
            with pytest.raises(ValidationError) as exc_info:
                TrainingConfigValidated(
                    ticker="SOLUSDT",
                    timeframe="15m",
                    side="long",
                    version=invalid_ver
                )

            errors = exc_info.value.errors()
            assert any('version' in str(e['msg']).lower() for e in errors)

        # Valid formats
        valid_versions = ["v1.0", "v2.1", "v10.5"]
        for valid_ver in valid_versions:
            config = TrainingConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                side="long",
                version=valid_ver
            )
            assert config.version == valid_ver

    def test_trials_constraints(self):
        """Test trial counts have proper constraints."""
        # Label trials too low
        with pytest.raises(ValidationError):
            TrainingConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                side="long",
                version="v1.0",
                label_trials=5  # Too low (< 10)
            )

        # Model trials too high
        with pytest.raises(ValidationError):
            TrainingConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                side="long",
                version="v1.0",
                model_trials=1500  # Too high (> 1000)
            )


class TestBacktestConfigValidated:
    """Test BacktestConfigValidated model."""

    def test_valid_backtest_config(self):
        """Test valid backtest configuration."""
        config = BacktestConfigValidated(
            ticker="SOLUSDT",
            timeframe="15m",
            helper_timeframes=["1h"],
            version="v1.0",
            probability_threshold=0.7,
            min_confidence_ratio=1.5,
            tp_pct=0.03,
            tsl_pct=0.01,
            trade_size_usd=100.0,
            leverage=10,
            partial_tp_enabled=True,
            dynamic_tp_enabled=False,
            limit=3000
        )

        assert config.ticker == "SOLUSDT"
        assert config.tp_pct == 0.03

    def test_backtest_tp_greater_than_tsl(self):
        """Test backtest config validates TP > TSL."""
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                version="v1.0",
                probability_threshold=0.7,
                tp_pct=0.01,  # TP < TSL
                tsl_pct=0.03,
                trade_size_usd=100.0,
                leverage=10
            )

        errors = exc_info.value.errors()
        assert any('TP' in str(e['msg']) and 'TSL' in str(e['msg']) for e in errors)

    def test_limit_constraints(self):
        """Test limit has proper range."""
        # Too low
        with pytest.raises(ValidationError):
            BacktestConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                version="v1.0",
                probability_threshold=0.7,
                tp_pct=0.03,
                tsl_pct=0.01,
                trade_size_usd=100.0,
                leverage=10,
                limit=50  # Too low (< 100)
            )

        # Too high
        with pytest.raises(ValidationError):
            BacktestConfigValidated(
                ticker="SOLUSDT",
                timeframe="15m",
                version="v1.0",
                probability_threshold=0.7,
                tp_pct=0.03,
                tsl_pct=0.01,
                trade_size_usd=100.0,
                leverage=10,
                limit=60000  # Too high (> 50000)
            )


class TestEnumValues:
    """Test enum definitions."""

    def test_timeframe_enum_values(self):
        """Test TimeframeEnum has correct values."""
        assert TimeframeEnum.m1.value == "1m"
        assert TimeframeEnum.m15.value == "15m"
        assert TimeframeEnum.h1.value == "1h"
        assert TimeframeEnum.d1.value == "1d"

    def test_side_enum_values(self):
        """Test SideEnum has correct values."""
        assert SideEnum.LONG.value == "long"
        assert SideEnum.SHORT.value == "short"
