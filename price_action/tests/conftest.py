# tests/conftest.py
"""
Pytest fixtures for testing adaptive trading bot.

Contains reusable test fixtures for:
- Sample OHLCV dataframes
- Test bot configurations
- Mock exchange adapters
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from bot import BotConfig


@pytest.fixture
def sample_ohlcv_df():
    """
    Sample OHLCV dataframe for testing.

    Returns:
        pd.DataFrame: Test dataframe with realistic OHLCV data
    """
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    np.random.seed(42)

    # Generate realistic price data with trends
    base_price = 100.0
    price_changes = np.random.randn(100).cumsum() * 0.5
    close = base_price + price_changes

    df = pd.DataFrame({
        'open': close + np.random.randn(100) * 0.2,
        'high': close + np.abs(np.random.randn(100)) * 0.5,
        'low': close - np.abs(np.random.randn(100)) * 0.5,
        'close': close,
        'volume': np.random.randint(1000, 10000, 100).astype(float),
        'turnover': close * np.random.randint(1000, 10000, 100).astype(float)
    }, index=dates)  # Set DatetimeIndex instead of timestamp column

    # Ensure OHLC logic is correct (high >= max(o,c), low <= min(o,c))
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def test_config():
    """
    Test bot configuration with safe defaults.

    Returns:
        BotConfig: Test configuration
    """
    config = BotConfig()
    config.TICKER = "SOLUSDT"
    config.TIMEFRAME = "15m"
    config.HELPER_TIMEFRAMES = []
    config.TRADE_SIZE_USD = 100.0
    config.LEVERAGE = 10
    config.TP_PCT = 0.03
    config.TSL_PCT = 0.01
    config.PROBABILITY_THRESHOLD = 0.7
    config.MIN_CONFIDENCE_RATIO = 1.5
    config.PARTIAL_TP_ENABLED = True
    config.DYNAMIC_TP_ENABLED = False
    config.HEDGE_MODE = False
    config.LIMIT_ORDER_MODE = False
    config.DCA_ENABLED = False

    return config


@pytest.fixture
def mock_bybit_adapter(mocker):
    """
    Mock Bybit adapter for testing without API calls.

    Args:
        mocker: pytest-mock fixture

    Returns:
        Mock: Mocked BybitAdapter instance
    """
    adapter = mocker.Mock()

    # Mock common methods
    adapter.get_position.return_value = None
    adapter.get_balance.return_value = {'available': 1000.0}
    adapter.get_ticker_price.return_value = 100.0
    adapter.market_open.return_value = {
        'orderId': 'test_order_123',
        'price': 100.0,
        'qty': 10.0,
        'side': 'Buy'
    }
    adapter.set_stop_loss.return_value = {'success': True}
    adapter.set_take_profit.return_value = {'success': True}

    return adapter


@pytest.fixture
def sample_position_long():
    """
    Sample LONG position for testing position management.

    Returns:
        dict: Position state dictionary
    """
    return {
        'side': 'Long',
        'entry_price': 100.0,
        'size': 1.0,
        'last_sl': 98.0,
        'initial_tp': 103.0,
        'highest_price': 100.0,
        'lowest_price': 100.0,
        'partial_tp_taken': False,
        'dynamic_tp_levels_taken': 0,
    }


@pytest.fixture
def sample_position_short():
    """
    Sample SHORT position for testing position management.

    Returns:
        dict: Position state dictionary
    """
    return {
        'side': 'Short',
        'entry_price': 100.0,
        'size': 1.0,
        'last_sl': 102.0,
        'initial_tp': 97.0,
        'highest_price': 100.0,
        'lowest_price': 100.0,
        'partial_tp_taken': False,
        'dynamic_tp_levels_taken': 0,
    }


@pytest.fixture
def sample_features_df(sample_ohlcv_df):
    """
    Sample dataframe with basic features for testing model predictions.

    Args:
        sample_ohlcv_df: OHLCV fixture

    Returns:
        pd.DataFrame: Dataframe with features
    """
    df = sample_ohlcv_df.copy()

    # Add basic features
    df['rsi_14'] = 50 + np.random.randn(len(df)) * 15  # RSI around 50
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['atr_14'] = df['high'] - df['low']  # Simplified ATR

    # Fill NaN values
    df = df.ffill().bfill()

    return df
