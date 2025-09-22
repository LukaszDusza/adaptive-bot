# test_live_api.py
"""
Test script for live trading API interface.
Tests the communication between position_manager and live_trader components.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

import config
from services.analysis_service import AnalysisService
from logic.position_manager import PositionManager


def create_mock_analysis():
    """Create mock analysis data for testing."""
    return {
        'current_price': 4500.0,
        'atr_value_5m': 45.0,
        'expert_opinions': {
            'momentum': {
                'prediction': 1,
                'confidence': 0.92
            },
            'reversion': {
                'prediction': 1,
                'confidence': 0.85
            },
            'pa': {
                'prediction': 0,
                'confidence': 0.65
            }
        }
    }


def create_mock_candle():
    """Create mock candle data for testing."""
    return pd.Series({
        'open': 4490.0,
        'high': 4510.0,
        'low': 4485.0,
        'close': 4500.0,
        'volume': 1000.0,
        'ATRr_14_5m': 45.0
    }, name=datetime.now())


def test_ml_predictions():
    """Test ML predictions API."""
    print("=== TESTING ML PREDICTIONS API ===")
    
    manager = PositionManager(config)
    analysis = create_mock_analysis()
    
    predictions = manager.get_ml_predictions(analysis)
    
    print(f"Votes - Long: {predictions['votes_long']}, Short: {predictions['votes_short']}")
    print(f"Recommendation: {predictions['recommendation']['signal']}")
    print(f"Consensus: {predictions['recommendation']['consensus']}")
    
    for expert, pred in predictions['predictions'].items():
        print(f"{expert}: prediction={pred['prediction']}, confidence={pred['confidence']:.3f}, eligible={pred['vote_eligible']}")
    
    assert predictions['votes_long'] == 2
    assert predictions['votes_short'] == 0
    assert predictions['recommendation']['signal'] == 'LONG'
    
    print("✅ ML predictions API test passed\n")


def test_trading_signal_long_entry():
    """Test trading signal for long entry."""
    print("=== TESTING LONG ENTRY SIGNAL ===")
    
    manager = PositionManager(config)
    analysis = create_mock_analysis()
    candle = create_mock_candle()
    capital = 1000.0
    
    signal = manager.get_trading_signal(candle, analysis, capital)
    
    print(f"Action: {signal['action']}")
    print(f"Strategy: {signal.get('strategy', 'N/A')}")
    print(f"Entry price: {signal.get('entry_price', 'N/A')}")
    print(f"Stop loss: {signal.get('stop_loss', 'N/A')}")
    print(f"Take profit: {signal.get('take_profit', 'N/A')}")
    print(f"Size: {signal.get('size', 'N/A')}")
    
    assert signal['action'] == 'OPEN_LONG'
    assert signal['strategy'] == 'long'
    assert 'entry_price' in signal
    assert 'stop_loss' in signal
    assert 'take_profit' in signal
    assert 'size' in signal
    
    print("✅ Long entry signal test passed\n")
    return signal


def test_trading_signal_short_entry():
    """Test trading signal for short entry."""
    print("=== TESTING SHORT ENTRY SIGNAL ===")
    
    manager = PositionManager(config)
    
    # Create analysis favoring short
    analysis = {
        'current_price': 4500.0,
        'atr_value_5m': 45.0,
        'expert_opinions': {
            'momentum': {
                'prediction': 0,
                'confidence': 0.92
            },
            'reversion': {
                'prediction': 0,
                'confidence': 0.85
            },
            'pa': {
                'prediction': 1,
                'confidence': 0.65
            }
        }
    }
    
    candle = create_mock_candle()
    capital = 1000.0
    
    signal = manager.get_trading_signal(candle, analysis, capital)
    
    print(f"Action: {signal['action']}")
    print(f"Strategy: {signal.get('strategy', 'N/A')}")
    print(f"Entry price: {signal.get('entry_price', 'N/A')}")
    print(f"Stop loss: {signal.get('stop_loss', 'N/A')}")
    print(f"Take profit: {signal.get('take_profit', 'N/A')}")
    
    assert signal['action'] == 'OPEN_SHORT'
    assert signal['strategy'] == 'short'
    
    print("✅ Short entry signal test passed\n")
    return signal


def test_position_management():
    """Test position management instructions."""
    print("=== TESTING POSITION MANAGEMENT ===")
    
    manager = PositionManager(config)
    
    # First create a position
    position_data = {
        'strategy': 'long',
        'entry_date': datetime.now(),
        'entry_price': 4500.0,
        'size': 0.1,
        'current_sl_price': 4455.0,
        'tp_price': 4612.5,
        'breakeven_trigger_price': 4556.25,
        'breakeven_sl_price': 4500.0,
        'trailing_trigger_price': 4567.5,
        'conf_momentum': 0.92,
        'conf_reversion': 0.85,
        'conf_pa': 0.65
    }
    
    manager.update_position_from_live_data(position_data)
    
    # Test position status
    status = manager.get_position_status()
    print(f"Has position: {status['has_position']}")
    print(f"Position strategy: {status['position']['strategy']}")
    print(f"Entry price: {status['position']['entry_price']}")
    
    assert status['has_position'] == True
    assert status['position']['strategy'] == 'long'
    
    # Test with candle that triggers breakeven
    high_candle = create_mock_candle()
    high_candle['high'] = 4570.0  # Above breakeven trigger
    
    analysis = create_mock_analysis()
    signal = manager.get_trading_signal(high_candle, analysis, 1000.0)
    
    print(f"Instructions: {len(signal['instructions'])}")
    for instruction in signal['instructions']:
        print(f"  - {instruction['type']}: {instruction.get('reason', 'N/A')}")
    
    assert len(signal['instructions']) > 0
    assert any(inst['type'] == 'MOVE_SL_TO_BREAKEVEN' for inst in signal['instructions'])
    
    print("✅ Position management test passed\n")


def test_hold_signal():
    """Test HOLD signal when no clear direction."""
    print("=== TESTING HOLD SIGNAL ===")
    
    manager = PositionManager(config)
    
    # Create weak/conflicting signals
    analysis = {
        'current_price': 4500.0,
        'atr_value_5m': 45.0,
        'expert_opinions': {
            'momentum': {
                'prediction': 1,
                'confidence': 0.75  # Below threshold
            },
            'reversion': {
                'prediction': 0,
                'confidence': 0.80  # Below threshold
            },
            'pa': {
                'prediction': 1,
                'confidence': 0.65  # Below threshold but above PA threshold
            }
        }
    }
    
    candle = create_mock_candle()
    signal = manager.get_trading_signal(candle, analysis, 1000.0)
    
    print(f"Action: {signal['action']}")
    
    assert signal['action'] == 'HOLD'
    
    print("✅ Hold signal test passed\n")


def test_exit_signal():
    """Test exit signal generation."""
    print("=== TESTING EXIT SIGNAL ===")
    
    manager = PositionManager(config)
    
    # Create a long position
    position_data = {
        'strategy': 'long',
        'entry_date': datetime.now(),
        'entry_price': 4500.0,
        'size': 0.1,
        'current_sl_price': 4455.0,
        'tp_price': 4612.5,
        'breakeven_trigger_price': 4556.25,
        'breakeven_sl_price': 4500.0,
        'trailing_trigger_price': 4567.5,
        'conf_momentum': 0.92,
        'conf_reversion': 0.85,
        'conf_pa': 0.65,
        'opposing_signal_count': 2  # Already has opposing signals
    }
    
    manager.update_position_from_live_data(position_data)
    
    # Create analysis with opposing signals (short signals for long position)
    analysis = {
        'current_price': 4480.0,
        'atr_value_5m': 45.0,
        'expert_opinions': {
            'momentum': {
                'prediction': 0,  # Opposing signal
                'confidence': 0.92
            },
            'reversion': {
                'prediction': 0,  # Opposing signal
                'confidence': 0.85
            },
            'pa': {
                'prediction': 1,
                'confidence': 0.65
            }
        }
    }
    
    candle = create_mock_candle()
    candle['close'] = 4480.0
    
    signal = manager.get_trading_signal(candle, analysis, 1000.0)
    
    print(f"Action: {signal['action']}")
    print(f"Exit reason: {signal.get('exit_reason', 'N/A')}")
    
    assert signal['action'] == 'CLOSE'
    assert 'exit_reason' in signal
    
    print("✅ Exit signal test passed\n")


def run_all_tests():
    """Run all API tests."""
    print("🚀 STARTING LIVE TRADING API TESTS\n")
    
    try:
        test_ml_predictions()
        test_trading_signal_long_entry()
        test_trading_signal_short_entry()
        test_position_management()
        test_hold_signal()
        test_exit_signal()
        
        print("🎉 ALL TESTS PASSED! The live trading API is working correctly.")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        raise
    
    print("\n" + "="*50)
    print("API INTERFACE READY FOR LIVE TRADING")
    print("="*50)


if __name__ == "__main__":
    run_all_tests()