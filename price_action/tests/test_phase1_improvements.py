"""
TEST SCRIPT FOR PHASE 1 IMPROVEMENTS
======================================

Tests:
1. Profit-Aware Optimizer module
2. Adaptive Labels module
3. Rolling Window Trainer integration

Usage:
    cd price_action
    python tests/test_phase1_improvements.py

Author: Łukasz + Claude
Created: 2024-11-23
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

logger = logging.getLogger(__name__)


def test_profit_aware_optimizer():
    """Test profit-aware optimization functions"""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Profit-Aware Optimizer (FIXED - No Look-Ahead Bias)")
    logger.info("="*80)

    from profit_aware_optimizer import (
        simulate_trades_forward,
        calculate_trading_metrics,
        profit_aware_objective_score
    )

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    # 70% HOLD, 30% SIGNAL (realistic imbalance)
    y_pred = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    # Prices (random walk)
    prices = pd.Series(100 * np.exp(np.random.randn(n_samples).cumsum() * 0.01))

    try:
        # Test trade simulation (FIXED VERSION - no y_true!)
        logger.info("\n1. Testing simulate_trades_forward() [NO LOOK-AHEAD BIAS]...")
        returns = simulate_trades_forward(
            y_pred=y_pred,
            prices=prices,
            side=1,  # LONG
            tp_pct=0.03,
            sl_pct=0.01,
            fee_pct=0.0006
        )
        logger.info(f"✓ Generated {len(returns)} trade returns")
        logger.info(f"   Mean return: {returns[returns != 0].mean()*100:.2f}%")

        # Test metrics calculation
        logger.info("\n2. Testing calculate_trading_metrics()...")
        metrics = calculate_trading_metrics(returns)
        logger.info(f"✓ Calculated metrics:")
        logger.info(f"   Sharpe: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"   Total PnL: {metrics['total_pnl']*100:.1f}%")
        logger.info(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
        logger.info(f"   Trades: {metrics['n_trades']}")

        # Test objective score (FIXED VERSION - no y_true!)
        logger.info("\n3. Testing profit_aware_objective_score() [NO LOOK-AHEAD BIAS]...")
        score = profit_aware_objective_score(
            y_pred=y_pred,
            prices=prices,
            side=1,  # LONG
            tp_pct=0.03,
            sl_pct=0.01,
            fee_pct=0.0006
        )
        logger.info(f"✓ Objective score: {score:.3f}")

        logger.info("\n✅ Profit-Aware Optimizer: ALL TESTS PASSED")
        return True

    except Exception as e:
        logger.error(f"\n❌ Profit-Aware Optimizer: TEST FAILED")
        logger.error(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_labels():
    """Test adaptive labels module"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Adaptive Labels")
    logger.info("="*80)

    from adaptive_labels import (
        calculate_atr_normalized,
        calculate_trend_strength,
        get_adaptive_triple_barrier_labels
    )

    try:
        # Generate synthetic OHLCV data
        logger.info("\n1. Generating synthetic OHLCV data...")
        np.random.seed(42)
        n_candles = 5000

        # Random walk prices
        base_price = 100
        returns = np.random.randn(n_candles) * 0.02  # 2% daily vol
        close = base_price * np.exp(returns.cumsum())

        # OHLC with realistic spread
        high = close * (1 + np.abs(np.random.randn(n_candles)) * 0.01)
        low = close * (1 - np.abs(np.random.randn(n_candles)) * 0.01)
        open_prices = close * (1 + np.random.randn(n_candles) * 0.005)
        volume = np.random.randint(1000, 10000, n_candles)

        # Create DataFrame
        df = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        df.index = pd.date_range('2024-01-01', periods=n_candles, freq='15min')

        logger.info(f"✓ Generated {len(df)} candles")

        # Test ATR calculation
        logger.info("\n2. Testing calculate_atr_normalized()...")
        atr = calculate_atr_normalized(df)
        logger.info(f"✓ Calculated ATR")
        logger.info(f"   Mean ATR: {atr.mean()*100:.2f}%")
        logger.info(f"   Min ATR: {atr.min()*100:.2f}%")
        logger.info(f"   Max ATR: {atr.max()*100:.2f}%")

        # Add ATR to df
        df['atr_normalized'] = atr

        # Test trend strength
        logger.info("\n3. Testing calculate_trend_strength()...")
        trend = calculate_trend_strength(df)
        logger.info(f"✓ Calculated trend strength")
        logger.info(f"   Mean: {trend.mean():.3f}")
        logger.info(f"   Std: {trend.std():.3f}")

        # Test adaptive labels
        logger.info("\n4. Testing get_adaptive_triple_barrier_labels()...")
        labels = get_adaptive_triple_barrier_labels(
            df=df,
            t_events=df.index[1000:4000],  # Middle portion
            base_barrier_pct=0.015,
            atr_multiplier=1.5,
            base_time_limit=24,
            verbose=True
        )

        logger.info(f"✓ Generated {len(labels)} labels")
        logger.info(f"   Label distribution:")
        for label, count in labels.value_counts().items():
            logger.info(f"     {label}: {count} ({count/len(labels)*100:.1f}%)")

        logger.info("\n✅ Adaptive Labels: ALL TESTS PASSED")
        return True

    except Exception as e:
        logger.error(f"\n❌ Adaptive Labels: TEST FAILED")
        logger.error(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rolling_window_config():
    """Test rolling window configuration"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Rolling Window Trainer Config")
    logger.info("="*80)

    from rolling_window_trainer import RollingWindowConfig, get_days_since_last_train

    try:
        # Test config creation
        logger.info("\n1. Testing RollingWindowConfig()...")
        config = RollingWindowConfig(
            window_months=6,
            retrain_interval_days=30,
            ticker="SOLUSDT",
            side="long"
        )

        logger.info(f"✓ Config created:")
        logger.info(f"   Window: {config.window_months} months")
        logger.info(f"   Retrain interval: {config.retrain_interval_days} days")
        logger.info(f"   Version name: {config.version_name}")
        logger.info(f"   Metadata file: {config.metadata_file}")

        # Test days since last train (should be None for new model)
        logger.info("\n2. Testing get_days_since_last_train()...")
        days = get_days_since_last_train("SOLUSDT", "long")
        if days is None:
            logger.info(f"✓ No previous training found (expected for new model)")
        else:
            logger.info(f"✓ Days since last train: {days}")

        logger.info("\n✅ Rolling Window Trainer Config: ALL TESTS PASSED")
        return True

    except Exception as e:
        logger.error(f"\n❌ Rolling Window Trainer Config: TEST FAILED")
        logger.error(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 1 IMPROVEMENTS - TEST SUITE")
    logger.info("="*80)

    results = {}

    # Run tests
    results['profit_aware'] = test_profit_aware_optimizer()
    results['adaptive_labels'] = test_adaptive_labels()
    results['rolling_window'] = test_rolling_window_config()

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())

    logger.info("="*80)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("\nPhase 1 improvements are ready to use:")
        logger.info("  1. Profit-Aware Optimization ✓")
        logger.info("  2. Adaptive Label Parameters ✓")
        logger.info("  3. Rolling Window Training ✓")
        logger.info("\nNext steps:")
        logger.info("  • Test with real data: python rolling_window_trainer.py --force")
        logger.info("  • Train model: python main.py --train --side long --version rolling_20241123")
    else:
        logger.info("❌ SOME TESTS FAILED - Fix errors before using")

    logger.info("="*80)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
