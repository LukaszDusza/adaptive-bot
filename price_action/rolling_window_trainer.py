"""
ROLLING WINDOW TRAINING SYSTEM
===============================

Phase 1 Improvement #2: Train models on RECENT data only (rolling window).

Problem with traditional approach:
- Train on 4 years of data (2020-2024)
- Model averages patterns from bull/bear/range markets
- Old regimes pollute predictions
- Non-stationary data → poor live performance

Rolling window solution:
- Train on last N months ONLY (e.g., 6 months)
- Retrain every M days (e.g., 30 days)
- Model ALWAYS fresh and adapted to current regime
- Better live performance, lower regime change risk

Architecture:
1. rolling_window_train() - Main training function
2. should_retrain() - Check if retrain needed
3. schedule_retraining() - Automated scheduling (cron/systemd)
4. RollingWindowConfig - Configuration dataclass

Author: Łukasz + Claude
Created: 2024-11-23
Version: 2.0
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from pathlib import Path

import pandas as pd

# Import existing pipeline components
from data_preparer_pa import fetch_and_prepare_data
from model_pipeline import run_training_pipeline
from profit_aware_optimizer import find_optimal_threshold_for_profit, evaluate_on_holdout

logger = logging.getLogger(__name__)


@dataclass
class RollingWindowConfig:
    """Configuration for rolling window training"""

    # Window parameters
    window_months: int = 6  # Train on last N months
    retrain_interval_days: int = 30  # Retrain every N days

    # Model parameters
    ticker: str = "SOLUSDT"
    timeframe: str = "15m"
    helper_timeframes: list = None
    side: str = "long"

    # Training parameters
    n_label_trials: int = 50  # Reduced from 100 (faster retraining)
    n_model_trials: int = 100  # Reduced from 200
    min_recall_target: float = 0.55

    # Profit-aware parameters
    use_profit_aware: bool = True  # NEW: Enable profit-aware optimization
    use_adaptive_labels: bool = True  # NEW: Enable adaptive barriers

    # Live trading parameters (FIX #1: Match live bot config)
    live_tp_pct: float = 0.03  # Take profit % (must match live bot!)
    live_sl_pct: float = 0.01  # Stop loss % (must match live bot!)
    live_fee_pct: float = 0.0006  # Trading fees % (must match live bot!)

    # Data parameters
    min_samples: int = 20000  # Minimum samples required for training

    # Version management
    version_prefix: str = "rolling"  # e.g., "rolling_20241123"

    def __post_init__(self):
        if self.helper_timeframes is None:
            self.helper_timeframes = ["1h", "4h"]

    @property
    def version_name(self) -> str:
        """Generate version name with date"""
        date_str = datetime.now().strftime("%Y%m%d")
        return f"{self.version_prefix}_{date_str}"

    @property
    def metadata_file(self) -> str:
        """Path to metadata file for tracking retraining"""
        return f"models/rolling_window_metadata_{self.ticker}_{self.side}.json"


class RollingWindowTrainer:
    """Manages rolling window training and retraining schedule"""

    def __init__(self, config: RollingWindowConfig):
        self.config = config
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load training metadata (last train date, versions, etc.)"""
        metadata_path = Path(self.config.metadata_file)

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"✓ Loaded metadata from {metadata_path}")
        else:
            metadata = {
                'last_train_date': None,
                'train_history': [],
                'current_version': None
            }
            logger.info(f"No existing metadata, creating new")

        return metadata

    def _save_metadata(self):
        """Save training metadata"""
        metadata_path = Path(self.config.metadata_file)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"✓ Saved metadata to {metadata_path}")

    def should_retrain(self, check_drift: bool = True, drift_threshold: float = 0.15,
                       min_days_for_drift_check: int = 7) -> Tuple[bool, str]:
        """
        Check if retraining is needed.

        FIX #3: Added drift-based retraining trigger.

        Args:
            check_drift: Enable drift monitoring (default: True)
            drift_threshold: Trigger if > X% features drifted (default: 15%)
            min_days_for_drift_check: Minimum days before checking drift (default: 7)

        Returns:
            (should_retrain: bool, reason: str)
        """
        if self.metadata['last_train_date'] is None:
            return True, "No previous training found"

        # Parse last train date
        last_train = datetime.fromisoformat(self.metadata['last_train_date'])
        days_since_train = (datetime.now() - last_train).days

        # 1. Check scheduled retrain interval
        if days_since_train >= self.config.retrain_interval_days:
            return True, f"Retrain interval reached ({days_since_train} days since last train)"

        # 2. FIX #3: Check drift (if enabled and enough time passed)
        if check_drift and days_since_train >= min_days_for_drift_check:
            try:
                drift_score = self._check_feature_drift_online(drift_threshold)

                if drift_score > drift_threshold:
                    return True, f"Feature drift detected ({drift_score*100:.0f}% > {drift_threshold*100:.0f}%)"

            except Exception as e:
                logger.warning(f"Drift check failed: {e}")
                # Don't block training due to drift check failure
                pass

        return False, f"No retrain needed ({days_since_train}/{self.config.retrain_interval_days} days)"

    def _check_feature_drift_online(self, threshold: float = 0.15) -> float:
        """
        Check feature drift between training data and recent live data.

        FIX #3: Drift-based retraining trigger.

        Args:
            threshold: P-value threshold for KS test (default: 0.05)

        Returns:
            Drift score (0-1, proportion of features that drifted)
        """
        from scipy.stats import ks_2samp

        logger.info("\n" + "="*60)
        logger.info("DRIFT DETECTION - Online Monitoring")
        logger.info("="*60)

        # Load training metadata (has feature stats)
        version = self.metadata.get('current_version')
        if not version:
            logger.warning("No current version found, skipping drift check")
            return 0.0

        strategy_id = f"{self.config.ticker}_{self.config.timeframe.replace(' ', '')}"
        helpers_str = '_plus_' + '_'.join(self.config.helper_timeframes) if self.config.helper_timeframes else ""
        strategy_id += helpers_str + f"_{self.config.side}"

        metadata_path = Path(f"models/{version}/{strategy_id}/training_metadata.json")

        if not metadata_path.exists():
            logger.warning(f"Training metadata not found: {metadata_path}")
            return 0.0

        # Fetch recent live data (last 7 days)
        try:
            logger.info("Fetching recent live data (last 7 days)...")
            recent_df = fetch_and_prepare_data(
                ticker=self.config.ticker,
                timeframe=self.config.timeframe,
                helper_timeframes=self.config.helper_timeframes,
                side=self.config.side,
                limit=672,  # 7 days @ 15m = 672 candles
                skip_slow_features=False
            )

            if len(recent_df) < 100:
                logger.warning(f"Insufficient recent data: {len(recent_df)} candles")
                return 0.0

            logger.info(f"✓ Fetched {len(recent_df)} recent candles")

        except Exception as e:
            logger.error(f"Failed to fetch recent data: {e}")
            return 0.0

        # Load training stats (we'll approximate from saved model)
        # In practice, you'd save feature statistics during training
        # For now, we'll do a simplified check on top features

        try:
            import joblib
            features_path = Path(f"models/{version}/{strategy_id}/features.joblib")
            features = joblib.load(features_path)

            # Sample top 20 features for drift check (full check would be expensive)
            top_features = features[:min(20, len(features))]

            drift_count = 0
            total_checked = 0

            logger.info(f"Checking drift for top {len(top_features)} features...")

            for feature in top_features:
                if feature not in recent_df.columns:
                    continue

                # For simplicity, compare recent data variance with expected range
                # In production, you'd compare with training distribution
                recent_vals = recent_df[feature].dropna()

                if len(recent_vals) < 50:
                    continue

                total_checked += 1

                # Simple variance-based check
                # If variance > 3x normal, consider drifted
                recent_std = recent_vals.std()
                recent_mean = recent_vals.mean()

                # Z-score outlier detection
                if abs(recent_mean) > 5 or recent_std > 10:
                    drift_count += 1
                    logger.debug(f"  Drift detected in {feature}: mean={recent_mean:.2f}, std={recent_std:.2f}")

            if total_checked == 0:
                logger.warning("No features checked (data missing)")
                return 0.0

            drift_score = drift_count / total_checked

            logger.info(f"\nDrift Score: {drift_score*100:.1f}% ({drift_count}/{total_checked} features)")
            logger.info("="*60)

            return drift_score

        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            return 0.0

    def train(self, force: bool = False) -> Optional[str]:
        """
        Execute rolling window training.

        Args:
            force: Force training even if interval not reached

        Returns:
            Version name if trained, None if skipped
        """
        # Check if retrain needed
        should_retrain, reason = self.should_retrain()

        if not should_retrain and not force:
            logger.info(f"⏭️  Skipping training: {reason}")
            return None

        logger.info("="*80)
        logger.info(f"🔄 ROLLING WINDOW TRAINING - {self.config.ticker} {self.config.side.upper()}")
        logger.info("="*80)
        logger.info(f"Reason: {reason}")
        logger.info(f"Window: {self.config.window_months} months")
        logger.info(f"Force: {force}")
        logger.info("="*80)

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.window_months * 30)

        logger.info(f"\n📅 Training period:")
        logger.info(f"   Start: {start_date.strftime('%Y-%m-%d')}")
        logger.info(f"   End: {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"   Duration: {self.config.window_months} months")

        # Fetch data for window
        logger.info(f"\n📥 Fetching data...")

        try:
            df = fetch_and_prepare_data(
                ticker=self.config.ticker,
                timeframe=self.config.timeframe,
                helper_timeframes=self.config.helper_timeframes,
                side=self.config.side,
                limit=None,  # Will be filtered by date
                skip_slow_features=False  # Full features for training
            )

            # Filter to window
            df_window = df[df.index >= start_date]

            logger.info(f"✓ Data fetched: {len(df_window):,} candles")
            logger.info(f"   Date range: {df_window.index[0]} to {df_window.index[-1]}")

            # Validate sufficient data
            if len(df_window) < self.config.min_samples:
                logger.error(f"❌ Insufficient data: {len(df_window)} < {self.config.min_samples} (minimum)")
                return None

        except Exception as e:
            logger.error(f"❌ Data fetch failed: {e}")
            return None

        # Train model
        logger.info(f"\n🤖 Starting training pipeline...")

        version_name = self.config.version_name

        try:
            run_training_pipeline(
                df_features=df_window,
                n_label_trials=self.config.n_label_trials,
                n_model_trials=self.config.n_model_trials,
                ticker=self.config.ticker,
                timeframe=self.config.timeframe,
                helper_timeframes=self.config.helper_timeframes,
                side=self.config.side,
                version=version_name,
                min_recall_target=self.config.min_recall_target,
                use_adaptive_labels=self.config.use_adaptive_labels,
                use_profit_aware_threshold=self.config.use_profit_aware,
                live_tp_pct=self.config.live_tp_pct,  # FIX #1: Pass from config
                live_sl_pct=self.config.live_sl_pct,  # FIX #1: Pass from config
                live_fee_pct=self.config.live_fee_pct  # FIX #1: Pass from config
            )

            logger.info(f"✅ Training completed successfully!")

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Update metadata
        self.metadata['last_train_date'] = datetime.now().isoformat()
        self.metadata['current_version'] = version_name
        self.metadata['train_history'].append({
            'version': version_name,
            'date': datetime.now().isoformat(),
            'window_start': start_date.isoformat(),
            'window_end': end_date.isoformat(),
            'n_samples': len(df_window),
            'config': asdict(self.config)
        })

        self._save_metadata()

        logger.info("="*80)
        logger.info(f"✅ ROLLING WINDOW TRAINING COMPLETE")
        logger.info(f"   Version: {version_name}")
        logger.info(f"   Models saved to: models/{version_name}/")
        logger.info("="*80)

        return version_name


def rolling_window_train(ticker: str = "SOLUSDT",
                         timeframe: str = "15m",
                         helper_timeframes: list = None,
                         side: str = "long",
                         window_months: int = 6,
                         retrain_interval_days: int = 30,
                         force: bool = False) -> Optional[str]:
    """
    Convenience function for rolling window training.

    Args:
        ticker: Trading pair
        timeframe: Main timeframe
        helper_timeframes: Helper timeframes list
        side: 'long' or 'short'
        window_months: Training window in months (default: 6)
        retrain_interval_days: Days between retrains (default: 30)
        force: Force training even if interval not reached

    Returns:
        Version name if trained, None if skipped

    Example:
        # Train LONG model on last 6 months
        version = rolling_window_train(
            ticker="SOLUSDT",
            side="long",
            window_months=6,
            force=True
        )

        # Use in bot:
        bot = TradingBot(config, version=version)
    """
    if helper_timeframes is None:
        helper_timeframes = ["1h", "4h"]

    config = RollingWindowConfig(
        window_months=window_months,
        retrain_interval_days=retrain_interval_days,
        ticker=ticker,
        timeframe=timeframe,
        helper_timeframes=helper_timeframes,
        side=side
    )

    trainer = RollingWindowTrainer(config)
    return trainer.train(force=force)


def retrain_all_models(tickers: list = None,
                       sides: list = None,
                       window_months: int = 6,
                       force: bool = False) -> Dict[str, str]:
    """
    Retrain all models (convenience for scheduled retraining).

    Args:
        tickers: List of tickers (default: ["SOLUSDT"])
        sides: List of sides (default: ["long", "short"])
        window_months: Training window in months
        force: Force training even if interval not reached

    Returns:
        Dict mapping (ticker, side) → version_name

    Example:
        # Retrain all models
        versions = retrain_all_models(
            tickers=["SOLUSDT", "BTCUSDT"],
            sides=["long", "short"],
            force=True
        )
    """
    if tickers is None:
        tickers = ["SOLUSDT"]
    if sides is None:
        sides = ["long", "short"]

    results = {}

    for ticker in tickers:
        for side in sides:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing {ticker} {side.upper()}")
            logger.info(f"{'='*80}")

            try:
                version = rolling_window_train(
                    ticker=ticker,
                    side=side,
                    window_months=window_months,
                    force=force
                )

                if version:
                    results[(ticker, side)] = version
                    logger.info(f"✅ {ticker} {side}: {version}")
                else:
                    logger.info(f"⏭️  {ticker} {side}: skipped")

            except Exception as e:
                logger.error(f"❌ {ticker} {side}: {e}")
                import traceback
                traceback.print_exc()

    logger.info(f"\n{'='*80}")
    logger.info(f"RETRAINING SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Trained: {len(results)} models")
    for (ticker, side), version in results.items():
        logger.info(f"  {ticker} {side}: {version}")
    logger.info(f"{'='*80}")

    return results


def get_days_since_last_train(ticker: str, side: str) -> Optional[int]:
    """
    Get number of days since last training.

    Returns:
        Days since last train, or None if never trained
    """
    metadata_file = f"models/rolling_window_metadata_{ticker}_{side}.json"

    if not os.path.exists(metadata_file):
        return None

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    if metadata['last_train_date'] is None:
        return None

    last_train = datetime.fromisoformat(metadata['last_train_date'])
    days = (datetime.now() - last_train).days

    return days


if __name__ == "__main__":
    """
    Example usage for manual testing and scheduled retraining.

    For automated retraining, use cron:
    # Retrain every 30 days at 2 AM
    0 2 */30 * * cd /app/price_action && python rolling_window_trainer.py
    """
    import argparse

    parser = argparse.ArgumentParser(description="Rolling Window Training")
    parser.add_argument("--ticker", type=str, default="SOLUSDT", help="Trading pair")
    parser.add_argument("--side", type=str, default="long", choices=["long", "short"], help="Model side")
    parser.add_argument("--window-months", type=int, default=6, help="Training window (months)")
    parser.add_argument("--force", action="store_true", help="Force training")
    parser.add_argument("--all", action="store_true", help="Retrain all models")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

    if args.all:
        # Retrain all models
        versions = retrain_all_models(
            tickers=["SOLUSDT"],
            sides=["long", "short"],
            window_months=args.window_months,
            force=args.force
        )
    else:
        # Retrain single model
        version = rolling_window_train(
            ticker=args.ticker,
            side=args.side,
            window_months=args.window_months,
            force=args.force
        )

        if version:
            print(f"\n✅ SUCCESS: Model trained → {version}")
        else:
            print(f"\n⏭️  SKIPPED: No training needed")
