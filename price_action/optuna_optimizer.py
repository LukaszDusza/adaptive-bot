#!/usr/bin/env python3
"""
Optuna Optimizer for Backtester Parameters
Optimizes PROB_THRESHOLD, TP_PCT, and TSL_PCT for MAXIMUM TOTAL PNL

OPTIMIZATION FOCUS: Total PnL (absolute profit in USD)
Secondary considerations: Drawdown control, Sharpe ratio

Scoring formula:
    score = total_pnl - (max_dd * 50) + (sharpe * 200)
    
This prioritizes strategies that make the most money while keeping
risk under control and maintaining good risk-adjusted returns.
"""

import argparse
import logging
import os
import warnings
import optuna
from optuna.visualization import (
    plot_pareto_front,
    plot_param_importances,
    plot_optimization_history,
    plot_slice,
    plot_parallel_coordinate,
    plot_contour
)
from optuna.importance import get_param_importances
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List
from tqdm import tqdm
from backtester import BacktestEngine, calculate_metrics
from data_preparer_pa import fetch_and_prepare_data

# Suppress Optuna experimental warnings for multivariate parameter
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ProgressCallback:
    """Callback to update tqdm progress bar during optimization"""

    def __init__(self, n_trials: int):
        self.pbar = tqdm(total=n_trials, desc="Optimization", unit="trial",
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Called after each trial completes"""
        self.pbar.update(1)

    def close(self):
        """Close progress bar"""
        self.pbar.close()


class BestTrialCallback:
    """Callback to display best trial information during optimization"""

    def __init__(self, show_limit_params: bool = False):
        self.best_pnl = float('-inf')
        self.best_trial_number = None
        self.best_params = None
        self.best_values = None
        self.last_update_trial = -1
        self.show_limit_params = show_limit_params

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Called after each trial completes"""
        # Get best trials from Pareto front
        best_trials = study.best_trials

        if best_trials:
            # Find trial with highest PnL (values[0])
            current_best = max(best_trials, key=lambda t: t.values[0])

            # Update if we found a better PnL
            if current_best.values[0] > self.best_pnl:
                self.best_pnl = current_best.values[0]
                self.best_trial_number = current_best.number
                self.best_params = current_best.params
                self.best_values = current_best.values
                self.last_update_trial = trial.number

                # Get winrate from trial user attributes
                winrate = current_best.user_attrs.get('win_rate', 0)

                # Build limit params string if optimizing
                limit_params_str = ""
                if self.show_limit_params:
                    limit_offset = self.best_params.get('limit_offset_pct', 0.005)
                    limit_candles = self.best_params.get('limit_order_candles', 5)
                    limit_params_str = f" limit_off={limit_offset:.3f} candles={limit_candles}"

                # Display best result found so far (use tqdm.write for clean output)
                tqdm.write(f"\n✨ Best: Trial #{self.best_trial_number} | "
                          f"PnL=${self.best_pnl:+.2f} | "
                          f"prob={self.best_params['prob_threshold']:.2f} "
                          f"tp={self.best_params['tp_pct']:.3f} "
                          f"tsl={self.best_params['tsl_pct']:.3f} "
                          f"min_diff={self.best_params['min_proba_diff']:.2f}{limit_params_str} | "
                          f"WR={winrate:.1f}% DD={-self.best_values[1]:.2f}%")


class BacktesterOptimizer:
    """Optimizer for backtester parameters using Optuna"""

    def __init__(self,
                 ticker: str,
                 timeframe: str,
                 helper_timeframes: List[str],
                 limit: int,
                 initial_capital: float = 10000.0,
                 risk_pct: float = 0.02,
                 enable_partial_tp: bool = True,
                 enable_dynamic_tp: bool = False,
                 enable_profit_protection: bool = False,
                 enable_limit_order: bool = False,
                 optimize_limit_params: bool = False,
                 version: str = 'v1.0',
                 n_jobs: int = -1,
                 holdout_pct: float = 0.2):

        self.ticker = ticker
        self.timeframe = timeframe
        self.helper_timeframes = helper_timeframes
        self.limit = limit
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
        self.enable_partial_tp = enable_partial_tp
        self.enable_dynamic_tp = enable_dynamic_tp
        self.enable_profit_protection = enable_profit_protection
        self.enable_limit_order = enable_limit_order
        self.optimize_limit_params = optimize_limit_params
        self.version = version
        self.n_jobs = n_jobs
        self.holdout_pct = holdout_pct

        if self.enable_partial_tp and self.enable_dynamic_tp:
            raise ValueError("Cannot enable both enable_partial_tp and enable_dynamic_tp. Choose only one.")

        # Estimate expected trades based on timeframe (for trade count scoring)
        timeframe_multipliers = {
            '1m': 10.0, '5m': 5.0, '15m': 2.0, '30m': 1.5,
            '1h': 0.5, '2h': 0.35, '4h': 0.25, '1d': 0.1, '1D': 0.1
        }
        tf_key = self.timeframe.replace(' ', '').lower()
        self.trade_count_multiplier = timeframe_multipliers.get(tf_key, 1.0)

        self._load_models()
        self.df_train, self.df_holdout = self._prepare_data()
        
    def _get_strategy_id(self, side: str) -> str:
        """Generate strategy ID"""
        helpers = '_plus_' + '_'.join(self.helper_timeframes) if self.helper_timeframes else ""
        return f"{self.ticker}_{self.timeframe.replace(' ', '')}{helpers}_{side}"
    
    def _load_models(self):
        """Load trained models for long and short"""
        long_id = self._get_strategy_id('long')
        short_id = self._get_strategy_id('short')
        
        try:
            self.model_long = joblib.load(f"models/{self.version}/{long_id}/model.joblib")
            self.scaler_long = joblib.load(f"models/{self.version}/{long_id}/scaler.joblib")
            self.features_long = joblib.load(f"models/{self.version}/{long_id}/features.joblib")
            
            self.model_short = joblib.load(f"models/{self.version}/{short_id}/model.joblib")
            self.scaler_short = joblib.load(f"models/{self.version}/{short_id}/scaler.joblib")
            self.features_short = joblib.load(f"models/{self.version}/{short_id}/features.joblib")
            
            logging.info(f"✓ Models loaded from version {self.version}")
        except FileNotFoundError as e:
            logging.error(f"Model files not found for version {self.version}: {e}")
            raise
    
    def _prepare_data(self) -> tuple:
        """
        Prepare data for backtesting with train/holdout split

        Returns:
            tuple: (df_train, df_holdout) - Training data for optimization, holdout for final validation
        """
        logging.info(f"Preparing data: {self.ticker} {self.timeframe}")
        df = fetch_and_prepare_data(
            ticker=self.ticker,
            timeframe=self.timeframe,
            limit=self.limit,
            helper_timeframes=self.helper_timeframes,
            side='backtest',
            version=self.version
        )

        if df.empty:
            raise ValueError("Failed to prepare data")

        # Split into train (for CV optimization) and holdout (for final validation)
        split_idx = int(len(df) * (1 - self.holdout_pct))
        df_train = df.iloc[:split_idx]
        df_holdout = df.iloc[split_idx:]

        logging.info(f"✓ Data prepared: {len(df)} candles total")
        logging.info(f"  - Training set: {len(df_train)} candles ({(1-self.holdout_pct)*100:.0f}%)")
        logging.info(f"  - Holdout set:  {len(df_holdout)} candles ({self.holdout_pct*100:.0f}%)")

        return df_train, df_holdout
    
    def objective(self, trial: optuna.Trial) -> tuple:
        """
        Optuna multi-objective function - OPTIMIZED FOR THREE OBJECTIVES
        
        Returns three objectives (all to maximize):
        1. Total PnL (USD profit) - HIGHEST PRIORITY
        2. Negative Max Drawdown (lower drawdown is better) - SECOND PRIORITY
        3. Trade count score (balanced trading activity) - THIRD PRIORITY
        
        The trade count score prevents Optuna from selecting strategies with too few trades
        while also avoiding over-trading. Optimal range: 50-200 trades for 10k candles.
        
        Returns: tuple of (pnl, -drawdown, trade_score)
        """
        
        # Suggest parameters
        # OPTIMIZED RANGES: Zawężone zakresy bazując na crypto volatility analysis
        prob_threshold = trial.suggest_float('prob_threshold', 0.5, 0.9, step=0.05)
        tp_pct = trial.suggest_float('tp_pct', 0.02, 0.06, step=0.005)  # Narrowed: 2-6% (was 2-15%)
        tsl_pct = trial.suggest_float('tsl_pct', 0.01, 0.03, step=0.005)  # Narrowed to match TP range
        min_proba_diff = trial.suggest_float('min_proba_diff', 0.1, 0.7, step=0.05)  # Narrowed: 10-70% (was 0-70%)

        # Limit order parameters (only optimize if optimize_limit_params=True)
        if self.optimize_limit_params:
            # Offset: 0.1% (tight) to 1.5% (wide spread)
            limit_offset_pct = trial.suggest_float('limit_offset_pct', 0.001, 0.015, step=0.001)
            # Candles to wait: 3 (quick) to 10 (patient)
            limit_order_candles = trial.suggest_int('limit_order_candles', 3, 10)
        else:
            # Use fixed defaults when not optimizing
            limit_offset_pct = 0.005  # 0.5% default
            limit_order_candles = 5    # 5 candles default

        # Ensure TSL is not greater than TP (logical constraint)
        if tsl_pct >= tp_pct:
            trial.set_user_attr('invalid_reason', 'tsl >= tp')
            raise optuna.TrialPruned("Invalid: TSL must be < TP")

        # Calculate SL as 50% of TP
        sl_pct = tp_pct * 0.5

        # ============================================================================
        # EXPANDING WINDOW 3-FOLD CROSS-VALIDATION (FIXED v2.1)
        # ============================================================================
        # Better for time-series: each fold expands from start
        # Fold 1: 0-33%   Fold 2: 0-66%   Fold 3: 0-100%
        # This respects temporal order and prevents look-ahead bias
        # ============================================================================

        # Calculate fold boundaries for expanding window
        n_samples = len(self.df_train)
        fold_size = n_samples // 3

        fold_metrics_list = []
        fold_dd_list = []  # Track DD separately for worst-case aggregation

        for fold_idx in range(3):
            # EXPANDING WINDOW: Always start from beginning
            fold_end = (fold_idx + 1) * fold_size if fold_idx < 2 else n_samples
            df_fold = self.df_train.iloc[:fold_end]  # From start to fold_end

            # Run backtest on this fold
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                enable_limit_order=self.enable_limit_order,
                limit_order_candles=limit_order_candles,
                limit_offset_pct=limit_offset_pct,
                timeframe=self.timeframe,
                suppress_limit_logs=True  # Suppress limit order logs during optimization
            )

            try:
                results = engine.run(
                    df=df_fold,
                    model_long=self.model_long,
                    scaler_long=self.scaler_long,
                    features_long=self.features_long,
                    model_short=self.model_short,
                    scaler_short=self.scaler_short,
                    features_short=self.features_short,
                    prob_threshold=prob_threshold,
                    risk_pct=self.risk_pct,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    tsl_pct=tsl_pct,
                    enable_partial_tp=self.enable_partial_tp,
                    enable_dynamic_tp=self.enable_dynamic_tp,
                    enable_profit_protection=self.enable_profit_protection,
                    min_proba_diff=min_proba_diff
                )

                # Calculate metrics for this fold
                fold_metrics = calculate_metrics(
                    results['trades'],
                    results['equity_curve'],
                    self.initial_capital
                )
                fold_metrics_list.append(fold_metrics)
                fold_dd_list.append(fold_metrics.get('max_drawdown_pct', 100))

                # EARLY STOPPING: After first fold, prune if performance is terrible
                # NOTE: trial.report() and trial.should_prune() NOT supported for multi-objective studies
                # Use simple heuristic instead: prune if PnL < -50% or drawdown > 80%
                if fold_idx == 0:
                    fold_pnl_pct = fold_metrics.get('total_return_pct', -999)
                    fold_dd_pct = fold_metrics.get('max_drawdown_pct', 100)

                    if fold_pnl_pct < -50 or fold_dd_pct > 80:
                        logging.warning(f"Pruning trial: Poor fold 0 performance (PnL={fold_pnl_pct:.1f}%, DD={fold_dd_pct:.1f}%)")
                        raise optuna.TrialPruned(f"Pruned after fold {fold_idx} (PnL={fold_pnl_pct:.1f}%, DD={fold_dd_pct:.1f}%)")

            except optuna.TrialPruned:
                # Re-raise TrialPruned to signal early stopping
                raise
            except Exception as e:
                # If fold fails, prune this trial (don't waste time on broken params)
                logging.error(f"Fold {fold_idx} failed: {str(e)}")
                trial.set_user_attr('error', str(e))
                raise optuna.TrialPruned(f"Trial failed on fold {fold_idx}: {str(e)}")

        # Aggregate metrics across folds
        # PnL, Return, Sharpe: Average across folds
        # Max DD: WORST CASE (maximum DD across all folds)
        # Trades: Sum across folds
        metrics = {
            'total_pnl_usd': np.mean([m.get('total_pnl_usd', -999999) for m in fold_metrics_list]),
            'total_return_pct': np.mean([m.get('total_return_pct', -999) for m in fold_metrics_list]),
            'max_drawdown_pct': np.max(fold_dd_list),  # WORST CASE (not average!)
            'sharpe_ratio': np.mean([m.get('sharpe_ratio', 0) for m in fold_metrics_list]),
            'total_trades': int(np.sum([m.get('total_trades', 0) for m in fold_metrics_list])),
            'win_rate': np.mean([m.get('win_rate', 0) for m in fold_metrics_list]),
            'profit_factor': np.mean([m.get('profit_factor', 0) for m in fold_metrics_list])
        }

        # Extract aggregated metrics
        total_pnl = metrics.get('total_pnl_usd', -999999)
        total_return = metrics.get('total_return_pct', -999)
        max_dd = abs(metrics.get('max_drawdown_pct', 100))
        sharpe = metrics.get('sharpe_ratio', 0)
        num_trades = metrics['total_trades']

        # Multi-objective optimization:
        # Objective 1: Maximize Total PnL (PRIMARY - highest priority)
        obj1_pnl = total_pnl

        # Objective 2: Minimize Drawdown (SECONDARY - return negative for maximization)
        obj2_drawdown = -max_dd

        # Objective 3: Optimize Trade Count (TERTIARY - balanced trading activity)
        # SCALED BY TIMEFRAME: 15m expects more trades than 4h
        min_trades = int(10 * self.trade_count_multiplier)
        optimal_start = int(30 * self.trade_count_multiplier)
        optimal_end = int(200 * self.trade_count_multiplier)

        if num_trades < min_trades:
            trial.set_user_attr('invalid_reason', f'too few trades: {num_trades}')
            raise optuna.TrialPruned(f"Too few trades: {num_trades} < {min_trades}")
        elif num_trades < optimal_start:
            obj3_trade_score = num_trades * 2  # Encourage more trades
        elif num_trades < optimal_end:
            obj3_trade_score = 100 + (num_trades - optimal_start) * 0.5  # Optimal range
        else:
            obj3_trade_score = 185 - (num_trades - optimal_end) * 0.1  # Slight penalty for over-trading

        # Log trial results (CV-averaged)
        limit_params_str = ""
        if self.optimize_limit_params:
            limit_params_str = f", limit_offset={limit_offset_pct:.3f}, limit_candles={limit_order_candles}"

        logging.info(
            f"Trial {trial.number} (3-fold CV avg): "
            f"prob={prob_threshold:.2f}, tp={tp_pct:.3f}, tsl={tsl_pct:.3f}, min_diff={min_proba_diff:.2f}{limit_params_str} | "
            f"Obj1(PnL)=${obj1_pnl:+.2f}, Obj2(DD)={obj2_drawdown:.2f}%, Obj3(Trades)={obj3_trade_score:.1f} | "
            f"Return={total_return:.2f}%, DD={max_dd:.2f}%, Sharpe={sharpe:.3f}, Trades={num_trades}"
        )

        # Store additional metrics in trial user attributes
        trial.set_user_attr('total_pnl_usd', total_pnl)
        trial.set_user_attr('total_return_pct', total_return)
        trial.set_user_attr('max_drawdown_pct', max_dd)
        trial.set_user_attr('sharpe_ratio', sharpe)
        trial.set_user_attr('total_trades', metrics['total_trades'])
        trial.set_user_attr('win_rate', metrics.get('win_rate', 0))
        trial.set_user_attr('profit_factor', metrics.get('profit_factor', 0))
        trial.set_user_attr('obj1_pnl', obj1_pnl)
        trial.set_user_attr('obj2_drawdown', obj2_drawdown)
        trial.set_user_attr('obj3_trade_score', obj3_trade_score)

        # Store limit order params if optimizing
        if self.optimize_limit_params:
            trial.set_user_attr('limit_offset_pct', limit_offset_pct)
            trial.set_user_attr('limit_order_candles', limit_order_candles)

        return obj1_pnl, obj2_drawdown, obj3_trade_score
    
    def optimize(self, n_trials: int = 100) -> Dict:
        """
        Run optimization
        
        Args:
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with best parameters and results
        """
        logging.info(f"\n{'='*70}")
        logging.info(f"{'STARTING OPTUNA OPTIMIZATION':^70}")
        logging.info(f"{'='*70}")
        logging.info(f"Trials: {n_trials}")
        logging.info(f"Ticker: {self.ticker} {self.timeframe}")
        logging.info(f"Training data: {len(self.df_train)} candles")
        logging.info(f"Holdout data:  {len(self.df_holdout)} candles")
        logging.info(f"{'='*70}\n")

        # Create database directory for persistent storage
        os.makedirs(f"models/{self.version}/optuna", exist_ok=True)

        # Generate unique study name based on ticker, timeframe, and helper timeframes
        # This allows resuming optimization for same configuration in the future
        helpers_str = '_'.join(self.helper_timeframes) if self.helper_timeframes else 'none'
        study_name = f"{self.ticker}_{self.timeframe}_{helpers_str}_multi_opt"

        # SQLite database storage for persistent optimization history
        storage_url = f"sqlite:///models/{self.version}/optuna/optuna_{self.ticker}_{self.timeframe}_{helpers_str}.db"

        logging.info(f"Database: {storage_url}")
        logging.info(f"Study name: {study_name}")
        logging.info(f"Load if exists: True (can resume previous optimizations)\n")

        # ============================================================================
        # SMART SAMPLING & PRUNING (v2.1 FIX)
        # ============================================================================
        # TPESampler: Bayesian optimization with multivariate correlations
        # MedianPruner: Early stopping for weak trials after first fold
        # ============================================================================
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=20,  # Random exploration first
            multivariate=True,    # Consider parameter correlations
            warn_independent_sampling=False,  # Suppress warnings for dynamic search space
            seed=42               # Reproducible results
        )

        # NOTE: MedianPruner requires trial.report() which is NOT supported for multi-objective studies
        # Use NopPruner (no-op) instead - we still have heuristic pruning in objective function
        pruner = optuna.pruners.NopPruner()

        # Create multi-objective study (3 objectives, all to maximize)
        # Objective 1: PnL (highest priority)
        # Objective 2: -Drawdown (second priority)
        # Objective 3: Trade count score (third priority)
        # load_if_exists=True allows resuming previous optimizations
        study = optuna.create_study(
            directions=['maximize', 'maximize', 'maximize'],
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner
        )
        
        # Check if resuming previous optimization
        existing_trials = len(study.trials)
        if existing_trials > 0:
            logging.info(f"{'='*70}")
            logging.info(f"RESUMING PREVIOUS OPTIMIZATION")
            logging.info(f"Found {existing_trials} existing trials in database")
            logging.info(f"Will run {n_trials} additional trials")
            logging.info(f"Total trials after completion: {existing_trials + n_trials}")
            logging.info(f"{'='*70}\n")
        
        # Suppress INFO logs during optimization (only show progress bar)
        # This hides position logs from backtester which clutter the console
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logging.getLogger().setLevel(logging.WARNING)

        # Create callbacks for progress tracking and best trial display
        progress_callback = ProgressCallback(n_trials=n_trials)
        best_trial_callback = BestTrialCallback(show_limit_params=self.optimize_limit_params)

        # ============================================================================
        # PARALLEL EXECUTION: n_jobs=-1 (speedup 4-8x)
        # ============================================================================
        # Uses all CPU cores for parallel trial execution
        # Single progress bar shows overall progress
        # BestTrialCallback displays improvements in real-time
        # ============================================================================
        import multiprocessing
        n_jobs = -1  # Use all available CPU cores
        actual_cores = multiprocessing.cpu_count()

        logging.info(f"\n🚀 PARALLEL OPTIMIZATION ENABLED")
        logging.info(f"   CPU Cores: {actual_cores}")
        logging.info(f"   Expected speedup: {min(actual_cores, 8)}x")
        logging.info(f"   Progress bar will track completed trials\n")

        try:
            study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs,
                          callbacks=[progress_callback, best_trial_callback])
        finally:
            # Ensure progress bar is closed even if optimization fails
            progress_callback.close()

        # Print final newline after optimization to separate from next output
        print("\n")
        
        # Restore INFO logging after optimization completes
        optuna.logging.set_verbosity(optuna.logging.INFO)
        logging.getLogger().setLevel(logging.INFO)
        
        # Get best results from Pareto front
        # For multi-objective, we get all Pareto-optimal trials
        best_trials = study.best_trials
        
        logging.info(f"\n{'='*70}")
        logging.info(f"Found {len(best_trials)} Pareto-optimal solutions")
        logging.info(f"{'='*70}\n")
        
        # Select the trial with highest PnL (primary objective) from Pareto front
        # This prioritizes PnL while still considering the multi-objective trade-offs
        best_trial = max(best_trials, key=lambda t: t.values[0])  # values[0] is PnL
        best_params = best_trial.params
        
        logging.info(f"Selected best trial based on highest PnL from Pareto front:")
        logging.info(f"  PnL: ${best_trial.values[0]:,.2f}")
        logging.info(f"  Drawdown: {-best_trial.values[1]:.2f}%")
        logging.info(f"  Trade Score: {best_trial.values[2]:.1f}")
        logging.info(f"  Parameters: {best_params}\n")
        
        # ============================================================================
        # HOLDOUT VALIDATION (v2.1 FIX)
        # ============================================================================
        # Run final backtest on HOLDOUT set (unseen data) to validate generalization
        # ============================================================================
        logging.info(f"\n{'='*70}")
        logging.info(f"{'OPTIMIZATION COMPLETE - HOLDOUT VALIDATION':^70}")
        logging.info(f"{'='*70}")
        logging.info(f"Running final backtest on holdout set ({len(self.df_holdout)} candles)...\n")

        # Extract limit order params from best trial (if optimized)
        limit_offset_pct = best_params.get('limit_offset_pct', 0.005)
        limit_order_candles = best_params.get('limit_order_candles', 5)

        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            enable_limit_order=self.enable_limit_order,
            limit_order_candles=limit_order_candles,
            limit_offset_pct=limit_offset_pct,
            timeframe=self.timeframe,
            suppress_limit_logs=False  # Show logs in final backtest (after optimization)
        )

        # Run on HOLDOUT set for unbiased evaluation
        final_results = engine.run(
            df=self.df_holdout,  # HOLDOUT (not training) for unbiased results
            model_long=self.model_long,
            scaler_long=self.scaler_long,
            features_long=self.features_long,
            model_short=self.model_short,
            scaler_short=self.scaler_short,
            features_short=self.features_short,
            prob_threshold=best_params['prob_threshold'],
            risk_pct=self.risk_pct,
            tp_pct=best_params['tp_pct'],
            sl_pct=best_params['tp_pct'] * 0.5,
            tsl_pct=best_params['tsl_pct'],
            enable_partial_tp=self.enable_partial_tp,
            enable_dynamic_tp=self.enable_dynamic_tp,
            enable_profit_protection=self.enable_profit_protection,
            min_proba_diff=best_params['min_proba_diff']
        )

        final_metrics = calculate_metrics(
            final_results['trades'],
            final_results['equity_curve'],
            self.initial_capital
        )
        
        # Print results
        self._print_optimization_results(best_params, final_metrics, study, best_trial, best_trials)

        # Generate visualizations and parameter importance analysis
        self._generate_visualizations(study, best_trial)

        # Save recommended live parameters to model directories
        self._save_recommended_live_params(best_params, final_metrics, best_trial)

        return {
            'best_params': best_params,
            'metrics': final_metrics,
            'study': study,
            'best_trial': best_trial,
            'pareto_trials': best_trials
        }

    def _generate_visualizations(self, study: optuna.Study, best_trial):
        """
        Generate Optuna visualization plots and parameter importance analysis

        Creates:
        - Pareto front plot (multi-objective trade-offs)
        - Parameter importances (which params matter most)
        - Optimization history (progress over trials)
        - Parameter relationships (slice plots)
        - Parallel coordinate plot (Pareto solutions)
        """
        # Create plots directory
        plots_dir = f"models/{self.version}/optuna/plots"
        os.makedirs(plots_dir, exist_ok=True)

        logging.info(f"\n{'='*70}")
        logging.info(f"GENERATING OPTUNA VISUALIZATIONS")
        logging.info(f"{'='*70}")

        try:
            # 1. Pareto Front (multi-objective trade-offs)
            logging.info("📊 Generating Pareto front plot...")
            fig = plot_pareto_front(study, target_names=['PnL (USD)', 'Drawdown (%)', 'Trade Score'])
            fig.write_html(f"{plots_dir}/pareto_front.html")
            logging.info(f"   ✓ Saved: {plots_dir}/pareto_front.html")

            # 2. Parameter Importances (which params matter most)
            logging.info("📊 Generating parameter importance plot...")
            try:
                # Calculate importances
                importances = get_param_importances(study, target=lambda t: t.values[0])  # PnL importance
                logging.info(f"\n   Parameter Importances (for PnL objective):")
                for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
                    logging.info(f"      {param:20s}: {importance:.4f}")

                fig = plot_param_importances(study, target=lambda t: t.values[0], target_name='PnL (USD)')
                fig.write_html(f"{plots_dir}/param_importances.html")
                logging.info(f"   ✓ Saved: {plots_dir}/param_importances.html")
            except Exception as e:
                logging.warning(f"   ⚠ Could not generate param importances: {e}")

            # 3. Optimization History (progress over trials)
            logging.info("📊 Generating optimization history plot...")
            fig = plot_optimization_history(study, target_name='PnL (USD)', target=lambda t: t.values[0])
            fig.write_html(f"{plots_dir}/optimization_history.html")
            logging.info(f"   ✓ Saved: {plots_dir}/optimization_history.html")

            # 4. Slice Plot (parameter relationships)
            logging.info("📊 Generating slice plot (parameter relationships)...")
            fig = plot_slice(study, target=lambda t: t.values[0], target_name='PnL (USD)')
            fig.write_html(f"{plots_dir}/slice_plot.html")
            logging.info(f"   ✓ Saved: {plots_dir}/slice_plot.html")

            # 5. Parallel Coordinate Plot (Pareto solutions)
            logging.info("📊 Generating parallel coordinate plot...")
            fig = plot_parallel_coordinate(study, target_names=['PnL (USD)', 'Drawdown (%)', 'Trade Score'])
            fig.write_html(f"{plots_dir}/parallel_coordinate.html")
            logging.info(f"   ✓ Saved: {plots_dir}/parallel_coordinate.html")

            # 6. Contour Plot (2D parameter interactions)
            logging.info("📊 Generating contour plots (parameter interactions)...")
            try:
                # tp_pct vs prob_threshold
                fig = plot_contour(study, params=['prob_threshold', 'tp_pct'],
                                   target=lambda t: t.values[0], target_name='PnL (USD)')
                fig.write_html(f"{plots_dir}/contour_prob_tp.html")
                logging.info(f"   ✓ Saved: {plots_dir}/contour_prob_tp.html")

                # tsl_pct vs tp_pct
                fig = plot_contour(study, params=['tp_pct', 'tsl_pct'],
                                   target=lambda t: t.values[0], target_name='PnL (USD)')
                fig.write_html(f"{plots_dir}/contour_tp_tsl.html")
                logging.info(f"   ✓ Saved: {plots_dir}/contour_tp_tsl.html")
            except Exception as e:
                logging.warning(f"   ⚠ Could not generate contour plots: {e}")

            logging.info(f"\n{'='*70}")
            logging.info(f"✓ All visualizations saved to: {plots_dir}/")
            logging.info(f"{'='*70}\n")

        except Exception as e:
            logging.error(f"❌ Error generating visualizations: {e}")
            logging.error(f"   Plots may be incomplete. Check {plots_dir}/")

    def _save_recommended_live_params(self, best_params: Dict, metrics: Dict, best_trial):
        """
        Save recommended live parameters to model directories

        This creates deployment-ready config files in each model directory:
        - recommended_live_params.json (from Optuna optimization)
        - deployment_config.json (combines training + optuna results)

        These files can be used for automatic deployment configuration.
        """
        import json
        from datetime import datetime

        logging.info(f"\n{'='*70}")
        logging.info(f"SAVING RECOMMENDED LIVE PARAMETERS")
        logging.info(f"{'='*70}")

        # Calculate SL based on TP
        sl_pct = best_params['tp_pct'] * 0.5

        # Determine TP mechanism
        if self.enable_dynamic_tp:
            tp_mechanism = "dynamic_tp"
        elif self.enable_partial_tp:
            tp_mechanism = "partial_tp"
        else:
            tp_mechanism = "none"

        # Recommended live parameters from Optuna
        recommended_params = {
            "optimization_date": datetime.now().isoformat(),
            "optimization_version": "v2.0_with_3fold_cv",
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "helper_timeframes": self.helper_timeframes,
            "version": self.version,
            "live_parameters": {
                "prob_threshold": best_params['prob_threshold'],
                "min_proba_diff": best_params['min_proba_diff'],
                "tp_pct": best_params['tp_pct'],
                "tsl_pct": best_params['tsl_pct'],
                "sl_pct": sl_pct,
                "tp_mechanism": tp_mechanism,
                "enable_partial_tp": self.enable_partial_tp,
                "enable_dynamic_tp": self.enable_dynamic_tp,
                "enable_profit_protection": self.enable_profit_protection
            },
            "performance_metrics": {
                "total_pnl_usd": metrics.get('total_pnl_usd', 0),
                "total_return_pct": metrics.get('total_return_pct', 0),
                "max_drawdown_pct": metrics.get('max_drawdown_pct', 0),
                "sharpe_ratio": metrics.get('sharpe_ratio', 0),
                "win_rate": metrics.get('win_rate', 0),
                "total_trades": metrics.get('total_trades', 0),
                "profit_factor": metrics.get('profit_factor', 0)
            },
            "optimization_info": {
                "trial_number": best_trial.number,
                "obj1_pnl": best_trial.values[0],
                "obj2_drawdown": -best_trial.values[1],
                "obj3_trade_score": best_trial.values[2],
                "cv_folds": 3,
                "backtest_candles_train": len(self.df_train),
                "backtest_candles_holdout": len(self.df_holdout),
                "backtest_candles_total": len(self.df_train) + len(self.df_holdout)
            }
        }

        # Save to both LONG and SHORT model directories
        for side in ['long', 'short']:
            strategy_id = self._get_strategy_id(side)
            model_dir = f"models/{self.version}/{strategy_id}"

            # Check if model directory exists
            if not os.path.exists(model_dir):
                logging.warning(f"   ⚠ Model directory not found: {model_dir} (skipping {side.upper()})")
                continue

            # Save recommended_live_params.json
            params_file = os.path.join(model_dir, "recommended_live_params.json")
            with open(params_file, 'w') as f:
                json.dump(recommended_params, f, indent=2)
            logging.info(f"   ✓ Saved: {params_file}")

            # Load training_metadata.json to get optimal_threshold
            training_metadata_file = os.path.join(model_dir, "training_metadata.json")
            optimal_threshold = None
            if os.path.exists(training_metadata_file):
                with open(training_metadata_file, 'r') as f:
                    training_metadata = json.load(f)
                    optimal_threshold = training_metadata.get('optimal_threshold')

            # Create deployment_config.json (combines everything)
            deployment_config = {
                "deployment_ready": True,
                "created_date": datetime.now().isoformat(),
                "version": self.version,
                "side": side,
                "ticker": self.ticker,
                "timeframe": self.timeframe,
                "helper_timeframes": self.helper_timeframes,
                "deployment_parameters": {
                    "prob_threshold": best_params['prob_threshold'],
                    "min_proba_diff": best_params['min_proba_diff'],
                    "tp_pct": best_params['tp_pct'],
                    "tsl_pct": best_params['tsl_pct'],
                    "sl_pct": sl_pct,
                    "tp_mechanism": tp_mechanism,
                    "enable_partial_tp": self.enable_partial_tp,
                    "enable_dynamic_tp": self.enable_dynamic_tp,
                    "enable_profit_protection": self.enable_profit_protection,
                    "enable_limit_order": self.enable_limit_order,
                    "limit_offset_pct": best_params.get('limit_offset_pct', 0.005),
                    "limit_order_candles": best_params.get('limit_order_candles', 5)
                },
                "model_info": {
                    "optimal_threshold_from_training": optimal_threshold,
                    "recommended_prob_threshold_from_optuna": best_params['prob_threshold'],
                    "note": "Use prob_threshold from Optuna (already optimized on live-like conditions)"
                },
                "expected_performance": {
                    "total_pnl_usd": metrics.get('total_pnl_usd', 0),
                    "total_return_pct": metrics.get('total_return_pct', 0),
                    "max_drawdown_pct": metrics.get('max_drawdown_pct', 0),
                    "sharpe_ratio": metrics.get('sharpe_ratio', 0),
                    "win_rate": metrics.get('win_rate', 0),
                    "total_trades": metrics.get('total_trades', 0),
                    "note": "Based on 3-fold CV backtest, actual results may vary"
                },
                "docker_compose_command_template": f"""
  --prob-threshold {best_params['prob_threshold']:.2f} \\
  --min-proba-diff {best_params['min_proba_diff']:.2f} \\
  --tp-pct {best_params['tp_pct']:.3f} \\
  --tsl-pct {best_params['tsl_pct']:.3f} \\
  --{'partial-tp' if self.enable_partial_tp else 'dynamic-tp' if self.enable_dynamic_tp else 'no-tp'}{' \\' if self.enable_limit_order else ''}
{f'  --limit-order \\' if self.enable_limit_order else ''}{f'''
  --limit-offset-pct {best_params.get('limit_offset_pct', 0.005):.3f} \\
  --limit-order-candles {best_params.get('limit_order_candles', 5)}''' if self.enable_limit_order else ''}
                """.strip()
            }

            # Save deployment_config.json
            deployment_file = os.path.join(model_dir, "deployment_config.json")
            with open(deployment_file, 'w') as f:
                json.dump(deployment_config, f, indent=2)
            logging.info(f"   ✓ Saved: {deployment_file}")

        logging.info(f"\n{'='*70}")
        logging.info(f"✓ Recommended parameters saved to model directories")
        logging.info(f"{'='*70}")
        logging.info(f"\nUse these files for deployment:")
        logging.info(f"  - recommended_live_params.json (Optuna optimization results)")
        logging.info(f"  - deployment_config.json (Complete deployment config)")
        logging.info(f"\nFor auto-deployment, run:")
        logging.info(f"  ./run_model_workflow.sh → Option 11) Optimize & Auto-Deploy")
        logging.info(f"{'='*70}\n")

    def _print_optimization_results(self, best_params: Dict, metrics: Dict, study: optuna.Study,
                                     best_trial, pareto_trials):
        """Print multi-objective optimization results in a nice format"""
        print("\n" + "="*70)
        print(f"{'MULTI-OBJECTIVE OPTIMIZATION RESULTS':^70}")
        print("="*70)
        
        print(f"\n{'OPTIMIZATION OBJECTIVES (Priority Order):':^70}")
        print(f"  1️⃣  Maximize PnL (USD profit) - PRIMARY")
        print(f"  2️⃣  Minimize Drawdown (risk control) - SECONDARY")
        print(f"  3️⃣  Optimize Trade Count (active trading) - TERTIARY")
        
        print(f"\n{'BEST PARAMETERS (Highest PnL from Pareto Front):':^70}")
        print(f"  PROB_THRESHOLD = {best_params['prob_threshold']:.2f}")
        print(f"  MIN_PROBA_DIFF = {best_params['min_proba_diff']:.2f}")
        print(f"  TP_PCT         = {best_params['tp_pct']:.3f}")
        print(f"  TSL_PCT        = {best_params['tsl_pct']:.3f}")
        print(f"  SL_PCT         = {best_params['tp_pct'] * 0.5:.3f} (auto: 50% of TP)")

        # Display limit order params if enabled
        if self.enable_limit_order:
            print(f"\n{'LIMIT ORDER PARAMETERS:':^70}")
            if self.optimize_limit_params:
                print(f"  LIMIT_OFFSET_PCT    = {best_params.get('limit_offset_pct', 0.005):.3f} (OPTIMIZED)")
                print(f"  LIMIT_ORDER_CANDLES = {best_params.get('limit_order_candles', 5)} (OPTIMIZED)")
            else:
                print(f"  LIMIT_OFFSET_PCT    = {best_params.get('limit_offset_pct', 0.005):.3f} (fixed)")
                print(f"  LIMIT_ORDER_CANDLES = {best_params.get('limit_order_candles', 5)} (fixed)")
        
        print(f"\n{'OBJECTIVE VALUES FOR SELECTED SOLUTION:':^70}")
        print(f"  🎯 Obj 1 - Total PnL:      ${best_trial.values[0]:+,.2f}  ⭐ PRIMARY")
        print(f"  🛡️  Obj 2 - Max Drawdown:   {-best_trial.values[1]:.2f}%  (lower is better)")
        print(f"  📊 Obj 3 - Trade Score:    {best_trial.values[2]:.1f}  (balanced activity)")
        
        print(f"\n{'PERFORMANCE WITH BEST PARAMETERS:':^70}")
        print(f"  Total PnL:        ${metrics.get('total_pnl_usd', 0):+.2f}")
        print(f"  Total Return:     {metrics.get('total_return_pct', 0):.2f}%")
        print(f"  Max Drawdown:     {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Sortino Ratio:    {metrics.get('sortino_ratio', 0):.3f}")
        
        print(f"\n{'TRADING STATISTICS:':^70}")
        print(f"  Total Trades:     {metrics.get('total_trades', 0)}")
        print(f"  Win Rate:         {metrics.get('win_rate', 0):.2f}%")
        print(f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
        print(f"  Expectancy:       ${metrics.get('expectancy', 0):.2f}")
        print(f"  Avg Win:          ${metrics.get('avg_win', 0):.2f}")
        print(f"  Avg Loss:         ${metrics.get('avg_loss', 0):.2f}")
        
        print(f"\n{'OPTIMIZATION INFO:':^70}")
        print(f"  Total Trials:         {len(study.trials)}")
        print(f"  Pareto-Optimal Solns: {len(pareto_trials)}")
        print(f"  Selected Trial:       #{best_trial.number}")
        
        print("="*70 + "\n")
        
        # Save results to file
        # Determine TP mechanism string for filename
        if self.enable_dynamic_tp:
            tp_mechanism = "dynamic-tp"
        elif self.enable_partial_tp:
            tp_mechanism = "partial-tp"
        else:
            tp_mechanism = "no-tp"
        
        results_file = f"models/{self.version}/optuna/optimization_results_{self.ticker}_{self.timeframe}_{tp_mechanism}_limit{self.limit}.txt"
        os.makedirs(f"models/{self.version}/optuna", exist_ok=True)
        
        with open(results_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"MULTI-OBJECTIVE OPTIMIZATION RESULTS - {self.ticker} {self.timeframe}\n")
            f.write("="*70 + "\n\n")
            
            f.write("OPTIMIZATION OBJECTIVES (Priority Order):\n")
            f.write("  1. Maximize PnL (USD profit) - PRIMARY\n")
            f.write("  2. Minimize Drawdown (risk control) - SECONDARY\n")
            f.write("  3. Optimize Trade Count (active trading) - TERTIARY\n\n")
            
            f.write("BEST PARAMETERS (Highest PnL from Pareto Front):\n")
            f.write(f"PROB_THRESHOLD={best_params['prob_threshold']:.2f}\n")
            f.write(f"MIN_PROBA_DIFF={best_params['min_proba_diff']:.2f}\n")
            f.write(f"TP_PCT={best_params['tp_pct']:.3f}\n")
            f.write(f"TSL_PCT={best_params['tsl_pct']:.3f}\n")
            f.write(f"SL_PCT={best_params['tp_pct'] * 0.5:.3f}\n")

            # Write limit order params if enabled
            if self.enable_limit_order:
                f.write(f"\nLIMIT ORDER PARAMETERS:\n")
                f.write(f"LIMIT_OFFSET_PCT={best_params.get('limit_offset_pct', 0.005):.3f}\n")
                f.write(f"LIMIT_ORDER_CANDLES={best_params.get('limit_order_candles', 5)}\n")
                if self.optimize_limit_params:
                    f.write(f"NOTE: Limit order params were OPTIMIZED\n")
                else:
                    f.write(f"NOTE: Limit order params were FIXED (not optimized)\n")

            f.write("\n")
            
            f.write("OBJECTIVE VALUES:\n")
            f.write(f"Objective 1 (PnL): ${best_trial.values[0]:+,.2f}\n")
            f.write(f"Objective 2 (Drawdown): {-best_trial.values[1]:.2f}%\n")
            f.write(f"Objective 3 (Trade Score): {best_trial.values[2]:.1f}\n\n")
            
            f.write("PERFORMANCE:\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nOPTIMIZATION INFO:\n")
            f.write(f"Total Trials: {len(study.trials)}\n")
            f.write(f"Pareto-Optimal Solutions: {len(pareto_trials)}\n")
            f.write(f"Selected Trial: #{best_trial.number}\n\n")

            f.write(f"VISUALIZATIONS:\n")
            f.write(f"Interactive plots saved to: models/{self.version}/optuna/plots/\n")
            f.write(f"  - pareto_front.html (multi-objective trade-offs)\n")
            f.write(f"  - param_importances.html (which params matter most)\n")
            f.write(f"  - optimization_history.html (progress over trials)\n")
            f.write(f"  - slice_plot.html (parameter relationships)\n")
            f.write(f"  - parallel_coordinate.html (Pareto solutions)\n")

        logging.info(f"✓ Results saved to: {results_file}")
        logging.info(f"✓ Visualizations saved to: models/{self.version}/optuna/plots/")


def main():
    parser = argparse.ArgumentParser(description="Optimize backtester parameters using Optuna")
    
    parser.add_argument('--ticker', type=str, required=True, help="Ticker symbol (e.g., SOLUSDT)")
    parser.add_argument('--timeframe', type=str, required=True, help="Main timeframe (e.g., 1h)")
    parser.add_argument('--helper-timeframes', nargs='*', default=None, help="Helper timeframes")
    parser.add_argument('--version', type=str, default='v1.0',
                        help='Model version to use (e.g., v1.0, v1.1)')
    parser.add_argument('--limit', type=int, default=2000, help="Number of candles for backtest")
    parser.add_argument('--trials', type=int, default=100, help="Number of Optuna trials")
    parser.add_argument('--initial-capital', type=float, default=10000.0, help="Initial capital")
    parser.add_argument('--risk-pct', type=float, default=0.02, help="Risk per trade (% of capital)")
    parser.add_argument('--partial-tp', action='store_true',
                        help='Enable old partial TP mechanism (50%% at halfway to TP)')
    parser.add_argument('--dynamic-tp', action='store_true',
                        help='Enable new dynamic TP mechanism (25%% at each of 4 levels: 25%%, 50%%, 75%%, 100%%)')
    parser.add_argument('--protect-profit', action='store_true',
                        help='Enable profit protection: move SL to breakeven if profit peaks >0.25%% but declines before hitting partial TP')
    parser.add_argument('--limit-order', action='store_true',
                        help='Enable limit order mode (place limit orders instead of market orders)')
    parser.add_argument('--optimize-limit-params', action='store_true',
                        help='Optimize limit order parameters (limit_offset_pct, limit_order_candles). Requires --limit-order.')

    args = parser.parse_args()

    if args.partial_tp and args.dynamic_tp:
        parser.error("Cannot use both --partial-tp and --dynamic-tp. Choose only one.")

    if args.optimize_limit_params and not args.limit_order:
        parser.error("--optimize-limit-params requires --limit-order to be enabled.")

    optimizer = BacktesterOptimizer(
        ticker=args.ticker,
        timeframe=args.timeframe,
        helper_timeframes=args.helper_timeframes,
        limit=args.limit,
        initial_capital=args.initial_capital,
        risk_pct=args.risk_pct,
        enable_partial_tp=args.partial_tp,
        enable_dynamic_tp=args.dynamic_tp,
        enable_profit_protection=args.protect_profit,
        enable_limit_order=args.limit_order,
        optimize_limit_params=args.optimize_limit_params,
        version=args.version
    )
    
    results = optimizer.optimize(n_trials=args.trials)
    
    # Determine TP mechanism string for filename
    if args.dynamic_tp:
        tp_mechanism = "dynamic-tp"
    elif args.partial_tp:
        tp_mechanism = "partial-tp"
    else:
        tp_mechanism = "no-tp"
    
    print("\n✓ Optimization complete!")
    print(f"✓ Best parameters saved to: models/{args.version}/optuna/optimization_results_{args.ticker}_{args.timeframe}_{tp_mechanism}_limit{args.limit}.txt")
    print("\nYou can now update your run_model_workflow.sh with these optimal parameters.")


if __name__ == "__main__":
    main()
