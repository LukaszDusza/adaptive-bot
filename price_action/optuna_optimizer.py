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
import optuna
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List
from backtester import BacktestEngine, calculate_metrics
from data_preparer_pa import fetch_and_prepare_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BestTrialCallback:
    """Callback to display best trial information during optimization"""
    
    def __init__(self):
        self.best_pnl = float('-inf')
        self.best_trial_number = None
        self.best_params = None
        self.best_values = None
        self.last_update_trial = -1
    
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
                
                # Display best result found so far (on new line, clean output)
                # This avoids interfering with tqdm progress bar
                print(f"\n✨ Best: Trial #{self.best_trial_number} | "
                      f"PnL=${self.best_pnl:+.2f} | "
                      f"prob={self.best_params['prob_threshold']:.2f} "
                      f"tp={self.best_params['tp_pct']:.3f} "
                      f"tsl={self.best_params['tsl_pct']:.3f} "
                      f"min_diff={self.best_params['min_proba_diff']:.2f} | "
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
                 enable_partial_tp: bool = True):
        
        self.ticker = ticker
        self.timeframe = timeframe
        self.helper_timeframes = helper_timeframes
        self.limit = limit
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
        self.enable_partial_tp = enable_partial_tp
        
        # Load models
        self._load_models()
        
        # Prepare data once
        self.df = self._prepare_data()
        
    def _get_strategy_id(self, side: str) -> str:
        """Generate strategy ID"""
        helpers = '_plus_' + '_'.join(self.helper_timeframes) if self.helper_timeframes else ""
        return f"{self.ticker}_{self.timeframe.replace(' ', '')}{helpers}_{side}"
    
    def _load_models(self):
        """Load trained models for long and short"""
        long_id = self._get_strategy_id('long')
        short_id = self._get_strategy_id('short')
        
        try:
            self.model_long = joblib.load(f"models/{long_id}_model.joblib")
            self.scaler_long = joblib.load(f"models/{long_id}_scaler.joblib")
            self.features_long = joblib.load(f"models/{long_id}_features.joblib")
            
            self.model_short = joblib.load(f"models/{short_id}_model.joblib")
            self.scaler_short = joblib.load(f"models/{short_id}_scaler.joblib")
            self.features_short = joblib.load(f"models/{short_id}_features.joblib")
            
            logging.info(f"✓ Models loaded successfully")
        except FileNotFoundError as e:
            logging.error(f"Model files not found: {e}")
            raise
    
    def _prepare_data(self) -> pd.DataFrame:
        """Prepare data for backtesting"""
        logging.info(f"Preparing data: {self.ticker} {self.timeframe}")
        df = fetch_and_prepare_data(
            ticker=self.ticker,
            timeframe=self.timeframe,
            limit=self.limit,
            helper_timeframes=self.helper_timeframes,
            side='backtest'
        )
        
        if df.empty:
            raise ValueError("Failed to prepare data")
        
        logging.info(f"✓ Data prepared: {len(df)} candles")
        return df
    
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
        prob_threshold = trial.suggest_float('prob_threshold', 0.5, 0.9, step=0.05)
        tp_pct = trial.suggest_float('tp_pct', 0.02, 0.15, step=0.01)
        tsl_pct = trial.suggest_float('tsl_pct', 0.01, 0.08, step=0.005)
        min_proba_diff = trial.suggest_float('min_proba_diff', 0.0, 0.7, step=0.05)
        
        # Ensure TSL is not greater than TP (logical constraint)
        if tsl_pct >= tp_pct:
            return -999999.0, -100.0, -999.0  # Invalid combination
        
        # Calculate SL as 50% of TP
        sl_pct = tp_pct * 0.5
        
        # Run backtest with suggested parameters
        engine = BacktestEngine(initial_capital=self.initial_capital)
        
        try:
            results = engine.run(
                df=self.df,
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
                min_proba_diff=min_proba_diff
            )
            
            # Calculate metrics
            metrics = calculate_metrics(
                results['trades'], 
                results['equity_curve'], 
                self.initial_capital
            )
            
            # Extract metrics
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
            # Penalize too few trades (prevents 1-trade strategies)
            # Reward optimal range: 50-200 trades for 10k candles
            # Slightly penalize excessive trading (> 300 trades)
            if num_trades < 10:
                return -999999.0, -100.0, -999.0  # Reject if too few trades
            elif num_trades < 30:
                obj3_trade_score = num_trades * 2  # Encourage more trades
            elif num_trades < 200:
                obj3_trade_score = 100 + (num_trades - 30) * 0.5  # Optimal range
            else:
                obj3_trade_score = 185 - (num_trades - 200) * 0.1  # Slight penalty for over-trading
            
            # Log trial results
            logging.info(
                f"Trial {trial.number}: "
                f"prob={prob_threshold:.2f}, tp={tp_pct:.3f}, tsl={tsl_pct:.3f}, min_diff={min_proba_diff:.2f} | "
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
            
            return obj1_pnl, obj2_drawdown, obj3_trade_score
            
        except Exception as e:
            logging.error(f"Trial {trial.number} failed: {e}")
            return -999999.0, -100.0, -999.0
    
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
        logging.info(f"Data points: {len(self.df)}")
        logging.info(f"{'='*70}\n")
        
        # Create database directory for persistent storage
        os.makedirs("optuna", exist_ok=True)
        
        # Generate unique study name based on ticker, timeframe, and helper timeframes
        # This allows resuming optimization for same configuration in the future
        helpers_str = '_'.join(self.helper_timeframes) if self.helper_timeframes else 'none'
        study_name = f"{self.ticker}_{self.timeframe}_{helpers_str}_multi_opt"
        
        # SQLite database storage for persistent optimization history
        storage_url = f"sqlite:///optuna/optuna_{self.ticker}_{self.timeframe}_{helpers_str}.db"
        
        logging.info(f"Database: {storage_url}")
        logging.info(f"Study name: {study_name}")
        logging.info(f"Load if exists: True (can resume previous optimizations)\n")
        
        # Create multi-objective study (3 objectives, all to maximize)
        # Objective 1: PnL (highest priority)
        # Objective 2: -Drawdown (second priority)
        # Objective 3: Trade count score (third priority)
        # load_if_exists=True allows resuming previous optimizations
        study = optuna.create_study(
            directions=['maximize', 'maximize', 'maximize'],
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True
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
        
        # Create callback to display best trial info during optimization
        callback = BestTrialCallback()
        
        # Run optimization with parallel execution (n_jobs=-1 uses all CPU cores)
        # Callback will display best results after each trial
        # Note: show_progress_bar works cleanly only with n_jobs=1 to avoid multiple progress bars
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1, callbacks=[callback])
        
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
        
        # Run final backtest with best parameters
        logging.info(f"\n{'='*70}")
        logging.info(f"{'OPTIMIZATION COMPLETE - RUNNING FINAL BACKTEST':^70}")
        logging.info(f"{'='*70}\n")
        
        engine = BacktestEngine(initial_capital=self.initial_capital)
        
        final_results = engine.run(
            df=self.df,
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
            min_proba_diff=best_params['min_proba_diff']
        )
        
        final_metrics = calculate_metrics(
            final_results['trades'],
            final_results['equity_curve'],
            self.initial_capital
        )
        
        # Print results
        self._print_optimization_results(best_params, final_metrics, study, best_trial, best_trials)
        
        return {
            'best_params': best_params,
            'metrics': final_metrics,
            'study': study,
            'best_trial': best_trial,
            'pareto_trials': best_trials
        }
    
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
        results_file = f"optuna/optimization_results_{self.ticker}_{self.timeframe}.txt"
        os.makedirs("optuna", exist_ok=True)
        
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
            f.write(f"SL_PCT={best_params['tp_pct'] * 0.5:.3f}\n\n")
            
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
            f.write(f"Selected Trial: #{best_trial.number}\n")
        
        logging.info(f"✓ Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Optimize backtester parameters using Optuna")
    
    parser.add_argument('--ticker', type=str, required=True, help="Ticker symbol (e.g., SOLUSDT)")
    parser.add_argument('--timeframe', type=str, required=True, help="Main timeframe (e.g., 1h)")
    parser.add_argument('--helper-timeframes', nargs='*', default=None, help="Helper timeframes")
    parser.add_argument('--limit', type=int, default=2000, help="Number of candles for backtest")
    parser.add_argument('--trials', type=int, default=100, help="Number of Optuna trials")
    parser.add_argument('--initial-capital', type=float, default=10000.0, help="Initial capital")
    parser.add_argument('--risk-pct', type=float, default=0.02, help="Risk per trade (% of capital)")
    parser.add_argument('--partial-tp', action='store_true', help="Enable partial take profit")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = BacktesterOptimizer(
        ticker=args.ticker,
        timeframe=args.timeframe,
        helper_timeframes=args.helper_timeframes,
        limit=args.limit,
        initial_capital=args.initial_capital,
        risk_pct=args.risk_pct,
        enable_partial_tp=args.partial_tp
    )
    
    # Run optimization
    results = optimizer.optimize(n_trials=args.trials)
    
    print("\n✓ Optimization complete!")
    print(f"✓ Best parameters saved to: optuna/optimization_results_{args.ticker}_{args.timeframe}.txt")
    print("\nYou can now update your run_solusdt_workflow.sh with these optimal parameters.")


if __name__ == "__main__":
    main()
