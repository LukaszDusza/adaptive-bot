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
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function - OPTIMIZED FOR MAXIMUM TOTAL PNL
        
        Scoring formula:
        score = total_pnl - (max_dd * 50) + (sharpe * 200)
        
        Priority:
        1. Maximize Total PnL (USD profit) - PRIMARY GOAL
        2. Limit Max Drawdown (risk control)
        3. Reward high Sharpe Ratio (quality filter)
        
        Returns: composite score (higher is better)
        """
        
        # Suggest parameters
        prob_threshold = trial.suggest_float('prob_threshold', 0.5, 0.9, step=0.05)
        tp_pct = trial.suggest_float('tp_pct', 0.02, 0.15, step=0.01)
        tsl_pct = trial.suggest_float('tsl_pct', 0.01, 0.08, step=0.005)
        
        # Ensure TSL is not greater than TP (logical constraint)
        if tsl_pct >= tp_pct:
            return -999.0  # Invalid combination
        
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
                enable_partial_tp=self.enable_partial_tp
            )
            
            # Calculate metrics
            metrics = calculate_metrics(
                results['trades'], 
                results['equity_curve'], 
                self.initial_capital
            )
            
            # Use composite score for optimization - FOCUS ON TOTAL PNL
            # Primary goal: Maximize absolute profit (Total PnL in USD)
            # Risk control: Penalize large drawdowns (DD * 50)
            # Quality filter: Reward good risk-adjusted returns (Sharpe * 200)
            if metrics['total_trades'] < 10:
                return -999.0  # Reject if too few trades
            
            total_pnl = metrics.get('total_pnl_usd', -999999)
            total_return = metrics.get('total_return_pct', -999)
            max_dd = abs(metrics.get('max_drawdown_pct', 100))
            sharpe = metrics.get('sharpe_ratio', 0)
            
            # Composite score: FOCUS ON TOTAL PNL
            # Primary: Total PnL (most important)
            # Secondary: Limit drawdown risk
            # Tertiary: Sharpe as quality filter
            score = total_pnl - (max_dd * 50) + (sharpe * 200)
            
            # Log trial results
            logging.info(
                f"Trial {trial.number}: "
                f"prob={prob_threshold:.2f}, tp={tp_pct:.3f}, tsl={tsl_pct:.3f} | "
                f"Score={score:.2f}, PnL=${total_pnl:+.2f}, Return={total_return:.2f}%, "
                f"DD={max_dd:.2f}%, Sharpe={sharpe:.3f}, Trades={metrics['total_trades']}"
            )
            
            # Store additional metrics in trial user attributes
            trial.set_user_attr('total_pnl_usd', total_pnl)
            trial.set_user_attr('total_return_pct', total_return)
            trial.set_user_attr('max_drawdown_pct', max_dd)
            trial.set_user_attr('sharpe_ratio', sharpe)
            trial.set_user_attr('total_trades', metrics['total_trades'])
            trial.set_user_attr('win_rate', metrics.get('win_rate', 0))
            trial.set_user_attr('profit_factor', metrics.get('profit_factor', 0))
            
            return score
            
        except Exception as e:
            logging.error(f"Trial {trial.number} failed: {e}")
            return -999.0
    
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
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=f"{self.ticker}_{self.timeframe}_optimization"
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best results
        best_trial = study.best_trial
        best_params = best_trial.params
        
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
            enable_partial_tp=self.enable_partial_tp
        )
        
        final_metrics = calculate_metrics(
            final_results['trades'],
            final_results['equity_curve'],
            self.initial_capital
        )
        
        # Print results
        self._print_optimization_results(best_params, final_metrics, study)
        
        return {
            'best_params': best_params,
            'metrics': final_metrics,
            'study': study
        }
    
    def _print_optimization_results(self, best_params: Dict, metrics: Dict, study: optuna.Study):
        """Print optimization results in a nice format"""
        print("\n" + "="*70)
        print(f"{'OPTIMIZATION RESULTS (OPTIMIZED FOR MAX TOTAL PNL)':^70}")
        print("="*70)
        
        print(f"\n{'BEST PARAMETERS:':^70}")
        print(f"  PROB_THRESHOLD = {best_params['prob_threshold']:.2f}")
        print(f"  TP_PCT         = {best_params['tp_pct']:.3f}")
        print(f"  TSL_PCT        = {best_params['tsl_pct']:.3f}")
        print(f"  SL_PCT         = {best_params['tp_pct'] * 0.5:.3f} (auto: 50% of TP)")
        
        print(f"\n{'PERFORMANCE WITH BEST PARAMETERS:':^70}")
        print(f"  🎯 Total PnL:     ${metrics.get('total_pnl_usd', 0):+.2f}  ⭐ PRIMARY METRIC")
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
        print(f"  Total Trials:     {len(study.trials)}")
        print(f"  Best Trial:       #{study.best_trial.number}")
        print(f"  Best Score:       {study.best_value:.3f}")
        
        print("="*70 + "\n")
        
        # Save results to file
        results_file = f"optuna/optimization_results_{self.ticker}_{self.timeframe}.txt"
        os.makedirs("optuna", exist_ok=True)
        
        with open(results_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"OPTIMIZATION RESULTS - {self.ticker} {self.timeframe}\n")
            f.write("="*70 + "\n\n")
            f.write("BEST PARAMETERS:\n")
            f.write(f"PROB_THRESHOLD={best_params['prob_threshold']:.2f}\n")
            f.write(f"TP_PCT={best_params['tp_pct']:.3f}\n")
            f.write(f"TSL_PCT={best_params['tsl_pct']:.3f}\n")
            f.write(f"SL_PCT={best_params['tp_pct'] * 0.5:.3f}\n\n")
            f.write("PERFORMANCE:\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
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
