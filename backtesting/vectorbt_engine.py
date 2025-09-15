#!/usr/bin/env python3
"""
Vectorbt-based Backtesting Engine for Adaptive Trading Bot

This module implements a high-performance backtesting system using vectorbt
to replace the custom backtesting engine, providing 100x faster execution
as recommended in ANALIZA_BOTA_ADAPTACYJNEGO.md
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Configure vectorbt logging to reduce file watching noise
def _configure_vectorbt_logging():
    """Configure vectorbt to reduce file watching noise"""
    # Set vectorbt and watchdog loggers to WARNING level
    logging.getLogger('vectorbt').setLevel(logging.WARNING)
    logging.getLogger('watchdog').setLevel(logging.WARNING)
    
    # Filter out cache-related messages
    class VectorbtCacheFilter(logging.Filter):
        def filter(self, record):
            message = record.getMessage()
            return not any(keyword in message for keyword in [
                'Ignoring changed path',
                '__pycache__',
                'vectorbt/labels',
                'Watched_paths:'
            ])
    
    # Apply filter to existing handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(VectorbtCacheFilter())

# Apply the fix immediately when module is imported
_configure_vectorbt_logging()

# Import our modules
from core.regime_detector import RegimeDetector, MarketRegime
from strategies.consolidation_strategy import ConsolidationStrategy, SignalType as ConsSignalType
from strategies.trend_strategy import TrendStrategy, TrendSignalType
from core.risk_manager import RiskManager, PositionSide, Position
from indicators.technical import TechnicalIndicators
from core.portfolio_sync import integrate_backtest_results

# Setup logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class VectorbtBacktestConfig:
    """Configuration for vectorbt backtesting"""
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.02
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    max_positions: int = 1
    enable_short: bool = True
    freq: str = '15min'

@dataclass
class VectorbtResults:
    """Vectorbt backtest results"""
    portfolio: vbt.Portfolio
    returns: pd.Series
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    regime_performance: Dict[str, Dict[str, float]]
    drawdowns: pd.Series
    underwater_curve: pd.Series
    
    def __post_init__(self):
        """Calculate additional metrics after initialization"""
        self._calculate_extended_metrics()
    
    def _calculate_extended_metrics(self):
        """Calculate additional performance metrics"""
        if self.portfolio is not None:
            stats = self.portfolio.stats()
            
            # Add custom metrics
            self.metrics.update({
                'sharpe_ratio': stats.get('Sharpe Ratio', 0),
                'calmar_ratio': stats.get('Calmar Ratio', 0),
                'max_drawdown': stats.get('Max Drawdown [%]', 0),
                'win_rate': stats.get('Win Rate [%]', 0),
                'profit_factor': stats.get('Profit Factor', 0),
                'total_trades': stats.get('# Trades', 0),
                'avg_trade_duration': stats.get('Avg Trade Duration', pd.Timedelta(0)),
            })

class VectorbtAdaptiveEngine:
    """
    High-performance adaptive trading bot backtesting using vectorbt
    
    This engine replaces the custom backtesting implementation with vectorbt's
    optimized vectorized operations for significant performance improvements.
    """
    
    def __init__(self, config: VectorbtBacktestConfig = None):
        """Initialize vectorbt backtesting engine"""
        self.config = config or VectorbtBacktestConfig()
        
        # Initialize components
        self.regime_detector = RegimeDetector()
        self.trend_strategy = TrendStrategy()
        self.consolidation_strategy = ConsolidationStrategy()
        self.risk_manager = RiskManager(
            initial_capital=self.config.initial_capital,
            risk_per_trade=self.config.risk_per_trade
        )
        self.technical = TechnicalIndicators()
        
        # Performance tracking
        self.regime_stats = {}
        
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with all necessary indicators and regime detection
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Enhanced DataFrame with indicators and regimes
        """
        logger.info("Preparing data with indicators and regime detection")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Add technical indicators
        df = self.technical.add_all_indicators(df)
        
        # Add regime detection
        regimes = []
        regime_confidence = []
        
        for i in range(len(df)):
            if i < 50:  # Need minimum data for regime detection
                regimes.append(MarketRegime.CONSOLIDATION.value)
                regime_confidence.append(0.5)
            else:
                current_data = df.iloc[:i+1]
                regime, confidence = self.regime_detector.detect_regime(current_data)
                regimes.append(regime.value)
                regime_confidence.append(confidence)
        
        df['regime'] = regimes
        df['regime_confidence'] = regime_confidence
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate buy/sell signals based on regime-adaptive strategies
        
        Args:
            data: Prepared DataFrame with indicators and regimes
            
        Returns:
            Tuple of (buy_signals, sell_signals) as boolean Series
        """
        logger.info("Generating adaptive trading signals")
        
        buy_signals = pd.Series(False, index=data.index)
        sell_signals = pd.Series(False, index=data.index)
        
        # Process signals for each regime
        for i in range(len(data)):
            if i < 50:  # Skip initial period
                continue
                
            current_data = data.iloc[:i+1]
            current_row = data.iloc[i]
            regime = MarketRegime(current_row['regime'])
            
            # Generate signals based on current regime
            if regime == MarketRegime.TRENDING:
                # Use trend following strategy
                signal = self._get_trend_signal(current_data, i)
                if signal == TrendSignalType.LONG_ENTRY:
                    buy_signals.iloc[i] = True
                elif signal == TrendSignalType.SHORT_ENTRY and self.config.enable_short:
                    sell_signals.iloc[i] = True
                    
            elif regime == MarketRegime.CONSOLIDATION:
                # Use mean reversion strategy
                signal = self._get_consolidation_signal(current_data, i)
                if signal == ConsSignalType.BUY:
                    buy_signals.iloc[i] = True
                elif signal == ConsSignalType.SELL and self.config.enable_short:
                    sell_signals.iloc[i] = True
        
        return buy_signals, sell_signals
    
    def _get_trend_signal(self, data: pd.DataFrame, idx: int) -> TrendSignalType:
        """Get trend following signal"""
        try:
            signal, confidence = self.trend_strategy.generate_signal(data.iloc[:idx+1])
            return signal
        except:
            return TrendSignalType.HOLD
    
    def _get_consolidation_signal(self, data: pd.DataFrame, idx: int) -> ConsSignalType:
        """Get consolidation/mean reversion signal"""
        try:
            signal, confidence = self.consolidation_strategy.generate_signal(data.iloc[:idx+1])
            return signal
        except:
            return ConsSignalType.HOLD
    
    def calculate_position_sizes(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate position sizes using risk management
        
        Args:
            data: OHLCV data
            signals: Trading signals
            
        Returns:
            Series with position sizes
        """
        position_sizes = pd.Series(0.0, index=data.index)
        
        for i, signal in enumerate(signals):
            if signal and i > 0:  # If there's a signal
                current_price = data.iloc[i]['close']
                
                # Calculate position size based on risk management
                # This is a simplified version - in production you'd use the full risk manager
                account_value = self.config.initial_capital  # Simplified
                risk_amount = account_value * self.config.risk_per_trade
                
                # Assume 2% stop loss for position sizing
                stop_distance = current_price * 0.02
                if stop_distance > 0:
                    position_size = risk_amount / stop_distance
                    # Convert to percentage of capital
                    position_sizes.iloc[i] = min(position_size / current_price / account_value, 1.0)
        
        return position_sizes
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        symbol: str = "BTC/USDT",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> VectorbtResults:
        """
        Run vectorbt-powered backtest
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            VectorbtResults with comprehensive metrics
        """
        logger.info(f"Running vectorbt backtest for {symbol}")
        
        # Filter data by date range if specified
        if start_date or end_date:
            mask = pd.Series(True, index=data.index)
            if start_date:
                mask = mask & (data.index >= start_date)
            if end_date:
                mask = mask & (data.index <= end_date)
            data = data[mask]
        
        # Prepare data
        prepared_data = self.prepare_data(data)
        
        # Generate signals
        buy_signals, sell_signals = self.generate_signals(prepared_data)
        
        # Calculate position sizes
        position_sizes = self.calculate_position_sizes(prepared_data, buy_signals | sell_signals)
        
        # Create vectorbt portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=prepared_data['close'],
            entries=buy_signals,
            exits=sell_signals,
            size=position_sizes,
            init_cash=self.config.initial_capital,
            fees=self.config.transaction_cost,
            slippage=self.config.slippage,
            freq=self.config.freq
        )
        
        # Calculate regime-specific performance
        regime_performance = self._calculate_regime_performance(
            prepared_data, portfolio
        )
        
        # Create results object
        # Note: vectorbt doesn't have underwater() method on drawdowns
        # We can calculate underwater curve manually or use alternative approach
        try:
            # Calculate underwater curve manually from equity curve
            equity_curve = portfolio.value()
            running_max = equity_curve.expanding().max()
            underwater_curve = (equity_curve - running_max) / running_max
        except Exception:
            # Fallback to empty series if calculation fails
            underwater_curve = pd.Series(0.0, index=portfolio.value().index)
        
        results = VectorbtResults(
            portfolio=portfolio,
            returns=portfolio.returns(),
            equity_curve=equity_curve,
            trades=portfolio.trades.records_readable,
            metrics={},
            regime_performance=regime_performance,
            drawdowns=portfolio.drawdowns.records_readable,
            underwater_curve=underwater_curve
        )
        
        logger.info(f"Backtest completed: {len(results.trades)} trades executed")
        
        # Integrate results with portfolio synchronization
        try:
            integrate_backtest_results(
                results=results,
                symbol=symbol,
                strategy="adaptive_bot",  # Could be made configurable
                timeframe="15m"  # Could be made configurable
            )
            logger.info(f"Backtest results integrated with portfolio synchronization for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to integrate backtest results with portfolio sync: {e}")
        
        return results
    
    def _calculate_regime_performance(
        self,
        data: pd.DataFrame,
        portfolio: vbt.Portfolio
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics by market regime"""
        regime_perf = {}
        
        for regime in MarketRegime:
            regime_mask = data['regime'] == regime.value
            if regime_mask.sum() == 0:
                continue
            
            # Get returns for this regime
            regime_returns = portfolio.returns()[regime_mask]
            
            if len(regime_returns) > 0:
                regime_perf[regime.name] = {
                    'total_return': regime_returns.sum(),
                    'avg_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    'periods': len(regime_returns),
                    'win_rate': (regime_returns > 0).mean()
                }
        
        return regime_perf
    
    def run_walk_forward_analysis(
        self,
        data: pd.DataFrame,
        symbol: str = "BTC/USDT",
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 1
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis using vectorbt
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol  
            train_months: Training period in months
            test_months: Testing period in months
            step_months: Step size in months
            
        Returns:
            Dictionary with walk-forward results
        """
        logger.info(f"Running walk-forward analysis for {symbol}")
        
        results = []
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_start = start_date
        
        while current_start + timedelta(days=30*(train_months + test_months)) <= end_date:
            # Define train and test periods
            train_end = current_start + timedelta(days=30*train_months)
            test_start = train_end
            test_end = test_start + timedelta(days=30*test_months)
            
            # Get train and test data
            train_data = data[(data.index >= current_start) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            
            if len(train_data) < 100 or len(test_data) < 10:
                current_start += timedelta(days=30*step_months)
                continue
            
            try:
                # Run backtest on test period
                test_results = self.run_backtest(
                    test_data,
                    symbol=symbol,
                    start_date=test_start,
                    end_date=test_end
                )
                
                period_result = {
                    'train_start': current_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'test_return': test_results.portfolio.total_return(),
                    'test_sharpe': test_results.metrics.get('sharpe_ratio', 0),
                    'test_drawdown': test_results.metrics.get('max_drawdown', 0),
                    'num_trades': len(test_results.trades)
                }
                
                results.append(period_result)
                logger.info(f"WFA period {test_start.date()} to {test_end.date()}: {period_result['test_return']:.2%}")
                
            except Exception as e:
                logger.error(f"Error in WFA period {test_start.date()}: {e}")
            
            # Move to next period
            current_start += timedelta(days=30*step_months)
        
        # Aggregate results
        if results:
            returns = [r['test_return'] for r in results]
            sharpes = [r['test_sharpe'] for r in results if r['test_sharpe'] != 0]
            
            summary = {
                'periods': len(results),
                'avg_return': np.mean(returns),
                'return_std': np.std(returns),
                'avg_sharpe': np.mean(sharpes) if sharpes else 0,
                'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                'best_period': max(returns),
                'worst_period': min(returns),
                'detailed_results': results
            }
        else:
            summary = {'error': 'No valid periods found'}
        
        return summary
    
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        param_ranges: Dict[str, List],
        symbol: str = "BTC/USDT"
    ) -> Dict[str, Any]:
        """
        Parameter optimization using vectorbt's optimization capabilities
        
        Args:
            data: OHLCV DataFrame
            param_ranges: Dictionary of parameter ranges to test
            symbol: Trading symbol
            
        Returns:
            Optimization results
        """
        logger.info("Running parameter optimization")
        
        # This is a simplified optimization - vectorbt has more advanced optimization features
        best_params = None
        best_score = float('-inf')
        results = []
        
        # Generate parameter combinations (simplified grid search)
        param_combinations = self._generate_param_combinations(param_ranges)
        
        for params in param_combinations[:50]:  # Limit combinations for demo
            try:
                # Update strategy parameters
                self._update_strategy_params(params)
                
                # Run backtest
                result = self.run_backtest(data, symbol)
                
                # Calculate optimization score (can be customized)
                score = result.portfolio.sharpe_ratio() if hasattr(result.portfolio, 'sharpe_ratio') else 0
                
                results.append({
                    'params': params.copy(),
                    'score': score,
                    'total_return': result.portfolio.total_return(),
                    'num_trades': len(result.trades)
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                logger.error(f"Error optimizing params {params}: {e}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def _generate_param_combinations(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """Generate parameter combinations for optimization"""
        # Simplified implementation - use itertools.product for full grid search
        combinations = []
        
        # For demonstration, just sample some combinations
        import random
        for _ in range(20):
            combo = {}
            for param, values in param_ranges.items():
                combo[param] = random.choice(values)
            combinations.append(combo)
        
        return combinations
    
    def _update_strategy_params(self, params: Dict[str, Any]):
        """Update strategy parameters for optimization"""
        # Update risk management
        if 'risk_per_trade' in params:
            self.risk_manager.risk_per_trade = params['risk_per_trade']
        
        # Update other parameters as needed
        # This would be expanded based on what parameters you want to optimize
    
    def generate_report(self, results: VectorbtResults, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive performance report
        
        Args:
            results: Backtest results
            save_path: Optional path to save report
            
        Returns:
            Report as string
        """
        if results.portfolio is None:
            return "No portfolio data available"
        
        # Get portfolio stats
        stats = results.portfolio.stats()
        
        report_lines = [
            "=" * 80,
            "VECTORBT ADAPTIVE BOT BACKTEST REPORT",
            "=" * 80,
            "",
            "OVERALL PERFORMANCE:",
            f"  Total Return: {stats.get('Total Return [%]', 0):.2f}%",
            f"  Sharpe Ratio: {stats.get('Sharpe Ratio', 0):.3f}",
            f"  Max Drawdown: {stats.get('Max Drawdown [%]', 0):.2f}%",
            f"  Win Rate: {stats.get('Win Rate [%]', 0):.1f}%",
            f"  Total Trades: {stats.get('# Trades', 0)}",
            f"  Profit Factor: {stats.get('Profit Factor', 0):.2f}",
            "",
            "REGIME-SPECIFIC PERFORMANCE:",
        ]
        
        # Add regime performance
        for regime, metrics in results.regime_performance.items():
            report_lines.extend([
                f"  {regime}:",
                f"    Total Return: {metrics['total_return']:.4f}",
                f"    Average Return: {metrics['avg_return']:.4f}",
                f"    Sharpe Ratio: {metrics['sharpe_ratio']:.3f}",
                f"    Win Rate: {metrics['win_rate']:.1%}",
                f"    Periods: {metrics['periods']}",
                ""
            ])
        
        report_lines.extend([
            "TRADE STATISTICS:",
            f"  Average Trade Duration: {stats.get('Avg Trade Duration', 'N/A')}",
            f"  Best Trade: {stats.get('Best Trade [%]', 0):.2f}%",
            f"  Worst Trade: {stats.get('Worst Trade [%]', 0):.2f}%",
            "",
            "=" * 80
        ])
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
        
        return report
    
    def generate_trading_chart(
        self,
        results: VectorbtResults,
        data: pd.DataFrame,
        symbol: str = "BTC/USDT",
        save_path: Optional[str] = None,
        show_regime: bool = True,
        show_levels: bool = True
    ) -> go.Figure:
        """
        Generate interactive trading chart with entry/exit points
        
        Args:
            results: Backtest results
            data: Original OHLCV data with indicators
            symbol: Trading symbol
            save_path: Optional path to save chart
            show_regime: Whether to show regime detection overlay
            show_levels: Whether to show stop-loss/take-profit levels
            
        Returns:
            Plotly figure object
        """
        logger.info(f"Generating trading chart for {symbol}")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{symbol} Price & Trading Signals',
                'Portfolio Value',
                'Market Regime'
            ),
            row_heights=[0.6, 0.25, 0.15]
        )
        
        # 1. Price chart with candlesticks
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'], 
                low=data['low'],
                close=data['close'],
                name=symbol,
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # 2. Add moving averages if available
        if 'ema_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['ema_50'],
                    mode='lines',
                    name='EMA 50',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        # 3. Add entry/exit markers from trades
        if hasattr(results, 'trades') and not results.trades.empty:
            trades_df = results.trades
            
            # Entry points (green triangles up)
            if 'Entry Timestamp' in trades_df.columns and 'Entry Price' in trades_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=trades_df['Entry Timestamp'],
                        y=trades_df['Entry Price'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green',
                            line=dict(width=2, color='darkgreen')
                        ),
                        name='Entry Points',
                        hovertemplate='<b>Entry</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Exit points (red triangles down) - color coded by P&L
            if 'Exit Timestamp' in trades_df.columns and 'Exit Price' in trades_df.columns:
                profitable_trades = trades_df[trades_df.get('PnL', trades_df.get('Return', 0)) > 0]
                losing_trades = trades_df[trades_df.get('PnL', trades_df.get('Return', 0)) <= 0]
                
                # Profitable exits (green)
                if not profitable_trades.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=profitable_trades['Exit Timestamp'],
                            y=profitable_trades['Exit Price'],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color='lightgreen',
                                line=dict(width=2, color='green')
                            ),
                            name='Profitable Exits',
                            hovertemplate='<b>Profitable Exit</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                
                # Losing exits (red)
                if not losing_trades.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=losing_trades['Exit Timestamp'],
                            y=losing_trades['Exit Price'],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color='lightcoral',
                                line=dict(width=2, color='red')
                            ),
                            name='Loss Exits',
                            hovertemplate='<b>Loss Exit</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
                        ),
                        row=1, col=1
                    )
        
        # 4. Portfolio value chart
        if hasattr(results, 'portfolio') and results.portfolio is not None:
            portfolio_value = results.portfolio.value()
            fig.add_trace(
                go.Scatter(
                    x=portfolio_value.index,
                    y=portfolio_value.values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2),
                    fill='tonexty' if len(fig.data) == 0 else None,
                    hovertemplate='<b>Portfolio Value</b><br>$%{y:,.2f}<br>%{x}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 5. Market Regime overlay
        if show_regime and 'regime' in data.columns:
            # Create regime color mapping
            regime_colors = {
                'TRENDING': 'rgba(0, 255, 0, 0.3)',
                'CONSOLIDATION': 'rgba(255, 255, 0, 0.3)', 
                'STAGNANT': 'rgba(128, 128, 128, 0.3)',
                'PANIC': 'rgba(255, 0, 0, 0.3)',
                'TRANSITION': 'rgba(128, 0, 128, 0.3)'
            }
            
            # Add regime background colors
            current_regime = None
            regime_start = None
            
            for i, (timestamp, row) in enumerate(data.iterrows()):
                regime = MarketRegime(row['regime']).name if pd.notna(row['regime']) else 'TRANSITION'
                
                if regime != current_regime:
                    # End previous regime
                    if current_regime is not None and regime_start is not None:
                        fig.add_vrect(
                            x0=regime_start,
                            x1=timestamp,
                            fillcolor=regime_colors.get(current_regime, 'rgba(128, 128, 128, 0.1)'),
                            layer="below",
                            line_width=0,
                            row=1, col=1
                        )
                    
                    # Start new regime
                    current_regime = regime
                    regime_start = timestamp
            
            # Add final regime
            if current_regime is not None and regime_start is not None:
                fig.add_vrect(
                    x0=regime_start,
                    x1=data.index[-1],
                    fillcolor=regime_colors.get(current_regime, 'rgba(128, 128, 128, 0.1)'),
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )
            
            # Add regime indicator in bottom subplot
            regime_numeric = data['regime'].map({
                MarketRegime.STAGNANT.value: 0,
                MarketRegime.CONSOLIDATION.value: 1,
                MarketRegime.TRENDING.value: 2,
                MarketRegime.PANIC.value: 3,
            }).fillna(0.5)  # TRANSITION
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=regime_numeric,
                    mode='lines',
                    name='Market Regime',
                    line=dict(color='purple', width=2),
                    hovertemplate='<b>Regime</b><br>%{text}<br>%{x}<extra></extra>',
                    text=[MarketRegime(r).name if pd.notna(r) else 'TRANSITION' for r in data['regime']]
                ),
                row=3, col=1
            )
        
        # 6. Add stop-loss and take-profit levels if available
        if show_levels and hasattr(results, 'trades') and not results.trades.empty:
            trades_df = results.trades
            
            # Add stop-loss levels as dashed red lines
            for _, trade in trades_df.iterrows():
                if 'Entry Timestamp' in trade and 'Exit Timestamp' in trade:
                    entry_time = trade['Entry Timestamp']
                    exit_time = trade['Exit Timestamp']
                    
                    # Estimate stop-loss level (2% below entry for long positions)
                    entry_price = trade.get('Entry Price', 0)
                    if entry_price > 0:
                        stop_loss_level = entry_price * 0.98  # Simplified 2% stop
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[entry_time, exit_time],
                                y=[stop_loss_level, stop_loss_level],
                                mode='lines',
                                line=dict(color='red', dash='dash', width=1),
                                name='Stop Loss',
                                showlegend=False,
                                opacity=0.7
                            ),
                            row=1, col=1
                        )
        
        # Update layout
        fig.update_layout(
            title=f'Trading Analysis: {symbol}',
            xaxis_title='Date',
            height=800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
        fig.update_yaxes(
            title_text="Regime", 
            row=3, col=1,
            tickmode='array',
            tickvals=[0, 1, 2, 3],
            ticktext=['Stagnant', 'Consolidation', 'Trending', 'Panic']
        )
        
        # Save chart if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Trading chart saved to {save_path}")
        
        return fig
    
    def generate_strategy_analysis(
        self,
        results: VectorbtResults,
        data: pd.DataFrame,
        symbol: str = "BTC/USDT"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive strategy analysis with improvement recommendations
        
        Args:
            results: Backtest results
            data: Original OHLCV data with indicators
            symbol: Trading symbol
            
        Returns:
            Dictionary with analysis and recommendations
        """
        logger.info(f"Generating strategy analysis for {symbol}")
        
        analysis = {
            'symbol': symbol,
            'performance_summary': {},
            'regime_analysis': {},
            'trade_analysis': {},
            'risk_analysis': {},
            'recommendations': [],
            'parameter_suggestions': {}
        }
        
        # 1. Performance Summary
        if hasattr(results, 'portfolio') and results.portfolio is not None:
            stats = results.portfolio.stats()
            
            analysis['performance_summary'] = {
                'total_return': stats.get('Total Return [%]', 0),
                'sharpe_ratio': stats.get('Sharpe Ratio', 0),
                'max_drawdown': stats.get('Max Drawdown [%]', 0),
                'win_rate': stats.get('Win Rate [%]', 0),
                'total_trades': stats.get('# Trades', 0),
                'profit_factor': stats.get('Profit Factor', 0),
                'avg_trade_duration': stats.get('Avg Trade Duration', pd.Timedelta(0)),
                'best_trade': stats.get('Best Trade [%]', 0),
                'worst_trade': stats.get('Worst Trade [%]', 0)
            }
        
        # 2. Regime-specific Analysis
        analysis['regime_analysis'] = results.regime_performance.copy()
        
        # Find best and worst performing regimes
        if results.regime_performance:
            regime_returns = {regime: metrics['total_return'] 
                            for regime, metrics in results.regime_performance.items()}
            best_regime = max(regime_returns, key=regime_returns.get)
            worst_regime = min(regime_returns, key=regime_returns.get)
            
            analysis['regime_analysis']['best_regime'] = best_regime
            analysis['regime_analysis']['worst_regime'] = worst_regime
        
        # 3. Trade Analysis
        if hasattr(results, 'trades') and not results.trades.empty:
            trades_df = results.trades
            
            # Calculate trade statistics
            if 'PnL' in trades_df.columns or 'Return' in trades_df.columns:
                pnl_col = 'PnL' if 'PnL' in trades_df.columns else 'Return'
                profitable_trades = trades_df[trades_df[pnl_col] > 0]
                losing_trades = trades_df[trades_df[pnl_col] <= 0]
                
                analysis['trade_analysis'] = {
                    'total_trades': len(trades_df),
                    'profitable_trades': len(profitable_trades),
                    'losing_trades': len(losing_trades),
                    'avg_profit': profitable_trades[pnl_col].mean() if len(profitable_trades) > 0 else 0,
                    'avg_loss': losing_trades[pnl_col].mean() if len(losing_trades) > 0 else 0,
                    'largest_winner': trades_df[pnl_col].max(),
                    'largest_loser': trades_df[pnl_col].min(),
                    'consecutive_wins': self._calculate_consecutive_streak(trades_df[pnl_col] > 0),
                    'consecutive_losses': self._calculate_consecutive_streak(trades_df[pnl_col] <= 0)
                }
        
        # 4. Risk Analysis
        if hasattr(results, 'portfolio') and results.portfolio is not None:
            returns = results.portfolio.returns()
            
            analysis['risk_analysis'] = {
                'volatility': returns.std() * np.sqrt(252),  # Annualized volatility
                'downside_deviation': returns[returns < 0].std() * np.sqrt(252),
                'var_95': returns.quantile(0.05),  # Value at Risk (95%)
                'max_daily_loss': returns.min(),
                'positive_days': (returns > 0).mean(),
                'negative_days': (returns < 0).mean()
            }
        
        # 5. Generate Recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        # 6. Parameter Optimization Suggestions
        analysis['parameter_suggestions'] = self._generate_parameter_suggestions(analysis)
        
        return analysis
    
    def _calculate_consecutive_streak(self, boolean_series: pd.Series) -> int:
        """Calculate maximum consecutive streak of True values"""
        if len(boolean_series) == 0:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for value in boolean_series:
            if value:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable improvement recommendations"""
        recommendations = []
        
        perf = analysis.get('performance_summary', {})
        regime_analysis = analysis.get('regime_analysis', {})
        trade_analysis = analysis.get('trade_analysis', {})
        risk_analysis = analysis.get('risk_analysis', {})
        
        # Performance-based recommendations
        sharpe_ratio = perf.get('sharpe_ratio', 0)
        if sharpe_ratio < 1.0:
            recommendations.append("🔴 LOW SHARPE RATIO: Consider tightening entry criteria or improving position sizing to reduce risk-adjusted returns.")
        elif sharpe_ratio > 2.0:
            recommendations.append("🟢 EXCELLENT SHARPE RATIO: Your strategy has strong risk-adjusted returns. Consider increasing position sizes carefully.")
        
        win_rate = perf.get('win_rate', 0)
        if win_rate < 45:
            recommendations.append("🔴 LOW WIN RATE: Focus on better entry timing or consider trend-following approaches in trending markets.")
        elif win_rate > 65:
            recommendations.append("🟢 HIGH WIN RATE: Consider taking larger positions or holding winners longer to maximize profits.")
        
        max_drawdown = abs(perf.get('max_drawdown', 0))
        if max_drawdown > 15:
            recommendations.append("🔴 HIGH DRAWDOWN: Implement tighter stop-losses or reduce position sizing during volatile periods.")
        elif max_drawdown < 5:
            recommendations.append("🟡 LOW DRAWDOWN: You might be too conservative. Consider slightly larger positions for better returns.")
        
        # Regime-based recommendations
        if regime_analysis:
            best_regime = regime_analysis.get('best_regime')
            worst_regime = regime_analysis.get('worst_regime')
            
            if best_regime and worst_regime:
                recommendations.append(f"📊 REGIME INSIGHT: Best performance in {best_regime} markets, worst in {worst_regime}. Consider avoiding trades in {worst_regime} conditions.")
            
            # Check regime-specific win rates
            for regime, metrics in regime_analysis.items():
                if isinstance(metrics, dict) and 'win_rate' in metrics:
                    regime_wr = metrics['win_rate']
                    if regime_wr < 0.3:  # Less than 30% win rate
                        recommendations.append(f"🔴 POOR {regime.upper()} PERFORMANCE: Consider disabling trading during {regime.lower()} market conditions.")
        
        # Trade analysis recommendations
        if trade_analysis:
            avg_profit = trade_analysis.get('avg_profit', 0)
            avg_loss = abs(trade_analysis.get('avg_loss', 0))
            
            if avg_loss > 0 and avg_profit > 0:
                profit_loss_ratio = avg_profit / avg_loss
                if profit_loss_ratio < 1.5:
                    recommendations.append("🔴 POOR PROFIT/LOSS RATIO: Let winners run longer or cut losses sooner to improve the average profit per trade.")
                elif profit_loss_ratio > 3.0:
                    recommendations.append("🟢 EXCELLENT PROFIT/LOSS RATIO: Your risk management is working well. Consider increasing trade frequency.")
            
            consecutive_losses = trade_analysis.get('consecutive_losses', 0)
            if consecutive_losses > 5:
                recommendations.append(f"🔴 HIGH CONSECUTIVE LOSSES ({consecutive_losses}): Implement circuit breakers to pause trading after 3-4 consecutive losses.")
        
        # Risk analysis recommendations
        if risk_analysis:
            volatility = risk_analysis.get('volatility', 0)
            if volatility > 0.3:  # 30% annual volatility
                recommendations.append("🔴 HIGH VOLATILITY: Consider position sizing based on market volatility (ATR-based sizing).")
            
            positive_days = risk_analysis.get('positive_days', 0)
            if positive_days < 0.4:  # Less than 40% positive days
                recommendations.append("🔴 LOW POSITIVE DAYS: Focus on trend identification to avoid choppy, sideways markets.")
        
        # General strategy improvements
        total_trades = trade_analysis.get('total_trades', 0)
        if total_trades < 20:
            recommendations.append("🟡 LOW TRADE COUNT: Consider longer backtesting period or relaxing entry criteria for more statistical significance.")
        elif total_trades > 200:
            recommendations.append("🟡 HIGH TRADE COUNT: Ensure you're not overtrading. Quality over quantity - focus on best setups only.")
        
        if not recommendations:
            recommendations.append("🟢 STRATEGY PERFORMING WELL: Continue monitoring and consider minor optimizations to parameter sensitivity.")
        
        return recommendations
    
    def _generate_parameter_suggestions(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate parameter optimization suggestions"""
        suggestions = {}
        
        perf = analysis.get('performance_summary', {})
        
        # Suggest parameter ranges for optimization
        if perf.get('sharpe_ratio', 0) < 1.0:
            suggestions['risk_per_trade'] = "Try testing 1%-3% range. Current risk might be too high for the strategy's edge."
            suggestions['adx_threshold'] = "Test ADX thresholds 20-30. Lower values might catch trends earlier."
        
        if perf.get('win_rate', 0) < 45:
            suggestions['entry_criteria'] = "Consider stricter entry filters: higher ADX, volume confirmation, or EMA alignment."
            suggestions['stop_loss'] = "Test tighter stops (1.5-2.5x ATR) if current stops are too wide."
        
        if abs(perf.get('max_drawdown', 0)) > 15:
            suggestions['position_sizing'] = "Reduce position sizes during high volatility periods (when ATR ratio > 1.5)."
            suggestions['regime_filter'] = "Avoid trading in PANIC regimes or when regime confidence < 0.7."
        
        # Regime-specific suggestions
        regime_analysis = analysis.get('regime_analysis', {})
        if regime_analysis:
            worst_regime = regime_analysis.get('worst_regime')
            if worst_regime:
                suggestions['regime_trading'] = f"Consider disabling {worst_regime} trading or using different parameters."
        
        return suggestions
    
    def generate_improvement_report(
        self,
        analysis: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive improvement report
        
        Args:
            analysis: Strategy analysis results
            save_path: Optional path to save report
            
        Returns:
            Report as string
        """
        symbol = analysis.get('symbol', 'Unknown')
        
        report_lines = [
            "=" * 80,
            f"STRATEGY IMPROVEMENT ANALYSIS: {symbol}",
            "=" * 80,
            "",
            "📊 PERFORMANCE OVERVIEW:",
        ]
        
        # Performance summary
        perf = analysis.get('performance_summary', {})
        report_lines.extend([
            f"  Total Return: {perf.get('total_return', 0):.2f}%",
            f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}",
            f"  Max Drawdown: {perf.get('max_drawdown', 0):.2f}%", 
            f"  Win Rate: {perf.get('win_rate', 0):.1f}%",
            f"  Total Trades: {perf.get('total_trades', 0)}",
            f"  Profit Factor: {perf.get('profit_factor', 0):.2f}",
            ""
        ])
        
        # Recommendations
        report_lines.extend([
            "🎯 IMPROVEMENT RECOMMENDATIONS:",
            ""
        ])
        
        for i, rec in enumerate(analysis.get('recommendations', []), 1):
            report_lines.append(f"  {i}. {rec}")
        
        report_lines.append("")
        
        # Parameter suggestions
        param_suggestions = analysis.get('parameter_suggestions', {})
        if param_suggestions:
            report_lines.extend([
                "⚙️ PARAMETER OPTIMIZATION SUGGESTIONS:",
                ""
            ])
            
            for param, suggestion in param_suggestions.items():
                report_lines.append(f"  • {param.upper()}: {suggestion}")
            
            report_lines.append("")
        
        # Regime analysis
        regime_analysis = analysis.get('regime_analysis', {})
        if regime_analysis and isinstance(regime_analysis, dict):
            report_lines.extend([
                "🌐 REGIME-SPECIFIC PERFORMANCE:",
                ""
            ])
            
            for regime, metrics in regime_analysis.items():
                if isinstance(metrics, dict) and 'total_return' in metrics:
                    report_lines.extend([
                        f"  {regime.upper()}:",
                        f"    Return: {metrics.get('total_return', 0):.4f}",
                        f"    Win Rate: {metrics.get('win_rate', 0):.1%}",
                        f"    Trades: {metrics.get('periods', 0)}",
                        ""
                    ])
        
        # Next steps
        report_lines.extend([
            "🚀 NEXT STEPS:",
            "  1. Implement the highest priority recommendations (marked 🔴)",
            "  2. Run parameter optimization tests on suggested ranges",
            "  3. Test regime-specific settings if performance varies significantly",
            "  4. Monitor out-of-sample performance with new parameters",
            "  5. Consider walk-forward analysis for robust optimization",
            "",
            "=" * 80
        ])
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Improvement report saved to {save_path}")
        
        return report