"""
Backtesting Validator Module

Comprehensive backtesting system for the adaptive trading bot.
Integrates all components and provides rigorous validation with walk-forward analysis.

Features:
- Regime-aware backtesting with strategy switching
- Walk-forward analysis for robust validation
- Detailed performance metrics and regime-specific analytics
- Position management during regime changes
- Re-entry mechanism testing
- Risk management validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from pathlib import Path

# Import our modules
from core.regime_detector import RegimeDetector, MarketRegime
from strategies.consolidation_strategy import ConsolidationStrategy, SignalType as ConsSignalType
from strategies.trend_strategy import TrendStrategy, TrendSignalType
from core.risk_manager import RiskManager, PositionSide, Position
from indicators.technical import TechnicalIndicators


@dataclass
class BacktestResults:
    """Backtesting results container"""
    # Performance metrics
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trading metrics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Regime-specific results
    consolidation_trades: int
    trend_trades: int
    regime_change_trades: int
    
    # Risk metrics
    volatility: float
    var_95: float
    max_consecutive_losses: int
    
    # Detailed data
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    regime_log: pd.DataFrame
    monthly_returns: pd.DataFrame
    
    # Additional statistics
    start_date: datetime
    end_date: datetime
    duration_days: int


class AdaptiveBotBacktester:
    """
    Comprehensive backtesting system for the adaptive trading bot.
    
    Simulates the complete trading system including regime detection,
    strategy switching, risk management, and position handling during regime changes.
    """
    
    def __init__(self,
                 initial_capital: float = 10000,
                 risk_per_trade: float = 0.02,
                 transaction_cost: float = 0.001,  # 0.1% per trade
                 slippage: float = 0.0005,         # 0.05% slippage
                 timeframe: str = "15min"):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital
            risk_per_trade: Risk per trade as fraction of capital
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
            slippage: Price slippage as fraction
            timeframe: Data timeframe (15min recommended)
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.timeframe = timeframe
        
        # Initialize components
        self.regime_detector = RegimeDetector()
        self.consolidation_strategy = ConsolidationStrategy()
        self.trend_strategy = TrendStrategy()
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade
        )
        self.indicators = TechnicalIndicators()
        
        # Results storage
        self.results: Optional[BacktestResults] = None
        self.equity_curve: List[Dict] = []
        self.trade_log: List[Dict] = []
        self.regime_log: List[Dict] = []
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and validate input data for backtesting.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            Prepared DataFrame with all indicators
        """
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure proper datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        # Sort by date
        df = df.sort_index()
        
        # Calculate all required indicators
        data = self.regime_detector.calculate_indicators(df)
        data = self.consolidation_strategy.prepare_data(data)
        data = self.trend_strategy.prepare_data(data)
        
        # Add volume confirmation
        data = self.indicators.volume_confirmation(data)
        
        return data
    
    def run_backtest(self, 
                    df: pd.DataFrame,
                    symbol: str = "BTC",
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    verbose: bool = True) -> BacktestResults:
        """
        Run comprehensive backtest on historical data.
        
        Args:
            df: Historical OHLCV data
            symbol: Trading symbol
            start_date: Backtest start date
            end_date: Backtest end date
            verbose: Whether to print progress
            
        Returns:
            BacktestResults with comprehensive metrics
        """
        # Prepare data
        if verbose:
            print("Preparing data...")
        data = self.prepare_data(df)
        
        # Filter date range if specified
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if len(data) < 200:
            raise ValueError("Insufficient data for backtesting (minimum 200 bars required)")
        
        # Reset components
        self._reset_components()
        
        # Main backtesting loop
        if verbose:
            print(f"Running backtest from {data.index[0]} to {data.index[-1]} ({len(data)} bars)")
        
        for i in range(100, len(data)):  # Start after enough data for indicators
            current_row = data.iloc[i]
            timestamp = data.index[i]
            
            # Update daily tracking
            self.risk_manager.reset_daily_tracking(timestamp)
            
            # Update regime detector
            regime_state = self.regime_detector.update_regime(data.iloc[:i+1])
            
            # Log regime information
            self.regime_log.append({
                'timestamp': timestamp,
                'regime': regime_state.regime.value,
                'adx': regime_state.adx_value,
                'atr_ratio': regime_state.atr_ratio,
                'confidence': regime_state.confidence,
                'can_trade': self.regime_detector.can_open_positions()
            })
            
            # Update existing positions
            self._update_positions(current_row, timestamp)
            
            # Check for position exits (stop loss, take profit)
            self._check_exits(current_row, timestamp)
            
            # Generate new signals if regime allows trading
            if self.regime_detector.can_open_positions():
                self._process_signals(data, i, current_row, timestamp, symbol)
            
            # Handle regime changes for existing positions
            self._handle_regime_changes(data, i)
            
            # Check for re-entry opportunities
            self._check_reentry_opportunities(data, i, current_row, timestamp, symbol)
            
            # Record equity curve
            self._record_equity_point(timestamp)
            
            if verbose and i % 1000 == 0:
                progress = (i / len(data)) * 100
                print(f"Progress: {progress:.1f}% - Active positions: {len(self.risk_manager.positions)}")
        
        # Close all remaining positions at the end
        final_price = data.iloc[-1]['close']
        final_timestamp = data.index[-1]
        self._close_all_positions(final_price, final_timestamp)
        
        # Calculate and return results
        if verbose:
            print("Calculating results...")
        self.results = self._calculate_results(data, symbol)
        
        return self.results
    
    def run_walk_forward_analysis(self,
                                 df: pd.DataFrame,
                                 symbol: str = "BTC",
                                 in_sample_months: int = 12,
                                 out_sample_months: int = 3,
                                 step_months: int = 3,
                                 verbose: bool = True) -> Dict[str, Any]:
        """
        Run walk-forward analysis for robust validation.
        
        Args:
            df: Historical data
            symbol: Trading symbol
            in_sample_months: Months of data for optimization
            out_sample_months: Months of data for testing
            step_months: Step size in months
            verbose: Print progress
            
        Returns:
            Walk-forward analysis results
        """
        if verbose:
            print("Starting Walk-Forward Analysis...")
        
        data = self.prepare_data(df)
        results = []
        
        # Calculate walk-forward windows
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_start = start_date
        
        while current_start < end_date - timedelta(days=in_sample_months * 30):
            # Define periods
            in_sample_end = current_start + timedelta(days=in_sample_months * 30)
            out_sample_start = in_sample_end
            out_sample_end = out_sample_start + timedelta(days=out_sample_months * 30)
            
            if out_sample_end > end_date:
                break
            
            if verbose:
                print(f"Testing period: {out_sample_start.strftime('%Y-%m-%d')} to {out_sample_end.strftime('%Y-%m-%d')}")
            
            # Run backtest on out-of-sample period
            try:
                result = self.run_backtest(
                    data,
                    symbol=symbol,
                    start_date=out_sample_start,
                    end_date=out_sample_end,
                    verbose=False
                )
                
                results.append({
                    'period_start': out_sample_start,
                    'period_end': out_sample_end,
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'total_trades': result.total_trades,
                    'win_rate': result.win_rate
                })
                
            except Exception as e:
                if verbose:
                    print(f"Error in period {out_sample_start} to {out_sample_end}: {e}")
                continue
            
            # Step forward
            current_start += timedelta(days=step_months * 30)
        
        # Aggregate walk-forward results
        if results:
            results_df = pd.DataFrame(results)
            summary = {
                'periods_tested': len(results),
                'avg_return': results_df['total_return'].mean(),
                'avg_sharpe': results_df['sharpe_ratio'].mean(),
                'avg_max_dd': results_df['max_drawdown'].mean(),
                'consistency': len(results_df[results_df['total_return'] > 0]) / len(results_df),
                'best_period': results_df.loc[results_df['total_return'].idxmax()],
                'worst_period': results_df.loc[results_df['total_return'].idxmin()],
                'detailed_results': results_df
            }
        else:
            summary = {'periods_tested': 0, 'error': 'No valid periods found'}
        
        return summary
    
    def _reset_components(self):
        """Reset all components for new backtest."""
        self.regime_detector = RegimeDetector()
        self.risk_manager = RiskManager(
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade
        )
        self.equity_curve = []
        self.trade_log = []
        self.regime_log = []
    
    def _update_positions(self, current_row: pd.Series, timestamp: pd.Timestamp):
        """Update all open positions with current market data."""
        for position_id in list(self.risk_manager.positions.keys()):
            position = self.risk_manager.positions[position_id]
            
            # Update position with current data
            self.risk_manager.update_position(
                position_id,
                current_row['close'],
                current_row['high'],
                current_row['low']
            )
            
            # Update trailing stops for trend positions
            if position.original_regime == MarketRegime.TRENDING:
                side = TrendSignalType.BUY if position.side == PositionSide.LONG else TrendSignalType.SELL
                new_trailing_stop = self.trend_strategy.update_trailing_stop(
                    position_id,
                    current_row['high'],
                    current_row['low'],
                    current_row['atr'],
                    side
                )
                
                # Update position with new trailing stop
                self.risk_manager.update_position(
                    position_id,
                    current_row['close'],
                    current_row['high'],
                    current_row['low'],
                    new_trailing_stop=new_trailing_stop
                )
    
    def _check_exits(self, current_row: pd.Series, timestamp: pd.Timestamp):
        """Check stop loss and take profit conditions."""
        # Apply transaction costs and slippage
        adjusted_price = current_row['close'] * (1 + self.slippage)
        
        for position_id in list(self.risk_manager.positions.keys()):
            # Check stop loss
            if self.risk_manager.check_stop_loss(position_id, adjusted_price, timestamp):
                # Apply transaction cost
                self._apply_transaction_cost(position_id)
                continue
            
            # Check take profit
            if self.risk_manager.check_take_profit(position_id, adjusted_price, timestamp):
                # Apply transaction cost
                self._apply_transaction_cost(position_id)
    
    def _process_signals(self, data: pd.DataFrame, idx: int, current_row: pd.Series, 
                        timestamp: pd.Timestamp, symbol: str):
        """Process trading signals based on current regime."""
        regime = self.regime_detector.current_regime.regime
        
        if regime == MarketRegime.CONSOLIDATION:
            self._process_consolidation_signals(data, idx, current_row, timestamp, symbol)
        elif regime == MarketRegime.TRENDING:
            self._process_trend_signals(data, idx, current_row, timestamp, symbol)
    
    def _process_consolidation_signals(self, data: pd.DataFrame, idx: int, 
                                     current_row: pd.Series, timestamp: pd.Timestamp, symbol: str):
        """Process consolidation strategy signals."""
        signal = self.consolidation_strategy.generate_signal(data, idx)
        if signal is None:
            return
        
        # Convert signal to position parameters
        if signal.signal_type == ConsSignalType.BUY:
            side = PositionSide.LONG
        elif signal.signal_type == ConsSignalType.SELL:
            side = PositionSide.SHORT
        else:
            return
        
        # Apply slippage to entry price
        entry_price = signal.entry_price * (1 + self.slippage if side == PositionSide.LONG else 1 - self.slippage)
        
        # Open position
        position = self.risk_manager.open_position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            regime=MarketRegime.CONSOLIDATION,
            atr=current_row['atr'],
            timestamp=timestamp
        )
        
        if position:
            # Apply transaction cost
            cost = position.quantity * position.entry_price * self.transaction_cost
            self.risk_manager.current_capital -= cost
    
    def _process_trend_signals(self, data: pd.DataFrame, idx: int, 
                             current_row: pd.Series, timestamp: pd.Timestamp, symbol: str):
        """Process trend strategy signals."""
        signal = self.trend_strategy.generate_signal(data, idx)
        if signal is None:
            return
        
        # Convert signal to position parameters
        if signal.signal_type == TrendSignalType.BUY:
            side = PositionSide.LONG
        elif signal.signal_type == TrendSignalType.SELL:
            side = PositionSide.SHORT
        else:
            return
        
        # Apply slippage to entry price
        entry_price = signal.entry_price * (1 + self.slippage if side == PositionSide.LONG else 1 - self.slippage)
        
        # Open position (no take profit for trend strategy)
        position = self.risk_manager.open_position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=signal.initial_stop_loss,
            take_profit=None,  # Trend strategy uses only trailing stops
            regime=MarketRegime.TRENDING,
            atr=current_row['atr'],
            timestamp=timestamp
        )
        
        if position:
            # Apply transaction cost
            cost = position.quantity * position.entry_price * self.transaction_cost
            self.risk_manager.current_capital -= cost
    
    def _handle_regime_changes(self, data: pd.DataFrame, idx: int):
        """Handle positions during regime changes."""
        current_regime = self.regime_detector.current_regime.regime
        
        for position_id, position in list(self.risk_manager.positions.items()):
            if position.original_regime != current_regime:
                # Regime has changed - adapt position management
                if position.original_regime == MarketRegime.CONSOLIDATION and current_regime == MarketRegime.TRENDING:
                    # Convert to trailing stop
                    self._convert_to_trailing_stop(position, data.iloc[idx])
                elif position.original_regime == MarketRegime.TRENDING and current_regime == MarketRegime.CONSOLIDATION:
                    # Lock trailing stop and set take profit
                    self._lock_trailing_stop(position, data.iloc[idx])
    
    def _convert_to_trailing_stop(self, position: Position, current_row: pd.Series):
        """Convert consolidation position to trailing stop (trend style)."""
        # Remove take profit and implement trailing stop
        position.take_profit = None
        
        # Initialize trailing stop based on current position
        side = TrendSignalType.BUY if position.side == PositionSide.LONG else TrendSignalType.SELL
        trailing_stop = self.trend_strategy.calculate_chandelier_exit(
            current_row['high'] if position.side == PositionSide.LONG else 0,
            current_row['low'] if position.side == PositionSide.SHORT else 0,
            current_row['atr'],
            side
        )
        
        self.risk_manager.update_position(
            position.id,
            current_row['close'],
            current_row['high'],
            current_row['low'],
            new_trailing_stop=trailing_stop
        )
    
    def _lock_trailing_stop(self, position: Position, current_row: pd.Series):
        """Lock trailing stop and set take profit."""
        # Keep current trailing stop as fixed stop
        if position.trailing_stop:
            position.stop_loss = position.trailing_stop
            position.trailing_stop = None
        
        # Set conservative take profit
        if position.side == PositionSide.LONG:
            current_profit = current_row['close'] - position.entry_price
            if current_profit > 0:
                take_profit = current_row['close'] + (current_profit * 0.5)  # Take 50% more
                position.take_profit = take_profit
        else:  # SHORT
            current_profit = position.entry_price - current_row['close']
            if current_profit > 0:
                take_profit = current_row['close'] - (current_profit * 0.5)
                position.take_profit = take_profit
    
    def _check_reentry_opportunities(self, data: pd.DataFrame, idx: int, 
                                   current_row: pd.Series, timestamp: pd.Timestamp, symbol: str):
        """Check for re-entry opportunities."""
        candidates = self.risk_manager.check_reentry_opportunity(
            symbol, self.regime_detector.current_regime.regime, 
            current_row['close'], timestamp
        )
        
        for candidate in candidates:
            # Check if we have a new valid signal in the same direction
            if candidate.original_regime == MarketRegime.CONSOLIDATION:
                signal = self.consolidation_strategy.generate_signal(data, idx)
                if (signal and 
                    ((candidate.side == PositionSide.LONG and signal.signal_type == ConsSignalType.BUY) or
                     (candidate.side == PositionSide.SHORT and signal.signal_type == ConsSignalType.SELL))):
                    # Re-enter position
                    self._process_consolidation_signals(data, idx, current_row, timestamp, symbol)
                    self.risk_manager.use_reentry_attempt(candidate.original_position_id)
            
            elif candidate.original_regime == MarketRegime.TRENDING:
                signal = self.trend_strategy.generate_signal(data, idx)
                if (signal and 
                    ((candidate.side == PositionSide.LONG and signal.signal_type == TrendSignalType.BUY) or
                     (candidate.side == PositionSide.SHORT and signal.signal_type == TrendSignalType.SELL))):
                    # Re-enter position
                    self._process_trend_signals(data, idx, current_row, timestamp, symbol)
                    self.risk_manager.use_reentry_attempt(candidate.original_position_id)
    
    def _apply_transaction_cost(self, position_id: str):
        """Apply transaction costs to closed position."""
        if position_id in [p.id for p in self.risk_manager.closed_positions]:
            position = next(p for p in self.risk_manager.closed_positions if p.id == position_id)
            cost = position.quantity * position.exit_price * self.transaction_cost
            self.risk_manager.current_capital -= cost
    
    def _close_all_positions(self, final_price: float, timestamp: pd.Timestamp):
        """Close all remaining positions at backtest end."""
        for position_id in list(self.risk_manager.positions.keys()):
            self.risk_manager._close_position(position_id, final_price, timestamp, 
                                            self.risk_manager.PositionStatus.CLOSED)
            self._apply_transaction_cost(position_id)
    
    def _record_equity_point(self, timestamp: pd.Timestamp):
        """Record equity curve point."""
        # Calculate unrealized P&L
        unrealized_pnl = 0
        for position in self.risk_manager.positions.values():
            if position.side == PositionSide.LONG:
                unrealized_pnl += (position.exit_price or 0 - position.entry_price) * position.quantity
            else:
                unrealized_pnl += (position.entry_price - (position.exit_price or 0)) * position.quantity
        
        total_equity = self.risk_manager.current_capital + unrealized_pnl
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': self.risk_manager.current_capital,
            'unrealized_pnl': unrealized_pnl,
            'active_positions': len(self.risk_manager.positions),
            'regime': self.regime_detector.current_regime.regime.value
        })
    
    def _calculate_results(self, data: pd.DataFrame, symbol: str) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        # Convert data to DataFrames
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        regime_df = pd.DataFrame(self.regime_log)
        regime_df.set_index('timestamp', inplace=True)
        
        # Get trade history
        trade_df = pd.DataFrame(self.risk_manager.trade_history)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cum_returns'] = (1 + equity_df['returns']).cumprod() - 1
        
        # Performance metrics
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital) - 1
        
        # Risk metrics
        returns = equity_df['returns'].dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4) if returns.std() > 0 else 0  # Annualized for 15min bars
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252 * 24 * 4) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Maximum drawdown
        peak = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = (total_return * 100) / abs(max_drawdown * 100) if max_drawdown != 0 else 0
        
        # Trading metrics
        if len(trade_df) > 0:
            win_trades = trade_df[trade_df['pnl'] > 0]
            loss_trades = trade_df[trade_df['pnl'] < 0]
            
            win_rate = len(win_trades) / len(trade_df)
            avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
            avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
            profit_factor = abs(win_trades['pnl'].sum() / loss_trades['pnl'].sum()) if loss_trades['pnl'].sum() != 0 else float('inf')
            
            # Regime-specific metrics
            consolidation_trades = len(trade_df[trade_df['regime'] == 'consolidation'])
            trend_trades = len(trade_df[trade_df['regime'] == 'trending'])
            regime_change_trades = len(trade_df) - consolidation_trades - trend_trades
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            consolidation_trades = trend_trades = regime_change_trades = 0
        
        # Additional metrics
        volatility = returns.std() * np.sqrt(252 * 24 * 4)  # Annualized
        var_95 = returns.quantile(0.05)
        
        # Consecutive losses
        if len(trade_df) > 0:
            trade_df['is_loss'] = trade_df['pnl'] < 0
            consecutive_losses = 0
            max_consecutive_losses = 0
            for is_loss in trade_df['is_loss']:
                if is_loss:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
        else:
            max_consecutive_losses = 0
        
        # Monthly returns
        equity_monthly = equity_df['equity'].resample('M').last()
        monthly_returns = equity_monthly.pct_change().dropna()
        
        return BacktestResults(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=len(trade_df),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            consolidation_trades=consolidation_trades,
            trend_trades=trend_trades,
            regime_change_trades=regime_change_trades,
            volatility=volatility,
            var_95=var_95,
            max_consecutive_losses=max_consecutive_losses,
            equity_curve=equity_df,
            trade_log=trade_df,
            regime_log=regime_df,
            monthly_returns=monthly_returns.to_frame('monthly_return'),
            start_date=data.index[0],
            end_date=data.index[-1],
            duration_days=(data.index[-1] - data.index[0]).days
        )
    
    def generate_report(self, results: BacktestResults, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive backtesting report.
        
        Args:
            results: BacktestResults object
            save_path: Optional path to save the report
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("ADAPTIVE TRADING BOT - BACKTEST RESULTS")
        report.append("=" * 80)
        report.append("")
        
        # Period information
        report.append(f"Backtest Period: {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')}")
        report.append(f"Duration: {results.duration_days} days")
        report.append(f"Initial Capital: ${self.initial_capital:,.2f}")
        report.append(f"Final Capital: ${self.initial_capital * (1 + results.total_return):,.2f}")
        report.append("")
        
        # Performance Metrics
        report.append("PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append(f"Total Return: {results.total_return:.2%}")
        report.append(f"Sharpe Ratio: {results.sharpe_ratio:.3f}")
        report.append(f"Sortino Ratio: {results.sortino_ratio:.3f}")
        report.append(f"Calmar Ratio: {results.calmar_ratio:.3f}")
        report.append(f"Maximum Drawdown: {results.max_drawdown:.2%}")
        report.append(f"Volatility (Annualized): {results.volatility:.2%}")
        report.append(f"Value at Risk (95%): {results.var_95:.2%}")
        report.append("")
        
        # Trading Metrics
        report.append("TRADING METRICS")
        report.append("-" * 40)
        report.append(f"Total Trades: {results.total_trades}")
        report.append(f"Win Rate: {results.win_rate:.2%}")
        report.append(f"Average Win: ${results.avg_win:.2f}")
        report.append(f"Average Loss: ${results.avg_loss:.2f}")
        report.append(f"Profit Factor: {results.profit_factor:.2f}")
        report.append(f"Max Consecutive Losses: {results.max_consecutive_losses}")
        report.append("")
        
        # Regime-Specific Analysis
        report.append("REGIME-SPECIFIC ANALYSIS")
        report.append("-" * 40)
        report.append(f"Consolidation Trades: {results.consolidation_trades}")
        report.append(f"Trend Trades: {results.trend_trades}")
        report.append(f"Regime Change Trades: {results.regime_change_trades}")
        report.append("")
        
        # Monthly Performance
        if len(results.monthly_returns) > 0:
            monthly_stats = results.monthly_returns['monthly_return'].describe()
            report.append("MONTHLY PERFORMANCE")
            report.append("-" * 40)
            report.append(f"Average Monthly Return: {monthly_stats['mean']:.2%}")
            report.append(f"Best Month: {monthly_stats['max']:.2%}")
            report.append(f"Worst Month: {monthly_stats['min']:.2%}")
            report.append(f"Monthly Volatility: {monthly_stats['std']:.2%}")
            report.append("")
        
        # Risk Assessment
        report.append("RISK ASSESSMENT")
        report.append("-" * 40)
        if results.total_return > 0:
            risk_score = "LOW" if results.max_drawdown > -0.10 else "MODERATE" if results.max_drawdown > -0.20 else "HIGH"
        else:
            risk_score = "HIGH"
        report.append(f"Risk Level: {risk_score}")
        report.append(f"Risk-Adjusted Return: {results.total_return / abs(results.max_drawdown) if results.max_drawdown != 0 else 'N/A'}")
        report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text