"""
Core Backtesting Engine

Contains the core backtesting functionality extracted from VectorbtAdaptiveEngine.
Handles data preparation, signal generation, and basic backtesting operations.
"""

import logging
import vectorbt as vbt
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Import bot components
from core.regime_detector import RegimeDetector, MarketRegime
from strategies.consolidation_strategy import ConsolidationStrategy, SignalType as ConsSignalType
from strategies.trend_strategy import TrendStrategy, TrendSignalType
from core.risk_manager import RiskManager
from indicators.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


def _configure_vectorbt_logging():
    """Configure vectorbt logging to reduce noise"""
    class VectorbtCacheFilter(logging.Filter):
        def filter(self, record):
            # Filter out cache and warning messages
            message = record.getMessage() if hasattr(record, 'getMessage') else str(record.msg)
            if any(keyword in message.lower() for keyword in ['cache', 'warning', 'deprecated']):
                return False
            return True

    # Apply filter to vectorbt loggers
    for logger_name in ['vectorbt', 'vectorbt.base', 'vectorbt.portfolio']:
        vbt_logger = logging.getLogger(logger_name)
        vbt_logger.addFilter(VectorbtCacheFilter())
        vbt_logger.setLevel(logging.ERROR)


@dataclass
class VectorbtBacktestConfig:
    """Configuration for vectorbt backtesting"""
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.02
    commission: float = 0.001
    enable_short: bool = False
    max_positions: int = 1


@dataclass
class VectorbtResults:
    """Backtesting results container"""
    portfolio: Optional[vbt.Portfolio] = None
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate extended metrics after initialization"""
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


class CoreBacktestEngine:
    """
    Core backtesting engine using vectorbt for high-performance calculations
    """
    
    def __init__(self, config: VectorbtBacktestConfig = None):
        """Initialize core backtesting engine"""
        self.config = config or VectorbtBacktestConfig()
        
        # Configure vectorbt logging
        _configure_vectorbt_logging()
        
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
            signal = self.consolidation_strategy.generate_signal(data.iloc[:idx+1])
            return signal.signal_type
        except:
            return ConsSignalType.HOLD

    def calculate_position_sizes(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate position sizes based on risk management rules
        
        Args:
            data: Price data DataFrame
            signals: Buy/sell signals
            
        Returns:
            Position sizes as Series
        """
        position_sizes = pd.Series(0.0, index=data.index)
        
        for i in range(len(signals)):
            if signals.iloc[i]:  # If there's a signal
                current_price = data['close'].iloc[i]
                
                # Calculate position size using risk management
                try:
                    # Simple fixed percentage for now
                    risk_amount = self.config.initial_capital * self.config.risk_per_trade
                    position_size = risk_amount / current_price
                    position_sizes.iloc[i] = position_size
                except:
                    position_sizes.iloc[i] = 0.0
        
        return position_sizes

    def run_backtest(
        self, 
        data: pd.DataFrame, 
        symbol: str = "BTC/USDT", 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None
    ) -> VectorbtResults:
        """
        Run comprehensive backtest using vectorbt
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            start_date: Start date for backtesting
            end_date: End date for backtesting
            
        Returns:
            VectorbtResults object with performance metrics
        """
        logger.info(f"Running vectorbt backtest for {symbol}")
        
        try:
            # Filter data by date range if specified
            if start_date or end_date:
                mask = pd.Series(True, index=data.index)
                if start_date:
                    mask &= (data.index >= start_date)
                if end_date:
                    mask &= (data.index <= end_date)
                data = data[mask]
            
            if len(data) < 100:
                raise ValueError("Insufficient data for backtesting")
            
            # Prepare data with indicators and regime detection
            prepared_data = self.prepare_data(data)
            
            # Generate signals
            buy_signals, sell_signals = self.generate_signals(prepared_data)
            
            # Calculate position sizes
            position_sizes = self.calculate_position_sizes(prepared_data, buy_signals)
            
            # Create vectorbt portfolio
            portfolio = vbt.Portfolio.from_signals(
                close=prepared_data['close'],
                entries=buy_signals,
                exits=sell_signals,
                size=position_sizes,
                init_cash=self.config.initial_capital,
                fees=self.config.commission,
                freq='1D'  # Adjust based on your data frequency
            )
            
            # Calculate regime-specific performance
            regime_performance = self._calculate_regime_performance(prepared_data, portfolio)
            
            # Extract key metrics
            stats = portfolio.stats()
            
            results = VectorbtResults(
                portfolio=portfolio,
                total_return=float(stats.get('Total Return [%]', 0)),
                sharpe_ratio=float(stats.get('Sharpe Ratio', 0)),
                max_drawdown=float(stats.get('Max Drawdown [%]', 0)),
                win_rate=float(stats.get('Win Rate [%]', 0)),
                total_trades=int(stats.get('# Trades', 0)),
                regime_performance=regime_performance
            )
            
            logger.info(f"Backtest completed: {results.total_trades} trades, "
                       f"{results.total_return:.2f}% return, "
                       f"{results.sharpe_ratio:.2f} Sharpe ratio")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            # Return empty results on failure
            return VectorbtResults()

    def _calculate_regime_performance(self, data: pd.DataFrame, portfolio: vbt.Portfolio) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each market regime"""
        try:
            regime_performance = {}
            
            # Get trades from portfolio
            trades = portfolio.trades.records_readable
            
            if len(trades) == 0:
                return regime_performance
            
            # Group trades by regime
            for regime in MarketRegime:
                regime_name = regime.value
                
                # Filter trades that occurred during this regime
                regime_mask = data['regime'] == regime_name
                regime_data = data[regime_mask]
                
                if len(regime_data) == 0:
                    continue
                
                # Calculate regime-specific metrics
                regime_trades = trades[
                    (trades['Entry Timestamp'].isin(regime_data.index)) |
                    (trades['Exit Timestamp'].isin(regime_data.index))
                ]
                
                if len(regime_trades) > 0:
                    total_pnl = regime_trades['PnL'].sum()
                    win_rate = (regime_trades['PnL'] > 0).mean() * 100
                    avg_return = regime_trades['Return [%]'].mean()
                    
                    regime_performance[regime_name] = {
                        'total_pnl': float(total_pnl),
                        'win_rate': float(win_rate),
                        'avg_return': float(avg_return),
                        'trade_count': len(regime_trades)
                    }
            
            return regime_performance
            
        except Exception as e:
            logger.warning(f"Failed to calculate regime performance: {str(e)}")
            return {}