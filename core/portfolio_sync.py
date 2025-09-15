#!/usr/bin/env python3
"""
Portfolio Synchronization Module

This module implements portfolio synchronization between backtesting results
and live trading as recommended in ANALIZA_BOTA_ADAPTACYJNEGO.md

Features:
- Sync optimal parameters from backtesting to live trading
- Portfolio allocation based on backtest performance
- Risk parameter adjustment based on historical results
- Strategy selection based on backtest outcomes
"""

import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class BacktestResults:
    """Backtest results structure for synchronization"""
    symbol: str
    strategy: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_duration: float
    profit_factor: float
    parameters: Dict[str, Any]
    tested_at: datetime
    timeframe: str
    test_period_days: int

@dataclass 
class PortfolioAllocation:
    """Portfolio allocation structure"""
    symbol: str
    allocation_pct: float
    strategy: str
    risk_per_trade: float
    max_positions: int
    confidence_threshold: float
    parameters: Dict[str, Any]

@dataclass
class SyncConfig:
    """Portfolio synchronization configuration"""
    min_sharpe_ratio: float = 1.0
    min_trades: int = 20
    max_drawdown_limit: float = 0.15
    min_win_rate: float = 0.4
    allocation_method: str = "sharpe_weighted"  # "equal", "sharpe_weighted", "kelly"
    max_single_allocation: float = 0.4
    sync_frequency_hours: int = 24
    results_storage_path: str = "backtest_results/"

class PortfolioSynchronizer:
    """Main portfolio synchronization class"""
    
    def __init__(self, config: SyncConfig = None):
        self.config = config or SyncConfig()
        self.results_storage = Path(self.config.results_storage_path)
        self.results_storage.mkdir(exist_ok=True)
        
        # Cache for backtest results
        self.cached_results: Dict[str, BacktestResults] = {}
        self.last_sync_time: Optional[datetime] = None
        
    def store_backtest_results(self, results: BacktestResults) -> bool:
        """
        Store backtest results for portfolio synchronization
        
        Args:
            results: Backtest results to store
            
        Returns:
            True if stored successfully
        """
        try:
            # Create filename with timestamp
            filename = f"{results.symbol}_{results.strategy}_{results.tested_at.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.results_storage / filename
            
            # Convert to dict and store
            results_dict = asdict(results)
            results_dict['tested_at'] = results.tested_at.isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            # Update cache
            cache_key = f"{results.symbol}_{results.strategy}"
            self.cached_results[cache_key] = results
            
            logger.info(f"Stored backtest results for {results.symbol} {results.strategy}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store backtest results: {e}")
            return False
    
    def load_recent_results(self, days_back: int = 30) -> List[BacktestResults]:
        """
        Load recent backtest results from storage
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            List of recent backtest results
        """
        results = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            for filepath in self.results_storage.glob("*.json"):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Parse datetime
                    tested_at = datetime.fromisoformat(data['tested_at'])
                    
                    # Skip old results
                    if tested_at < cutoff_date:
                        continue
                    
                    # Convert back to BacktestResults
                    data['tested_at'] = tested_at
                    result = BacktestResults(**data)
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to load result from {filepath}: {e}")
                    continue
            
            logger.info(f"Loaded {len(results)} recent backtest results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to load recent results: {e}")
            return []
    
    def filter_quality_results(self, results: List[BacktestResults]) -> List[BacktestResults]:
        """
        Filter backtest results based on quality criteria
        
        Args:
            results: List of backtest results
            
        Returns:
            Filtered list of quality results
        """
        quality_results = []
        
        for result in results:
            # Apply quality filters
            if (result.sharpe_ratio >= self.config.min_sharpe_ratio and
                result.total_trades >= self.config.min_trades and
                abs(result.max_drawdown) <= self.config.max_drawdown_limit and
                result.win_rate >= self.config.min_win_rate and
                result.total_return > 0):
                
                quality_results.append(result)
        
        logger.info(f"Filtered to {len(quality_results)} quality results from {len(results)} total")
        return quality_results
    
    def calculate_portfolio_allocations(self, results: List[BacktestResults]) -> List[PortfolioAllocation]:
        """
        Calculate optimal portfolio allocations based on backtest results
        
        Args:
            results: Quality backtest results
            
        Returns:
            List of portfolio allocations
        """
        if not results:
            logger.warning("No quality results available for allocation calculation")
            return []
        
        allocations = []
        
        if self.config.allocation_method == "equal":
            # Equal allocation
            allocation_pct = min(1.0 / len(results), self.config.max_single_allocation)
            
            for result in results:
                allocation = PortfolioAllocation(
                    symbol=result.symbol,
                    allocation_pct=allocation_pct,
                    strategy=result.strategy,
                    risk_per_trade=0.02,  # Default
                    max_positions=1,
                    confidence_threshold=0.6,
                    parameters=result.parameters
                )
                allocations.append(allocation)
        
        elif self.config.allocation_method == "sharpe_weighted":
            # Sharpe ratio weighted allocation
            sharpe_ratios = [max(result.sharpe_ratio, 0.1) for result in results]
            total_sharpe = sum(sharpe_ratios)
            
            for i, result in enumerate(results):
                raw_allocation = sharpe_ratios[i] / total_sharpe
                capped_allocation = min(raw_allocation, self.config.max_single_allocation)
                
                allocation = PortfolioAllocation(
                    symbol=result.symbol,
                    allocation_pct=capped_allocation,
                    strategy=result.strategy,
                    risk_per_trade=min(0.02 * (result.sharpe_ratio / 2.0), 0.05),  # Scale with Sharpe
                    max_positions=1,
                    confidence_threshold=max(0.6, min(0.8, result.win_rate)),
                    parameters=result.parameters
                )
                allocations.append(allocation)
        
        elif self.config.allocation_method == "kelly":
            # Kelly criterion based allocation
            for result in results:
                # Calculate Kelly fraction
                if result.win_rate > 0 and result.win_rate < 1:
                    avg_win = result.total_return / (result.total_trades * result.win_rate) if result.win_rate > 0 else 0
                    avg_loss = abs(result.max_drawdown) / (result.total_trades * (1 - result.win_rate)) if result.win_rate < 1 else 0.01
                    
                    if avg_loss > 0:
                        kelly_fraction = result.win_rate - (1 - result.win_rate) * (avg_win / avg_loss)
                        kelly_fraction = max(0, min(kelly_fraction, self.config.max_single_allocation))
                    else:
                        kelly_fraction = self.config.max_single_allocation / len(results)
                else:
                    kelly_fraction = self.config.max_single_allocation / len(results)
                
                allocation = PortfolioAllocation(
                    symbol=result.symbol,
                    allocation_pct=kelly_fraction,
                    strategy=result.strategy,
                    risk_per_trade=min(kelly_fraction * 0.5, 0.03),  # Conservative Kelly
                    max_positions=1,
                    confidence_threshold=0.7,
                    parameters=result.parameters
                )
                allocations.append(allocation)
        
        # Normalize allocations to sum to 1.0
        total_allocation = sum(alloc.allocation_pct for alloc in allocations)
        if total_allocation > 0:
            for allocation in allocations:
                allocation.allocation_pct /= total_allocation
        
        logger.info(f"Calculated {len(allocations)} portfolio allocations using {self.config.allocation_method}")
        return allocations
    
    def generate_live_trading_config(self, allocations: List[PortfolioAllocation]) -> Dict[str, Any]:
        """
        Generate live trading configuration based on portfolio allocations
        
        Args:
            allocations: Portfolio allocations
            
        Returns:
            Live trading configuration dictionary
        """
        if not allocations:
            logger.warning("No allocations provided, returning default config")
            return {
                'symbols': ['BTC/USDT'],
                'risk_per_trade': 0.02,
                'max_positions': 1,
                'strategy_parameters': {}
            }
        
        # Extract symbols and their weights
        symbols = [alloc.symbol for alloc in allocations]
        symbol_weights = {alloc.symbol: alloc.allocation_pct for alloc in allocations}
        
        # Calculate aggregate risk parameters
        avg_risk_per_trade = np.average([alloc.risk_per_trade for alloc in allocations], 
                                       weights=[alloc.allocation_pct for alloc in allocations])
        
        total_max_positions = sum(alloc.max_positions for alloc in allocations)
        
        # Collect strategy parameters by symbol
        strategy_parameters = {}
        for alloc in allocations:
            strategy_parameters[alloc.symbol] = {
                'strategy': alloc.strategy,
                'parameters': alloc.parameters,
                'confidence_threshold': alloc.confidence_threshold,
                'allocation_weight': alloc.allocation_pct
            }
        
        config = {
            'symbols': symbols,
            'symbol_weights': symbol_weights,
            'risk_per_trade': avg_risk_per_trade,
            'max_positions': min(total_max_positions, 5),  # Cap at 5
            'strategy_parameters': strategy_parameters,
            'sync_timestamp': datetime.now().isoformat(),
            'allocation_method': self.config.allocation_method
        }
        
        logger.info(f"Generated live trading config for {len(symbols)} symbols")
        return config
    
    def sync_portfolio(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Main portfolio synchronization method
        
        Args:
            force: Force sync even if recently synced
            
        Returns:
            Live trading configuration or None if sync not needed/failed
        """
        # Check if sync is needed
        if not force and self.last_sync_time:
            time_since_sync = datetime.now() - self.last_sync_time
            if time_since_sync.total_seconds() < (self.config.sync_frequency_hours * 3600):
                logger.info("Portfolio sync not needed yet")
                return None
        
        logger.info("Starting portfolio synchronization...")
        
        try:
            # Load recent backtest results
            results = self.load_recent_results()
            
            if not results:
                logger.warning("No recent backtest results available for sync")
                return None
            
            # Filter for quality results
            quality_results = self.filter_quality_results(results)
            
            if not quality_results:
                logger.warning("No quality backtest results available for sync")
                return None
            
            # Calculate portfolio allocations
            allocations = self.calculate_portfolio_allocations(quality_results)
            
            if not allocations:
                logger.warning("Failed to calculate portfolio allocations")
                return None
            
            # Generate live trading configuration
            config = self.generate_live_trading_config(allocations)
            
            # Update sync time
            self.last_sync_time = datetime.now()
            
            logger.info("Portfolio synchronization completed successfully")
            return config
            
        except Exception as e:
            logger.error(f"Portfolio synchronization failed: {e}")
            return None
    
    def get_sync_status(self) -> Dict[str, Any]:
        """
        Get portfolio synchronization status
        
        Returns:
            Sync status information
        """
        status = {
            'last_sync_time': self.last_sync_time.isoformat() if self.last_sync_time else None,
            'cached_results_count': len(self.cached_results),
            'sync_frequency_hours': self.config.sync_frequency_hours,
            'allocation_method': self.config.allocation_method,
            'quality_criteria': {
                'min_sharpe_ratio': self.config.min_sharpe_ratio,
                'min_trades': self.config.min_trades,
                'max_drawdown_limit': self.config.max_drawdown_limit,
                'min_win_rate': self.config.min_win_rate
            }
        }
        
        # Check if sync is due
        if self.last_sync_time:
            time_since_sync = datetime.now() - self.last_sync_time
            sync_due = time_since_sync.total_seconds() >= (self.config.sync_frequency_hours * 3600)
            status['sync_due'] = sync_due
            status['hours_since_last_sync'] = time_since_sync.total_seconds() / 3600
        else:
            status['sync_due'] = True
            status['hours_since_last_sync'] = None
        
        return status


# Integration functions for existing codebase

def integrate_backtest_results(results: Any, symbol: str, strategy: str, timeframe: str) -> bool:
    """
    Integration function to store vectorbt backtest results
    
    Args:
        results: Vectorbt backtest results object
        symbol: Trading symbol
        strategy: Strategy name
        timeframe: Timeframe used
        
    Returns:
        True if integrated successfully
    """
    try:
        # Extract metrics from vectorbt results
        if hasattr(results, 'portfolio') and hasattr(results, 'metrics'):
            total_return = results.portfolio.total_return() if hasattr(results.portfolio, 'total_return') else 0.0
            sharpe_ratio = results.metrics.get('sharpe_ratio', 0.0)
            max_drawdown = results.metrics.get('max_drawdown', 0.0)
            
            # Calculate additional metrics
            trades = getattr(results, 'trades', None)
            if trades is not None and hasattr(trades, 'records'):
                total_trades = len(trades.records)
                if total_trades > 0:
                    win_trades = len([t for t in trades.records if t.pnl > 0])
                    win_rate = win_trades / total_trades
                else:
                    win_rate = 0.0
            else:
                total_trades = 0
                win_rate = 0.0
            
            # Create BacktestResults
            backtest_result = BacktestResults(
                symbol=symbol,
                strategy=strategy,
                total_return=float(total_return),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown=float(max_drawdown),
                win_rate=float(win_rate),
                total_trades=total_trades,
                avg_trade_duration=0.0,  # Could be calculated from trades
                profit_factor=1.0,  # Could be calculated
                parameters={},  # Could be passed from strategy
                tested_at=datetime.now(),
                timeframe=timeframe,
                test_period_days=365  # Default
            )
            
            # Store results
            synchronizer = PortfolioSynchronizer()
            return synchronizer.store_backtest_results(backtest_result)
        
    except Exception as e:
        logger.error(f"Failed to integrate backtest results: {e}")
        
    return False

def get_synced_trading_config() -> Optional[Dict[str, Any]]:
    """
    Get synchronized trading configuration for live trading
    
    Returns:
        Synced trading configuration or None
    """
    try:
        synchronizer = PortfolioSynchronizer()
        return synchronizer.sync_portfolio()
    except Exception as e:
        logger.error(f"Failed to get synced trading config: {e}")
        return None