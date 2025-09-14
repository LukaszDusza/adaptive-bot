"""
Position Management Module

Handles position management, exits, and portfolio tracking for backtesting.
Extracted from the original validator.py to improve code organization.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from core.risk_manager import RiskManager, PositionSide, Position

logger = logging.getLogger(__name__)


class BacktestPositionManager:
    """Manages positions during backtesting"""
    
    def __init__(self, risk_manager: RiskManager, transaction_cost: float = 0.001, slippage: float = 0.0005):
        """Initialize position manager"""
        self.risk_manager = risk_manager
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # Portfolio tracking
        self.equity_curve = []
        self.portfolio_value = risk_manager.portfolio_balance
        
    def update_positions(self, current_row: pd.Series, timestamp: pd.Timestamp):
        """Update all open positions with current market data"""
        updated_positions = {}
        
        for position_id, position in self.risk_manager.positions.items():
            if position.status == 'open':
                # Update position with current price
                current_price = current_row['close']
                position.current_price = current_price
                
                # Calculate unrealized P&L
                if position.side == PositionSide.LONG:
                    position.unrealized_pnl = (current_price - position.entry_price) * position.size
                else:  # SHORT
                    position.unrealized_pnl = (position.entry_price - current_price) * position.size
                
                # Update trailing stops if applicable
                if hasattr(position, 'trailing_stop_active') and position.trailing_stop_active:
                    self._update_trailing_stop(position, current_row)
                
                updated_positions[position_id] = position
        
        # Update portfolio balance with unrealized P&L
        unrealized_total = sum(pos.unrealized_pnl for pos in updated_positions.values())
        self.portfolio_value = self.risk_manager.portfolio_balance + unrealized_total

    def check_exits(self, current_row: pd.Series, timestamp: pd.Timestamp) -> List[str]:
        """Check and execute position exits"""
        exits_executed = []
        current_price = current_row['close']
        
        for position_id, position in list(self.risk_manager.positions.items()):
            if position.status != 'open':
                continue
                
            should_exit = False
            exit_reason = ""
            
            # Check stop loss
            if position.side == PositionSide.LONG and current_price <= position.stop_loss:
                should_exit = True
                exit_reason = "Stop Loss"
            elif position.side == PositionSide.SHORT and current_price >= position.stop_loss:
                should_exit = True
                exit_reason = "Stop Loss"
            
            # Check take profit
            if position.take_profit:
                if position.side == PositionSide.LONG and current_price >= position.take_profit:
                    should_exit = True
                    exit_reason = "Take Profit"
                elif position.side == PositionSide.SHORT and current_price <= position.take_profit:
                    should_exit = True
                    exit_reason = "Take Profit"
            
            # Execute exit if needed
            if should_exit:
                self._execute_exit(position, current_price, timestamp, exit_reason)
                exits_executed.append(position_id)
        
        return exits_executed

    def _execute_exit(self, position: Position, exit_price: float, timestamp: pd.Timestamp, reason: str):
        """Execute position exit"""
        try:
            # Apply slippage
            if position.side == PositionSide.LONG:
                exit_price = exit_price * (1 - self.slippage)
            else:
                exit_price = exit_price * (1 + self.slippage)
            
            # Close position through risk manager
            self.risk_manager.close_position(position.id, exit_price, timestamp)
            
            # Apply transaction costs
            self._apply_transaction_cost(position.id)
            
            logger.debug(f"Closed position {position.id}: {reason} at {exit_price}")
            
        except Exception as e:
            logger.error(f"Error closing position {position.id}: {e}")

    def _update_trailing_stop(self, position: Position, current_row: pd.Series):
        """Update trailing stop loss levels"""
        current_price = current_row['close']
        
        if position.side == PositionSide.LONG:
            # For long positions, trail stop up as price rises
            new_stop = current_price * (1 - position.trailing_stop_distance)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                logger.debug(f"Updated trailing stop for {position.id}: {new_stop}")
        else:
            # For short positions, trail stop down as price falls
            new_stop = current_price * (1 + position.trailing_stop_distance)
            if new_stop < position.stop_loss:
                position.stop_loss = new_stop
                logger.debug(f"Updated trailing stop for {position.id}: {new_stop}")

    def convert_to_trailing_stop(self, position: Position, current_row: pd.Series, trailing_distance: float = 0.05):
        """Convert position to trailing stop"""
        position.trailing_stop_active = True
        position.trailing_stop_distance = trailing_distance
        
        current_price = current_row['close']
        
        if position.side == PositionSide.LONG:
            # Set initial trailing stop below current price
            position.stop_loss = current_price * (1 - trailing_distance)
        else:
            # Set initial trailing stop above current price
            position.stop_loss = current_price * (1 + trailing_distance)
        
        logger.debug(f"Converted position {position.id} to trailing stop: {position.stop_loss}")

    def lock_trailing_stop(self, position: Position, current_row: pd.Series, lock_distance: float = 0.02):
        """Lock in profits with tight trailing stop"""
        current_price = current_row['close']
        
        if position.side == PositionSide.LONG:
            # Lock profits for long position
            locked_stop = current_price * (1 - lock_distance)
            if locked_stop > position.stop_loss:
                position.stop_loss = locked_stop
                logger.debug(f"Locked trailing stop for {position.id}: {locked_stop}")
        else:
            # Lock profits for short position
            locked_stop = current_price * (1 + lock_distance)
            if locked_stop < position.stop_loss:
                position.stop_loss = locked_stop
                logger.debug(f"Locked trailing stop for {position.id}: {locked_stop}")

    def close_all_positions(self, final_price: float, timestamp: pd.Timestamp):
        """Close all open positions at market close"""
        for position_id, position in list(self.risk_manager.positions.items()):
            if position.status == 'open':
                self._execute_exit(position, final_price, timestamp, "Market Close")

    def _apply_transaction_cost(self, position_id: str):
        """Apply transaction costs to completed trade"""
        position = self.risk_manager.positions.get(position_id)
        if position and position.status == 'closed':
            transaction_fee = position.realized_pnl * self.transaction_cost
            position.realized_pnl -= transaction_fee
            self.risk_manager.portfolio_balance -= transaction_fee

    def record_equity_point(self, timestamp: pd.Timestamp):
        """Record portfolio value for equity curve"""
        # Calculate total portfolio value including unrealized P&L
        total_unrealized = sum(
            pos.unrealized_pnl for pos in self.risk_manager.positions.values() 
            if pos.status == 'open'
        )
        
        current_equity = self.risk_manager.portfolio_balance + total_unrealized
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'cash': self.risk_manager.portfolio_balance,
            'unrealized_pnl': total_unrealized,
            'open_positions': len([p for p in self.risk_manager.positions.values() if p.status == 'open'])
        })
        
        self.portfolio_value = current_equity

    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get current portfolio metrics"""
        total_trades = len([p for p in self.risk_manager.positions.values() if p.status == 'closed'])
        winning_trades = len([p for p in self.risk_manager.positions.values() 
                            if p.status == 'closed' and p.realized_pnl > 0])
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            avg_win = np.mean([p.realized_pnl for p in self.risk_manager.positions.values() 
                             if p.status == 'closed' and p.realized_pnl > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([p.realized_pnl for p in self.risk_manager.positions.values() 
                              if p.status == 'closed' and p.realized_pnl <= 0]) if (total_trades - winning_trades) > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'portfolio_value': self.portfolio_value,
            'cash_balance': self.risk_manager.portfolio_balance
        }