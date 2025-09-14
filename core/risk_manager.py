"""
Risk Management Module

Implements comprehensive risk management for the adaptive trading bot.
Includes dynamic position sizing, stop loss management, and intelligent re-entry logic.

Key Features:
- Dynamic position sizing based on ATR (constant risk per trade)
- Intelligent re-entry mechanism after stop loss hits
- Position management during regime changes
- Portfolio-level risk controls
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import logging

from core.regime_detector import MarketRegime

# Database integration
try:
    from database.models import (
        get_position_repository, 
        PositionRepository, 
        ExitReason,
        PositionStatus as DbPositionStatus
    )
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logging.warning("Database models not available. Position history will not be persisted.")

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side enumeration"""
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = "open"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"


@dataclass
class Position:
    """Trading position data structure"""
    id: str
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    entry_time: pd.Timestamp
    stop_loss: float
    take_profit: Optional[float]
    original_regime: MarketRegime
    entry_atr: float
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    pnl: Optional[float] = None
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    trailing_stop: Optional[float] = None
    re_entry_attempts: int = 0


@dataclass
class ReEntryCandidate:
    """Re-entry candidate tracking"""
    original_position_id: str
    symbol: str
    side: PositionSide
    stop_out_time: pd.Timestamp
    stop_out_price: float
    original_regime: MarketRegime
    original_stop_distance: float
    attempts_used: int = 0
    max_attempts: int = 1


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    total_risk: float
    available_risk: float
    active_positions: int
    total_exposure: float
    max_position_size: float
    daily_pnl: float
    drawdown: float


class RiskManager:
    """
    Comprehensive risk management system for the adaptive trading bot.
    
    Manages position sizing, stop losses, re-entries, and overall portfolio risk.
    """
    
    def __init__(self,
                 initial_capital: float = 10000,
                 risk_per_trade: float = 0.02,  # 2% of capital
                 max_positions: int = 5,
                 max_daily_loss: float = 0.05,  # 5% daily loss limit
                 max_drawdown: float = 0.15,    # 15% max drawdown
                 re_entry_window_bars: int = 3,
                 max_re_entry_attempts: int = 1,
                 re_entry_threshold_multiplier: float = 1.5,
                 position_correlation_limit: float = 0.7):
        """
        Initialize risk manager.
        
        Args:
            initial_capital: Starting capital
            risk_per_trade: Risk per trade as fraction of capital
            max_positions: Maximum concurrent positions
            max_daily_loss: Maximum daily loss as fraction of capital
            max_drawdown: Maximum drawdown from peak
            re_entry_window_bars: Number of bars to allow re-entry
            max_re_entry_attempts: Maximum re-entry attempts per original signal
            re_entry_threshold_multiplier: Multiplier for re-entry stop distance threshold
            position_correlation_limit: Maximum correlation between positions
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.re_entry_window_bars = re_entry_window_bars
        self.max_re_entry_attempts = max_re_entry_attempts
        self.re_entry_threshold_multiplier = re_entry_threshold_multiplier
        self.position_correlation_limit = position_correlation_limit
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.re_entry_candidates: Dict[str, ReEntryCandidate] = {}
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.last_reset_date = None
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        
        # Database integration
        self.position_repo: Optional[PositionRepository] = None
        if DATABASE_AVAILABLE:
            try:
                self.position_repo = get_position_repository()
                logger.info("Database position tracking enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize database: {e}")
                self.position_repo = None
        
    def calculate_position_size(self, 
                              entry_price: float, 
                              stop_loss: float,
                              symbol: str = "BTC") -> float:
        """
        Calculate optimal position size based on risk management rules.
        
        Args:
            entry_price: Planned entry price
            stop_loss: Stop loss price
            symbol: Trading symbol
            
        Returns:
            Position size (quantity)
        """
        # Calculate risk amount in currency terms
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Calculate stop loss distance
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0.0
        
        # Calculate position size
        position_size = risk_amount / stop_distance
        
        # Apply maximum position size constraints
        max_position_value = self.current_capital * 0.2  # Max 20% of capital per position
        max_size_by_value = max_position_value / entry_price
        
        position_size = min(position_size, max_size_by_value)
        
        return position_size
    
    def can_open_position(self, 
                         symbol: str,
                         regime: MarketRegime,
                         entry_price: float,
                         stop_loss: float) -> Tuple[bool, str]:
        """
        Check if a new position can be opened based on risk constraints.
        
        Args:
            symbol: Trading symbol
            regime: Current market regime
            entry_price: Planned entry price
            stop_loss: Stop loss price
            
        Returns:
            Tuple of (can_open, reason)
        """
        # Check maximum positions limit
        active_positions = len([p for p in self.positions.values() if p.status == PositionStatus.OPEN])
        if active_positions >= self.max_positions:
            return False, f"Maximum positions limit reached ({active_positions}/{self.max_positions})"
        
        # Check daily loss limit
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown >= self.max_drawdown:
            return False, f"Maximum drawdown exceeded ({current_drawdown:.2%})"
        
        if abs(self.daily_pnl) >= self.max_daily_loss * self.current_capital:
            return False, f"Daily loss limit exceeded ({self.daily_pnl:.2f})"
        
        # Check if position size would be meaningful
        position_size = self.calculate_position_size(entry_price, stop_loss, symbol)
        min_position_value = 10  # Minimum $10 position
        if position_size * entry_price < min_position_value:
            return False, "Position size too small to be meaningful"
        
        return True, "Position can be opened"
    
    def open_position(self,
                     symbol: str,
                     side: PositionSide,
                     entry_price: float,
                     stop_loss: float,
                     take_profit: Optional[float],
                     regime: MarketRegime,
                     atr: float,
                     timestamp: pd.Timestamp) -> Optional[Position]:
        """
        Open a new position with proper risk management.
        
        Args:
            symbol: Trading symbol
            side: Position side (LONG/SHORT)
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price (optional)
            regime: Market regime when position opened
            atr: ATR at position opening
            timestamp: Entry timestamp
            
        Returns:
            Position object if successful, None otherwise
        """
        can_open, reason = self.can_open_position(symbol, regime, entry_price, stop_loss)
        if not can_open:
            return None
        
        # Calculate position size
        quantity = self.calculate_position_size(entry_price, stop_loss, symbol)
        
        # Create position
        position = Position(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            original_regime=regime,
            entry_atr=atr
        )
        
        # Add to positions
        self.positions[position.id] = position
        
        # Save to database if available
        if self.position_repo:
            try:
                position_data = {
                    'position_id': position.id,
                    'symbol': position.symbol,
                    'side': position.side.value,
                    'entry_price': entry_price,
                    'entry_time': timestamp,
                    'quantity': quantity,
                    'entry_regime': regime.value,
                    'entry_atr': atr,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'status': DbPositionStatus.OPEN.value
                }
                self.position_repo.create_position(position_data)
                logger.info(f"Position {position.id} saved to database")
            except Exception as e:
                logger.error(f"Failed to save position to database: {e}")
        
        return position
    
    def update_position(self, 
                       position_id: str,
                       current_price: float,
                       current_high: float,
                       current_low: float,
                       new_stop_loss: Optional[float] = None,
                       new_trailing_stop: Optional[float] = None) -> bool:
        """
        Update position with current market data.
        
        Args:
            position_id: Position ID to update
            current_price: Current market price
            current_high: Current high price
            current_low: Current low price
            new_stop_loss: New stop loss level
            new_trailing_stop: New trailing stop level
            
        Returns:
            True if position updated successfully
        """
        if position_id not in self.positions:
            return False
        
        position = self.positions[position_id]
        if position.status != PositionStatus.OPEN:
            return False
        
        # Update stop loss if provided
        if new_stop_loss is not None:
            position.stop_loss = new_stop_loss
        
        # Update trailing stop if provided
        if new_trailing_stop is not None:
            position.trailing_stop = new_trailing_stop
        
        # Calculate current P&L
        if position.side == PositionSide.LONG:
            unrealized_pnl = (current_price - position.entry_price) * position.quantity
            # Update max favorable/adverse excursion
            position.max_favorable_excursion = max(
                position.max_favorable_excursion,
                (current_high - position.entry_price) * position.quantity
            )
            position.max_adverse_excursion = min(
                position.max_adverse_excursion,
                (current_low - position.entry_price) * position.quantity
            )
        else:  # SHORT
            unrealized_pnl = (position.entry_price - current_price) * position.quantity
            # Update max favorable/adverse excursion
            position.max_favorable_excursion = max(
                position.max_favorable_excursion,
                (position.entry_price - current_low) * position.quantity
            )
            position.max_adverse_excursion = min(
                position.max_adverse_excursion,
                (position.entry_price - current_high) * position.quantity
            )
        
        return True
    
    def check_stop_loss(self, 
                       position_id: str,
                       current_price: float,
                       timestamp: pd.Timestamp) -> bool:
        """
        Check if position should be stopped out.
        
        Args:
            position_id: Position ID to check
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            True if position should be closed
        """
        if position_id not in self.positions:
            return False
        
        position = self.positions[position_id]
        if position.status != PositionStatus.OPEN:
            return False
        
        # Check trailing stop first (if exists)
        if position.trailing_stop is not None:
            if position.side == PositionSide.LONG:
                if current_price <= position.trailing_stop:
                    self._close_position(position_id, current_price, timestamp, PositionStatus.STOPPED_OUT)
                    return True
            else:  # SHORT
                if current_price >= position.trailing_stop:
                    self._close_position(position_id, current_price, timestamp, PositionStatus.STOPPED_OUT)
                    return True
        
        # Check regular stop loss
        if position.side == PositionSide.LONG:
            if current_price <= position.stop_loss:
                self._close_position(position_id, current_price, timestamp, PositionStatus.STOPPED_OUT)
                return True
        else:  # SHORT
            if current_price >= position.stop_loss:
                self._close_position(position_id, current_price, timestamp, PositionStatus.STOPPED_OUT)
                return True
        
        return False
    
    def check_take_profit(self, 
                         position_id: str,
                         current_price: float,
                         timestamp: pd.Timestamp) -> bool:
        """
        Check if position should hit take profit.
        
        Args:
            position_id: Position ID to check
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            True if position was closed at take profit
        """
        if position_id not in self.positions:
            return False
        
        position = self.positions[position_id]
        if position.status != PositionStatus.OPEN or position.take_profit is None:
            return False
        
        if position.side == PositionSide.LONG:
            if current_price >= position.take_profit:
                self._close_position(position_id, current_price, timestamp, PositionStatus.CLOSED)
                return True
        else:  # SHORT
            if current_price <= position.take_profit:
                self._close_position(position_id, current_price, timestamp, PositionStatus.CLOSED)
                return True
        
        return False
    
    def _close_position(self, 
                       position_id: str,
                       exit_price: float,
                       timestamp: pd.Timestamp,
                       status: PositionStatus):
        """
        Close a position and update capital.
        
        Args:
            position_id: Position ID to close
            exit_price: Exit price
            timestamp: Exit timestamp
            status: Exit status
        """
        if position_id not in self.positions:
            return
        
        position = self.positions[position_id]
        
        # Calculate P&L
        if position.side == PositionSide.LONG:
            pnl = (exit_price - position.entry_price) * position.quantity
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.quantity
        
        # Update position
        position.exit_price = exit_price
        position.exit_time = timestamp
        position.pnl = pnl
        position.status = status
        
        # Update capital and tracking
        self.current_capital += pnl
        self.daily_pnl += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        # Add to trade history
        self.trade_history.append({
            'id': position.id,
            'symbol': position.symbol,
            'side': position.side.value,
            'entry_time': position.entry_time,
            'exit_time': timestamp,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'regime': position.original_regime.value,
            'status': status.value
        })
        
        # Calculate position duration
        duration_minutes = None
        if position.entry_time and timestamp:
            duration_minutes = int((timestamp - position.entry_time).total_seconds() / 60)
        
        # Map status to database exit reason
        exit_reason_map = {
            PositionStatus.CLOSED: ExitReason.TAKE_PROFIT if pnl > 0 else ExitReason.MANUAL,
            PositionStatus.STOPPED_OUT: ExitReason.STOP_LOSS
        }
        exit_reason = exit_reason_map.get(status, ExitReason.MANUAL)
        
        # Update database if available
        if self.position_repo:
            try:
                pnl_percentage = (pnl / (position.entry_price * position.quantity)) * 100 if position.quantity > 0 else 0
                
                updates = {
                    'exit_price': exit_price,
                    'exit_time': timestamp,
                    'exit_reason': exit_reason.value,
                    'pnl': pnl,
                    'pnl_percentage': pnl_percentage,
                    'duration_minutes': duration_minutes,
                    'status': DbPositionStatus.CLOSED.value if status == PositionStatus.CLOSED else DbPositionStatus.STOPPED_OUT.value,
                    'max_favorable_excursion': position.max_favorable_excursion,
                    'max_adverse_excursion': position.max_adverse_excursion
                }
                self.position_repo.update_position(position.id, updates)
                logger.info(f"Position {position.id} closed in database: {exit_reason.value} P&L: {pnl:.2f}")
            except Exception as e:
                logger.error(f"Failed to update position in database: {e}")
        
        # If stopped out, create re-entry candidate
        if status == PositionStatus.STOPPED_OUT:
            self._create_reentry_candidate(position)
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[position_id]
    
    def close_position(self, position_id: str, exit_price: float, timestamp: pd.Timestamp = None) -> bool:
        """
        Public method to close a position (for testing and manual closure).
        
        Args:
            position_id: Position ID to close
            exit_price: Exit price
            timestamp: Exit timestamp (optional, defaults to now)
            
        Returns:
            True if position was closed successfully
        """
        if position_id not in self.positions:
            return False
        
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        self._close_position(position_id, exit_price, timestamp, PositionStatus.CLOSED)
        return True
    
    def _create_reentry_candidate(self, position: Position):
        """
        Create a re-entry candidate from a stopped out position.
        
        Args:
            position: Stopped out position
        """
        if position.re_entry_attempts >= self.max_re_entry_attempts:
            return
        
        stop_distance = abs(position.entry_price - position.stop_loss)
        
        candidate = ReEntryCandidate(
            original_position_id=position.id,
            symbol=position.symbol,
            side=position.side,
            stop_out_time=position.exit_time,
            stop_out_price=position.exit_price,
            original_regime=position.original_regime,
            original_stop_distance=stop_distance
        )
        
        self.re_entry_candidates[position.id] = candidate
    
    def check_reentry_opportunity(self,
                                symbol: str,
                                current_regime: MarketRegime,
                                current_price: float,
                                timestamp: pd.Timestamp) -> List[ReEntryCandidate]:
        """
        Check for valid re-entry opportunities.
        
        Args:
            symbol: Trading symbol
            current_regime: Current market regime
            current_price: Current price
            timestamp: Current timestamp
            
        Returns:
            List of valid re-entry candidates
        """
        valid_candidates = []
        expired_candidates = []
        
        for candidate_id, candidate in self.re_entry_candidates.items():
            # Check if candidate is for this symbol
            if candidate.symbol != symbol:
                continue
            
            # Check time window
            bars_elapsed = (timestamp - candidate.stop_out_time).total_seconds() / (15 * 60)  # Assuming 15min bars
            if bars_elapsed > self.re_entry_window_bars:
                expired_candidates.append(candidate_id)
                continue
            
            # Check if regime is still the same
            if current_regime != candidate.original_regime:
                expired_candidates.append(candidate_id)
                continue
            
            # Check price movement threshold (avoid if price moved too far)
            price_move = abs(current_price - candidate.stop_out_price)
            threshold = self.re_entry_threshold_multiplier * candidate.original_stop_distance
            if price_move > threshold:
                expired_candidates.append(candidate_id)
                continue
            
            # Check attempts limit
            if candidate.attempts_used >= candidate.max_attempts:
                expired_candidates.append(candidate_id)
                continue
            
            valid_candidates.append(candidate)
        
        # Clean up expired candidates
        for candidate_id in expired_candidates:
            del self.re_entry_candidates[candidate_id]
        
        return valid_candidates
    
    def use_reentry_attempt(self, candidate_id: str) -> bool:
        """
        Mark a re-entry attempt as used.
        
        Args:
            candidate_id: Re-entry candidate ID
            
        Returns:
            True if attempt was recorded successfully
        """
        if candidate_id not in self.re_entry_candidates:
            return False
        
        candidate = self.re_entry_candidates[candidate_id]
        candidate.attempts_used += 1
        
        # Remove if max attempts reached
        if candidate.attempts_used >= candidate.max_attempts:
            del self.re_entry_candidates[candidate_id]
        
        return True
    
    def reset_daily_tracking(self, current_date: datetime):
        """
        Reset daily tracking metrics.
        
        Args:
            current_date: Current date
        """
        if self.last_reset_date is None or current_date.date() != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date.date()
    
    def get_risk_metrics(self) -> RiskMetrics:
        """
        Get current risk metrics.
        
        Returns:
            RiskMetrics object with current risk information
        """
        active_positions = [p for p in self.positions.values() if p.status == PositionStatus.OPEN]
        
        total_exposure = sum(p.quantity * p.entry_price for p in active_positions)
        max_single_risk = self.current_capital * self.risk_per_trade
        
        return RiskMetrics(
            total_risk=len(active_positions) * max_single_risk,
            available_risk=self.current_capital * self.risk_per_trade,
            active_positions=len(active_positions),
            total_exposure=total_exposure,
            max_position_size=max_single_risk,
            daily_pnl=self.daily_pnl,
            drawdown=self.current_drawdown
        )
    
    def get_performance_summary(self) -> Dict:
        """
        Get performance summary statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trade_history:
            return {"total_trades": 0}
        
        trades = pd.DataFrame(self.trade_history)
        
        total_pnl = trades['pnl'].sum()
        win_trades = trades[trades['pnl'] > 0]
        loss_trades = trades[trades['pnl'] < 0]
        
        win_rate = len(win_trades) / len(trades) if len(trades) > 0 else 0
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
        
        return {
            "total_trades": len(trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            "current_capital": self.current_capital,
            "total_return": (self.current_capital - self.initial_capital) / self.initial_capital,
            "max_drawdown": self.current_drawdown,
            "active_positions": len(self.positions)
        }