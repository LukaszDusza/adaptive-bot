"""
Consolidation Strategy Module

Implements mean-reversion strategy for consolidating markets using Stochastic Oscillator.
Active when ADX < 25 and volatility is in normal range.

Strategy Logic:
- Buy signals: Stochastic in oversold zone with bullish crossover near support
- Sell signals: Stochastic in overbought zone with bearish crossover near resistance
- Fixed take profit targets at opposite range boundaries
- ATR-based stop losses
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from indicators.technical import TechnicalIndicators, StochasticSignal


class SignalType(Enum):
    """Signal types for consolidation strategy"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class ConsolidationSignal:
    """Consolidation strategy signal"""
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    volume_confirmed: bool = False


class ConsolidationStrategy:
    """
    Mean-reversion strategy for consolidating markets.
    
    Uses Stochastic Oscillator to identify oversold/overbought conditions
    and generates signals when price is near support/resistance levels.
    """
    
    @classmethod
    def get_strategy_info(cls) -> Dict[str, Any]:
        """
        Get comprehensive strategy information for GUI display.
        
        Returns:
            Dictionary with strategy description, logic, and parameters
        """
        return {
            'name': 'Consolidation (Mean Reversion) Strategy',
            'short_description': 'Stochastic-based mean reversion with support/resistance targeting',
            'detailed_description': """
            **Consolidation Strategy** is designed for sideways, ranging markets where price oscillates between support and resistance levels.
            
            **Core Logic:**
            • Uses Stochastic Oscillator to identify oversold/overbought conditions
            • Generates buy signals when Stochastic is oversold (<20) with bullish crossover near support
            • Generates sell signals when Stochastic is overbought (>80) with bearish crossover near resistance
            • Uses fixed take profit targets at opposite range boundaries
            • ATR-based stop losses for risk management
            • Optional volume confirmation to validate breakout attempts
            
            **When Active:**
            • Market regime: CONSOLIDATION (ADX < 25)
            • Normal volatility conditions (not stagnant or panic)
            • Price trading within established support/resistance range
            
            **Risk Management:**
            • Stop loss: Entry price ± (ATR × stop multiplier)
            • Take profit: Target opposite support/resistance level
            • Quick profits in ranging markets - doesn't let positions run indefinitely
            """,
            'entry_criteria': [
                'Stochastic in oversold (<20) or overbought (>80) zone',
                'Stochastic %K crosses above %D (bullish) or below %D (bearish)',
                'Price near support level (for longs) or resistance level (for shorts)',
                'Distance to support/resistance < max_distance_to_level × ATR',
                'Volume confirmation (optional - above average volume)'
            ],
            'exit_criteria': [
                'Take profit hit (at opposite support/resistance level)',
                'Stop loss hit (ATR-based protective stop)',
                'Opposite Stochastic signal (momentum reversal)',
                'Market regime changes (e.g., to TRENDING or PANIC)',
                'Support/resistance level broken (range breakdown)'
            ],
            'market_conditions': {
                'best_for': 'Sideways, ranging markets with clear support/resistance levels',
                'worst_for': 'Strong trending markets or highly volatile breakout conditions',
                'regime': 'CONSOLIDATION (ADX < 25)'
            },
            'advantages': [
                'Excels in sideways markets where trend strategies fail',
                'Quick profit targets prevent giving back gains',
                'Support/resistance targeting improves entry/exit timing',
                'Stochastic filters out false signals in choppy conditions'
            ],
            'disadvantages': [
                'Suffers in strong trending markets (fights the trend)',
                'Fixed take profits may cut winners short in breakouts',
                'Relies on support/resistance holding - vulnerable to breakouts',
                'Multiple small losses possible in transition periods'
            ]
        }
    
    @classmethod
    def get_parameter_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed parameter information for GUI sliders and optimization.
        
        Returns:
            Dictionary with parameter details including ranges, descriptions, and impact
        """
        return {
            'stoch_k_period': {
                'name': 'Stochastic %K Period',
                'description': 'Period for calculating Stochastic %K oscillator',
                'type': 'int',
                'default': 5,
                'min_value': 3,
                'max_value': 14,
                'step': 1,
                'impact': 'Lower = more sensitive to price changes, Higher = smoother but slower signals',
                'optimization_range': [5, 8, 10, 14]
            },
            'stoch_k_smooth': {
                'name': 'Stochastic %K Smoothing',
                'description': 'Smoothing periods for Stochastic %K',
                'type': 'int',
                'default': 3,
                'min_value': 1,
                'max_value': 5,
                'step': 1,
                'impact': 'Higher = smoother signals, less noise but slower response',
                'optimization_range': [1, 3, 5]
            },
            'stoch_d_smooth': {
                'name': 'Stochastic %D Smoothing',
                'description': 'Smoothing periods for Stochastic %D signal line',
                'type': 'int',
                'default': 3,
                'min_value': 1,
                'max_value': 5,
                'step': 1,
                'impact': 'Higher = smoother signal line for crossover detection',
                'optimization_range': [1, 3, 5]
            },
            'oversold_level': {
                'name': 'Oversold Threshold',
                'description': 'Stochastic level below which market is considered oversold',
                'type': 'float',
                'default': 20.0,
                'min_value': 10.0,
                'max_value': 30.0,
                'step': 5.0,
                'impact': 'Lower = fewer but stronger oversold signals, Higher = more frequent signals',
                'optimization_range': [15.0, 20.0, 25.0, 30.0]
            },
            'overbought_level': {
                'name': 'Overbought Threshold',
                'description': 'Stochastic level above which market is considered overbought',
                'type': 'float',
                'default': 80.0,
                'min_value': 70.0,
                'max_value': 90.0,
                'step': 5.0,
                'impact': 'Higher = fewer but stronger overbought signals, Lower = more frequent signals',
                'optimization_range': [70.0, 75.0, 80.0, 85.0]
            },
            'atr_stop_multiplier': {
                'name': 'ATR Stop Loss Multiplier',
                'description': 'ATR multiplier for stop loss distance',
                'type': 'float',
                'default': 2.0,
                'min_value': 1.0,
                'max_value': 4.0,
                'step': 0.25,
                'impact': 'Lower = tighter stops (more stopped out), Higher = wider stops (larger losses)',
                'optimization_range': [1.5, 2.0, 2.5, 3.0]
            },
            'support_resistance_window': {
                'name': 'Support/Resistance Window',
                'description': 'Lookback period for identifying support and resistance levels',
                'type': 'int',
                'default': 20,
                'min_value': 10,
                'max_value': 50,
                'step': 5,
                'impact': 'Lower = more recent levels, Higher = more significant historical levels',
                'optimization_range': [15, 20, 25, 30]
            },
            'max_distance_to_level': {
                'name': 'Max Distance to S/R Level',
                'description': 'Maximum distance to support/resistance as ATR multiple',
                'type': 'float',
                'default': 0.25,
                'min_value': 0.1,
                'max_value': 1.0,
                'step': 0.05,
                'impact': 'Lower = must be very close to levels, Higher = more flexible entry timing',
                'optimization_range': [0.2, 0.25, 0.35, 0.5]
            },
            'volume_threshold': {
                'name': 'Volume Confirmation Threshold',
                'description': 'Volume multiplier for confirming signals',
                'type': 'float',
                'default': 1.2,
                'min_value': 1.0,
                'max_value': 2.0,
                'step': 0.1,
                'impact': 'Higher = require stronger volume confirmation for signals',
                'optimization_range': [1.0, 1.2, 1.5, 1.8]
            }
        }
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter values.
        
        Returns:
            Dictionary of current parameter values
        """
        return {
            'stoch_k_period': self.stoch_k_period,
            'stoch_k_smooth': self.stoch_k_smooth,
            'stoch_d_smooth': self.stoch_d_smooth,
            'oversold_level': self.oversold_level,
            'overbought_level': self.overbought_level,
            'atr_stop_multiplier': self.atr_stop_multiplier,
            'support_resistance_window': self.support_resistance_window,
            'max_distance_to_level': self.max_distance_to_level,
            'volume_confirmation': self.volume_confirmation,
            'volume_threshold': self.volume_threshold
        }
    
    def update_parameters(self, new_params: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.
        
        Args:
            new_params: Dictionary of parameter updates
        """
        for param, value in new_params.items():
            if hasattr(self, param):
                setattr(self, param, value)
    
    def __init__(self,
                 stoch_k_period: int = 5,
                 stoch_k_smooth: int = 3,
                 stoch_d_smooth: int = 3,
                 oversold_level: float = 20,
                 overbought_level: float = 80,
                 atr_stop_multiplier: float = 2.0,
                 support_resistance_window: int = 20,
                 max_distance_to_level: float = 0.25,  # As multiple of ATR
                 volume_confirmation: bool = True,
                 volume_threshold: float = 1.2):
        """
        Initialize consolidation strategy.
        
        Args:
            stoch_k_period: Stochastic %K period
            stoch_k_smooth: Stochastic %K smoothing
            stoch_d_smooth: Stochastic %D smoothing
            oversold_level: Oversold threshold
            overbought_level: Overbought threshold
            atr_stop_multiplier: ATR multiplier for stop loss
            support_resistance_window: Window for S/R identification
            max_distance_to_level: Max distance to S/R level (ATR multiple)
            volume_confirmation: Whether to require volume confirmation
            volume_threshold: Volume threshold for confirmation
        """
        self.stoch_k_period = stoch_k_period
        self.stoch_k_smooth = stoch_k_smooth
        self.stoch_d_smooth = stoch_d_smooth
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
        self.atr_stop_multiplier = atr_stop_multiplier
        self.support_resistance_window = support_resistance_window
        self.max_distance_to_level = max_distance_to_level
        self.volume_confirmation = volume_confirmation
        self.volume_threshold = volume_threshold
        
        self.indicators = TechnicalIndicators()
        self.current_signals: list = []
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with all required indicators.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with calculated indicators
        """
        # Calculate stochastic oscillator
        data = self.indicators.stochastic(
            df,
            k_period=self.stoch_k_period,
            k_smooth=self.stoch_k_smooth,
            d_smooth=self.stoch_d_smooth,
            oversold_level=self.oversold_level,
            overbought_level=self.overbought_level
        )
        
        # Add ATR for stop loss calculations
        data['atr'] = pd.DataFrame(df).ta.atr(length=14)
        
        # Add volume confirmation if required
        if self.volume_confirmation:
            data = self.indicators.volume_confirmation(
                data,
                volume_threshold=self.volume_threshold
            )
        
        return data
    
    def identify_support_resistance(self, df: pd.DataFrame) -> Dict[str, list]:
        """
        Identify current support and resistance levels.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary with support and resistance levels
        """
        return self.indicators.support_resistance_levels(
            df,
            window=self.support_resistance_window
        )
    
    def check_near_level(self, 
                        price: float, 
                        level: float, 
                        atr: float) -> bool:
        """
        Check if price is near a support/resistance level.
        
        Args:
            price: Current price
            level: Support/resistance level
            atr: Current ATR value
            
        Returns:
            True if price is near the level
        """
        distance = abs(price - level)
        max_distance = self.max_distance_to_level * atr
        return distance <= max_distance
    
    def calculate_stop_loss(self, 
                           entry_price: float, 
                           atr: float, 
                           signal_type: SignalType,
                           support_level: Optional[float] = None,
                           resistance_level: Optional[float] = None) -> float:
        """
        Calculate stop loss level.
        
        Args:
            entry_price: Entry price
            atr: Current ATR
            signal_type: Buy or sell signal
            support_level: Support level (for buy signals)
            resistance_level: Resistance level (for sell signals)
            
        Returns:
            Stop loss level
        """
        if signal_type == SignalType.BUY:
            # For buy signals, stop below support or entry - ATR
            if support_level is not None:
                return support_level - (self.atr_stop_multiplier * atr)
            else:
                return entry_price - (self.atr_stop_multiplier * atr)
        
        elif signal_type == SignalType.SELL:
            # For sell signals, stop above resistance or entry + ATR
            if resistance_level is not None:
                return resistance_level + (self.atr_stop_multiplier * atr)
            else:
                return entry_price + (self.atr_stop_multiplier * atr)
        
        return entry_price
    
    def calculate_take_profit(self, 
                             entry_price: float,
                             signal_type: SignalType,
                             support_level: Optional[float] = None,
                             resistance_level: Optional[float] = None,
                             atr: float = None) -> float:
        """
        Calculate take profit level.
        
        Args:
            entry_price: Entry price
            signal_type: Buy or sell signal
            support_level: Support level
            resistance_level: Resistance level
            atr: Current ATR (fallback if no levels)
            
        Returns:
            Take profit level
        """
        if signal_type == SignalType.BUY:
            # Target resistance level for buy signals
            if resistance_level is not None:
                return resistance_level
            else:
                # Fallback: use ATR-based target
                return entry_price + (2 * atr) if atr else entry_price * 1.02
        
        elif signal_type == SignalType.SELL:
            # Target support level for sell signals
            if support_level is not None:
                return support_level
            else:
                # Fallback: use ATR-based target
                return entry_price - (2 * atr) if atr else entry_price * 0.98
        
        return entry_price
    
    def generate_signal(self, 
                       df: pd.DataFrame, 
                       current_idx: int) -> Optional[ConsolidationSignal]:
        """
        Generate trading signal for current market conditions.
        
        Args:
            df: DataFrame with prepared data
            current_idx: Current index
            
        Returns:
            ConsolidationSignal or None if no signal
        """
        if current_idx < 1 or current_idx >= len(df):
            return None
        
        # Get current market data
        current_row = df.iloc[current_idx]
        current_price = current_row['close']
        current_atr = current_row['atr']
        
        # Get stochastic signal
        stoch_signal = self.indicators.get_stochastic_signal(df, current_idx)
        if stoch_signal is None:
            return None
        
        # Identify support and resistance levels
        levels = self.identify_support_resistance(df.iloc[:current_idx+1])
        
        # Find nearest levels
        nearest_support = self.indicators.find_nearest_level(
            current_price, levels['support'], max_distance_pct=2.0
        )
        nearest_resistance = self.indicators.find_nearest_level(
            current_price, levels['resistance'], max_distance_pct=2.0
        )
        
        # Check volume confirmation
        volume_confirmed = True
        if self.volume_confirmation:
            volume_confirmed = current_row.get('high_volume', False)
        
        # Generate buy signal
        if (stoch_signal.in_oversold_zone and 
            stoch_signal.bullish_crossover and
            nearest_support is not None and
            self.check_near_level(current_price, nearest_support, current_atr)):
            
            # Calculate stop loss and take profit
            stop_loss = self.calculate_stop_loss(
                current_price, current_atr, SignalType.BUY, 
                support_level=nearest_support
            )
            
            take_profit = self.calculate_take_profit(
                current_price, SignalType.BUY,
                resistance_level=nearest_resistance,
                atr=current_atr
            )
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(
                stoch_signal, current_price, nearest_support, 
                nearest_resistance, volume_confirmed, SignalType.BUY
            )
            
            return ConsolidationSignal(
                signal_type=SignalType.BUY,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                reason=f"Stoch oversold crossover near support {nearest_support:.4f}",
                support_level=nearest_support,
                resistance_level=nearest_resistance,
                volume_confirmed=volume_confirmed
            )
        
        # Generate sell signal
        elif (stoch_signal.in_overbought_zone and 
              stoch_signal.bearish_crossover and
              nearest_resistance is not None and
              self.check_near_level(current_price, nearest_resistance, current_atr)):
            
            # Calculate stop loss and take profit
            stop_loss = self.calculate_stop_loss(
                current_price, current_atr, SignalType.SELL,
                resistance_level=nearest_resistance
            )
            
            take_profit = self.calculate_take_profit(
                current_price, SignalType.SELL,
                support_level=nearest_support,
                atr=current_atr
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                stoch_signal, current_price, nearest_support,
                nearest_resistance, volume_confirmed, SignalType.SELL
            )
            
            return ConsolidationSignal(
                signal_type=SignalType.SELL,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                reason=f"Stoch overbought crossover near resistance {nearest_resistance:.4f}",
                support_level=nearest_support,
                resistance_level=nearest_resistance,
                volume_confirmed=volume_confirmed
            )
        
        return None
    
    def _calculate_confidence(self, 
                             stoch_signal: StochasticSignal,
                             current_price: float,
                             support_level: Optional[float],
                             resistance_level: Optional[float],
                             volume_confirmed: bool,
                             signal_type: SignalType) -> float:
        """
        Calculate confidence score for the signal.
        
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.0
        
        # Base confidence for stochastic crossover
        if signal_type == SignalType.BUY and stoch_signal.bullish_crossover:
            confidence += 0.3
        elif signal_type == SignalType.SELL and stoch_signal.bearish_crossover:
            confidence += 0.3
        
        # Confidence for being in extreme zones
        if (signal_type == SignalType.BUY and stoch_signal.in_oversold_zone) or \
           (signal_type == SignalType.SELL and stoch_signal.in_overbought_zone):
            confidence += 0.3
        
        # Confidence for being near support/resistance
        relevant_level = support_level if signal_type == SignalType.BUY else resistance_level
        if relevant_level is not None:
            confidence += 0.2
        
        # Volume confirmation bonus
        if volume_confirmed:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def get_strategy_summary(self) -> dict:
        """
        Get summary of strategy parameters and current state.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            "strategy_type": "consolidation_mean_reversion",
            "stochastic_periods": f"({self.stoch_k_period},{self.stoch_k_smooth},{self.stoch_d_smooth})",
            "oversold_level": self.oversold_level,
            "overbought_level": self.overbought_level,
            "atr_stop_multiplier": self.atr_stop_multiplier,
            "volume_confirmation": self.volume_confirmation,
            "active_signals": len(self.current_signals)
        }