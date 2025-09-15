"""
Trend Strategy Module

Implements trend-following strategy for trending markets using DMI system.
Active when ADX > 25 and volatility is in normal range.

Strategy Logic:
- Buy signals: +DI crosses above -DI with strong ADX (> 25)
- Sell signals: -DI crosses above +DI with strong ADX (> 25)
- Open take profit (trailing stop only)
- ATR-based initial stops, then Chandelier Exit trailing stops
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from indicators.technical import TechnicalIndicators, DMISignal


class TrendSignalType(Enum):
    """Signal types for trend strategy"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    LONG_ENTRY = "long_entry"
    SHORT_ENTRY = "short_entry"
    EXIT = "exit"


@dataclass
class TrendSignal:
    """Trend strategy signal"""
    signal_type: TrendSignalType
    entry_price: float
    initial_stop_loss: float
    trailing_stop_multiplier: float
    confidence: float
    reason: str
    adx_value: float
    plus_di: float
    minus_di: float
    adx_slope: float
    ema_confirmed: bool = False


class TrendStrategy:
    """
    Trend-following strategy for trending markets.
    
    Uses Directional Movement Index (DMI) to identify trend direction
    and generates signals on +DI/-DI crossovers with strong ADX.
    Employs Chandelier Exit for trailing stops.
    """
    
    @classmethod
    def get_strategy_info(cls) -> Dict[str, Any]:
        """
        Get comprehensive strategy information for GUI display.
        
        Returns:
            Dictionary with strategy description, logic, and parameters
        """
        return {
            'name': 'Trend Following Strategy',
            'short_description': 'DMI-based trend following with Chandelier Exit trailing stops',
            'detailed_description': """
            **Trend Following Strategy** is designed for trending markets where price shows clear directional movement.
            
            **Core Logic:**
            • Uses Directional Movement Index (DMI) to identify trend strength and direction
            • Generates buy signals when +DI crosses above -DI with strong ADX (> 25)
            • Generates sell signals when -DI crosses above +DI with strong ADX (> 25)
            • Uses ATR-based initial stops, then switches to Chandelier Exit trailing stops
            • Optional EMA filter to confirm trend direction
            • Optional volume confirmation to validate breakouts
            
            **When Active:**
            • Market regime: TRENDING (ADX > 25)
            • Normal volatility conditions (not stagnant or panic)
            
            **Risk Management:**
            • Initial stop loss: Entry price ± (ATR × stop multiplier)
            • Trailing stop: Chandelier Exit (Highest/Lowest ± ATR × chandelier multiplier)
            • No fixed take profit - lets winners run with trailing stops
            """,
            'entry_criteria': [
                'ADX > threshold (confirms trend strength)',
                '+DI crosses above -DI (bullish) or -DI crosses above +DI (bearish)',
                'ADX slope > minimum (trend is strengthening)',
                'Price above/below EMA (optional trend filter)',
                'Volume above threshold (optional confirmation)'
            ],
            'exit_criteria': [
                'Stop loss hit (ATR-based initial or Chandelier trailing)',
                'Opposite DMI crossover signal',
                'ADX falls below threshold (trend weakening)',
                'Market regime changes (e.g., to CONSOLIDATION or PANIC)'
            ],
            'market_conditions': {
                'best_for': 'Strong trending markets with clear directional bias',
                'worst_for': 'Sideways, choppy, or highly volatile markets',
                'regime': 'TRENDING (ADX > 25)'
            },
            'advantages': [
                'Captures large trending moves effectively',
                'Trailing stops protect profits while letting winners run',
                'DMI system filters out false signals in consolidation',
                'Adaptive to market volatility via ATR-based stops'
            ],
            'disadvantages': [
                'Suffers in sideways markets with frequent whipsaws',
                'Late entry/exit signals (trend confirmation required)',
                'Requires significant trend strength to be profitable'
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
            'dmi_period': {
                'name': 'DMI Period',
                'description': 'Period for calculating Directional Movement Index (DMI)',
                'type': 'int',
                'default': 14,
                'min_value': 5,
                'max_value': 30,
                'step': 1,
                'impact': 'Lower = more sensitive to short-term moves, Higher = smoother but slower signals',
                'optimization_range': [10, 12, 14, 16, 18, 20]
            },
            'adx_threshold': {
                'name': 'ADX Threshold',
                'description': 'Minimum ADX value to confirm trend strength',
                'type': 'float',
                'default': 25.0,
                'min_value': 15.0,
                'max_value': 40.0,
                'step': 2.5,
                'impact': 'Lower = more signals in weaker trends, Higher = only very strong trends',
                'optimization_range': [20.0, 22.5, 25.0, 27.5, 30.0]
            },
            'atr_stop_multiplier': {
                'name': 'ATR Stop Multiplier',
                'description': 'ATR multiplier for initial stop loss distance',
                'type': 'float',
                'default': 2.0,
                'min_value': 1.0,
                'max_value': 4.0,
                'step': 0.25,
                'impact': 'Lower = tighter stops (more stopped out), Higher = wider stops (larger losses)',
                'optimization_range': [1.5, 2.0, 2.5, 3.0]
            },
            'chandelier_multiplier': {
                'name': 'Chandelier Exit Multiplier',
                'description': 'ATR multiplier for Chandelier Exit trailing stops',
                'type': 'float',
                'default': 3.0,
                'min_value': 2.0,
                'max_value': 5.0,
                'step': 0.25,
                'impact': 'Lower = tighter trailing (secure profits faster), Higher = looser trailing (ride trends longer)',
                'optimization_range': [2.5, 3.0, 3.5, 4.0]
            },
            'ema_period': {
                'name': 'EMA Filter Period',
                'description': 'EMA period for trend direction filter',
                'type': 'int',
                'default': 50,
                'min_value': 20,
                'max_value': 100,
                'step': 10,
                'impact': 'Lower = more responsive filter, Higher = stronger trend confirmation',
                'optimization_range': [30, 40, 50, 60, 80]
            },
            'min_adx_slope': {
                'name': 'Minimum ADX Slope',
                'description': 'Minimum ADX slope to confirm strengthening trend',
                'type': 'float',
                'default': 0.5,
                'min_value': 0.0,
                'max_value': 2.0,
                'step': 0.25,
                'impact': 'Higher = only enter when trend is clearly accelerating',
                'optimization_range': [0.0, 0.5, 1.0, 1.5]
            },
            'volume_threshold': {
                'name': 'Volume Confirmation Threshold',
                'description': 'Volume multiplier for confirming breakouts',
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
            'dmi_period': self.dmi_period,
            'adx_threshold': self.adx_threshold,
            'atr_stop_multiplier': self.atr_stop_multiplier,
            'chandelier_multiplier': self.chandelier_multiplier,
            'ema_period': self.ema_period,
            'use_ema_filter': self.use_ema_filter,
            'min_adx_slope': self.min_adx_slope,
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
                 dmi_period: int = 14,
                 adx_threshold: float = 25.0,
                 atr_stop_multiplier: float = 2.0,
                 chandelier_multiplier: float = 3.0,
                 ema_period: int = 50,
                 use_ema_filter: bool = True,
                 min_adx_slope: float = 0.5,
                 volume_confirmation: bool = False,
                 volume_threshold: float = 1.2):
        """
        Initialize trend strategy.
        
        Args:
            dmi_period: Period for DMI calculation
            adx_threshold: Minimum ADX for trend confirmation
            atr_stop_multiplier: ATR multiplier for initial stop loss
            chandelier_multiplier: ATR multiplier for Chandelier Exit
            ema_period: EMA period for trend filter
            use_ema_filter: Whether to use EMA filter
            min_adx_slope: Minimum ADX slope for signal validation
            volume_confirmation: Whether to require volume confirmation
            volume_threshold: Volume threshold for confirmation
        """
        self.dmi_period = dmi_period
        self.adx_threshold = adx_threshold
        self.atr_stop_multiplier = atr_stop_multiplier
        self.chandelier_multiplier = chandelier_multiplier
        self.ema_period = ema_period
        self.use_ema_filter = use_ema_filter
        self.min_adx_slope = min_adx_slope
        self.volume_confirmation = volume_confirmation
        self.volume_threshold = volume_threshold
        
        self.indicators = TechnicalIndicators()
        self.current_signals: list = []
        self.position_highs: Dict[str, float] = {}  # Track highs for trailing stops
        self.position_lows: Dict[str, float] = {}   # Track lows for trailing stops
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with all required indicators.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with calculated indicators
        """
        # Calculate DMI system
        data = self.indicators.dmi_system(df, period=self.dmi_period)
        
        # Add ATR for stop loss calculations
        data['atr'] = pd.DataFrame(df).ta.atr(length=14)
        
        # Calculate ADX slope
        data['adx_slope'] = data['adx'].diff(3) / 3  # 3-period slope
        
        # Add EMA filter if required
        if self.use_ema_filter:
            data = self.indicators.ema_filter(data, period=self.ema_period)
        
        # Add volume confirmation if required
        if self.volume_confirmation:
            data = self.indicators.volume_confirmation(
                data,
                volume_threshold=self.volume_threshold
            )
        
        return data
    
    def calculate_initial_stop(self, 
                              entry_price: float, 
                              atr: float, 
                              signal_type: TrendSignalType) -> float:
        """
        Calculate initial stop loss level.
        
        Args:
            entry_price: Entry price
            atr: Current ATR
            signal_type: Buy or sell signal
            
        Returns:
            Initial stop loss level
        """
        if signal_type == TrendSignalType.BUY:
            return entry_price - (self.atr_stop_multiplier * atr)
        elif signal_type == TrendSignalType.SELL:
            return entry_price + (self.atr_stop_multiplier * atr)
        
        return entry_price
    
    def calculate_chandelier_exit(self, 
                                 highest_high: float,
                                 lowest_low: float, 
                                 atr: float, 
                                 signal_type: TrendSignalType) -> float:
        """
        Calculate Chandelier Exit trailing stop level.
        
        Args:
            highest_high: Highest high since entry (for long positions)
            lowest_low: Lowest low since entry (for short positions)
            atr: Current ATR
            signal_type: Buy or sell signal
            
        Returns:
            Chandelier Exit level
        """
        if signal_type == TrendSignalType.BUY:
            # For long positions: Highest High - (N × ATR)
            return highest_high - (self.chandelier_multiplier * atr)
        elif signal_type == TrendSignalType.SELL:
            # For short positions: Lowest Low + (N × ATR)
            return lowest_low + (self.chandelier_multiplier * atr)
        
        return 0.0
    
    def check_ema_filter(self, 
                        price: float, 
                        ema_value: float, 
                        signal_type: TrendSignalType) -> bool:
        """
        Check EMA filter condition.
        
        Args:
            price: Current price
            ema_value: Current EMA value
            signal_type: Buy or sell signal
            
        Returns:
            True if EMA filter is satisfied
        """
        if not self.use_ema_filter or pd.isna(ema_value):
            return True
        
        if signal_type == TrendSignalType.BUY:
            return price > ema_value
        elif signal_type == TrendSignalType.SELL:
            return price < ema_value
        
        return False
    
    def generate_signal(self, 
                       df: pd.DataFrame, 
                       current_idx: int) -> Optional[TrendSignal]:
        """
        Generate trading signal for current market conditions.
        
        Args:
            df: DataFrame with prepared data
            current_idx: Current index
            
        Returns:
            TrendSignal or None if no signal
        """
        if current_idx < 1 or current_idx >= len(df):
            return None
        
        # Get current market data
        current_row = df.iloc[current_idx]
        current_price = current_row['close']
        current_atr = current_row['atr']
        current_adx = current_row['adx']
        current_adx_slope = current_row.get('adx_slope', 0)
        
        # Check ADX threshold
        if pd.isna(current_adx) or current_adx < self.adx_threshold:
            return None
        
        # Check ADX slope (trend must be strengthening)
        if current_adx_slope < self.min_adx_slope:
            return None
        
        # Get DMI signal
        dmi_signal = self.indicators.get_dmi_signal(df, current_idx)
        if dmi_signal is None:
            return None
        
        # Check EMA filter
        ema_value = current_row.get(f'ema_{self.ema_period}', current_price)
        
        # Check volume confirmation
        volume_confirmed = True
        if self.volume_confirmation:
            volume_confirmed = current_row.get('high_volume', False)
        
        # Generate buy signal
        if (dmi_signal.bullish_crossover and 
            dmi_signal.trend_direction == "bullish" and
            self.check_ema_filter(current_price, ema_value, TrendSignalType.BUY)):
            
            # Calculate initial stop loss
            initial_stop = self.calculate_initial_stop(
                current_price, current_atr, TrendSignalType.BUY
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                dmi_signal, current_adx, current_adx_slope, 
                volume_confirmed, TrendSignalType.BUY
            )
            
            return TrendSignal(
                signal_type=TrendSignalType.BUY,
                entry_price=current_price,
                initial_stop_loss=initial_stop,
                trailing_stop_multiplier=self.chandelier_multiplier,
                confidence=confidence,
                reason=f"+DI crossed above -DI, ADX={current_adx:.1f}, slope={current_adx_slope:.2f}",
                adx_value=current_adx,
                plus_di=dmi_signal.plus_di,
                minus_di=dmi_signal.minus_di,
                adx_slope=current_adx_slope,
                ema_confirmed=self.use_ema_filter and current_price > ema_value
            )
        
        # Generate sell signal
        elif (dmi_signal.bearish_crossover and 
              dmi_signal.trend_direction == "bearish" and
              self.check_ema_filter(current_price, ema_value, TrendSignalType.SELL)):
            
            # Calculate initial stop loss
            initial_stop = self.calculate_initial_stop(
                current_price, current_atr, TrendSignalType.SELL
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                dmi_signal, current_adx, current_adx_slope,
                volume_confirmed, TrendSignalType.SELL
            )
            
            return TrendSignal(
                signal_type=TrendSignalType.SELL,
                entry_price=current_price,
                initial_stop_loss=initial_stop,
                trailing_stop_multiplier=self.chandelier_multiplier,
                confidence=confidence,
                reason=f"-DI crossed above +DI, ADX={current_adx:.1f}, slope={current_adx_slope:.2f}",
                adx_value=current_adx,
                plus_di=dmi_signal.plus_di,
                minus_di=dmi_signal.minus_di,
                adx_slope=current_adx_slope,
                ema_confirmed=self.use_ema_filter and current_price < ema_value
            )
        
        return None
    
    def update_trailing_stop(self, 
                           position_id: str,
                           current_high: float,
                           current_low: float,
                           atr: float,
                           signal_type: TrendSignalType) -> float:
        """
        Update trailing stop using Chandelier Exit method.
        
        Args:
            position_id: Unique position identifier
            current_high: Current high price
            current_low: Current low price
            atr: Current ATR
            signal_type: Position type (buy/sell)
            
        Returns:
            Updated trailing stop level
        """
        if signal_type == TrendSignalType.BUY:
            # Track highest high since position opened
            if position_id not in self.position_highs:
                self.position_highs[position_id] = current_high
            else:
                self.position_highs[position_id] = max(
                    self.position_highs[position_id], current_high
                )
            
            return self.calculate_chandelier_exit(
                self.position_highs[position_id], 
                0,  # Not used for long positions
                atr, 
                signal_type
            )
        
        elif signal_type == TrendSignalType.SELL:
            # Track lowest low since position opened
            if position_id not in self.position_lows:
                self.position_lows[position_id] = current_low
            else:
                self.position_lows[position_id] = min(
                    self.position_lows[position_id], current_low
                )
            
            return self.calculate_chandelier_exit(
                0,  # Not used for short positions
                self.position_lows[position_id],
                atr,
                signal_type
            )
        
        return 0.0
    
    def close_position_tracking(self, position_id: str):
        """
        Clean up position tracking data when position is closed.
        
        Args:
            position_id: Position identifier to clean up
        """
        self.position_highs.pop(position_id, None)
        self.position_lows.pop(position_id, None)
    
    def _calculate_confidence(self, 
                             dmi_signal: DMISignal,
                             adx_value: float,
                             adx_slope: float,
                             volume_confirmed: bool,
                             signal_type: TrendSignalType) -> float:
        """
        Calculate confidence score for the signal.
        
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.0
        
        # Base confidence for DMI crossover
        if signal_type == TrendSignalType.BUY and dmi_signal.bullish_crossover:
            confidence += 0.3
        elif signal_type == TrendSignalType.SELL and dmi_signal.bearish_crossover:
            confidence += 0.3
        
        # ADX strength bonus
        if dmi_signal.trend_strength == "strong":
            confidence += 0.2
        elif dmi_signal.trend_strength == "very_strong":
            confidence += 0.3
        elif dmi_signal.trend_strength == "moderate":
            confidence += 0.1
        
        # ADX slope bonus (strengthening trend)
        if adx_slope > 1.0:
            confidence += 0.2
        elif adx_slope > 0.5:
            confidence += 0.1
        
        # Volume confirmation bonus
        if volume_confirmed:
            confidence += 0.1
        
        # EMA filter bonus
        if self.use_ema_filter:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def should_exit_on_regime_change(self, 
                                   current_adx: float,
                                   original_regime_adx_threshold: float = 25.0) -> bool:
        """
        Check if position should be exited due to regime change.
        
        Args:
            current_adx: Current ADX value
            original_regime_adx_threshold: Original ADX threshold for trend
            
        Returns:
            True if regime has changed and position should be managed differently
        """
        return current_adx < original_regime_adx_threshold
    
    def get_strategy_summary(self) -> dict:
        """
        Get summary of strategy parameters and current state.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            "strategy_type": "trend_following_dmi",
            "dmi_period": self.dmi_period,
            "adx_threshold": self.adx_threshold,
            "atr_stop_multiplier": self.atr_stop_multiplier,
            "chandelier_multiplier": self.chandelier_multiplier,
            "ema_filter": self.use_ema_filter,
            "ema_period": self.ema_period if self.use_ema_filter else None,
            "min_adx_slope": self.min_adx_slope,
            "volume_confirmation": self.volume_confirmation,
            "active_signals": len(self.current_signals),
            "tracking_positions": len(self.position_highs) + len(self.position_lows)
        }