"""
Market Regime Detection Module

This module implements the four-state market regime classification system
based on ADX (trend strength) and ATR (volatility) indicators.

Market Regimes:
- Trending: ADX > 25, normal volatility
- Consolidation: ADX < 25, normal volatility  
- Stagnant: Any ADX, very low volatility
- Panic: Any ADX, extremely high volatility
"""

import pandas as pd
import pandas_ta as ta
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING = "trending"
    CONSOLIDATION = "consolidation"
    STAGNANT = "stagnant"
    PANIC = "panic"
    TRANSITION = "transition"


@dataclass
class RegimeState:
    """Current market regime state"""
    regime: MarketRegime
    adx_value: float
    atr_ratio: float
    confidence: float
    adx_slope: float
    last_change: Optional[pd.Timestamp] = None


class RegimeDetector:
    """
    Detects market regimes based on ADX and ATR analysis.
    
    Uses a four-state classification system:
    1. Trending: Strong directional movement (ADX > 25)
    2. Consolidation: Sideways movement (ADX < 25) 
    3. Stagnant: Very low volatility (ATR < 0.5 * ATR_SMA)
    4. Panic: Extremely high volatility (ATR > 3.0 * ATR_SMA)
    """
    
    def __init__(self, 
                 adx_period: int = 14,
                 atr_period: int = 14, 
                 atr_sma_period: int = 100,
                 adx_trend_threshold: float = 25.0,
                 adx_transition_lower: float = 20.0,
                 atr_stagnant_ratio: float = 0.5,
                 atr_panic_ratio: float = 3.0,
                 slope_periods: int = 3):
        """
        Initialize the regime detector.
        
        Args:
            adx_period: Period for ADX calculation
            atr_period: Period for ATR calculation
            atr_sma_period: Period for ATR moving average
            adx_trend_threshold: ADX threshold for trend identification
            adx_transition_lower: Lower ADX threshold for transition zone
            atr_stagnant_ratio: ATR ratio threshold for stagnant markets
            atr_panic_ratio: ATR ratio threshold for panic markets
            slope_periods: Periods for calculating ADX slope
        """
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.atr_sma_period = atr_sma_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_transition_lower = adx_transition_lower
        self.atr_stagnant_ratio = atr_stagnant_ratio
        self.atr_panic_ratio = atr_panic_ratio
        self.slope_periods = slope_periods
        
        self.current_regime: Optional[RegimeState] = None
        self.regime_history: list = []
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX, ATR and related indicators.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with calculated indicators
        """
        data = df.copy()
        
        # Calculate ADX and Directional Movement Indicators
        adx_data = ta.adx(data['high'], data['low'], data['close'],
                         length=self.adx_period)
        
        # Handle case where ADX returns None due to insufficient data
        if adx_data is not None:
            adx_col = f'ADX_{self.adx_period}'
            plus_di_col = f'DMP_{self.adx_period}'
            minus_di_col = f'DMN_{self.adx_period}'
            
            data['adx'] = adx_data[adx_col] if adx_col in adx_data.columns else np.nan
            data['plus_di'] = adx_data[plus_di_col] if plus_di_col in adx_data.columns else np.nan
            data['minus_di'] = adx_data[minus_di_col] if minus_di_col in adx_data.columns else np.nan
        else:
            data['adx'] = np.nan
            data['plus_di'] = np.nan
            data['minus_di'] = np.nan
        
        # Calculate ATR and its moving average
        data['atr'] = ta.atr(data['high'], data['low'], data['close'], 
                           length=self.atr_period)
        data['atr_sma'] = data['atr'].rolling(window=self.atr_sma_period).mean()
        data['atr_ratio'] = data['atr'] / data['atr_sma']
        
        # Calculate ADX slope for trend strength assessment
        data['adx_slope'] = data['adx'].diff(self.slope_periods) / self.slope_periods
        
        return data
    
    def classify_regime(self, adx: float, atr_ratio: float, adx_slope: float = 0) -> Tuple[MarketRegime, float]:
        """
        Classify market regime based on ADX and ATR values.
        
        Args:
            adx: Current ADX value
            atr_ratio: Current ATR to ATR_SMA ratio
            adx_slope: ADX slope (rate of change)
            
        Returns:
            Tuple of (MarketRegime, confidence_score)
        """
        # First check volatility extremes (override ADX)
        if atr_ratio < self.atr_stagnant_ratio:
            confidence = min(1.0, (self.atr_stagnant_ratio - atr_ratio) / self.atr_stagnant_ratio)
            return MarketRegime.STAGNANT, confidence
        
        if atr_ratio > self.atr_panic_ratio:
            confidence = min(1.0, (atr_ratio - self.atr_panic_ratio) / self.atr_panic_ratio)
            return MarketRegime.PANIC, confidence
        
        # Normal volatility range - classify by ADX
        if pd.isna(adx):
            return MarketRegime.TRANSITION, 0.0
        
        # Transition zone
        if self.adx_transition_lower <= adx <= self.adx_trend_threshold:
            return MarketRegime.TRANSITION, 0.5
        
        # Clear trend
        if adx > self.adx_trend_threshold:
            # Higher confidence for stronger trends and positive slope
            base_confidence = min(1.0, (adx - self.adx_trend_threshold) / 25.0)
            slope_bonus = max(0, adx_slope / 5.0)  # Bonus for rising ADX
            confidence = min(1.0, base_confidence + slope_bonus)
            return MarketRegime.TRENDING, confidence
        
        # Consolidation
        if adx < self.adx_transition_lower:
            confidence = min(1.0, (self.adx_transition_lower - adx) / self.adx_transition_lower)
            return MarketRegime.CONSOLIDATION, confidence
        
        return MarketRegime.TRANSITION, 0.5
    
    def detect_regime(self, df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Detect market regime from data - wrapper for compatibility.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Tuple of (MarketRegime, confidence_score)
        """
        regime_state = self.update_regime(df)
        return regime_state.regime, regime_state.confidence
    
    def update_regime(self, df: pd.DataFrame) -> RegimeState:
        """
        Update the current market regime based on latest data.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Current RegimeState
        """
        # Calculate indicators
        data = self.calculate_indicators(df)
        
        # Get latest values
        latest_idx = data.index[-1]
        adx = data.loc[latest_idx, 'adx']
        atr_ratio = data.loc[latest_idx, 'atr_ratio']
        adx_slope = data.loc[latest_idx, 'adx_slope']
        
        # Classify regime
        regime, confidence = self.classify_regime(adx, atr_ratio, adx_slope)
        
        # Check for regime change
        last_change = None
        if (self.current_regime is None or 
            self.current_regime.regime != regime):
            last_change = latest_idx
        
        # Create new regime state
        new_state = RegimeState(
            regime=regime,
            adx_value=adx if not pd.isna(adx) else 0.0,
            atr_ratio=atr_ratio if not pd.isna(atr_ratio) else 1.0,
            confidence=confidence,
            adx_slope=adx_slope if not pd.isna(adx_slope) else 0.0,
            last_change=last_change
        )
        
        # Update history
        if last_change is not None:
            self.regime_history.append({
                'timestamp': latest_idx,
                'old_regime': self.current_regime.regime if self.current_regime else None,
                'new_regime': regime,
                'adx': adx,
                'atr_ratio': atr_ratio
            })
        
        self.current_regime = new_state
        return new_state
    
    def can_open_positions(self) -> bool:
        """
        Check if new positions can be opened in current regime.
        
        Returns:
            True if positions can be opened, False otherwise
        """
        if self.current_regime is None:
            return False
        
        return self.current_regime.regime in [MarketRegime.TRENDING, MarketRegime.CONSOLIDATION]
    
    def should_tighten_stops(self) -> bool:
        """
        Check if stop losses should be tightened.
        
        Returns:
            True if stops should be tightened (panic regime)
        """
        if self.current_regime is None:
            return False
        
        return self.current_regime.regime == MarketRegime.PANIC
    
    def get_regime_summary(self) -> dict:
        """
        Get a summary of the current regime state.
        
        Returns:
            Dictionary with regime information
        """
        if self.current_regime is None:
            return {"regime": "unknown", "can_trade": False}
        
        return {
            "regime": self.current_regime.regime.value,
            "adx": self.current_regime.adx_value,
            "atr_ratio": self.current_regime.atr_ratio,
            "confidence": self.current_regime.confidence,
            "adx_slope": self.current_regime.adx_slope,
            "can_trade": self.can_open_positions(),
            "tighten_stops": self.should_tighten_stops(),
            "last_change": self.current_regime.last_change
        }