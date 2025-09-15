"""
Technical Indicators Module

Provides technical analysis indicators specifically tailored for the adaptive trading bot.
Includes Stochastic Oscillator for consolidation strategy and DMI system for trend strategy.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Some indicators may not work.")
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StochasticSignal:
    """Stochastic oscillator signal data"""
    k_value: float
    d_value: float
    k_prev: float
    d_prev: float
    oversold: bool
    overbought: bool
    bullish_crossover: bool
    bearish_crossover: bool
    in_oversold_zone: bool
    in_overbought_zone: bool
    signal_type: str = "HOLD"


@dataclass
class DMISignal:
    """Directional Movement Index signal data"""
    plus_di: float
    minus_di: float
    adx: float
    plus_di_prev: float
    minus_di_prev: float
    bullish_crossover: bool
    bearish_crossover: bool
    trend_strength: str  # "weak", "moderate", "strong", "very_strong"
    trend_direction: str  # "bullish", "bearish", "neutral"


class TechnicalIndicators:
    """
    Technical indicators calculator for the adaptive trading bot.
    
    Provides optimized calculations for:
    - Stochastic Oscillator (for consolidation strategy)
    - Directional Movement Index - DMI (for trend strategy)
    - Support/Resistance levels
    - Volume analysis
    """
    
    def __init__(self):
        pass
    
    def stochastic(self, 
                   df: pd.DataFrame,
                   k_period: int = 5,
                   k_smooth: int = 3,
                   d_smooth: int = 3,
                   oversold_level: float = 20,
                   overbought_level: float = 80) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator with signal detection.
        
        Args:
            df: DataFrame with OHLC data
            k_period: Period for %K calculation
            k_smooth: Smoothing period for %K
            d_smooth: Smoothing period for %D
            oversold_level: Oversold threshold
            overbought_level: Overbought threshold
            
        Returns:
            DataFrame with stochastic values and signals
        """
        data = df.copy()
        
        # Calculate stochastic values
        stoch = ta.stoch(data['high'], data['low'], data['close'],
                        k=k_period, d=d_smooth, smooth_k=k_smooth)
        
        # Handle case where stochastic returns None due to insufficient data
        if stoch is not None:
            k_col = f'STOCHk_{k_period}_{k_smooth}_{d_smooth}'
            d_col = f'STOCHd_{k_period}_{k_smooth}_{d_smooth}'
            data['stoch_k'] = stoch[k_col] if k_col in stoch.columns else np.nan
            data['stoch_d'] = stoch[d_col] if d_col in stoch.columns else np.nan
        else:
            data['stoch_k'] = np.nan
            data['stoch_d'] = np.nan
        
        # Calculate previous values for crossover detection
        data['stoch_k_prev'] = data['stoch_k'].shift(1)
        data['stoch_d_prev'] = data['stoch_d'].shift(1)
        
        # Generate signals
        data['stoch_oversold'] = data['stoch_k'] < oversold_level
        data['stoch_overbought'] = data['stoch_k'] > overbought_level
        
        # Crossover signals
        data['stoch_bullish_cross'] = (
            (data['stoch_k'] > data['stoch_d']) & 
            (data['stoch_k_prev'] <= data['stoch_d_prev'])
        )
        
        data['stoch_bearish_cross'] = (
            (data['stoch_k'] < data['stoch_d']) & 
            (data['stoch_k_prev'] >= data['stoch_d_prev'])
        )
        
        # Zone signals (both K and D in oversold/overbought)
        data['stoch_in_oversold'] = (
            (data['stoch_k'] < oversold_level) & 
            (data['stoch_d'] < oversold_level)
        )
        
        data['stoch_in_overbought'] = (
            (data['stoch_k'] > overbought_level) & 
            (data['stoch_d'] > overbought_level)
        )
        
        return data
    
    def get_stochastic_signal(self, df: pd.DataFrame, idx: int) -> Optional[StochasticSignal]:
        """
        Get stochastic signal for a specific index.
        
        Args:
            df: DataFrame with stochastic data
            idx: Index to get signal for
            
        Returns:
            StochasticSignal object or None if data insufficient
        """
        if idx < 1 or idx >= len(df):
            return None
        
        required_cols = ['stoch_k', 'stoch_d', 'stoch_k_prev', 'stoch_d_prev']
        if not all(col in df.columns for col in required_cols):
            return None
        
        row = df.iloc[idx]
        
        # Determine signal type
        if row['stoch_bullish_cross'] and row['stoch_in_oversold']:
            signal_type = "BUY"
        elif row['stoch_bearish_cross'] and row['stoch_in_overbought']:
            signal_type = "SELL"
        else:
            signal_type = "HOLD"
        
        return StochasticSignal(
            k_value=row['stoch_k'],
            d_value=row['stoch_d'],
            k_prev=row['stoch_k_prev'],
            d_prev=row['stoch_d_prev'],
            oversold=row['stoch_oversold'],
            overbought=row['stoch_overbought'],
            bullish_crossover=row['stoch_bullish_cross'],
            bearish_crossover=row['stoch_bearish_cross'],
            in_oversold_zone=row['stoch_in_oversold'],
            in_overbought_zone=row['stoch_in_overbought'],
            signal_type=signal_type
        )
    
    def dmi_system(self, 
                   df: pd.DataFrame,
                   period: int = 14) -> pd.DataFrame:
        """
        Calculate Directional Movement Index (DMI) system.
        
        Args:
            df: DataFrame with OHLC data
            period: Period for DMI calculation
            
        Returns:
            DataFrame with DMI values and signals
        """
        data = df.copy()
        
        # Calculate DMI components
        adx_data = ta.adx(data['high'], data['low'], data['close'], length=period)
        
        # Handle case where ADX returns None due to insufficient data
        if adx_data is not None:
            adx_col = f'ADX_{period}'
            plus_di_col = f'DMP_{period}'
            minus_di_col = f'DMN_{period}'
            
            data['adx'] = adx_data[adx_col] if adx_col in adx_data.columns else np.nan
            data['plus_di'] = adx_data[plus_di_col] if plus_di_col in adx_data.columns else np.nan
            data['minus_di'] = adx_data[minus_di_col] if minus_di_col in adx_data.columns else np.nan
        else:
            data['adx'] = np.nan
            data['plus_di'] = np.nan
            data['minus_di'] = np.nan
        
        # Calculate previous values for crossover detection
        data['plus_di_prev'] = data['plus_di'].shift(1)
        data['minus_di_prev'] = data['minus_di'].shift(1)
        
        # Generate crossover signals
        data['dmi_bullish_cross'] = (
            (data['plus_di'] > data['minus_di']) & 
            (data['plus_di_prev'] <= data['minus_di_prev'])
        )
        
        data['dmi_bearish_cross'] = (
            (data['plus_di'] < data['minus_di']) & 
            (data['plus_di_prev'] >= data['minus_di_prev'])
        )
        
        # Trend direction
        data['dmi_trend_bullish'] = data['plus_di'] > data['minus_di']
        data['dmi_trend_bearish'] = data['plus_di'] < data['minus_di']
        
        # ADX strength categories
        data['adx_strength'] = pd.cut(
            data['adx'], 
            bins=[0, 25, 40, 50, 100],
            labels=['weak', 'moderate', 'strong', 'very_strong'],
            include_lowest=True
        )
        
        return data
    
    def get_dmi_signal(self, df: pd.DataFrame, idx: int) -> Optional[DMISignal]:
        """
        Get DMI signal for a specific index.
        
        Args:
            df: DataFrame with DMI data
            idx: Index to get signal for
            
        Returns:
            DMISignal object or None if data insufficient
        """
        if idx < 1 or idx >= len(df):
            return None
        
        required_cols = ['plus_di', 'minus_di', 'adx', 'plus_di_prev', 'minus_di_prev']
        if not all(col in df.columns for col in required_cols):
            return None
        
        row = df.iloc[idx]
        
        # Determine trend direction
        if row['plus_di'] > row['minus_di']:
            trend_direction = "bullish"
        elif row['plus_di'] < row['minus_di']:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"
        
        # Determine trend strength
        if pd.isna(row['adx']):
            trend_strength = "weak"
        elif row['adx'] < 25:
            trend_strength = "weak"
        elif row['adx'] < 40:
            trend_strength = "moderate"
        elif row['adx'] < 50:
            trend_strength = "strong"
        else:
            trend_strength = "very_strong"
        
        return DMISignal(
            plus_di=row['plus_di'],
            minus_di=row['minus_di'],
            adx=row['adx'],
            plus_di_prev=row['plus_di_prev'],
            minus_di_prev=row['minus_di_prev'],
            bullish_crossover=row['dmi_bullish_cross'],
            bearish_crossover=row['dmi_bearish_cross'],
            trend_strength=trend_strength,
            trend_direction=trend_direction
        )
    
    def support_resistance_levels(self, 
                                df: pd.DataFrame, 
                                window: int = 20,
                                min_touches: int = 2) -> Dict[str, list]:
        """
        Identify support and resistance levels using pivot points.
        
        Args:
            df: DataFrame with OHLC data
            window: Window for pivot point detection
            min_touches: Minimum touches to confirm a level
            
        Returns:
            Dictionary with support and resistance levels
        """
        data = df.copy()
        
        # Find pivot highs and lows
        data['pivot_high'] = data['high'].rolling(window=window*2+1, center=True).max() == data['high']
        data['pivot_low'] = data['low'].rolling(window=window*2+1, center=True).min() == data['low']
        
        # Extract pivot points
        pivot_highs = data[data['pivot_high']]['high'].values
        pivot_lows = data[data['pivot_low']]['low'].values
        
        # Cluster similar levels
        def cluster_levels(levels, tolerance=0.001):
            if len(levels) == 0:
                return []
            
            clustered = []
            sorted_levels = np.sort(levels)
            current_cluster = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                    current_cluster.append(level)
                else:
                    clustered.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clustered.append(np.mean(current_cluster))
            return clustered
        
        support_levels = cluster_levels(pivot_lows)
        resistance_levels = cluster_levels(pivot_highs)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    def find_nearest_level(self, price: float, levels: list, max_distance_pct: float = 2.0) -> Optional[float]:
        """
        Find the nearest support or resistance level to current price.
        
        Args:
            price: Current price
            levels: List of support/resistance levels
            max_distance_pct: Maximum distance as percentage of price
            
        Returns:
            Nearest level within max_distance or None
        """
        if not levels:
            return None
        
        distances = [abs(price - level) / price * 100 for level in levels]
        min_idx = np.argmin(distances)
        
        if distances[min_idx] <= max_distance_pct:
            return levels[min_idx]
        
        return None
    
    def volume_confirmation(self, 
                          df: pd.DataFrame, 
                          volume_ma_period: int = 20,
                          volume_threshold: float = 1.2) -> pd.DataFrame:
        """
        Add volume confirmation signals.
        
        Args:
            df: DataFrame with OHLCV data
            volume_ma_period: Period for volume moving average
            volume_threshold: Threshold for significant volume (multiple of MA)
            
        Returns:
            DataFrame with volume signals
        """
        data = df.copy()
        
        if 'volume' not in data.columns:
            # Create dummy volume data if not available
            data['volume'] = 1000
        
        # Calculate volume moving average
        data['volume_ma'] = data['volume'].rolling(window=volume_ma_period).mean()
        
        # Volume signals
        data['high_volume'] = data['volume'] > (data['volume_ma'] * volume_threshold)
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        return data
    
    def ema_filter(self, 
                   df: pd.DataFrame, 
                   period: int = 50) -> pd.DataFrame:
        """
        Add EMA filter for trend confirmation.
        
        Args:
            df: DataFrame with OHLC data
            period: EMA period
            
        Returns:
            DataFrame with EMA and trend signals
        """
        data = df.copy()
        
        # Calculate EMA
        ema_result = ta.ema(data['close'], length=period)
        data[f'ema_{period}'] = ema_result if ema_result is not None else np.nan
        
        # Trend signals - handle NaN values safely
        data['above_ema'] = (data['close'] > data[f'ema_{period}']).fillna(False)
        data['below_ema'] = (data['close'] < data[f'ema_{period}']).fillna(False)
        
        return data
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame - for test compatibility.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with all indicators added
        """
        data = df.copy()
        
        # Add basic indicators using pandas-ta with null checks
        data['rsi'] = ta.rsi(data['close'])
        data['atr'] = ta.atr(data['high'], data['low'], data['close'])
        
        # Handle ADX which can return None for insufficient data
        adx_result = ta.adx(data['high'], data['low'], data['close'])
        if adx_result is not None and 'ADX_14' in adx_result.columns:
            data['adx'] = adx_result['ADX_14']
        else:
            data['adx'] = np.nan
            
        data['ema_20'] = ta.ema(data['close'], length=20)
        data['ema_50'] = ta.ema(data['close'], length=50)
        data['sma_200'] = ta.sma(data['close'], length=200)
        
        # Add custom indicators
        data = self.stochastic(data)
        data = self.dmi_system(data)
        data = self.volume_confirmation(data)
        data = self.ema_filter(data)
        
        # Add TA-Lib enhanced indicators
        data = self.talib_rsi(data)
        data = self.talib_macd(data)
        data = self.talib_bbands(data)
        data = self.talib_adx(data)
        
        return data
    
    def talib_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate RSI using TA-Lib for enhanced accuracy.
        
        Args:
            df: DataFrame with OHLC data
            period: RSI period
            
        Returns:
            DataFrame with TA-Lib RSI
        """
        data = df.copy()
        
        if TALIB_AVAILABLE and len(data) >= period:
            try:
                data[f'talib_rsi_{period}'] = talib.RSI(data['close'].values, timeperiod=period)
                # Add signal zones
                data['rsi_oversold'] = data[f'talib_rsi_{period}'] < 30
                data['rsi_overbought'] = data[f'talib_rsi_{period}'] > 70
                data['rsi_divergence_bullish'] = (
                    (data[f'talib_rsi_{period}'] > data[f'talib_rsi_{period}'].shift(1)) &
                    (data['close'] < data['close'].shift(1))
                )
                data['rsi_divergence_bearish'] = (
                    (data[f'talib_rsi_{period}'] < data[f'talib_rsi_{period}'].shift(1)) &
                    (data['close'] > data['close'].shift(1))
                )
            except Exception as e:
                print(f"TA-Lib RSI calculation failed: {e}")
                data[f'talib_rsi_{period}'] = np.nan
        else:
            # Fallback to pandas-ta
            data[f'talib_rsi_{period}'] = ta.rsi(data['close'], length=period)
        
        return data
    
    def talib_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD using TA-Lib for enhanced accuracy.
        
        Args:
            df: DataFrame with OHLC data
            fast: Fast EMA period
            slow: Slow EMA period  
            signal: Signal line period
            
        Returns:
            DataFrame with TA-Lib MACD
        """
        data = df.copy()
        
        if TALIB_AVAILABLE and len(data) >= slow:
            try:
                macd_line, macd_signal, macd_hist = talib.MACD(
                    data['close'].values, 
                    fastperiod=fast, 
                    slowperiod=slow, 
                    signalperiod=signal
                )
                data['talib_macd'] = macd_line
                data['talib_macd_signal'] = macd_signal
                data['talib_macd_histogram'] = macd_hist
                
                # Add crossover signals
                data['macd_bullish_cross'] = (
                    (data['talib_macd'] > data['talib_macd_signal']) &
                    (data['talib_macd'].shift(1) <= data['talib_macd_signal'].shift(1))
                )
                data['macd_bearish_cross'] = (
                    (data['talib_macd'] < data['talib_macd_signal']) &
                    (data['talib_macd'].shift(1) >= data['talib_macd_signal'].shift(1))
                )
                data['macd_zero_cross_up'] = (
                    (data['talib_macd'] > 0) & (data['talib_macd'].shift(1) <= 0)
                )
                data['macd_zero_cross_down'] = (
                    (data['talib_macd'] < 0) & (data['talib_macd'].shift(1) >= 0)
                )
            except Exception as e:
                print(f"TA-Lib MACD calculation failed: {e}")
                data['talib_macd'] = np.nan
                data['talib_macd_signal'] = np.nan
                data['talib_macd_histogram'] = np.nan
        else:
            # Fallback to pandas-ta
            macd_result = ta.macd(data['close'], fast=fast, slow=slow, signal=signal)
            if macd_result is not None:
                data['talib_macd'] = macd_result[f'MACD_{fast}_{slow}_{signal}']
                data['talib_macd_signal'] = macd_result[f'MACDs_{fast}_{slow}_{signal}']
                data['talib_macd_histogram'] = macd_result[f'MACDh_{fast}_{slow}_{signal}']
        
        return data
    
    def talib_bbands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands using TA-Lib for enhanced accuracy.
        
        Args:
            df: DataFrame with OHLC data
            period: Moving average period
            std: Standard deviation multiplier
            
        Returns:
            DataFrame with TA-Lib Bollinger Bands
        """
        data = df.copy()
        
        if TALIB_AVAILABLE and len(data) >= period:
            try:
                upper, middle, lower = talib.BBANDS(
                    data['close'].values, 
                    timeperiod=period, 
                    nbdevup=std, 
                    nbdevdn=std, 
                    matype=0
                )
                data['talib_bb_upper'] = upper
                data['talib_bb_middle'] = middle
                data['talib_bb_lower'] = lower
                
                # Convert numpy arrays to pandas Series for proper shift operations
                upper_series = pd.Series(upper, index=data.index)
                middle_series = pd.Series(middle, index=data.index)
                lower_series = pd.Series(lower, index=data.index)
                
                # Add BB signals
                data['bb_squeeze'] = (upper_series - lower_series) / middle_series < 0.1  # Band squeeze
                data['bb_expansion'] = (upper_series - lower_series) / middle_series > 0.15  # Band expansion
                data['bb_upper_touch'] = data['close'] >= upper_series * 0.99
                data['bb_lower_touch'] = data['close'] <= lower_series * 1.01
                data['bb_middle_cross_up'] = (
                    (data['close'] > middle_series) & (data['close'].shift(1) <= middle_series.shift(1))
                )
                data['bb_middle_cross_down'] = (
                    (data['close'] < middle_series) & (data['close'].shift(1) >= middle_series.shift(1))
                )
                
                # %B indicator
                data['bb_percent_b'] = (data['close'] - lower_series) / (upper_series - lower_series)
                
            except Exception as e:
                print(f"TA-Lib Bollinger Bands calculation failed: {e}")
                data['talib_bb_upper'] = np.nan
                data['talib_bb_middle'] = np.nan
                data['talib_bb_lower'] = np.nan
        else:
            # Fallback to pandas-ta
            bb_result = ta.bbands(data['close'], length=period, std=std)
            if bb_result is not None:
                data['talib_bb_upper'] = bb_result[f'BBU_{period}_{std}']
                data['talib_bb_middle'] = bb_result[f'BBM_{period}_{std}']
                data['talib_bb_lower'] = bb_result[f'BBL_{period}_{std}']
        
        return data
    
    def talib_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate ADX using TA-Lib for enhanced accuracy.
        
        Args:
            df: DataFrame with OHLC data
            period: ADX period
            
        Returns:
            DataFrame with TA-Lib ADX
        """
        data = df.copy()
        
        if TALIB_AVAILABLE and len(data) >= period:
            try:
                adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
                plus_di = talib.PLUS_DI(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
                minus_di = talib.MINUS_DI(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
                
                data[f'talib_adx_{period}'] = adx
                data[f'talib_plus_di_{period}'] = plus_di
                data[f'talib_minus_di_{period}'] = minus_di
                
                # Add trend strength classification
                data['adx_trend_weak'] = adx < 25
                data['adx_trend_strong'] = adx > 50
                data['adx_trend_very_strong'] = adx > 75
                
            except Exception as e:
                print(f"TA-Lib ADX calculation failed: {e}")
                data[f'talib_adx_{period}'] = np.nan
                data[f'talib_plus_di_{period}'] = np.nan
                data[f'talib_minus_di_{period}'] = np.nan
        else:
            # Fallback to pandas-ta
            adx_result = ta.adx(data['high'], data['low'], data['close'], length=period)
            if adx_result is not None:
                data[f'talib_adx_{period}'] = adx_result[f'ADX_{period}']
                data[f'talib_plus_di_{period}'] = adx_result[f'DMP_{period}']
                data[f'talib_minus_di_{period}'] = adx_result[f'DMN_{period}']
        
        return data
    
    def get_stochastic_signals(self, df: pd.DataFrame, idx: int) -> str:
        """
        Get stochastic signals - wrapper for compatibility.
        
        Args:
            df: DataFrame with stochastic indicators
            idx: Current index
            
        Returns:
            Signal string
        """
        signal = self.get_stochastic_signal(df, idx)
        if signal.signal_type == "BUY":
            return "BUY"
        elif signal.signal_type == "SELL":
            return "SELL"
        else:
            return "HOLD"