"""
DATA PREPARER - PRICE ACTION + ICT/SMART MONEY
==============================================

Ten moduł przygotowuje cechy (features) dla modelu ML tradingowego.

STRUKTURA CECH (TIERS):
=======================

TIER 0: OHLCV (basic data)
  - open, high, low, close, volume, turnover

TIER 1: Podstawowe wskaźniki techniczne
  - RSI, SMA, VWAP, ATR, Bollinger Bands
  - Volume analysis, Support/Resistance
  - Candlestick patterns

TIER 2: Zaawansowane Price Action
  - Order Flow proxies
  - Volume Profile approximation
  - Market regime detection
  - Mikrostruktura rynku

TIER 3: Wskaźniki kompozytowe
  - Market State Indicator (MSI)
  - Momentum Regime
  - Volume Confirmation Score
  - Multi-Factor Sentiment

TIER 4: ICT & SMART MONEY CONCEPTS ⭐ NOWE! ⭐
  ┌──────────────────────────────────────────────┐
  │  30+ cech wykrywających działania            │
  │  instytucjonalnych traderów:                 │
  │                                              │
  │  • Fair Value Gaps (FVG)                     │
  │  • Liquidity Sweeps                          │
  │  • Order Blocks                              │
  │  • Breaker Blocks                            │
  │  • Market Structure Shifts (MSS)             │
  │  • Institutional Candles                     │
  │  • Liquidity Voids                           │
  │                                              │
  │  COMPOSITE: ict_composite_score              │
  │  (master score - najważniejsza cecha!)       │
  └──────────────────────────────────────────────┘

KLUCZOWE CECHY DLA MODELU (High Importance):
=============================================
1. ict_composite_score          ← MASTER ICT SCORE
2. liquidity_sweep_with_volume  ← Sweep z potwierdzeniem
3. high_conviction_sweep        ← Najsilniejszy sygnał
4. ob_with_fvg                  ← Order Block + FVG
5. market_structure_shift       ← Change of Character

FLOW:
=====
1. fetch_and_prepare_data() 
   ↓
2. _calculate_base_features()    ← Tu dodawane są wszystkie cechy
   ├─ Basic features (1-10)
   ├─ Advanced PA (11-20)
   ├─ Composite indicators (21)
   └─ ICT & Smart Money (22) ⭐ NOWE!
   ↓
3. _add_multi_timeframe_features()
   ↓
4. remove_correlated_features()  ← Usuwa skorelowane, ZACHOWUJE ICT
   ↓
5. Target generation (will_pump_X%)

DOKUMENTACJA:
=============
- Szczegóły ICT: ICT_FEATURES_DOCUMENTATION.md
- Quick start: ICT_UPDATE_README.md
- Testing: test_ict_features.py
- Checklist: ICT_CHECKLIST.md

Author: Łukasz + Claude
Updated: 2025-01-15 (ICT implementation)
Version: 2.0 (with ICT/Smart Money)
"""

import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from dotenv import load_dotenv
from bybit_adapter import BybitAdapter
import asyncio
from scipy.signal import find_peaks
import json
from typing import Tuple, List
import logging

# ============================================================================
# NOWE FUNKCJE: Wskaźniki kompozytowe i usuwanie korelacji
# ============================================================================

def add_oversold_overbought_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prosty binarny wskaźnik oversold/overbought
    
    Zwraca:
    -1: Overbought (RSI > 70, potencjalna korekta w dół)
     0: Neutral (RSI między 30 a 70)
     1: Oversold (RSI < 30, potencjalny odbicie w górę)
    """
    # Inicjalizuj jako 0 (neutral) dla wszystkich wierszy
    signal = pd.Series(0, index=df.index, dtype=np.int8)
    
    if 'rsi_14' in df.columns:
        # Oversold: RSI < 30 -> bullish signal (potencjalny buy)
        signal[df['rsi_14'] < 30] = 1
        # Overbought: RSI > 70 -> bearish signal (potencjalny sell)
        signal[df['rsi_14'] > 70] = -1
        # Wszystkie pozostałe (30 <= RSI <= 70) pozostają jako 0 (neutral)
    
    df['oversold_overbought_signal'] = signal
    return df


def add_market_state_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market State Indicator (MSI) - kompleksowy sygnał stanu rynku
    
    Łączy RSI + Trend + Volume w jeden wskaźnik od -3 do 3:
    -3: Silny niedźwiedzi (oversold extreme)
    -2: Niedźwiedzi oversold
    -1: Słaby niedźwiedzi
     0: Neutralny
     1: Słaby byczy
     2: Byczy overbought
     3: Silny byczy (overbought extreme)
    """
    msi = pd.Series(0, index=df.index)
    
    # RSI component
    rsi_signal = pd.Series(0, index=df.index)
    if 'rsi_14' in df.columns:
        rsi_signal[df['rsi_14'] < 30] = -2  # oversold
        rsi_signal[df['rsi_14'] < 20] = -3  # extreme oversold
        rsi_signal[df['rsi_14'] > 70] = 2   # overbought
        rsi_signal[df['rsi_14'] > 80] = 3   # extreme overbought
    
    # Trend component (using SMAs if available)
    trend_signal = pd.Series(0, index=df.index)
    if 'above_sma_20' in df.columns and 'above_sma_50' in df.columns:
        trend_signal = (df['above_sma_20'] + df['above_sma_50']) - 1  # -1, 0, or 1
    
    # Volume component
    volume_signal = pd.Series(0, index=df.index)
    if 'volume_vs_ma_20' in df.columns:
        volume_signal[df['volume_vs_ma_20'] > 1.5] = 1  # high volume
        volume_signal[df['volume_vs_ma_20'] < 0.7] = -1  # low volume
    
    # Composite MSI
    msi = rsi_signal + trend_signal + volume_signal
    msi = msi.clip(-3, 3)
    
    df['market_state_indicator'] = msi
    return df


def add_momentum_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum Regime - prosty trójstanowy wskaźnik momentum
    
    Łączy momentum z wielu timeframe'ów w jeden sygnał:
    -1: Bearish momentum (multiple timeframes negative)
     0: Neutral/Choppy
     1: Bullish momentum (multiple timeframes positive)
    """
    momentum_score = pd.Series(0, index=df.index)
    
    # Short-term momentum
    if 'price_change_pct_5' in df.columns:
        momentum_score += np.sign(df['price_change_pct_5'])
    
    # Medium-term momentum
    if 'roc_10' in df.columns:
        momentum_score += np.sign(df['roc_10'])
    
    # RSI momentum direction
    if 'rsi_14' in df.columns:
        momentum_score += np.sign(df['rsi_14'] - 50)
    
    # Normalize to [-1, 0, 1]
    momentum_regime = pd.Series(0, index=df.index)
    momentum_regime[momentum_score <= -2] = -1
    momentum_regime[momentum_score >= 2] = 1
    
    df['momentum_regime'] = momentum_regime
    return df


def add_volume_confirmation_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume Confirmation Score - czy wolumen potwierdza ruch cenowy
    
    Zwraca:
    -1: Volume confirms downward move
     0: No clear volume confirmation
     1: Volume confirms upward move
    """
    vcs = pd.Series(0, index=df.index)
    
    # Price direction
    price_up = (df['close'] > df['open']).astype(int)
    price_down = (df['close'] < df['open']).astype(int)
    
    # High volume condition
    high_volume = False
    if 'volume_vs_ma_20' in df.columns:
        high_volume = (df['volume_vs_ma_20'] > 1.2)
    
    # Confirmation logic
    vcs[(price_up == 1) & high_volume] = 1   # Bullish with volume
    vcs[(price_down == 1) & high_volume] = -1  # Bearish with volume
    
    df['volume_confirmation_score'] = vcs
    return df


def add_multi_factor_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-Factor Sentiment Score - agreguje wiele sygnałów w jeden
    
    Wartość od -5 do 5:
    < -3: Strong bearish
    -3 to -1: Bearish
    -1 to 1: Neutral
    1 to 3: Bullish
    > 3: Strong bullish
    """
    sentiment = pd.Series(0.0, index=df.index)
    
    # RSI component
    if 'rsi_14' in df.columns:
        rsi_score = (df['rsi_14'] - 50) / 50  # Normalize to [-1, 1]
        sentiment += rsi_score
    
    # Momentum component
    if 'momentum_regime' in df.columns:
        sentiment += df['momentum_regime']
    
    # Volume component
    if 'volume_confirmation_score' in df.columns:
        sentiment += df['volume_confirmation_score']
    
    # Trend component
    if 'above_sma_20' in df.columns and 'above_sma_50' in df.columns:
        trend_score = (df['above_sma_20'] + df['above_sma_50']) - 1
        sentiment += trend_score
    
    df['multi_factor_sentiment'] = sentiment.clip(-5, 5)
    return df


def remove_correlated_features(df: pd.DataFrame, 
                               target_col: str = None,
                               correlation_threshold: float = 0.85,
                               keep_important: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Usuwa cechy silnie skorelowane ze sobą
    
    Args:
        df: DataFrame z cechami
        target_col: Nazwa kolumny targetu (zostanie pominięta w analizie)
        correlation_threshold: Próg korelacji powyżej którego usuwamy cechy (domyślnie 0.85)
        keep_important: Lista nazw cech które zawsze zachowujemy
    
    Returns:
        Tuple[DataFrame, List[str]]: DataFrame z usuniętymi cechami, lista usuniętych cech
    """
    print(f"\n{'='*60}")
    print("ANALIZA KORELACJI CECH")
    print(f"{'='*60}")
    print(f"Próg korelacji: {correlation_threshold}")
    print(f"Początkowa liczba cech: {df.shape[1]}")
    
    # Domyślna lista ważnych cech
    if keep_important is None:
        keep_important = [
            # Podstawowe wskaźniki (klasyczne)
            'rsi_14', 'volume_vs_ma_20', 'dist_from_vwap', 'atr_normalized',
            
            # Wskaźniki kompozytowe (już istniejące)
            'market_state_indicator', 'momentum_regime', 'volume_confirmation_score',
            'multi_factor_sentiment', 'oversold_overbought_signal',
            
            # ====================================================================
            # ICT & SMART MONEY - NAJWYŻSZY PRIORYTET (nowe)
            # ====================================================================
            'ict_composite_score',           # Master ICT score - KLUCZOWA CECHA
            'fvg_signal',                    # Fair Value Gaps
            'fvg_size',                      # Wielkość FVG
            'liquidity_sweep',               # Liquidity Sweeps - bardzo ważne
            'liquidity_sweep_strength',      # Siła sweep
            'order_block',                   # Order Blocks - gdzie smart money
            'order_block_strength',          # Siła OB
            'breaker_block',                 # Breaker Blocks
            'market_structure_shift',        # MSS - Change of Character
            'market_structure_direction',    # Kierunek struktury
            'institutional_candle',          # Świece instytucjonalne
            'institutional_candle_strength', # Siła inst. candle
            
            # Smart Money Context Features (kombinacje)
            'ob_with_fvg',                   # OB + FVG = silny sygnał
            'high_conviction_sweep',         # Sweep + volume
            'structure_aligned_ob',          # OB aligned z trendem
            'fvg_fill_reversal',             # FVG fill + reversal
        ]
    
    # Cechy numeryczne (bez targetu)
    cols_to_analyze = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in cols_to_analyze:
        cols_to_analyze.remove(target_col)
    
    # Macierz korelacji
    print(f"Obliczanie macierzy korelacji dla {len(cols_to_analyze)} cech...")
    corr_matrix = df[cols_to_analyze].corr().abs()
    
    # Górny trójkąt
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Znajdź pary o wysokiej korelacji
    to_drop = set()
    high_corr_pairs = []
    
    for column in upper_triangle.columns:
        correlated_features = upper_triangle.index[
            upper_triangle[column] > correlation_threshold
        ].tolist()
        
        if correlated_features:
            for corr_feature in correlated_features:
                high_corr_pairs.append({
                    'feature1': column,
                    'feature2': corr_feature,
                    'correlation': upper_triangle.loc[corr_feature, column]
                })
                
                # Decyzja którą cechę usunąć
                # PRIORITY 1: Never drop features in keep_important
                if column in keep_important and corr_feature in keep_important:
                    # Both are important - keep both (don't add either to drop list)
                    continue
                elif column in keep_important and corr_feature not in keep_important:
                    to_drop.add(corr_feature)
                elif corr_feature in keep_important and column not in keep_important:
                    to_drop.add(column)
                else:
                    # Neither is important - drop the one with lower variance
                    var_col = df[column].var()
                    var_corr = df[corr_feature].var()
                    if var_col < var_corr:
                        to_drop.add(column)
                    else:
                        to_drop.add(corr_feature)
    
    # Raport
    print(f"\nZnaleziono {len(high_corr_pairs)} par cech o korelacji > {correlation_threshold}")
    
    if high_corr_pairs:
        print("\nTop 10 najwyższych korelacji:")
        sorted_pairs = sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)
        for pair in sorted_pairs[:10]:
            print(f"  {pair['feature1']:40s} <-> {pair['feature2']:40s} : {pair['correlation']:.3f}")
    
    to_drop_list = list(to_drop)
    print(f"\nUsuwam {len(to_drop_list)} skorelowanych cech")
    
    if to_drop_list:
        print("\nPrzykładowe usunięte cechy (max 20):")
        for feature in sorted(to_drop_list)[:20]:
            print(f"  - {feature}")
        if len(to_drop_list) > 20:
            print(f"  ... i {len(to_drop_list) - 20} więcej")
    
    # Usuń
    df_cleaned = df.drop(columns=to_drop_list, errors='ignore')
    
    print(f"\nKońcowa liczba cech: {df_cleaned.shape[1]}")
    print(f"Usunięto: {len(to_drop_list)} cech ({len(to_drop_list)/len(cols_to_analyze)*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return df_cleaned, to_drop_list


# ============================================================================
# ICT & SMART MONEY CONCEPTS
# ============================================================================

def detect_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fair Value Gaps (FVG) - ICT Core Concept
    
    FVG Bullish: gdy low[i] > high[i-2] (gap w górę - brak ceny między świecami)
    FVG Bearish: gdy high[i] < low[i-2] (gap w dół - brak ceny między świecami)
    
    Te gap'y często działają jak magnesy - cena wraca aby je wypełnić.
    Smart Money używa ich jako entry zones.
    """
    # Bullish FVG: obecny low > high sprzed 2 świec (zwróć boolean Series!)
    fvg_bullish = (df['low'] > df['high'].shift(2))
    
    # Bearish FVG: obecny high < low sprzed 2 świec (zwróć boolean Series!)
    fvg_bearish = (df['high'] < df['low'].shift(2))
    
    # Signal: 1 = bullish FVG, -1 = bearish FVG, 0 = brak
    df['fvg_signal'] = fvg_bullish.astype(int) - fvg_bearish.astype(int)
    
    # Rozmiar gap'u (znormalizowany przez cenę)
    df['fvg_size'] = np.where(
        fvg_bullish, 
        (df['low'] - df['high'].shift(2)) / df['close'],
        np.where(
            fvg_bearish, 
            (df['low'].shift(2) - df['high']) / df['close'], 
            0
        )
    )
    
    # Czy FVG został wypełniony w ostatnich N świecach
    df['fvg_filled'] = 0
    for i in range(1, 6):  # sprawdź ostatnie 5 świec
        # Bullish FVG wypełniony jeśli cena wróciła w dół
        # POPRAWKA: użyj boolean Series przed operatorem &
        filled_bull = fvg_bullish.shift(i) & (df['low'] <= df['high'].shift(i+2))
        # Bearish FVG wypełniony jeśli cena wróciła w górę
        filled_bear = fvg_bearish.shift(i) & (df['high'] >= df['low'].shift(i+2))
        df['fvg_filled'] += (filled_bull.astype(int) - filled_bear.astype(int))
    
    return df


def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Liquidity Sweeps - ICT Core Concept
    
    Smart Money często "sweepuje" (zbiera) płynność z retail stop lossów
    umieszczonych za oczywistymi high/low, a następnie reversal.
    
    Bullish Sweep: cena bierze recent low + natychmiastowy reversal w górę
    Bearish Sweep: cena bierze recent high + natychmiastowy reversal w dół
    """
    # Znajdź recent extreme levels
    recent_high = df['high'].rolling(lookback).max()
    recent_low = df['low'].rolling(lookback).min()
    
    # Bullish Liquidity Sweep: 
    # - cena bierze recent low (sweep stop lossów)
    # - świeca zamyka się wyżej (reversal)
    sweep_low = (
        (df['low'] <= recent_low.shift(1)) &  # zbiera płynność
        (df['close'] > df['open']) &           # bullish reversal candle
        (df['close'] > recent_low.shift(1))    # zamyka powyżej swept level
    )
    
    # Bearish Liquidity Sweep:
    # - cena bierze recent high (sweep stop lossów)
    # - świeca zamyka się niżej (reversal)
    sweep_high = (
        (df['high'] >= recent_high.shift(1)) &  # zbiera płynność
        (df['close'] < df['open']) &             # bearish reversal candle
        (df['close'] < recent_high.shift(1))     # zamyka poniżej swept level
    )
    
    df['liquidity_sweep'] = sweep_low.astype(int) - sweep_high.astype(int)
    
    # Siła sweep'u - jak daleko cena poszła poza level
    df['liquidity_sweep_strength'] = np.where(
        sweep_low,
        (recent_low.shift(1) - df['low']) / df['close'],
        np.where(
            sweep_high,
            (df['high'] - recent_high.shift(1)) / df['close'],
            0
        )
    )
    
    # Czy sweep miał wysokie volume (bardziej wiarygodny)
    if 'volume_vs_ma_20' in df.columns:
        df['liquidity_sweep_with_volume'] = (
            (df['liquidity_sweep'] != 0) & 
            (df['volume_vs_ma_20'] > 1.3)
        ).astype(int) * df['liquidity_sweep']
    
    return df


def detect_order_blocks(df: pd.DataFrame, impulse_threshold: float = 0.015) -> pd.DataFrame:
    """
    Order Blocks (OB) - ICT Core Concept
    
    Order Block = ostatnia przeciwna świeca przed silnym impulsem.
    To miejsce gdzie Smart Money złożyło duże zlecenia.
    
    Bullish OB: ostatnia bearish świeca przed silnym ruchem w górę
    Bearish OB: ostatnia bullish świeca przed silnym ruchem w dół
    """
    # Wykryj silne impulsy (>1.5% w 3 świece)
    price_change_3 = df['close'].pct_change(3)
    bullish_impulse = (price_change_3 > impulse_threshold)
    bearish_impulse = (price_change_3 < -impulse_threshold)
    
    # Sprawdź czy poprzednia świeca była przeciwna
    bearish_candle = (df['close'] < df['open'])
    bullish_candle = (df['close'] > df['open'])
    
    # Bullish Order Block: bearish świeca przed bullish impulsem
    bullish_ob = bullish_impulse & bearish_candle.shift(1)
    
    # Bearish Order Block: bullish świeca przed bearish impulsem
    bearish_ob = bearish_impulse & bullish_candle.shift(1)
    
    df['order_block'] = bullish_ob.astype(int) - bearish_ob.astype(int)
    
    # Siła Order Block = wielkość impulsu
    df['order_block_strength'] = np.where(
        bullish_ob,
        price_change_3,
        np.where(bearish_ob, -price_change_3, 0)
    )
    
    # Dystans do Order Block (jak daleko jesteśmy od ostatniego OB)
    last_bullish_ob_price = df['close'].where(bullish_ob).ffill()
    last_bearish_ob_price = df['close'].where(bearish_ob).ffill()
    
    df['dist_from_bullish_ob'] = (df['close'] - last_bullish_ob_price) / df['close']
    df['dist_from_bearish_ob'] = (df['close'] - last_bearish_ob_price) / df['close']
    
    # Czy testujemy Order Block (w zasięgu ±0.5%)
    df['testing_bullish_ob'] = (df['dist_from_bullish_ob'].abs() < 0.005).astype(int)
    df['testing_bearish_ob'] = (df['dist_from_bearish_ob'].abs() < 0.005).astype(int)
    
    return df


def detect_breaker_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Breaker Blocks - ICT Advanced Concept
    
    Breaker = Order Block który został przebity i zmienił swoją rolę.
    Support który został złamany staje się resistance (i odwrotnie).
    
    To sign of "Change of Character" - zmiana struktury rynku.
    """
    # Wykryj czy poprzedni support został przebity
    support_level = df['low'].rolling(10).min()
    support_broken = (
        (df['low'].shift(3) == support_level.shift(3)) &  # był support
        (df['close'] < support_level.shift(3)) &           # został przebity w dół
        (df['close'].shift(1) >= support_level.shift(3))   # dopiero co przebiliśmy
    )
    
    # Wykryj czy poprzedni resistance został przebity
    resistance_level = df['high'].rolling(10).max()
    resistance_broken = (
        (df['high'].shift(3) == resistance_level.shift(3)) &  # był resistance
        (df['close'] > resistance_level.shift(3)) &            # został przebity w górę
        (df['close'].shift(1) <= resistance_level.shift(3))    # dopiero co przebiliśmy
    )
    
    # Breaker signal: 1 = bullish break, -1 = bearish break
    df['breaker_block'] = resistance_broken.astype(int) - support_broken.astype(int)
    
    # Siła breakout (jak daleko poszliśmy poza level)
    df['breaker_strength'] = np.where(
        resistance_broken,
        (df['close'] - resistance_level.shift(3)) / df['close'],
        np.where(
            support_broken,
            (support_level.shift(3) - df['close']) / df['close'],
            0
        )
    )
    
    return df


def detect_market_structure_shift(df: pd.DataFrame, swing_period: int = 10) -> pd.DataFrame:
    """
    Market Structure Shift (MSS) - ICT Core Concept
    
    MSS = Change of Character - moment gdy struktura rynku się zmienia:
    - Bullish: higher highs & higher lows
    - Bearish: lower highs & lower lows
    
    MSS występuje gdy ta struktura zostaje złamana.
    """
    # Znajdź swing points (lokalne ekstrema)
    # Swing High = highest high w oknie
    highs_window = df['high'].rolling(window=swing_period*2+1, center=True).max()
    is_swing_high = (df['high'] == highs_window)
    
    # Swing Low = lowest low w oknie
    lows_window = df['low'].rolling(window=swing_period*2+1, center=True).min()
    is_swing_low = (df['low'] == lows_window)
    
    # Track ostatniego swing high/low
    last_swing_high = df['high'].where(is_swing_high).ffill()
    last_swing_low = df['low'].where(is_swing_low).ffill()
    
    # Bullish MSS: przebiliśmy ostatni swing high (breaking resistance)
    bullish_mss = (df['close'] > last_swing_high.shift(1)) & (last_swing_high.shift(1).notna())
    
    # Bearish MSS: przebiliśmy ostatni swing low (breaking support)
    bearish_mss = (df['close'] < last_swing_low.shift(1)) & (last_swing_low.shift(1).notna())
    
    df['market_structure_shift'] = bullish_mss.astype(int) - bearish_mss.astype(int)
    
    # Jak dawno był ostatni MSS (0 = właśnie teraz, rośnie z czasem)
    mss_occurred = (df['market_structure_shift'] != 0)
    df['bars_since_mss'] = (~mss_occurred).cumsum() - (~mss_occurred).cumsum().where(~mss_occurred).ffill().fillna(0)
    
    # Kierunek obecnej struktury (1 = uptrend, -1 = downtrend, 0 = neutral)
    df['market_structure_direction'] = df['market_structure_shift'].replace(0, np.nan).ffill().fillna(0)
    
    return df


def detect_institutional_candles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Institutional Candles - Smart Money Detection
    
    Duże świece z wysokim volume = prawdopodobnie działanie smart money.
    
    Kryteria:
    1. Range > 2x średnia (duża świeca)
    2. Volume > 1.5x średnia (wysokie zainteresowanie)
    3. Close w górnej/dolnej części range (pokazuje dominację)
    """
    # Wielkość świecy vs średnia
    candle_range = df['high'] - df['low']
    avg_range = candle_range.rolling(50).mean()
    large_range = (candle_range > avg_range * 2)
    
    # Wysoki volume
    high_volume = (df['volume'] > df['volume'].rolling(50).mean() * 1.5)
    
    # Pozycja close w range świecy
    close_position = (df['close'] - df['low']) / (candle_range + 1e-8)
    
    # Bullish institutional: duża świeca, high vol, close w górnych 30%
    close_in_upper = (close_position > 0.7)
    bullish_institutional = (large_range & high_volume & close_in_upper)
    
    # Bearish institutional: duża świeca, high vol, close w dolnych 30%
    close_in_lower = (close_position < 0.3)
    bearish_institutional = (large_range & high_volume & close_in_lower)
    
    df['institutional_candle'] = (
        bullish_institutional.astype(int) - bearish_institutional.astype(int)
    )
    
    # Siła świecy instytucjonalnej
    df['institutional_candle_strength'] = np.where(
        bullish_institutional | bearish_institutional,
        (candle_range / avg_range) * (df['volume'] / df['volume'].rolling(50).mean()),
        0
    )
    
    return df


def detect_liquidity_voids(df: pd.DataFrame, volume_threshold: float = 0.5) -> pd.DataFrame:
    """
    Liquidity Voids - ICT Concept
    
    Obszary z bardzo niskim volume = brak płynności.
    Cena często szybko przechodzi przez te obszary (swift move).
    
    Retail lubi handlować w tych obszarach, Smart Money ich unika.
    """
    # Nisko-wolumenowe świece (< 50% średniej)
    avg_volume = df['volume'].rolling(50).mean()
    low_volume = (df['volume'] < avg_volume * volume_threshold)
    
    # Zlicz ile ostatnich świec miało niski volume
    df['liquidity_void_depth'] = low_volume.astype(int).rolling(5).sum()
    
    # Czy jesteśmy w strefie liquidity void (3+ świece z niskim volume)
    df['in_liquidity_void'] = (df['liquidity_void_depth'] >= 3).astype(int)
    
    # Siła void (jak niskie było volume)
    df['liquidity_void_strength'] = np.where(
        low_volume,
        1 - (df['volume'] / avg_volume),
        0
    )
    
    return df


def add_ict_smart_money_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    MASTER FUNCTION: Dodaje wszystkie ICT & Smart Money Features
    
    Te cechy mają WYSOKĄ WAGĘ dla modelu bo reprezentują działania
    smart money i institutional traders.
    """
    print("  === ICT & SMART MONEY CONCEPTS ===")
    
    print("    → Fair Value Gaps (FVG)...")
    df = detect_fair_value_gaps(df)
    
    print("    → Liquidity Sweeps...")
    df = detect_liquidity_sweeps(df, lookback=20)
    
    print("    → Order Blocks...")
    df = detect_order_blocks(df, impulse_threshold=0.015)
    
    print("    → Breaker Blocks...")
    df = detect_breaker_blocks(df)
    
    print("    → Market Structure Shifts (MSS)...")
    df = detect_market_structure_shift(df, swing_period=10)
    
    print("    → Institutional Candles...")
    df = detect_institutional_candles(df)
    
    print("    → Liquidity Voids...")
    df = detect_liquidity_voids(df, volume_threshold=0.5)
    
    # ========================================================================
    # COMPOSITE ICT SCORE - agregacja wszystkich sygnałów ICT
    # To jest KLUCZOWA cecha - model powinien jej dać dużą wagę
    # ========================================================================
    print("    → Composite ICT Score (HIGH IMPORTANCE)...")
    
    df['ict_composite_score'] = (
        df['fvg_signal'] * 0.15 +                    # FVG jako entry zones
        df['liquidity_sweep'] * 0.25 +               # Sweeps = silny sygnał
        df['order_block'] * 0.20 +                   # OB = gdzie smart money siedzi
        df['breaker_block'] * 0.15 +                 # Breaker = zmiana charakteru
        df['market_structure_shift'] * 0.20 +        # MSS = change of trend
        df['institutional_candle'] * 0.05            # Potwierdzenie przez volume
    )
    
    # Znormalizuj do zakresu [-1, 1] dla łatwiejszej interpretacji
    max_abs = df['ict_composite_score'].abs().max()
    if max_abs > 0:
        df['ict_composite_score'] = df['ict_composite_score'] / max_abs
    
    # ========================================================================
    # SMART MONEY CONTEXT FEATURES - dodatkowe cechy kontekstowe
    # ========================================================================
    print("    → Smart Money Context Features...")
    
    # Czy jesteśmy przy Order Block + FVG (bardzo silny sygnał)
    df['ob_with_fvg'] = (
        ((df['testing_bullish_ob'] == 1) & (df['fvg_signal'] == 1)) |
        ((df['testing_bearish_ob'] == 1) & (df['fvg_signal'] == -1))
    ).astype(int)
    
    # Liquidity sweep + institutional volume (highest conviction)
    if 'liquidity_sweep_with_volume' in df.columns:
        df['high_conviction_sweep'] = (
            (df['liquidity_sweep_with_volume'] != 0) & 
            (df['institutional_candle'] != 0)
        ).astype(int) * np.sign(df['liquidity_sweep_with_volume'])
    
    # Market structure aligned with OB (trend confirmation)
    df['structure_aligned_ob'] = (
        ((df['market_structure_direction'] == 1) & (df['order_block'] == 1)) |
        ((df['market_structure_direction'] == -1) & (df['order_block'] == -1))
    ).astype(int)
    
    # FVG being filled = potential reversal point
    df['fvg_fill_reversal'] = (
        (df['fvg_filled'] != 0) & 
        (df['liquidity_sweep'] != 0)
    ).astype(int)
    
    print("    ✓ ICT/Smart Money: 30+ nowych cech dodanych")
    
    return df


# ============================================================================
# KONIEC NOWYCH FUNKCJI
# ============================================================================

def find_hidden_divergence(prices: pd.Series, indicator: pd.Series, lookback: int) -> pd.Series:
    """
    Znajduje ukrytą byczą i niedźwiedzią dywergencję.
    Zwraca: 1 dla byczej, -1 dla niedźwiedziej, 0 w pozostałych przypadkach.
    """
    price_peaks, _ = find_peaks(prices, distance=5)
    price_troughs, _ = find_peaks(-prices, distance=5)
    ind_peaks, _ = find_peaks(indicator, distance=5)
    ind_troughs, _ = find_peaks(-indicator, distance=5)

    signals = pd.Series(0, index=prices.index)

    # Ukryta dywergencja niedźwiedzia (niższy szczyt ceny, wyższy szczyt wskaźnika)
    for i in range(1, len(price_peaks)):
        for j in range(max(0, i - lookback), i):
            if prices.iloc[price_peaks[i]] < prices.iloc[price_peaks[j]]:
                corresponding_ind_peak_i = ind_peaks[(ind_peaks >= price_peaks[i]-2) & (ind_peaks <= price_peaks[i]+2)]
                corresponding_ind_peak_j = ind_peaks[(ind_peaks >= price_peaks[j]-2) & (ind_peaks <= price_peaks[j]+2)]
                if len(corresponding_ind_peak_i) > 0 and len(corresponding_ind_peak_j) > 0:
                    if indicator.iloc[corresponding_ind_peak_i[0]] > indicator.iloc[corresponding_ind_peak_j[0]]:
                        signals.iloc[price_peaks[i]] = -1
                        break

    # Ukryta dywergencja bycza (wyższy dołek ceny, niższy dołek wskaźnika)
    for i in range(1, len(price_troughs)):
        for j in range(max(0, i - lookback), i):
            if prices.iloc[price_troughs[i]] > prices.iloc[price_troughs[j]]:
                corresponding_ind_trough_i = ind_troughs[(ind_troughs >= price_troughs[i]-2) & (ind_troughs <= price_troughs[i]+2)]
                corresponding_ind_trough_j = ind_troughs[(ind_troughs >= price_troughs[j]-2) & (ind_troughs <= price_troughs[j]+2)]
                if len(corresponding_ind_trough_i) > 0 and len(corresponding_ind_trough_j) > 0:
                    if indicator.iloc[corresponding_ind_trough_i[0]] < indicator.iloc[corresponding_ind_trough_j[0]]:
                        signals.iloc[price_troughs[i]] = 1
                        break
    return signals


def _detect_engulfing(df: pd.DataFrame) -> pd.Series:
    """Detects bullish and bearish engulfing patterns."""
    engulfing = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        prev_open = df['open'].iloc[i-1]
        prev_close = df['close'].iloc[i-1]
        curr_open = df['open'].iloc[i]
        curr_close = df['close'].iloc[i]

        # Bullish engulfing: previous bearish, current bullish and larger
        if prev_close < prev_open and curr_close > curr_open:
            if curr_open <= prev_close and curr_close >= prev_open:
                engulfing.iloc[i] = 1
        # Bearish engulfing: previous bullish, current bearish and larger
        elif prev_close > prev_open and curr_close < curr_open:
            if curr_open >= prev_close and curr_close <= prev_open:
                engulfing.iloc[i] = -1

    return engulfing


def _detect_hammer(df: pd.DataFrame) -> pd.Series:
    """Detects hammer and inverted hammer patterns."""
    hammer = pd.Series(0, index=df.index)

    for i in range(len(df)):
        open_price = df['open'].iloc[i]
        close_price = df['close'].iloc[i]
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]

        body = abs(close_price - open_price)
        lower_wick = min(open_price, close_price) - low_price
        upper_wick = high_price - max(open_price, close_price)
        total_range = high_price - low_price

        if total_range > 0:
            body_pct = body / total_range
            lower_wick_pct = lower_wick / total_range
            upper_wick_pct = upper_wick / total_range

            # Hammer: small body at top, long lower wick
            if body_pct < 0.3 and lower_wick_pct > 2 * body_pct and upper_wick_pct < body_pct:
                hammer.iloc[i] = 1
            # Inverted hammer: small body at bottom, long upper wick
            elif body_pct < 0.3 and upper_wick_pct > 2 * body_pct and lower_wick_pct < body_pct:
                hammer.iloc[i] = -1

    return hammer


def _detect_doji(df: pd.DataFrame) -> pd.Series:
    """Detects doji patterns."""
    doji = pd.Series(0, index=df.index)

    for i in range(len(df)):
        open_price = df['open'].iloc[i]
        close_price = df['close'].iloc[i]
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]

        body = abs(close_price - open_price)
        total_range = high_price - low_price

        if total_range > 0:
            body_pct = body / total_range
            # Doji: very small or no body, wicks roughly equal
            if body_pct < 0.1:
                doji.iloc[i] = 1

    return doji


def _detect_three_line_strike(df: pd.DataFrame) -> pd.Series:
    """Three Line Strike pattern - strong reversal signal (TIER 3)."""
    pattern = pd.Series(0, index=df.index)
    
    for i in range(3, len(df)):
        # Bullish three line strike: 3 bearish candles + 1 large bullish
        if all(df['close'].iloc[i-j] < df['open'].iloc[i-j] for j in range(1, 4)):
            if (df['close'].iloc[i] > df['open'].iloc[i] and 
                df['close'].iloc[i] > df['open'].iloc[i-3]):
                pattern.iloc[i] = 1
        
        # Bearish three line strike
        elif all(df['close'].iloc[i-j] > df['open'].iloc[i-j] for j in range(1, 4)):
            if (df['close'].iloc[i] < df['open'].iloc[i] and 
                df['close'].iloc[i] < df['open'].iloc[i-3]):
                pattern.iloc[i] = -1
    
    return pattern


def _detect_morning_evening_star(df: pd.DataFrame) -> pd.Series:
    """Morning/Evening Star patterns - reversal indicators (TIER 3)."""
    pattern = pd.Series(0, index=df.index)
    
    for i in range(2, len(df)):
        body_0 = abs(df['close'].iloc[i-2] - df['open'].iloc[i-2])
        body_1 = abs(df['close'].iloc[i-1] - df['open'].iloc[i-1])
        body_2 = abs(df['close'].iloc[i] - df['open'].iloc[i])
        
        # Morning star (bullish reversal)
        if (df['close'].iloc[i-2] < df['open'].iloc[i-2] and  # bearish
            body_1 < body_0 * 0.3 and  # small middle candle
            df['close'].iloc[i] > df['open'].iloc[i] and  # bullish
            body_2 > body_0 * 0.5):  # strong bullish
            pattern.iloc[i] = 1
        
        # Evening star (bearish reversal)
        elif (df['close'].iloc[i-2] > df['open'].iloc[i-2] and  # bullish
              body_1 < body_0 * 0.3 and  # small middle
              df['close'].iloc[i] < df['open'].iloc[i] and  # bearish
              body_2 > body_0 * 0.5):  # strong bearish
            pattern.iloc[i] = -1
    
    return pattern


def _calculate_base_features(df_out: pd.DataFrame):
    print("Obliczanie pełnego zestawu cech dla interwału bazowego...")
    SWING_WINDOW, VOLUME_MA_WINDOW, BBANDS_LEN, BBANDS_STD = 50, 20, 20, 2

    print("  1. Struktura rynku...")
    swing_high, swing_low = df_out['high'].rolling(window=SWING_WINDOW).max(), df_out['low'].rolling(
        window=SWING_WINDOW).min()
    df_out[f'dist_from_swing_high_{SWING_WINDOW}'] = (df_out['close'] - swing_high) / swing_high
    df_out[f'dist_from_swing_low_{SWING_WINDOW}'] = (df_out['close'] - swing_low) / swing_low
    df_daily = df_out.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    pivot_p = (df_daily['high'].shift(1) + df_daily['low'].shift(1) + df_daily['close'].shift(1)) / 3
    pivot_r1, pivot_s1 = (2 * pivot_p) - df_daily['low'].shift(1), (2 * pivot_p) - df_daily['high'].shift(1)
    pivots = pd.DataFrame({'pivot': pivot_p, 'r1': pivot_r1, 's1': pivot_s1})
    df_out.sort_index(inplace=True)
    pivots.sort_index(inplace=True)
    df_out = pd.merge_asof(df_out, pivots, left_index=True, right_index=True, direction='backward')

    df_out.drop(columns=['pivot'], inplace=True, errors='ignore')
    df_out['dist_from_s1'] = (df_out['close'] - df_out['s1']) / df_out['s1']
    df_out['dist_from_r1'] = (df_out['close'] - df_out['r1']) / df_out['r1']

    print("  2. Analiza wolumenu...")
    df_out[f'volume_vs_ma_{VOLUME_MA_WINDOW}'] = df_out['volume'] / df_out['volume'].rolling(
        window=VOLUME_MA_WINDOW).mean()
    up_volume, down_volume = df_out['volume'].where(df_out['close'] > df_out['open'], 0), df_out['volume'].where(
        df_out['close'] < df_out['open'], 0)
    df_out['rvol_ratio'] = up_volume.rolling(window=50).sum() / (down_volume.rolling(window=50).sum() + 1e-5)

    print("  3. Cechy świecowe...")
    body_size, wick_size = abs(df_out['close'] - df_out['open']), df_out['high'] - df_out['low']
    df_out['body_to_wick_ratio'] = body_size / wick_size
    df_out['upper_wick_size'] = (df_out['high'] - np.maximum(df_out['open'], df_out['close'])) / wick_size
    df_out['lower_wick_size'] = (np.minimum(df_out['open'], df_out['close']) - df_out['low']) / wick_size

    print("  4. Zmienność i prędkość...")
    df_out['atr_normalized'] = df_out.ta.atr(length=14) / df_out['close']
    df_out['price_change_pct_5'] = df_out['close'].pct_change(periods=5)
    bbands = df_out.ta.bbands(length=BBANDS_LEN, std=BBANDS_STD)

    bbu_col, bbl_col, bbm_col = None, None, None
    if bbands is not None and not bbands.empty:
        for col in bbands.columns:
            if 'BBU' in col:
                bbu_col = col
            elif 'BBL' in col:
                bbl_col = col
            elif 'BBM' in col:
                bbm_col = col

    if all([bbu_col, bbl_col, bbm_col]):
        bbw = (bbands[bbu_col] - bbands[bbl_col]) / bbands[bbm_col]
        df_out['bbw'], df_out['bbw_squeeze'] = bbw, bbw / bbw.rolling(window=100).mean()

    print("  5. Cechy oparte o VWAP...")
    df_out['vwap'] = ta.vwap(high=df_out['high'], low=df_out['low'], close=df_out['close'], volume=df_out['volume'])
    df_out['dist_from_vwap'] = (df_out['close'] - df_out['vwap']) / df_out['vwap']

    print("  6. Momentum i sygnały odwrócenia...")

    rsi = df_out.ta.rsi(length=14)
    df_out['rsi_14'] = rsi

    df_out['hidden_divergence'] = find_hidden_divergence(df_out['close'], rsi, lookback=60)

    df_out['engulfing'] = _detect_engulfing(df_out)
    df_out['hammer'] = _detect_hammer(df_out)
    df_out['doji'] = _detect_doji(df_out)
    
    # OPTIMIZATION: Additional candlestick patterns for better price action signals
    df_out['three_white_soldiers'] = (
        (df_out['close'] > df_out['open']) &
        (df_out['close'].shift(1) > df_out['open'].shift(1)) &
        (df_out['close'].shift(2) > df_out['open'].shift(2)) &
        (df_out['close'] > df_out['close'].shift(1)) &
        (df_out['close'].shift(1) > df_out['close'].shift(2))
    ).astype(int)
    
    # OPTIMIZATION: Price structure features - trend quality indicators
    print("  6B. Price structure analysis (higher highs/lows)...")
    window = 20
    higher_highs = pd.Series(0, index=df_out.index)
    higher_lows = pd.Series(0, index=df_out.index)
    for i in range(window):
        higher_highs += (df_out['high'].shift(i) > df_out['high'].shift(i+1)).astype(int)
        higher_lows += (df_out['low'].shift(i) > df_out['low'].shift(i+1)).astype(int)
    
    df_out['higher_highs_count_20'] = higher_highs
    df_out['higher_lows_count_20'] = higher_lows
    df_out['trend_structure_score'] = ((higher_highs + higher_lows) / (2 * window)) * 2 - 1

    print("  7. Cechy mikrostruktury rynku...")
    # Order Flow Proxy (bez orderbook)
    df_out['price_volume_trend'] = ((df_out['close'] - df_out['close'].shift(1)) / 
                                     df_out['close'].shift(1)) * df_out['volume']
    df_out['cumulative_delta'] = df_out['price_volume_trend'].rolling(window=20).sum()
    
    # Volume-Weighted Price Momentum
    df_out['vwap_momentum'] = (df_out['close'] - df_out['vwap'].rolling(window=10).mean()) / df_out['vwap']
    
    # Tick direction proxy
    df_out['tick_direction'] = np.sign(df_out['close'] - df_out['close'].shift(1))
    df_out['tick_persistence'] = df_out['tick_direction'].rolling(window=5).sum()

    print("  8. Zaawansowane cechy wolumenu...")
    # Volume Profile Approximation
    df_out['volume_ma_ratio_short'] = df_out['volume'] / df_out['volume'].rolling(5).mean()
    df_out['volume_ma_ratio_long'] = df_out['volume'] / df_out['volume'].rolling(50).mean()
    
    # Volume Acceleration
    df_out['volume_acceleration'] = df_out['volume'].diff() / df_out['volume'].shift(1)
    
    # Turnover-based features (wykorzystaj kolumnę 'turnover')
    # POPRAWKA: avg_trade_size powinien być ZNORMALIZOWANY względem ceny
    # aby nie był skorelowany z surową ceną
    typical_price = (df_out['high'] + df_out['low'] + df_out['close']) / 3
    df_out['avg_trade_size_raw'] = df_out['turnover'] / (df_out['volume'] + 1e-8)
    
    # Normalizuj przez typową cenę aby usunąć korelację z poziomem ceny
    df_out['avg_trade_size_norm'] = df_out['avg_trade_size_raw'] / (typical_price + 1e-8)
    
    # Rolling average dla porównania (czy trade size rośnie/maleje)
    df_out['avg_trade_size_momentum'] = (
        df_out['avg_trade_size_raw'] / (df_out['avg_trade_size_raw'].rolling(50).mean() + 1e-8)
    )
    
    # Usuń raw version (jest skorelowana z ceną)
    df_out.drop(columns=['avg_trade_size_raw'], inplace=True, errors='ignore')

    print("  8B. Order Flow Proxies (bez orderbook)...")
    
    # PERFORMANCE FIX: Batch adding to avoid DataFrame fragmentation
    # BATCH 1: Base features (needed as dependencies for later calculations)
    base_order_flow = {}
    base_order_flow['aggressive_buy_volume'] = df_out['volume'].where(
        (df_out['close'] > df_out['open']) & (df_out['close'] > df_out['close'].shift(1)), 0)
    base_order_flow['aggressive_sell_volume'] = df_out['volume'].where(
        (df_out['close'] < df_out['open']) & (df_out['close'] < df_out['close'].shift(1)), 0)
    df_out = pd.concat([df_out, pd.DataFrame(base_order_flow, index=df_out.index)], axis=1)
    
    # BATCH 2: Derived features (dependent on batch 1)
    derived_order_flow = {}
    derived_order_flow['buy_sell_imbalance_20'] = (
        df_out['aggressive_buy_volume'].rolling(20).sum() - 
        df_out['aggressive_sell_volume'].rolling(20).sum()
    ) / (df_out['volume'].rolling(20).sum() + 1e-8)
    derived_order_flow['buy_sell_imbalance_50'] = (
        df_out['aggressive_buy_volume'].rolling(50).sum() - 
        df_out['aggressive_sell_volume'].rolling(50).sum()
    ) / (df_out['volume'].rolling(50).sum() + 1e-8)
    derived_order_flow['buy_pressure'] = (df_out['close'] - df_out['low']) / (df_out['high'] - df_out['low'] + 1e-8)
    derived_order_flow['vwap_buy_pressure'] = (derived_order_flow['buy_pressure'] * df_out['volume']).rolling(20).sum() / (df_out['volume'].rolling(20).sum() + 1e-8)
    derived_order_flow['volume_delta'] = df_out['volume'].diff()
    derived_order_flow['volume_delta_pct'] = df_out['volume'].pct_change()
    df_out = pd.concat([df_out, pd.DataFrame(derived_order_flow, index=df_out.index)], axis=1)
    
    # BATCH 3: Advanced features (dependent on batch 2)
    advanced_order_flow = {}
    advanced_order_flow['volume_delta_positive'] = df_out['volume_delta'].where(df_out['close'] > df_out['open'], 0).rolling(10).sum()
    advanced_order_flow['volume_delta_negative'] = df_out['volume_delta'].where(df_out['close'] < df_out['open'], 0).rolling(10).sum()
    
    volume_mean = df_out['volume'].rolling(50).mean()
    volume_std = df_out['volume'].rolling(50).std()
    advanced_order_flow['volume_z_score'] = (df_out['volume'] - volume_mean) / (volume_std + 1e-8)
    advanced_order_flow['large_buy_trade'] = ((advanced_order_flow['volume_z_score'] > 2) & (df_out['close'] > df_out['open'])).astype(int)
    advanced_order_flow['large_sell_trade'] = ((advanced_order_flow['volume_z_score'] > 2) & (df_out['close'] < df_out['open'])).astype(int)
    advanced_order_flow['tape_speed_20'] = (df_out['volume'] > volume_mean).astype(int).rolling(20).sum()
    
    price_momentum = df_out['close'].pct_change(5)
    volume_momentum = df_out['volume'].pct_change(5)
    advanced_order_flow['momentum_volume_divergence'] = price_momentum - volume_momentum
    
    df_out = pd.concat([df_out, pd.DataFrame(advanced_order_flow, index=df_out.index)], axis=1)
    
    # TIER 2B: Advanced Order Flow (dodatkowe cechy dla lepszej separacji)
    # Collect all new features in a dictionary to add at once
    tier2b_features = {}
    
    # Volume Profile Approximation
    tier2b_features['price_range'] = df_out['high'] - df_out['low']
    tier2b_features['volume_per_price_unit'] = df_out['volume'] / (tier2b_features['price_range'] + 1e-8)
    
    # Buy/Sell Volume Ratio (stosunek, nie różnica)
    tier2b_features['buy_sell_ratio'] = (df_out['aggressive_buy_volume'] + 1) / (df_out['aggressive_sell_volume'] + 1)
    tier2b_features['buy_sell_ratio_ma20'] = tier2b_features['buy_sell_ratio'].rolling(20).mean()
    
    # Cumulative Buy/Sell Pressure - POPRAWIONE
    # Zamiast używać (1 - buy_pressure), użyjemy osobno obliczonego sell_pressure
    tier2b_features['cumulative_buy_pressure'] = df_out['buy_pressure'].rolling(50).sum()
    # Sell pressure jako osobna cecha (nie matematyczna odwrotność)
    sell_pressure = (df_out['high'] - df_out['close']) / (df_out['high'] - df_out['low'] + 1e-8)
    tier2b_features['cumulative_sell_pressure'] = sell_pressure.rolling(50).sum()
    # Netowa presja (różnica między buy i sell)
    tier2b_features['net_buy_sell_pressure'] = tier2b_features['cumulative_buy_pressure'] - tier2b_features['cumulative_sell_pressure']
    
    # Volume-Weighted Price Action
    tier2b_features['vwap_distance_weighted'] = df_out['dist_from_vwap'] * df_out['volume_vs_ma_20']
    
    # Delta Volume Change Rate
    tier2b_features['volume_change_rate'] = df_out['volume'].pct_change()
    tier2b_features['volume_acceleration_ma'] = tier2b_features['volume_change_rate'].rolling(10).mean()
    
    # Add TIER 2B features to dataframe
    df_out = pd.concat([df_out, pd.DataFrame(tier2b_features, index=df_out.index)], axis=1)
    
    # TIER 2C: Confidence/Meta Features (dla precision boost)
    tier2c_features = {}
    
    # Multi-timeframe alignment strength
    if 'sma_20_slope_4h' in df_out.columns and 'sma_20_slope_12h' in df_out.columns:
        tier2c_features['multi_tf_momentum_strength'] = (
            np.sign(df_out['price_change_pct_5']) + 
            np.sign(df_out['sma_20_slope_4h']) + 
            np.sign(df_out['sma_20_slope_12h'])
        )
        tier2c_features['multi_tf_alignment'] = (tier2c_features['multi_tf_momentum_strength'].abs() == 3).astype(int)
    
    # Volume confirmation score (ile z ostatnich N świec miało wysokie volume przy ruchu w górę)
    tier2c_features['bullish_volume_count'] = (
        (df_out['close'] > df_out['open']) & 
        (df_out['volume'] > df_out['volume'].rolling(20).mean())
    ).astype(int).rolling(10).sum()
    
    # Trend strength composite
    if 'adx_14' in df_out.columns:
        tier2c_features['trend_strength_composite'] = (
            (df_out['adx_14'] > 25).astype(int) * 2 +  # Strong trend
            (df_out['price_change_pct_5'] > 0).astype(int)  # Positive momentum
        )
    
    # Regime consistency (czy jesteśmy w stabilnym reżimie?)
    if 'volatility_regime' in df_out.columns:
        tier2c_features['regime_stability'] = 1 - df_out['volatility_regime'].rolling(20).std().fillna(0)
    
    # Add TIER 2C features to dataframe
    df_out = pd.concat([df_out, pd.DataFrame(tier2c_features, index=df_out.index)], axis=1)

    print("  9. Cechy zmienności i range'u...")
    volatility_features = {}
    
    # True Range Percentile
    atr_14 = df_out.ta.atr(length=14)
    volatility_features['tr_percentile'] = atr_14.rank(pct=True).rolling(window=50).mean()
    
    # High-Low Range vs Body
    volatility_features['range_to_body'] = (df_out['high'] - df_out['low']) / (abs(df_out['close'] - df_out['open']) + 1e-8)
    
    # Volatility Regime Detection
    atr_20 = df_out.ta.atr(length=20)
    volatility_features['volatility_regime'] = (atr_20 - atr_20.rolling(100).mean()) / atr_20.rolling(100).std()
    
    # Add volatility features to dataframe
    df_out = pd.concat([df_out, pd.DataFrame(volatility_features, index=df_out.index)], axis=1)

    print("  10. Kombinacje cech (interakcje)...")
    # RSI x Volume (momentum z potwierdzeniem wolumenu)
    interaction_features = {}
    interaction_features['rsi_volume_interaction'] = df_out['rsi_14'] * df_out['volume_vs_ma_20']
    
    # Price Position x Volume (czy cena przy kluczowych poziomach ma wsparcie wolumenu)
    interaction_features['pivot_volume_interaction'] = df_out['dist_from_s1'] * df_out['volume_vs_ma_20']
    
    # Volatility x Momentum
    interaction_features['volatility_momentum'] = df_out['atr_normalized'] * df_out['price_change_pct_5']
    
    # BBW Squeeze x Volume (czy ściskanie bollinger bands następuje przy niskim/wysokim wolumenie)
    if 'bbw_squeeze' in df_out.columns:
        interaction_features['bbw_volume_interaction'] = df_out['bbw_squeeze'] * df_out['volume_vs_ma_20']
    
    # Distance from VWAP x RSI (czy odchylenie od VWAP jest ekstremalnie + oversold/overbought)
    interaction_features['vwap_rsi_interaction'] = df_out['dist_from_vwap'] * (df_out['rsi_14'] - 50) / 50
    
    df_out = pd.concat([df_out, pd.DataFrame(interaction_features, index=df_out.index)], axis=1)

    print("  11. Zaawansowane momentum indicators...")
    momentum_features = {}
    
    # Stochastic Oscillator
    stoch = df_out.ta.stoch(high='high', low='low', close='close', k=14, d=3)
    if stoch is not None and not stoch.empty:
        momentum_features['stoch_k'] = stoch.iloc[:, 0]  # %K
        momentum_features['stoch_d'] = stoch.iloc[:, 1]  # %D
        momentum_features['stoch_cross'] = (stoch.iloc[:, 0] - stoch.iloc[:, 1]).apply(np.sign)
    
    # Rate of Change (ROC) - momentum velocity
    momentum_features['roc_5'] = df_out['close'].pct_change(5) * 100
    momentum_features['roc_10'] = df_out['close'].pct_change(10) * 100
    momentum_features['roc_20'] = df_out['close'].pct_change(20) * 100
    
    # Momentum Acceleration (second derivative)
    momentum_features['momentum_accel'] = df_out['close'].pct_change(5).diff() * 100
    
    # RSI-Price Divergence (czy RSI robi nowe high/low gdy cena tego nie robi)
    momentum_features['rsi_price_divergence'] = (
        (df_out['rsi_14'].diff(5) > 0).astype(int) - 
        (df_out['close'].diff(5) > 0).astype(int)
    )
    
    # Money Flow Index (MFI) - RSI z uwzględnieniem wolumenu
    mfi = df_out.ta.mfi(high='high', low='low', close='close', volume='volume', length=14)
    if mfi is not None:
        momentum_features['mfi_14'] = mfi
    
    # Commodity Channel Index (CCI)
    cci = df_out.ta.cci(high='high', low='low', close='close', length=20)
    if cci is not None:
        momentum_features['cci_20'] = cci
    
    df_out = pd.concat([df_out, pd.DataFrame(momentum_features, index=df_out.index)], axis=1)

    print("  12. Cechy czasowe (temporal features) - REMOVED per optimization recommendations...")
    # OPTIMIZATION: Temporal features removed to reduce overfitting to time patterns
    # Model should focus on price action, not clock-based patterns
    # Previously: hour_sin, hour_cos, day_sin, day_cos, is_weekend, session
    # These features were dominating (19.8% combined importance) but causing poor generalization
    
    # Defragment DataFrame mid-way to avoid PerformanceWarnings in subsequent sections
    df_out = df_out.copy()

    print("  13. Volume-Price Divergence...")
    # On-Balance Volume (OBV)
    df_out['obv'] = (np.sign(df_out['close'].diff()) * df_out['volume']).fillna(0).cumsum()
    df_out['obv_ma_20'] = df_out['obv'].rolling(20).mean()
    df_out['obv_divergence'] = (df_out['obv'] - df_out['obv_ma_20']) / (df_out['obv_ma_20'].abs() + 1e-8)
    
    # Volume-Price Trend (VPT) - similar to OBV but weighted by % price change
    df_out['vpt'] = (df_out['close'].pct_change() * df_out['volume']).fillna(0).cumsum()
    df_out['vpt_ma_20'] = df_out['vpt'].rolling(20).mean()
    
    # Accumulation/Distribution Line
    # A/D = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    ad_multiplier = ((df_out['close'] - df_out['low']) - (df_out['high'] - df_out['close'])) / (df_out['high'] - df_out['low'] + 1e-8)
    df_out['ad_line'] = (ad_multiplier * df_out['volume']).cumsum()
    df_out['ad_line_ma_20'] = df_out['ad_line'].rolling(20).mean()
    
    # Chaikin Money Flow (CMF)
    df_out['cmf_20'] = (ad_multiplier * df_out['volume']).rolling(20).sum() / (df_out['volume'].rolling(20).sum() + 1e-8)

    print("  14. TIER 2A: Market Regime Classification...")
    # ADX (Average Directional Index) - siła trendu
    adx = df_out.ta.adx(high='high', low='low', close='close', length=14)
    if adx is not None and not adx.empty:
        df_out['adx_14'] = adx['ADX_14']
        # ADX > 25 = trending, < 20 = ranging
        df_out['is_trending'] = (df_out['adx_14'] > 25).astype(int)
    
    # Volatility clustering (GARCH-like proxy)
    returns = df_out['close'].pct_change()
    df_out['return_volatility'] = returns.rolling(20).std()
    df_out['volatility_clustering'] = df_out['return_volatility'] / (df_out['return_volatility'].rolling(100).mean() + 1e-8)
    
    # Trend consistency (ile z ostatnich N świec było bullish/bearish)
    df_out['bullish_candle'] = (df_out['close'] > df_out['open']).astype(int)
    df_out['trend_consistency_10'] = df_out['bullish_candle'].rolling(10).mean()  # 0 = all bearish, 1 = all bullish
    df_out['trend_consistency_20'] = df_out['bullish_candle'].rolling(20).mean()
    
    # Choppiness Index (czy rynek jest choppy/ranging)
    # Simplified version
    tr = df_out.ta.atr(length=1)  # true range without smoothing
    tr_sum = tr.rolling(14).sum()
    high_low_diff = df_out['high'].rolling(14).max() - df_out['low'].rolling(14).min()
    df_out['choppiness_14'] = 100 * np.log10(tr_sum / (high_low_diff + 1e-8)) / np.log10(14)
    
    df_out.drop(columns=['bullish_candle'], inplace=True, errors='ignore')

    print("  15. TIER 2: Dodatkowe interakcje cech...")
    # Volume × Volatility (czy wysokie volume = wysokie volatility?)
    if 'volatility_regime' in df_out.columns:
        df_out['volume_volatility_interaction'] = df_out['volume_vs_ma_20'] * df_out['volatility_regime']
    
    # Momentum × Trend Strength (czy momentum jest zgodne z trendem?)
    if 'adx_14' in df_out.columns:
        df_out['momentum_trend_strength'] = df_out['price_change_pct_5'] * df_out['adx_14']
    
    # Candlestick patterns × Volume (czy pattern ma volume confirmation?)
    df_out['engulfing_volume'] = df_out['engulfing'] * df_out['volume_vs_ma_20']
    df_out['hammer_volume'] = df_out['hammer'] * df_out['volume_vs_ma_20']
    
    # Time-based × Volatility (czy pewne godziny są bardziej volatile?)
    if 'hour_sin' in df_out.columns and 'volatility_regime' in df_out.columns:
        df_out['time_volatility_interaction'] = df_out['hour_sin'] * df_out['volatility_regime']

    print("  16. TIER 2: Bid-ask spread proxies...")
    # High-Low spread jako proxy dla liquidity
    df_out['hl_spread'] = (df_out['high'] - df_out['low']) / (df_out['close'] + 1e-8)
    df_out['hl_spread_ma'] = df_out['hl_spread'].rolling(20).mean()
    df_out['hl_spread_normalized'] = df_out['hl_spread'] / (df_out['hl_spread_ma'] + 1e-8)
    
    # Close position within candle (gdzie close jest względem high/low)
    df_out['close_position'] = (df_out['close'] - df_out['low']) / (df_out['high'] - df_out['low'] + 1e-8)
    
    # Roll model estimate of spread (correlation of price changes)
    price_changes = df_out['close'].diff()
    df_out['price_change_autocorr'] = price_changes.rolling(20).apply(lambda x: x.autocorr(lag=1) if len(x.dropna()) > 1 else 0, raw=False)
    
    # Effective spread proxy: |close - (high+low)/2| / midpoint
    midpoint = (df_out['high'] + df_out['low']) / 2
    df_out['effective_spread_proxy'] = abs(df_out['close'] - midpoint) / (midpoint + 1e-8)

    print("  17. TIER 3: Support/Resistance Detection...")
    # Rolling highs/lows jako S/R proxies (prostsza wersja bez scipy)
    df_out['resistance_50'] = df_out['high'].rolling(50).max()
    df_out['support_50'] = df_out['low'].rolling(50).min()
    df_out['dist_from_resistance'] = (df_out['close'] - df_out['resistance_50']) / (df_out['resistance_50'] + 1e-8)
    df_out['dist_from_support'] = (df_out['close'] - df_out['support_50']) / (df_out['support_50'] + 1e-8)
    
    # Czy cena testuje resistance/support (0.5% threshold)
    df_out['testing_resistance'] = (abs(df_out['dist_from_resistance']) < 0.005).astype(int)
    df_out['testing_support'] = (abs(df_out['dist_from_support']) < 0.005).astype(int)
    
    # Dodatkowe S/R z dłuższych okresów dla multi-level detection
    df_out['resistance_100'] = df_out['high'].rolling(100).max()
    df_out['support_100'] = df_out['low'].rolling(100).min()
    df_out['near_resistance_100'] = (abs((df_out['close'] - df_out['resistance_100']) / (df_out['resistance_100'] + 1e-8)) < 0.01).astype(int)
    df_out['near_support_100'] = (abs((df_out['close'] - df_out['support_100']) / (df_out['support_100'] + 1e-8)) < 0.01).astype(int)
    
    # S/R strength: ile razy cena odbijała się od poziomu
    # Proxy: czy w ciągu ostatnich 20 świec cena była blisko aktualnego S/R
    df_out['resistance_strength'] = (df_out['high'] >= df_out['resistance_50'] * 0.995).astype(int).rolling(20).sum()
    df_out['support_strength'] = (df_out['low'] <= df_out['support_50'] * 1.005).astype(int).rolling(20).sum()
    
    # OPTIMIZATION: Enhanced S/R strength features
    print("  17B. Enhanced S/R strength analysis...")
    sr_threshold = 0.005  # 0.5%
    lookback = 100
    
    # Touch count for resistance
    resistance_touches = (abs((df_out['close'] - df_out['resistance_50']) / df_out['resistance_50']) < sr_threshold)
    df_out['resistance_touch_count'] = resistance_touches.astype(int).rolling(window=lookback).sum()
    
    # Touch count for support
    support_touches = (abs((df_out['close'] - df_out['support_50']) / df_out['support_50']) < sr_threshold)
    df_out['support_touch_count'] = support_touches.astype(int).rolling(window=lookback).sum()
    
    # Bounce ratio (how often price bounces vs breaks through)
    resistance_bounces = resistance_touches & (df_out['close'].shift(-1) < df_out['close'])
    df_out['resistance_bounce_ratio'] = (
        resistance_bounces.astype(int).rolling(window=lookback).sum() / 
        (df_out['resistance_touch_count'] + 1)
    )
    
    support_bounces = support_touches & (df_out['close'].shift(-1) > df_out['close'])
    df_out['support_bounce_ratio'] = (
        support_bounces.astype(int).rolling(window=lookback).sum() / 
        (df_out['support_touch_count'] + 1)
    )
    
    # Break momentum - volume and price change when breaking through S/R
    resistance_breaks = (df_out['close'].shift(1) < df_out['resistance_50']) & (df_out['close'] > df_out['resistance_50'])
    price_change_on_break = (df_out['close'] - df_out['resistance_50']) / df_out['resistance_50']
    volume_ratio_on_break = df_out['volume'] / df_out['volume'].rolling(window=20).mean()
    df_out['resistance_break_momentum'] = (price_change_on_break * volume_ratio_on_break).where(resistance_breaks, 0)
    
    support_breaks = (df_out['close'].shift(1) > df_out['support_50']) & (df_out['close'] < df_out['support_50'])
    price_change_on_break_down = (df_out['support_50'] - df_out['close']) / df_out['support_50']
    df_out['support_break_momentum'] = (price_change_on_break_down * volume_ratio_on_break).where(support_breaks, 0)

    print("  18. TIER 3: Advanced Price Action Patterns...")
    # Użyj nowych funkcji pattern detection
    df_out['three_line_strike'] = _detect_three_line_strike(df_out)
    df_out['morning_evening_star'] = _detect_morning_evening_star(df_out)
    
    # Gap detection (dla crypto mniej użyteczne, ale możliwe przy niskiej płynności)
    # Gap up: low obecnej świecy > high poprzedniej świecy
    df_out['gap_up'] = ((df_out['low'] - df_out['high'].shift(1)) / (df_out['high'].shift(1) + 1e-8) > 0.002).astype(int)
    # Gap down: high obecnej świecy < low poprzedniej świecy
    df_out['gap_down'] = ((df_out['high'] - df_out['low'].shift(1)) / (df_out['low'].shift(1) + 1e-8) < -0.002).astype(int)
    
    # Czy poprzednia świeca zamknęła się blisko high/low (potencjalne continuation)
    df_out['closed_near_high'] = ((df_out['close'].shift(1) - df_out['low'].shift(1)) / (df_out['high'].shift(1) - df_out['low'].shift(1) + 1e-8) > 0.8).astype(int)
    df_out['closed_near_low'] = ((df_out['close'].shift(1) - df_out['low'].shift(1)) / (df_out['high'].shift(1) - df_out['low'].shift(1) + 1e-8) < 0.2).astype(int)

    print("  19. TIER 3: Support/Resistance Interaction Features...")
    # Support/Resistance × Volume (czy testy S/R mają volume confirmation?)
    df_out['resistance_volume_interaction'] = df_out['dist_from_resistance'] * df_out['volume_vs_ma_20']
    df_out['support_volume_interaction'] = df_out['dist_from_support'] * df_out['volume_vs_ma_20']
    
    # Testing S/R × Volume (czy test S/R ma wysokie volume?)
    df_out['testing_resistance_with_volume'] = df_out['testing_resistance'] * df_out['volume_vs_ma_20']
    df_out['testing_support_with_volume'] = df_out['testing_support'] * df_out['volume_vs_ma_20']
    
    # S/R × Momentum (czy przy S/R jest silny momentum?)
    df_out['resistance_momentum_interaction'] = df_out['testing_resistance'] * abs(df_out['price_change_pct_5'])
    df_out['support_momentum_interaction'] = df_out['testing_support'] * abs(df_out['price_change_pct_5'])
    
    # S/R × RSI (czy przy S/R RSI jest w overbought/oversold?)
    if 'rsi_14' in df_out.columns:
        df_out['resistance_rsi_interaction'] = df_out['testing_resistance'] * (df_out['rsi_14'] / 100)
        df_out['support_rsi_interaction'] = df_out['testing_support'] * (1 - df_out['rsi_14'] / 100)

    # POPRAWKA #3: TIER 4 - Enhanced Momentum Features dla LONG
    print("  20. POPRAWKA #3: TIER 4 - Enhanced Momentum Features dla LONG...")
    enhanced_momentum_features = {}
    
    # Price momentum acceleration (2nd derivative) - wykrywa przyspieszenie ruchu
    enhanced_momentum_features['price_accel_3'] = df_out['close'].pct_change(3).diff()
    enhanced_momentum_features['price_accel_5'] = df_out['close'].pct_change(5).diff()
    
    # Multi-period momentum consensus (0 = bearish, 1 = bullish)
    enhanced_momentum_features['momentum_consensus'] = (
        (df_out['close'].pct_change(5) > 0).astype(int) +
        (df_out['close'].pct_change(10) > 0).astype(int) +
        (df_out['close'].pct_change(20) > 0).astype(int)
    ) / 3
    
    # Volume-confirmed momentum (czy momentum jest wspierany przez wolumen)
    enhanced_momentum_features['vol_confirmed_momentum'] = (
        (df_out['close'] > df_out['close'].shift(5)) & 
        (df_out['volume'] > df_out['volume'].rolling(20).mean())
    ).astype(int)
    
    # Breakout detection (price crossing above resistance with volume)
    if 'resistance_50' in df_out.columns:
        enhanced_momentum_features['breakout_signal'] = (
            (df_out['close'] > df_out['resistance_50']) &
            (df_out['close'].shift(1) <= df_out['resistance_50'].shift(1)) &
            (df_out['volume'] > df_out['volume'].rolling(20).mean() * 1.5)
        ).astype(int)
    
    # Trend acceleration (czy trend przyspiesza czy zwalnia)
    if 'sma_20_slope_4h' in df_out.columns:
        enhanced_momentum_features['trend_acceleration'] = df_out['sma_20_slope_4h'].diff()
    
    # Momentum strength score (composite indicator)
    enhanced_momentum_features['momentum_strength_score'] = (
        enhanced_momentum_features['momentum_consensus'] * 0.3 +
        enhanced_momentum_features['vol_confirmed_momentum'] * 0.4 +
        (df_out['rsi_14'] / 100) * 0.3
    )
    
    df_out = pd.concat([df_out, pd.DataFrame(enhanced_momentum_features, index=df_out.index)], axis=1)

    # ========================================================================
    # NOWE: Dodanie wskaźników kompozytowych
    # ========================================================================
    print("  21. Wskaźniki kompozytowe (uproszczone sygnały)...")
    df_out = add_oversold_overbought_signal(df_out)
    df_out = add_market_state_indicator(df_out)
    df_out = add_momentum_regime(df_out)
    df_out = add_volume_confirmation_score(df_out)
    df_out = add_multi_factor_sentiment(df_out)
    print("     ✓ Dodano 5 nowych wskaźników kompozytowych")
    # ========================================================================

    # ========================================================================
    # ICT & SMART MONEY CONCEPTS - WYSOKIE PRIORYTETY DLA MODELU
    # ========================================================================
    print("  22. ICT & Smart Money Concepts (HIGH PRIORITY)...")
    df_out = add_ict_smart_money_features(df_out)
    print("     ✓ ICT/Smart Money: 30+ cech dodanych z wysokim priorytetem")
    # ========================================================================

    df_out.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Defragment DataFrame to avoid PerformanceWarning
    df_out = df_out.copy()
    
    return df_out

def _calculate_helper_features(df: pd.DataFrame):
    """Calculate features for helper timeframes including TIER 2A: Multi-timeframe Trend Alignment."""
    df[f'atr_normalized'] = df.ta.atr(length=14) / df['close']
    swing_high, swing_low = df['high'].rolling(window=50).max(), df['low'].rolling(window=50).min()
    df[f'dist_from_swing_high_50'] = (df['close'] - swing_high) / swing_high
    df[f'dist_from_swing_low_50'] = (df['close'] - swing_low) / swing_low
    up_volume, down_volume = df['volume'].where(df['close'] > df['open'], 0), df['volume'].where(
        df['close'] < df['open'], 0)

    df['rvol_ratio'] = up_volume.rolling(window=50).sum() / (down_volume.rolling(window=50).sum() + 1e-5)

    # TIER 2A: Multi-timeframe Trend Alignment
    # Moving averages dla różnych okresów
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_100'] = df['close'].rolling(100).mean()
    
    # Trend direction (czy cena > MA)
    df['above_sma_20'] = (df['close'] > df['sma_20']).astype(int)
    df['above_sma_50'] = (df['close'] > df['sma_50']).astype(int)
    df['above_sma_100'] = (df['close'] > df['sma_100']).astype(int)
    
    # MA slope (czy MA rośnie/spada)
    df['sma_20_slope'] = df['sma_20'].pct_change(5)
    df['sma_50_slope'] = df['sma_50'].pct_change(5)
    
    # Trend strength: odległość od MA
    df['dist_from_sma_20'] = (df['close'] - df['sma_20']) / (df['sma_20'] + 1e-8)
    df['dist_from_sma_50'] = (df['close'] - df['sma_50']) / (df['sma_50'] + 1e-8)

    key_features = [
        'atr_normalized', 'dist_from_swing_high_50', 'dist_from_swing_low_50', 'rvol_ratio',
        'above_sma_20', 'above_sma_50', 'above_sma_100',
        'sma_20_slope', 'sma_50_slope',
        'dist_from_sma_20', 'dist_from_sma_50'
    ]
    df.dropna(inplace=True)
    return df[key_features]


def fetch_and_prepare_data(ticker: str, timeframe: str, limit: int, helper_timeframes: list = None, side: str = 'long', date_from: str = None, version: str = 'v1.0', model_features_to_preserve: list = None):
    logging.info(f"📊 Fetching data: ticker={ticker}, timeframe={timeframe}, limit={limit}, helpers={helper_timeframes}")
    if date_from:
        logging.warning(f"⚠️  UWAGA: Dane będą pobrane wstecz od daty: {date_from}")
    load_dotenv()
    api_key, api_secret = os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET")
    base_url = os.getenv("BYBIT_BASE_URL")
    if not api_key or not api_secret: raise ValueError("Brak kluczy API w .env")
    adapter = BybitAdapter(api_key=api_key, api_secret=api_secret, base_url=base_url)

    def to_dataframe(raw_data):
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        df = pd.DataFrame(raw_data, columns=cols)
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    logging.info(f"🔄 Calling fetch_ohlcv for {ticker} {timeframe}...")
    base_raw_data = adapter.fetch_ohlcv(symbol=ticker, timeframe=timeframe, limit=limit, end_date=date_from)
    if not base_raw_data:
        logging.error(f"❌ fetch_ohlcv returned EMPTY DATA for {ticker} {timeframe}! Check API connection, symbol validity, or timeframe format.")
        return pd.DataFrame()

    base_df = to_dataframe(base_raw_data)
    base_df = base_df.iloc[:-1]
    base_df.sort_index(inplace=True)
    logging.info(f"✅ Pobrano {len(base_df)} zamkniętych świec dla interwału bazowego {timeframe}.")
    logging.info(f"📊 Base DataFrame shape after loading: {base_df.shape}")

    if helper_timeframes:
        for helper_tf in helper_timeframes:
            print(f"Przetwarzanie interwału pomocniczego: {helper_tf}...")
            try:
                base_duration_mins = pd.to_timedelta(timeframe).total_seconds() / 60
                helper_duration_mins = pd.to_timedelta(helper_tf).total_seconds() / 60
                helper_limit = int((limit * base_duration_mins) / helper_duration_mins) + 100
            except (ValueError, TypeError):
                helper_limit = limit // 4 if 'h' in helper_tf else limit // 24

            helper_raw_data = adapter.fetch_ohlcv(symbol=ticker, timeframe=helper_tf, limit=helper_limit, end_date=date_from)
            if not helper_raw_data: continue

            helper_df = to_dataframe(helper_raw_data)
            helper_df = helper_df.iloc[:-1]
            helper_df.sort_index(inplace=True)
            helper_features = _calculate_helper_features(helper_df.copy())
            helper_features.rename(columns=lambda x: f"{x}_{helper_tf}", inplace=True)

            base_df = pd.merge_asof(base_df, helper_features, left_index=True, right_index=True, direction='backward')
            print(f"Dodano cechy z interwału {helper_tf}.")

    # FIX: Remove duplicate index values that may have been introduced during merge operations
    if base_df.index.duplicated().any():
        n_duplicates = base_df.index.duplicated().sum()
        print(f"⚠️  Wykryto {n_duplicates} duplikatów w indeksie. Usuwanie duplikatów (zachowuję pierwszy wpis)...")
        base_df = base_df[~base_df.index.duplicated(keep='first')]
        print(f"✓ Usunięto duplikaty. Pozostało {len(base_df)} unikalnych wpisów.")

    logging.info(f"📊 DataFrame shape before feature calculation: {base_df.shape}")
    final_df = _calculate_base_features(base_df)
    logging.info(f"📊 DataFrame shape after feature calculation: {final_df.shape}")

    # OPTIMIZATION: Multi-timeframe confluence features
    if helper_timeframes:
        print("\n🔄 Calculating multi-timeframe confluence features...")
        
        # Trend alignment score - are trends aligned across timeframes?
        trend_signals = []
        for helper_tf in helper_timeframes:
            slope_col = f'sma_20_slope_{helper_tf}'
            if slope_col in final_df.columns:
                trend_signals.append((final_df[slope_col] > 0).astype(int))
        
        if trend_signals:
            final_df['trend_alignment_score'] = sum(trend_signals) / len(trend_signals)
            print(f"   ✓ Added trend_alignment_score across {len(trend_signals)} timeframes")
        
        # Swing alignment score - are we near swing highs/lows across timeframes?
        swing_high_signals = []
        swing_low_signals = []
        for helper_tf in helper_timeframes:
            swing_high_col = f'dist_from_swing_high_50_{helper_tf}'
            swing_low_col = f'dist_from_swing_low_50_{helper_tf}'
            if swing_high_col in final_df.columns:
                # Negative distance means below swing high
                swing_high_signals.append((final_df[swing_high_col] > -0.02).astype(int))  # Within 2% of swing high
            if swing_low_col in final_df.columns:
                # Positive distance means above swing low
                swing_low_signals.append((final_df[swing_low_col] < 0.02).astype(int))  # Within 2% of swing low
        
        if swing_high_signals:
            final_df['near_swing_high_alignment'] = sum(swing_high_signals) / len(swing_high_signals)
            print(f"   ✓ Added near_swing_high_alignment across {len(swing_high_signals)} timeframes")
        
        if swing_low_signals:
            final_df['near_swing_low_alignment'] = sum(swing_low_signals) / len(swing_low_signals)
            print(f"   ✓ Added near_swing_low_alignment across {len(swing_low_signals)} timeframes")
        
        # Multi-timeframe momentum consensus
        momentum_signals = []
        for helper_tf in helper_timeframes:
            # Check if above SMA on each timeframe
            above_sma_col = f'above_sma_20_{helper_tf}'
            if above_sma_col in final_df.columns:
                momentum_signals.append(final_df[above_sma_col])
        
        if momentum_signals:
            final_df['momentum_alignment_score'] = sum(momentum_signals) / len(momentum_signals)
            print(f"   ✓ Added momentum_alignment_score across {len(momentum_signals)} timeframes")
        
        print("✓ Multi-timeframe confluence features completed\n")

    # Dynamic feature removal: Load weak features from file if exists
    print("Usuwanie najsłabszych cech w celu redukcji szumu...")
    
    # Construct strategy_id to match the naming convention in model_pipeline.py
    helpers_str = '_plus_' + '_'.join(helper_timeframes) if helper_timeframes else ""
    strategy_id = f"{ticker}_{timeframe.replace(' ', '')}{helpers_str}_{side}"
    
    # Use model_features_to_preserve parameter directly (passed from backtester/bot)
    if not model_features_to_preserve:
        if model_features_to_preserve is None:
            model_features_to_preserve = []
    
    print(f"🔧 Model features to preserve: {len(model_features_to_preserve)}")
    if not model_features_to_preserve:

        long_features_path = os.path.join("models", f"{strategy_id.replace('_backtest', '_long')}_features.joblib")
        short_features_path = os.path.join("models", f"{strategy_id.replace('_backtest', '_short')}_features.joblib")
        
        try:
            import joblib
            if os.path.exists(long_features_path):
                long_features = joblib.load(long_features_path)
                model_features_to_preserve.extend(long_features)
                print(f"   ✓ Loaded {len(long_features)} LONG model features from {os.path.basename(long_features_path)}")
            if os.path.exists(short_features_path):
                short_features = joblib.load(short_features_path)
                model_features_to_preserve.extend(short_features)
                print(f"   ✓ Loaded {len(short_features)} SHORT model features from {os.path.basename(short_features_path)}")
            model_features_to_preserve = list(set(model_features_to_preserve))  # Remove duplicates
            print(f"   ✓ Total unique model features to preserve: {len(model_features_to_preserve)}")
            print(f"   ✓ These features will NOT be removed during correlation/weak feature cleanup\n")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load model features: {e}")
            print(f"   Continuing without model feature preservation...\n")
    
    weak_features_path = os.path.join("models", f"{strategy_id}_weak_features.json")
    
    # Load weak features from file if exists, otherwise use empty list
    features_to_remove = []
    if os.path.exists(weak_features_path):
        try:
            with open(weak_features_path, 'r') as f:
                features_to_remove = json.load(f)
            print(f"Załadowano {len(features_to_remove)} słabych cech z pliku: {weak_features_path}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Ostrzeżenie: Nie udało się wczytać pliku słabych cech: {e}")
            print("Używam pustej listy cech do usunięcia.")
            features_to_remove = []
    else:
        print(f"Plik słabych cech nie istnieje: {weak_features_path}")
        print("To normalne podczas pierwszego uruchomienia. Żadne cechy nie zostaną usunięte.")
    
    # BACKTEST FIX: Exclude model features from removal
    if model_features_to_preserve:
        features_to_remove = [f for f in features_to_remove if f not in model_features_to_preserve]
        print(f"🔧 Excluded {len(model_features_to_preserve)} model features from weak feature removal")
    
    existing_features_to_remove = [col for col in features_to_remove if col in final_df.columns]
    if existing_features_to_remove:
        final_df.drop(columns=existing_features_to_remove, inplace=True)
        print(f"Usunięto następujące cechy: {existing_features_to_remove}")
    elif features_to_remove:
        print(f"Ostrzeżenie: Żadna z {len(features_to_remove)} cech do usunięcia nie została znaleziona w danych.")

    print(f"\nKształt danych przed czyszczeniem (usunięciem wierszy z NaN): {final_df.shape}")
    initial_rows = len(final_df)
    final_df.dropna(inplace=True)
    final_rows = len(final_df)
    print(f"Usunięto {initial_rows - final_rows} początkowych wierszy z powodu okresu 'burn-in' dla wskaźników.")

    # ========================================================================
    # NOWE: Usuwanie skorelowanych cech (opcjonalne)
    # ========================================================================
    # WŁĄCZONE: Na podstawie raportu korelacji - mamy 50 par > 0.85!
    REMOVE_CORRELATED = True  # ✅ WŁĄCZONO po analizie korelacji
    CORRELATION_THRESHOLD = 0.90  # Próg korelacji (zwiększony do 0.90 przez użytkownika)
    
    if REMOVE_CORRELATED:
        print(f"\n🔍 Wykrywanie i usuwanie cech skorelowanych powyżej {CORRELATION_THRESHOLD}...")
        
        # ⚠️ KROK 1: USUŃ surowe OHLC PRZED analizą korelacji
        # Te cechy są zawsze skorelowane (reprezentują poziom ceny)
        # Zachowamy tylko CECHY POCHODNE (dist_from_*, price_change_pct, etc.)
        # BACKTEST FIX: Keep OHLCV for backtesting (needed for trade execution)
        if side != 'backtest':
            print("\n🗑️  Usuwanie surowych cech OHLC (zawsze skorelowane)...")
            ohlc_to_remove = ['open', 'high', 'low']  # Zachowujemy 'close' jako reference point
            existing_ohlc = [col for col in ohlc_to_remove if col in final_df.columns]
            if existing_ohlc:
                final_df.drop(columns=existing_ohlc, inplace=True)
                print(f"   ✓ Usunięto: {existing_ohlc}")
                print(f"   ✓ Zachowano: 'close' (jako punkt odniesienia)")
                print(f"   ✓ Zachowano: cechy pochodne (dist_from_*, price_change_pct, body_to_wick_ratio, etc.)\n")
        else:
            print("\n🔧 BACKTEST MODE: Keeping OHLCV columns for trade execution")
            print(f"   ✓ Preserved: open, high, low, close, volume\n")
        
        # ⚠️ KROK 2: USUŃ pivoty S1/R1 (skorelowane z ceną)
        # Zachowamy tylko DISTANCE od pivotów (dist_from_s1, dist_from_r1)
        print("🗑️  Usuwanie surowych pivotów (zachowujemy distance)...")
        pivot_to_remove = ['s1', 'r1', 'resistance_50', 'resistance_100', 'support_50', 'support_100']
        existing_pivots = [col for col in pivot_to_remove if col in final_df.columns]
        if existing_pivots:
            final_df.drop(columns=existing_pivots, inplace=True)
            print(f"   ✓ Usunięto surowe pivoty: {len(existing_pivots)} cech")
            print(f"   ✓ Zachowano: dist_from_s1, dist_from_r1, dist_from_resistance, dist_from_support\n")
        
        print(f"📊 Kształt danych po usunięciu surowych cech: {final_df.shape}\n")
        print(f"\n🔍 Wykrywanie i usuwanie cech skorelowanych powyżej {CORRELATION_THRESHOLD}...")
        
        # KROK 3: Usuń POZOSTAŁE skorelowane cechy (po usunięciu OHLC i pivotów)
        print(f"🔍 Analiza pozostałych korelacji (threshold > {CORRELATION_THRESHOLD})...\n")
        
        # Lista ważnych cech które zawsze zachowujemy
        important_features = [
            # Podstawowe wskaźniki
            'close',  # Reference price point
            'rsi_14', 'volume_vs_ma_20', 'dist_from_vwap', 'atr_normalized',
            # Nowe wskaźniki kompozytowe (MUST KEEP!)
            'market_state_indicator', 'momentum_regime', 'volume_confirmation_score',
            'multi_factor_sentiment', 'oversold_overbought_signal',
            # Kluczowe wskaźniki techniczne
            'adx_14', 'obv_divergence', 'cmf_20', 'mfi_14', 'cci_20',
            # Volume features
            'volume', 'turnover', 'rvol_ratio',
            # Volume-Price Divergence features (MUST KEEP!)
            'obv', 'obv_ma_20', 'vpt', 'vpt_ma_20', 'ad_line', 'ad_line_ma_20',
            # Key price derivatives (NIE surowe OHLC!)
            'dist_from_s1', 'dist_from_r1', 'dist_from_resistance', 'dist_from_support',
            'price_change_pct_5', 'roc_10', 'roc_20', 'momentum_consensus',
            # VWAP
            'vwap', 'dist_from_vwap', 'vwap_momentum',
            # Candlestick features (NIE surowe OHLC!)
            'body_to_wick_ratio', 'upper_wick_size', 'lower_wick_size',
            # Order flow
            'buy_pressure', 'net_buy_sell_pressure',  # POPRAWIONE!
            'buy_sell_imbalance_20', 'buy_sell_imbalance_50',
            # Trend
            'momentum_strength_score', 'vol_confirmed_momentum',
            # OPTIMIZATION: New price action features (MUST KEEP!)
            'three_white_soldiers', 'higher_highs_count_20', 'higher_lows_count_20', 
            'trend_structure_score',
            # OPTIMIZATION: Enhanced S/R features (MUST KEEP!)
            'resistance_touch_count', 'support_touch_count', 
            'resistance_bounce_ratio', 'support_bounce_ratio',
            'resistance_break_momentum', 'support_break_momentum',
            # OPTIMIZATION: Multi-timeframe confluence (MUST KEEP!)
            'trend_alignment_score', 'near_swing_high_alignment', 
            'near_swing_low_alignment', 'momentum_alignment_score'
        ]
        
        # BACKTEST FIX: Merge model features with important features
        if model_features_to_preserve:
            important_features.extend(model_features_to_preserve)
            important_features = list(set(important_features))  # Remove duplicates
            print(f"🔧 Added {len(model_features_to_preserve)} model features to preservation list")
            print(f"   Total features to preserve: {len(important_features)}")
            print(f"   Model features sample: {model_features_to_preserve[:10]}\n")
        
        # BACKTEST FIX: Add OHLCV columns to important features for backtest mode
        if side == 'backtest':
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            important_features.extend(ohlcv_cols)
            important_features = list(set(important_features))
            print(f"🔧 BACKTEST MODE: Added OHLCV columns to preservation list")
            print(f"   Total features to preserve: {len(important_features)}\n")
        
        print(f"📊 Features before correlation removal: {final_df.shape[1]}")
        print(f"   Checking if model features exist: hl_spread={('hl_spread' in final_df.columns)}, dist_from_swing_high_50_4h={('dist_from_swing_high_50_4h' in final_df.columns)}\n")
        
        final_df, removed_corr_features = remove_correlated_features(
            final_df,
            target_col=None,  # Jeśli masz target, podaj tu nazwę kolumny
            correlation_threshold=CORRELATION_THRESHOLD,
            keep_important=important_features
        )
        
        print(f"\n📊 Features after correlation removal: {final_df.shape[1]}")
        print(f"   Checking if model features exist: hl_spread={('hl_spread' in final_df.columns)}, dist_from_swing_high_50_4h={('dist_from_swing_high_50_4h' in final_df.columns)}\n")
        
        # Opcjonalnie: zapisz listę usuniętych cech skorelowanych
        if removed_corr_features:
            helpers_str = '_plus_' + '_'.join(helper_timeframes) if helper_timeframes else ""
            corr_features_path = os.path.join("models", version, f"{ticker}_{timeframe.replace(' ', '')}{helpers_str}_{side}_correlated_features.json")
            os.makedirs(os.path.dirname(corr_features_path), exist_ok=True)
            with open(corr_features_path, 'w') as f:
                json.dump(removed_corr_features, f, indent=2)
            print(f"💾 Zapisano listę {len(removed_corr_features)} skorelowanych cech do: {corr_features_path}")
    # ========================================================================

    print(f"\nPrzygotowywanie cech zakończone. Finalny kształt danych: {final_df.shape}")
    return final_df
