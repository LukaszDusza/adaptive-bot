"""
PROPOZYCJE NOWYCH WSKAŹNIKÓW KOMPOZYTOWYCH
===========================================

1. MARKET STATE INDICATOR (MSI) - kompleksowy sygnał stanu rynku
2. MOMENTUM REGIME - uproszczony sygnał momentum
3. VOLUME CONFIRMATION SCORE - czy wolumen potwierdza ruch
4. SUPPORT/RESISTANCE PROXIMITY - jak blisko jesteśmy kluczowych poziomów
5. Feature Correlation Removal - mechanizm usuwania skorelowanych cech
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def add_market_state_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market State Indicator (MSI) - kompozytowy wskaźnik stanu rynku
    
    Zwraca wartości:
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
    
    # Clip to range [-3, 3]
    msi = msi.clip(-3, 3)
    
    df['market_state_indicator'] = msi
    return df


def add_momentum_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum Regime - prosty trójstanowy wskaźnik momentum
    
    Zwraca:
    -1: Bearish momentum (multiple timeframes negative)
     0: Neutral/Choppy
     1: Bullish momentum (multiple timeframes positive)
    """
    momentum_score = pd.Series(0, index=df.index)
    
    # Short-term momentum
    if 'price_change_pct_5' in df.columns:
        momentum_score += np.sign(df['price_change_pct_5'])
    
    # Medium-term momentum (ROC if available)
    if 'roc_10' in df.columns:
        momentum_score += np.sign(df['roc_10'])
    elif 'price_change_pct_5' in df.columns:
        # Fallback to longer price change
        momentum_score += np.sign(df['close'].pct_change(10))
    
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
    elif 'volume' in df.columns:
        volume_ma = df['volume'].rolling(20).mean()
        high_volume = (df['volume'] > volume_ma * 1.2)
    
    # Confirmation logic
    vcs[(price_up == 1) & high_volume] = 1   # Bullish with volume
    vcs[(price_down == 1) & high_volume] = -1  # Bearish with volume
    
    df['volume_confirmation_score'] = vcs
    return df


def add_sr_proximity_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Support/Resistance Proximity Indicator
    
    Zwraca:
    -2: At resistance with rejection signs
    -1: Near resistance
     0: In the middle
     1: Near support
     2: At support with bounce signs
    """
    sr_prox = pd.Series(0, index=df.index)
    
    if 'dist_from_resistance' in df.columns and 'dist_from_support' in df.columns:
        # Near resistance (within 1%)
        near_resistance = (df['dist_from_resistance'].abs() < 0.01)
        # Near support (within 1%)
        near_support = (df['dist_from_support'].abs() < 0.01)
        
        # Check for rejection/bounce using price action
        rejection = (df['close'] < df['open'])  # Red candle
        bounce = (df['close'] > df['open'])     # Green candle
        
        # Set signals
        sr_prox[near_resistance] = -1
        sr_prox[near_support] = 1
        sr_prox[near_resistance & rejection] = -2
        sr_prox[near_support & bounce] = 2
    
    df['sr_proximity_indicator'] = sr_prox
    return df


def add_oversold_overbought_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prosty binarny wskaźnik oversold/overbought (dokładnie to o co prosił użytkownik)
    
    Zwraca:
    -1: Overbought (potencjalna korekta w dół)
     0: Neutral
     1: Oversold (potencjalny odbicie w górę)
    """
    signal = pd.Series(0, index=df.index)
    
    if 'rsi_14' in df.columns:
        signal[df['rsi_14'] < 30] = 1   # Oversold -> bullish signal
        signal[df['rsi_14'] > 70] = -1  # Overbought -> bearish signal
    
    df['oversold_overbought_signal'] = signal
    return df


def add_multi_factor_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-Factor Sentiment Score - agregacja wielu sygnałów w jeden
    
    Zbiera sygnały z:
    - RSI (oversold/overbought)
    - Momentum (wzrost/spadek)
    - Volume (potwierdzenie)
    - Trend (czy w trendzie)
    - S/R (bliskość kluczowych poziomów)
    
    Zwraca wartość od -5 do 5, gdzie:
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
    
    # S/R component
    if 'sr_proximity_indicator' in df.columns:
        sentiment += df['sr_proximity_indicator'] * 0.5
    
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
        target_col: Nazwa kolumny targetu (zostanie pominięta w analizie korelacji)
        correlation_threshold: Próg korelacji powyżej którego usuwamy cechy (domyślnie 0.85)
        keep_important: Lista nazw cech które zawsze zachowujemy (np. kluczowe wskaźniki)
    
    Returns:
        Tuple[DataFrame, List[str]]: 
            - DataFrame z usuniętymi skorelowanymi cechami
            - Lista nazw usuniętych cech
    """
    print(f"\n{'='*60}")
    print("ANALIZA KORELACJI CECH")
    print(f"{'='*60}")
    print(f"Próg korelacji: {correlation_threshold}")
    print(f"Początkowa liczba cech: {df.shape[1]}")
    
    # Domyślna lista ważnych cech do zachowania
    if keep_important is None:
        keep_important = [
            'rsi_14', 'volume_vs_ma_20', 'dist_from_vwap', 'atr_normalized',
            'market_state_indicator', 'momentum_regime', 'volume_confirmation_score',
            'sr_proximity_indicator', 'multi_factor_sentiment', 'oversold_overbought_signal'
        ]
    
    # Wydziel cechy numeryczne (bez targetu jeśli podano)
    cols_to_analyze = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in cols_to_analyze:
        cols_to_analyze.remove(target_col)
    
    # Oblicz macierz korelacji
    print(f"Obliczanie macierzy korelacji dla {len(cols_to_analyze)} cech...")
    corr_matrix = df[cols_to_analyze].corr().abs()
    
    # Górny trójkąt macierzy (unikamy duplikatów)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Znajdź pary cech o wysokiej korelacji
    to_drop = set()
    high_corr_pairs = []
    
    for column in upper_triangle.columns:
        # Znajdź cechy skorelowane z obecną cechą
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
                # Priorytet: zachowaj cechy z keep_important
                if column in keep_important and corr_feature not in keep_important:
                    to_drop.add(corr_feature)
                elif corr_feature in keep_important and column not in keep_important:
                    to_drop.add(column)
                else:
                    # Jeśli obie/żadna nie jest ważna, usuń tę z mniejszą wariancją
                    # (mniej informatywna)
                    var_col = df[column].var()
                    var_corr = df[corr_feature].var()
                    if var_col < var_corr:
                        to_drop.add(column)
                    else:
                        to_drop.add(corr_feature)
    
    # Raport
    print(f"\nZnaleziono {len(high_corr_pairs)} par cech o korelacji > {correlation_threshold}")
    
    if high_corr_pairs:
        print("\nNajwyższe korelacje:")
        sorted_pairs = sorted(high_corr_pairs, key=lambda x: x['correlation'], reverse=True)
        for pair in sorted_pairs[:10]:  # Pokaż top 10
            print(f"  {pair['feature1']:40s} <-> {pair['feature2']:40s} : {pair['correlation']:.3f}")
    
    to_drop_list = list(to_drop)
    print(f"\nUsuwam {len(to_drop_list)} skorelowanych cech")
    
    if to_drop_list:
        print("\nUsunięte cechy:")
        for feature in sorted(to_drop_list):
            print(f"  - {feature}")
    
    # Usuń skorelowane cechy
    df_cleaned = df.drop(columns=to_drop_list, errors='ignore')
    
    print(f"\nKońcowa liczba cech: {df_cleaned.shape[1]}")
    print(f"Usunięto: {len(to_drop_list)} cech ({len(to_drop_list)/len(cols_to_analyze)*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return df_cleaned, to_drop_list


def add_all_composite_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje wszystkie zaproponowane wskaźniki kompozytowe
    """
    print("\nDodawanie wskaźników kompozytowych...")
    
    df = add_oversold_overbought_binary(df)
    print("  ✓ Oversold/Overbought Signal")
    
    df = add_market_state_indicator(df)
    print("  ✓ Market State Indicator")
    
    df = add_momentum_regime(df)
    print("  ✓ Momentum Regime")
    
    df = add_volume_confirmation_score(df)
    print("  ✓ Volume Confirmation Score")
    
    df = add_sr_proximity_indicator(df)
    print("  ✓ Support/Resistance Proximity")
    
    df = add_multi_factor_sentiment(df)
    print("  ✓ Multi-Factor Sentiment")
    
    return df


# PRZYKŁAD UŻYCIA
if __name__ == "__main__":
    # Przykładowe użycie (zakładając że masz już df z cechami)
    # df = fetch_and_prepare_data(...)
    
    # 1. Dodaj kompozytowe wskaźniki
    # df = add_all_composite_indicators(df)
    
    # 2. Usuń skorelowane cechy
    # df_cleaned, removed_features = remove_correlated_features(
    #     df, 
    #     target_col='target',
    #     correlation_threshold=0.85,
    #     keep_important=['rsi_14', 'market_state_indicator', 'multi_factor_sentiment']
    # )
    
    pass
