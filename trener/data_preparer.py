# utils/data_preparer.py
import pandas as pd
import pandas_ta as ta
import numpy as np


def add_fibonacci_features(df, window=100):
    """
    Oblicza i dodaje cechy oparte na zniesieniach Fibonacciego.
    """
    # 1. Automatyczne wykrywanie swingu high/low w oknie
    swing_high = df['high'].rolling(window=window).max()
    swing_low = df['low'].rolling(window=window).min()
    swing_range = swing_high - swing_low
    swing_range = swing_range.replace(0, np.nan)  # Unikamy dzielenia przez zero

    # 2. Obliczanie poziomów Fibo
    fibo_levels = {
        'FIBO_0.0': swing_high,
        'FIBO_23.6': swing_high - swing_range * 0.236,
        'FIBO_38.2': swing_high - swing_range * 0.382,
        'FIBO_50.0': swing_high - swing_range * 0.5,
        'FIBO_61.8': swing_high - swing_range * 0.618,
        'FIBO_100.0': swing_low
    }
    fibo_df = pd.DataFrame(fibo_levels, index=df.index)

    # 3. Tworzenie cech dla modelu

    # Cecha 1: Względna pozycja ceny w swingu (0 = na dołku, 1 = na szczycie)
    df['FIBO_relative_position'] = (df['close'] - swing_low) / swing_range

    # Cecha 2 & 3: Dystans i nazwa najbliższego poziomu Fibo
    # Obliczamy dystans od ceny 'close' do każdego z poziomów Fibo
    distances = fibo_df.sub(df['close'], axis=0).abs()

    # Znajdujemy, który poziom jest najbliżej
    df['FIBO_nearest_level'] = distances.idxmin(axis=1)
    df['FIBO_distance_to_nearest'] = distances.min(axis=1) / swing_range  # Normalizujemy dystans

    # Przekształcamy nazwy poziomów na wartości liczbowe (np. 'FIBO_38.2' -> 38.2)
    df['FIBO_nearest_level'] = df['FIBO_nearest_level'].str.replace('FIBO_', '').astype(float)

    return df

def add_divergence_feature(df, indicator_col, price_high_col='high', price_low_col='low', window=28):
    """
    Oblicza i dodaje cechę dywergencji dla danego wskaźnika.
    Zwraca 1 dla dywergencji byczej, -1 dla niedźwiedziej, 0 w pozostałych przypadkach.
    """
    low_price_lookback = df[price_low_col].rolling(window=window).min().shift(1)
    low_indicator_lookback = df[indicator_col].rolling(window=window).min().shift(1)

    high_price_lookback = df[price_high_col].rolling(window=window).max().shift(1)
    high_indicator_lookback = df[indicator_col].rolling(window=window).max().shift(1)

    bullish_divergence = (df[price_low_col] < low_price_lookback) & \
                         (df[indicator_col] > low_indicator_lookback)

    bearish_divergence = (df[price_high_col] > high_price_lookback) & \
                         (df[indicator_col] < high_indicator_lookback)

    div_col_name = f'DIVERGENCE_{indicator_col}'
    df[div_col_name] = 0
    df.loc[bullish_divergence, div_col_name] = 1
    df.loc[bearish_divergence, div_col_name] = -1

    return df


def prepare_feature_set_for_timeframe(df_5m_raw: pd.DataFrame, base_tf: str = '5m'):
    """
    Agreguje dane, oblicza wskaźniki i wszystkie zaawansowane cechy,
    a następnie łączy je w jeden DataFrame.
    """
    print(f"Przygotowywanie zestawu cech dla interwału bazowego: {base_tf}...")

    timeframes = {'5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h'}
    if base_tf not in timeframes:
        raise ValueError(f"Nieobsługiwany interwał bazowy: {base_tf}.")

    ohlc = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'turnover': 'sum'}

    all_dfs = {}
    for tf_name, tf_pandas in timeframes.items():
        if tf_name == '5m':
            all_dfs['5m'] = df_5m_raw.copy()
        else:
            all_dfs[tf_name] = df_5m_raw.resample(tf_pandas).agg(ohlc).dropna()

    for tf_name, df in all_dfs.items():
        print(f"Obliczanie wskaźników dla interwału {tf_name}...")
        # --- Standardowe wskaźniki ---
        df.ta.rsi(append=True)
        df.ta.atr(append=True)
        df.ta.macd(append=True)
        df.ta.bbands(append=True)
        df.ta.stoch(append=True)
        df.ta.adx(append=True)
        df.ta.obv(append=True)
        df.ta.vwap(append=True)
        df.ta.cci(append=True)
        df.ta.mfi(append=True)
        df.ta.aroon(append=True)

        # --- Zaawansowane wskaźniki i cechy ---

        # Dywergencja dla RSI
        rsi_col_name = 'RSI_14'
        if rsi_col_name in df.columns:
            df = add_divergence_feature(df, indicator_col=rsi_col_name)

        # SuperTrend (tylko kierunek)
        st_df = df.ta.supertrend(append=False)
        if st_df is not None and not st_df.empty:
            direction_col_name = next((col for col in st_df.columns if 'SUPERTd' in col), None)
            if direction_col_name:
                df[direction_col_name] = st_df[direction_col_name]

        # Ichimoku
        ichimoku_df = df.ta.ichimoku(append=False)[0]
        df = pd.concat([df, ichimoku_df], axis=1)

        # Cechy wyższego rzędu (pęd i zmienność wskaźników)
        if 'RSI_14' in df.columns:
            df['RSI_14_roc_1'] = df['RSI_14'].diff()
            df['RSI_14_vol_10'] = df['RSI_14'].rolling(window=10).std()
        if 'MACDh_12_26_9' in df.columns:
            df['MACDh_12_26_9_roc_1'] = df['MACDh_12_26_9'].diff()

        # TTM Squeeze
        df.ta.squeeze(append=True)
        # Kanały Donchiana
        df.ta.donchian(append=True)

        # Parabolic SAR (w inteligentny sposób, aby uniknąć NaN)
        psar_df = df.ta.psar(append=False)
        if psar_df is not None and not psar_df.empty:
            reversal_col = next((col for col in psar_df.columns if 'PSARr' in col), None)
            if reversal_col:
                df[reversal_col] = psar_df[reversal_col]
        # =====================================

        # Automatyczne rozpoznawanie formacji świecowych
        df.ta.cdl_pattern(name="all", append=True)

        # === DODANIE CECH FIBONACCIEGO ===
        print(f"Dodawanie cech Fibonacciego dla interwału {tf_name}...")
        df = add_fibonacci_features(df, window=100)  # Okno 100 świec do znalezienia swingu
        # ==================================

        all_dfs[tf_name] = df

    # --- Łączenie interwałów ---
    base_df = all_dfs[base_tf].add_suffix(f'_{base_tf}')
    base_df.rename(columns={
        f'open_{base_tf}': 'open', f'high_{base_tf}': 'high', f'low_{base_tf}': 'low',
        f'close_{base_tf}': 'close', f'volume_{base_tf}': 'volume', f'turnover_{base_tf}': 'turnover'
    }, inplace=True)

    final_df = base_df
    for tf_name, df_to_merge in all_dfs.items():
        if tf_name == base_tf: continue
        df_with_suffix = df_to_merge.drop(columns=['open', 'high', 'low', 'close', 'volume', 'turnover'],
                                          errors='ignore').add_suffix(f'_{tf_name}')
        final_df = pd.merge_asof(final_df, df_with_suffix, left_index=True, right_index=True, direction='backward')

    # --- Cechy Price Action ---
    atr_col_name = f'ATRr_14_{base_tf}'
    if atr_col_name not in final_df.columns:
        final_df.ta.atr(col_names=(atr_col_name,), append=True)

    pa_df = final_df[['open', 'high', 'low', 'close', 'volume', atr_col_name]].copy()
    pa_df['impulse_strength'] = (pa_df['close'] - pa_df['open']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    pa_df['volatility_burst'] = (pa_df['high'] - pa_df['low']) / pa_df[atr_col_name].replace(0, 1)
    pa_df['closing_position'] = (pa_df['close'] - pa_df['low']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    volume_rolling_mean = pa_df['volume'].rolling(window=20).mean().replace(0, 1)
    pa_df['volume_spike'] = pa_df['volume'] / volume_rolling_mean

    for col in ['impulse_strength', 'volatility_burst', 'closing_position', 'volume_spike']:
        for n in [1, 2, 3]:
            pa_df[f'{col}_lag_{n}'] = pa_df[col].shift(n)

    pa_features_to_add = [col for col in pa_df.columns if
                          col not in ['open', 'high', 'low', 'close', 'volume', atr_col_name]]
    final_df = pd.concat([final_df, pa_df[pa_features_to_add]], axis=1)

    # --- Cechy Reżimu Rynkowego i Czasowe ---
    adx_col_name = f'ADX_14_{base_tf}'
    if adx_col_name in final_df.columns:
        final_df['market_regime_trending'] = (final_df[adx_col_name] > 25).astype(int)

    print("Dodawanie cech czasowych...")
    if not pd.api.types.is_datetime64_any_dtype(final_df.index):
        final_df.index = pd.to_datetime(final_df.index)

    day_of_week = final_df.index.dayofweek
    hour_of_day = final_df.index.hour

    final_df['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
    final_df['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
    final_df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    final_df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    # --- Finalizacja ---
    print(f"Zakończono przygotowywanie cech. Finalny kształt danych: {final_df.shape}")
    return final_df