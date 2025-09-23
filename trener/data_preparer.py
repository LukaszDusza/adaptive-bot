# utils/data_preparer.py
import pandas as pd
import pandas_ta as ta
import numpy as np


# W pliku data_preparer.py

def add_zigzag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oblicza wskaźnik ZigZag i tworzy z niego użyteczne cechy.
    Wersja poprawiona, aby unikać błędu 'ValueError: Length of values...'.
    """
    print(f"Dodawanie cech ZigZag...")
    zigzag_df = df.ta.zigzag(high='high', low='low', append=False)
    if zigzag_df is None or zigzag_df.empty:
        return df

    df['zigzag_signal'] = zigzag_df.iloc[:, 1]

    # === POPRAWIONY BLOK LOGIKI ===
    # Krok 1: Stwórz "rzadką" serię, która ma wartości tylko w punktach zwrotnych
    pivot_prices = df['close'].where(df['zigzag_signal'].notna())
    pivot_times = df.index.to_series().where(df['zigzag_signal'].notna())

    # Krok 2: Użyj ffill (forward-fill), aby wypełnić puste miejsca ostatnią znaną wartością
    df['last_pivot_price'] = pivot_prices.ffill()
    last_pivot_time = pivot_times.ffill()

    # Krok 3: Oblicz cechy na podstawie wypełnionych danych
    df['dist_from_last_pivot'] = (df['close'] - df['last_pivot_price']) / df['last_pivot_price']

    time_diff = df.index.to_series().diff().median()
    if pd.notna(time_diff):
        df['bars_since_last_pivot'] = (df.index - last_pivot_time) / time_diff
    else:
        df['bars_since_last_pivot'] = 0
    # ===============================

    df.fillna({'zigzag_signal': 0, 'dist_from_last_pivot': 0, 'bars_since_last_pivot': 0}, inplace=True)
    df.drop(columns=['last_pivot_price'], inplace=True, errors='ignore')

    return df

def add_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oblicza dzienne Pivot Points i dołącza je do danych intraday.
    Wersja ostateczna, używająca groupby zamiast zepsutego resample.
    """
    print("Obliczanie dziennych Pivot Points...")
    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index)

    # === OSTATECZNA POPRAWKA: Używamy groupby zamiast resample ===
    # Krok 1: Stwórz tymczasową kolumnę z samą datą
    df_copy['date_for_grouping'] = df_copy.index.date

    # Krok 2: Grupuj po dacie i agreguj, aby uzyskać dane dzienne
    daily_agg = {
        'high': 'max', 'low': 'min', 'close': 'last'
    }
    df_daily = df_copy.groupby('date_for_grouping').agg(daily_agg)
    # =============================================================

    print(f"-> Znaleziono {len(df_daily)} dni z danymi do obliczenia pivotów.")

    if df_daily.empty:
        print("Ostrzeżenie: Brak danych dziennych do obliczenia Pivot Points. Pomijanie kroku.")
        return df.drop(columns=['date_for_grouping'], errors='ignore')

    prev_day = df_daily.shift(1).dropna()

    if prev_day.empty:
        print("Ostrzeżenie: Brak danych z poprzedniego dnia. Pomijanie kroku.")
        return df.drop(columns=['date_for_grouping'], errors='ignore')

    # 2. Oblicz poziomy Pivot Points
    pp = (prev_day['high'] + prev_day['low'] + prev_day['close']) / 3
    r1 = 2 * pp - prev_day['low']
    s1 = 2 * pp - prev_day['high']
    r2 = pp + (prev_day['high'] - prev_day['low'])
    s2 = pp - (prev_day['high'] - prev_day['low'])

    pivots_for_log = pd.DataFrame({'PP': pp, 'R1': r1, 'S1': s1}, index=prev_day.index)
    print("-> Przykładowe obliczone poziomy Pivot (pierwsze 3 dni):")
    print(pivots_for_log.head(3).to_string())

    # Indeksem prev_day jest teraz data, więc nie musimy używać .date
    prev_day_date_index = prev_day.index

    pp_series = pd.Series(pp.values, index=prev_day_date_index)
    r1_series = pd.Series(r1.values, index=prev_day_date_index)
    s1_series = pd.Series(s1.values, index=prev_day_date_index)
    r2_series = pd.Series(r2.values, index=prev_day_date_index)
    s2_series = pd.Series(s2.values, index=prev_day_date_index)

    pivots_map = {'PP': pp_series, 'R1': r1_series, 'S1': s1_series, 'R2': r2_series, 'S2': s2_series}

    # Używamy tej samej kolumny tymczasowej do mapowania
    df_copy['date_map'] = df_copy.index.date

    for level_name, level_series in pivots_map.items():
        pivot_values = df_copy['date_map'].map(level_series)
        df_copy[f'dist_to_{level_name}'] = \
            (df_copy['close'] - pivot_values) / pivot_values.replace(0, np.nan)

    # Usuń obie kolumny tymczasowe
    df_copy.drop(columns=['date_map', 'date_for_grouping'], inplace=True, errors='ignore')

    print("-> Cechy oparte na Pivot Points zostały dodane.")

    return df_copy

def add_fibonacci_features(df, window=100):
    """
    Oblicza i dodaje cechy oparte na zniesieniach Fibonacciego.
    Wersja poprawiona, aby unikać FutureWarning.
    """
    # 1. Automatyczne wykrywanie swingu high/low w oknie
    swing_high = df['high'].rolling(window=window).max()
    swing_low = df['low'].rolling(window=window).min()
    swing_range = swing_high - swing_low
    swing_range = swing_range.replace(0, np.nan)

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
    df['FIBO_relative_position'] = (df['close'] - swing_low) / swing_range

    distances = fibo_df.sub(df['close'], axis=0).abs()

    # === POPRAWIONY FRAGMENT KODU ===
    # Obliczamy najbliższy poziom. To wygeneruje NaN na początku.
    nearest_level_series = distances.idxmin(axis=1)

    # Wypełniamy początkowe NaNy domyślną wartością, np. poziomem 100.0 (dołek swingu)
    # To rozwiązuje problem i sprawia, że kod jest gotowy na przyszłe wersje pandas.
    df['FIBO_nearest_level'] = nearest_level_series.fillna('FIBO_100.0')
    # =================================

    df['FIBO_distance_to_nearest'] = distances.min(axis=1) / swing_range

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
    bullish_divergence = (df[price_low_col] < low_price_lookback) & (df[indicator_col] > low_indicator_lookback)
    bearish_divergence = (df[price_high_col] > high_price_lookback) & (df[indicator_col] < high_indicator_lookback)
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

    ohlc = {'open': 'first', 'high': 'max', 'low': 'min', 'close': lambda x: x.iloc[-1] if not x.empty else np.nan, 'volume': 'sum', 'turnover': 'sum'}

    all_dfs = {}
    for tf_name, tf_pandas in timeframes.items():
        if tf_name == '5m':
            all_dfs['5m'] = df_5m_raw.copy()
        else:
            all_dfs[tf_name] = df_5m_raw.resample(tf_pandas).agg(ohlc).dropna()

    for tf_name, df in all_dfs.items():
        print(f"Obliczanie wskaźników dla interwału {tf_name}...")
        df.ta.rsi(append=True); df.ta.atr(append=True); df.ta.macd(append=True); df.ta.bbands(append=True)
        df.ta.stoch(append=True); df.ta.adx(append=True); df.ta.obv(append=True); df.ta.vwap(append=True)
        df.ta.cci(append=True); df.ta.mfi(append=True); df.ta.aroon(append=True)

        df.ta.ema(length=20, append=True); df.ta.ema(length=50, append=True); df.ta.ema(length=200, append=True)
        df.ta.dema(length=50, append=True); df.ta.tema(length=50, append=True)

        if 'EMA_200' in df.columns:
            df['dist_from_ema_200'] = (df['close'] - df['EMA_200']) / df['EMA_200']
        if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
            df['ema_cross_signal'] = (df['EMA_20'] > df['EMA_50']).astype(int)

        # --- Zaawansowane wskaźniki i cechy ---

        # Dywergencja dla RSI
        rsi_col_name = 'RSI_14'
        if rsi_col_name in df.columns:
            df = add_divergence_feature(df, indicator_col=rsi_col_name)

        # SuperTrend (tylko kierunek)
        st_df = df.ta.supertrend(append=False)
        if st_df is not None and not st_df.empty:
            direction_col_name = next((col for col in st_df.columns if 'SUPERTd' in col), None)
            if direction_col_name: df[direction_col_name] = st_df[direction_col_name]

        # Ichimoku
        ichimoku_df = df.ta.ichimoku(append=False)[0]
        df = pd.concat([df, ichimoku_df], axis=1)

        # Cechy wyższego rzędu (pęd i zmienność wskaźników)
        if 'RSI_14' in df.columns:
            df['RSI_14_roc_1'] = df['RSI_14'].diff()
            df['RSI_14_vol_10'] = df['RSI_14'].rolling(window=10).std()
        if 'MACDh_12_26_9' in df.columns:
            df['MACDh_12_26_9_roc_1'] = df['MACDh_12_26_9'].diff()

        df.ta.squeeze(append=True)
        df.ta.donchian(append=True)
        df.ta.pvo(append=True)
        df.ta.kvo(append=True)

        # Parabolic SAR (w inteligentny sposób, aby uniknąć NaN)
        psar_df = df.ta.psar(append=False)
        if psar_df is not None and not psar_df.empty:
            reversal_col = next((col for col in psar_df.columns if 'PSARr' in col), None)
            if reversal_col: df[reversal_col] = psar_df[reversal_col]

        # Automatyczne rozpoznawanie formacji świecowych
        df.ta.cdl_pattern(name="all", append=True)

        # === DODANIE CECH FIBONACCIEGO ===
        print(f"Dodawanie cech Fibonacciego dla interwału {tf_name}...")
        df = add_fibonacci_features(df, window=100)

        # === NOWY BLOK: Dodanie cech ZigZag ===
        df = add_zigzag_features(df)
        # ====================================

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
        df_with_suffix = df_to_merge.drop(columns=['open', 'high', 'low', 'close', 'volume', 'turnover'], errors='ignore').add_suffix(f'_{tf_name}')
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

    pa_features_to_add = [col for col in pa_df.columns if col not in ['open', 'high', 'low', 'close', 'volume', atr_col_name]]
    final_df = pd.concat([final_df, pa_df[pa_features_to_add]], axis=1)

    # === TUTAJ WKLEJ BLOK DIAGNOSTYCZNY ===
    print("\n[DIAGNOSTYKA] Sprawdzanie indeksu PRZED wywołaniem add_pivot_points:")
    print(f"Typ indeksu: {type(final_df.index)}")
    print("Pierwsze 5 wartości indeksu:")
    print(final_df.index[:5])
    # ======================================

    # === DODANIE DZIENNYCH PIVOT POINTS ===
    final_df = add_pivot_points(final_df)

    adx_col_name = f'ADX_14_{base_tf}'
    if adx_col_name in final_df.columns:
        final_df['market_regime_trending'] = (final_df[adx_col_name] > 25).astype(int)

    print("Dodawanie cech czasowych...")
    if not pd.api.types.is_datetime64_any_dtype(final_df.index):
        final_df.index = pd.to_datetime(final_df.index)

    day_of_week, hour_of_day = final_df.index.dayofweek, final_df.index.hour
    final_df['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
    final_df['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
    final_df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    final_df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    print(f"Zakończono przygotowywanie cech. Finalny kształt danych: {final_df.shape}")
    return final_df