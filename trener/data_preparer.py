import numpy as np
import pandas as pd
import pandas_ta as ta
import config
from hurst import compute_Hc
import pywt
import statsmodels.api as sm
from tqdm import tqdm
import nolds
import warnings


def add_sample_entropy(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    Oblicza Entropię Próbkową (Sample Entropy) w ruchomym oknie.
    Zawiera mechanizm wyciszający ostrzeżenia i czyszczący wyniki.
    """
    new_col_name = f'SAMP_ENTROPY_{window}'

    log_returns = np.log(df['close'].replace(0, np.nan)).diff().dropna()

    def calculate_sampen(x):
        std_dev = np.std(x)
        if std_dev == 0:
            return np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = nolds.sampen(x, tolerance=0.25 * std_dev)
        return result

    entropy_series = (
        log_returns.rolling(window=window)
        .apply(calculate_sampen, raw=True)
    )

    df[new_col_name] = entropy_series

    # ZMIANA: Usunięcie `inplace=True` i jawne przypisanie wyniku
    df[new_col_name] = df[new_col_name].replace([np.inf, -np.inf], np.nan)

    print(f"-> Cecha Entropii Próbkowej '{new_col_name}' została dodana.")
    return df


def orthogonalize_feature(df: pd.DataFrame, base_feature: str, feature_to_orthogonalize: str,
                          window: int = 100) -> pd.DataFrame:
    """
    Ortogonalizuje `feature_to_orthogonalize` względem `base_feature` za pomocą jawnej pętli.
    Zwraca sygnał resztkowy (residual), który jest niezależny od sygnału bazowego.
    """
    new_col_name = f'{feature_to_orthogonalize}_ortho_vs_{base_feature}'

    # Inicjalizujemy nową kolumnę z wartościami NaN
    df[new_col_name] = np.nan

    # Pobieramy dane jako tablice NumPy dla wydajności
    feature_y = df[feature_to_orthogonalize].values
    feature_x = df[base_feature].values

    # Iterujemy przez dane, zaczynając od pierwszego pełnego okna
    # Używamy tqdm dla paska postępu, bo to może chwilę potrwać
    print(f"Obliczanie cechy ortogonalnej: {new_col_name}...")
    for i in tqdm(range(window, len(df)), leave=False, ncols=100):
        # Wycinamy okno z danych
        window_y = feature_y[i - window:i]
        window_x = feature_x[i - window:i]

        # Sprawdzamy, czy w oknie nie ma wartości NaN
        if np.isnan(window_y).any() or np.isnan(window_x).any():
            continue

        # Dodajemy stałą do modelu regresji (intercept)
        x_with_const = sm.add_constant(window_x, prepend=True)

        # Dopasowujemy model regresji OLS
        model = sm.OLS(window_y, x_with_const).fit()

        # Bierzemy ostatni residual i przypisujemy do odpowiedniego wiersza
        # .iloc[] jest potrzebne, aby przypisać wartość do właściwego indeksu w DataFrame
        df.iloc[i, df.columns.get_loc(new_col_name)] = model.resid[-1]

    print(f"-> Cecha ortogonalna '{new_col_name}' została dodana.")
    return df

def add_wavelet_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje cechy oparte na Ciągłej Transformacji Falkowej (CWT).
    Analizuje sygnał cenowy pod kątem energii w różnych pasmach częstotliwości (cyklach).
    """
    try:
        # Sygnał wejściowy - używamy zlogarytmowanej ceny, aby ustabilizować wariancję
        signal = np.log(df['close'].values)

        # Wybór falki - 'morl' (Morlet) jest popularna w analizie finansowej
        wavelet = 'morl'

        # Definicja skali (odpowiadającej okresom cykli w barach)
        # Analizujemy cykle od 2 do 128 barów
        scales = np.arange(2, 129)

        # Wykonanie Ciągłej Transformacji Falkowej
        # `coeffs` to macierz, gdzie wiersze to skale, a kolumny to czas
        coeffs, _ = pywt.cwt(signal, scales, wavelet)

        # Obliczenie energii (kwadrat absolutnej wartości współczynników)
        energy = np.square(np.abs(coeffs))

        # Agregacja energii do zdefiniowanych pasm cykli
        # Te pasma można dostosować w zależności od charakterystyki rynku
        bands = {
            'WAVELET_NRG_2_8': (2, 8),  # Cykle bardzo krótkoterminowe
            'WAVELET_NRG_9_24': (9, 24),  # Cykle krótkoterminowe
            'WAVELET_NRG_25_64': (25, 64),  # Cykle średnioterminowe
            'WAVELET_NRG_65_128': (65, 128)  # Cykle długoterminowe
        }

        for band_name, (min_scale, max_scale) in bands.items():
            # Znajdujemy indeksy wierszy odpowiadające danemu pasmu
            band_indices = np.where((scales >= min_scale) & (scales <= max_scale))[0]
            # Sumujemy energię w danym pasmie dla każdego punktu w czasie
            df[band_name] = np.sum(energy[band_indices, :], axis=0)

        print("-> Cechy oparte na Analizie Falkowej (CWT) zostały dodane.")

    except Exception as e:
        print(f"Błąd podczas obliczania cech falkowych: {e}")

    return df

def add_hurst_exponent(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    Oblicza Wykładnik Hursta w ruchomym oknie.
    """
    hurst_series = df['close'].rolling(window=window).apply(
        lambda x: compute_Hc(x, kind='price')[0],
        raw=True
    )
    df[f'HURST_{window}'] = hurst_series
    return df

def _add_stationary_features(df: pd.DataFrame, columns_to_transform: list, window: int) -> pd.DataFrame:
    """
    Oblicza stacjonarne wersje podanych wskaźników (diff, z-score, rank).
    """
    for col in columns_to_transform:
        if col in df.columns:
            # 1. Różnicowanie (zmiana od poprzedniego kroku)
            df[f'{col}_diff_1'] = df[col].diff(1)

            # 2. Standaryzacja w ruchomym oknie (Z-Score)
            rolling_mean = df[col].rolling(window=window).mean()
            rolling_std = df[col].rolling(window=window).std()
            # Dzielenie przez zero jest zabezpieczone przez replace
            df[f'{col}_zscore_{window}'] = (df[col] - rolling_mean) / rolling_std.replace(0, np.nan)

            # 3. Ranking Procentowy w ruchomym oknie
            df[f'{col}_rank_{window}'] = df[col].rolling(window=window).rank(pct=True)

    return df

def add_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    print("Obliczanie dziennych Pivot Points...")
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
        df_copy.index = pd.to_datetime(df_copy.index)

    df_copy['date_for_grouping'] = df_copy.index.date
    daily_agg = {'high': 'max', 'low': 'min', 'close': 'last'}
    df_daily = df_copy.groupby('date_for_grouping').agg(daily_agg)

    if len(df_daily) < 2:
        print("Ostrzeżenie: Zbyt mało dni, aby obliczyć Pivot Points. Pomijanie kroku.")
        return df

    prev_day = df_daily.shift(1)

    pp = (prev_day['high'] + prev_day['low'] + prev_day['close']) / 3
    r1 = 2 * pp - prev_day['low']
    s1 = 2 * pp - prev_day['high']
    r2 = pp + (prev_day['high'] - prev_day['low']);
    s2 = pp - (prev_day['high'] - prev_day['low'])

    pivots_map = {'PP': pp, 'R1': r1, 'S1': s1, 'R2': r2, 'S2': s2}

    df_copy['date_map'] = df_copy.index.date
    for level_name, level_series in pivots_map.items():
        pivot_values = df_copy['date_map'].map(level_series)
        df_copy[f'dist_to_{level_name}'] = (df_copy['close'] - pivot_values) / pivot_values.replace(0, np.nan)

    df_copy.drop(columns=['date_map', 'date_for_grouping'], inplace=True, errors='ignore')
    print("-> Cechy oparte na Pivot Points zostały dodane.")
    return df_copy

def add_fibonacci_features(df, window=config.FeatureConfig.FIBO_WINDOW):
    swing_high = df['high'].rolling(window=window).max()
    swing_low = df['low'].rolling(window=window).min()
    swing_range = swing_high - swing_low
    swing_range = swing_range.replace(0, np.nan)

    fibo_levels = {
        'FIBO_0.0': swing_high, 'FIBO_23.6': swing_high - swing_range * 0.236,
        'FIBO_38.2': swing_high - swing_range * 0.382, 'FIBO_50.0': swing_high - swing_range * 0.5,
        'FIBO_61.8': swing_high - swing_range * 0.618, 'FIBO_100.0': swing_low
    }
    fibo_df = pd.DataFrame(fibo_levels, index=df.index)

    df['FIBO_relative_position'] = (df['close'] - swing_low) / swing_range
    distances = fibo_df.sub(df['close'], axis=0).abs()

    # === POPRAWIONY FRAGMENT KODU ===
    # Krok 1: Zidentyfikuj wiersze, w których wszystkie odległości to NaN
    all_na_rows = distances.isnull().all(axis=1)

    # Krok 2: Zainicjuj serię z wartością domyślną
    nearest_level_series = pd.Series('FIBO_100.0', index=df.index)

    # Krok 3: Oblicz idxmin() tylko dla prawidłowych wierszy i nadpisz domyślne wartości
    # Używamy `loc`, aby uniknąć ostrzeżeń o przypisywaniu do kopii
    nearest_level_series.loc[~all_na_rows] = distances[~all_na_rows].idxmin(axis=1)

    df['FIBO_nearest_level'] = nearest_level_series
    # =================================

    df['FIBO_distance_to_nearest'] = distances.min(axis=1) / swing_range
    df['FIBO_nearest_level'] = df['FIBO_nearest_level'].str.replace('FIBO_', '').astype(float)
    return df

def add_divergence_feature(df, indicator_col, price_high_col='high', price_low_col='low',
                           window=config.FeatureConfig.DIVERGENCE_WINDOW):
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

def add_ichimoku_relational_features(df: pd.DataFrame, tenkan_col='ITS_9', kijun_col='IKS_26', span_a_col='ISA_9',
                                     span_b_col='ISB_26') -> pd.DataFrame:
    """
    Tworzy cechy relacyjne na podstawie komponentów Ichimoku.
    Należy ją wywołać PO df.ta.ichimoku().
    """
    # Sprawdzenie, czy kolumny istnieją
    required_cols = [tenkan_col, kijun_col, span_a_col, span_b_col]
    if not all(col in df.columns for col in required_cols):
        print("Ostrzeżenie: Brak kolumn Ichimoku. Pomijanie tworzenia cech relacyjnych.")
        return df

    # 1. Pozycja ceny względem Chmury (Kumo)
    # ZMIANA: Użycie operatora & zamiast 'and' i dodanie nawiasów
    conditions = [
        (df['close'] > df[span_a_col]) & (df['close'] > df[span_b_col]),
        (df['close'] < df[span_a_col]) & (df['close'] < df[span_b_col])
    ]
    choices = [1, -1]
    df['ICHIMOKU_price_vs_cloud'] = np.select(conditions, choices, default=0)

    # 2. Kolor i grubość chmury (znormalizowana przez cenę)
    df['ICHIMOKU_cloud_color'] = np.where(df[span_a_col] > df[span_b_col], 1, -1)
    # Zabezpieczenie przed dzieleniem przez zero, jeśli cena byłaby zerem
    df['ICHIMOKU_cloud_thickness'] = (df[span_a_col] - df[span_b_col]).abs() / df['close'].replace(0, np.nan)

    # 3. Przecięcie Tenkan/Kijun (sygnał pędu)
    df['ICHIMOKU_tk_cross'] = np.where(df[tenkan_col] > df[kijun_col], 1, -1)

    # 4. Odległość ceny od linii Tenkan i Kijun (znormalizowana)
    df['ICHIMOKU_dist_price_tenkan'] = (df['close'] - df[tenkan_col]) / df['close'].replace(0, np.nan)
    df['ICHIMOKU_dist_price_kijun'] = (df['close'] - df[kijun_col]) / df['close'].replace(0, np.nan)

    # Usuwamy oryginalne, przesunięte w przyszłość kolumny
    # UWAGA: W Twojej wersji pandas-ta kolumny ISA i ISB nie są przesunięte,
    # więc ich usunięcie jest opcjonalne, ale utrzymuje kod w czystości.
    df.drop(columns=[span_a_col, span_b_col], inplace=True, errors='ignore')

    return df

def prepare_feature_set_for_timeframe(df_5m_raw: pd.DataFrame, base_tf: str = config.BASE_TIMEFRAME):
    print(f"Przygotowywanie zestawu cech dla interwału bazowego: {base_tf}...")
    timeframes_map = {'5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h'}
    ohlc = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'turnover': 'sum'}
    all_dfs = {}
    for tf_name in config.FEATURE_TIMEFRAMES:
        all_dfs[tf_name] = df_5m_raw.resample(timeframes_map[tf_name]).agg(
            ohlc).dropna() if tf_name != '5m' else df_5m_raw.copy()

    cfg = config.FeatureConfig
    for tf_name, df in all_dfs.items():
        print(f"Obliczanie wskaźników dla interwału {tf_name}...")

        df.ta.rsi(length=cfg.RSI_LENGTH, append=True)

        df.ta.atr(length=cfg.ATR_LENGTH, append=True)

        macd = df.ta.macd(fast=cfg.MACD_FAST, slow=cfg.MACD_SLOW, signal=cfg.MACD_SIGNAL)
        if macd is not None and not macd.empty:
            df[f'MACDh_{cfg.MACD_FAST}_{cfg.MACD_SLOW}_{cfg.MACD_SIGNAL}'] = macd[
                f'MACDh_{cfg.MACD_FAST}_{cfg.MACD_SLOW}_{cfg.MACD_SIGNAL}']

        df.ta.bbands(length=cfg.BBANDS_LENGTH, append=True)

        # df.ta.stoch(k=cfg.STOCH_K, append=True)

        df.ta.adx(length=cfg.ADX_LENGTH, append=True)

        df.ta.obv(append=True)
        df.ta.vwap(append=True)

        # df.ta.cci(length=cfg.CCI_LENGTH, append=True)
        # df.ta.mfi(length=cfg.MFI_LENGTH, append=True)

        df.ta.aroon(length=cfg.AROON_LENGTH, append=True)

        df.ta.ema(length=cfg.EMA_FAST_LEN, append=True)
        df.ta.ema(length=cfg.EMA_SLOW_LEN, append=True)
        df.ta.ema(length=cfg.EMA_TREND_LEN, append=True)

        df.ta.ichimoku(append=True)
        df = add_ichimoku_relational_features(df)

        # df.ta.cdl_pattern(name="all", append=True)

        print(f"Dodawanie cech Fibonacciego dla interwału {tf_name}...")
        df = add_fibonacci_features(df, window=cfg.FIBO_WINDOW)

        print(f"Obliczanie Wykładnika Hursta dla interwału {tf_name}...")
        df = add_hurst_exponent(df, window=100)

        print(f"Dodawanie cech stacjonarnych dla interwału {tf_name}...")
        df = _add_stationary_features(df, window=cfg.STATIONARY_WINDOW, columns_to_transform=cfg.STATIONARITY_TARGET_INDICATORS)

        all_dfs[tf_name] = df

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

    print("Obliczanie pivot_points...")
    final_df = add_pivot_points(final_df)

    print("Obliczanie wavelet...")
    final_df = add_wavelet_features(final_df)

    base_rsi_col = f'RSI_{cfg.RSI_LENGTH}_{base_tf}'
    base_stoch_col = f'STOCHk_{cfg.STOCH_K}_3_3_{base_tf}'

    if base_rsi_col in final_df.columns and base_stoch_col in final_df.columns:
        print("Obliczanie orthogonalize...")
        final_df = orthogonalize_feature(final_df,
                                         base_feature=base_rsi_col,
                                         feature_to_orthogonalize=base_stoch_col,
                                         window=100)

    # print("Obliczanie entropy...") - cos nie dziala
    # final_df = add_sample_entropy(final_df, window=100)

    print("Dodawanie cech czasowych...")
    final_df.index = pd.to_datetime(final_df.index)
    day_of_week, hour_of_day = final_df.index.dayofweek, final_df.index.hour
    final_df['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
    final_df['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
    final_df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    final_df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    print(f"Zakończono przygotowywanie cech. Finalny kształt danych: {final_df.shape}")

    print("Czyszczenie wartości nieskończonych ('inf')...")
    final_df = final_df.replace([np.inf, -np.inf], np.nan)

    return final_df
