# utils/data_preparer.py
import numpy as np
import pandas as pd

import config


# === NOWA FUNKCJA POMOCNICZA DLA CECH STACJONARNYCH ===
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
    r1 = 2 * pp - prev_day['low'];
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
    nearest_level_series = distances.idxmin(axis=1)
    df['FIBO_nearest_level'] = nearest_level_series.fillna('FIBO_100.0')
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


def prepare_feature_set_for_timeframe(df_5m_raw: pd.DataFrame, base_tf: str = config.BASE_TIMEFRAME):
    print(f"Przygotowywanie zestawu cech dla interwału bazowego: {base_tf}...")
    timeframes_map = {'5m': '5T', '15m': '15T', '1h': '1H', '4h': '4H'}
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
        df.ta.macd(fast=cfg.MACD_FAST, slow=cfg.MACD_SLOW, signal=cfg.MACD_SIGNAL, append=True)
        df.ta.bbands(length=cfg.BBANDS_LENGTH, append=True)
        df.ta.stoch(k=cfg.STOCH_K, append=True)  # Domyślnie d=3, smooth_k=3
        df.ta.adx(length=cfg.ADX_LENGTH, append=True)
        df.ta.obv(append=True);
        df.ta.vwap(append=True)
        df.ta.cci(length=cfg.CCI_LENGTH, append=True)
        df.ta.mfi(length=cfg.MFI_LENGTH, append=True)
        df.ta.aroon(length=cfg.AROON_LENGTH, append=True)
        df.ta.ema(length=cfg.EMA_FAST_LEN, append=True)
        df.ta.ema(length=cfg.EMA_SLOW_LEN, append=True)
        df.ta.ema(length=cfg.EMA_TREND_LEN, append=True)
        df.ta.cdl_pattern(name="all", append=True)

        if f'RSI_{cfg.RSI_LENGTH}' in df.columns:
            df = add_divergence_feature(df, indicator_col=f'RSI_{cfg.RSI_LENGTH}')

        print(f"Dodawanie cech Fibonacciego dla interwału {tf_name}...")
        df = add_fibonacci_features(df, window=cfg.FIBO_WINDOW)

        print(f"Dodawanie cech stacjonarnych dla interwału {tf_name}...")
        cols_map = {
            'RSI': f'RSI_{cfg.RSI_LENGTH}',
            'MFI': f'MFI_{cfg.MFI_LENGTH}',
            'CCI': f'CCI_{cfg.CCI_LENGTH}',
            'MACDh': f'MACDh_{cfg.MACD_FAST}_{cfg.MACD_SLOW}_{cfg.MACD_SIGNAL}',
            'STOCHk': f'STOCHk_{cfg.STOCH_K}_3_3',
            'STOCHd': f'STOCHd_{cfg.STOCH_K}_3_3',
            'ADX': f'ADX_{cfg.ADX_LENGTH}'
        }
        cols_to_transform = [cols_map[ind] for ind in cfg.STATIONARITY_TARGET_INDICATORS if ind in cols_map]
        df = _add_stationary_features(df, columns_to_transform=cols_to_transform, window=cfg.STATIONARY_WINDOW)
        # ===============================================

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

    final_df = add_pivot_points(final_df)

    print("Dodawanie cech czasowych...")
    final_df.index = pd.to_datetime(final_df.index)
    day_of_week, hour_of_day = final_df.index.dayofweek, final_df.index.hour
    final_df['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
    final_df['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
    final_df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    final_df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    print(f"Zakończono przygotowywanie cech. Finalny kształt danych: {final_df.shape}")
    return final_df
