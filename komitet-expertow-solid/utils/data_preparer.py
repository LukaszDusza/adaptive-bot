# utils/data_preparer.py
import pandas as pd
import pandas_ta as ta


def prepare_full_feature_set(df_5m_raw: pd.DataFrame):
    print("Agregowanie danych i obliczanie wszystkich wskaźników oraz cech...")
    ohlc = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    df_15m_raw = df_5m_raw.resample('15min').agg(ohlc).dropna()
    df_1h_raw = df_5m_raw.resample('1h').agg(ohlc).dropna()
    all_dataframes = {'5m': df_5m_raw, '15m': df_15m_raw, '1h': df_1h_raw}
    for tf_name, df in all_dataframes.items():
        df.ta.rsi(append=True);
        df.ta.atr(append=True);
        df.ta.macd(append=True);
        df.ta.bbands(append=True);
        df.ta.stoch(append=True);
        df.ta.adx(append=True)

    df_5m = all_dataframes['5m'].add_suffix('_5m').rename(
        columns={'open_5m': 'open', 'high_5m': 'high', 'low_5m': 'low', 'close_5m': 'close', 'volume_5m': 'volume'})
    df_15m = all_dataframes['15m'].drop(columns=['open', 'high', 'low', 'close', 'volume']).add_suffix('_15m')
    df_1h = all_dataframes['1h'].drop(columns=['open', 'high', 'low', 'close', 'volume']).add_suffix('_1h')

    final_df = pd.merge_asof(df_5m, df_15m, left_index=True, right_index=True, direction='backward')
    final_df = pd.merge_asof(final_df, df_1h, left_index=True, right_index=True, direction='backward')

    pa_df = final_df[['open', 'high', 'low', 'close', 'volume', 'ATRr_14_5m']].copy()
    pa_df['impulse_strength'] = (pa_df['close'] - pa_df['open']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    pa_df['volatility_burst'] = (pa_df['high'] - pa_df['low']) / pa_df['ATRr_14_5m'].replace(0, 1)
    pa_df['closing_position'] = (pa_df['close'] - pa_df['low']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    pa_df['volume_spike'] = pa_df['volume'] / pa_df['volume'].rolling(window=20).mean().replace(0, 1)
    for col in ['impulse_strength', 'volatility_burst', 'closing_position', 'volume_spike']:
        for n in [1, 2, 3]: pa_df[f'{col}_lag_{n}'] = pa_df[col].shift(n)

    pa_features_to_add = [col for col in pa_df.columns if
                          col not in ['open', 'high', 'low', 'close', 'volume', 'ATRr_14_5m']]
    final_df = pd.concat([final_df, pa_df[pa_features_to_add]], axis=1)
    final_df.dropna(inplace=True)
    return final_df