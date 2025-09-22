# utils/data_preparer.py
import pandas as pd
import pandas_ta as ta


def prepare_feature_set_for_timeframe(df_5m_raw: pd.DataFrame, base_tf: str = '5m'):
    """
    Agreguje dane do wybranych interwałów, oblicza wskaźniki techniczne i cechy
    price action, a następnie łączy je w jeden DataFrame bazujący na wybranym
    interwale (base_tf).

    Args:
        df_5m_raw (pd.DataFrame): Surowe dane w interwale 5-minutowym.
        base_tf (str): Interwał bazowy ('5m', '15m', '1h'), na którym trenowany będzie model.

    Returns:
        pd.DataFrame: Finalny DataFrame z cechami, gotowy do treningu.
    """
    print(f"Przygotowywanie zestawu cech dla interwału bazowego: {base_tf}...")

    # Definicje interwałów i mapowanie
    timeframes = {
        '5m': '5min',
        '15m': '15min',
        '1h': '1h',
        '4h': '4h'
    }

    if base_tf not in timeframes:
        raise ValueError(f"Nieobsługiwany interwał bazowy: {base_tf}. Dostępne: {list(timeframes.keys())}")

    ohlc = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}

    # 1. Stwórz wszystkie potrzebne DataFrame'y interwałowe na podstawie danych 5m
    all_dfs = {}
    for tf_name, tf_pandas in timeframes.items():
        if tf_name == '5m':
            all_dfs['5m'] = df_5m_raw.copy()
        else:
            all_dfs[tf_name] = df_5m_raw.resample(tf_pandas).agg(ohlc).dropna()

    # 2. Oblicz wskaźniki techniczne dla każdego interwału
    for tf_name, df in all_dfs.items():
        # Klasyczne wskaźniki
        df.ta.rsi(append=True)
        df.ta.atr(append=True)
        df.ta.macd(append=True)
        df.ta.bbands(append=True)
        df.ta.stoch(append=True)
        df.ta.adx(append=True)

        # NOWE DODANE WSKAŹNIKI
        df.ta.obv(append=True)
        df.ta.vwap(append=True)
        df.ta.cci(append=True)

        # Ichimoku zwraca kilka kolumn, więc dołączamy je wszystkie osobno
        ichimoku_df = df.ta.ichimoku(append=False)[0]
        df = pd.concat([df, ichimoku_df], axis=1)

        # Zaktualizuj DataFrame w głównym słowniku
        all_dfs[tf_name] = df

    # 3. Wybierz bazowy DataFrame i przygotuj go
    base_df = all_dfs[base_tf].add_suffix(f'_{base_tf}')
    base_df.rename(columns={
        f'open_{base_tf}': 'open',
        f'high_{base_tf}': 'high',
        f'low_{base_tf}': 'low',
        f'close_{base_tf}': 'close',
        f'volume_{base_tf}': 'volume'
    }, inplace=True)

    final_df = base_df

    # 4. Dołącz wskaźniki z INNYCH interwałów
    for tf_name, df_to_merge in all_dfs.items():
        if tf_name == base_tf:
            continue

        df_with_suffix = df_to_merge.drop(columns=['open', 'high', 'low', 'close', 'volume']).add_suffix(f'_{tf_name}')
        final_df = pd.merge_asof(final_df, df_with_suffix, left_index=True, right_index=True, direction='backward')

    # 5. Oblicz cechy Price Action na bazowym interwale
    atr_col_name = f'ATRr_14_{base_tf}'
    if atr_col_name not in final_df.columns:
        raise KeyError(f"Brak kolumny ATR '{atr_col_name}' w DataFrame.")

    pa_df = final_df[['open', 'high', 'low', 'close', 'volume', atr_col_name]].copy()
    pa_df['impulse_strength'] = (pa_df['close'] - pa_df['open']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    pa_df['volatility_burst'] = (pa_df['high'] - pa_df['low']) / pa_df[atr_col_name].replace(0, 1)
    pa_df['closing_position'] = (pa_df['close'] - pa_df['low']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    pa_df['volume_spike'] = pa_df['volume'] / pa_df['volume'].rolling(window=20).mean().replace(0, 1)

    for col in ['impulse_strength', 'volatility_burst', 'closing_position', 'volume_spike']:
        for n in [1, 2, 3]:
            pa_df[f'{col}_lag_{n}'] = pa_df[col].shift(n)

    pa_features_to_add = [col for col in pa_df.columns if
                          col not in ['open', 'high', 'low', 'close', 'volume', atr_col_name]]
    final_df = pd.concat([final_df, pa_df[pa_features_to_add]], axis=1)

    # 6. DODANIE CECHY REŻIMU RYNKOWEGO
    adx_col_name = f'ADX_14_{base_tf}'
    if adx_col_name in final_df.columns:
        final_df['market_regime_trending'] = (final_df[adx_col_name] > 25).astype(int)

    # 7. Oczyszczenie
    final_df.dropna(inplace=True)

    print(f"Zakończono przygotowywanie cech. Finalny kształt danych: {final_df.shape}")
    return final_df