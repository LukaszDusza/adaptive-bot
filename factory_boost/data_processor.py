import pandas as pd
import pandas_ta as ta
import numpy as np


def process_data_from_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df.rename(columns={'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_price': 'close'},
              inplace=True)

    # --- POCZĄTEK POPRAWKI ---
    # Sprawdzamy, czy kolumna 'timeframe' wymaga standaryzacji.
    # Jeśli pierwsza wartość jest liczbą lub literą 'D', uruchamiamy konwersję.
    # W przeciwnym razie zakładamy, że format jest już poprawny (np. '1h', '4h').
    if not df.empty and (pd.api.types.is_numeric_dtype(df['timeframe']) or 'D' in df['timeframe'].unique()):
        print("Standaryzacja kolumny 'timeframe'...")
        df['timeframe'] = df['timeframe'].replace({'D': 1440})
        df['timeframe'] = pd.to_numeric(df['timeframe'], errors='coerce')
        df.dropna(subset=['timeframe'], inplace=True)
        df['timeframe'] = df['timeframe'].astype(int)
        timeframe_map = {5: '5m', 15: '15m', 60: '1h', 240: '4h', 1440: '1D'}
        df['timeframe'] = df['timeframe'].map(timeframe_map)
    else:
        print("Kolumna 'timeframe' jest już w poprawnym formacie tekstowym. Pomijanie standaryzacji.")
    # --- KONIEC POPRAWKI ---

    master_timeframes = ['5m', '15m', '1h', '4h', '1D']
    detected_timeframes = df['timeframe'].unique()
    timeframes_to_process = [tf for tf in master_timeframes if tf in detected_timeframes]

    print(f"Wykryto następujące interwały do przetworzenia: {timeframes_to_process}")

    if '1h' not in timeframes_to_process:
        print("BŁĄD KRYTYCZNY: W dostarczonych danych brakuje interwału bazowego '1h'.")
        return None

    all_dataframes = {}

    for tf in timeframes_to_process:
        print(f"Przetwarzanie interwału: {tf}...")
        df_tf = df[df['timeframe'] == tf].copy()
        df_tf.set_index('timestamp', inplace=True)
        df_tf.sort_index(inplace=True)

        df_tf.ta.atr(append=True);
        df_tf.ta.obv(append=True);
        df_tf.ta.rsi(append=True)
        df_tf.ta.stoch(append=True);
        df_tf.ta.ao(append=True);
        df_tf.ta.bbands(append=True)
        df_tf.ta.sma(length=20, append=True);
        df_tf.ta.vwap(append=True)

        df_tf['hour'] = df_tf.index.hour
        df_tf['dayofweek'] = df_tf.index.dayofweek
        df_tf['body_size'] = abs(df_tf['close'] - df_tf['open'])
        df_tf['candle_size'] = abs(df_tf['high'] - df_tf['low'])
        if 'SMA_20' in df_tf.columns:
            df_tf['dist_from_sma_20'] = df_tf.apply(
                lambda r: r['close'] / r['SMA_20'] - 1 if r['SMA_20'] not in [0, np.nan] else 0, axis=1)
        if 'RSI_14' in df_tf.columns:
            df_tf['rsi_lag_1'] = df_tf['RSI_14'].shift(1)
        if 'ATRr_14' in df_tf.columns:
            df_tf['atr_lag_1'] = df_tf['ATRr_14'].shift(1)

        suffix = f'_{tf}'
        if tf == '1h':
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            df_tf.rename(columns={c: f"{c}{suffix}" for c in df_tf.columns if c not in ohlcv_cols}, inplace=True)
        else:
            df_tf = df_tf.add_suffix(suffix)
        all_dataframes[tf] = df_tf

    print("Łączenie przetworzonych danych...")
    final_df = all_dataframes['1h']
    for tf in timeframes_to_process:
        if tf == '1h': continue
        df_to_merge = all_dataframes[tf]
        final_df = pd.merge_asof(final_df, df_to_merge, left_index=True, right_index=True, direction='backward')

    final_df.dropna(inplace=True)
    print("--- Przetwarzanie danych zakończone ---")
    return final_df


def process_data_from_single_csv(csv_path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    print("--- Uruchamianie procesora danych z pliku CSV ---")
    try:
        df_raw = pd.read_csv(csv_path, parse_dates=['timestamp'])
    except ValueError:
        try:
            print("Nie znaleziono kolumny 'timestamp'. Próba wczytania pliku bez nagłówka...")
            column_names = ['id', 'timeframe', 'timestamp', 'open_price', 'high_price', 'low_price', 'close_price',
                            'volume', 'created_at']
            df_raw = pd.read_csv(csv_path, header=None, names=column_names, parse_dates=['timestamp'])
        except Exception as e:
            print(f"BŁĄD: Nie udało się wczytać pliku CSV w żadnym trybie. Szczegóły: {e}")
            return None
    except Exception as e:
        print(f"BŁĄD podczas wczytywania CSV: {e}")
        return None

    if start_date and end_date:
        mask = (df_raw['timestamp'] >= start_date) & (df_raw['timestamp'] <= end_date)
        df_raw = df_raw.loc[mask]

    return process_data_from_dataframe(df_raw)