import pandas as pd
import pandas_ta as ta
from typing import List, Union


def load_and_prepare_timeframe(
        file_path_or_list: Union[str, List[str]],
        interval_suffix: str,
        include_ohlcv: bool = False
) -> pd.DataFrame:
    """
    Wczytuje historyczne dane z pliku CSV lub listy plików, oblicza wskaźniki techniczne
    i przygotowuje DataFrame do dalszej analizy.

    Args:
        file_path_or_list: Ścieżka do pliku CSV lub lista ścieżek.
        interval_suffix: Przyrostek do dodania do nazw kolumn (np. '1h', '4h').
        include_ohlcv: Jeśli True, w wynikowym DataFrame pozostaną kolumny OHLCV.

    Returns:
        DataFrame gotowy do użycia w modelu lub backtesterze.
    """

    # 1. Wczytywanie danych
    column_names = ['id', 'interval', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'updated_at']
    ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    if isinstance(file_path_or_list, list):
        # Wczytaj i połącz wiele plików
        df_list = [pd.read_csv(file, header=None, names=column_names, usecols=ohlcv_cols) for file in file_path_or_list]
        df = pd.concat(df_list, ignore_index=True)
    else:
        # Wczytaj pojedynczy plik
        df = pd.read_csv(file_path_or_list, header=None, names=column_names, usecols=ohlcv_cols)

    # 2. Porządkowanie danych
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]  # Usuń ewentualne duplikaty timestampów

    # 3. Obliczanie wskaźników technicznych
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.stoch(length=14, append=True)
    df.ta.ao(append=True)
    df['candle_size'] = df['high'] - df['low']
    df['body_size'] = (df['close'] - df['open']).abs()

    # Zapisz kolumny OHLCV, jeśli są potrzebne (tylko dla interwału 1h w Twoim przypadku)
    if include_ohlcv:
        ohlcv_data = df[['open', 'high', 'low', 'close', 'volume']].copy()
        ohlcv_data = ohlcv_data.add_suffix(f'_{interval_suffix}')

    # 4. Finalne przygotowanie zbioru
    # Usuwamy oryginalne kolumny OHLCV, aby zostały same wskaźniki
    df = df.drop(columns=['open', 'high', 'low', 'close', 'volume'])

    # Dodajemy sufiks do wszystkich kolumn ze wskaźnikami
    df = df.add_suffix(f'_{interval_suffix}')

    # Jeśli potrzebowaliśmy OHLCV, dołączamy je z powrotem
    if include_ohlcv:
        df = pd.concat([ohlcv_data, df], axis=1)

    return df