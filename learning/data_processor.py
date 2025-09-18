import pandas as pd
import pandas_ta as ta
from typing import Union, List


def load_and_prepare_timeframe(source: Union[str, pd.DataFrame, List[str]], interval_suffix: str, include_ohlcv=False):
    """
    Centralna, elastyczna funkcja do wczytywania i przygotowywania danych.
    Akceptuje ścieżkę do pliku, listę ścieżek lub gotowy DataFrame.
    """
    if isinstance(source, list):
        df = pd.concat([pd.read_csv(f, header=None) for f in source])
    elif isinstance(source, str):
        df = pd.read_csv(source, header=None)
    else:
        df = source.copy()

    column_names = ['id', 'interval', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'updated_at']
    df.columns = column_names[:len(df.columns)]

    use_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[use_cols]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    # Obliczanie wskaźników
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.stoch(append=True)
    df.ta.ao(append=True)
    df['candle_size'] = df['high'] - df['low']
    df['body_size'] = (df['close'] - df['open']).abs()

    # Wybierz, które kolumny zwrócić
    if include_ohlcv:
        # Zwróć wszystko i dodaj przyrostki do wszystkich kolumn
        return df.add_suffix(f'_{interval_suffix}')
    else:
        # Zwróć tylko wskaźniki
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        return df[feature_cols].add_suffix(f'_{interval_suffix}')

