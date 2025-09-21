# services/bybit_service.py
import os
import sys
import time
from datetime import datetime
import pandas as pd
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

class BybitService:
    def __init__(self, mode='live'):
        load_dotenv()
        self.session = HTTP(testnet=False) # Uproszczone dla przykładu
        print("Bybit Service zainicjalizowany.")

    def fetch_historical_data_range(self, ticker, start_date, end_date, interval_minutes=5):
        # ... (bez zmian - ta metoda jest idealna dla backtestera) ...
        print(f"Pobieranie PEŁNEJ historii {interval_minutes}m dla backtestera...")
        # (pełna implementacja tej funkcji pozostaje bez zmian)
        # Zwraca duży DataFrame z danymi 5m
        pass

    def fetch_recent_candles(self, symbol: str, interval_minutes: int, limit: int) -> pd.DataFrame:
        """
        Pobiera określoną liczbę najnowszych świec dla danego interwału.
        TO JEST KLUCZOWA FUNKCJA DLA LIVE TRADERA.
        """
        print(f"Pobieranie OSTATNICH {limit} świec {interval_minutes}m dla live tradera...")
        try:
            response = self.session.get_kline(
                category="linear", symbol=symbol, interval=str(interval_minutes), limit=limit
            )
            if response['retCode'] == 0 and response['result']['list']:
                data = response['result']['list']
                df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
                df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                df.set_index('timestamp', inplace=True)
                # Sortujemy od najstarszej do najnowszej
                return df.sort_index()
            else:
                print(f"Nie udało się pobrać danych dla {symbol}: {response.get('retMsg')}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Wyjątek podczas pobierania danych z Bybit dla {symbol}: {e}")
            return pd.DataFrame()