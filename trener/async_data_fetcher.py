import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import config

# --- Konfiguracja ---
MAX_CONCURRENT_REQUESTS = 10
API_SLEEP_SECONDS = 0.1
CACHE_DIR = config.RAW_DATA_CACHE_DIR


def _convert_dataframe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')

    df.set_index('timestamp', inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def get_bybit_session():
    load_dotenv()
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        sys.exit("BŁĄD: Klucze API nie są ustawione.")
    try:
        return HTTP(testnet=False, api_key=api_key, api_secret=api_secret, timeout=20)
    except Exception as e:
        sys.exit(f"Błąd inicjalizacji sesji Bybit: {e}")


async def _fetch_chunk(session, semaphore, ticker, start_ts, end_ts):
    all_data = []
    current_ts = start_ts
    while current_ts < end_ts:
        async with semaphore:
            try:
                response = await asyncio.to_thread(
                    session.get_kline,
                    category="linear", symbol=ticker, interval='5',
                    start=current_ts, limit=1000
                )
                await asyncio.sleep(API_SLEEP_SECONDS)
                if response and response.get('retCode') == 0 and response['result']['list']:
                    data = response['result']['list']
                    if not data: break
                    data.sort(key=lambda k: int(k[0]))
                    all_data.extend(data)
                    current_ts = int(data[-1][0]) + (5 * 60 * 1000)
                else:
                    await asyncio.sleep(5)
                    continue
            except Exception as e:
                print(f"Wystąpił błąd podczas pobierania danych: {e}. Ponawianie próby...")
                await asyncio.sleep(10)
    return all_data


async def fetch_data_for_trainer_async(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_filename = f"{CACHE_DIR}/{ticker}_{start_date.strip()}_{end_date.strip()}.csv"

    if os.path.exists(cache_filename):
        print(f"Znaleziono dane w cache. Wczytywanie z pliku: {cache_filename}")
        df = pd.read_csv(cache_filename, parse_dates=['timestamp'])
        df = _convert_dataframe_numeric(df)
        return df

    print("Brak danych w cache. Rozpoczynanie pobierania z API...")
    session = get_bybit_session()
    start_dt = datetime.strptime(start_date.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)

    date_chunks = []
    current_start = start_dt
    while current_start < end_dt:
        current_end = current_start + timedelta(days=30)
        date_chunks.append((int(current_start.timestamp() * 1000), int(min(current_end, end_dt).timestamp() * 1000)))
        current_start = current_end

    print(f"Planowane pobranie {len(date_chunks)} fragmentów danych...")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [
        asyncio.create_task(_fetch_chunk(session, semaphore, ticker, start_ts, end_ts))
        for start_ts, end_ts in date_chunks
    ]
    results = await asyncio.gather(*tasks)
    print("Wszystkie fragmenty pobrane, łączenie wyników...")

    all_data_flat = [item for sublist in results for item in sublist]
    if not all_data_flat: return pd.DataFrame()

    df = pd.DataFrame(all_data_flat, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df.sort_values(by='timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)

    df = _convert_dataframe_numeric(df)

    print(f"\nZakończono pobieranie. Pobrane {len(df)} unikalnych świec.")
    df.to_csv(cache_filename)
    print(f"Zapisano dane do cache: {cache_filename}")

    return df
