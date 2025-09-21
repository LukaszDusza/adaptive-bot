# utils/async_data_fetcher.py
import os
import sys
import asyncio
from datetime import datetime, timezone, timedelta
import pandas as pd
from pybit.unified_trading import HTTP
from tqdm.asyncio import tqdm as asyncio_tqdm
from dotenv import load_dotenv

# --- Konfiguracja ---
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
MAX_CONCURRENT_REQUESTS = 10  # Ile zapytań do API wysyłać jednocześnie
API_SLEEP_SECONDS = 0.1  # Krótka przerwa między zapytaniami w ramach jednego zadania


def get_bybit_session():
    """Tworzy i zwraca sesję API Bybit."""
    if not API_KEY or not API_SECRET:
        print("BŁĄD: Klucze API BYBIT_API_KEY i BYBIT_API_SECRET nie są ustawione.")
        sys.exit(1)
    try:
        return HTTP(testnet=False, api_key=API_KEY, api_secret=API_SECRET)
    except Exception as e:
        print(f"Błąd podczas inicjalizacji sesji Bybit: {e}")
        sys.exit(1)


async def _fetch_chunk(session, semaphore, ticker, start_ts, end_ts):
    """Pobiera pojedynczy fragment danych w pętli, aż do skutku."""
    all_data = []
    current_ts = start_ts
    while current_ts < end_ts:
        async with semaphore:
            response = await asyncio.to_thread(
                session.get_kline,
                category="linear", symbol=ticker, interval=5,
                start=current_ts, limit=1000
            )
            await asyncio.sleep(API_SLEEP_SECONDS)

        if response and response.get('retCode') == 0 and response['result']['list']:
            data = response['result']['list']
            if not data: break
            all_data.extend(data)
            current_ts = int(data[-1][0]) + (5 * 60 * 1000)
        else:
            # W razie błędu API, ponów próbę dla tego samego fragmentu po chwili
            await asyncio.sleep(1)
            continue
    return all_data


async def fetch_data_for_trainer_async(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Główna funkcja orkiestrująca. Dzieli zadany okres na mniejsze części
    i pobiera je wszystkie współbieżnie.
    """
    session = get_bybit_session()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Dzielimy cały okres na 30-dniowe fragmenty
    date_chunks = []
    current_start = start_dt
    while current_start < end_dt:
        current_end = current_start + timedelta(days=30)
        date_chunks.append((
            int(current_start.timestamp() * 1000),
            int(min(current_end, end_dt).timestamp() * 1000)
        ))
        current_start = current_end

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    print(f"Dzielenie zadania na {len(date_chunks)} mniejszych części i uruchamianie współbieżnego pobierania...")

    tasks = [
        _fetch_chunk(session, semaphore, ticker, start_ts, end_ts)
        for start_ts, end_ts in date_chunks
    ]

    results = await asyncio_tqdm.gather(tasks, desc="Pobieranie danych")

    # Składamy wszystkie części w jedną całość
    all_data_flat = [item for sublist in results for item in sublist]
    if not all_data_flat:
        return pd.DataFrame()

    df = pd.DataFrame(all_data_flat, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')

    # Sortowanie i usuwanie duplikatów dla pewności
    df.sort_values(by='timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.set_index('timestamp', inplace=True)

    print(f"\nZakończono pobieranie. Pobrane {len(df)} unikalnych świec 5-minutowych.")
    return df