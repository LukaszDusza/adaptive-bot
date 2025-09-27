import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
import ccxt.async_support as ccxt
import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import config

# --- Konfiguracja ---
MAX_CONCURRENT_REQUESTS = 10
API_SLEEP_SECONDS = 0.1
CACHE_DIR = config.RAW_DATA_CACHE_DIR


async def fetch_liquidations_async(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Pobiera historię likwidacji dla danego symbolu z Bybit.
    """
    print(f"Pobieranie danych o likwidacjach dla {ticker}...")
    exchange = ccxt.bybit({'enableRateLimit': True})

    try:
        since = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ms = int(pd.to_datetime(end_date).timestamp() * 1000)

        all_liq_data = []

        while since < end_ms:
            # Używamy zunifikowanej metody fetch_liquidations
            liquidations = await exchange.fetch_liquidations(ticker, since, limit=1000)  # Bybit zwraca max 1000

            if not liquidations:
                break

            all_liq_data.extend(liquidations)
            since = liquidations[-1]['timestamp'] + 1
            print(
                f"  Pobrano {len(liquidations)} rekordów likwidacji, ostatnia data: {pd.to_datetime(since, unit='ms')}")

        if not all_liq_data:
            print("Nie udało się pobrać danych o likwidacjach.")
            return None

        df_liq = pd.DataFrame([liq['info'] for liq in all_liq_data])
        df_liq['timestamp'] = pd.to_datetime(df_liq['execTime'], unit='ms')
        df_liq.set_index('timestamp', inplace=True)
        df_liq['qty'] = pd.to_numeric(df_liq['qty'])
        df_liq['price'] = pd.to_numeric(df_liq['price'])
        df_liq['liquidated_usd'] = df_liq['qty'] * df_liq['price']

        # Wybieramy tylko potrzebne kolumny
        df_liq = df_liq[['side', 'liquidated_usd']]

        print(f"Pobrano łącznie {len(df_liq)} pojedynczych rekordów likwidacji.")
        return df_liq

    except Exception as e:
        print(f"Wystąpił błąd podczas pobierania danych o likwidacjach: {e}")
        return None
    finally:
        await exchange.close()

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

async def fetch_funding_rate_async(ticker: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Pobiera historyczne dane o Funding Rate dla danego symbolu z Bybit.
    """
    print(f"Pobieranie danych Funding Rate dla {ticker}...")
    exchange = ccxt.bybit({'enableRateLimit': True})

    try:
        since = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ms = int(pd.to_datetime(end_date).timestamp() * 1000)

        all_funding_data = []

        while since < end_ms:
            funding_history = await exchange.fetch_funding_rate_history(ticker, since,
                                                                        limit=200)  # Bybit zwraca max 200

            if not funding_history:
                break

            all_funding_data.extend(funding_history)
            since = funding_history[-1]['timestamp'] + 1
            print(
                f"  Pobrano {len(funding_history)} rekordów Funding Rate, ostatnia data: {pd.to_datetime(since, unit='ms')}")

        if not all_funding_data:
            print("Nie udało się pobrać danych Funding Rate.")
            return None

        df_funding = pd.DataFrame(all_funding_data)
        df_funding['timestamp'] = pd.to_datetime(df_funding['timestamp'], unit='ms')
        df_funding.set_index('timestamp', inplace=True)
        df_funding = df_funding[['fundingRate']].rename(columns={'fundingRate': 'funding_rate'})
        df_funding['funding_rate'] = pd.to_numeric(df_funding['funding_rate'])

        print(f"Pobrano łącznie {len(df_funding)} rekordów Funding Rate.")
        return df_funding

    except Exception as e:
        print(f"Wystąpił błąd podczas pobierania danych Funding Rate: {e}")
        return None
    finally:
        await exchange.close()

async def fetch_open_interest_async(ticker: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Pobiera historyczne dane Open Interest dla danego symbolu i interwału z Binance.
    """
    print(f"Pobieranie danych Open Interest dla {ticker} (interwał: {timeframe})...")
    exchange = ccxt.bybit({
        'options': {'defaultType': 'future'},
        'enableRateLimit': True,
    })

    try:
        # Konwersja dat na milisekundy
        since = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ms = int(pd.to_datetime(end_date).timestamp() * 1000)

        all_oi_data = []

        while since < end_ms:
            # Binance API zwraca dane w formacie: [timestamp, open_interest]
            # Używamy `fetchOpenInterestHistory`
            oi_history = await exchange.fetch_open_interest_history(ticker, timeframe, since, limit=500)

            if not oi_history:
                break

            all_oi_data.extend(oi_history)
            # Ustaw 'since' na timestamp następnego bara, aby kontynuować pobieranie
            since = oi_history[-1]['timestamp'] + 1
            print(f"  Pobrano {len(oi_history)} rekordów OI, ostatnia data: {pd.to_datetime(since, unit='ms')}")

        if not all_oi_data:
            print("Nie udało się pobrać danych Open Interest.")
            return None

        # Tworzenie DataFrame
        df_oi = pd.DataFrame(all_oi_data)
        df_oi = df_oi.rename(columns={'info': 'open_interest', 'timestamp': 'timestamp_ms'})
        df_oi['open_interest'] = pd.to_numeric(df_oi['open_interest'])
        df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp_ms'], unit='ms')
        df_oi.set_index('timestamp', inplace=True)

        # Wybieramy tylko potrzebne kolumny i usuwamy duplikaty
        df_oi = df_oi[['open_interest']].drop_duplicates()

        print(f"Pobrano łącznie {len(df_oi)} unikalnych rekordów Open Interest.")
        return df_oi

    except Exception as e:
        print(f"Wystąpił błąd podczas pobierania danych Open Interest: {e}")
        return None
    finally:
        await exchange.close()

