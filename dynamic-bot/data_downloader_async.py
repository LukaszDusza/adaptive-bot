import os
import sys
import time
import argparse
import asyncio
from datetime import datetime, timezone
import pandas as pd
from pybit.unified_trading import HTTP

# --- Konfiguracja (bez zmian) ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

MAX_LIMIT_PER_REQUEST = 1000
API_SLEEP_SECONDS = 0.05

def get_bybit_session():
    if not API_KEY or not API_SECRET:
        print("BŁĄD: Klucze API BYBIT_API_KEY i BYBIT_API_SECRET nie są ustawione w zmiennych środowiskowych.")
        sys.exit(1)
    try:
        session = HTTP(testnet=False, api_key=API_KEY, api_secret=API_SECRET)
        print("Pomyślnie zainicjowano sesję z Bybit.")
        return session
    except Exception as e:
        print(f"Błąd podczas inicjalizacji sesji Bybit: {e}")
        sys.exit(1)


def map_interval_to_bybit(interval_str: str) -> str:
    mapping = {'1m': '1', '5m': '5', '15m': '15', '1h': '60', '4h': '240', '1D': 'D', '1W': 'W'}
    if interval_str not in mapping:
        raise ValueError(f"Nieobsługiwany interwał: {interval_str}. Dostępne: {list(mapping.keys())}")
    return mapping[interval_str]


def map_bybit_interval_to_minutes(interval_str: str) -> int:
    mapping = {'1': 1, '5': 5, '15': 15, '60': 60, '240': 240, 'D': 1440, 'W': 10080}
    if interval_str not in mapping:
        raise ValueError(f"Nieobsługiwany interwał Bybit: {interval_str}")
    return mapping[interval_str]


async def fetch_historical_data_async(
        session: HTTP,
        semaphore: asyncio.Semaphore,
        symbol: str,
        interval: str,
        start_dt: datetime,
        end_dt: datetime
) -> pd.DataFrame:
    bybit_interval = map_interval_to_bybit(interval)
    all_data = []

    current_end_timestamp_ms = int(end_dt.timestamp() * 1000)
    start_timestamp_ms = int(start_dt.timestamp() * 1000)

    print(f"--- [ASYNC] Rozpoczynanie pobierania dla interwału: {interval} ---")

    while True:
        response = None
        try:
            async with semaphore:
                response = await asyncio.to_thread(
                    session.get_kline,
                    category="linear", symbol=symbol, interval=bybit_interval,
                    end=current_end_timestamp_ms, limit=MAX_LIMIT_PER_REQUEST
                )
                await asyncio.sleep(API_SLEEP_SECONDS)

            if response and response.get('retCode') == 0 and response['result']['list']:
                data = response['result']['list']
                oldest_timestamp_ms = int(data[-1][0])
                all_data.extend(data)
                print(
                    f"[{interval}] Pobrano {len(data)} rekordów. Najstarszy: {datetime.fromtimestamp(oldest_timestamp_ms / 1000, tz=timezone.utc)}")

                if oldest_timestamp_ms <= start_timestamp_ms:
                    print(f"[{interval}] Osiągnięto zadaną datę początkową. Zakończono.")
                    break

                current_end_timestamp_ms = oldest_timestamp_ms
            else:
                ret_msg = response.get('retMsg', 'Brak danych') if response else 'Brak odpowiedzi'
                print(f"[{interval}] Brak więcej danych lub błąd API ({ret_msg}). Zakończono.")
                break
        except Exception as e:
            print(f"[{interval}] Wystąpił krytyczny błąd podczas pętli pobierania: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms', utc=True)
    df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]

    # Usuwamy duplikaty na poziomie świeżo pobranych danych
    df.drop_duplicates(subset='timestamp', inplace=True)
    return df


async def main():
    parser = argparse.ArgumentParser(description="Pobiera i aktualizuje dane historyczne z Bybit w pliku CSV.")
    # ... argumenty bez zmian ...
    parser.add_argument("--symbol", type=str, required=True, help="Symbol do pobrania (np. ETHUSDT)")
    parser.add_argument("--intervals", nargs='+', required=True, help="Lista interwałów do pobrania (np. 1h 4h 1D)")
    parser.add_argument("--start-date", type=str, required=True, help="Data początkowa w formacie YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="Data końcowa w formacie YYYY-MM-DD")
    parser.add_argument("--output-file", type=str, required=True, help="Nazwa pliku wyjściowego CSV")
    args = parser.parse_args()

    existing_df = pd.DataFrame()
    if os.path.exists(args.output_file):
        print(f"Znaleziono istniejący plik: {args.output_file}. Dane zostaną zaktualizowane.")
        try:
            # Wczytujemy istniejące dane, upewniając się, że timestamp jest poprawnie parsowany
            existing_df = pd.read_csv(args.output_file, parse_dates=['timestamp'])
        except Exception as e:
            print(f"BŁĄD: Nie udało się wczytać istniejącego pliku. Przerwanie. Błąd: {e}")
            return
    else:
        print(f"Plik {args.output_file} nie istnieje. Zostanie utworzony nowy.")

    session = get_bybit_session()

    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=23, minute=59, second=59)

    semaphore = asyncio.Semaphore(5)

    tasks = [
        asyncio.create_task(
            fetch_historical_data_async(session, semaphore, args.symbol, interval, start_dt, end_dt)
        ) for interval in args.intervals
    ]

    newly_fetched_dfs = await asyncio.gather(*tasks)

    # Przetwarzanie nowo pobranych danych
    processed_new_dfs = []
    for i, df in enumerate(newly_fetched_dfs):
        if not df.empty:
            interval_str = args.intervals[i]
            df['timeframe'] = map_bybit_interval_to_minutes(map_interval_to_bybit(interval_str))
            processed_new_dfs.append(df)

    if not processed_new_dfs and existing_df.empty:
        print("Nie udało się pobrać żadnych nowych danych i brak danych istniejących. Kończenie pracy.")
        return

    # --- NOWOŚĆ: Logika łączenia, deduplikacji i porządkowania ---
    print("\nŁączenie, usuwanie duplikatów i porządkowanie danych...")

    # Łączymy istniejące dane z nowo pobranymi
    all_dfs = [existing_df] + processed_new_dfs
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Zmieniamy nazwy kolumn, aby ujednolicić format przed deduplikacją
    final_df.rename(columns={
        'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_price': 'close'
    }, inplace=True)

    # Kluczowy krok: usuwanie duplikatów na podstawie unikalnej pary (timestamp, timeframe)
    # 'keep=first' zachowa dane, które były w pliku jako pierwsze (czyli te z existing_df)
    initial_rows = len(final_df)
    final_df.drop_duplicates(subset=['timestamp', 'timeframe'], keep='first', inplace=True)
    removed_rows = initial_rows - len(final_df)
    if removed_rows > 0:
        print(f"Usunięto {removed_rows} zduplikowanych wierszy.")

    # Sortowanie całej tabeli od najnowszych do najstarszych
    final_df.sort_values(by='timestamp', ascending=False, inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    # Regeneracja kolumny 'id', aby była spójna i po kolei
    final_df['id'] = range(1, len(final_df) + 1)
    final_df['created_at'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

    # Przywracanie nazw kolumn do formatu docelowego
    final_df.rename(columns={
        'open': 'open_price', 'high': 'high_price', 'low': 'low_price', 'close': 'close_price'
    }, inplace=True)

    # Ustawienie ostatecznej kolejności kolumn
    final_df = final_df[[
        'id', 'timeframe', 'timestamp', 'open_price', 'high_price',
        'low_price', 'close_price', 'volume', 'created_at'
    ]]

    # Zapis do pliku CSV (nadpisując starą wersję nową, zaktualizowaną)
    final_df.to_csv(args.output_file, index=False)
    print(f"\n--- Zakończono! Zapisano {len(final_df)} unikalnych rekordów do pliku: {args.output_file} ---")


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        sys.exit("Ten skrypt wymaga Pythona w wersji 3.9 lub nowszej.")
    asyncio.run(main())