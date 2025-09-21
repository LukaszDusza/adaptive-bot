import os
import sys
import time
import argparse
from datetime import datetime, timezone
import pandas as pd
from pybit.unified_trading import HTTP

# --- Konfiguracja i połączenie z API Bybit ---
# Używamy zmiennych środowiskowych, tak jak w pozostałych skryptach.
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")

# Maksymalna liczba świeczek do pobrania w jednym zapytaniu API
MAX_LIMIT_PER_REQUEST = 1000
# Krótka pauza między zapytaniami, aby nie przekroczyć limitów API (5 zapytań na sekundę)
API_SLEEP_SECONDS = 0.25


def get_bybit_session():
    """Inicjalizuje i zwraca sesję API Bybit."""
    if not api_key or not api_secret:
        print("BŁĄD: Klucze API BYBIT_API_KEY i BYBIT_API_SECRET nie są ustawione w zmiennych środowiskowych.")
        sys.exit(1)
    try:
        session = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
        print("Pomyślnie zainicjowano sesję z Bybit.")
        return session
    except Exception as e:
        print(f"Błąd podczas inicjalizacji sesji Bybit: {e}")
        sys.exit(1)


def map_interval_to_bybit(interval_str: str) -> str:
    """Mapuje interwały '1h', '4h', '1D' na format używany przez API Bybit."""
    mapping = {'1m': '1', '5m': '5', '15m': '15', '1h': '60', '4h': '240', '1D': 'D', '1W': 'W'}
    if interval_str not in mapping:
        raise ValueError(f"Nieobsługiwany interwał: {interval_str}. Dostępne: {list(mapping.keys())}")
    return mapping[interval_str]


def map_bybit_interval_to_minutes(interval_str: str) -> int:
    """Mapuje interwały w formacie Bybit na minuty, aby pasowały do formatu z Twojego pliku."""
    # To jest format, który widzieliśmy w Twoim przykładowym CSV (60 dla 1h)
    mapping = {'1': 1, '5': 5, '15': 15, '60': 60, '240': 240, 'D': 1440, 'W': 10080}
    if interval_str not in mapping:
        raise ValueError(f"Nieobsługiwany interwał Bybit: {interval_str}")
    return mapping[interval_str]


def fetch_historical_data(session: HTTP, symbol: str, interval: str, start_dt: datetime,
                          end_dt: datetime) -> pd.DataFrame:
    """Pobiera dane historyczne dla danego symbolu i interwału w zadanym zakresie dat, obsługując paginację."""

    bybit_interval = map_interval_to_bybit(interval)
    all_data = []

    current_end_timestamp_ms = int(end_dt.timestamp() * 1000)
    start_timestamp_ms = int(start_dt.timestamp() * 1000)

    print(f"\n--- Rozpoczynanie pobierania dla interwału: {interval} ---")

    while True:
        try:
            response = session.get_kline(
                category="linear",
                symbol=symbol,
                interval=bybit_interval,
                end=current_end_timestamp_ms,
                limit=MAX_LIMIT_PER_REQUEST
            )

            if response['retCode'] == 0 and response['result']['list']:
                data = response['result']['list']
                oldest_timestamp_ms = int(data[-1][0])

                # Dodajemy dane do naszej listy
                all_data.extend(data)

                print(
                    f"Pobrano {len(data)} rekordów. Najstarszy: {datetime.fromtimestamp(oldest_timestamp_ms / 1000, tz=timezone.utc)}")

                # Jeśli najstarszy rekord jest już starszy niż nasza data początkowa, kończymy
                if oldest_timestamp_ms <= start_timestamp_ms:
                    print("Osiągnięto zadaną datę początkową. Zakończono pobieranie dla tego interwału.")
                    break

                # Ustawiamy 'end' dla kolejnego zapytania na timestamp najstarszego rekordu, który właśnie pobraliśmy
                current_end_timestamp_ms = oldest_timestamp_ms
            else:
                print("Brak więcej danych lub błąd API. Zakończono pobieranie.")
                break

        except Exception as e:
            print(f"Wystąpił błąd podczas zapytania do API: {e}")
            break

        time.sleep(API_SLEEP_SECONDS)

    if not all_data:
        return pd.DataFrame()

    # Tworzenie DataFrame i wstępne czyszczenie
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms', utc=True)

    # Usuwamy duplikaty i sortujemy
    df.drop_duplicates(subset='timestamp', inplace=True)
    # Filtrujemy dane, aby upewnić się, że są w zadanym zakresie
    df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]

    return df


def main():
    parser = argparse.ArgumentParser(description="Pobiera dane historyczne z Bybit i zapisuje do pliku CSV.")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol do pobrania (np. ICPUSDT)")
    parser.add_argument("--intervals", nargs='+', required=True, help="Lista interwałów do pobrania (np. 1h 4h 1D)")
    parser.add_argument("--start-date", type=str, required=True, help="Data początkowa w formacie YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="Data końcowa w formacie YYYY-MM-DD")
    parser.add_argument("--output-file", type=str, required=True, help="Nazwa pliku wyjściowego CSV")
    args = parser.parse_args()

    session = get_bybit_session()

    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=23, minute=59, second=59)

    all_dfs = []
    for interval in args.intervals:
        df = fetch_historical_data(session, args.symbol, interval, start_dt, end_dt)
        if not df.empty:
            df['timeframe'] = map_bybit_interval_to_minutes(map_interval_to_bybit(interval))
            all_dfs.append(df)

    if not all_dfs:
        print("Nie udało się pobrać żadnych danych. Kończenie pracy.")
        return

    print("\nŁączenie i formatowanie wszystkich danych...")
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Sortowanie od najnowszych do najstarszych, tak jak w Twoim przykładzie
    final_df.sort_values(by='timestamp', ascending=False, inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    # Zmiana nazw kolumn na format "_price"
    final_df.rename(columns={
        'open': 'open_price',
        'high': 'high_price',
        'low': 'low_price',
        'close': 'close_price'
    }, inplace=True)

    # Dodanie brakujących kolumn, aby pasowały do docelowej struktury
    final_df['id'] = range(1, len(final_df) + 1)
    final_df['created_at'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

    # Ustawienie ostatecznej kolejności kolumn
    final_df = final_df[[
        'id', 'timeframe', 'timestamp', 'open_price', 'high_price',
        'low_price', 'close_price', 'volume', 'created_at'
    ]]

    # Zapis do pliku CSV
    final_df.to_csv(args.output_file, index=False)
    print(f"\n--- Zakończono! Zapisano {len(final_df)} rekordów do pliku: {args.output_file} ---")


if __name__ == "__main__":
    main()