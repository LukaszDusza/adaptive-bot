# utils/data_fetcher.py
import os
import sys
import time
import pandas as pd
from datetime import datetime, timezone
from urllib.parse import urlencode
from dotenv import load_dotenv
import requests
import hmac
import hashlib


def fetch_data_for_trainer(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pobiera dane 5m, używając 'surowych' zapytań HTTP, bez paska postępu tqdm.
    """
    print("Uruchamianie pobierania danych (wersja diagnostyczna bez paska postępu)...")

    load_dotenv()
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        print("BŁĄD: Klucze API nie są ustawione.")
        sys.exit(1)

    all_data = []
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    current_ts = start_ts

    base_url = "https://api.bybit.com"
    endpoint = "/v5/market/kline"

    chunk_count = 1

    while current_ts < end_ts:
        params = {
            "category": "linear", "symbol": ticker, "interval": "5",
            "start": current_ts, "limit": 1000
        }

        timestamp = str(int(time.time() * 1000))
        query_string = urlencode(params)
        signature_payload = timestamp + api_key + "5000" + query_string
        signature = hmac.new(api_secret.encode('utf-8'), signature_payload.encode('utf-8'), hashlib.sha256).hexdigest()

        headers = {
            'X-BAPI-API-KEY': api_key, 'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': "5000", 'X-BAPI-SIGN': signature
        }

        url = base_url + endpoint + "?" + query_string

        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            json_response = response.json()

            if json_response.get("retCode") == 0 and json_response['result']['list']:
                data = json_response['result']['list']
                all_data.extend(data)
                last_ts_in_chunk = int(data[-1][0])

                last_date = datetime.fromtimestamp(last_ts_in_chunk / 1000, tz=timezone.utc)
                print(f"Pobrano fragment #{chunk_count}. Najnowsza data w paczce: {last_date.strftime('%Y-%m-%d')}")

                chunk_count += 1
                current_ts = last_ts_in_chunk + (5 * 60 * 1000)
            else:
                if not json_response.get('result', {}).get('list'):
                    print("\nBrak więcej danych od giełdy. Zakończono.")
                    break
                else:
                    print(f"Błąd API Bybit: {json_response.get('retMsg')}. Czekam 10s...")
                    time.sleep(10)

        except requests.exceptions.RequestException as e:
            print(f"Błąd połączenia: {e}. Czekam 10s...")
            time.sleep(10)

        time.sleep(0.1)

    if not all_data: return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
    df.sort_values(by='timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.set_index('timestamp', inplace=True)

    print(f"\nZakończono pobieranie. Pobrane {len(df)} unikalnych świec.")
    return df