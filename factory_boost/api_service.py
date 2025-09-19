from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
import os
from pybit.unified_trading import HTTP

from data_processor import process_data_from_dataframe

api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")

if not api_key or not api_secret:
    print("###################################################################")
    print("### OSTRZEŻENIE: Klucze API nie znalezione w zmiennych środowiskowych.")
    print("### Używam kluczy demonstracyjnych zahardcodowanych w kodzie.")
    print("### PAMIĘTAJ: Nigdy nie umieszczaj prawdziwych kluczy w kodzie!")
    print("###################################################################")
    api_key = "pvXUTTIFpBuHDndmpf"
    api_secret = "uZnjNGSE4OZ3uHNyNU53XFMr2q9X2dlEJk46"

try:
    session = HTTP(testnet=True, api_key=api_key, api_secret=api_secret)
    print("Pomyślnie zainicjowano sesję z Bybit Testnet.")
except Exception as e:
    print(f"Błąd podczas inicjalizacji sesji Bybit: {e}")
    session = None


def fetch_ohlcv_from_exchange(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    if not session:
        print("Sesja Bybit nie jest aktywna. Nie można pobrać danych.")
        return pd.DataFrame()
    bybit_interval_map = {'1h': '60', '4h': '240', '1D': 'D'}
    if timeframe not in bybit_interval_map:
        print(f"Błąd: Nieobsługiwany interwał czasowy: {timeframe}")
        return pd.DataFrame()
    try:
        print(f"Pobieranie danych z Bybit: {symbol}, interwał: {timeframe}, limit: {limit}...")
        response = session.get_kline(
            category="linear", symbol=symbol,
            interval=bybit_interval_map[timeframe], limit=limit
        )
        if response['retCode'] == 0 and response['result']['list']:
            data = response['result']['list']
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])

            # --- POCZĄTEK POPRAWKI (FutureWarning #1) ---
            # Jawnie konwertujemy kolumnę timestamp na typ numeryczny przed użyciem 'unit'
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms', utc=True)
            # --- KONIEC POPRAWKI ---

            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            return df.iloc[::-1].reset_index(drop=True)
        else:
            print(f"Błąd API Bybit: {response['retMsg']}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Wystąpił wyjątek podczas pobierania danych z Bybit: {e}")
        return pd.DataFrame()


def get_historical_data(symbol: str) -> pd.DataFrame:
    timeframes_config = {'1h': 200, '4h': 200, '1D': 50}
    all_dfs = []
    for tf_text, limit in timeframes_config.items():
        df = fetch_ohlcv_from_exchange(symbol, tf_text, limit)
        if not df.empty:
            df['timeframe'] = tf_text
            all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.rename(columns={'open': 'open_price', 'high': 'high_price', 'low': 'low_price', 'close': 'close_price'},
                       inplace=True)
    return combined_df


try:
    print("Ładowanie modeli, scalerów i listy cech...")
    MODEL_LONG = joblib.load('trading_model_long.joblib')
    SCALER_LONG = joblib.load('scaler_long.joblib')
    MODEL_SHORT = joblib.load('trading_model_short.joblib')
    SCALER_SHORT = joblib.load('scaler_short.joblib')
    with open('feature_names_long.json', 'r') as f:
        FEATURE_NAMES = json.load(f)
    print("Modele załadowane pomyślnie.")
except FileNotFoundError as e:
    print(f"BŁĄD KRYTYCZNY: Nie znaleziono pliku modelu/scalera: {e.filename}.")
    MODEL_LONG = SCALER_LONG = MODEL_SHORT = SCALER_SHORT = FEATURE_NAMES = None

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if not all([MODEL_LONG, SCALER_LONG, FEATURE_NAMES]):
        return jsonify({"error": "Modele nie są załadowane na serwerze."}), 500
    data = request.get_json()
    if not data:
        return jsonify({"error": "Brak danych w formacie JSON w requeście."}), 400
    symbol = data.get('symbol')
    strategy = data.get('strategy')
    if not symbol or not strategy:
        return jsonify({"error": "W requeście JSON brakuje pól 'symbol' lub 'strategy'."}), 400
    raw_data_df = get_historical_data(symbol)
    if raw_data_df is None or raw_data_df.empty:
        return jsonify({"error": "Nie udało się pobrać danych z API giełdy."}), 500
    raw_data_df['timestamp'] = pd.to_datetime(raw_data_df['timestamp'])
    now = pd.Timestamp.now(tz='UTC')
    df_1h = raw_data_df[raw_data_df['timeframe'] == '1h']

    # --- POCZĄTEK POPRAWKI (FutureWarning #2) ---
    # Używamy małej litery 'h' zamiast dużej 'H'
    df_4h = raw_data_df[(raw_data_df['timeframe'] == '4h') & (raw_data_df['timestamp'] < now.floor('4h'))]
    # --- KONIEC POPRAWKI ---

    df_1d = raw_data_df[(raw_data_df['timeframe'] == '1D') & (raw_data_df['timestamp'] < now.floor('1D'))]
    sanitized_df = pd.concat([df_1h, df_4h, df_1d])
    features_df = process_data_from_dataframe(sanitized_df)
    if features_df is None or features_df.empty:
        return jsonify({"error": "Nie udało się wygenerować cech z pobranych danych."}), 500
    last_row = features_df.iloc[[-1]]
    if strategy == 'long':
        model, scaler = MODEL_LONG, SCALER_LONG
    elif strategy == 'short':
        model, scaler = MODEL_SHORT, SCALER_SHORT
    else:
        return jsonify({"error": "Nieznana strategia. Wybierz 'long' lub 'short'."}), 400
    X_live = last_row[FEATURE_NAMES]
    X_live_scaled = scaler.transform(X_live)
    prediction_numeric = model.predict(X_live_scaled)[0]
    prediction_proba = model.predict_proba(X_live_scaled)[0]
    prediction_map = {0: "SELL/HOLD", 1: "NEUTRAL", 2: "BUY"}
    result = {
        "timestamp": last_row.index[0].isoformat(), "strategy": strategy,
        "prediction": prediction_map.get(prediction_numeric, "UNKNOWN"),
        "confidence_buy": float(round(prediction_proba[2], 4)) if len(prediction_proba) > 2 else 0.0,
        "confidence_sell": float(round(prediction_proba[0], 4)) if len(prediction_proba) > 0 else 0.0
    }
    print(f"Zwrócono predykcję: {result}")
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)