from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
import os
from pybit.unified_trading import HTTP
import numpy as np

from data_processor import process_data_from_dataframe

# --- Konfiguracja i połączenie z API Bybit ---
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
    session = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
    print("Pomyślnie zainicjowano sesję z Bybit.")
except Exception as e:
    print(f"Błąd podczas inicjalizacji sesji Bybit: {e}")
    session = None


# --- Funkcje pobierania danych ---
def fetch_ohlcv_from_exchange(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    if not session: return pd.DataFrame()
    bybit_interval_map = {'1h': '60', '4h': '240', '1D': 'D'}
    if timeframe not in bybit_interval_map: return pd.DataFrame()
    try:
        response = session.get_kline(category="linear", symbol=symbol, interval=bybit_interval_map[timeframe],
                                     limit=limit)
        if response['retCode'] == 0 and response['result']['list']:
            data = response['result']['list']
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms', utc=True)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            return df.iloc[::-1].reset_index(drop=True)
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Wyjątek podczas pobierania danych z Bybit: {e}")
        return pd.DataFrame()


def get_historical_data(symbol: str) -> pd.DataFrame:
    timeframes_config = {'1h': 200, '4h': 200, '1D': 50}
    all_dfs = []
    for tf_text, limit in timeframes_config.items():
        df = fetch_ohlcv_from_exchange(symbol, tf_text, limit)
        if not df.empty:
            df['timeframe'] = tf_text
            all_dfs.append(df)
    if not all_dfs: return pd.DataFrame()
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.rename(columns={'open': 'open_price', 'high': 'high_price', 'low': 'low_price', 'close': 'close_price'},
                       inplace=True)
    return combined_df


# --- Ładowanie modeli (bez zmian) ---
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

# --- Inicjalizacja Aplikacji Flask (bez zmian) ---
app = Flask(__name__)


# --- Główny punkt końcowy API /predict (Z DODANYM ZABEZPIECZENIEM) ---
@app.route('/predict', methods=['POST'])
def predict():
    if not all([MODEL_LONG, SCALER_LONG, FEATURE_NAMES]):
        return jsonify({"error": "Modele nie są załadowane na serwerze."}), 500
    data = request.get_json()
    if not data: return jsonify({"error": "Brak danych w formacie JSON."}), 400
    symbol = data.get('symbol');
    strategy = data.get('strategy')
    if not symbol or not strategy: return jsonify({"error": "Brak 'symbol' lub 'strategy' w JSON."}), 400

    raw_data_df = get_historical_data(symbol)
    if raw_data_df is None or raw_data_df.empty: return jsonify(
        {"error": "Nie udało się pobrać danych z API giełdy."}), 500

    raw_data_df['timestamp'] = pd.to_datetime(raw_data_df['timestamp'])
    now = pd.Timestamp.now(tz='UTC')
    df_1h = raw_data_df[raw_data_df['timeframe'] == '1h']
    df_4h = raw_data_df[(raw_data_df['timeframe'] == '4h') & (raw_data_df['timestamp'] < now.floor('4h'))]
    df_1d = raw_data_df[(raw_data_df['timeframe'] == '1D') & (raw_data_df['timestamp'] < now.floor('1D'))]
    sanitized_df = pd.concat([df_1h, df_4h, df_1d])

    features_df = process_data_from_dataframe(sanitized_df)
    if features_df is None or features_df.empty: return jsonify(
        {"error": "Nie udało się wygenerować cech z danych."}), 500

    last_row = features_df.iloc[[-1]]
    if strategy == 'long':
        model, scaler = MODEL_LONG, SCALER_LONG
    elif strategy == 'short':
        model, scaler = MODEL_SHORT, SCALER_SHORT
    else:
        return jsonify({"error": "Nieznana strategia."}), 400

    X_live = last_row[FEATURE_NAMES];
    X_live_scaled = scaler.transform(X_live)
    prediction_numeric = model.predict(X_live_scaled)[0]
    prediction_proba = model.predict_proba(X_live_scaled)[0]

    action = "DO_NOTHING"
    confidence = 0.0
    take_profit_price = None
    stop_loss_price = None

    # <<< NOWE ZABEZPIECZENIE: Sprawdź, czy przewidziany indeks jest w granicach tablicy prawdopodobieństw >>>
    if prediction_numeric < len(prediction_proba):
        confidence = float(round(prediction_proba[prediction_numeric], 4))

        if strategy == 'long':
            if prediction_numeric == 2:
                action = "ENTER_LONG"
            elif prediction_numeric == 0:
                action = "EXIT_LONG"
        elif strategy == 'short':
            if prediction_numeric == 2:
                action = "ENTER_SHORT"
            elif prediction_numeric == 0:
                action = "EXIT_SHORT"

        if action in ["ENTER_LONG", "ENTER_SHORT"]:
            entry_price = float(last_row['close'].iloc[0])
            atr_value = float(last_row['ATRr_14_1h'].iloc[0])
            atr_profit_multiplier = 2.0
            atr_loss_multiplier = 1.0

            if action == "ENTER_LONG":
                take_profit_price = round(entry_price + (atr_value * atr_profit_multiplier), 4)
                stop_loss_price = round(entry_price - (atr_value * atr_loss_multiplier), 4)
            elif action == "ENTER_SHORT":
                take_profit_price = round(entry_price - (atr_value * atr_profit_multiplier), 4)
                stop_loss_price = round(entry_price + (atr_value * atr_loss_multiplier), 4)
    else:
        print(
            f"OSTRZEŻENIE: Niespójność modelu! Przewidziano klasę {prediction_numeric}, ale dostępne są tylko prawdopodobieństwa dla {len(prediction_proba)} klas. Ignorowanie sygnału.")
        # Pozostawiamy action="DO_NOTHING" i confidence=0.0

    result = {
        "timestamp": last_row.index[0].isoformat(),
        "strategy": strategy,
        "action": action,
        "confidence": confidence,
        "take_profit_price": take_profit_price,
        "stop_loss_price": stop_loss_price
    }

    print(f"Zwrócono predykcję: {result}")
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)