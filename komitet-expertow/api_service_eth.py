from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
import os
from pybit.unified_trading import HTTP
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler

# --- Konfiguracja Serwisu ---
TICKER = "ETHUSDT"
MODELS_TICKER = "ETH"  # Używane w nazwach plików modeli
INTERVALS_NEEDED = ['5m', '15m', '1h']
print(f"--- Uruchamianie API Service dla {TICKER} ---")

# --- Krok 1: Wczytaj wszystkich 3 ekspertów i ich zasoby ---
print("Wczytywanie modeli 'Komitetu Ekspertów'...")
models = {}
features = {}
try:
    for expert in ['momentum', 'reversion', 'pa']:
        models[expert] = joblib.load(f'expert_{expert}_{MODELS_TICKER}_5m.joblib')
        with open(f'features_{expert}_{MODELS_TICKER}_5m.json', 'r') as f:
            features[expert] = json.load(f)
    scaler_pa = joblib.load(f'scaler_pa_{MODELS_TICKER}_5m.joblib')
    print("Wszystkie modele i zasoby załadowane pomyślnie.")
except FileNotFoundError as e:
    print(f"BŁĄD KRYTYCZNY: Nie znaleziono plików modelu! {e.filename}")
    exit()  # Zakończ działanie, jeśli brakuje modeli

# --- Krok 2: Logika pobierania i przetwarzania danych (wzorowana na backtesterze) ---
session = HTTP(testnet=False)  # Używamy sesji bez kluczy do danych publicznych


def fetch_and_prepare_data(symbol: str) -> pd.DataFrame:
    print("Pobieranie i przetwarzanie świeżych danych rynkowych...")
    all_dataframes = {}
    interval_map = {'5m': 5, '15m': 15, '1h': 60}

    for interval_str in INTERVALS_NEEDED:
        bybit_interval = interval_map[interval_str]
        try:
            # Pobieramy więcej danych, aby wskaźniki i cechy opóźnione miały z czego liczyć
            response = session.get_kline(category="spot", symbol=symbol, interval=bybit_interval, limit=400)
            if response['retCode'] == 0 and response['result']['list']:
                df = pd.DataFrame(response['result']['list'],
                                  columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df.astype(float).sort_index()

                # Obliczanie wskaźników
                df.ta.rsi(append=True);
                df.ta.atr(append=True);
                df.ta.macd(append=True);
                df.ta.bbands(append=True);
                df.ta.stoch(append=True);
                df.ta.adx(append=True);
                df.ta.ema(length=50, append=True);
                df.ta.ema(length=200, append=True)

                all_dataframes[interval_str] = df
            else:
                print(f"OSTRZEŻENIE: Nie udało się pobrać danych dla interwału {interval_str}.")
                return None
        except Exception as e:
            print(f"Błąd podczas pobierania danych dla {interval_str}: {e}")
            return None

    # Łączenie danych (tak jak w backtesterze)
    df_5m = all_dataframes['5m'].add_suffix('_5m').rename(
        columns={'open_5m': 'open', 'high_5m': 'high', 'low_5m': 'low', 'close_5m': 'close', 'volume_5m': 'volume'})
    df_15m = all_dataframes['15m'].drop(columns=['open', 'high', 'low', 'close', 'volume']).add_suffix('_15m')
    df_1h = all_dataframes['1h'].drop(columns=['open', 'high', 'low', 'close', 'volume']).add_suffix('_1h')

    final_df = pd.merge_asof(df_5m, df_15m, left_index=True, right_index=True, direction='backward')
    final_df = pd.merge_asof(final_df, df_1h, left_index=True, right_index=True, direction='backward')

    # <<< START: Uzupełnione tworzenie cech Price Action >>>
    pa_df = final_df[['open', 'high', 'low', 'close', 'volume', 'ATRr_14_5m']].copy()

    # Podstawowe cechy PA
    pa_df['impulse_strength'] = (pa_df['close'] - pa_df['open']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    pa_df['closing_position'] = (pa_df['close'] - pa_df['low']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    pa_df['volume_spike'] = pa_df['volume'] / pa_df['volume'].rolling(window=20).mean().replace(0, 1)
    pa_df['volatility_burst'] = (pa_df['high'] - pa_df['low']) / pa_df['ATRr_14_5m'].replace(0, 1)

    # Cechy opóźnione (lags)
    pa_feature_names = ['impulse_strength', 'closing_position', 'volume_spike', 'volatility_burst']
    for feature in pa_feature_names:
        for i in range(1, 4):
            pa_df[f'{feature}_lag_{i}'] = pa_df[feature].shift(i)

    # Wybieramy tylko nowo utworzone kolumny do dodania
    pa_features_to_add = [col for col in pa_df.columns if
                          col not in ['open', 'high', 'low', 'close', 'volume', 'ATRr_14_5m']]
    final_df = pd.concat([final_df, pa_df[pa_features_to_add]], axis=1)
    # <<< END: Uzupełnione tworzenie cech Price Action >>>

    final_df.dropna(inplace=True)
    return final_df.tail(1)  # Zwracamy tylko ostatni, w pełni przetworzony wiersz

# --- Krok 3: Aplikacja Flask i endpoint predykcji ---
app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    print("\nOtrzymano zapytanie o predykcję...")

    # 1. Pobierz i przygotuj dane
    latest_data = fetch_and_prepare_data(TICKER)
    if latest_data is None or latest_data.empty:
        return jsonify({"error": "Nie udało się pobrać lub przetworzyć danych rynkowych."}), 500

    # 2. Wygeneruj "opinie" od wszystkich ekspertów
    expert_opinions = {}
    for expert in ['momentum', 'reversion', 'pa']:
        X = latest_data[features[expert]]
        if expert == 'pa':
            X.replace([np.inf, -np.inf], 0, inplace=True);
            X.fillna(0, inplace=True)
            X = scaler_pa.transform(X)

        prediction = int(models[expert].predict(X)[0])
        confidence = float(models[expert].predict_proba(X).max())
        expert_opinions[expert] = {"prediction": prediction, "confidence": confidence}
        print(f"Opinia Eksperta '{expert.upper()}': Predykcja={prediction}, Pewność={confidence:.2f}")

    # 3. Zwróć kompletny raport
    response = {
        "timestamp": latest_data.index[0].isoformat(),
        "ticker": TICKER,
        "current_price": float(latest_data['close'].iloc[0]),
        "atr_value_5m": float(latest_data['ATRr_14_5m'].iloc[0]),
        "expert_opinions": expert_opinions
    }

    print("Wysłano kompletną analizę do bota.")
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)