import pandas as pd
import numpy as np
import joblib
import json
import argparse
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier


# ... (funkcja process_data_from_single_csv bez zmian) ...
def process_data_from_single_csv(csv_path: str):
    print("--- Uruchamianie procesora danych z jednego pliku CSV ---")
    try:
        df_raw = pd.read_csv(csv_path, parse_dates=['timestamp'])
    except Exception as e:
        print(f"BŁĄD KRYTYCZNY: Nie udało się wczytać pliku CSV. Szczegóły: {e}")
        return None
    df_raw.rename(columns={'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_price': 'close'},
                  inplace=True)
    timeframe_map = {5: '5m', 15: '15m', 60: '1h'};
    df_raw['timeframe'] = df_raw['timeframe'].map(timeframe_map)
    df_raw.dropna(subset=['timeframe'], inplace=True)
    all_dataframes = {}
    for tf_name, tf_df in df_raw.groupby('timeframe'):
        df_processed = tf_df.copy();
        df_processed.set_index('timestamp', inplace=True);
        df_processed.sort_index(inplace=True)
        df_processed.ta.rsi(append=True);
        df_processed.ta.atr(append=True);
        df_processed.ta.macd(append=True);
        df_processed.ta.bbands(append=True);
        df_processed.ta.stoch(append=True);
        df_processed.ta.adx(append=True);
        df_processed.ta.ema(length=50, append=True);
        df_processed.ta.ema(length=200, append=True)
        all_dataframes[tf_name] = df_processed
    df_5m = all_dataframes['5m'].add_suffix('_5m').rename(
        columns={'open_5m': 'open', 'high_5m': 'high', 'low_5m': 'low', 'close_5m': 'close', 'volume_5m': 'volume'})
    df_15m = all_dataframes['15m'].drop(columns=['open', 'high', 'low', 'close', 'volume']).add_suffix('_15m')
    df_1h = all_dataframes['1h'].drop(columns=['open', 'high', 'low', 'close', 'volume']).add_suffix('_1h')
    final_df = pd.merge_asof(df_5m, df_15m, left_index=True, right_index=True, direction='backward')
    final_df = pd.merge_asof(final_df, df_1h, left_index=True, right_index=True, direction='backward')
    final_df.dropna(inplace=True)
    return final_df


def main(args):
    # ... (Kroki 1, 2, 3 bez zmian) ...
    print(f"--- Rozpoczęcie procesu treningowego 'Komitetu Ekspertów' dla {args.ticker} 5m ---")
    print("\n[KROK 1/4] Przetwarzanie danych z jednego pliku...")
    final_df = process_data_from_single_csv(args.data_file)
    if final_df is None or final_df.empty: return

    LOOKAHEAD_BARS = 12

    print("\n[KROK 2/4] Trening Eksperta #1: Model Pędu (Momentum)...")
    momentum_features = [col for col in final_df.columns if 'RSI' in col or 'MACD' in col or 'ADX' in col]
    final_df['target_momentum'] = (final_df['close'].shift(-LOOKAHEAD_BARS) > final_df['close']).astype(int)
    model_data_momentum = final_df.dropna(subset=['target_momentum'])
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(model_data_momentum[momentum_features],
                                                                model_data_momentum['target_momentum'], test_size=0.2,
                                                                shuffle=False)
    momentum_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
    momentum_model.fit(X_train_m, y_train_m)
    joblib.dump(momentum_model, f'expert_momentum_{args.ticker}_5m.joblib')
    with open(f'features_momentum_{args.ticker}_5m.json', 'w') as f:
        json.dump(momentum_features, f)

    print("\n[KROK 3/4] Trening Eksperta #2: Model Powrotu do Średniej (Mean Reversion)...")
    reversion_features = [col for col in final_df.columns if 'BB' in col or 'STOCH' in col]
    final_df['target_reversion'] = (final_df['close'].shift(-LOOKAHEAD_BARS) > final_df['close']).astype(int)
    model_data_reversion = final_df.dropna(subset=['target_reversion'])
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(model_data_reversion[reversion_features],
                                                                model_data_reversion['target_reversion'], test_size=0.2,
                                                                shuffle=False)
    reversion_model = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
    reversion_model.fit(X_train_r, y_train_r)
    joblib.dump(reversion_model, f'expert_reversion_{args.ticker}_5m.joblib')
    with open(f'features_reversion_{args.ticker}_5m.json', 'w') as f:
        json.dump(reversion_features, f)

    # --- TRENING EKSPERTA #3: PRICE ACTION (Z POPRAWKĄ) ---
    print("\n[KROK 4/4] Trening Eksperta #3: Model Wzorców Świecowych (Price Action)...")

    print("Tworzenie 'CECH NARRACYJNYCH' dla modelu Price Action...")

    # --- KLUCZOWA POPRAWKA: Dodajemy ATRr_14_5m do kopiowanych danych ---
    pa_df = final_df[['open', 'high', 'low', 'close', 'volume', 'ATRr_14_5m']].copy()

    # Siła impulsu ostatniej świecy
    pa_df['impulse_strength'] = (pa_df['close'] - pa_df['open']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    # Kontekst zmienności
    pa_df['volatility_burst'] = (pa_df['high'] - pa_df['low']) / pa_df['ATRr_14_5m'].replace(0, 1)
    # Psychologia świecy
    pa_df['closing_position'] = (pa_df['close'] - pa_df['low']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    # Dynamika wolumenu
    pa_df['volume_spike'] = pa_df['volume'] / pa_df['volume'].rolling(window=20).mean().replace(0, 1)

    for col in ['impulse_strength', 'volatility_burst', 'closing_position', 'volume_spike']:
        for n in [1, 2, 3]:
            pa_df[f'{col}_lag_{n}'] = pa_df[col].shift(n)

    # Usuwamy kolumny robocze, zostawiając tylko finalne cechy
    price_action_features = [col for col in pa_df.columns if
                             col not in ['open', 'high', 'low', 'close', 'volume', 'ATRr_14_5m']]
    final_df = pd.concat([final_df, pa_df[price_action_features]], axis=1)
    print(f"Wybrano {len(price_action_features)} 'CECH NARRACYJNYCH'.")

    final_df['target_pa'] = (final_df['close'].shift(-LOOKAHEAD_BARS) > final_df['close']).astype(int)
    model_data_pa = final_df.dropna(subset=['target_pa'] + price_action_features)
    X_pa = model_data_pa[price_action_features]
    y_pa = model_data_pa['target_pa']

    X_pa.replace([np.inf, -np.inf], np.nan, inplace=True);
    X_pa.fillna(0, inplace=True)
    scaler = StandardScaler()
    X_pa_scaled = scaler.fit_transform(X_pa)

    X_train_pa, X_test_pa, y_train_pa, y_test_pa = train_test_split(X_pa_scaled, y_pa, test_size=0.2, shuffle=False)

    print("Trenowanie OSTATECZNEGO modelu Price Action (Sieć Neuronowa)...")
    pa_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True,
                             n_iter_no_change=10, learning_rate_init=0.001)
    pa_model.fit(X_train_pa, y_train_pa)

    print("\nOcena OSTATECZNEGO Eksperta #3 (Price Action) na danych testowych:")
    y_pred_pa = pa_model.predict(X_test_pa)
    print(classification_report(y_test_pa, y_pred_pa))

    joblib.dump(pa_model, f'expert_pa_{args.ticker}_5m.joblib')
    joblib.dump(scaler, f'scaler_pa_{args.ticker}_5m.joblib')
    with open(f'features_pa_{args.ticker}_5m.json', 'w') as f:
        json.dump(price_action_features, f)
    print(f"Zapisano OSTATECZNY model, skaler oraz listę cech dla Eksperta #3.")

    print("\n--- WSZYSCY EKSPERCI ZOSTALI WYTRENOWANI. PROCES ZAKOŃCZONY POMYŚLNIE! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trener 'Komitetu Ekspertów' dla strategii 5m.")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Ścieżka do JEDNEGO, dużego pliku CSV z danymi 5m, 15m i 1h.")
    parser.add_argument("--ticker", type=str, default="ETH", help="Nazwa tickera (np. ETH).")
    args = parser.parse_args()
    main(args)