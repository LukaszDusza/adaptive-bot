# model_trainer.py
import pandas as pd
import numpy as np
import joblib
import json
import asyncio  # <-- Przywracamy asyncio
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

import config_trainer
# Przywracamy import do naszego asynchronicznego fetchera
from async_data_fetcher import fetch_data_for_trainer_async
from utils.data_preparer import prepare_full_feature_set


async def main():  # <-- Funkcja znowu jest asynchroniczna
    print(f"--- Rozpoczęcie procesu treningowego 'Komitetu Ekspertów' dla {config_trainer.TICKER_NAME_FOR_MODELS} ---")

    # Krok 1: Używamy szybkiej, asynchronicznej funkcji z obsługą cache
    df_raw = await fetch_data_for_trainer_async(
        ticker=config_trainer.TICKER,
        start_date=config_trainer.TRAIN_START_DATE,
        end_date=config_trainer.TRAIN_END_DATE
    )
    if df_raw.empty:
        print("Nie udało się pobrać/wczytać danych. Przerwano.")
        return

    # --- Kroki od 2 do 5 pozostają BEZ ZMIAN ---
    # ... (cała logika treningu modeli pozostaje identyczna) ...
    print("\n[KROK 2/5] Przetwarzanie danych za pomocą standardowego preparera...")
    final_df = prepare_full_feature_set(df_raw)

    LOOKAHEAD_BARS = 12
    TARGET_COLUMN_NAME = 'target'
    final_df[TARGET_COLUMN_NAME] = (final_df['close'].shift(-LOOKAHEAD_BARS) > final_df['close']).astype(int)

    print("\n[KROK 3/5] Trening Eksperta #1: Model Pędu (Momentum)...")
    momentum_features = [col for col in final_df.columns if 'RSI' in col or 'MACD' in col or 'ADX' in col]
    model_data_momentum = final_df.dropna(subset=[TARGET_COLUMN_NAME])
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(model_data_momentum[momentum_features],
                                                                model_data_momentum[TARGET_COLUMN_NAME], test_size=0.2,
                                                                shuffle=False)
    momentum_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1);
    momentum_model.fit(X_train_m, y_train_m)
    joblib.dump(momentum_model, f'expert_momentum_{config_trainer.TICKER_NAME_FOR_MODELS}_5m.joblib')
    with open(f'features_momentum_{config_trainer.TICKER_NAME_FOR_MODELS}_5m.json', 'w') as f: json.dump(
        momentum_features, f)

    print("\n[KROK 4/5] Trening Eksperta #2: Model Powrotu do Średniej (Mean Reversion)...")
    reversion_features = [col for col in final_df.columns if 'BB' in col or 'STOCH' in col]
    model_data_reversion = final_df.dropna(subset=[TARGET_COLUMN_NAME])
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(model_data_reversion[reversion_features],
                                                                model_data_reversion[TARGET_COLUMN_NAME], test_size=0.2,
                                                                shuffle=False)
    reversion_model = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1);
    reversion_model.fit(X_train_r, y_train_r)
    joblib.dump(reversion_model, f'expert_reversion_{config_trainer.TICKER_NAME_FOR_MODELS}_5m.joblib')
    with open(f'features_reversion_{config_trainer.TICKER_NAME_FOR_MODELS}_5m.json', 'w') as f: json.dump(
        reversion_features, f)

    print("\n[KROK 5/5] Trening Eksperta #3: Model Wzorców Świecowych (Price Action)...")
    price_action_features = [col for col in final_df.columns if any(
        pa_feat in col for pa_feat in ['impulse_strength', 'volatility_burst', 'closing_position', 'volume_spike'])]
    model_data_pa = final_df.dropna(subset=[TARGET_COLUMN_NAME] + price_action_features)
    X_pa = model_data_pa[price_action_features];
    y_pa = model_data_pa[TARGET_COLUMN_NAME]
    X_pa.replace([np.inf, -np.inf], 0, inplace=True);
    X_pa.fillna(0, inplace=True)
    scaler = StandardScaler();
    X_pa_scaled = scaler.fit_transform(X_pa)
    X_train_pa, X_test_pa, y_train_pa, y_test_pa = train_test_split(X_pa_scaled, y_pa, test_size=0.2, shuffle=False)
    pa_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True);
    pa_model.fit(X_train_pa, y_train_pa)
    y_pred_pa = pa_model.predict(X_test_pa)
    print("Ocena Eksperta #3:\n", classification_report(y_test_pa, y_pred_pa, zero_division=0))
    joblib.dump(pa_model, f'expert_pa_{config_trainer.TICKER_NAME_FOR_MODELS}_5m.joblib')
    joblib.dump(scaler, f'scaler_pa_{config_trainer.TICKER_NAME_FOR_MODELS}_5m.joblib')
    with open(f'features_pa_{config_trainer.TICKER_NAME_FOR_MODELS}_5m.json', 'w') as f: json.dump(
        price_action_features, f)

    print("\n--- WSZYSCY EKSPERCI ZOSTALI WYTRENOWANI. ---")


if __name__ == "__main__":
    asyncio.run(main())  # <-- Uruchamiamy znowu przez asyncio.run()