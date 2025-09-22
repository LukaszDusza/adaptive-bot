# model_trainer.py
import pandas as pd
import numpy as np
import joblib
import json
import asyncio
from typing import Dict, Any

# --- NOWE IMPORTY ---
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

import config_trainer
from async_data_fetcher import fetch_data_for_trainer_async
from utils.data_preparer import prepare_full_feature_set

# --- Konfiguracja Treningu ---
TARGET_COLUMN_NAME = 'target'
LOOKAHEAD_BARS = 1  # NAPRAWIONY: Używamy 1 bar zamiast 12 aby uniknąć lookahead bias
TEST_SIZE = 0.2  # Ten parametr nie jest już używany do walidacji, ale może być przydatny w przyszłości
RANDOM_STATE = 42
CV_SPLITS = 5  # Liczba podziałów w walidacji krzyżowej
HOLDOUT_SIZE = 0.2  # 20% ostatnich danych jako holdout test set

# --- Konfiguracja Modeli Ekspertów ---
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "momentum": {
        "model": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1),
        "feature_keywords": ['RSI', 'MACD', 'ADX'],
        "description": "Ekspert #1: Model Pędu (Momentum)"
    },
    "reversion": {
        "model": LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1),
        "feature_keywords": ['BB', 'STOCH'],
        "description": "Ekspert #2: Model Powrotu do Średniej (Mean Reversion)"
    },
    "price_action": {
        "model": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1),
        "feature_keywords": ['impulse_strength', 'volatility_burst', 'closing_position', 'volume_spike'],
        "description": "Ekspert #3: Model Wzorców Świecowych (Price Action)"
    }
}


def train_and_evaluate_expert(
        df: pd.DataFrame,
        model_name: str,
        model_config: Dict[str, Any],
        ticker_name: str
) -> None:
    """
    Trenuje i ocenia standardowy model eksperta przy użyciu proper holdout validation
    i purged cross-validation dla szeregów czasowych.
    """
    print(f"\n[KROK] Walidacja i Trening: {model_config['description']}...")

    # 1. Wybór cech
    features = [col for col in df.columns if any(keyword in col for keyword in model_config['feature_keywords'])]

    # 2. Przygotowanie danych
    model_data = df.dropna(subset=[TARGET_COLUMN_NAME] + features)
    
    # 3. Podział na train/holdout (temporal split)
    holdout_split_idx = int(len(model_data) * (1 - HOLDOUT_SIZE))
    train_data = model_data.iloc[:holdout_split_idx]
    holdout_data = model_data.iloc[holdout_split_idx:]
    
    print(f"Train data: {len(train_data)} samples, Holdout data: {len(holdout_data)} samples")
    
    X_train_full = train_data[features]
    y_train_full = train_data[TARGET_COLUMN_NAME]
    X_holdout = holdout_data[features]
    y_holdout = holdout_data[TARGET_COLUMN_NAME]

    # 4. Purged Cross-Validation na danych treningowych
    print(f"Przeprowadzanie purged cross-validation dla '{model_name}'...")
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=3)  # Gap=3 usuwa overlapping samples
    scores = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X_train_full)):
        X_train, X_test = X_train_full.iloc[train_index], X_train_full.iloc[test_index]
        y_train, y_test = y_train_full.iloc[train_index], y_train_full.iloc[test_index]

        model_for_cv = clone(model_config['model'])
        model_for_cv.fit(X_train, y_train)

        y_pred = model_for_cv.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
        print(f"  Fold {fold + 1}/{CV_SPLITS} | Accuracy: {score:.4f}")

    print(f"-> Średnia dokładność CV: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # 5. Finalny trening na wszystkich danych treningowych (bez holdout)
    print(f"Trening finalnego modelu '{model_name}' na danych treningowych...")
    final_model = clone(model_config['model'])
    final_model.fit(X_train_full, y_train_full)
    
    # 6. Ewaluacja na holdout set
    y_holdout_pred = final_model.predict(X_holdout)
    holdout_accuracy = accuracy_score(y_holdout, y_holdout_pred)
    print(f"-> HOLDOUT ACCURACY: {holdout_accuracy:.4f}")

    # 7. Zapisanie finalnego modelu i cech
    joblib.dump(final_model, f'expert_{model_name}_{ticker_name}_5m.joblib')
    with open(f'features_{model_name}_{ticker_name}_5m.json', 'w') as f:
        json.dump(features, f)
    print(f"Model '{model_name}' został pomyślnie zapisany.")


def train_price_action_expert(
        df: pd.DataFrame,
        model_name: str,
        model_config: Dict[str, Any],
        ticker_name: str
) -> None:
    """
    Specjalistyczna funkcja do trenowania i walidacji modelu Price Action z proper
    holdout validation i purged cross-validation, oraz dodatkowym skalowaniem danych.
    """
    print(f"\n[KROK] Walidacja i Trening: {model_config['description']}...")

    # 1. Wybór cech
    features = [col for col in df.columns if any(keyword in col for keyword in model_config['feature_keywords'])]

    # 2. Przygotowanie danych
    model_data = df.dropna(subset=[TARGET_COLUMN_NAME] + features)
    
    # 3. Podział na train/holdout (temporal split)
    holdout_split_idx = int(len(model_data) * (1 - HOLDOUT_SIZE))
    train_data = model_data.iloc[:holdout_split_idx]
    holdout_data = model_data.iloc[holdout_split_idx:]
    
    print(f"Train data: {len(train_data)} samples, Holdout data: {len(holdout_data)} samples")
    
    X_train_full = train_data[features]
    y_train_full = train_data[TARGET_COLUMN_NAME]
    X_holdout = holdout_data[features]
    y_holdout = holdout_data[TARGET_COLUMN_NAME]

    # 4. Purged Cross-Validation z poprawnym skalowaniem
    print(f"Przeprowadzanie purged cross-validation dla '{model_name}'...")
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=3)  # Gap=3 usuwa overlapping samples
    scores = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X_train_full)):
        X_train, X_test = X_train_full.iloc[train_index], X_train_full.iloc[test_index]
        y_train, y_test = y_train_full.iloc[train_index], y_train_full.iloc[test_index]

        # Skaler musi być dopasowany wewnątrz pętli na danych treningowych!
        scaler_cv = StandardScaler()
        X_train_scaled = scaler_cv.fit_transform(X_train)
        X_test_scaled = scaler_cv.transform(X_test)

        model_for_cv = clone(model_config['model'])
        model_for_cv.fit(X_train_scaled, y_train)

        y_pred = model_for_cv.predict(X_test_scaled)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
        print(f"  Fold {fold + 1}/{CV_SPLITS} | Accuracy: {score:.4f}")

    print(f"-> Średnia dokładność CV: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # 5. Finalny trening na danych treningowych (bez holdout)
    print(f"Trening finalnego modelu '{model_name}' na danych treningowych...")
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train_full)

    final_model = clone(model_config['model'])
    final_model.fit(X_train_scaled, y_train_full)
    
    # 6. Ewaluacja na holdout set
    X_holdout_scaled = final_scaler.transform(X_holdout)
    y_holdout_pred = final_model.predict(X_holdout_scaled)
    holdout_accuracy = accuracy_score(y_holdout, y_holdout_pred)
    print(f"-> HOLDOUT ACCURACY: {holdout_accuracy:.4f}")

    # 7. Zapisanie finalnego modelu, skalera i cech
    joblib.dump(final_model, f'expert_{model_name}_{ticker_name}_5m.joblib')
    joblib.dump(final_scaler, f'scaler_{model_name}_{ticker_name}_5m.joblib')
    with open(f'features_{model_name}_{ticker_name}_5m.json', 'w') as f:
        json.dump(features, f)
    print(f"Model '{model_name}' i skaler zostały pomyślnie zapisane.")


async def main() -> None:
    """Główna funkcja orkiestrująca proces treningu modeli."""
    print(f"--- Rozpoczęcie procesu treningowego 'Komitetu Ekspertów' dla {config_trainer.TICKER_NAME_FOR_MODELS} ---")

    # Krok 1: Pobieranie danych
    print("\n[KROK 1/3] Pobieranie i cachowanie danych...")
    df_raw = await fetch_data_for_trainer_async(
        ticker=config_trainer.TICKER,
        start_date=config_trainer.TRAIN_START_DATE,
        end_date=config_trainer.TRAIN_END_DATE
    )
    if df_raw.empty:
        print("Nie udało się pobrać/wczytać danych. Przerwano.")
        return

    # Krok 2: Przetwarzanie danych i tworzenie zmiennej docelowej
    print("\n[KROK 2/3] Przetwarzanie danych i tworzenie cech...")
    final_df = prepare_full_feature_set(df_raw)
    final_df[TARGET_COLUMN_NAME] = (final_df['close'].shift(-LOOKAHEAD_BARS) > final_df['close']).astype(int)

    # Krok 3: Walidacja i trening modeli ekspertów
    print("\n[KROK 3/3] Uruchamianie procedur walidacji i treningu dla ekspertów...")

    # Walidacja i trening ekspertów momentum i reversion
    train_and_evaluate_expert(final_df, "momentum", MODEL_CONFIGS["momentum"], config_trainer.TICKER_NAME_FOR_MODELS)
    train_and_evaluate_expert(final_df, "reversion", MODEL_CONFIGS["reversion"], config_trainer.TICKER_NAME_FOR_MODELS)
    
    # Walidacja i trening eksperta price_action (wymaga specjalnego skalowania)
    train_price_action_expert(final_df, "pa", MODEL_CONFIGS["price_action"], config_trainer.TICKER_NAME_FOR_MODELS)

    print("\n--- WSZYSCY EKSPERCI ZOSTALI ZWALIDOWANI, WYTRENOWANI I ZAPISANI. ---")


if __name__ == "__main__":
    asyncio.run(main())