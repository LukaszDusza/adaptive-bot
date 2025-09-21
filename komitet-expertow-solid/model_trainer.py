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
LOOKAHEAD_BARS = 12
TEST_SIZE = 0.2  # Ten parametr nie jest już używany do walidacji, ale może być przydatny w przyszłości
RANDOM_STATE = 42
CV_SPLITS = 5  # Liczba podziałów w walidacji krzyżowej

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
    Trenuje i ocenia standardowy model eksperta przy użyciu walidacji krzyżowej
    dla szeregów czasowych, a na końcu trenuje go na pełnym zbiorze danych i zapisuje.
    """
    print(f"\n[KROK] Walidacja i Trening: {model_config['description']}...")

    # 1. Wybór cech
    features = [col for col in df.columns if any(keyword in col for keyword in model_config['feature_keywords'])]

    # 2. Przygotowanie danych
    model_data = df.dropna(subset=[TARGET_COLUMN_NAME] + features)
    X = model_data[features]
    y = model_data[TARGET_COLUMN_NAME]

    # 3. Walidacja krzyżowa (Cross-Validation)
    print(f"Przeprowadzanie walidacji krzyżowej dla '{model_name}'...")
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    scores = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model_for_cv = clone(model_config['model'])
        model_for_cv.fit(X_train, y_train)

        y_pred = model_for_cv.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
        print(f"  Fold {fold + 1}/{CV_SPLITS} | Accuracy: {score:.4f}")

    print(f"-> Średnia dokładność (Accuracy) z walidacji: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # 4. Finalny trening na pełnym zbiorze danych
    print(f"Trening finalnego modelu '{model_name}' na wszystkich danych...")
    final_model = model_config['model']
    final_model.fit(X, y)

    # 5. Zapisanie finalnego modelu i cech
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
    Specjalistyczna funkcja do trenowania i walidacji modelu Price Action (MLP),
    który wymaga dodatkowego skalowania danych.
    """
    print(f"\n[KROK] Walidacja i Trening: {model_config['description']}...")

    # 1. Wybór cech
    features = [col for col in df.columns if any(keyword in col for keyword in model_config['feature_keywords'])]

    # 2. Przygotowanie danych
    model_data = df.dropna(subset=[TARGET_COLUMN_NAME] + features)
    X = model_data[features]
    y = model_data[TARGET_COLUMN_NAME]

    # 3. Walidacja krzyżowa z poprawnym skalowaniem
    print(f"Przeprowadzanie walidacji krzyżowej dla '{model_name}'...")
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    scores = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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

    print(f"-> Średnia dokładność (Accuracy) z walidacji: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # 4. Finalny trening na pełnym zbiorze danych
    print(f"Trening finalnego modelu '{model_name}' na wszystkich danych...")
    # Skalujemy cały zbiór danych przy użyciu finalnego skalera
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    final_model = model_config['model']
    final_model.fit(X_scaled, y)

    # 5. Zapisanie finalnego modelu, skalera i cech
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

    # Walidacja i trening wszystkich ekspertów przy użyciu tej samej, standardowej funkcji
    train_and_evaluate_expert(final_df, "momentum", MODEL_CONFIGS["momentum"], config_trainer.TICKER_NAME_FOR_MODELS)
    train_and_evaluate_expert(final_df, "reversion", MODEL_CONFIGS["reversion"], config_trainer.TICKER_NAME_FOR_MODELS)
    train_and_evaluate_expert(final_df, "price_action", MODEL_CONFIGS["price_action"],
                              config_trainer.TICKER_NAME_FOR_MODELS)

    print("\n--- WSZYSCY EKSPERCI ZOSTALI ZWALIDOWANI, WYTRENOWANI I ZAPISANI. ---")


if __name__ == "__main__":
    asyncio.run(main())