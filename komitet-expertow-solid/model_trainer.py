# model_trainer.py
import pandas as pd
import numpy as np
import joblib
import json
import asyncio
from typing import List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

import config_trainer
from async_data_fetcher import fetch_data_for_trainer_async
from utils.data_preparer import prepare_full_feature_set

# --- Konfiguracja Treningu ---
# Przeniesione tutaj dla łatwiejszego dostępu i modyfikacji
TARGET_COLUMN_NAME = 'target'
LOOKAHEAD_BARS = 12
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Konfiguracja Modeli Ekspertów ---
# Centralne miejsce do definiowania modeli, ich parametrów i cech
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
        "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=RANDOM_STATE,
                               early_stopping=True),
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
    Trenuje, ocenia i zapisuje standardowy model eksperta (XGBoost, LightGBM).

    Args:
        df (pd.DataFrame): Pełny DataFrame z danymi i cechami.
        model_name (str): Nazwa modelu (np. 'momentum').
        model_config (Dict[str, Any]): Słownik konfiguracyjny dla danego modelu.
        ticker_name (str): Nazwa tickera używana w nazwach plików.
    """
    print(f"\n[KROK] Trening: {model_config['description']}...")

    # 1. Wybór cech
    features = [col for col in df.columns if any(keyword in col for keyword in model_config['feature_keywords'])]

    # 2. Przygotowanie danych
    model_data = df.dropna(subset=[TARGET_COLUMN_NAME] + features)
    X = model_data[features]
    y = model_data[TARGET_COLUMN_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )

    # 3. Trening modelu
    model = model_config['model']
    model.fit(X_train, y_train)

    # 4. Ocena i zapisanie wyników
    y_pred = model.predict(X_test)
    print(f"Ocena Eksperta '{model_name}':\n", classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, f'expert_{model_name}_{ticker_name}_5m.joblib')
    with open(f'features_{model_name}_{ticker_name}_5m.json', 'w') as f:
        json.dump(features, f)


def train_price_action_expert(
        df: pd.DataFrame,
        model_name: str,
        model_config: Dict[str, Any],
        ticker_name: str
) -> None:
    """
    Specjalistyczna funkcja do trenowania, oceny i zapisu modelu Price Action (MLP),
    który wymaga dodatkowego skalowania danych.

    Args:
        df (pd.DataFrame): Pełny DataFrame z danymi i cechami.
        model_name (str): Nazwa modelu ('price_action').
        model_config (Dict[str, Any]): Słownik konfiguracyjny dla modelu.
        ticker_name (str): Nazwa tickera używana w nazwach plików.
    """
    print(f"\n[KROK] Trening: {model_config['description']}...")

    # 1. Wybór cech
    features = [col for col in df.columns if any(keyword in col for keyword in model_config['feature_keywords'])]

    # 2. Przygotowanie danych
    model_data = df.dropna(subset=[TARGET_COLUMN_NAME] + features)
    X = model_data[features]
    y = model_data[TARGET_COLUMN_NAME]

    # Specyficzna obróbka dla Price Action
    X.replace([np.inf, -np.inf], 0, inplace=True)
    X.fillna(0, inplace=True)

    # 3. Skalowanie danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, shuffle=False
    )

    # 4. Trening modelu
    model = model_config['model']
    model.fit(X_train, y_train)

    # 5. Ocena i zapisanie wyników
    y_pred = model.predict(X_test)
    print(f"Ocena Eksperta '{model_name}':\n", classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, f'expert_{model_name}_{ticker_name}_5m.joblib')
    joblib.dump(scaler, f'scaler_{model_name}_{ticker_name}_5m.joblib')  # Zapisujemy również scaler
    with open(f'features_{model_name}_{ticker_name}_5m.json', 'w') as f:
        json.dump(features, f)


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

    # Krok 3: Trening, ocena i zapisywanie modeli ekspertów
    print("\n[KROK 3/3] Uruchamianie procedur treningowych dla ekspertów...")

    # Trening standardowych ekspertów
    train_and_evaluate_expert(final_df, "momentum", MODEL_CONFIGS["momentum"], config_trainer.TICKER_NAME_FOR_MODELS)
    train_and_evaluate_expert(final_df, "reversion", MODEL_CONFIGS["reversion"], config_trainer.TICKER_NAME_FOR_MODELS)

    # Trening eksperta wymagającego specjalnego traktowania
    train_price_action_expert(final_df, "price_action", MODEL_CONFIGS["price_action"],
                              config_trainer.TICKER_NAME_FOR_MODELS)

    print("\n--- WSZYSCY EKSPERCI ZOSTALI WYTRENOWANI I ZAPISANI. ---")


if __name__ == "__main__":
    asyncio.run(main())