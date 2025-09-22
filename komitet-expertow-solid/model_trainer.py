# model_trainer.py
import pandas as pd
import numpy as np
import joblib
import json
import asyncio
from typing import Dict, Any, List

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping

import config_trainer
from async_data_fetcher import fetch_data_for_trainer_async
from utils.data_preparer import prepare_feature_set_for_timeframe

# --- Konfiguracja Treningu ---
BASE_TIMEFRAME = '5m'
TARGET_COLUMN_NAME = 'target'
LOOKAHEAD_BARS = 1
RANDOM_STATE = 42
CV_SPLITS = 5
HOLDOUT_SIZE = 0.2
TOP_N_FEATURES = 40

# --- Konfiguracja Modeli Ekspertów ---
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "momentum": {
        "model": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1),
        "feature_keywords": ['RSI', 'MACD', 'ADX'],
        "description": "Ekspert #1: Model Pędu (Momentum)"
    },
    "reversion": {
        "model": LGBMClassifier(n_estimators=1000, max_depth=7, learning_rate=0.05, random_state=RANDOM_STATE,
                                n_jobs=-1),
        "fit_params": {"callbacks": [early_stopping(stopping_rounds=20, verbose=-1)]},
        "feature_keywords": ['BB', 'STOCH'],
        "description": "Ekspert #2: Model Powrotu do Średniej (Mean Reversion)"
    },
    "price_action": {
        "model": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1),
        "feature_keywords": ['impulse_strength', 'volatility_burst', 'closing_position', 'volume_spike'],
        "description": "Ekspert #3: Model Wzorców Świecowych (Price Action)"
    },
    "overall_best": {
        "model": LGBMClassifier(n_estimators=1000, max_depth=7, learning_rate=0.05, random_state=RANDOM_STATE,
                                n_jobs=-1),
        "fit_params": {"callbacks": [early_stopping(stopping_rounds=20, verbose=-1)]},
        "description": "Ekspert #4: Model Ogólny (na wybranych cechach)",
        "feature_keywords": None
    }
}


def train_and_evaluate_expert(
        df: pd.DataFrame,
        model_name: str,
        model_config: Dict[str, Any],
        ticker_name: str,
        timeframe: str,
        selected_features: List[str] = None
) -> None:
    print(f"\n[KROK] Walidacja i Trening: {model_config['description']}...")

    if selected_features:
        print(f"Używanie {len(selected_features)} preselekcjonowanych cech.")
        features = selected_features
    else:
        features = [col for col in df.columns if any(keyword in col for keyword in model_config['feature_keywords'])]

    model_data = df.dropna(subset=[TARGET_COLUMN_NAME] + features)
    holdout_split_idx = int(len(model_data) * (1 - HOLDOUT_SIZE))
    train_data, holdout_data = model_data.iloc[:holdout_split_idx], model_data.iloc[holdout_split_idx:]
    print(f"Train data: {len(train_data)} samples, Holdout data: {len(holdout_data)} samples")

    # Zabezpieczenie przed duplikatami w tym konkretnym podzbiorze cech
    X_train_full = train_data[features].loc[:, ~train_data[features].columns.duplicated()]
    y_train_full = train_data[TARGET_COLUMN_NAME]
    X_holdout = holdout_data[features].loc[:, ~holdout_data[features].columns.duplicated()]
    y_holdout = holdout_data[TARGET_COLUMN_NAME]

    print(f"Przeprowadzanie purged cross-validation dla '{model_name}'...")
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=3)
    scores = []
    for fold, (train_index, test_index) in enumerate(tscv.split(X_train_full)):
        X_train, X_test = X_train_full.iloc[train_index], X_train_full.iloc[test_index]
        y_train, y_test = y_train_full.iloc[train_index], y_train_full.iloc[test_index]
        model_for_cv = clone(model_config['model'])

        fit_params = model_config.get("fit_params", {}).copy()
        if "callbacks" in fit_params:
            fit_params["eval_set"] = [(X_test, y_test)]

        model_for_cv.fit(X_train, y_train, **fit_params)

        y_pred = model_for_cv.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
        print(f"  Fold {fold + 1}/{CV_SPLITS} | Accuracy: {score:.4f}")
    print(f"-> Średnia dokładność CV: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    print(f"Trening finalnego modelu '{model_name}' na danych treningowych...")
    final_model = clone(model_config['model'])
    final_model.fit(X_train_full, y_train_full)

    y_holdout_pred = final_model.predict(X_holdout)
    holdout_accuracy = accuracy_score(y_holdout, y_holdout_pred)
    print(f"-> HOLDOUT ACCURACY: {holdout_accuracy:.4f}")

    joblib.dump(final_model, f'expert_{model_name}_{ticker_name}_{timeframe}.joblib')
    with open(f'features_{model_name}_{ticker_name}_{timeframe}.json', 'w') as f:
        json.dump(features, f)
    print(f"Model '{model_name}' ({timeframe}) został pomyślnie zapisany.")


def train_price_action_expert(
        df: pd.DataFrame,
        model_name: str,
        model_config: Dict[str, Any],
        ticker_name: str,
        timeframe: str
) -> None:
    print(f"\n[KROK] Walidacja i Trening: {model_config['description']}...")
    features = [col for col in df.columns if any(keyword in col for keyword in model_config['feature_keywords'])]

    # Zabezpieczenie przed duplikatami
    unique_features = df[features].loc[:, ~df[features].columns.duplicated()].columns.tolist()

    model_data = df.dropna(subset=[TARGET_COLUMN_NAME] + unique_features)
    holdout_split_idx = int(len(model_data) * (1 - HOLDOUT_SIZE))
    train_data, holdout_data = model_data.iloc[:holdout_split_idx], model_data.iloc[holdout_split_idx:]
    print(f"Train data: {len(train_data)} samples, Holdout data: {len(holdout_data)} samples")
    X_train_full, y_train_full = train_data[unique_features], train_data[TARGET_COLUMN_NAME]
    X_holdout, y_holdout = holdout_data[unique_features], holdout_data[TARGET_COLUMN_NAME]

    print(f"Przeprowadzanie purged cross-validation dla '{model_name}'...")
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=3)
    scores = []
    for fold, (train_index, test_index) in enumerate(tscv.split(X_train_full)):
        X_train, X_test = X_train_full.iloc[train_index], X_train_full.iloc[test_index]
        y_train, y_test = y_train_full.iloc[train_index], y_train_full.iloc[test_index]
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

    print(f"Trening finalnego modelu '{model_name}' na danych treningowych...")
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train_full)
    final_model = clone(model_config['model'])
    final_model.fit(X_train_scaled, y_train_full)
    X_holdout_scaled = final_scaler.transform(X_holdout)
    y_holdout_pred = final_model.predict(X_holdout_scaled)
    holdout_accuracy = accuracy_score(y_holdout, y_holdout_pred)
    print(f"-> HOLDOUT ACCURACY: {holdout_accuracy:.4f}")

    joblib.dump(final_model, f'expert_{model_name}_{ticker_name}_{timeframe}.joblib')
    joblib.dump(final_scaler, f'scaler_{model_name}_{ticker_name}_{timeframe}.joblib')
    with open(f'features_{model_name}_{ticker_name}_{timeframe}.json', 'w') as f:
        json.dump(unique_features, f)
    print(f"Model '{model_name}' ({timeframe}) i skaler zostały pomyślnie zapisane.")


async def main() -> None:
    print(
        f"--- Rozpoczęcie procesu treningowego dla {config_trainer.TICKER_NAME_FOR_MODELS} na interwale {BASE_TIMEFRAME} ---")

    print("\n[KROK 1/5] Pobieranie i cachowanie danych 5m...")
    df_raw = await fetch_data_for_trainer_async(
        ticker=config_trainer.TICKER,
        start_date=config_trainer.TRAIN_START_DATE,
        end_date=config_trainer.TRAIN_END_DATE
    )
    if df_raw.empty: return

    print(f"\n[KROK 2/5] Przetwarzanie danych i tworzenie cech dla interwału {BASE_TIMEFRAME}...")
    final_df = prepare_feature_set_for_timeframe(df_raw, base_tf=BASE_TIMEFRAME)
    final_df[TARGET_COLUMN_NAME] = (final_df['close'].shift(-LOOKAHEAD_BARS) > final_df['close']).astype(int)
    final_df.dropna(subset=[TARGET_COLUMN_NAME], inplace=True)

    print(f"\n[KROK 3/5] Selekcja {TOP_N_FEATURES} najważniejszych cech...")
    holdout_split_idx = int(len(final_df) * (1 - HOLDOUT_SIZE))
    train_df_fs = final_df.iloc[:holdout_split_idx]

    all_features_for_selection = [col for col in final_df.columns if
                                  col not in ['open', 'high', 'low', 'close', 'volume', TARGET_COLUMN_NAME]]

    # --- KLUCZOWA POPRAWKA ---
    # Usuwamy zduplikowane kolumny przed przekazaniem danych do modelu
    train_df_fs_unique = train_df_fs.loc[:, ~train_df_fs.columns.duplicated()]
    all_features_for_selection = [col for col in train_df_fs_unique.columns if
                                  col not in ['open', 'high', 'low', 'close', 'volume', TARGET_COLUMN_NAME]]
    # --- KONIEC POPRAWKI ---

    X_train_fs = train_df_fs_unique[all_features_for_selection]
    y_train_fs = train_df_fs_unique[TARGET_COLUMN_NAME]

    selector_config = MODEL_CONFIGS["reversion"]
    selector_model = clone(selector_config['model'])
    selector_model.fit(X_train_fs, y_train_fs)

    feature_importances = pd.DataFrame({
        'feature': all_features_for_selection,
        'importance': selector_model.feature_importances_
    }).sort_values('importance', ascending=False)

    best_features = feature_importances.head(TOP_N_FEATURES)['feature'].tolist()

    print(f"-> Wybrano {len(best_features)} najlepszych cech. Top 10:")
    print(feature_importances.head(10).to_string(index=False))

    with open('best_features.json', 'w') as f:
        json.dump(best_features, f, indent=4)
    print("-> Lista najlepszych cech została zapisana do 'best_features.json'.")

    print("\n[KROK 4/5] Uruchamianie procedur walidacji i treningu dla ekspertów...")
    train_and_evaluate_expert(final_df, "momentum", MODEL_CONFIGS["momentum"], config_trainer.TICKER_NAME_FOR_MODELS,
                              BASE_TIMEFRAME)
    train_and_evaluate_expert(final_df, "reversion", MODEL_CONFIGS["reversion"], config_trainer.TICKER_NAME_FOR_MODELS,
                              BASE_TIMEFRAME)
    train_price_action_expert(final_df, "pa", MODEL_CONFIGS["price_action"], config_trainer.TICKER_NAME_FOR_MODELS,
                              BASE_TIMEFRAME)

    print("\n[KROK 5/5] Uruchamianie procedury dla eksperta na wybranych cechach...")
    train_and_evaluate_expert(final_df, "overall_best", MODEL_CONFIGS["overall_best"],
                              config_trainer.TICKER_NAME_FOR_MODELS, BASE_TIMEFRAME, selected_features=best_features)

    print("\n--- WSZYSCY EKSPERCI ZOSTALI ZWALIDOWANI, WYTRENOWANI I ZAPISANI. ---")


if __name__ == "__main__":
    asyncio.run(main())