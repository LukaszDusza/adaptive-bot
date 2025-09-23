# model_trainer.py
import pandas as pd
import numpy as np
import joblib
import json
import asyncio
from typing import Dict, Any, List
from sklearn.metrics import f1_score

from optuna import trial
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from tqdm import tqdm

import optuna
from lightgbm import LGBMClassifier

import config
from async_data_fetcher import fetch_data_for_trainer_async
from data_preparer import prepare_feature_set_for_timeframe


def calculate_multiclass_target(df: pd.DataFrame, target_pct: float, horizon: int) -> pd.Series:
    """Oblicza target dla klasyfikacji wieloklasowej (UP, DOWN, SIDEWAYS)."""
    print(f"Obliczanie celu (target {target_pct * 100:.1f}%, horyzont {horizon} barów)...")
    outcomes = pd.Series(0, index=df.index, dtype=int)

    for i in tqdm(range(len(df) - horizon), desc="Obliczanie celu", leave=False, ncols=100):
        entry_price = df['close'].iloc[i]
        upper_barrier = entry_price * (1 + target_pct)
        lower_barrier = entry_price * (1 - target_pct)
        future_window = df.iloc[i + 1: i + 1 + horizon]
        hit_tp_time = future_window[future_window['high'] >= upper_barrier].index.min()
        hit_sl_time = future_window[future_window['low'] <= lower_barrier].index.min()
        if pd.notna(hit_tp_time) and pd.notna(hit_sl_time):
            if hit_tp_time < hit_sl_time:
                outcomes.iloc[i] = 1
            else:
                outcomes.iloc[i] = -1
        elif pd.notna(hit_tp_time):
            outcomes.iloc[i] = 1
        elif pd.notna(hit_sl_time):
            outcomes.iloc[i] = -1
    return outcomes


# === ZMIANA 3: Funkcja przyjmuje model jako argument i ZWRACA wynik ===
def train_unified_model(df: pd.DataFrame, model_for_trial: LGBMClassifier) -> float:
    """Przeprowadza trening i walidację, a na końcu zwraca dokładność na zbiorze holdout."""

    print(f"\n[KROK 3/4] Selekcja {config.TOP_N_FEATURES} najważniejszych cech...")
    df = df.loc[:, ~df.columns.duplicated()]
    all_features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
    holdout_split_idx = int(len(df) * (1 - config.HOLDOUT_SIZE))
    train_val_df = df.iloc[:holdout_split_idx]
    holdout_df = df.iloc[holdout_split_idx:]

    X_train_val_fs = train_val_df[all_features]
    y_train_val_fs = train_val_df['target']

    selector_model = clone(model_for_trial)
    selector_model.fit(X_train_val_fs, y_train_val_fs)

    feature_importances = pd.DataFrame({
        'feature': all_features,
        'importance': selector_model.feature_importances_
    }).sort_values('importance', ascending=False)

    best_features = feature_importances.head(config.TOP_N_FEATURES)['feature'].tolist()

    print(f"[KROK 4/4] Walidacja krzyżowa i finalny trening...")
    X_train_val = train_val_df[best_features]
    y_train_val = train_val_df['target']
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS, gap=5)
    scores = []

    for fold, (train_index, test_index) in enumerate(
            tqdm(tscv.split(X_train_val), total=config.CV_SPLITS, desc="Walidacja krzyżowa", leave=False, ncols=100)):
        X_train, X_test = X_train_val.iloc[train_index], X_train_val.iloc[test_index]
        y_train, y_test = y_train_val.iloc[train_index], y_train_val.iloc[test_index]
        scaler = clone(config.SCALER)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_for_cv = clone(model_for_trial)
        model_for_cv.fit(X_train_scaled, y_train)
        y_pred = model_for_cv.predict(X_test_scaled)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)

    print(f"-> Średnia dokładność CV: {np.mean(scores):.4f}")

    X_holdout = holdout_df[best_features]
    y_holdout = holdout_df['target']
    final_scaler = clone(config.SCALER)
    X_train_val_scaled = final_scaler.fit_transform(X_train_val)

    final_model = clone(model_for_trial)
    final_model.fit(X_train_val_scaled, y_train_val)

    X_holdout_scaled = final_scaler.transform(X_holdout)
    y_holdout_pred = final_model.predict(X_holdout_scaled)
    holdout_accuracy = accuracy_score(y_holdout, y_holdout_pred)

    f1 = f1_score(y_holdout, y_holdout_pred, average=None, labels=[0, 2])
    mean_f1_score = np.mean(f1)

    return mean_f1_score


def objective(trial: optuna.Trial, df_features: pd.DataFrame) -> float:
    """
    Pojedyncza iteracja (trial) procesu optymalizacji.
    Optuna wywołuje tę funkcję wielokrotnie z różnymi parametrami.
    """
    # -- Krok 1: Optuna sugeruje hiperparametry do przetestowania w tej iteracji --
    # target_pct = trial.suggest_float('PRICE_TARGET_PCT', 0.005, 0.015, step=0.005)
    # horizon = trial.suggest_int('HORIZON_BARS', 6, 24, step=6)
    target_pct = 0.01
    horizon = 32

    # Krok 2: Optuna sugeruje już tylko hiperparametry modelu --
    model_params = {
        'objective': 'multiclass',
        'n_estimators': trial.suggest_int('n_estimators', 200, 1200, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'random_state': config.RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    }
    model_for_trial = LGBMClassifier(**model_params)

    # Krok 3: Uruchomienie procesu z użyciem zadanych parametrów --
    print(f"\n--- Rozpoczynanie Trial #{trial.number} ---")
    # Zmieniony print, aby odzwierciedlał stałe parametry
    print(f"Parametry: Target={target_pct * 100:.1f}% (stały), Horyzont={horizon} (stały)")
    print(f"Parametry modelu: n_est={model_params['n_estimators']}, lr={model_params['learning_rate']:.4f}, max_depth={model_params['max_depth']}")


    df = df_features.copy()
    targets = calculate_multiclass_target(df, target_pct, horizon)
    df['target'] = targets.map({-1: 0, 0: 1, 1: 2})
    df.dropna(inplace=True)

    if len(df) < 500:
        print("Zbyt mało danych po obróbce, pomijanie triala.")
        return 0.0

    # Zmieniliśmy metrykę na F1-score
    f1_result = train_unified_model(df, model_for_trial)
    print(f"--- Trial #{trial.number} Zakończony | Średni F1-score (UP/DOWN): {f1_result:.4f} ---")
    return f1_result

async def main() -> None:
    # Krok 1: Pobranie i przygotowanie cech (bez zmian, wykonuje się raz na start workera)
    print("--- Jednorazowe pobieranie i przygotowywanie cech ---")
    df_raw = await fetch_data_for_trainer_async(
        ticker=config.TICKER, start_date=config.TRAIN_START_DATE, end_date=config.TRAIN_END_DATE
    )
    if df_raw.empty:
        print("Nie pobrano danych, zakończono.")
        return

    df_features = prepare_feature_set_for_timeframe(df_raw, base_tf=config.BASE_TIMEFRAME)

    # === KLUCZOWA ZMIANA DLA WSPÓŁBIEŻNOŚCI ===
    # Definiujemy nazwę dla naszej optymalizacji i wskazujemy plik bazy danych
    study_name = f"optimization_{config.TICKER}_{config.BASE_TIMEFRAME}"
    storage_name = f"sqlite:///{study_name}.db"

    # Tworzymy 'study', która będzie zapisywana na dysku i współdzielona
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True  # NAJWAŻNIEJSZE: jeśli study już istnieje, dołącz do niej
    )
    # ============================================

    # Wyświetlamy informację, jeśli dołączamy do istniejącego study
    if len(study.trials) > 0:
        print(f"Dołączono do istniejącego study. Aktualna liczba prób: {len(study.trials)}")


    study.optimize(
        lambda trial: objective(trial, df_features),
        n_trials=50
    )

    # Krok 3: Wyświetlenie wyników (wykona się w każdym workerze po zakończeniu jego pętli)
    print("\n--- Optymalizacja zakończona! ---")
    print(f"Najlepsza dokładność (holdout) w całym study: {study.best_value:.4f}")
    print("Najlepsze znalezione parametry w całym study:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\nGenerowanie wykresu historii optymalizacji...")
    loaded_study = optuna.load_study(study_name=study_name, storage=storage_name)
    fig = optuna.visualization.plot_optimization_history(loaded_study)
    fig.show()

if __name__ == "__main__":
    asyncio.run(main())

