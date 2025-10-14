import pandas as pd
import numpy as np
import optuna
import joblib
import os
import json
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, fbeta_score, precision_recall_curve
from numba import njit
from imblearn.over_sampling import SMOTE

# Wyłączenie szczegółowych logów z Optuny dla większej czytelności
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Sprawdzenie dostępności GPU i ustawienie urządzenia
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Urządzenie dla TabNet: {DEVICE.upper()} ---")


# --- Ta sekcja jest skopiowana z Twojego oryginalnego pliku, działa świetnie i nie wymaga zmian ---
@njit
def _compute_labels_fast(prices: np.ndarray, event_indices: np.ndarray, profit_take_pct: float, stop_loss_pct: float,
                         time_limit: int):
    n_events = len(event_indices)
    outcomes = np.zeros(n_events, dtype=np.int64)
    for i in range(n_events):
        event_idx = event_indices[i]
        start_price = prices[event_idx]
        upper_barrier = start_price * (1 + profit_take_pct)
        lower_barrier = start_price * (1 - stop_loss_pct)
        for j in range(1, time_limit + 1):
            current_idx = event_idx + j
            if current_idx >= len(prices): break
            price = prices[current_idx]
            if price >= upper_barrier:
                outcomes[i] = 1;
                break
            if price <= lower_barrier:
                outcomes[i] = 2;
                break
    return outcomes


def get_triple_barrier_labels(prices: pd.Series, t_events: pd.Index, profit_take_pct: float, stop_loss_pct: float,
                              time_limit: int, verbose=True):
    if verbose:
        print("Rozpoczynanie etykietowania danych (Triple Barrier)...")
    prices_arr = prices.to_numpy()
    event_indices = prices.index.get_indexer(t_events)
    outcomes = _compute_labels_fast(prices_arr, event_indices, profit_take_pct, stop_loss_pct, time_limit)
    labels = pd.Series(outcomes, index=t_events)
    if verbose:
        print(f"Etykietowanie zakończone. Rozkład etykiet:\n{labels.value_counts(normalize=True)}")
    return labels


def _get_strategy_id(ticker, timeframe, helper_timeframes, side: str):
    helpers_str = '_plus_' + '_'.join(helper_timeframes) if helper_timeframes else ""
    return f"TabNet_{ticker}_{timeframe.replace(' ', '')}{helpers_str}_{side}"


# --- Koniec sekcji bez zmian ---


def _run_feature_selection(X: pd.DataFrame, y: pd.Series, strategy_id: str, importance_threshold: float = 0.90):
    """
    Selekcja cech oparta na ważności cech z modelu TabNet.
    Trenuje prosty model TabNet na foldach, aby uzyskać stabilne wagi cech,
    a następnie wybiera te, które odpowiadają za określony próg skumulowanej ważności.
    """
    print("\n--- ETAP 1.5: Rozpoczynanie selekcji cech (TabNet Feature Importance) ---")
    print(f"Cechy początkowe: {len(X.columns)}")

    cv_splitter = TimeSeriesSplit(n_splits=3)
    feature_importances = np.zeros(len(X.columns))

    print("Trenowanie modeli TabNet z cross-validation dla agregacji ważności cech...")
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X), 1):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Prosty model TabNet do oceny cech
        model = TabNetClassifier(
            verbose=0,
            seed=42,
            device_name=DEVICE
        )
        model.fit(
            X_train=X_train_scaled, y_train=y_train_fold.values,
            eval_set=[(X_val_scaled, y_val_fold.values)],
            patience=5, max_epochs=50,
            batch_size=1024
        )
        feature_importances += model.feature_importances_
        print(f"  Fold {fold_idx}/3 ukończony")

    feature_importances /= cv_splitter.get_n_splits()

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    importance_df['cumulative_importance'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()

    selected_features = importance_df[importance_df['cumulative_importance'] <= importance_threshold][
        'feature'].tolist()

    min_features = max(10, int(len(X.columns) * 0.2))
    if len(selected_features) < min_features:
        selected_features = importance_df.head(min_features)['feature'].tolist()

    print(f"\nSelekcja zakończona. Wybrano {len(selected_features)} z {len(X.columns)} cech.")
    print(f"Próg skumulowanej ważności: {importance_threshold * 100}%")
    print("Top 10 najważniejszych cech:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f} (skumulowane: {row['cumulative_importance']:.2%})")

    removed_features = [f for f in X.columns if f not in selected_features]
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    weak_features_path = os.path.join(models_dir, f"{strategy_id}_weak_features.json")

    with open(weak_features_path, 'w') as f:
        json.dump(removed_features, f, indent=2)

    print(f"\nZapisano {len(removed_features)} słabych cech do: {weak_features_path}")
    return selected_features


def _run_model_optimization(X: pd.DataFrame, y: pd.Series, n_model_trials: int, strategy_id: str):
    """Optymalizacja hiperparametrów dla modelu TabNet."""

    def objective_model(trial):
        # Definiowanie przestrzeni hiperparametrów dla TabNet
        mask_type = trial.suggest_categorical("mask_type", ["sparsemax", "entmax"])
        n_da = trial.suggest_int("n_da", 16, 64, step=8)
        n_steps = trial.suggest_int("n_steps", 3, 10, step=1)
        gamma = trial.suggest_float("gamma", 1.0, 2.0, step=0.1)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)

        tabnet_params = {
            "n_d": n_da,
            "n_a": n_da,
            "n_steps": n_steps,
            "gamma": gamma,
            "lambda_sparse": lambda_sparse,
            "mask_type": mask_type,
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=trial.suggest_float("lr", 1e-3, 3e-2, log=True)),
            "scheduler_params": {"mode": "min", "patience": 5, "factor": 0.5},
            "scheduler_fn": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "seed": 42,
            "verbose": 0,
            "device_name": DEVICE
        }

        tscv = TimeSeriesSplit(n_splits=4)
        scores = []
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            if y_train.nunique() < 2: continue

            smote = SMOTE(random_state=42,
                          k_neighbors=max(1, min(len(y_train[y_train == 1]), len(y_train[y_train == 0])) - 1))
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_resampled)
            X_val_scaled = scaler.transform(X_val)

            model = TabNetClassifier(**tabnet_params)
            model.fit(
                X_train=X_train_scaled, y_train=y_train_resampled.values,
                eval_set=[(X_val_scaled, y_val.values)],
                eval_metric=['logloss'],
                max_epochs=200,
                patience=15,  # Wczesne zatrzymywanie
                batch_size=1024,
                drop_last=False
            )
            preds = model.predict(X_val_scaled)
            scores.append(fbeta_score(y_val, preds, beta=2.0, average='macro'))

        return np.mean(scores) if scores else 0.0

    storage_name = f"sqlite:///optuna/{strategy_id}_model_study.db"
    os.makedirs("optuna", exist_ok=True)
    study = optuna.create_study(study_name=f"{strategy_id}_model_optimization", storage=storage_name,
                                direction='maximize', load_if_exists=True)
    study.optimize(objective_model, n_trials=n_model_trials, show_progress_bar=True)

    # Zwracamy tylko te parametry, które są potrzebne do inicjalizacji TabNetClassifier
    best_params = {
        "mask_type": study.best_params["mask_type"],
        "n_d": study.best_params["n_da"],
        "n_a": study.best_params["n_da"],
        "n_steps": study.best_params["n_steps"],
        "gamma": study.best_params["gamma"],
        "lambda_sparse": study.best_params["lambda_sparse"],
        "optimizer_params": {"lr": study.best_params["lr"]}
    }
    return best_params


def run_training_pipeline(df_features: pd.DataFrame, n_label_trials: int, n_model_trials: int, ticker: str,
                          timeframe: str, helper_timeframes: list = None, side: str = 'long'):
    strategy_id = _get_strategy_id(ticker, timeframe, helper_timeframes, side)
    df_model_base = df_features.drop(columns=['open', 'high', 'low', 'close', 'volume', 'turnover'], errors='ignore')
    holdout_size = int(len(df_model_base) * 0.2)
    train_val_df = df_model_base.iloc[:-holdout_size]
    holdout_df = df_model_base.iloc[-holdout_size:]

    # --- ETAP 1: Optymalizacja Etykiet (bez zmian) ---
    def objective_labels(trial):
        pt = trial.suggest_float('pt_pct', 0.01, 0.09, log=True)
        sl = trial.suggest_float('sl_pct', 0.01, 0.09, log=True)
        time_limit = trial.suggest_int('time_limit', 4, 48)
        labels = get_triple_barrier_labels(df_features['close'], df_features.index, pt, sl, time_limit, verbose=False)
        y = labels.reindex(train_val_df.index)
        if y.nunique() < 3: return 0.0
        ideal_dist = {0: 0.50, 1: 0.25, 2: 0.25}
        actual_dist = y.value_counts(normalize=True)
        for i in range(3):
            if i not in actual_dist: actual_dist[i] = 0
        mse = ((actual_dist.sort_index() - pd.Series(ideal_dist).sort_index()) ** 2).mean()
        return np.exp(-10 * mse)

    print("\n--- ETAP 1: Rozpoczynanie optymalizacji parametrów etykiet ---")
    storage_name_labels = f"sqlite:///optuna/{strategy_id}_labels_study.db"
    os.makedirs("optuna", exist_ok=True)
    study_labels = optuna.create_study(study_name=f"{strategy_id}_labels_optimization", storage=storage_name_labels,
                                       direction='maximize', load_if_exists=True)
    study_labels.optimize(objective_labels, n_trials=n_label_trials, show_progress_bar=True)
    best_label_params = study_labels.best_params
    print(f"\nNajlepsze parametry etykiet: {best_label_params}")

    final_labels = get_triple_barrier_labels(df_features['close'], df_features.index, best_label_params['pt_pct'],
                                             best_label_params['sl_pct'], best_label_params['time_limit'])

    print(f"\n--- Przygotowywanie binarnego zestawu danych dla modelu '{side.upper()}' ---")
    if side == 'long':
        y_full_multi = final_labels.reindex(train_val_df.index)
        is_long_or_hold = y_full_multi.isin([0, 1])
        X_full, y_full = train_val_df[is_long_or_hold], y_full_multi[is_long_or_hold]
        target_names = ['HOLD (0)', 'BUY (1)']
    elif side == 'short':
        y_full_multi = final_labels.reindex(train_val_df.index)
        is_short_or_hold = y_full_multi.isin([0, 2])
        X_full, y_full = train_val_df[is_short_or_hold], y_full_multi[is_short_or_hold]
        y_full = y_full.replace(2, 1)
        target_names = ['HOLD (0)', 'SELL (1)']
    else:
        raise ValueError("Parametr 'side' musi być 'long' lub 'short'.")
    print(f"Zachowano {len(y_full)} etykiet. Nowy rozkład:\n{y_full.value_counts(normalize=True)}")

    # --- ETAP 1.5: Selekcja cech dla TabNet ---
    selected_features = _run_feature_selection(X_full, y_full, strategy_id)
    X_full = X_full[selected_features]

    # --- ETAP 2: Optymalizacja modelu TabNet ---
    print("\n--- ETAP 2: Rozpoczynanie optymalizacji hiperparametrów modelu TabNet ---")
    best_model_params = _run_model_optimization(X_full, y_full, n_model_trials, strategy_id)
    print(f"Najlepsze parametry modelu: {best_model_params}")

    # --- ETAP 3: Trening finalnego modelu ---
    print("\n--- Trenowanie finalnego modelu binarnego TabNet ... ---")
    smote_final = SMOTE(random_state=42,
                        k_neighbors=max(1, min(len(y_full[y_full == 1]), len(y_full[y_full == 0])) - 1))
    X_resampled, y_resampled = smote_final.fit_resample(X_full, y_full)

    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X_resampled)

    final_model = TabNetClassifier(
        **best_model_params,
        optimizer_fn=torch.optim.Adam,
        scheduler_params={"mode": "min", "patience": 10, "factor": 0.5, "verbose": False},
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        seed=42,
        verbose=1,
        device_name=DEVICE
    )
    final_model.fit(
        X_train=X_scaled, y_train=y_resampled.values,
        max_epochs=300, patience=25,
        batch_size=1024
    )

    # --- Zapisywanie artefaktów ---
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{strategy_id}_model")  # TabNet zapisuje jako .zip
    scaler_path = os.path.join(models_dir, f"{strategy_id}_scaler.joblib")
    features_path = os.path.join(models_dir, f"{strategy_id}_features.joblib")

    final_model.save_model(model_path)  # Specjalna metoda zapisu dla TabNet
    joblib.dump(final_scaler, scaler_path)
    joblib.dump(selected_features, features_path)
    print(f"Model, skaler i lista cech zostały zapisane w katalogu '{models_dir}'.")

    # --- Ewaluacja na zbiorze Holdout ---
    print("\n--- Ocena modelu na danych Holdout ---")
    y_holdout_multi = final_labels.reindex(holdout_df.index).dropna()
    X_holdout_multi = holdout_df.loc[y_holdout_multi.index]

    if side == 'long':
        X_holdout = X_holdout_multi[y_holdout_multi.isin([0, 1])]
        y_holdout = y_holdout_multi[y_holdout_multi.isin([0, 1])]
    else:  # short
        X_holdout = X_holdout_multi[y_holdout_multi.isin([0, 2])]
        y_holdout = y_holdout_multi[y_holdout_multi.isin([0, 2])].replace(2, 1)

    X_holdout = X_holdout[selected_features]
    X_holdout_scaled = final_scaler.transform(X_holdout)

    holdout_probas = final_model.predict_proba(X_holdout_scaled)

    # --- Strojenie progu decyzyjnego ---
    print("\n--- ANALIZA PROGÓW DECYZYJNYCH (Threshold Tuning) ---")
    precisions, recalls, thresholds = precision_recall_curve(y_holdout, holdout_probas[:, 1])

    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)

    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]

    holdout_preds_optimized = (holdout_probas[:, 1] >= optimal_threshold).astype(int)

    print(
        f"\nRaport klasyfikacji na zbiorze holdout dla modelu '{side.upper()}' (Próg zoptymalizowany: {optimal_threshold:.3f}):")
    if y_holdout.nunique() < 2:
        print("Nie można wygenerować raportu: zbiór testowy holdout zawiera tylko jedną klasę.")
    else:
        print(classification_report(y_holdout, holdout_preds_optimized, target_names=target_names))

    # Zapisywanie wyników
    results_df = pd.DataFrame(holdout_probas, columns=[f'proba_{c}' for c in target_names], index=X_holdout.index)
    results_df['y_true'] = y_holdout
    results_df['y_pred_optimized'] = holdout_preds_optimized
    results_df['optimal_threshold'] = optimal_threshold

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{strategy_id}_holdout_predictions.csv")
    results_df.to_csv(results_path)
    print(f"Szczegółowe wyniki ze zbioru holdout zapisano w: {results_path}")