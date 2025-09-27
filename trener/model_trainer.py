import asyncio
import os
import sys
import json
import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
from sklearn.feature_selection import RFE
import hashlib
import inspect
from numba import jit

import config
from data_preparer import prepare_feature_set_for_timeframe
from async_data_fetcher import fetch_data_for_trainer_async

def get_cache_hash() -> str:
    """
    Tworzy unikalny hash na podstawie parametrów w config.FeatureConfig
    oraz zawartości pliku data_preparer.py.
    """
    # 1. Pobierz parametry z FeatureConfig
    config_params = inspect.getmembers(config.FeatureConfig, lambda a: not(inspect.isroutine(a)))
    config_dict = {m[0]: m[1] for m in config_params if not m[0].startswith('__')}

    # 2. Odczytaj zawartość pliku data_preparer.py
    try:
        with open('data_preparer.py', 'r') as f:
            data_preparer_code = f.read()
    except FileNotFoundError:
        data_preparer_code = ""

    # 3. Połącz wszystko w jeden string i oblicz hash
    combined_string = json.dumps(config_dict, sort_keys=True) + data_preparer_code

    # Używamy sha256 dla bezpieczeństwa i skracamy do 12 znaków dla czytelności
    return hashlib.sha256(combined_string.encode()).hexdigest()[:12]

@jit(nopython=True, cache=True)
def _calculate_target_loop_numba(close, high, low, atr, horizon, tp_multiplier, sl_multiplier):
    n = len(close)
    outcomes = np.full(n, np.nan)
    for i in range(n - horizon):
        entry_price = close[i]
        current_atr = atr[i]

        if np.isnan(current_atr) or current_atr == 0:
            continue

        upper_barrier = entry_price + (current_atr * tp_multiplier)
        lower_barrier = entry_price - (current_atr * sl_multiplier)

        outcome = 0  # Domyślnie ruch boczny

        for j in range(1, horizon + 1):
            hit_tp = high[i + j] >= upper_barrier
            hit_sl = low[i + j] <= lower_barrier

            if hit_tp and hit_sl:
                # Jeśli w tej samej świecy dotknięto obu, wybieramy TP
                outcome = 1
                break
            elif hit_tp:
                outcome = 1
                break
            elif hit_sl:
                outcome = -1
                break

        outcomes[i] = outcome
    return outcomes

def calculate_multiclass_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    print(f"Obliczanie celu z użyciem Numba (Typ: {config.TARGET_TYPE}, Horyzont: {horizon} barów)...")

    atr_col_name = f'ATRr_{config.FeatureConfig.ATR_LENGTH}_{config.BASE_TIMEFRAME}'
    if atr_col_name not in df.columns:
        raise ValueError(f"Brak kolumny ATR '{atr_col_name}' w DataFrame.")

    # Przekazujemy dane jako tablice NumPy do skompilowanej funkcji
    outcomes_np = _calculate_target_loop_numba(
        df['close'].values,
        df['high'].values,
        df['low'].values,
        df[atr_col_name].values,
        horizon,
        config.ATR_TP_MULTIPLIER,
        config.ATR_SL_MULTIPLIER
    )

    return pd.Series(outcomes_np, index=df.index)

def train_unified_model(df: pd.DataFrame, model_for_trial: LGBMClassifier, full_run: bool = False):
    best_features = [col for col in df.columns if
                     col not in ['open', 'high', 'low', 'close', 'volume', 'turnover', 'target']]

    holdout_split_idx = int(len(df) * (1 - config.HOLDOUT_SIZE))
    train_val_df = df.iloc[:holdout_split_idx]
    holdout_df = df.iloc[holdout_split_idx:]

    print(f"Stosowanie embarga: usuwanie ostatnich {config.HORIZON_BARS} rekordów ze zbioru treningowego...")
    train_val_df = train_val_df.iloc[:-config.HORIZON_BARS]
    print(f"Rozmiar zbioru treningowego po embargu: {len(train_val_df)}")

    x_train_val = train_val_df[best_features]
    y_train_val = train_val_df['target']

    if not full_run:
        print(f"[KROK Wewnątrz triala] Uruchamianie walidacji krzyżowej...")
        tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS, gap=config.HORIZON_BARS)
        cv_scores = []
        for fold, (train_index, test_index) in enumerate(tscv.split(x_train_val)):
            x_train, x_test = x_train_val.iloc[train_index], x_train_val.iloc[test_index]
            y_train, y_test = y_train_val.iloc[train_index], y_train_val.iloc[test_index]

            scaler = clone(config.SCALER).set_output(transform="pandas")

            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            model_for_cv = clone(model_for_trial)
            model_for_cv.fit(x_train_scaled, y_train)
            y_pred = model_for_cv.predict(x_test_scaled)
            score = f1_score(y_test, y_pred, average='weighted')
            cv_scores.append(score)
        print(f"-> Średni F1-score z walidacji krzyżowej: {np.mean(cv_scores):.4f}")

    final_scaler = clone(config.SCALER).set_output(transform="pandas")

    x_train_val_scaled = final_scaler.fit_transform(x_train_val)
    final_model = clone(model_for_trial)
    final_model.fit(x_train_val_scaled, y_train_val)

    if full_run:
        joblib.dump(final_model, "final_model.joblib")
        print("\n> Wytrenowany model zapisany do 'final_model.joblib'")
        joblib.dump(final_scaler, "final_scaler.joblib")
        print("> Dopasowany skaler zapisany do 'final_scaler.joblib'")
        with open('best_features.json', 'w') as f:
            json.dump(best_features, f)
        print("> Lista najlepszych cech zapisana do 'best_features.json'")

    x_holdout = holdout_df[best_features]
    y_holdout = holdout_df['target']
    x_holdout_scaled = final_scaler.transform(x_holdout)
    y_holdout_pred = final_model.predict(x_holdout_scaled)
    y_holdout_proba = final_model.predict_proba(x_holdout_scaled)
    final_f1_score = f1_score(y_holdout, y_holdout_pred, average='weighted')

    return y_holdout_pred, y_holdout_proba, final_f1_score, holdout_df, best_features

def objective(trial: optuna.Trial, df_with_selected_features: pd.DataFrame) -> float:
    model_params = {
        'objective': 'binary',
        'metric': 'logloss',
        'random_state': config.RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,

        # --- Istniejące parametry ---
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'num_leaves': trial.suggest_int('num_leaves', 15, 60),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),

        # Parametry próbkowania (subsampling)
        'subsample': trial.suggest_float('subsample', 0.7, 1.0, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0, step=0.05),

        # Parametr złożoności liścia
        'min_child_samples': trial.suggest_int('min_child_samples', 15, 40),
    }
    model_for_trial = LGBMClassifier(**model_params)
    print(f"\n--- Rozpoczynanie Trial #{trial.number} ---")
    _, _, f1_result, _, _ = train_unified_model(df_with_selected_features.copy(), model_for_trial)
    print(f"--- Trial #{trial.number} Zakończony | F1-score na Holdout: {f1_result:.4f} ---")
    return f1_result

async def main() -> None:
    raw_cache_filepath = f"{config.RAW_DATA_CACHE_DIR}/{config.TICKER}_{config.TRAIN_START_DATE}_{config.TRAIN_END_DATE}.csv"
    if os.path.exists(raw_cache_filepath):
        print(f"Znaleziono surowe dane w cache: {os.path.basename(raw_cache_filepath)}")
        df_raw = pd.read_csv(raw_cache_filepath, index_col='timestamp', parse_dates=True)
    else:
        print("Pobieranie danych...")
        os.makedirs(config.RAW_DATA_CACHE_DIR, exist_ok=True)
        df_raw = await fetch_data_for_trainer_async(ticker=config.TICKER, start_date=config.TRAIN_START_DATE,
                                                    end_date=config.TRAIN_END_DATE, timeframe=config.BASE_TIMEFRAME,
                                                    cache_dir=config.RAW_DATA_CACHE_DIR)
        if df_raw is None or df_raw.empty: return

    cache_hash = get_cache_hash()
    features_cache_filepath = f"{config.FEATURES_CACHE_DIR}/features_{config.TICKER}_{cache_hash}.parquet"

    if os.path.exists(features_cache_filepath):
        print(f"Znaleziono gotowe cechy w cache (hash: {cache_hash}). Wczytywanie...")
        df_features = pd.read_parquet(features_cache_filepath)
    else:
        print(f"Brak cache dla bieżącej konfiguracji (hash: {cache_hash}). Uruchamianie przygotowania cech...")
        os.makedirs(config.FEATURES_CACHE_DIR, exist_ok=True)
        df_features = prepare_feature_set_for_timeframe(df_raw, base_tf=config.BASE_TIMEFRAME)
        df_features.to_parquet(features_cache_filepath)
        print(f"Przetworzone cechy zostały zapisane do cache.")

    target_params_str = f"H{config.HORIZON_BARS}_TP{config.ATR_TP_MULTIPLIER}_SL{config.ATR_SL_MULTIPLIER}"
    target_cache_filename = f"targets_{config.TICKER}_{target_params_str}_{cache_hash}.parquet"
    target_cache_filepath = os.path.join(config.FEATURES_CACHE_DIR, target_cache_filename)

    if os.path.exists(target_cache_filepath):
        print(f"Znaleziono gotowe etykiety w cache: {target_cache_filename}")
        targets = pd.read_parquet(target_cache_filepath).squeeze()
    else:
        print("Obliczanie etykiet celu...")
        targets = calculate_multiclass_target(df_features, config.HORIZON_BARS)
        targets.to_frame(name='target').to_parquet(target_cache_filepath)
        print(f"Obliczone etykiety zostały zapisane do cache.")

    df_features['target'] = targets.map({-1: 0, 0: 1, 1: 2})

    print("\n>>> Konwersja problemu na binarny (WZROST vs SPADEK)...")
    df_features = df_features[df_features['target'] != 1].copy()
    df_features['target'] = df_features['target'].map({0: 0, 2: 1})
    df_features.dropna(inplace=True)
    if len(df_features) < 1000: return

    # --- Uruchamianie jednorazowej, BARDZO DOKŁADNEJ selekcji cech (RFE) ---
    from sklearn.feature_selection import RFE
    print("\n--- Uruchamianie jednorazowej selekcji cech (RFE) ---")

    all_features = [col for col in df_features.columns if
                    col not in ['open', 'high', 'low', 'close', 'volume', 'turnover', 'target']]

    holdout_split_idx_fs = int(len(df_features) * (1 - config.HOLDOUT_SIZE))
    train_val_df_fs = df_features.iloc[:holdout_split_idx_fs]

    # Inicjalizujemy model, który będzie używany wewnątrz RFE
    # Używamy prostszych parametrów, bo będzie trenowany wielokrotnie
    selector_model_for_rfe = LGBMClassifier(n_estimators=100, random_state=config.RANDOM_STATE, objective='binary')

    print(f"Uruchamianie RFE w celu wybrania {config.TOP_N_FEATURES} cech... (to będzie długo trwało!)")
    # "step=0.1" oznacza, że w każdym kroku będziemy usuwać 10% najgorszych cech
    selector = RFE(estimator=selector_model_for_rfe, n_features_to_select=config.TOP_N_FEATURES, step=0.1, verbose=1)
    selector.fit(train_val_df_fs[all_features], train_val_df_fs['target'])

    # Pobieramy listę wybranych cech
    best_features_fs = [feature for feature, selected in zip(all_features, selector.support_) if selected]

    # Możemy też zobaczyć ranking wszystkich cech (1 - najlepsza)
    feature_ranking = pd.DataFrame({
        'feature': all_features,
        'ranking': selector.ranking_
    }).sort_values('ranking')

    print(f"Wybrano {len(best_features_fs)} najlepszych cech do optymalizacji.")
    feature_ranking.to_csv("feature_ranking_rfe.csv", index=False)
    print("> Pełen ranking cech (RFE) zapisany do 'feature_ranking_rfe.csv'")

    # Dalsza część kodu pozostaje bez zmian
    final_columns_to_keep = best_features_fs + ['target']
    df_features_selected = df_features[final_columns_to_keep]

    final_columns_to_keep = best_features_fs + ['target']
    df_features_selected = df_features[final_columns_to_keep]

    study_name = f"optimization_binary_{config.TICKER}_{config.BASE_TIMEFRAME}"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize', load_if_exists=True)

    # study.optimize(lambda trial: objective(trial, df_features_selected), n_trials=config.OPTUNA_TRIALS)

    try:
        best_params = study.best_params
        print("\nPomyślnie znaleziono najlepsze parametry w Optunie.")
    except (ValueError, AttributeError):
        print("\nBŁĄD: Nie znaleziono żadnych ukończonych traiali w studium Optuny.")
        sys.exit(1)

    print("\n--- Trenowanie finalnego modelu z najlepszymi parametrami... ---")
    best_model = LGBMClassifier(objective='binary', random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1,
                                **best_params)

    preds, probas, f1, df_holdout, top_features_final = train_unified_model(df_features_selected.copy(), best_model,
                                                                            full_run=True)

    # 1. Stwórz bazową ramkę danych z tego, co mamy: target, prediction, probas
    results_df = pd.DataFrame({
        'target': df_holdout['target'],
        'prediction': preds,
        'proba_DOWN(0)': probas[:, 0],
        'proba_UP(1)': probas[:, 1]
    }, index=df_holdout.index)

    # 2. Dołącz brakujące kolumny OHLC z pełnego zbioru `df_features`
    ohlc_cols = ['open', 'high', 'low', 'close']
    results_df = results_df.join(df_features[ohlc_cols])

    # 3. Ustaw pożądaną kolejność kolumn
    final_cols_order = ohlc_cols + ['target', 'prediction', 'proba_DOWN(0)', 'proba_UP(1)']
    results_df = results_df[final_cols_order]
    # =================================================================

    print(f"\nWynik F1-score na zbiorze testowym (holdout): {f1:.4f}")
    print("\n" + "=" * 50)
    print("Pełny Raport Klasyfikacji na Zbiorze Holdout:")
    print(classification_report(y_true=df_holdout['target'], y_pred=preds, target_names=['SPADEK (0)', 'WZROST (1)']))
    print("=" * 50)

    print("\nTop 10 cech użytych w finalnym modelu:")
    print(top_features_final[:10])

    print("\n--- Przykładowe predykcje finalnego modelu na danych testowych: ---")
    print(results_df.head(15).to_string())
    results_df.to_csv("final_predictions.csv")

if __name__ == "__main__":
    asyncio.run(main())