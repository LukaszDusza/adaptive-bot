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

import config
from data_preparer import prepare_feature_set_for_timeframe
from async_data_fetcher import fetch_data_for_trainer_async


def calculate_multiclass_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    print(f"Obliczanie celu (Typ: {config.TARGET_TYPE}, Horyzont: {horizon} barów)...")
    outcomes = pd.Series(np.nan, index=df.index)

    if config.TARGET_TYPE == 'DYNAMIC_ATR':
        atr_col_name = f'ATRr_{config.FeatureConfig.ATR_LENGTH}_{config.BASE_TIMEFRAME}'
        if atr_col_name not in df.columns:
            raise ValueError(f"Brak kolumny ATR '{atr_col_name}' w DataFrame.")

    for i in tqdm(range(len(df) - horizon), desc="Obliczanie celu", leave=False, ncols=100):
        entry_price = df['close'].iloc[i]

        if config.TARGET_TYPE == 'DYNAMIC_ATR':
            current_atr = df[atr_col_name].iloc[i]
            if pd.isna(current_atr) or current_atr == 0:
                continue
            upper_barrier = entry_price + (current_atr * config.ATR_TP_MULTIPLIER)
            lower_barrier = entry_price - (current_atr * config.ATR_SL_MULTIPLIER)
        else:
            upper_barrier = entry_price * (1 + config.PRICE_TARGET_PCT)
            lower_barrier = entry_price * (1 - config.PRICE_TARGET_PCT)

        future_window = df.iloc[i + 1: i + 1 + horizon]
        hit_tp_time = future_window[future_window['high'] >= upper_barrier].index.min()
        hit_sl_time = future_window[future_window['low'] <= lower_barrier].index.min()

        outcome = 0
        if pd.notna(hit_tp_time) and pd.notna(hit_sl_time):
            outcome = 1 if hit_tp_time < hit_sl_time else -1
        elif pd.notna(hit_tp_time):
            outcome = 1
        elif pd.notna(hit_sl_time):
            outcome = -1
        outcomes.iloc[i] = outcome

    return outcomes


def train_unified_model(df: pd.DataFrame, model_for_trial: LGBMClassifier, full_run: bool = False):
    all_features = [col for col in df.columns if
                    col not in ['open', 'high', 'low', 'close', 'volume', 'turnover', 'target']]

    holdout_split_idx = int(len(df) * (1 - config.HOLDOUT_SIZE))
    train_val_df = df.iloc[:holdout_split_idx]
    holdout_df = df.iloc[holdout_split_idx:]

    print(f"Stosowanie embarga: usuwanie ostatnich {config.HORIZON_BARS} rekordów ze zbioru treningowego...")
    train_val_df = train_val_df.iloc[:-config.HORIZON_BARS]
    print(f"Rozmiar zbioru treningowego po embargu: {len(train_val_df)}")

    x_train_val_fs = train_val_df[all_features]
    y_train_val_fs = train_val_df['target']

    selector_model = clone(model_for_trial)
    selector_model.fit(x_train_val_fs, y_train_val_fs)

    feature_importances = pd.DataFrame({
        'feature': all_features, 'importance': selector_model.feature_importances_
    }).sort_values('importance', ascending=False)

    best_features = feature_importances.head(config.TOP_N_FEATURES)['feature'].tolist()

    x_train_val = train_val_df[best_features]
    y_train_val = train_val_df['target']

    if not full_run:
        print(f"-> Top 10 najważniejszych cech w tym trialu:")
        print(feature_importances.head(10).to_string(index=False))
        print(f"[KROK 4/4] Walidacja krzyżowa...")

        tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS, gap=config.HORIZON_BARS)
        cv_scores = []
        for fold, (train_index, test_index) in enumerate(tscv.split(x_train_val)):
            x_train, x_test = x_train_val.iloc[train_index], x_train_val.iloc[test_index]
            y_train, y_test = y_train_val.iloc[train_index], y_train_val.iloc[test_index]
            scaler = clone(config.SCALER)
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            model_for_cv = clone(model_for_trial)
            model_for_cv.fit(x_train_scaled, y_train)
            y_pred = model_for_cv.predict(x_test_scaled)
            score = f1_score(y_test, y_pred, average='weighted')
            cv_scores.append(score)
        print(f"-> Średni F1-score z walidacji krzyżowej: {np.mean(cv_scores):.4f}")

    final_scaler = clone(config.SCALER)
    x_train_val_scaled = final_scaler.fit_transform(x_train_val)
    final_model = clone(model_for_trial)
    final_model.fit(x_train_val_scaled, y_train_val)

    if full_run:
        feature_importances.to_csv("feature_importances.csv", index=False)
        print("\n> Ważność cech zapisana do 'feature_importances.csv'")
        joblib.dump(final_model, "final_model.joblib")
        print("> Wytrenowany model zapisany do 'final_model.joblib'")
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


def objective(trial: optuna.Trial, df_with_target: pd.DataFrame) -> float:
    model_params = {
        'objective': 'binary',
        'metric': 'logloss',
        'n_estimators': trial.suggest_int('n_estimators', 200, 1200, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'random_state': config.RANDOM_STATE,
        'n_jobs': -1, 'verbose': -1
    }
    model_for_trial = LGBMClassifier(**model_params)
    print(f"\n--- Rozpoczynanie Trial #{trial.number} ---")
    _, _, f1_result, _, _ = train_unified_model(df_with_target.copy(), model_for_trial)
    print(f"--- Trial #{trial.number} Zakończony | F1-score na Holdout: {f1_result:.4f} ---")
    return f1_result


async def main() -> None:
    # --- Krok 1: Inteligentne wczytywanie surowych danych (Cache-Then-Fetch) ---
    raw_cache_filepath = f"{config.RAW_DATA_CACHE_DIR}/{config.TICKER}_{config.TRAIN_START_DATE}_{config.TRAIN_END_DATE}.csv"

    if os.path.exists(raw_cache_filepath):
        print(f"Znaleziono surowe dane w cache. Wczytywanie z pliku: {os.path.basename(raw_cache_filepath)}")
        df_raw = pd.read_csv(raw_cache_filepath, index_col='timestamp', parse_dates=True)
    else:
        print("Nie znaleziono surowych danych w cache. Uruchamianie pobierania danych...")
        os.makedirs(config.RAW_DATA_CACHE_DIR, exist_ok=True)
        df_raw = await fetch_data_for_trainer_async(
            ticker=config.TICKER, start_date=config.TRAIN_START_DATE, end_date=config.TRAIN_END_DATE,
            timeframe=config.BASE_TIMEFRAME, cache_dir=config.RAW_DATA_CACHE_DIR
        )
        if df_raw is None or df_raw.empty:
            print("BŁĄD: Pobieranie danych nie powiodło się. Zakończenie skryptu.")
            return

    # === ZMIANA: Implementacja cache dla przetworzonych cech ===
    features_cache_filepath = f"{config.FEATURES_CACHE_DIR}/features_{config.TICKER}_{config.TRAIN_START_DATE}_{config.TRAIN_END_DATE}.parquet"

    if os.path.exists(features_cache_filepath):
        print(f"Znaleziono gotowe cechy w cache. Wczytywanie z pliku: {os.path.basename(features_cache_filepath)}")
        df_features = pd.read_parquet(features_cache_filepath)
    else:
        print("Nie znaleziono gotowych cech w cache. Uruchamianie pełnego przygotowania cech...")
        os.makedirs(config.FEATURES_CACHE_DIR, exist_ok=True)
        df_features = prepare_feature_set_for_timeframe(df_raw, base_tf=config.BASE_TIMEFRAME)

        # Zapis do cache
        df_features.to_parquet(features_cache_filepath)
        print(f"Przetworzone cechy zostały zapisane do cache: {features_cache_filepath}")
    # ==========================================================

    targets = calculate_multiclass_target(df_features, config.HORIZON_BARS)
    df_features['target'] = targets.map({-1: 0, 0: 1, 1: 2})

    print("\n>>> Konwersja problemu na binarny (WZROST vs SPADEK) i usunięcie klasy BOK...")
    df_features = df_features[df_features['target'] != 1].copy()
    df_features['target'] = df_features['target'].map({0: 0, 2: 1})

    df_features.dropna(inplace=True)
    if len(df_features) < 1000:
        print("Za mało danych po przetworzeniu. Zakończenie.")
        return

    # Optymalizacja (lub wczytanie wyników)
    study_name = f"optimization_binary_{config.TICKER}_{config.BASE_TIMEFRAME}"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize', load_if_exists=True)

    # Odkomentuj, aby uruchomić optymalizację
    study.optimize(lambda trial: objective(trial, df_features), n_trials=config.OPTUNA_TRIALS)

    try:
        best_params = study.best_params
        print("\nPomyślnie wczytano najlepsze parametry z Optuny.")
    except (ValueError, AttributeError):
        print("\nBŁĄD: Nie znaleziono żadnych ukończonych traiali w studium Optuny.")
        print(f"Upewnij się, że plik bazy danych '{storage_name}' istnieje i zawiera wyniki,")
        print("lub odkomentuj linię 'study.optimize(...)' w kodzie.")
        sys.exit(1)

    # Finalny trening
    print("\n--- Trenowanie finalnego modelu z najlepszymi parametrami... ---")
    best_model = LGBMClassifier(objective='binary', random_state=config.RANDOM_STATE, n_jobs=-1, verbose=--1,
                                **best_params)

    preds, probas, f1, df_holdout, top_features = train_unified_model(df_features.copy(), best_model, full_run=True)

    results_df = df_holdout[['open', 'high', 'low', 'close', 'target']].copy()
    results_df['prediction'] = preds
    results_df['proba_DOWN(0)'] = probas[:, 0]
    results_df['proba_UP(1)'] = probas[:, 1]

    print(f"\nWynik F1-score na zbiorze testowym (holdout): {f1:.4f}")
    print("\n" + "=" * 50)
    print("Pełny Raport Klasyfikacji na Zbiorze Holdout:")
    print(classification_report(y_true=df_holdout['target'], y_pred=preds, target_names=['SPADEK (0)', 'WZROST (1)']))
    print("=" * 50)

    print("\nTop 10 cech użytych w finalnym modelu:")
    print(top_features[:10])

    print("\n--- Przykładowe predykcje finalnego modelu na danych testowych: ---")
    print(results_df.head(15).to_string())

    results_df.to_csv("final_predictions.csv")


if __name__ == "__main__":
    asyncio.run(main())
