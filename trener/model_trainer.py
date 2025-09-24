import asyncio
import os

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

import config
from async_data_fetcher import fetch_data_for_trainer_async
from data_preparer import prepare_feature_set_for_timeframe


def calculate_multiclass_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    print(f"Obliczanie celu (Typ: {config.TARGET_TYPE}, Horyzont: {horizon} barów)...")

    outcomes = pd.Series(np.nan, index=df.index)

    if config.TARGET_TYPE == 'DYNAMIC_ATR':
        atr_col_name = f'ATRr_{config.FeatureConfig.ATR_LENGTH}_{config.BASE_TIMEFRAME}'
        if atr_col_name not in df.columns:
            raise ValueError(f"Brak kolumny ATR '{atr_col_name}' w DataFrame. Sprawdź konfigurację.")

    for i in tqdm(range(len(df) - horizon), desc="Obliczanie celu", leave=False, ncols=100):
        entry_price = df['close'].iloc[i]
        outcomes.iloc[i] = 0 # Domyślna wartość "neutral" dla próbek w pętli

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
        if pd.notna(hit_tp_time) and pd.notna(hit_sl_time):
            outcomes.iloc[i] = 1 if hit_tp_time < hit_sl_time else -1
        elif pd.notna(hit_tp_time):
            outcomes.iloc[i] = 1
        elif pd.notna(hit_sl_time):
            outcomes.iloc[i] = -1
    return outcomes


def train_unified_model(df: pd.DataFrame, model_for_trial: LGBMClassifier, full_run: bool = False):
    if not full_run:
        print(f"\n[KROK 3/4] Selekcja {config.TOP_N_FEATURES} najważniejszych cech...")
    df = df.loc[:, ~df.columns.duplicated()]
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

    if not full_run:
        print(f"-> Top 10 najważniejszych cech w tym trialu:")
        print(feature_importances.head(10).to_string(index=False))
        print(f"[KROK 4/4] Walidacja krzyżowa...")

    x_train_val = train_val_df[best_features]
    y_train_val = train_val_df['target']

    # Walidacja krzyżowa
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS, gap=5)
    cv_scores = []

    if not full_run:
        for fold, (train_index, test_index) in enumerate(tscv.split(x_train_val)):
            x_train, x_test = x_train_val.iloc[train_index], x_train_val.iloc[test_index]
            y_train, y_test = y_train_val.iloc[train_index], y_train_val.iloc[test_index]
            scaler = clone(config.SCALER)
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            model_for_cv = clone(model_for_trial)
            model_for_cv.fit(x_train_scaled, y_train)

            y_pred = model_for_cv.predict(x_test_scaled)
            f1 = f1_score(y_test, y_pred, average=None, labels=[0, 2])
            score = np.mean(f1) if f1.size > 0 else 0.0
            cv_scores.append(score)

        print(f"-> Średni F1-score z walidacji krzyżowej: {np.mean(cv_scores):.4f}")

    final_scaler = clone(config.SCALER)

    x_train_val_scaled = final_scaler.fit_transform(x_train_val)
    final_model = clone(model_for_trial)
    final_model.fit(x_train_val_scaled, y_train_val)

    # Predykcja na zbiorze holdout (testowym)
    x_holdout = holdout_df[best_features]
    y_holdout = holdout_df['target']
    x_holdout_scaled = final_scaler.transform(x_holdout)

    y_holdout_proba = final_model.predict_proba(x_holdout_scaled)
    y_holdout_pred = np.argmax(y_holdout_proba, axis=1)

    f1 = f1_score(y_holdout, y_holdout_pred, average=None, labels=[0, 2])
    mean_f1_score = np.mean(f1) if f1.size > 0 else 0.0

    return y_holdout_pred, y_holdout_proba, mean_f1_score, holdout_df, best_features


def objective(trial: optuna.Trial, df_with_target: pd.DataFrame) -> float:
    model_params = {
        'objective': 'multiclass',
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
    df_raw = await fetch_data_for_trainer_async(
        ticker=config.TICKER, start_date=config.TRAIN_START_DATE, end_date=config.TRAIN_END_DATE
    )
    if df_raw.empty: return

    features_cache_filename = f"{config.FEATURES_CACHE_DIR}/features_{config.TICKER}_{config.TRAIN_START_DATE}_{config.TRAIN_END_DATE}_{config.BASE_TIMEFRAME}.parquet"
    if os.path.exists(features_cache_filename):
        df_features = pd.read_parquet(features_cache_filename)
    else:
        df_features = prepare_feature_set_for_timeframe(df_raw, base_tf=config.BASE_TIMEFRAME)
        df_features.to_parquet(features_cache_filename)

    targets = calculate_multiclass_target(df_features, config.HORIZON_BARS)
    df_features['target'] = targets.map({-1: 0, 0: 1, 1: 2})
    df_features.dropna(inplace=True)

    if len(df_features) < 1000: return

    study_name = f"optimization_{config.TICKER}_{config.BASE_TIMEFRAME}"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize', load_if_exists=True)
    # study.optimize(lambda trial: objective(trial, df_features), n_trials=config.OPTUNA_TRIALS)

    print("\n" + "=" * 50)
    print("--- Optymalizacja Zakończona ---")
    print(f"Najlepszy F1-score w study: {study.best_value:.4f}")
    print("Najlepsze parametry:", study.best_params)
    print("=" * 50 + "\n")

    # Finalny trening z najlepszymi parametrami i pokazanie wyników
    print("--- Trenowanie finalnego modelu z najlepszymi parametrami... ---")
    best_params = study.best_params
    best_model = LGBMClassifier(objective='multiclass', random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1,
                                **best_params)

    # Uruchamiamy trening ostatni raz, aby uzyskać finalne predykcje i prawdopodobieństwa
    preds, probas, f1, df_holdout, top_features = train_unified_model(df_features.copy(), best_model, full_run=True)

    results_df = df_holdout[['open', 'high', 'low', 'close', 'target']].copy()
    results_df['prediction'] = preds
    results_df['proba_DOWN(0)'] = probas[:, 0]
    results_df['proba_SIDE(1)'] = probas[:, 1]
    results_df['proba_UP(2)'] = probas[:, 2]

    print(f"\nWynik F1-score na zbiorze testowym (holdout): {f1:.4f}")
    print("\nTop 10 cech użytych w finalnym modelu:")
    print(top_features[:10])

    print("\n--- Przykładowe predykcje finalnego modelu na danych testowych: ---")
    print(results_df.head(15).to_string())

    # Możesz zapisać wyniki do pliku
    results_df.to_csv("final_predictions.csv")

if __name__ == "__main__":
    asyncio.run(main())
