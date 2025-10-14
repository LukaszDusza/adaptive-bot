import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import os
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, fbeta_score, precision_recall_curve, precision_score, recall_score
from numba import njit
from imblearn.over_sampling import SMOTE

optuna.logging.set_verbosity(optuna.logging.WARNING)


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
                outcomes[i] = 1
                break
            if price <= lower_barrier:
                outcomes[i] = 2
                break
    return outcomes


def get_triple_barrier_labels(prices: pd.Series, t_events: pd.Index, profit_take_pct: float, stop_loss_pct: float,
                              time_limit: int, verbose=True):
    if verbose:
        print("Rozpoczynanie etykietowania danych...")
    prices_arr = prices.to_numpy()
    event_indices = prices.index.get_indexer(t_events)
    outcomes = _compute_labels_fast(prices_arr, event_indices, profit_take_pct, stop_loss_pct, time_limit)
    labels = pd.Series(outcomes, index=t_events)
    if verbose:
        print(f"Etykietowanie zakończone. Rozkład etykiet:\n{labels.value_counts(normalize=True)}")
    return labels


def _get_strategy_id(ticker, timeframe, helper_timeframes, side: str):
    helpers_str = '_plus_' + '_'.join(helper_timeframes) if helper_timeframes else ""
    return f"{ticker}_{timeframe.replace(' ', '')}{helpers_str}_{side}"


def walk_forward_split(X, n_splits=6, test_size=0.15):
    """
    Walk-Forward Validation dla szeregów czasowych.
    Expanding window: train window rośnie, test size stały.
    """
    n_samples = len(X)
    test_samples = int(n_samples * test_size)
    min_train_samples = int(n_samples * 0.30)
    available_range = n_samples - min_train_samples - test_samples
    step_size = available_range // (n_splits - 1) if n_splits > 1 else 0
    
    for i in range(n_splits):
        train_end = min_train_samples + (i * step_size)
        test_start = train_end
        test_end = test_start + test_samples
        
        if test_end > n_samples:
            break
        
        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))
        
        yield train_idx, test_idx


def _run_feature_selection(X: pd.DataFrame, y: pd.Series, strategy_id: str, importance_threshold: float = 0.85):
    """
    Selekcja cech oparta na feature importance z LightGBM.
    OPTYMALIZACJA: Threshold 0.85 dla zachowania większej liczby cech (lepszy precision)
    """
    print("\n--- ETAP 1.5: Rozpoczynanie optymalizowanej selekcji cech (Feature Importance) ---")
    print(f"Cechy początkowe: {len(X.columns)}")

    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    X_scaled = scaler.fit_transform(X)

    model = lgb.LGBMClassifier(
        random_state=42,
        objective='binary',
        n_estimators=100,
        learning_rate=0.05,
        verbose=-1
    )
    model.fit(X_scaled, y, feature_name=X.columns.to_list())

    feature_importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    importance_df['cumulative_importance'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()

    selected_features = importance_df[importance_df['cumulative_importance'] <= importance_threshold]['feature'].tolist()

    min_features = max(5, int(len(X.columns) * 0.1))
    if len(selected_features) < min_features:
        selected_features = importance_df.head(min_features)['feature'].tolist()

    print(f"\nSelekcja zakończona. Wybrano {len(selected_features)} z {len(X.columns)} cech.")
    print(f"Próg skumulowanej ważności: {importance_threshold * 100}%")
    print(f"Top 10 najważniejszych cech:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f} (skumulowane: {row['cumulative_importance']:.2%})")

    removed_features = [f for f in X.columns if f not in selected_features]
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    weak_features_path = os.path.join(models_dir, f"{strategy_id}_weak_features.json")

    with open(weak_features_path, 'w') as f:
        json.dump(removed_features, f, indent=2)

    print(f"\nZapisano {len(removed_features)} słabych cech do: {weak_features_path}")
    if removed_features:
        print(f"Przykłady usuniętych cech: {removed_features[:5]}")

    return selected_features


def _run_model_optimization(X: pd.DataFrame, y: pd.Series, n_model_trials: int, strategy_id: str, side: str = 'long'):
    """
    POPRAWKA #1: RECALL-FOCUSED OPTIMIZATION dla LONG
    Zmieniono objective function aby priorytetyzować recall (łapanie okazji)
    przy zachowaniu minimum 50% precision (jakość sygnałów).
    """
    def objective_model(trial):
        params = {
            'objective': 'binary',
            'metric': 'logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'class_weight': 'balanced',
            # POPRAWKA #5: Zwiększona regularyzacja
            'n_estimators': trial.suggest_int('n_estimators', 300, 1200),  # Było: 200-1000
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),  # Było: 0.01-0.1
            'reg_alpha': trial.suggest_float('reg_alpha', 10.0, 100.0, log=True),  # Było: 1e-3 - 50.0
            'reg_lambda': trial.suggest_float('reg_lambda', 10.0, 100.0, log=True),  # Było: 1e-3 - 50.0
            'num_leaves': trial.suggest_int('num_leaves', 15, 50),  # Było: 20-90 (prostsze drzewa)
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),  # Było: 0.5-0.9
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),  # Było: 0.6-0.9
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),  # Było: 1-7
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 200),  # Było: 30-150
        }
        
        scores = []
        for train_index, val_index in walk_forward_split(X, n_splits=6, test_size=0.15):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            if y_train.nunique() < 2:
                continue

            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            scaler = StandardScaler()
            scaler.set_output(transform="pandas")
            X_train_scaled = scaler.fit_transform(X_train_resampled)
            X_val_scaled = scaler.transform(X_val)

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_scaled, y_train_resampled, eval_set=[(X_val_scaled, y_val)], eval_metric='logloss',
                      callbacks=[lgb.early_stopping(15, verbose=False)], feature_name=X_train.columns.to_list())

            probas = model.predict_proba(X_val_scaled)
            
            # POPRAWKA #1: RECALL-FOCUSED OPTIMIZATION dla LONG
            # Strategia: Maksymalizuj recall przy minimum 50% precision
            best_score = 0
            for thresh in [0.40, 0.45, 0.50, 0.55, 0.60]:  # Było: [0.50, 0.60, 0.70, 0.80]
                preds_at_thresh = (probas[:, 1] > thresh).astype(int)
                prec = precision_score(y_val, preds_at_thresh, zero_division=0)
                rec = recall_score(y_val, preds_at_thresh, zero_division=0)
                
                # Minimum precision threshold (jakość sygnałów: min 50% = 1:1 TP:FP ratio)
                if prec >= 0.50:  # Było: rec >= 0.15
                    # Recall-weighted score: 60% recall + 40% precision
                    score = 0.60 * rec + 0.40 * prec  # Było: 0.70 * prec + 0.30 * rec
                    best_score = max(best_score, score)
            
            scores.append(best_score if best_score > 0 else 0.0)

        if not scores:
            return 0.0
        return np.mean(scores)

    storage_name = f"sqlite:///optuna/{strategy_id}_model_study.db"
    study = optuna.create_study(study_name=f"{strategy_id}_model_optimization", storage=storage_name,
                                direction='maximize', load_if_exists=True)
    study.optimize(objective_model, n_trials=n_model_trials, show_progress_bar=True)
    return study.best_params


def run_training_pipeline(df_features: pd.DataFrame, n_label_trials: int, n_model_trials: int, ticker: str,
                          timeframe: str, helper_timeframes: list = None, side: str = 'long'):
    strategy_id = _get_strategy_id(ticker, timeframe, helper_timeframes, side)
    df_model_base = df_features.drop(columns=['open', 'high', 'low', 'close', 'volume', 'turnover'], errors='ignore')
    holdout_size = int(len(df_model_base) * 0.2)
    train_val_df = df_model_base.iloc[:-holdout_size]
    holdout_df = df_model_base.iloc[-holdout_size:]

    def objective_labels(trial):
        # POPRAWKA #2: ZBALANSOWANE BARRIERS dla LONG
        base_barrier = trial.suggest_float('base_barrier', 0.010, 0.030, log=True)  # Było: 0.015-0.040 (1-3%)
        
        if side == 'long':
            pt_multiplier = trial.suggest_float('pt_multiplier', 1.5, 2.5)  # Było: 2.5-4.5 (PT: 1.5-7.5%)
            sl_multiplier = trial.suggest_float('sl_multiplier', 0.6, 1.2)  # Było: 0.4-1.0 (SL: 0.6-3.6%)
            pt = base_barrier * pt_multiplier
            sl = base_barrier * sl_multiplier
        elif side == 'short':
            pt_multiplier = trial.suggest_float('pt_multiplier', 1.5, 2.5)
            sl_multiplier = trial.suggest_float('sl_multiplier', 0.6, 1.2)
            pt = base_barrier * pt_multiplier
            sl = base_barrier * sl_multiplier
        else:
            pt = base_barrier
            sl = base_barrier

        time_limit = trial.suggest_int('time_limit', 8, 24)  # Było: 12-40 (8-24h dla 1h TF)
        labels = get_triple_barrier_labels(df_features['close'], df_features.index, pt, sl, time_limit, verbose=False)

        X = train_val_df.copy()
        y = labels.reindex(X.index)

        if y.nunique() < 3:
            return 0.0

        ideal_dist = {0: 0.50, 1: 0.25, 2: 0.25}
        actual_dist = y.value_counts(normalize=True)
        for i in range(3):
            if i not in actual_dist:
                actual_dist[i] = 0
        mse = ((actual_dist.sort_index() - pd.Series(ideal_dist).sort_index()) ** 2).mean()
        balance_penalty = np.exp(-10 * mse)

        scores = []
        for train_index, val_index in walk_forward_split(X, n_splits=6, test_size=0.15):
            X_train, X_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]
            if y_train.nunique() < 3:
                continue
            scaler = StandardScaler()
            scaler.set_output(transform="pandas")
            X_train_scaled, X_val_scaled = scaler.fit_transform(X_train), scaler.transform(X_val)
            probe_model = lgb.LGBMClassifier(random_state=42, objective='multiclass', num_class=3, verbose=-1)
            probe_model.fit(X_train_scaled, y_train, feature_name=X_train.columns.to_list())
            preds = probe_model.predict(X_val_scaled)
            scores.append(f1_score(y_val, preds, average='macro'))

        if not scores:
            return 0.0

        base_score = np.mean(scores)
        final_score = base_score * balance_penalty
        trial.set_user_attr("base_score", base_score)
        trial.set_user_attr("balance_penalty", balance_penalty)
        return final_score

    print("\n--- ETAP 1: Rozpoczynanie optymalizacji parametrów etykiet ---")
    print("POPRAWKA #2: Nowe zakresy barriers (niższy PT, szerszy SL, krótszy time_limit)")
    storage_name_labels = f"sqlite:///optuna/{strategy_id}_labels_study.db"
    study_labels = optuna.create_study(study_name=f"{strategy_id}_labels_optimization", storage=storage_name_labels,
                                       direction='maximize', load_if_exists=True)
    study_labels.optimize(objective_labels, n_trials=n_label_trials, show_progress_bar=True)
    best_label_params = study_labels.best_params
    print(f"\nNajlepsze parametry etykiet: {best_label_params}")
    
    # Zapisz label_params do pliku JSON
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    label_params_path = os.path.join(models_dir, f"{strategy_id}_label_params.json")
    with open(label_params_path, 'w') as f:
        json.dump(best_label_params, f, indent=2)
    print(f"Parametry labelowania zapisane do: {label_params_path}")

    base_barrier = best_label_params['base_barrier']
    if side == 'long':
        pt = base_barrier * best_label_params['pt_multiplier']
        sl = base_barrier * best_label_params['sl_multiplier']
    elif side == 'short':
        pt = base_barrier * best_label_params['pt_multiplier']
        sl = base_barrier * best_label_params['sl_multiplier']
    else:
        pt = base_barrier
        sl = base_barrier

    final_labels = get_triple_barrier_labels(df_features['close'], df_features.index, pt, sl,
                                             best_label_params['time_limit'])

    print(f"\n--- Przygotowywanie binarnego zestawu danych dla modelu '{side.upper()}' ---")
    X_full_multi = train_val_df.copy()
    y_full_multi = final_labels.reindex(train_val_df.index)

    if side == 'long':
        is_long_or_hold = y_full_multi.isin([0, 1])
        X_full, y_full = X_full_multi[is_long_or_hold], y_full_multi[is_long_or_hold]
        target_names = ['HOLD (0)', 'BUY (1)']
        print(f"Zachowano {len(y_full)} etykiet (HOLD i BUY). Nowy rozkład:\n{y_full.value_counts(normalize=True)}")
    elif side == 'short':
        is_short_or_hold = y_full_multi.isin([0, 2])
        X_full, y_full = X_full_multi[is_short_or_hold], y_full_multi[is_short_or_hold]
        y_full = y_full.replace(2, 1)
        target_names = ['HOLD (0)', 'SELL (1)']
        print(f"Zachowano {len(y_full)} etykiet (HOLD i SELL). Nowy rozkład:\n{y_full.value_counts(normalize=True)}")
    else:
        raise ValueError("Parametr 'side' musi być 'long' lub 'short'.")

    selected_features = _run_feature_selection(X_full, y_full, strategy_id)

    X_full = X_full[selected_features]

    print("\n--- ETAP 2: Rozpoczynanie optymalizacji hiperparametrów modelu binarnego ---")
    print("POPRAWKA #1 & #5: Recall-focused optimization + mocniejsza regularyzacja")
    best_model_params = _run_model_optimization(X_full, y_full, n_model_trials, strategy_id, side)
    print(f"Najlepsze parametry modelu: {best_model_params}")

    print("\n--- Trenowanie finalnego modelu binarnego ... ---")
    smote_final = SMOTE(random_state=42)
    X_resampled, y_resampled = smote_final.fit_resample(X_full, y_full)

    final_scaler = StandardScaler()
    final_scaler.set_output(transform="pandas")
    X_scaled = final_scaler.fit_transform(X_resampled)

    final_model = lgb.LGBMClassifier(objective='binary', **best_model_params)
    final_model.fit(X_scaled, y_resampled, feature_name=X_full.columns.to_list())

    model_path = os.path.join(models_dir, f"{strategy_id}_model.joblib")
    scaler_path = os.path.join(models_dir, f"{strategy_id}_scaler.joblib")
    features_path = os.path.join(models_dir, f"{strategy_id}_features.joblib")

    joblib.dump(final_model, model_path)
    joblib.dump(final_scaler, scaler_path)
    joblib.dump(selected_features, features_path)

    print(f"Model, skaler i lista cech zostały zapisane w katalogu '{models_dir}'.")

    y_holdout_multi = final_labels.reindex(holdout_df.index).dropna()
    X_holdout_multi = holdout_df.loc[y_holdout_multi.index]

    if side == 'long':
        is_long_or_hold_holdout = y_holdout_multi.isin([0, 1])
        X_holdout, y_holdout = X_holdout_multi[is_long_or_hold_holdout], y_holdout_multi[is_long_or_hold_holdout]
    elif side == 'short':
        is_short_or_hold_holdout = y_holdout_multi.isin([0, 2])
        X_holdout, y_holdout = X_holdout_multi[is_short_or_hold_holdout], y_holdout_multi[is_short_or_hold_holdout]
        y_holdout = y_holdout.replace(2, 1)

    X_holdout = X_holdout[selected_features]

    X_holdout_scaled = final_scaler.transform(X_holdout)
    holdout_preds = final_model.predict(X_holdout_scaled)
    holdout_probas = final_model.predict_proba(X_holdout_scaled)

    print("\n--- ANALIZA PROGÓW DECYZYJNYCH (Threshold Tuning) ---")
    print("POPRAWKA #4: Threshold optimization - target 55% recall")
    precisions, recalls, thresholds = precision_recall_curve(y_holdout, holdout_probas[:, 1])

    # POPRAWKA #4: Zmiana minimum recall z 0.70 na 0.55
    min_recall = 0.55  # Było: 0.70
    valid_indices = np.where(recalls[:-1] >= min_recall)[0]

    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(precisions[valid_indices])]
        optimal_threshold = thresholds[best_idx]
        optimal_precision = precisions[best_idx]
        optimal_recall = recalls[best_idx]
    else:
        best_idx = np.argmax(recalls[:-1])
        optimal_threshold = thresholds[best_idx]
        optimal_precision = precisions[best_idx]
        optimal_recall = recalls[best_idx]

    default_idx = np.argmin(np.abs(thresholds - 0.5))
    default_precision = precisions[default_idx]
    default_recall = recalls[default_idx]

    print(f"Próg domyślny (0.5):")
    print(f"  Precision: {default_precision:.3f}, Recall: {default_recall:.3f}, F1: {2 * default_precision * default_recall / (default_precision + default_recall + 1e-8):.3f}")
    print(f"\nPróg optymalny ({optimal_threshold:.3f}) dla recall >= {min_recall}:")
    print(f"  Precision: {optimal_precision:.3f}, Recall: {optimal_recall:.3f}, F1: {2 * optimal_precision * optimal_recall / (optimal_precision + optimal_recall + 1e-8):.3f}")

    if optimal_recall >= min_recall:
        print(f"✓ Osiągnięto minimalny recall {min_recall}")
    else:
        print(f"⚠ Nie udało się osiągnąć recall >= {min_recall}. Najlepszy recall: {optimal_recall:.3f}")

    holdout_preds_optimized = (holdout_probas[:, 1] >= optimal_threshold).astype(int)

    print(f"\nRaport klasyfikacji na zbiorze holdout dla modelu '{side.upper()}':")

    if y_holdout.nunique() < 2:
        print("Nie można wygenerować raportu: zbiór testowy holdout zawiera tylko jedną klasę.")
        print(f"Unikalne klasy w y_holdout: {y_holdout.unique()}")
    else:
        print("Próg domyślny (0.5):")
        print(classification_report(y_holdout, holdout_preds, target_names=target_names))

        print(f"\nPróg optymalny ({optimal_threshold:.3f}):")
        print(classification_report(y_holdout, holdout_preds_optimized, target_names=target_names))

    results_df = pd.DataFrame(holdout_probas, columns=[f'proba_{target_names[0]}', f'proba_{target_names[1]}'],
                              index=X_holdout.index)
    results_df['y_true'] = y_holdout
    results_df['y_pred_default'] = holdout_preds
    results_df['y_pred_optimized'] = holdout_preds_optimized
    results_df['optimal_threshold'] = optimal_threshold

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{strategy_id}_holdout_predictions.csv")
    results_df.to_csv(results_path)
    print(f"Szczegółowe wyniki ze zbioru holdout zapisano w: {results_path}")

    print("\n" + "=" * 60)
    print("--- AUTOMATYCZNE URUCHAMIANIE ANALIZY MODELU ---")
    print("=" * 60)

    analysis_output_dir = os.path.join(strategy_id)
    os.makedirs(analysis_output_dir, exist_ok=True)
    print(f"Wyniki analizy zostały zapisane w katalogu: {analysis_output_dir}/")

    try:
        from analysis import run_analysis_with_args

        class AnalysisArgs:
            def __init__(self, ticker, timeframe, helper_timeframes, side):
                self.ticker = ticker
                self.timeframe = timeframe
                self.helper_timeframes = helper_timeframes
                self.side = side

        args = AnalysisArgs(ticker, timeframe, helper_timeframes, side)
        run_analysis_with_args(args, output_dir=analysis_output_dir)

        print(f"\n✓ Analiza modelu zakończona pomyślnie. Wszystkie wyniki zapisane w: {analysis_output_dir}/")
    except Exception as e:
        print(f"\n⚠ Błąd podczas automatycznej analizy: {e}")
        print("Model został wytrenowany poprawnie, ale analiza nie powiodła się.")
        print(f"Możesz uruchomić analizę ręcznie używając: python analysis.py --side {side} --ticker {ticker} --timeframe {timeframe}")
