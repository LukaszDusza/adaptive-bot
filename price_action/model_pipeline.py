import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import os
import json
import gc
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, precision_score, recall_score
from numba import njit
from typing import Tuple, List
from scipy.stats import ks_2samp

from data_preparer_pa import remove_correlated_features

from profit_aware_optimizer import find_optimal_threshold_for_profit
from adaptive_labels import (
    get_adaptive_triple_barrier_labels,
    analyze_barrier_distribution
)

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)


def setup_logging(log_dir: str = "logs", module_name: str = "model_pipeline"):
    """
    Configure logging with file and console handlers.

    ETAP 2.1: Professional logging setup:
    - File handler: DEBUG+ to logs/training_YYYY-MM-DD_HH-MM-SS.log
    - Console handler: WARNING+ (clean output for user)
    - Rotating logs: 10 files × 10MB each
    - Format: [timestamp] [LEVEL] module:line - message

    Args:
        log_dir: Directory for log files
        module_name: Module name for log file naming
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_path / f"{module_name}_{timestamp}.log"

    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("="*80)
    logger.info(f"Logging initialized for {module_name}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Console level: WARNING+")
    logger.info(f"File level: DEBUG+")
    logger.info("="*80)

    return logger

_logger = setup_logging(log_dir="logs", module_name="model_pipeline")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _validate_input_data(df: pd.DataFrame, min_samples: int = 10000) -> None:
    """
    Walidacja input data przed treningiem.

    ENHANCEMENT #4: Input validation - fail-fast zamiast cryptic errors

    Args:
        df: DataFrame do walidacji
        min_samples: Minimalna liczba sampli (default: 10000)

    Raises:
        ValueError: Jeśli walidacja nie przejdzie
    """
    errors = []

    if df.empty:
        errors.append("DataFrame is empty")
    if len(df) < min_samples:
        errors.append(f"Insufficient samples: {len(df)} < {min_samples}")

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index must be DatetimeIndex")
    if df.index.duplicated().any():
        errors.append(f"Duplicate indices: {df.index.duplicated().sum()}")
    if not df.index.is_monotonic_increasing:
        errors.append("Index must be sorted chronologically")

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        nan_pct = (nan_counts[nan_cols] / len(df) * 100).round(2)
        errors.append(f"NaN values in {len(nan_cols)} columns: {dict(list(nan_pct.items())[:5])}")

    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
    inf_cols = inf_counts[inf_counts > 0]
    if len(inf_cols) > 0:
        errors.append(f"Inf values in {len(inf_cols)} columns: {dict(list(inf_cols.items())[:5])}")

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric = [col for col in non_numeric if col in df.columns]
    if non_numeric:
        errors.append(f"Non-numeric columns (will be dropped): {non_numeric[:10]}")

    if errors:
        error_msg = "Input validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    logging.info(f"✓ Input validation passed: {len(df):,} samples, {df.shape[1]} features")
def _check_feature_drift(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        selected_features: List[str] = None,
                        drift_threshold: float = 0.05) -> Tuple[List[str], pd.DataFrame]:
    """
    POPRAWKA #9: Monitoruje drift między train i test sets używając KS test.

    Args:
        train_df: Training dataset
        test_df: Test dataset (calibration lub holdout)
        selected_features: Lista cech do sprawdzenia (top 20 jeśli None)
        drift_threshold: P-value threshold (default: 0.05)

    Returns:
        Tuple[List[str], pd.DataFrame]: Lista features z driftem, DataFrame z wynikami
    """
    if selected_features is None:
        selected_features = train_df.columns[:20].tolist()

    drift_results = []
    drifted_features = []

    for feature in selected_features:
        if feature not in train_df.columns or feature not in test_df.columns:
            continue

        train_vals = train_df[feature].dropna().values
        test_vals = test_df[feature].dropna().values

        if len(train_vals) < 10 or len(test_vals) < 10:
            continue

        statistic, p_value = ks_2samp(train_vals, test_vals)

        drift_results.append({
            'feature': feature,
            'ks_statistic': statistic,
            'p_value': p_value,
            'has_drift': p_value < drift_threshold
        })

        if p_value < drift_threshold:
            drifted_features.append(feature)

    results_df = pd.DataFrame(drift_results).sort_values('p_value')
    return drifted_features, results_df


def _check_feature_drift_enhanced(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: List[str],
    target_col: str = None,
    drift_threshold: float = 0.05,
    alert_threshold: int = 10
) -> dict:
    """
    ENHANCEMENT #5: Enhanced drift monitoring z alertami i rekomendacjami.

    Rozszerzona wersja _check_feature_drift z:
    - Target drift detection (distribution shift)
    - Alert generation
    - Automatic retraining recommendations

    Args:
        train_df: Training dataset
        test_df: Test dataset
        selected_features: Lista cech do sprawdzenia
        target_col: Nazwa kolumny target (opcjonalne)
        drift_threshold: P-value threshold (default: 0.05)
        alert_threshold: Liczba features z driftem powyżej której generujemy alert

    Returns:
        Dict z kluczami:
        - drifted_features: Lista features z driftem
        - drift_df: DataFrame z wynikami
        - target_drift: Bool - czy target ma drift
        - target_drift_pvalue: P-value dla target drift
        - alerts: Lista alertów (strings)
        - should_retrain: Bool - czy rekomendowany retraining
    """
    from scipy.stats import chisquare

    drifted_features, results_df = _check_feature_drift(
        train_df, test_df, selected_features, drift_threshold
    )

    target_drift = False
    target_drift_pvalue = None

    if target_col and target_col in train_df.columns and target_col in test_df.columns:
        y_train = train_df[target_col]
        y_test = test_df[target_col]

        train_dist = y_train.value_counts(normalize=True).sort_index()
        test_dist = y_test.value_counts(normalize=True).sort_index()

        all_classes = sorted(set(train_dist.index) | set(test_dist.index))
        train_dist = train_dist.reindex(all_classes, fill_value=1e-10)
        test_dist = test_dist.reindex(all_classes, fill_value=1e-10)

        try:
            chi2_stat, chi2_p = chisquare(test_dist.values, train_dist.values)
            target_drift = chi2_p < drift_threshold
            target_drift_pvalue = chi2_p
        except Exception as e:
            logging.warning(f"⚠ Could not compute target drift: {e}")
    alerts = []
    if len(drifted_features) > alert_threshold:
        alerts.append(
            f"⚠ ALERT: {len(drifted_features)} features have significant drift (> {alert_threshold} threshold)"
        )

    if target_drift:
        alerts.append(
            f"⚠ ALERT: Target distribution has drifted (p={target_drift_pvalue:.4f} < {drift_threshold})"
        )

    should_retrain = len(drifted_features) > alert_threshold or target_drift

    return {
        'drifted_features': drifted_features,
        'drift_df': results_df,
        'target_drift': target_drift,
        'target_drift_pvalue': target_drift_pvalue,
        'alerts': alerts,
        'should_retrain': should_retrain,
        'n_drifted': len(drifted_features),
        'n_total': len(selected_features)
    }


@njit
def _compute_labels_fast(prices: np.ndarray, event_indices: np.ndarray, profit_take_pct: float, stop_loss_pct: float,
                         time_limit: int):
    """
    CRITICAL FIX v2.2: Removed parallel=True to fix crash with Optuna n_jobs=-1.

    PROBLEM: @njit(parallel=True) + Optuna n_jobs=-1 creates nested parallelism:
      - Optuna spawns multiple Python threads (one per trial)
      - Each thread calls this function with parallel=True
      - Numba workqueue threading layer is NOT thread-safe
      - Result: "Concurrent access has been detected" crash

    SOLUTION: Use sequential Numba + parallel Optuna trials (safer, still fast).
    Speedup comes from Optuna level parallelism, not Numba level.
    """
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
        logging.info("Rozpoczynanie etykietowania danych...")
    if not prices.index.is_unique:
        if verbose:
            logging.warning(f"Warning: prices index has {prices.index.duplicated().sum()} duplicates. Keeping first occurrence.")
        prices = prices[~prices.index.duplicated(keep='first')]

    prices_arr = prices.to_numpy()
    event_indices = prices.index.get_indexer(t_events)
    outcomes = _compute_labels_fast(prices_arr, event_indices, profit_take_pct, stop_loss_pct, time_limit)
    labels = pd.Series(outcomes, index=t_events)
    if verbose:
        logging.info(f"Etykietowanie zakończone. Rozkład etykiet:\n{labels.value_counts(normalize=True)}")
    return labels


def _get_strategy_id(ticker, timeframe, helper_timeframes, side: str):
    helpers_str = '_plus_' + '_'.join(helper_timeframes) if helper_timeframes else ""
    return f"{ticker}_{timeframe.replace(' ', '')}{helpers_str}_{side}"


def _get_optuna_storage(version: str, study_type: str = "study") -> str:
    """
    ENHANCEMENT #3: PostgreSQL support dla Optuna - eliminuje SQLite bottleneck.

    Zwraca storage URL dla Optuna study bazując na zmiennych środowiskowych.

    Zmienne środowiskowe (opcjonalne):
        OPTUNA_STORAGE_TYPE: "postgresql" lub "sqlite" (default: "sqlite")
        OPTUNA_DB_HOST: PostgreSQL host (default: "localhost")
        OPTUNA_DB_PORT: PostgreSQL port (default: "5432")
        OPTUNA_DB_NAME: Database name (default: "optuna")
        OPTUNA_DB_USER: Username (default: "optuna")
        OPTUNA_DB_PASSWORD: Password (required dla PostgreSQL)

    Args:
        version: Model version (e.g., "v1.2")
        study_type: Type of study (default: "study")

    Returns:
        Storage URL string dla Optuna

    Examples:
        # SQLite (default):
        sqlite:///models/v1.2/optuna/study.db

        # PostgreSQL:
        postgresql://user:pass@localhost:5432/optuna
    """
    storage_type = os.environ.get('OPTUNA_STORAGE_TYPE', 'sqlite').lower()

    if storage_type == 'postgresql':
        host = os.environ.get('OPTUNA_DB_HOST', 'localhost')
        port = os.environ.get('OPTUNA_DB_PORT', '5432')
        db_name = os.environ.get('OPTUNA_DB_NAME', 'optuna')
        user = os.environ.get('OPTUNA_DB_USER', 'optuna')
        password = os.environ.get('OPTUNA_DB_PASSWORD', '')

        if not password:
            logging.warning("⚠ WARNING: OPTUNA_DB_PASSWORD not set, falling back to SQLite")
            storage_type = 'sqlite'
        else:
            storage_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
            logging.info(f"✓ Using PostgreSQL storage: {user}@{host}:{port}/{db_name}")
            return storage_url

    optuna_dir = os.path.join("models", version, "optuna")
    os.makedirs(optuna_dir, exist_ok=True)
    storage_url = f"sqlite:///{optuna_dir}/{study_type}.db"
    logging.info(f"✓ Using SQLite storage: {storage_url}")
    return storage_url


def walk_forward_split(X, n_splits=6, test_size=0.15, gap_size=0.02):
    """
    Walk-Forward Validation dla szeregów czasowych z GAP między train/test.

    ============================================================================
    FIX #3 (HIGH): GAP BETWEEN TRAIN/TEST - Eliminacja data leakage
    ============================================================================
    PROBLEM v1.1: Test starts immediately after train → features w test[0] zawierają
                  prices from train[-1,-2,...] → forward contamination.
    FIX v2.0: Add gap = 2% (~700 candles = ~28 dni @ 15m) between train/test.
              Większość ICT patterns resolve w 3-7 dni, więc 28 dni = safe buffer.

    Expected impact:
    - Validation accuracy: -3% (more honest, no leakage)
    - Live robustness: +8% (eliminates autocorrelation artifacts)
    ============================================================================

    Args:
        X: DataFrame z features
        n_splits: Liczba fold'ów (default: 6, ale często używamy 3 dla speedu)
        test_size: Fraction of data for test (default: 0.15 = 15%)
        gap_size: Fraction of data to SKIP between train/test (default: 0.02 = 2%)
    """
    n_samples = len(X)
    test_samples = int(n_samples * test_size)
    gap_samples = max(200, int(n_samples * gap_size))
    min_train_samples = int(n_samples * 0.30)

    available_range = n_samples - min_train_samples - test_samples - gap_samples
    step_size = available_range // (n_splits - 1) if n_splits > 1 else 0

    for i in range(n_splits):
        train_end = min_train_samples + (i * step_size)
        test_start = train_end + gap_samples
        test_end = test_start + test_samples

        if test_end > n_samples:
            break

        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))

        if i == 0:
            gap_days = gap_samples * 15 / (60 * 24)
            logging.info(f"   CV Fold {i+1}: Train[0:{train_end}], GAP[{train_end}:{test_start}] ({gap_days:.1f} days), Test[{test_start}:{test_end}]")
        yield train_idx, test_idx


def _run_feature_selection(X: pd.DataFrame, y: pd.Series, strategy_id: str, version_dir: str,
                           importance_threshold: float = 0.90, top_n_features: int = None):
    """
    Selekcja cech oparta na feature importance z LightGBM.

    EXPERIMENT 2A: Threshold 0.90 (było: 0.85) → Zachowuje więcej sparse features (ICT)
    POPRAWKA #2: CV-based feature importance (5 folds) → Bardziej stabilne rankings
    ENHANCEMENT #2: Explicit memory cleanup - prevents memory leaks
    ENHANCEMENT v2.2: top_n_features parameter - select only top N features (overrides threshold)

    Args:
        X: Feature DataFrame
        y: Target Series
        strategy_id: Strategy identifier
        version_dir: Directory to save results
        importance_threshold: Cumulative importance threshold (default 0.90)
        top_n_features: If set, select only top N features (ignores threshold)

    Returns:
        List of selected feature names
    """
    logging.info("\n--- ETAP 1.5: Rozpoczynanie optymalizowanej selekcji cech (Feature Importance + CV) ---")
    logging.info(f"Cechy początkowe: {len(X.columns)}")
    importance_accumulator = np.zeros(len(X.columns))
    n_folds = 5

    logging.info(f"POPRAWKA #2: Uśrednianie feature importance z {n_folds} CV folds...")
    for fold_idx, (train_idx, val_idx) in enumerate(walk_forward_split(X, n_splits=n_folds, test_size=0.15)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]

        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
        X_scaled = scaler.fit_transform(X_train_fold)

        model = lgb.LGBMClassifier(
            random_state=42,
            objective='binary',
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5,
            num_leaves=15,
            verbose=-1
        )
        model.fit(X_scaled, y_train_fold, feature_name=X.columns.to_list())

        importance_accumulator += model.feature_importances_

        del X_train_fold, y_train_fold, X_scaled, model, scaler
        gc.collect()

        logging.info(f"  Fold {fold_idx+1}/{n_folds} complete (memory freed)")
    feature_importances = importance_accumulator / n_folds

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    importance_df['cumulative_importance'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()

    if top_n_features is not None:
        selected_features = importance_df.head(top_n_features)['feature'].tolist()
        logging.info(f"\n✂️  FEATURE PRUNING MODE: Selecting top {top_n_features} features")
        final_cumulative = importance_df.iloc[top_n_features-1]['cumulative_importance']
        logging.info(f"   → Cumulative importance at position {top_n_features}: {final_cumulative:.1%}")
    else:
        selected_features = importance_df[importance_df['cumulative_importance'] <= importance_threshold]['feature'].tolist()

        min_features = max(5, int(len(X.columns) * 0.1))
        if len(selected_features) < min_features:
            selected_features = importance_df.head(min_features)['feature'].tolist()

    logging.info(f"\nSelekcja zakończona. Wybrano {len(selected_features)} z {len(X.columns)} cech ({len(selected_features)/len(X.columns):.1%}).")
    if top_n_features is None:
        logging.info(f"Próg skumulowanej ważności: {importance_threshold * 100}%")
    logging.info(f"Top 10 najważniejszych cech:")
    for idx, row in importance_df.head(10).iterrows():
        logging.info(f"  {row['feature']}: {row['importance']:.4f} (skumulowane: {row['cumulative_importance']:.2%})")
    removed_features = [f for f in X.columns if f not in selected_features]
    logging.info(f"\nUsunięto {len(removed_features)} słabych cech z feature selection")
    if removed_features:
        logging.info(f"Przykłady usuniętych cech: {removed_features[:5]}")

    importance_rankings_path = os.path.join(version_dir, "feature_importance_rankings.csv")
    importance_df.to_csv(importance_rankings_path, index=False)
    logging.info(f"\n💾 ENHANCEMENT #9: Feature importance rankings saved to: {importance_rankings_path}")
    logging.info(f"   Top features: {', '.join(importance_df.head(5)['feature'].tolist())}")
    logging.info(f"   Weakest features: {', '.join(importance_df.tail(5)['feature'].tolist())}")

    return selected_features


def _rescue_interaction_features(
    X: pd.DataFrame,
    y: pd.Series,
    selected_features: List[str],
    improvement_threshold: float = 0.005,
    max_features_to_test: int = 50
) -> List[str]:
    """
    ENHANCEMENT #8: Rescue features that have low individual importance but critical interactions.

    Problem: Feature A może mieć low importance, ale być critical w kombinacji z Feature B.
    Przykład: FVG + Order Block mogą razem dawać silny sygnał.

    Args:
        X: Feature DataFrame
        y: Target Series
        selected_features: Lista już wybranych features
        improvement_threshold: Minimum improvement (accuracy) żeby rescue feature
        max_features_to_test: Limit ile dropped features testować (dla speedu)

    Returns:
        Lista rescued features
    """
    dropped_features = [f for f in X.columns if f not in selected_features]

    if len(dropped_features) == 0:
        logging.info("  ✓ No dropped features to check for interactions")
        return []

    logging.info(f"\n--- ENHANCEMENT #8: Checking feature interactions ---")
    logging.info(f"  Testing {min(len(dropped_features), max_features_to_test)} dropped features for interactions...")
    X_base = X[selected_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_base)
    base_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=15,
        random_state=42,
        verbose=-1
    )
    base_model.fit(X_scaled, y)
    base_score = base_model.score(X_scaled, y)

    rescue_candidates = []
    features_to_test = dropped_features[:max_features_to_test]

    for i, feat in enumerate(features_to_test):
        if (i + 1) % 10 == 0:
            logging.info(f"    Progress: {i+1}/{len(features_to_test)} features tested")
        X_test = X[selected_features + [feat]]
        X_test_scaled = scaler.fit_transform(X_test)
        test_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            num_leaves=15,
            random_state=42,
            verbose=-1
        )
        test_model.fit(X_test_scaled, y)
        test_score = test_model.score(X_test_scaled, y)

        improvement = test_score - base_score
        if improvement > improvement_threshold:
            rescue_candidates.append((feat, improvement))

    rescue_candidates = sorted(rescue_candidates, key=lambda x: x[1], reverse=True)

    if rescue_candidates:
        logging.info(f"\n  ✓ Rescued {len(rescue_candidates)} features due to interaction effects:")
        for feat, imp in rescue_candidates[:10]:
            logging.info(f"      {feat}: +{imp:.4f} accuracy")
        if len(rescue_candidates) > 10:
            logging.info(f"      ... and {len(rescue_candidates)-10} more")
        rescued_features = [f for f, _ in rescue_candidates]
        return rescued_features
    else:
        logging.info(f"  ✓ No features rescued (no significant interactions detected)")
        return []


def get_adaptive_hyperparameter_ranges(n_samples: int, n_features: int, df_close: pd.Series = None, ticker: str = None, timeframe: str = '15m') -> dict:
    """
    ENHANCEMENT #6: Adaptive hyperparameter ranges bazując na charakterystykach datasetu.
    ENHANCEMENT #17: Computed volatility zamiast hardcoded ticker lists.

    Różne datasety wymagają różnych ranges:
    - Małe datasets → silniejsza regularyzacja
    - Duże datasets → więcej trees, bardziej złożone modele
    - High-dimensional → więcej feature sampling
    - Volatile assets → szersze barriers, krótszy time limit (computed z historical data)

    Args:
        n_samples: Liczba sampli w datasecie
        n_features: Liczba features
        df_close: Series z close prices (dla computed volatility)
        ticker: Ticker symbol (opcjonalne, fallback jeśli df_close brak)
        timeframe: Timeframe string (e.g., '15m', '1h', '4h') for volatility calculation

    Returns:
        Dict z ranges dla każdego hyperparametra
    """
    ranges = {
        'n_estimators': (300, 1200),
        'learning_rate': (0.005, 0.05),
        'reg_alpha': (5.0, 80.0),
        'reg_lambda': (5.0, 80.0),
        'num_leaves': (15, 50),
        'colsample_bytree': (0.6, 0.8),
        'subsample': (0.7, 0.9),
        'subsample_freq': (1, 5),
        'min_child_samples': (50, 200),

        'barrier_size': (0.015, 0.040),
        'time_limit': (12, 48)
    }

    if n_samples < 30000:
        logging.info(f"  ➤ Small dataset detected ({n_samples:,} samples) - increasing regularization")
        ranges['reg_alpha'] = (20.0, 150.0)
        ranges['reg_lambda'] = (20.0, 150.0)
        ranges['num_leaves'] = (10, 30)
        ranges['min_child_samples'] = (100, 300)
        ranges['n_estimators'] = (200, 800)
    elif n_samples > 100000:
        logging.info(f"  ➤ Large dataset detected ({n_samples:,} samples) - increasing model capacity")
        ranges['n_estimators'] = (500, 2000)
        ranges['num_leaves'] = (20, 80)
        ranges['reg_alpha'] = (1.0, 50.0)
        ranges['reg_lambda'] = (1.0, 50.0)

    if n_features > 200:
        logging.info(f"  ➤ High-dimensional dataset ({n_features} features) - increasing feature sampling")
        ranges['colsample_bytree'] = (0.4, 0.7)

    volatility = None
    volatility_classification = "medium"

    if df_close is not None and len(df_close) > 100:
        try:
            if timeframe.endswith('m'):
                minutes_per_candle = int(timeframe[:-1])
            elif timeframe.endswith('h'):
                minutes_per_candle = int(timeframe[:-1]) * 60
            elif timeframe.endswith('D'):
                minutes_per_candle = int(timeframe[:-1]) * 1440
            elif timeframe.endswith('W'):
                minutes_per_candle = int(timeframe[:-1]) * 10080
            else:
                logging.warning(f"Unknown timeframe format: {timeframe}, defaulting to 15m")
                minutes_per_candle = 15
        except (ValueError, AttributeError, TypeError) as e:
            logging.warning(f"Failed to parse timeframe: {timeframe} ({type(e).__name__}: {e}), defaulting to 15m")
            minutes_per_candle = 15

        candles_per_year = (365 * 24 * 60) / minutes_per_candle

        candles_per_day = (24 * 60) / minutes_per_candle
        lookback_days = min(30, int(len(df_close) / candles_per_day))
        lookback_candles = int(lookback_days * candles_per_day)
        recent_close = df_close.iloc[-lookback_candles:] if lookback_candles < len(df_close) else df_close

        returns = recent_close.pct_change().dropna()
        candle_vol = returns.std()

        volatility = candle_vol * np.sqrt(candles_per_year)

        if volatility > 1.0:
            volatility_classification = "high"
        elif volatility < 0.5:
            volatility_classification = "low"
        else:
            volatility_classification = "medium"

        logging.info(f"  ➤ Computed volatility: {volatility*100:.1f}% annualized (classification: {volatility_classification})")
    elif ticker:
        volatile_tickers = ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT']
        stable_tickers = ['BTCUSDT', 'ETHUSDT']

        if ticker in volatile_tickers:
            volatility_classification = "high"
            logging.info(f"  ➤ Ticker-based classification: {ticker} → high volatility (DEPRECATED: use df_close)")
        elif ticker in stable_tickers:
            volatility_classification = "low"
            logging.info(f"  ➤ Ticker-based classification: {ticker} → low volatility (DEPRECATED: use df_close)")
        else:
            logging.info(f"  ➤ Unknown ticker {ticker}, using medium volatility defaults")
    if volatility_classification == "high":
        logging.info(f"  ➤ High volatility detected - wider barriers, shorter time limits")
        ranges['barrier_size'] = (0.015, 0.050)
        ranges['time_limit'] = (8, 36)
    elif volatility_classification == "low":
        logging.info(f"  ➤ Low volatility detected - tighter barriers, longer time limits")
        ranges['barrier_size'] = (0.012, 0.030)
        ranges['time_limit'] = (16, 60)

    return ranges


def _failed_trial_callback(study, trial):
    """
    ENHANCEMENT #1: Callback dla failed trials - logging i diagnostyka.
    """
    if trial.state == optuna.trial.TrialState.FAIL:
        logging.error(f"⚠ Trial {trial.number} FAILED with exception")
        if trial.user_attrs:
            logging.info(f"  Last known state: {trial.user_attrs}")
def _run_model_optimization(X: pd.DataFrame, y: pd.Series, n_model_trials: int, strategy_id: str, version: str, side: str = 'long', ticker: str = 'SOLUSDT', df_close: pd.Series = None, timeframe: str = '15m'):
    """
    POPRAWKA #1: RECALL-FOCUSED OPTIMIZATION dla LONG
    Zmieniono objective function aby priorytetyzować recall (łapanie okazji)
    przy zachowaniu minimum 50% precision (jakość sygnałów).

    OPTYMALIZACJA SZYBKOŚCI:
    - Zredukowano liczbę CV splits z 6 do 3 (2x szybciej)
    - Dodano pruning dla słabych trials (early stopping)
    - Włączono równoległe wykonywanie trials (n_jobs=-1)
    - Zoptymalizowano sampler (fewer startup trials)

    ENHANCEMENT #1: Failed trials handling - kontynuacja mimo błędów
    ENHANCEMENT #6 + #17: Adaptive hyperparameter ranges z computed volatility
    """
    logging.info("\n--- ENHANCEMENT #6 + #17: Computing adaptive hyperparameter ranges ---")
    ranges = get_adaptive_hyperparameter_ranges(len(X), len(X.columns), df_close=df_close, ticker=ticker, timeframe=timeframe)

    def objective_model(trial):
        params = {
            'objective': 'binary',
            'metric': 'logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'class_weight': 'balanced',
            'n_estimators': trial.suggest_int('n_estimators', *ranges['n_estimators']),
            'learning_rate': trial.suggest_float('learning_rate', *ranges['learning_rate'], log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', *ranges['reg_alpha'], log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', *ranges['reg_lambda'], log=True),
            'num_leaves': trial.suggest_int('num_leaves', *ranges['num_leaves']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *ranges['colsample_bytree']),
            'subsample': trial.suggest_float('subsample', *ranges['subsample']),
            'subsample_freq': trial.suggest_int('subsample_freq', *ranges['subsample_freq']),
            'min_child_samples': trial.suggest_int('min_child_samples', *ranges['min_child_samples']),
        }
        
        scores = []
        for fold_idx, (train_index, val_index) in enumerate(walk_forward_split(X, n_splits=3, test_size=0.15)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            if y_train.nunique() < 2:
                continue


            scaler = StandardScaler()
            scaler.set_output(transform="pandas")
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], eval_metric='logloss',
                      callbacks=[lgb.early_stopping(15, verbose=False)], feature_name=X_train.columns.to_list())

            probas = model.predict_proba(X_val_scaled)

            best_score = 0
            for thresh in np.arange(0.30, 0.71, 0.02):
                preds_at_thresh = (probas[:, 1] > thresh).astype(int)
                prec = precision_score(y_val, preds_at_thresh, zero_division=0)
                rec = recall_score(y_val, preds_at_thresh, zero_division=0)

                if prec >= 0.50:
                    score = 0.60 * rec + 0.40 * prec
                    best_score = max(best_score, score)

            scores.append(best_score if best_score > 0 else 0.0)

            if fold_idx == 1 and len(scores) > 1:
                trial.report(np.mean(scores), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if not scores:
            return 0.0
        return np.mean(scores)

    storage_name = _get_optuna_storage(version, f"{strategy_id}_model_study")

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=5,
        multivariate=True,
        warn_independent_sampling=False,
        seed=42
    )
    
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=0,
        interval_steps=1
    )
    
    study = optuna.create_study(
        study_name=f"{strategy_id}_model_optimization",
        storage=storage_name,
        direction='maximize',
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )
    
    study.optimize(
        objective_model,
        n_trials=n_model_trials,
        n_jobs=-1,
        show_progress_bar=True,
        catch=(Exception,),
        callbacks=[_failed_trial_callback]
    )

    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    if failed_trials:
        logging.error(f"\n⚠ {len(failed_trials)} trials failed during optimization (continued anyway)")
    return study.best_params


def run_training_pipeline(df_features: pd.DataFrame, n_label_trials: int, n_model_trials: int, ticker: str,
                          timeframe: str, helper_timeframes: list = None, side: str = 'long', version: str = 'v1.0',
                          min_recall_target: float = 0.55, top_n_features: int = None,
                          use_adaptive_labels: bool = True, use_profit_aware_threshold: bool = True,
                          live_tp_pct: float = 0.03, live_sl_pct: float = 0.01, live_fee_pct: float = 0.0006):
    strategy_id = _get_strategy_id(ticker, timeframe, helper_timeframes, side)

    version_dir = os.path.join("models", version, strategy_id)
    os.makedirs(version_dir, exist_ok=True)
    os.makedirs(os.path.join("models", version, "optuna"), exist_ok=True)

    logging.info(f"\n{'='*60}")
    logging.info(f"Training pipeline initialized for version: {version}")
    logging.info(f"Output directory: {version_dir}")
    logging.info(f"{'='*60}\n")
    logging.info(f"\n{'='*60}")
    logging.info("ENHANCEMENT #4: Input Data Validation")
    logging.info(f"{'='*60}")
    _validate_input_data(df_features, min_samples=10000)
    logging.info("")
    df_model_base = df_features.drop(columns=['open', 'high', 'low', 'close', 'volume', 'turnover'], errors='ignore')
    logging.info(f"📊 Features after removing OHLCV: {df_model_base.shape[1]}")
    logging.info(f"\n{'='*60}")
    logging.info("ETAP 0.5: Usuwanie skorelowanych cech (przed feature selection)")
    logging.info(f"{'='*60}")
    df_model_base, removed_corr_features = remove_correlated_features(
        df_model_base,
        target_col=None,
        correlation_threshold=0.90,
        keep_important=None
    )

    if removed_corr_features:
        corr_features_path = os.path.join(version_dir, "correlated_features_removed.json")
        with open(corr_features_path, 'w') as f:
            json.dump(removed_corr_features, f, indent=2)
        logging.info(f"💾 Zapisano listę {len(removed_corr_features)} skorelowanych cech do: {corr_features_path}")
    calibration_size = int(len(df_model_base) * 0.1)
    holdout_size = int(len(df_model_base) * 0.1)
    train_val_size = len(df_model_base) - calibration_size - holdout_size

    train_val_df = df_model_base.iloc[:train_val_size]
    calibration_df = df_model_base.iloc[train_val_size:train_val_size+calibration_size]
    holdout_df = df_model_base.iloc[train_val_size+calibration_size:]

    logging.info(f"\n📊 FIX #5: 3-way data split (80/10/10 - UPDATED v1.2):")
    logging.info(f"  Train/Val: {len(train_val_df):,} samples ({len(train_val_df)/len(df_model_base)*100:.1f}%)")
    logging.info(f"  Calibration: {len(calibration_df):,} samples ({len(calibration_df)/len(df_model_base)*100:.1f}%)")
    logging.info(f"  Holdout: {len(holdout_df):,} samples ({len(holdout_df)/len(df_model_base)*100:.1f}%)")
    logging.info(f"  ✅ Model trenuje na 80% (było 60%) = ~7 więcej miesięcy recent data")
    logging.info("\n--- ENHANCEMENT #6 + #17: Computing adaptive ranges for label optimization ---")
    label_ranges = get_adaptive_hyperparameter_ranges(len(train_val_df), len(train_val_df.columns), df_close=df_features['close'], ticker=ticker, timeframe=timeframe)

    def objective_labels(trial):

        barrier_size = trial.suggest_float('barrier_size', *label_ranges['barrier_size'], log=True)

        pt = barrier_size
        sl = barrier_size


        time_limit = trial.suggest_int('time_limit', *label_ranges['time_limit'])
        labels = get_triple_barrier_labels(df_features['close'], df_features.index, pt, sl, time_limit, verbose=False)

        X = train_val_df.copy()
        y = labels.reindex(X.index)

        if y.nunique() < 3:
            return 0.0


        actual_dist = y.value_counts(normalize=True)
        trial.set_user_attr("label_distribution", actual_dist.to_dict())

        scores = []
        for fold_idx, (train_index, val_index) in enumerate(walk_forward_split(X, n_splits=5, test_size=0.15)):
            X_train, X_val, y_train, y_val = X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]
            if y_train.nunique() < 3:
                continue
            scaler = StandardScaler()
            scaler.set_output(transform="pandas")
            X_train_scaled, X_val_scaled = scaler.fit_transform(X_train), scaler.transform(X_val)
            probe_model = lgb.LGBMClassifier(
                random_state=42,
                objective='multiclass',
                num_class=3,
                n_estimators=50,
                learning_rate=0.1,
                max_depth=5,
                verbose=-1
            )
            probe_model.fit(X_train_scaled, y_train, feature_name=X_train.columns.to_list())
            preds = probe_model.predict(X_val_scaled)
            fold_score = f1_score(y_val, preds, average='macro')
            scores.append(fold_score)

            if fold_idx == 1 and len(scores) > 1:
                trial.report(np.mean(scores), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if not scores:
            return 0.0

        base_score = np.mean(scores)

        regularization_penalty = 1.0
        if barrier_size < 0.008 or barrier_size > 0.035:
            regularization_penalty = 0.85

        final_score = base_score * regularization_penalty

        trial.set_user_attr("f1_score", base_score)
        trial.set_user_attr("regularization_penalty", regularization_penalty)
        return final_score

    logging.info("\n--- ETAP 1: Rozpoczynanie optymalizacji parametrów etykiet ---")
    logging.info("OPTYMALIZACJA: Zredukowano CV splits (6→3), dodano pruning i równoległe wykonywanie")
    storage_name_labels = _get_optuna_storage(version, f"{strategy_id}_labels_study")

    sampler_labels = optuna.samplers.TPESampler(
        n_startup_trials=5,
        multivariate=True,
        warn_independent_sampling=False,
        seed=42
    )
    
    pruner_labels = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=0,
        interval_steps=1
    )
    
    study_labels = optuna.create_study(
        study_name=f"{strategy_id}_labels_optimization",
        storage=storage_name_labels,
        direction='maximize',
        load_if_exists=True,
        sampler=sampler_labels,
        pruner=pruner_labels
    )
    
    study_labels.optimize(
        objective_labels,
        n_trials=n_label_trials,
        n_jobs=-1,
        show_progress_bar=True,
        catch=(Exception,),
        callbacks=[_failed_trial_callback]
    )

    failed_trials = [t for t in study_labels.trials if t.state == optuna.trial.TrialState.FAIL]
    if failed_trials:
        logging.error(f"\n⚠ {len(failed_trials)} trials failed during label optimization (continued anyway)")
    best_label_params = study_labels.best_params
    logging.info(f"\nNajlepsze parametry etykiet: {best_label_params}")
    label_params_path = os.path.join(version_dir, "label_params.json")
    with open(label_params_path, 'w') as f:
        json.dump(best_label_params, f, indent=2)
    logging.info(f"Parametry labelowania zapisane do: {label_params_path}")
    barrier_size = best_label_params['barrier_size']
    pt = barrier_size
    sl = barrier_size

    logging.info(f"\n📊 FIX #1: Symmetric barriers applied:")
    logging.info(f"  Profit Target: {pt*100:.2f}%")
    logging.info(f"  Stop Loss: {sl*100:.2f}%")
    logging.info(f"  Time Limit: {best_label_params['time_limit']} candles")
    logging.info(f"  Expected 50% win rate on random walk (unbiased)")

    if use_adaptive_labels:
        logging.info(f"\n🔥 PHASE 1 IMPROVEMENT #3: Using ADAPTIVE barriers (ATR-based)")
        final_labels = get_adaptive_triple_barrier_labels(
            df=df_features,
            t_events=df_features.index,
            base_barrier_pct=pt,
            atr_multiplier=1.5,
            base_time_limit=best_label_params['time_limit'],
            min_barrier_pct=0.005,
            max_barrier_pct=0.05,
            verbose=True
        )

        logging.info("\n📊 Analyzing adaptive barrier distribution...")
        barrier_analysis = analyze_barrier_distribution(
            df_features,
            final_labels,
            base_barrier_pct=pt,
            atr_multiplier=1.5
        )

        barrier_analysis_path = os.path.join(version_dir, "adaptive_barrier_analysis.csv")
        barrier_analysis.to_csv(barrier_analysis_path)
        logging.info(f"💾 Barrier analysis saved to: {barrier_analysis_path}")
    else:
        logging.info(f"\n⚠️  Using STATIC barriers (legacy mode)")
        final_labels = get_triple_barrier_labels(df_features['close'], df_features.index, pt, sl,
                                                 best_label_params['time_limit'])

    logging.info(f"\n--- Przygotowywanie binarnego zestawu danych dla modelu '{side.upper()}' ---")
    X_full_multi = train_val_df.copy()
    y_full_multi = final_labels.reindex(train_val_df.index)

    if side == 'long':
        is_long_or_hold = y_full_multi.isin([0, 1])
        X_full, y_full = X_full_multi[is_long_or_hold], y_full_multi[is_long_or_hold]
        target_names = ['HOLD (0)', 'BUY (1)']
        logging.info(f"Zachowano {len(y_full)} etykiet (HOLD i BUY). Nowy rozkład:\n{y_full.value_counts(normalize=True)}")
    elif side == 'short':
        is_short_or_hold = y_full_multi.isin([0, 2])
        X_full, y_full = X_full_multi[is_short_or_hold], y_full_multi[is_short_or_hold]
        y_full = y_full.replace(2, 1)
        target_names = ['HOLD (0)', 'SELL (1)']
        logging.info(f"Zachowano {len(y_full)} etykiet (HOLD i SELL). Nowy rozkład:\n{y_full.value_counts(normalize=True)}")
    else:
        raise ValueError("Parametr 'side' musi być 'long' lub 'short'.")

    selected_features = _run_feature_selection(X_full, y_full, strategy_id, version_dir, top_n_features=top_n_features)

    rescued_features = _rescue_interaction_features(
        X_full, y_full,
        selected_features,
        improvement_threshold=0.005,
        max_features_to_test=50
    )

    if rescued_features:
        selected_features = selected_features + rescued_features
        logging.info(f"\n✓ Final feature count: {len(selected_features)} (including {len(rescued_features)} rescued)")
    X_full = X_full[selected_features]

    logging.info("\n--- ETAP 2: Rozpoczynanie optymalizacji hiperparametrów modelu binarnego ---")
    logging.info("POPRAWKA #1 & #5 + ENHANCEMENT #6 + #17: Recall-focused optimization + computed volatility")
    best_model_params = _run_model_optimization(X_full, y_full, n_model_trials, strategy_id, version, side, ticker, df_features['close'], timeframe)
    logging.info(f"Najlepsze parametry modelu: {best_model_params}")
    logging.info("\n--- Trenowanie finalnego modelu binarnego ... ---")

    val_split_idx = int(len(X_full) * 0.8)
    X_train_final = X_full.iloc[:val_split_idx]
    X_val_final = X_full.iloc[val_split_idx:]
    y_train_final = y_full.iloc[:val_split_idx]
    y_val_final = y_full.iloc[val_split_idx:]

    final_scaler = StandardScaler()
    final_scaler.set_output(transform="pandas")
    X_train_scaled = final_scaler.fit_transform(X_train_final)
    X_val_scaled = final_scaler.transform(X_val_final)

    final_model = lgb.LGBMClassifier(objective='binary', **best_model_params)
    final_model.fit(
        X_train_scaled, y_train_final,
        eval_set=[(X_val_scaled, y_val_final)],
        eval_metric='logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        feature_name=X_train_final.columns.to_list()
    )
    logging.info(f"✓ Final model trained with early stopping. Best iteration: {final_model.best_iteration_}")
    model_path = os.path.join(version_dir, "model.joblib")
    scaler_path = os.path.join(version_dir, "scaler.joblib")
    features_path = os.path.join(version_dir, "features.joblib")

    joblib.dump(final_model, model_path)
    joblib.dump(final_scaler, scaler_path)
    joblib.dump(selected_features, features_path)

    logging.info(f"Model, skaler i lista cech zostały zapisane w: {version_dir}")
    logging.info("\n--- Przygotowanie zbiorów CALIBRATION i HOLDOUT ---")
    y_calib_multi = final_labels.reindex(calibration_df.index).dropna()
    X_calib_multi = calibration_df.loc[y_calib_multi.index]

    if side == 'long':
        is_long_or_hold_calib = y_calib_multi.isin([0, 1])
        X_calib, y_calib = X_calib_multi[is_long_or_hold_calib], y_calib_multi[is_long_or_hold_calib]
    elif side == 'short':
        is_short_or_hold_calib = y_calib_multi.isin([0, 2])
        X_calib, y_calib = X_calib_multi[is_short_or_hold_calib], y_calib_multi[is_short_or_hold_calib]
        y_calib = y_calib.replace(2, 1)

    X_calib = X_calib[selected_features]
    X_calib_scaled = final_scaler.transform(X_calib)
    calib_probas = final_model.predict_proba(X_calib_scaled)

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

    logging.info(f"Calibration set: {len(y_calib):,} samples (for threshold tuning)")
    logging.info(f"Holdout set: {len(y_holdout):,} samples (for final evaluation)")
    logging.info("\n--- POPRAWKA #9 + ENHANCEMENT #5: Enhanced Feature Drift Monitoring ---")
    top_20_features = selected_features[:20] if len(selected_features) >= 20 else selected_features

    X_full_with_target = X_full[selected_features].copy()
    X_full_with_target['target'] = y_full

    X_calib_with_target = X_calib[selected_features].copy()
    X_calib_with_target['target'] = y_calib

    X_holdout_with_target = X_holdout[selected_features].copy()
    X_holdout_with_target['target'] = y_holdout

    drift_calib_result = _check_feature_drift_enhanced(
        X_full_with_target, X_calib_with_target,
        selected_features=top_20_features,
        target_col='target',
        drift_threshold=0.05,
        alert_threshold=10
    )

    drift_holdout_result = _check_feature_drift_enhanced(
        X_full_with_target, X_holdout_with_target,
        selected_features=top_20_features,
        target_col='target',
        drift_threshold=0.05,
        alert_threshold=10
    )

    logging.info(f"✓ Enhanced drift monitoring complete:")
    logging.info(f"  Train → Calibration: {drift_calib_result['n_drifted']}/{drift_calib_result['n_total']} features with drift")
    if drift_calib_result['target_drift']:
        logging.warning(f"    ⚠ Target drift detected (p={drift_calib_result['target_drift_pvalue']:.4f})")
    logging.info(f"  Train → Holdout: {drift_holdout_result['n_drifted']}/{drift_holdout_result['n_total']} features with drift")
    if drift_holdout_result['target_drift']:
        logging.warning(f"    ⚠ Target drift detected (p={drift_holdout_result['target_drift_pvalue']:.4f})")
    if drift_calib_result['alerts']:
        logging.info(f"\n📢 CALIBRATION SET ALERTS:")
        for alert in drift_calib_result['alerts']:
            logging.info(f"  {alert}")
    if drift_holdout_result['alerts']:
        logging.info(f"\n📢 HOLDOUT SET ALERTS:")
        for alert in drift_holdout_result['alerts']:
            logging.info(f"  {alert}")
    if drift_calib_result['should_retrain'] or drift_holdout_result['should_retrain']:
        logging.info(f"\n🚨 RETRAINING RECOMMENDED 🚨")
        logging.info(f"  Significant drift detected - consider retraining model with fresh data")
    drift_report_path = os.path.join(version_dir, "feature_drift_report.csv")
    drift_calib_df = drift_calib_result['drift_df'].copy()
    drift_calib_df['set'] = 'calibration'
    drift_holdout_df = drift_holdout_result['drift_df'].copy()
    drift_holdout_df['set'] = 'holdout'
    pd.concat([drift_calib_df, drift_holdout_df]).to_csv(drift_report_path, index=False)
    logging.info(f"💾 Drift report saved to: {drift_report_path}")
    logging.info("\n--- ANALIZA PROGÓW DECYZYJNYCH (Threshold Tuning na CALIBRATION) ---")
    logging.info("FIX #5: Using CALIBRATION set (never seen by model during training)")

    if use_profit_aware_threshold:
        logging.info(f"\n🔥 PHASE 1 IMPROVEMENT #1: Using PROFIT-AWARE threshold optimization")

        calib_prices = df_features.loc[X_calib.index, 'close']

        optimal_threshold, profit_metrics = find_optimal_threshold_for_profit(
            model=final_model,
            X_calib=X_calib,
            prices=calib_prices,
            side=side,
            tp_pct=live_tp_pct,
            sl_pct=live_sl_pct,
            fee_pct=live_fee_pct,
            threshold_range=(0.30, 0.85),
            threshold_step=0.01,
            min_trades=10,
            max_holding_period=best_label_params['time_limit']
        )

        profit_metrics_path = os.path.join(version_dir, "profit_aware_threshold_metrics.json")
        with open(profit_metrics_path, 'w') as f:
            metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                   for k, v in profit_metrics.items()}
            json.dump(metrics_serializable, f, indent=2)
        logging.info(f"💾 Profit metrics saved to: {profit_metrics_path}")

    else:
        logging.info(f"\n⚠️  Using LEGACY threshold optimization (recall-focused)")
        logging.info(f"POPRAWKA #4 & #6: Threshold optimization - target {min_recall_target*100:.0f}% recall")

        precisions, recalls, thresholds = precision_recall_curve(y_calib, calib_probas[:, 1])

        min_recall = min_recall_target
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

        logging.info(f"Próg domyślny (0.5):")
        logging.info(f"  Precision: {default_precision:.3f}, Recall: {default_recall:.3f}, F1: {2 * default_precision * default_recall / (default_precision + default_recall + 1e-8):.3f}")
        logging.info(f"\nPróg optymalny ({optimal_threshold:.3f}) dla recall >= {min_recall}:")
        logging.info(f"  Precision: {optimal_precision:.3f}, Recall: {optimal_recall:.3f}, F1: {2 * optimal_precision * optimal_recall / (optimal_precision + optimal_recall + 1e-8):.3f}")
        if optimal_recall >= min_recall:
            logging.info(f"✓ Osiągnięto minimalny recall {min_recall}")
        else:
            logging.warning(f"⚠ Nie udało się osiągnąć recall >= {min_recall}. Najlepszy recall: {optimal_recall:.3f}")
    recommended_min_confidence_ratio = round(1.0 + optimal_threshold * 0.5, 2)
    logging.info(f"\n--- POPRAWKA #1: Rekomendacja min_confidence_ratio ---")
    logging.info(f"  Recommended min_confidence_ratio: {recommended_min_confidence_ratio:.2f}")
    logging.info(f"  (Heurystyka: 50% of optimal_threshold)")
    logging.warning(f"  ⚠ Dla pełnej optymalizacji użyj optuna_optimizer.py z DUAL models")
    holdout_preds_optimized = (holdout_probas[:, 1] >= optimal_threshold).astype(int)

    logging.info(f"\nRaport klasyfikacji na zbiorze holdout dla modelu '{side.upper()}':")
    if y_holdout.nunique() < 2:
        logging.info("Nie można wygenerować raportu: zbiór testowy holdout zawiera tylko jedną klasę.")
        logging.info(f"Unikalne klasy w y_holdout: {y_holdout.unique()}")
    else:
        logging.info("Próg domyślny (0.5):")
        logging.info(classification_report(y_holdout, holdout_preds, target_names=target_names))
        logging.info(f"\nPróg optymalny ({optimal_threshold:.3f}):")
        logging.info(classification_report(y_holdout, holdout_preds_optimized, target_names=target_names))
    results_df = pd.DataFrame(holdout_probas, columns=[f'proba_{target_names[0]}', f'proba_{target_names[1]}'],
                              index=X_holdout.index)
    results_df['y_true'] = y_holdout
    results_df['y_pred_default'] = holdout_preds
    results_df['y_pred_optimized'] = holdout_preds_optimized
    results_df['optimal_threshold'] = optimal_threshold

    results_path = os.path.join(version_dir, "holdout_predictions.csv")
    results_df.to_csv(results_path)
    logging.info(f"Szczegółowe wyniki ze zbioru holdout zapisano w: {results_path}")
    training_metadata = {
        "version": version,
        "ticker": ticker,
        "timeframe": timeframe,
        "helper_timeframes": helper_timeframes,
        "side": side,
        "n_label_trials": n_label_trials,
        "n_model_trials": n_model_trials,
        "best_label_params": best_label_params,
        "best_model_params": best_model_params,
        "optimal_threshold": float(optimal_threshold),
        "recommended_min_confidence_ratio": float(recommended_min_confidence_ratio),
        "n_features_selected": len(selected_features),
        "n_features_total": len(df_model_base.columns),
        "n_samples_train": len(X_full),
        "n_samples_calibration": len(X_calib),
        "n_samples_holdout": len(X_holdout),
        "pipeline_version": "v2.1",
        "fixes_applied": [
            "FIX #1: Symmetric barriers (PT=SL) - eliminates positive bias",
            "FIX #2: Removed SMOTE - use class_weight='balanced'",
            "FIX #3: Added 2% gap to walk-forward CV - prevents data leakage",
            "FIX #4: Removed balance penalty - natural label distribution",
            "FIX #5: 3-way split (train/calib/holdout) - proper threshold tuning"
        ],
        "enhancements_v2.1": [
            "POPRAWKA #1: Heuristic min_confidence_ratio recommendation",
            "POPRAWKA #2: CV-based feature selection (5 folds)",
            "POPRAWKA #4: Pruning after fold 1 (was fold 0)",
            "POPRAWKA #5: Early stopping for final model",
            "POPRAWKA #6: Parametrized min_recall_target",
            "POPRAWKA #7: Gap size respects feature lookback (min 200)",
            "POPRAWKA #8: Expanded hyperparameter ranges",
            "POPRAWKA #9: Feature drift monitoring (KS test)",
            "POPRAWKA #10: Fine-grained threshold search (21 values)"
        ]
    }
    
    metadata_path = os.path.join(version_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=2)
    logging.info(f"Metadata treningu zapisane w: {metadata_path}")
    logging.info("\n" + "=" * 60)
    logging.info("--- AUTOMATYCZNE URUCHAMIANIE ANALIZY MODELU ---")
    logging.info("=" * 60)
    analysis_output_dir = os.path.join(version_dir, "analysis")
    os.makedirs(analysis_output_dir, exist_ok=True)
    logging.info(f"Wyniki analizy zostały zapisane w katalogu: {analysis_output_dir}/")
    try:
        from analysis import run_analysis_with_args

        class AnalysisArgs:
            def __init__(self, ticker, timeframe, helper_timeframes, side, version):
                self.ticker = ticker
                self.timeframe = timeframe
                self.helper_timeframes = helper_timeframes
                self.side = side
                self.version = version

        args = AnalysisArgs(ticker, timeframe, helper_timeframes, side, version)
        run_analysis_with_args(args, output_dir=analysis_output_dir)

        logging.info(f"\n✓ Analiza modelu zakończona pomyślnie. Wszystkie wyniki zapisane w: {analysis_output_dir}/")
    except Exception as e:
        logging.warning(f"\n⚠ Błąd podczas automatycznej analizy: {e}")
        logging.info("Model został wytrenowany poprawnie, ale analiza nie powiodła się.")
        logging.info(f"Możesz uruchomić analizę ręcznie używając: python analysis.py --side {side} --ticker {ticker} --timeframe {timeframe}")
