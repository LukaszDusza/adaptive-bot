"""
FEATURE VALIDATION UTILITIES

Tools for validating feature quality, detecting drift, and monitoring production features.

Usage:
    from feature_validation import validate_ohlc, check_feature_drift, monitor_feature_stats

    # Before training
    validate_ohlc(df)

    # After training
    stats = monitor_feature_stats(df, feature_name='ict_composite_score')

    # In production
    drift_report = check_feature_drift(train_df, live_df)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp
import logging

logger = logging.getLogger(__name__)


def validate_ohlc(df: pd.DataFrame, raise_on_error: bool = True) -> Dict[str, bool]:
    """
    Validate OHLC data consistency.

    Checks:
    - High >= Low
    - Close between High and Low
    - Open between High and Low
    - Volume >= 0
    - No NaN in critical columns

    Args:
        df: DataFrame with OHLC data
        raise_on_error: If True, raise ValueError on validation failure

    Returns:
        Dict with validation results

    Raises:
        ValueError: If validation fails and raise_on_error=True
    """
    results = {
        'high_gte_low': True,
        'close_in_range': True,
        'open_in_range': True,
        'volume_positive': True,
        'no_critical_nans': True
    }

    errors = []

    # Check: High >= Low
    if not (df['high'] >= df['low']).all():
        results['high_gte_low'] = False
        invalid_count = (df['high'] < df['low']).sum()
        errors.append(f"Found {invalid_count} rows where High < Low")

    # Check: Close in range
    if not (df['close'] <= df['high']).all():
        results['close_in_range'] = False
        invalid_count = (df['close'] > df['high']).sum()
        errors.append(f"Found {invalid_count} rows where Close > High")

    if not (df['close'] >= df['low']).all():
        results['close_in_range'] = False
        invalid_count = (df['close'] < df['low']).sum()
        errors.append(f"Found {invalid_count} rows where Close < Low")

    # Check: Open in range
    if not (df['open'] <= df['high']).all():
        results['open_in_range'] = False
        invalid_count = (df['open'] > df['high']).sum()
        errors.append(f"Found {invalid_count} rows where Open > High")

    if not (df['open'] >= df['low']).all():
        results['open_in_range'] = False
        invalid_count = (df['open'] < df['low']).sum()
        errors.append(f"Found {invalid_count} rows where Open < Low")

    # Check: Volume >= 0
    if not (df['volume'] >= 0).all():
        results['volume_positive'] = False
        invalid_count = (df['volume'] < 0).sum()
        errors.append(f"Found {invalid_count} rows with negative volume")

    # Check: No NaN in critical columns
    critical_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in critical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                results['no_critical_nans'] = False
                errors.append(f"Found {nan_count} NaN values in {col}")

    # Raise or log errors
    if errors:
        error_msg = "OHLC validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        if raise_on_error:
            raise ValueError(error_msg)
        else:
            logger.warning(error_msg)

    return results


def monitor_feature_stats(
    df: pd.DataFrame,
    feature_name: str,
    save_path: str = None
) -> Dict:
    """
    Calculate and save feature statistics for monitoring.

    Args:
        df: DataFrame with feature
        feature_name: Name of feature to monitor
        save_path: Optional path to save stats JSON

    Returns:
        Dict with feature statistics
    """
    if feature_name not in df.columns:
        raise ValueError(f"Feature {feature_name} not found in DataFrame")

    series = df[feature_name].dropna()

    stats = {
        'feature_name': feature_name,
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(series),
        'n_zeros': (series == 0).sum(),
        'pct_zeros': (series == 0).mean() * 100,
        'n_nans': df[feature_name].isna().sum(),
        'pct_nans': df[feature_name].isna().mean() * 100,
        'mean': float(series.mean()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        'median': float(series.median()),
        'q25': float(series.quantile(0.25)),
        'q75': float(series.quantile(0.75)),
        'q95': float(series.quantile(0.95)),
        'q99': float(series.quantile(0.99)),
    }

    # Add sparsity classification
    if stats['pct_zeros'] > 90:
        stats['sparsity'] = 'very_sparse'
    elif stats['pct_zeros'] > 70:
        stats['sparsity'] = 'sparse'
    elif stats['pct_zeros'] > 30:
        stats['sparsity'] = 'moderate'
    else:
        stats['sparsity'] = 'dense'

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing stats if available
        if save_path.exists():
            with open(save_path, 'r') as f:
                history = json.load(f)
        else:
            history = []

        history.append(stats)

        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"Feature stats saved to {save_path}")

    return stats


def check_feature_drift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str] = None,
    drift_threshold: float = 0.05
) -> Dict:
    """
    Check for feature drift between train and test sets using KS test.

    Args:
        train_df: Training DataFrame
        test_df: Test/production DataFrame
        features: List of features to check (all numeric if None)
        drift_threshold: P-value threshold for drift detection

    Returns:
        Dict with drift analysis results
    """
    if features is None:
        # Check all numeric columns
        features = train_df.select_dtypes(include=[np.number]).columns.tolist()

    drift_results = []
    drifted_features = []

    for feature in features:
        if feature not in train_df.columns or feature not in test_df.columns:
            continue

        train_vals = train_df[feature].dropna().values
        test_vals = test_df[feature].dropna().values

        if len(train_vals) < 10 or len(test_vals) < 10:
            continue

        # Kolmogorov-Smirnov test
        statistic, pvalue = ks_2samp(train_vals, test_vals)

        has_drift = pvalue < drift_threshold

        result = {
            'feature': feature,
            'ks_statistic': float(statistic),
            'p_value': float(pvalue),
            'drift_detected': has_drift,
            'train_mean': float(train_vals.mean()),
            'test_mean': float(test_vals.mean()),
            'mean_shift_pct': float((test_vals.mean() - train_vals.mean()) / train_vals.mean() * 100) if train_vals.mean() != 0 else 0,
        }

        drift_results.append(result)

        if has_drift:
            drifted_features.append(feature)

    # Sort by p-value (most drifted first)
    drift_results = sorted(drift_results, key=lambda x: x['p_value'])

    # Summary
    summary = {
        'total_features_checked': len(features),
        'features_with_drift': len(drifted_features),
        'drift_percentage': len(drifted_features) / len(features) * 100 if features else 0,
        'threshold_used': drift_threshold,
        'drifted_features': drifted_features,
        'results': drift_results
    }

    # Alerts
    alerts = []
    if len(drifted_features) > len(features) * 0.2:
        alerts.append(f"⚠️  HIGH DRIFT: {len(drifted_features)}/{len(features)} features ({summary['drift_percentage']:.1f}%) have significant drift")
        alerts.append("   → Consider retraining model with recent data")

    if len(drifted_features) > 0:
        top_drifted = drift_results[:5]
        alerts.append("\nTop drifted features:")
        for r in top_drifted:
            if r['drift_detected']:
                alerts.append(f"  - {r['feature']}: p={r['p_value']:.4f}, shift={r['mean_shift_pct']:.1f}%")

    summary['alerts'] = alerts

    return summary


def compare_feature_importance_stability(
    importance_df1: pd.DataFrame,
    importance_df2: pd.DataFrame,
    top_n: int = 20
) -> Dict:
    """
    Compare feature importance rankings between two trainings.

    Checks if top features are stable across trainings.

    Args:
        importance_df1: First importance DataFrame (columns: feature, importance)
        importance_df2: Second importance DataFrame
        top_n: Number of top features to compare

    Returns:
        Dict with stability analysis
    """
    # Get top N features from each
    top1 = set(importance_df1.head(top_n)['feature'].tolist())
    top2 = set(importance_df2.head(top_n)['feature'].tolist())

    # Overlap analysis
    overlap = top1 & top2
    only_in_1 = top1 - top2
    only_in_2 = top2 - top1

    overlap_pct = len(overlap) / top_n * 100

    # Rank correlation for overlapping features
    overlap_features = list(overlap)
    if overlap_features:
        ranks1 = importance_df1.set_index('feature').loc[overlap_features, 'importance'].rank(ascending=False)
        ranks2 = importance_df2.set_index('feature').loc[overlap_features, 'importance'].rank(ascending=False)
        rank_corr = ranks1.corr(ranks2, method='spearman')
    else:
        rank_corr = 0.0

    result = {
        'top_n': top_n,
        'overlap_count': len(overlap),
        'overlap_percentage': overlap_pct,
        'rank_correlation': float(rank_corr),
        'stability_score': (overlap_pct / 100 + rank_corr) / 2 * 100,  # 0-100 score
        'overlapping_features': list(overlap),
        'only_in_training_1': list(only_in_1),
        'only_in_training_2': list(only_in_2),
    }

    # Classification
    if result['stability_score'] > 80:
        result['stability'] = 'STABLE'
        result['recommendation'] = "✅ Feature importance is stable across trainings"
    elif result['stability_score'] > 60:
        result['stability'] = 'MODERATE'
        result['recommendation'] = "⚠️  Some feature importance instability - monitor"
    else:
        result['stability'] = 'UNSTABLE'
        result['recommendation'] = "🔴 High feature importance instability - investigate data/model"

    return result


def validate_no_lookahead_bias(
    df: pd.DataFrame,
    current_time: pd.Timestamp = None
) -> bool:
    """
    Validate that no future data is present (all timestamps are in the past).

    Args:
        df: DataFrame with timestamp index
        current_time: Reference time (default: now)

    Returns:
        True if no look-ahead bias detected

    Raises:
        ValueError: If future data detected
    """
    if current_time is None:
        current_time = pd.Timestamp.now(tz='UTC')

    # Convert to timezone-aware if needed
    if df.index.tz is None and current_time.tz is not None:
        df_index = df.index.tz_localize('UTC')
    else:
        df_index = df.index

    future_rows = df_index > current_time

    if future_rows.any():
        n_future = future_rows.sum()
        first_future = df_index[future_rows].min()
        raise ValueError(
            f"Look-ahead bias detected: {n_future} rows with future timestamps\n"
            f"First future timestamp: {first_future}\n"
            f"Current time: {current_time}"
        )

    return True


def generate_feature_quality_report(
    df: pd.DataFrame,
    output_path: str = "feature_quality_report.json"
) -> Dict:
    """
    Generate comprehensive feature quality report.

    Args:
        df: DataFrame with features
        output_path: Path to save report

    Returns:
        Dict with quality metrics
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(df),
        'n_features': len(df.columns),
        'features': {}
    }

    # Analyze each feature
    for col in df.columns:
        if col in ['open', 'high', 'low', 'close', 'volume']:
            continue  # Skip OHLC

        series = df[col]

        feature_stats = {
            'dtype': str(series.dtype),
            'n_unique': int(series.nunique()),
            'n_zeros': int((series == 0).sum()),
            'pct_zeros': float((series == 0).mean() * 100),
            'n_nans': int(series.isna().sum()),
            'pct_nans': float(series.isna().mean() * 100),
        }

        # Numeric features
        if pd.api.types.is_numeric_dtype(series):
            feature_stats.update({
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
            })

            # Classify feature type
            if feature_stats['n_unique'] == 2:
                feature_stats['type'] = 'binary'
            elif feature_stats['pct_zeros'] > 90:
                feature_stats['type'] = 'very_sparse'
            elif feature_stats['pct_zeros'] > 70:
                feature_stats['type'] = 'sparse'
            else:
                feature_stats['type'] = 'continuous'

        report['features'][col] = feature_stats

    # Summary statistics
    feature_types = [f['type'] for f in report['features'].values() if 'type' in f]
    report['summary'] = {
        'n_binary': feature_types.count('binary'),
        'n_sparse': feature_types.count('sparse') + feature_types.count('very_sparse'),
        'n_continuous': feature_types.count('continuous'),
    }

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Feature quality report saved to {output_path}")

    return report


if __name__ == '__main__':
    # Example usage
    print("Feature Validation Utilities loaded successfully")
    print("\nAvailable functions:")
    print("  - validate_ohlc(df)")
    print("  - monitor_feature_stats(df, feature_name)")
    print("  - check_feature_drift(train_df, test_df)")
    print("  - compare_feature_importance_stability(imp1, imp2)")
    print("  - validate_no_lookahead_bias(df)")
    print("  - generate_feature_quality_report(df)")
