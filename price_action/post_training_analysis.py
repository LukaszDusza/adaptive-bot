"""
Post-Training Analysis Script for TIER 3 Model Enhancement

This script implements the three recommended analyses from TIER_3_IMPLEMENTATION_SUMMARY.md:
1. Feature importance analysis: Remove features with importance < 0.1%
2. Correlation matrix: Check if TIER 3 features are too correlated with existing ones
3. Holdout validation: Enhanced metrics comparison and validation

Usage:
    python post_training_analysis.py --ticker SOLUSDT --timeframe 15m --side long --helper_timeframes 1h 2h 4h

Author: AI Assistant
Date: 2025-10-10
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    cohen_kappa_score
)


def _get_strategy_id(ticker, timeframe, helper_timeframes, side: str):
    """Generate strategy ID from parameters."""
    helpers_str = '_plus_' + '_'.join(helper_timeframes) if helper_timeframes else ""
    return f"{ticker}_{timeframe.replace(' ', '')}{helpers_str}_{side}"


def load_model_data(strategy_id: str):
    """Load all necessary model data for analysis."""
    models_dir = "models"
    results_dir = "results"
    
    model_path = os.path.join(models_dir, f"{strategy_id}_model.joblib")
    scaler_path = os.path.join(models_dir, f"{strategy_id}_scaler.joblib")
    features_path = os.path.join(models_dir, f"{strategy_id}_features.joblib")
    holdout_path = os.path.join(results_dir, f"{strategy_id}_holdout_predictions.csv")
    
    # Check if all files exist
    missing_files = []
    for path, name in [(model_path, "Model"), (scaler_path, "Scaler"), 
                        (features_path, "Features"), (holdout_path, "Holdout predictions")]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print(f"❌ Błąd: Nie znaleziono następujących plików dla strategii '{strategy_id}':")
        for missing_file in missing_files:
            print(f"  - {missing_file}")
        return None
    
    # Load all data
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    holdout_df = pd.read_csv(holdout_path, index_col='timestamp', parse_dates=True)
    
    return {
        'model': model,
        'scaler': scaler,
        'features': features,
        'holdout_df': holdout_df
    }


def analyze_feature_importance(model, features, threshold_pct=0.1, strategy_id="model"):
    """
    Analyze feature importance and identify features below threshold.
    
    Args:
        model: Trained LightGBM model
        features: List of feature names
        threshold_pct: Minimum importance percentage (default 0.1%)
        strategy_id: Strategy identifier for saving plots
    
    Returns:
        dict with analysis results
    """
    print("\n" + "="*80)
    print("📊 ANALIZA 1: FEATURE IMPORTANCE - Filtrowanie cech o niskiej wartości")
    print("="*80)
    
    # Get feature importance
    importance_values = model.feature_importances_
    feature_names = model.feature_name_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False)
    
    # Calculate percentage
    total_importance = importance_df['importance'].sum()
    importance_df['importance_pct'] = (importance_df['importance'] / total_importance) * 100
    importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()
    
    # Filter by threshold
    low_importance = importance_df[importance_df['importance_pct'] < threshold_pct]
    high_importance = importance_df[importance_df['importance_pct'] >= threshold_pct]
    
    print(f"\n📈 Statystyki Feature Importance:")
    print(f"  Całkowita liczba cech: {len(importance_df)}")
    print(f"  Cechy o importance >= {threshold_pct}%: {len(high_importance)} ({len(high_importance)/len(importance_df)*100:.1f}%)")
    print(f"  Cechy o importance < {threshold_pct}%: {len(low_importance)} ({len(low_importance)/len(importance_df)*100:.1f}%)")
    
    if len(low_importance) > 0:
        print(f"\n⚠️  REKOMENDACJA: Rozważ usunięcie {len(low_importance)} cech o niskiej importance:")
        print(f"\n{'Feature':<50} {'Importance %':>12}")
        print("-" * 65)
        for _, row in low_importance.head(20).iterrows():
            print(f"{row['feature']:<50} {row['importance_pct']:>11.4f}%")
        
        if len(low_importance) > 20:
            print(f"... i {len(low_importance) - 20} więcej")
    else:
        print(f"\n✅ Wszystkie cechy mają importance >= {threshold_pct}%")
    
    # Top 30 features
    print(f"\n🏆 TOP 30 najważniejszych cech:")
    print(f"\n{'Rank':<6} {'Feature':<50} {'Importance %':>12} {'Cumulative %':>12}")
    print("-" * 85)
    for idx, row in importance_df.head(30).iterrows():
        rank = importance_df.index.get_loc(idx) + 1
        print(f"{rank:<6} {row['feature']:<50} {row['importance_pct']:>11.4f}% {row['cumulative_pct']:>11.2f}%")
    
    # Visualizations
    output_dir = os.path.join("results", "post_training_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Top 30 features
    plt.figure(figsize=(12, 10))
    top_30 = importance_df.head(30)
    sns.barplot(x='importance_pct', y='feature', data=top_30, palette='viridis')
    plt.xlabel('Importance (%)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top 30 Feature Importance - {strategy_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{strategy_id}_feature_importance_top30.png"), dpi=300)
    plt.close()
    
    # Plot 2: Cumulative importance
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(importance_df) + 1), importance_df['cumulative_pct'].values, linewidth=2)
    plt.axhline(y=95, color='r', linestyle='--', label='95% threshold')
    plt.axhline(y=99, color='orange', linestyle='--', label='99% threshold')
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Cumulative Importance (%)', fontsize=12)
    plt.title(f'Cumulative Feature Importance - {strategy_id}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{strategy_id}_cumulative_importance.png"), dpi=300)
    plt.close()
    
    print(f"\n📁 Wykresy zapisane w: {output_dir}/")
    
    # Save low importance features to file
    if len(low_importance) > 0:
        low_importance_path = os.path.join(output_dir, f"{strategy_id}_low_importance_features.csv")
        low_importance.to_csv(low_importance_path, index=False)
        print(f"📄 Lista cech o niskiej importance zapisana: {low_importance_path}")
    
    return {
        'importance_df': importance_df,
        'low_importance': low_importance,
        'high_importance': high_importance,
        'threshold_pct': threshold_pct
    }


def identify_tier3_features():
    """
    Identify TIER 3 features based on TIER_3_IMPLEMENTATION_SUMMARY.md
    
    Returns:
        dict with categorized TIER 3 features
    """
    tier3_features = {
        'support_resistance': [
            'resistance_50', 'support_50', 'dist_from_resistance', 'dist_from_support',
            'testing_resistance', 'testing_support', 'resistance_100', 'support_100',
            'near_resistance_100', 'near_support_100', 'resistance_strength', 'support_strength'
        ],
        'advanced_patterns': [
            'three_line_strike', 'morning_evening_star', 'gap_up', 'gap_down',
            'closed_near_high', 'closed_near_low'
        ],
        'sr_interactions': [
            'resistance_volume_interaction', 'support_volume_interaction',
            'testing_resistance_with_volume', 'testing_support_with_volume',
            'resistance_momentum_interaction', 'support_momentum_interaction',
            'resistance_rsi_interaction', 'support_rsi_interaction'
        ]
    }
    
    # Add helper timeframe variants (e.g., resistance_50_1h, resistance_50_2h)
    helper_suffixes = ['_1h', '_2h', '_4h', '_1D']
    tier3_with_helpers = {}
    
    for category, features in tier3_features.items():
        expanded_features = features.copy()
        for feature in features:
            for suffix in helper_suffixes:
                expanded_features.append(feature + suffix)
        tier3_with_helpers[category] = expanded_features
    
    return tier3_with_helpers


def analyze_correlation_matrix(model, features, strategy_id="model"):
    """
    Analyze correlation between TIER 3 features and existing features.
    
    Args:
        model: Trained LightGBM model
        features: List of feature names
        strategy_id: Strategy identifier for saving plots
    
    Returns:
        dict with correlation analysis results
    """
    print("\n" + "="*80)
    print("🔗 ANALIZA 2: CORRELATION MATRIX - Sprawdzenie korelacji cech TIER 3")
    print("="*80)
    
    # Identify TIER 3 features
    tier3_dict = identify_tier3_features()
    all_tier3_features = []
    for category, feats in tier3_dict.items():
        all_tier3_features.extend(feats)
    
    # Find which TIER 3 features are actually in the model
    present_tier3 = [f for f in all_tier3_features if f in features]
    
    if len(present_tier3) == 0:
        print("⚠️  Nie znaleziono żadnych cech TIER 3 w modelu.")
        print("    Model prawdopodobnie nie został jeszcze retrenowany z TIER 3 features.")
        return None
    
    print(f"\n✅ Znaleziono {len(present_tier3)} cech TIER 3 w modelu:")
    for category, feats in tier3_dict.items():
        category_present = [f for f in feats if f in features]
        if category_present:
            print(f"  📌 {category}: {len(category_present)} cech")
            for feat in category_present[:5]:  # Show first 5
                print(f"     - {feat}")
            if len(category_present) > 5:
                print(f"     ... i {len(category_present) - 5} więcej")
    
    # Load feature data for correlation analysis
    # We need the actual feature values - load from training data if available
    print("\n⚠️  Uwaga: Analiza korelacji wymaga danych treningowych.")
    print("    Aby wykonać pełną analizę korelacji, potrzebny jest DataFrame z wartościami cech.")
    print("    Obecnie możemy tylko zidentyfikować, które cechy TIER 3 są obecne.")
    
    # Create a simple report showing which TIER 3 features exist
    output_dir = os.path.join("results", "post_training_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    tier3_report = pd.DataFrame({
        'tier3_feature': present_tier3,
        'category': [
            'support_resistance' if f in tier3_dict['support_resistance'] else
            'advanced_patterns' if f in tier3_dict['advanced_patterns'] else
            'sr_interactions' if f in tier3_dict['sr_interactions'] else 'unknown'
            for f in present_tier3
        ]
    })
    
    report_path = os.path.join(output_dir, f"{strategy_id}_tier3_features_present.csv")
    tier3_report.to_csv(report_path, index=False)
    print(f"\n📄 Raport cech TIER 3 zapisany: {report_path}")
    
    # Recommendations based on what we can see
    print("\n💡 REKOMENDACJE dot. korelacji:")
    print("  1. Cechy S/R mogą korelować z pivot points (dist_from_r1, dist_from_s1)")
    print("  2. Interaction features (X × Y) mogą mieć wysoką korelację z bazowymi cechami")
    print("  3. Jeśli feature importance dla TIER 3 jest niska, może to wskazywać na redundancję")
    print("\n  ✅ Akcja: Sprawdź feature importance dla cech TIER 3 w Analizie 1")
    
    return {
        'present_tier3': present_tier3,
        'tier3_report': tier3_report,
        'categories': {cat: [f for f in feats if f in features] 
                      for cat, feats in tier3_dict.items()}
    }


def analyze_holdout_validation(holdout_df, side='long', strategy_id="model"):
    """
    Enhanced holdout validation metrics analysis.
    
    Args:
        holdout_df: DataFrame with holdout predictions
        side: 'long' or 'short'
        strategy_id: Strategy identifier for saving plots
    
    Returns:
        dict with validation metrics
    """
    print("\n" + "="*80)
    print("✅ ANALIZA 3: HOLDOUT VALIDATION - Szczegółowa analiza metryk")
    print("="*80)
    
    signal_name = 'BUY' if side == 'long' else 'SELL'
    proba_col = f'proba_{signal_name} (1)'
    
    # Calculate all metrics
    y_true = holdout_df['y_true']
    y_pred_default = holdout_df['y_pred_default']
    y_pred_optimized = holdout_df['y_pred_optimized']
    proba = holdout_df[proba_col]
    
    metrics = {}
    
    # Metrics for default threshold (0.5)
    metrics['default'] = {
        'threshold': 0.5,
        'precision': precision_score(y_true, y_pred_default),
        'recall': recall_score(y_true, y_pred_default),
        'f1': f1_score(y_true, y_pred_default),
        'mcc': matthews_corrcoef(y_true, y_pred_default),
        'roc_auc': roc_auc_score(y_true, proba),
        'pr_auc': average_precision_score(y_true, proba),
        'kappa': cohen_kappa_score(y_true, y_pred_default)
    }
    
    # Metrics for optimized threshold
    optimal_threshold = holdout_df['optimal_threshold'].iloc[0]
    metrics['optimized'] = {
        'threshold': optimal_threshold,
        'precision': precision_score(y_true, y_pred_optimized),
        'recall': recall_score(y_true, y_pred_optimized),
        'f1': f1_score(y_true, y_pred_optimized),
        'mcc': matthews_corrcoef(y_true, y_pred_optimized),
        'roc_auc': roc_auc_score(y_true, proba),
        'pr_auc': average_precision_score(y_true, proba),
        'kappa': cohen_kappa_score(y_true, y_pred_optimized)
    }
    
    # Confusion matrices
    cm_default = confusion_matrix(y_true, y_pred_default)
    cm_optimized = confusion_matrix(y_true, y_pred_optimized)
    
    # Additional metrics from confusion matrix (default)
    tn, fp, fn, tp = cm_default.ravel()
    metrics['default']['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['default']['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['default']['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Additional metrics from confusion matrix (optimized)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_optimized.ravel()
    metrics['optimized']['specificity'] = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0
    metrics['optimized']['fpr'] = fp_opt / (fp_opt + tn_opt) if (fp_opt + tn_opt) > 0 else 0
    metrics['optimized']['fnr'] = fn_opt / (fn_opt + tp_opt) if (fn_opt + tp_opt) > 0 else 0
    
    # Win rate and signal frequency
    metrics['default']['win_rate'] = (y_true == y_pred_default).mean()
    metrics['default']['signal_freq'] = y_pred_default.mean()
    metrics['optimized']['win_rate'] = (y_true == y_pred_optimized).mean()
    metrics['optimized']['signal_freq'] = y_pred_optimized.mean()
    
    # Probability analysis
    tp_proba = holdout_df[(y_true == 1) & (y_pred_default == 1)][proba_col].mean()
    fp_proba = holdout_df[(y_true == 0) & (y_pred_default == 1)][proba_col].mean()
    confidence_gap = tp_proba - fp_proba
    
    metrics['confidence'] = {
        'tp_avg_proba': tp_proba,
        'fp_avg_proba': fp_proba,
        'confidence_gap': confidence_gap
    }
    
    # Print results
    print(f"\n📊 METRYKI HOLDOUT - Porównanie progów:")
    print(f"\n{'Metryka':<25} {'Próg 0.5':>12} {'Próg {:.3f}'.format(optimal_threshold):>12} {'Zmiana':>12}")
    print("-" * 65)
    
    metric_names = {
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'mcc': 'MCC',
        'roc_auc': 'ROC AUC',
        'pr_auc': 'PR AUC',
        'kappa': "Cohen's Kappa",
        'specificity': 'Specificity',
        'fpr': 'False Positive Rate',
        'fnr': 'False Negative Rate',
        'win_rate': 'Win Rate',
        'signal_freq': 'Signal Frequency'
    }
    
    for key, name in metric_names.items():
        default_val = metrics['default'][key]
        optimized_val = metrics['optimized'][key]
        change = optimized_val - default_val
        change_str = f"{change:+.4f}" if key != 'roc_auc' else "same"
        print(f"{name:<25} {default_val:>12.4f} {optimized_val:>12.4f} {change_str:>12}")
    
    print(f"\n🎯 CONFIDENCE ANALYSIS:")
    print(f"  TP avg probability: {tp_proba:.4f}")
    print(f"  FP avg probability: {fp_proba:.4f}")
    print(f"  Confidence Gap: {confidence_gap:.4f}")
    
    # Recommendations based on metrics
    print(f"\n💡 REKOMENDACJE:")
    
    # Check if metrics meet targets from TIER_3_IMPLEMENTATION_SUMMARY
    targets = {
        'ROC AUC': (metrics['default']['roc_auc'], 0.75, "✅" if metrics['default']['roc_auc'] >= 0.75 else "❌"),
        'MCC': (metrics['default']['mcc'], 0.40, "✅" if metrics['default']['mcc'] >= 0.40 else "❌"),
        'Precision': (metrics['default']['precision'], 0.60, "✅" if metrics['default']['precision'] >= 0.60 else "❌"),
        'Recall': (metrics['optimized']['recall'], 0.60, "✅" if metrics['optimized']['recall'] >= 0.60 else "❌"),
        'PR AUC': (metrics['default']['pr_auc'], 0.65, "✅" if metrics['default']['pr_auc'] >= 0.65 else "❌"),
        'Confidence Gap': (confidence_gap, 0.10, "✅" if confidence_gap >= 0.10 else "❌")
    }
    
    print(f"\n  📈 Porównanie z celami (TIER 3):")
    print(f"  {'Metryka':<20} {'Wartość':>10} {'Cel':>10} {'Status':>8}")
    print("  " + "-" * 50)
    
    for metric_name, (value, target, status) in targets.items():
        print(f"  {metric_name:<20} {value:>10.4f} {target:>10.2f} {status:>8}")
    
    passed = sum(1 for _, _, status in targets.values() if status == "✅")
    total = len(targets)
    
    print(f"\n  🎯 Wynik: {passed}/{total} celów osiągniętych")
    
    if passed >= 5:
        print(f"  ✅ Model spełnia większość celów TIER 3! Gotowy do paper tradingu.")
    elif passed >= 3:
        print(f"  🟡 Model częściowo spełnia cele. Rozważ dalsze tuning lub ensemble.")
    else:
        print(f"  ❌ Model nie spełnia celów. Rozważ zmianę asset, timeframe lub labeling strategy.")
    
    # Save metrics to JSON
    output_dir = os.path.join("results", "post_training_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_path = os.path.join(output_dir, f"{strategy_id}_holdout_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n📄 Metryki holdout zapisane: {metrics_path}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Metrics comparison
    comparison_metrics = ['precision', 'recall', 'f1', 'mcc']
    default_values = [metrics['default'][m] for m in comparison_metrics]
    optimized_values = [metrics['optimized'][m] for m in comparison_metrics]
    
    x = np.arange(len(comparison_metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, default_values, width, label='Threshold 0.5', alpha=0.8)
    axes[0].bar(x + width/2, optimized_values, width, label=f'Threshold {optimal_threshold:.3f}', alpha=0.8)
    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Threshold Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.upper() for m in comparison_metrics])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Target achievement
    target_names = list(targets.keys())
    target_values = [value for value, _, _ in targets.values()]
    target_goals = [target for _, target, _ in targets.values()]
    
    axes[1].barh(target_names, target_values, alpha=0.8, label='Actual')
    axes[1].scatter(target_goals, target_names, color='red', s=100, marker='|', linewidths=3, label='Target', zorder=3)
    axes[1].set_xlabel('Value')
    axes[1].set_title('Target Achievement')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{strategy_id}_holdout_comparison.png"), dpi=300)
    plt.close()
    
    print(f"📊 Wykres porównawczy zapisany: {strategy_id}_holdout_comparison.png")
    
    return metrics


def generate_summary_report(feature_analysis, correlation_analysis, validation_metrics, strategy_id):
    """Generate comprehensive summary report."""
    print("\n" + "="*80)
    print("📝 PODSUMOWANIE ANALIZY POST-TRAINING")
    print("="*80)
    
    output_dir = os.path.join("results", "post_training_analysis")
    report_path = os.path.join(output_dir, f"{strategy_id}_analysis_summary.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("POST-TRAINING ANALYSIS SUMMARY\n")
        f.write(f"Strategy: {strategy_id}\n")
        f.write("="*80 + "\n\n")
        
        # Feature importance summary
        f.write("1. FEATURE IMPORTANCE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        if feature_analysis:
            low_count = len(feature_analysis['low_importance'])
            total_count = len(feature_analysis['importance_df'])
            f.write(f"Total features: {total_count}\n")
            f.write(f"Features with importance < {feature_analysis['threshold_pct']}%: {low_count}\n")
            f.write(f"Features with importance >= {feature_analysis['threshold_pct']}%: {total_count - low_count}\n\n")
            
            if low_count > 0:
                f.write(f"RECOMMENDATION: Consider removing {low_count} low-importance features\n")
                f.write("See: {}_low_importance_features.csv\n".format(strategy_id))
            else:
                f.write("All features have sufficient importance.\n")
        f.write("\n")
        
        # Correlation analysis summary
        f.write("2. CORRELATION MATRIX ANALYSIS (TIER 3 Features)\n")
        f.write("-" * 80 + "\n")
        if correlation_analysis:
            tier3_count = len(correlation_analysis['present_tier3'])
            f.write(f"TIER 3 features found in model: {tier3_count}\n")
            for cat, feats in correlation_analysis['categories'].items():
                if feats:
                    f.write(f"  - {cat}: {len(feats)} features\n")
            f.write("\nRECOMMENDATION: Check feature importance for TIER 3 features\n")
            f.write("If TIER 3 importance is low, may indicate redundancy with existing features.\n")
        else:
            f.write("No TIER 3 features detected. Model may not be retrained with TIER 3 yet.\n")
        f.write("\n")
        
        # Holdout validation summary
        f.write("3. HOLDOUT VALIDATION METRICS\n")
        f.write("-" * 80 + "\n")
        if validation_metrics:
            f.write("Key Metrics (Default Threshold 0.5):\n")
            for key in ['precision', 'recall', 'f1', 'mcc', 'roc_auc', 'pr_auc']:
                value = validation_metrics['default'][key]
                f.write(f"  {key.upper():<15}: {value:.4f}\n")
            
            conf_gap = validation_metrics['confidence']['confidence_gap']
            f.write(f"  CONFIDENCE GAP: {conf_gap:.4f}\n\n")
            
            f.write("Target Achievement:\n")
            targets = {
                'ROC AUC': (validation_metrics['default']['roc_auc'], 0.75),
                'MCC': (validation_metrics['default']['mcc'], 0.40),
                'Precision': (validation_metrics['default']['precision'], 0.60),
                'PR AUC': (validation_metrics['default']['pr_auc'], 0.65),
                'Confidence Gap': (conf_gap, 0.10)
            }
            
            passed = 0
            for name, (value, target) in targets.items():
                status = "PASS" if value >= target else "FAIL"
                if status == "PASS":
                    passed += 1
                f.write(f"  {name:<20}: {value:.4f} / {target:.2f} [{status}]\n")
            
            f.write(f"\nOverall: {passed}/{len(targets)} targets achieved\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n📄 Raport podsumowujący zapisany: {report_path}")
    print(f"\n✅ Analiza zakończona! Wszystkie wyniki w: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Post-Training Analysis for TIER 3 Model Enhancement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python post_training_analysis.py --ticker SOLUSDT --timeframe 15m --side long --helper_timeframes 1h 2h 4h
  python post_training_analysis.py --ticker BTCUSDT --timeframe 1h --side short
        """
    )
    
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol (e.g., SOLUSDT)')
    parser.add_argument('--timeframe', type=str, required=True, help='Timeframe (e.g., 15m, 1h, 4h)')
    parser.add_argument('--side', type=str, required=True, choices=['long', 'short'], help='Trading side')
    parser.add_argument('--helper_timeframes', nargs='*', default=None, help='Helper timeframes (e.g., 1h 2h 4h)')
    parser.add_argument('--importance_threshold', type=float, default=0.1, 
                       help='Feature importance threshold percentage (default: 0.1%%)')
    
    args = parser.parse_args()
    
    # Generate strategy ID
    strategy_id = _get_strategy_id(args.ticker, args.timeframe, args.helper_timeframes, args.side)
    
    print("="*80)
    print("🚀 POST-TRAINING ANALYSIS - TIER 3 Model Enhancement")
    print("="*80)
    print(f"Strategy: {strategy_id}")
    print(f"Importance threshold: {args.importance_threshold}%")
    print("="*80)
    
    # Load data
    print("\n📂 Loading model data...")
    data = load_model_data(strategy_id)
    
    if data is None:
        print("\n❌ Nie można kontynuować analizy bez wymaganych plików.")
        return
    
    print("✅ Wszystkie pliki załadowane pomyślnie!")
    
    # Analysis 1: Feature Importance
    feature_analysis = analyze_feature_importance(
        data['model'], 
        data['features'], 
        args.importance_threshold,
        strategy_id
    )
    
    # Analysis 2: Correlation Matrix
    correlation_analysis = analyze_correlation_matrix(
        data['model'],
        data['features'],
        strategy_id
    )
    
    # Analysis 3: Holdout Validation
    validation_metrics = analyze_holdout_validation(
        data['holdout_df'],
        args.side,
        strategy_id
    )
    
    # Generate summary report
    generate_summary_report(
        feature_analysis,
        correlation_analysis,
        validation_metrics,
        strategy_id
    )
    
    print("\n" + "="*80)
    print("🎉 ANALIZA ZAKOŃCZONA POMYŚLNIE!")
    print("="*80)


if __name__ == '__main__':
    main()
