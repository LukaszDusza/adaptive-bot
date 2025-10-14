import argparse
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    cohen_kappa_score, roc_curve, precision_recall_curve, fbeta_score
)
from sklearn.calibration import calibration_curve
import json
from datetime import datetime


def _get_strategy_id(ticker, timeframe, helper_timeframes, side: str):
    helpers_str = '_plus_' + '_'.join(helper_timeframes) if helper_timeframes else ""
    return f"{ticker}_{timeframe.replace(' ', '')}{helpers_str}_{side}"


def export_analysis_to_json(df, model, strategy_id, args, plots_dir):
    """
    Eksportuje wszystkie statystyki do JSON dla dalszej analizy.
    """

    signal_name = 'BUY' if args.side == 'long' else 'SELL'
    target_names = ['HOLD (0)', f'{signal_name} (1)']
    proba_col = f'proba_{target_names[1]}'

    cm = confusion_matrix(df['y_true'], df['y_pred_default'])
    tn, fp, fn, tp = cm.ravel()

    # TIER 1: Metryki bazowe
    tier1_metrics = {
        "basic_metrics": {
            "accuracy": float((df['y_true'] == df['y_pred_default']).mean()),
            "precision": float(precision_score(df['y_true'], df['y_pred_default'], zero_division=0)),
            "recall": float(recall_score(df['y_true'], df['y_pred_default'], zero_division=0)),
            "f1_score": float(f1_score(df['y_true'], df['y_pred_default'], zero_division=0)),
            "f2_score": float(fbeta_score(df['y_true'], df['y_pred_default'], beta=2.0, zero_division=0)),
            "roc_auc": float(roc_auc_score(df['y_true'], df[proba_col])),
            "pr_auc": float(average_precision_score(df['y_true'], df[proba_col])),
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }

    # TIER 2: Metryki zaawansowane
    specificity = float(tn / (tn + fp) if (tn + fp) > 0 else 0)
    npv = float(tn / (tn + fn) if (tn + fn) > 0 else 0)
    fpr = float(fp / (fp + tn) if (fp + tn) > 0 else 0)
    fnr = float(fn / (fn + tp) if (fn + tp) > 0 else 0)
    sensitivity = float(tp / (tp + fn) if (tp + fn) > 0 else 0)
    mcc = float(matthews_corrcoef(df['y_true'], df['y_pred_default']))
    kappa = float(cohen_kappa_score(df['y_true'], df['y_pred_default']))

    tier2_metrics = {
        "advanced_metrics": {
            "sensitivity_recall": sensitivity,
            "specificity_tnr": specificity,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "negative_predictive_value": npv,
            "matthews_correlation_coefficient": mcc,
            "cohen_kappa": kappa,
        },
        "trading_metrics": {
            # POPRAWKA: Zmieniono "win_rate_percent" na "classification_accuracy_percent"
            "classification_accuracy_percent": float((df['y_true'] == df['y_pred_default']).mean() * 100),
            "signal_frequency_percent": float(df['y_pred_default'].mean() * 100),
            "total_predictions": int(len(df)),
            "total_signals": int(df['y_pred_default'].sum()),
        }
    }

    # TIER 3: Analiza prawdopodobieństw
    tp_proba = df[(df['y_true'] == 1) & (df['y_pred_default'] == 1)][proba_col].mean()
    fp_proba = df[(df['y_true'] == 0) & (df['y_pred_default'] == 1)][proba_col].mean()
    tn_proba = df[(df['y_true'] == 0) & (df['y_pred_default'] == 0)][proba_col].mean()
    fn_proba = df[(df['y_true'] == 1) & (df['y_pred_default'] == 0)][proba_col].mean()

    tier3_metrics = {
        "probability_analysis": {
            "mean_proba_true_positives": float(tp_proba) if not np.isnan(tp_proba) else None,
            "mean_proba_false_positives": float(fp_proba) if not np.isnan(fp_proba) else None,
            "mean_proba_true_negatives": float(tn_proba) if not np.isnan(tn_proba) else None,
            "mean_proba_false_negatives": float(fn_proba) if not np.isnan(fn_proba) else None,
            "proba_separation": {
                "tp_minus_fp": float(tp_proba - fp_proba) if not (np.isnan(tp_proba) or np.isnan(fp_proba)) else None,
                "tn_minus_fn": float(tn_proba - fn_proba) if not (np.isnan(tn_proba) or np.isnan(fn_proba)) else None,
            }
        }
    }

    # TIER 4: Analiza pewności modelu
    df['correct'] = df['y_true'] == df['y_pred_default']
    df['confidence'] = df[proba_col].apply(lambda x: max(x, 1 - x))

    correct_confidence = df[df['correct']]['confidence'].mean()
    incorrect_confidence = df[~df['correct']]['confidence'].mean()

    tier4_metrics = {
        "model_confidence": {
            "mean_confidence_correct_predictions": float(correct_confidence) if not np.isnan(
                correct_confidence) else None,
            "mean_confidence_incorrect_predictions": float(incorrect_confidence) if not np.isnan(
                incorrect_confidence) else None,
            "confidence_gap": float(correct_confidence - incorrect_confidence) if not (
                        np.isnan(correct_confidence) or np.isnan(incorrect_confidence)) else None,
        }
    }

    # TIER 5: Analiza progów
    thresholds = [i / 100.0 for i in range(30, 96, 5)]
    threshold_analysis = []
    for threshold in thresholds:
        y_pred = (df[proba_col] > threshold).astype(int)
        prec = precision_score(df['y_true'], y_pred, zero_division=0)
        rec = recall_score(df['y_true'], y_pred, zero_division=0)
        f1 = f1_score(df['y_true'], y_pred, zero_division=0)
        threshold_analysis.append({
            "threshold": float(threshold),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "signal_count": int(y_pred.sum())
        })

    tier5_metrics = {
        "threshold_optimization": threshold_analysis
    }

    # TIER 6: Rolling performance (ostatnie 100, 50, 20)
    windows = [100, 50, 20]
    tier6_metrics = {
        "rolling_performance": {}
    }

    for window in windows:
        if len(df) >= window:
            rolling_acc = (df['y_true'] == df['y_pred_default']).rolling(window).mean()
            recent_acc = rolling_acc.iloc[-1]
            tier6_metrics["rolling_performance"][f"window_{window}"] = {
                "recent_accuracy": float(recent_acc) if not np.isnan(recent_acc) else None,
                "mean_accuracy": float(rolling_acc.mean()) if not np.isnan(rolling_acc.mean()) else None,
                "std_accuracy": float(rolling_acc.std()) if not np.isnan(rolling_acc.std()) else None,
            }

    # TIER 7: Feature importance (top 30)
    tier7_metrics = {"feature_importance": None}
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': model.feature_name_,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        top_features = feature_importance.head(30)
        tier7_metrics["feature_importance"] = {
            "top_30_features": [
                {
                    "feature": row['feature'],
                    "importance": float(row['importance']),
                    "cumulative_pct": float(feature_importance.iloc[:idx + 1]['importance'].sum() / feature_importance[
                        'importance'].sum() * 100)
                }
                for idx, (_, row) in enumerate(top_features.iterrows())
            ],
            "total_features": int(len(model.feature_name_))
        }

    # Combine all tiers
    full_analysis = {
        "metadata": {
            "strategy_id": strategy_id,
            "side": args.side.upper(),
            "ticker": args.ticker,
            "timeframe": args.timeframe,
            "helper_timeframes": args.helper_timeframes if args.helper_timeframes else [],
            "timestamp": datetime.now().isoformat(),
            "test_set_size": int(len(df)),
        },
        "tier_1_basic": tier1_metrics,
        "tier_2_advanced": tier2_metrics,
        "tier_3_probability": tier3_metrics,
        "tier_4_confidence": tier4_metrics,
        "tier_5_thresholds": tier5_metrics,
        "tier_6_rolling": tier6_metrics,
        "tier_7_features": tier7_metrics,
    }

    # Save to JSON
    json_path = os.path.join(plots_dir, f"{strategy_id}_analysis.json")
    with open(json_path, 'w') as f:
        json.dump(full_analysis, f, indent=2)

    print(f"\n{'=' * 60}")
    print("✓ Pełna analiza wyeksportowana do JSON:")
    print(f"  {json_path}")
    print(f"{'=' * 60}\n")

    return full_analysis


def run_analysis_with_args(args, output_dir=None):
    """
    Uruchamia analizę z obiektu argumentów przekazanego z main.py.

    Args:
        args: Obiekt argumentów z atrybutami ticker, timeframe, helper_timeframes, side
        output_dir: Opcjonalny katalog wyjściowy dla wyników analizy. Jeśli None, używa domyślnego 'results/plots'
    """
    strategy_id = _get_strategy_id(args.ticker, args.timeframe, args.helper_timeframes, args.side)

    results_dir = "results"
    models_dir = "models"

    # Use custom output directory if provided, otherwise use default
    if output_dir:
        plots_dir = output_dir
    else:
        plots_dir = os.path.join(results_dir, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    results_path = os.path.join(results_dir, f"{strategy_id}_holdout_predictions.csv")
    model_path = os.path.join(models_dir, f"{strategy_id}_model.joblib")

    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append(f"Model: {model_path}")
    if not os.path.exists(results_path):
        missing_files.append(f"Wyniki holdout: {results_path}")

    if missing_files:
        print(f"Błąd: Nie znaleziono następujących plików dla strategii '{strategy_id}':")
        for missing_file in missing_files:
            print(f"  - {missing_file}")
        print("Upewnij się, że model został wytrenowany.")
        return

    df = pd.read_csv(results_path, index_col='timestamp', parse_dates=True)
    model = joblib.load(model_path)

    signal_name = 'BUY' if args.side == 'long' else 'SELL'
    target_names = ['HOLD (0)', f'{signal_name} (1)']
    proba_col = f'proba_{target_names[1]}'

    print("--- Raport Klasyfikacji (domyślny próg 0.5) ---")
    print(classification_report(df['y_true'], df['y_pred_default'], target_names=target_names))

    cm = confusion_matrix(df['y_true'], df['y_pred_default'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Macierz Pomyłek dla modelu {args.side.upper()} (próg 0.5)')
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Przewidywana etykieta')
    plt.savefig(os.path.join(plots_dir, f"{strategy_id}_confusion_matrix_thresh_0.5.png"))
    plt.close()
    print(f"Zapisano macierz pomyłek w katalogu '{plots_dir}'.")

    # === DODATKOWE METRYKI WYDAJNOŚCI ===
    print("\n" + "=" * 60)
    print("--- Dodatkowe Metryki Wydajności ---")
    print("=" * 60)

    f1 = f1_score(df['y_true'], df['y_pred_default'])
    mcc = matthews_corrcoef(df['y_true'], df['y_pred_default'])
    roc_auc = roc_auc_score(df['y_true'], df[proba_col])
    pr_auc = average_precision_score(df['y_true'], df[proba_col])
    kappa = cohen_kappa_score(df['y_true'], df['y_pred_default'])

    print(f"F1-Score: {f1:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    # === METRYKI Z CONFUSION MATRIX ===
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n--- Dodatkowe Metryki z Macierzy Pomyłek ---")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity (TNR): {specificity:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"Negative Predictive Value (NPV): {npv:.4f}")

    # === CLASSIFICATION ACCURACY I SIGNAL FREQUENCY ===
    # POPRAWKA: Zmieniono mylący komunikat "Win Rate" na "Classification Accuracy"
    classification_accuracy = (df['y_true'] == df['y_pred_default']).mean()
    signal_freq = df['y_pred_default'].mean()

    print(f"\n--- Statystyki Klasyfikacji ---")
    print(f"⚠️  UWAGA: To NIE jest win rate tradingowy!")
    print(f"Classification Accuracy: {classification_accuracy:.2%}")
    print(f"  └─ % poprawnie sklasyfikowanych próbek (TP+TN)/(TP+TN+FP+FN)")
    print(f"Signal Frequency (% {signal_name}): {signal_freq:.2%}")
    print(f"  └─ % sygnałów {signal_name} wygenerowanych przez model")

    # === ANALIZA PRAWDOPODOBIEŃSTW ===
    print(f"\n--- Analiza Prawdopodobieństw dla Różnych Typów Predykcji ---")
    tp_proba = df[(df['y_true'] == 1) & (df['y_pred_default'] == 1)][proba_col].mean()
    fp_proba = df[(df['y_true'] == 0) & (df['y_pred_default'] == 1)][proba_col].mean()
    tn_proba = df[(df['y_true'] == 0) & (df['y_pred_default'] == 0)][proba_col].mean()
    fn_proba = df[(df['y_true'] == 1) & (df['y_pred_default'] == 0)][proba_col].mean()

    print(f"Średnie prawdopodobieństwo {signal_name}:")
    print(f"  True Positives (TP): {tp_proba:.4f}")
    print(f"  False Positives (FP): {fp_proba:.4f}")
    print(f"  True Negatives (TN): {tn_proba:.4f}")
    print(f"  False Negatives (FN): {fn_proba:.4f}")

    # === ANALIZA PEWNOŚCI MODELU ===
    df['correct'] = df['y_true'] == df['y_pred_default']
    df['confidence'] = df[proba_col].apply(lambda x: max(x, 1 - x))

    correct_confidence = df[df['correct']]['confidence'].mean()
    incorrect_confidence = df[~df['correct']]['confidence'].mean()

    print(f"\n--- Pewność Modelu ---")
    print(f"Średnia pewność dla poprawnych predykcji: {correct_confidence:.4f}")
    print(f"Średnia pewność dla błędnych predykcji: {incorrect_confidence:.4f}")

    # === ROZKŁAD PRAWDOPODOBIEŃSTW ===
    plt.figure(figsize=(10, 6))
    plt.hist([df[df['y_true'] == 0][proba_col],
              df[df['y_true'] == 1][proba_col]],
             bins=20, label=['HOLD (y=0)', f'{signal_name} (y=1)'], alpha=0.7, edgecolor='black')
    plt.xlabel(f'Przewidywane Prawdopodobieństwo {signal_name}')
    plt.ylabel('Częstotliwość')
    plt.legend()
    plt.title(f'Rozkład Prawdopodobieństw dla Różnych Klas - {args.side.upper()}')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f"{strategy_id}_probability_distribution.png"))
    plt.close()
    print(f"\nZapisano wykres rozkładu prawdopodobieństw w katalogu '{plots_dir}'.")

    # === ROC CURVE ===
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(df['y_true'], df[proba_col])

    plt.figure(figsize=(8, 8))
    plt.plot(fpr_roc, tpr_roc, label=f'ROC curve (AUC = {roc_auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Krzywa ROC dla modelu {args.side.upper()}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f"{strategy_id}_roc_curve.png"))
    plt.close()
    print(f"Zapisano krzywą ROC w katalogu '{plots_dir}'.")

    # === PRECISION-RECALL CURVE ===
    precision_pr, recall_pr, thresholds_pr = precision_recall_curve(df['y_true'], df[proba_col])

    plt.figure(figsize=(8, 8))
    plt.plot(recall_pr, precision_pr, label=f'PR curve (AUC = {pr_auc:.4f})', linewidth=2)
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title(f'Krzywa Precision-Recall dla modelu {args.side.upper()}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f"{strategy_id}_precision_recall_curve.png"))
    plt.close()
    print(f"Zapisano krzywą Precision-Recall w katalogu '{plots_dir}'.")

    # === CALIBRATION CURVE ===
    prob_true, prob_pred = calibration_curve(df['y_true'], df[proba_col], n_bins=10)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=1)
    plt.plot(prob_pred, prob_true, marker='o', label='Model', linewidth=2, markersize=8)
    plt.xlabel('Przewidywane Prawdopodobieństwo')
    plt.ylabel('Rzeczywista Częstotliwość')
    plt.title(f'Krzywa Kalibracji dla modelu {args.side.upper()}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, f"{strategy_id}_calibration_curve.png"))
    plt.close()
    print(f"Zapisano krzywą kalibracji w katalogu '{plots_dir}'.")

    # === ROLLING ACCURACY ===
    window = 50
    df['rolling_accuracy'] = (df['y_true'] == df['y_pred_default']).rolling(window).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['rolling_accuracy'], linewidth=1.5)
    plt.xlabel('Czas')
    plt.ylabel(f'Rolling Accuracy (okno={window})')
    plt.title(f'Wydajność Modelu w Czasie - {args.side.upper()}')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=classification_accuracy, color='r', linestyle='--', label=f'Średnia Accuracy: {classification_accuracy:.2%}')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{strategy_id}_rolling_accuracy.png"))
    plt.close()
    print(f"Zapisano wykres rolling accuracy w katalogu '{plots_dir}'.")

    thresholds = [i / 100.0 for i in range(40, 81, 5)]
    metrics = []
    for threshold in thresholds:
        y_pred = (df[proba_col] > threshold).astype(int)
        precision = precision_score(df['y_true'], y_pred, zero_division=0)
        recall = recall_score(df['y_true'], y_pred, zero_division=0)
        metrics.append({'threshold': threshold, f'precision_{signal_name.lower()}': precision,
                        f'recall_{signal_name.lower()}': recall})

    metrics_df = pd.DataFrame(metrics).set_index('threshold')
    print("\n--- Analiza progów prawdopodobieństwa ---")
    print("Tabela metryk dla różnych progów:")
    print(metrics_df)

    plt.figure(figsize=(10, 6))
    metrics_df[f'precision_{signal_name.lower()}'].plot(label=f'Precision {signal_name}', marker='o')
    metrics_df[f'recall_{signal_name.lower()}'].plot(label=f'Recall {signal_name}', marker='x')
    plt.title(f'Analiza Precyzji i Czułości dla modelu {args.side.upper()}')
    plt.xlabel('Próg prawdopodobieństwa')
    plt.ylabel('Wynik')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{strategy_id}_threshold_analysis.png"))
    plt.close()
    print(f"Zapisano wykres analizy progów w katalogu '{plots_dir}'.")

    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({'feature': model.feature_name_, 'importance': model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(30)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title(f'Ważność Cech dla modelu {args.side.upper()}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{strategy_id}_feature_importance.png"))
        plt.close()
        print(f"Zapisano wykres ważności cech w katalogu '{plots_dir}'.")

    # EXPORT DO JSON - NOWE
    stats_json = export_analysis_to_json(df, model, strategy_id, args, plots_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analiza wyników modelu predykcyjnego.")
    parser.add_argument('--ticker', type=str, default="SOLUSDT")
    parser.add_argument('--timeframe', type=str, default="1h")
    parser.add_argument('--helper-timeframes', nargs='*', default=["4h", "12h", "1D"])
    parser.add_argument('--side', type=str, choices=['long', 'short'], required=True,
                        help="Określ, który model analizować: 'long' czy 'short'.")
    args = parser.parse_args()

    run_analysis_with_args(args)
