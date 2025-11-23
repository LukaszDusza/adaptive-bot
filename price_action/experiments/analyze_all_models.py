#!/usr/bin/env python3
"""
Comprehensive model analysis script.
Extracts metrics from all trained models and compares them.
"""
import json
import os
from pathlib import Path
import pandas as pd
from typing import Dict, Any

def load_model_data(model_dir: Path) -> Dict[str, Any]:
    """Load all relevant data for a model."""
    model_name = model_dir.name

    # Parse model name
    parts = model_name.split("_plus_")
    ticker_tf = parts[0]
    ticker = ticker_tf.split("_")[0]
    main_tf = ticker_tf.split("_")[1]
    helper_tfs = parts[1] if len(parts) > 1 else ""
    side = "long" if model_name.endswith("_long") else "short"

    data = {
        "ticker": ticker,
        "side": side,
        "main_tf": main_tf,
        "helper_tfs": helper_tfs.replace(f"_{side}", ""),
        "model_name": model_name,
    }

    # Load training metadata
    metadata_file = model_dir / "training_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            data["optimal_threshold_training"] = metadata.get("optimal_threshold", 0.5)
            data["training_auc"] = metadata.get("test_auc", 0)

    # Load analysis data
    analysis_file = model_dir / "analysis" / f"{model_name}_analysis.json"
    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)

            # Tier 1: Basic metrics (at training threshold)
            tier1 = analysis.get("tier_1_basic", {}).get("basic_metrics", {})
            data["accuracy"] = tier1.get("accuracy", 0)
            data["precision"] = tier1.get("precision", 0)
            data["recall"] = tier1.get("recall", 0)
            data["f1_score"] = tier1.get("f1_score", 0)
            data["roc_auc"] = tier1.get("roc_auc", 0)
            data["pr_auc"] = tier1.get("pr_auc", 0)

            # Confusion matrix
            cm = analysis.get("tier_1_basic", {}).get("confusion_matrix", {})
            data["true_positives"] = cm.get("true_positives", 0)
            data["false_positives"] = cm.get("false_positives", 0)
            data["true_negatives"] = cm.get("true_negatives", 0)
            data["false_negatives"] = cm.get("false_negatives", 0)

            # Trading metrics
            trade_metrics = analysis.get("tier_1_basic", {}).get("trading_metrics", {})
            data["signal_frequency_pct"] = trade_metrics.get("signal_frequency_percent", 0)
            data["total_signals"] = trade_metrics.get("total_signals", 0)

            # Tier 5: Find best threshold by F1
            tier5 = analysis.get("tier_5_thresholds", {}).get("threshold_optimization", [])
            if tier5:
                # Find threshold with best F1 score
                best_f1 = max(tier5, key=lambda x: x['f1_score'])
                data["best_threshold_f1"] = best_f1.get("threshold", 0.5)
                data["best_f1_score"] = best_f1.get("f1_score", 0)
                data["best_f1_precision"] = best_f1.get("precision", 0)
                data["best_f1_recall"] = best_f1.get("recall", 0)
                data["best_f1_signals"] = best_f1.get("signal_count", 0)

                # Find threshold with best precision (min 50% recall)
                high_recall_thresholds = [t for t in tier5 if t['recall'] >= 0.50]
                if high_recall_thresholds:
                    best_prec = max(high_recall_thresholds, key=lambda x: x['precision'])
                    data["best_prec_threshold"] = best_prec.get("threshold", 0.5)
                    data["best_prec_precision"] = best_prec.get("precision", 0)
                    data["best_prec_recall"] = best_prec.get("recall", 0)
                    data["best_prec_f1"] = best_prec.get("f1_score", 0)
                else:
                    data["best_prec_threshold"] = data["best_threshold_f1"]
                    data["best_prec_precision"] = data["best_f1_precision"]
                    data["best_prec_recall"] = data["best_f1_recall"]
                    data["best_prec_f1"] = data["best_f1_score"]

    return data

def main():
    models_base = Path("models")
    results = []

    print("\n" + "="*120)
    print("🔍 SCANNING ALL MODELS...")
    print("="*120)

    for version_dir in sorted(models_base.glob("v1.2.*")):
        for model_dir in sorted(version_dir.glob("*USDT*")):
            print(f"Processing: {model_dir.name}")
            data = load_model_data(model_dir)
            results.append(data)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(["ticker", "side"])

    # Save full data
    output_file = "models_analysis_full.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Full data saved to {output_file}")

    # Display key metrics
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.precision', 3)

    print("\n" + "="*120)
    print("📊 MODEL PERFORMANCE SUMMARY")
    print("="*120)

    summary_cols = [
        "ticker", "side",
        "roc_auc", "pr_auc",
        "optimal_threshold_training",
        "precision", "recall", "f1_score",
        "best_threshold_f1", "best_f1_score",
        "signal_frequency_pct"
    ]

    print(df[summary_cols].to_string(index=False))

    # Best models by AUC
    print("\n" + "="*120)
    print("🏆 TOP MODELS BY ROC AUC")
    print("="*120)
    top_models = df.nlargest(10, 'roc_auc')[['ticker', 'side', 'roc_auc', 'pr_auc', 'f1_score', 'precision', 'recall']]
    print(top_models.to_string(index=False))

    # Models in production
    print("\n" + "="*120)
    print("🚀 MODELS IN PRODUCTION (from docker-compose.yaml)")
    print("="*120)

    active_bots = [
        ("DOGEUSDT", "both"),
        ("SOLUSDT", "both"),
        ("ETHUSDT", "both"),
        ("LINKUSDT", "both"),
        ("HBARUSDT", "both"),
        ("SUIUSDT", "both"),
    ]

    for ticker, _ in active_bots:
        ticker_df = df[df['ticker'] == ticker].sort_values('side')
        if not ticker_df.empty:
            print(f"\n{ticker}:")
            for _, row in ticker_df.iterrows():
                print(f"  {row['side'].upper():5s} - AUC: {row['roc_auc']:.3f}, "
                      f"PR-AUC: {row['pr_auc']:.3f}, "
                      f"F1: {row['f1_score']:.3f}, "
                      f"P: {row['precision']:.3f}, "
                      f"R: {row['recall']:.3f}")
                print(f"         Training threshold: {row['optimal_threshold_training']:.3f}, "
                      f"Best F1 threshold: {row['best_threshold_f1']:.2f} "
                      f"(F1={row['best_f1_score']:.3f}, P={row['best_f1_precision']:.3f}, R={row['best_f1_recall']:.3f})")

    # Recommendations
    print("\n" + "="*120)
    print("💡 RECOMMENDATIONS FOR docker-compose.yaml")
    print("="*120)
    print("\nCurrent config uses:")
    print("  - prob-threshold: 0.58-0.60")
    print("  - min-proba-diff: 0.29-0.35")
    print("  - tp-pct: 0.02 (2%)")
    print("  - tsl-pct: 0.014 (1.4%)")
    print("\nModel analysis suggests:")
    for ticker, _ in active_bots:
        ticker_df = df[df['ticker'] == ticker].sort_values('side')
        if not ticker_df.empty:
            long_row = ticker_df[ticker_df['side'] == 'long'].iloc[0]
            short_row = ticker_df[ticker_df['side'] == 'short'].iloc[0]

            # Average recommended threshold
            avg_threshold = (long_row['best_threshold_f1'] + short_row['best_threshold_f1']) / 2

            print(f"\n{ticker}:")
            print(f"  Recommended prob-threshold: {avg_threshold:.2f}")
            print(f"    LONG  best F1 @ {long_row['best_threshold_f1']:.2f} "
                  f"(F1={long_row['best_f1_score']:.3f}, P={long_row['best_f1_precision']:.3f}, R={long_row['best_f1_recall']:.3f})")
            print(f"    SHORT best F1 @ {short_row['best_threshold_f1']:.2f} "
                  f"(F1={short_row['best_f1_score']:.3f}, P={short_row['best_f1_precision']:.3f}, R={short_row['best_f1_recall']:.3f})")

    print("\n" + "="*120)
    print("⚠️  NOTE: These are ML-only thresholds. For optimal live trading parameters")
    print("   (including tp-pct, tsl-pct, min-proba-diff), run Optuna optimization:")
    print("   python optuna_optimizer.py --ticker <TICKER> --timeframe 15m --helper-timeframes <TFS> --trials 100")
    print("="*120)

if __name__ == "__main__":
    main()
