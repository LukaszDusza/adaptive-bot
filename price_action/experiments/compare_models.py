#!/usr/bin/env python3
import json
import os
from pathlib import Path
import pandas as pd

def extract_model_metrics():
    models_base = Path("models")
    results = []

    for version_dir in models_base.glob("v1.2.*"):
        for model_dir in version_dir.glob("*USDT*"):
            model_name = model_dir.name

            # Get ticker, timeframes, side
            parts = model_name.split("_plus_")
            ticker_tf = parts[0]  # e.g., "SOLUSDT_15m"
            ticker = ticker_tf.split("_")[0]
            main_tf = ticker_tf.split("_")[1]

            helper_tfs = parts[1] if len(parts) > 1 else ""
            side = "long" if model_name.endswith("_long") else "short"

            # Read training metadata
            metadata_file = model_dir / "training_metadata.json"
            analysis_file = model_dir / "analysis" / f"{model_name}_analysis.json"

            data = {
                "ticker": ticker,
                "side": side,
                "main_tf": main_tf,
                "helper_tfs": helper_tfs.replace(f"_{side}", ""),
                "model_path": str(model_dir),
            }

            # Get training metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    data["optimal_threshold"] = metadata.get("optimal_threshold", "N/A")
                    data["training_auc"] = metadata.get("test_auc", "N/A")
                    data["training_recall"] = metadata.get("test_recall", "N/A")
                    data["training_precision"] = metadata.get("test_precision", "N/A")

            # Get analysis data
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    analysis = json.load(f)
                    data["holdout_auc"] = analysis.get("overall_metrics", {}).get("auc_roc", "N/A")
                    data["holdout_recall"] = analysis.get("overall_metrics", {}).get("recall", "N/A")
                    data["holdout_precision"] = analysis.get("overall_metrics", {}).get("precision", "N/A")
                    data["holdout_f1"] = analysis.get("overall_metrics", {}).get("f1_score", "N/A")
                    data["positive_samples"] = analysis.get("class_distribution", {}).get("positive_samples", "N/A")
                    data["negative_samples"] = analysis.get("class_distribution", {}).get("negative_samples", "N/A")

                    # Threshold recommendations
                    thresh_rec = analysis.get("threshold_recommendations", {})
                    data["recommended_threshold"] = thresh_rec.get("recommended_threshold", "N/A")
                    data["expected_precision"] = thresh_rec.get("expected_precision_at_recommended", "N/A")
                    data["expected_recall"] = thresh_rec.get("expected_recall_at_recommended", "N/A")

            results.append(data)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(["ticker", "side"])

    return df

if __name__ == "__main__":
    df = extract_model_metrics()

    print("\n" + "="*120)
    print("MODEL COMPARISON - ALL METRICS")
    print("="*120)

    # Format and display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 30)

    print(df.to_string(index=False))

    # Save to CSV
    output_file = "models_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to {output_file}")

    # Summary statistics
    print("\n" + "="*120)
    print("SUMMARY BY TICKER")
    print("="*120)

    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker]
        print(f"\n{ticker}:")
        print(f"  LONG - AUC: {ticker_df[ticker_df['side']=='long']['holdout_auc'].values[0]:.3f}, "
              f"Recall: {ticker_df[ticker_df['side']=='long']['holdout_recall'].values[0]:.3f}, "
              f"Precision: {ticker_df[ticker_df['side']=='long']['holdout_precision'].values[0]:.3f}")
        print(f"  SHORT - AUC: {ticker_df[ticker_df['side']=='short']['holdout_auc'].values[0]:.3f}, "
              f"Recall: {ticker_df[ticker_df['side']=='short']['holdout_recall'].values[0]:.3f}, "
              f"Precision: {ticker_df[ticker_df['side']=='short']['holdout_precision'].values[0]:.3f}")
