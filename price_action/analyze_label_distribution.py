#!/usr/bin/env python3
"""
Skrypt do analizy rozkładu klas w wytrenowanych modelach.

Usage:
    python analyze_label_distribution.py
    python analyze_label_distribution.py --version v1.2
    python analyze_label_distribution.py --version v1.2 --detailed

Output:
    - Tabela z rozkładem klas dla wszystkich modeli
    - Statystyki (średnia, min, max)
    - Rekomendacje dla poprawek
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path


def analyze_model(model_dir: Path) -> dict:
    """Analyze single model's label distribution."""
    metadata_path = model_dir / "training_metadata.json"
    holdout_path = model_dir / "holdout_predictions.csv"

    if not metadata_path.exists() or not holdout_path.exists():
        return None

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load holdout predictions
    df = pd.read_csv(holdout_path)
    dist = df['y_true'].value_counts(normalize=True).sort_index()

    # Calculate metrics
    hold_pct = dist.get(0, 0) * 100
    action_pct = dist.get(1, 0) * 100

    barrier = metadata['best_label_params']['barrier_size']
    time_limit = metadata['best_label_params']['time_limit']

    return {
        'model': model_dir.name,
        'ticker': metadata.get('ticker', 'unknown'),
        'side': metadata.get('side', 'unknown'),
        'version': metadata.get('version', 'unknown'),
        'barrier%': barrier * 100,
        'time_limit': time_limit,
        'HOLD%': hold_pct,
        'ACTION%': action_pct,
        'samples': len(df),
        'f1_score': metadata.get('optimal_threshold', None),  # Proxy dla quality
        'n_features': metadata.get('n_features_selected', None)
    }


def generate_recommendations(results_df: pd.DataFrame) -> list:
    """Generate recommendations based on analysis."""
    recommendations = []

    avg_action = results_df['ACTION%'].mean()
    min_action = results_df['ACTION%'].min()
    max_action = results_df['ACTION%'].max()

    # Recommendation 1: Overall balance
    if avg_action < 35:
        recommendations.append(
            f"🔴 CRITICAL: Średni ACTION%={avg_action:.1f}% jest bardzo niski (cel: 50%).\n"
            f"   → Zastosuj POPRAWKĘ #11 (balance constraint + zwiększ time_limit/barrier)"
        )
    elif avg_action < 45:
        recommendations.append(
            f"🟡 WARNING: Średni ACTION%={avg_action:.1f}% jest niski (cel: 50%).\n"
            f"   → Rozważ zwiększenie time_limit range do (24, 72)"
        )
    else:
        recommendations.append(
            f"✅ OK: Średni ACTION%={avg_action:.1f}% jest w akceptowalnym zakresie (40-60%)"
        )

    # Recommendation 2: Consistency
    if max_action - min_action > 20:
        recommendations.append(
            f"🟡 WARNING: Duża wariancja ACTION% ({min_action:.1f}% - {max_action:.1f}%).\n"
            f"   → Balance constraint pomoże ustabilizować rozkład"
        )

    # Recommendation 3: Barrier size
    avg_barrier = results_df['barrier%'].mean()
    if avg_barrier < 1.8:
        recommendations.append(
            f"🟡 INFO: Średni barrier={avg_barrier:.2f}% jest bardzo niski.\n"
            f"   → Zwiększ dolną granicę barrier_size do 0.020 (2.0%)"
        )

    # Recommendation 4: Time limit
    avg_time = results_df['time_limit'].mean()
    if avg_time < 20:
        recommendations.append(
            f"🟡 INFO: Średni time_limit={avg_time:.0f} candles jest krótki.\n"
            f"   → Zwiększ dolną granicę time_limit do 24 candles"
        )

    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Analyze label distribution in trained models")
    parser.add_argument('--version', type=str, default=None,
                        help='Model version to analyze (e.g., v1.2). If not specified, analyzes all.')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed per-model breakdown')
    parser.add_argument('--export', type=str, default=None,
                        help='Export results to CSV file')
    args = parser.parse_args()

    # Find models directory
    models_dir = Path("models")
    if not models_dir.exists():
        print(f"❌ Error: models/ directory not found in {Path.cwd()}")
        return

    # Collect all models
    results = []
    for root, dirs, files in os.walk(models_dir):
        if 'training_metadata.json' in files and 'holdout_predictions.csv' in files:
            model_dir = Path(root)
            result = analyze_model(model_dir)
            if result:
                # Filter by version if specified
                if args.version is None or result['version'] == args.version:
                    results.append(result)

    if not results:
        if args.version:
            print(f"❌ No models found for version: {args.version}")
        else:
            print(f"❌ No models with holdout predictions found in models/")
        return

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Display results
    print("=" * 100)
    print("📊 ANALIZA ROZKŁADU KLAS W ETYKIETACH")
    print("=" * 100)
    print(f"\nZnaleziono {len(results_df)} modeli")
    if args.version:
        print(f"Filtrowanie: version={args.version}")
    print()

    if args.detailed:
        # Detailed view
        print(results_df.to_string(index=False))
    else:
        # Summary view
        summary_cols = ['ticker', 'side', 'barrier%', 'time_limit', 'HOLD%', 'ACTION%']
        print(results_df[summary_cols].to_string(index=False))

    # Statistics
    print("\n" + "=" * 100)
    print("📈 STATYSTYKI")
    print("=" * 100)
    print(f"\nROZKŁAD KLAS:")
    print(f"  Średni HOLD%:   {results_df['HOLD%'].mean():.1f}%  (cel: 50%)")
    print(f"  Średni ACTION%: {results_df['ACTION%'].mean():.1f}%  (cel: 50%)")
    print(f"  Min ACTION%:    {results_df['ACTION%'].min():.1f}%")
    print(f"  Max ACTION%:    {results_df['ACTION%'].max():.1f}%")
    print(f"  Std ACTION%:    {results_df['ACTION%'].std():.1f}%")

    print(f"\nPARAMETRY ETYKIETOWANIA:")
    print(f"  Średni barrier:     {results_df['barrier%'].mean():.2f}%")
    print(f"  Średni time_limit:  {results_df['time_limit'].mean():.0f} candles")
    print(f"  Zakres barrier:     {results_df['barrier%'].min():.2f}% - {results_df['barrier%'].max():.2f}%")
    print(f"  Zakres time_limit:  {results_df['time_limit'].min():.0f} - {results_df['time_limit'].max():.0f} candles")

    # Recommendations
    print("\n" + "=" * 100)
    print("💡 REKOMENDACJE")
    print("=" * 100)
    recommendations = generate_recommendations(results_df)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")

    # Export if requested
    if args.export:
        results_df.to_csv(args.export, index=False)
        print(f"\n💾 Results exported to: {args.export}")

    print("\n" + "=" * 100)
    print("📖 NEXT STEPS")
    print("=" * 100)
    print("\n1. Przeczytaj szczegółową diagnozę:")
    print("   cat LABEL_OPTIMIZATION_DIAGNOSIS.md")
    print("\n2. Zastosuj poprawki:")
    print("   cat LABEL_FIX_README.md")
    print("\n3. Retrain model:")
    print("   python main.py --train --side long --ticker SOLUSDT --timeframe 15m \\")
    print("     --helper-timeframes 1h 4h --version v1.3.test \\")
    print("     --label-trials 500 --model-trials 400")
    print("\n4. Re-run this script to verify:")
    print(f"   python analyze_label_distribution.py --version v1.3.test")
    print()


if __name__ == "__main__":
    main()