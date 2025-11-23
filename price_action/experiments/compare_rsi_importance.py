#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare RSI feature importance between profitable vs unprofitable trades.
"""

import pandas as pd
import numpy as np
import joblib
from collections import defaultdict

def load_model_and_features(symbol: str):
    """Load model and features for symbol."""
    model_map = {
        'SOLUSDT': 'v1.2.sol',
        'DOGEUSDT': 'v1.2.doge',
        'ETHUSDT': 'v1.2.eth',
        'LINKUSDT': 'v1.2.link',
        'HBARUSDT': 'v1.2.hbar',
        'NEARUSDT': 'v1.2.near',
        'WIFUSDT': 'v1.2.wif',
        '1000PEPEUSDT': 'v1.2.1000pepe',
        'BRETTUSDT': 'v1.2.brett',
        'ONDOUSDT': 'v1.2.ondo',
        'PONKEUSDT': 'v1.2.ponke',
        'SUIUSDT': 'v1.2.sui',
        'KASUSDT': 'v1.2.kas',
        'TRXUSDT': 'v1.2.trx',
        'XRPUSDT': 'v1.2.xrp'
    }

    helpers_map = {
        'SOLUSDT': ['1h', '4h', '12h', '1d'],
        'DOGEUSDT': ['1h', '4h', '6h', '12h', '1d'],
        'ETHUSDT': ['1h', '2h', '4h', '6h', '12h', '1d'],
        'LINKUSDT': ['1h', '2h', '4h', '6h', '12h', '1d'],
        'HBARUSDT': ['1h', '2h', '4h', '6h', '12h', '1d'],
        'NEARUSDT': ['1h', '2h', '4h', '6h', '12h', '1d'],
        'WIFUSDT': ['1h', '2h', '4h', '12h', '1d'],
        '1000PEPEUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'BRETTUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'ONDOUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'PONKEUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'SUIUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'KASUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'TRXUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'XRPUSDT': ['1h', '2h', '4h', '12h', '1d']
    }

    if symbol not in model_map:
        return None, None, None, None

    version = model_map[symbol]
    helpers = helpers_map.get(symbol, ['1h', '4h', '12h', '1d'])
    helpers_str = '_'.join(helpers)

    try:
        model_long_path = f"models/{version}/{symbol}_15m_plus_{helpers_str}_long/model.joblib"
        model_short_path = f"models/{version}/{symbol}_15m_plus_{helpers_str}_short/model.joblib"
        features_long_path = f"models/{version}/{symbol}_15m_plus_{helpers_str}_long/features.joblib"
        features_short_path = f"models/{version}/{symbol}_15m_plus_{helpers_str}_short/features.joblib"

        model_long = joblib.load(model_long_path)
        model_short = joblib.load(model_short_path)
        features_long = joblib.load(features_long_path)
        features_short = joblib.load(features_short_path)

        return model_long, model_short, features_long, features_short

    except Exception as e:
        return None, None, None, None

def analyze_feature_importance_by_outcome():
    """Analyze feature importance for profitable vs unprofitable trades."""

    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║   RSI FEATURE IMPORTANCE: PROFITABLE vs UNPROFITABLE TRADES               ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    # Load trades
    df = pd.read_csv('live_trades_nov1-11_2025.csv')
    df['createdTime'] = pd.to_datetime(df['createdTime'])

    # Separate profitable vs unprofitable
    profitable = df[df['closedPnl'] > 0]
    unprofitable = df[df['closedPnl'] <= 0]

    print(f"\n📊 DATASET:")
    print(f"   Total Trades: {len(df)}")
    print(f"   Profitable: {len(profitable)} (${profitable['closedPnl'].sum():.2f})")
    print(f"   Unprofitable: {len(unprofitable)} (${unprofitable['closedPnl'].sum():.2f})")

    # Get symbols
    symbols = df['symbol'].unique()

    # Collect feature importance for each group
    profitable_importance = defaultdict(list)
    unprofitable_importance = defaultdict(list)

    for symbol in symbols:
        # Get trades for this symbol
        symbol_profitable = profitable[profitable['symbol'] == symbol]
        symbol_unprofitable = unprofitable[unprofitable['symbol'] == symbol]

        if len(symbol_profitable) == 0 and len(symbol_unprofitable) == 0:
            continue

        # Load model
        model_long, model_short, features_long, features_short = load_model_and_features(symbol)

        if model_long is None:
            continue

        print(f"\n📊 {symbol}:")
        print(f"   Profitable: {len(symbol_profitable)} trades")
        print(f"   Unprofitable: {len(symbol_unprofitable)} trades")

        # Get feature importance for both models
        for model, features, model_name in [
            (model_long, features_long, "LONG"),
            (model_short, features_short, "SHORT")
        ]:
            if model is None or features is None:
                continue

            try:
                importances = model.feature_importances_

                # Create feature importance dict
                fi_dict = dict(zip(features, importances))

                # Filter for this model's side
                if model_name == "LONG":
                    side_profitable = symbol_profitable[symbol_profitable['side'] == 'Buy']
                    side_unprofitable = symbol_unprofitable[symbol_unprofitable['side'] == 'Buy']
                else:
                    side_profitable = symbol_profitable[symbol_profitable['side'] == 'Sell']
                    side_unprofitable = symbol_unprofitable[symbol_unprofitable['side'] == 'Sell']

                # Store importance for each feature
                for feature, importance in fi_dict.items():
                    if len(side_profitable) > 0:
                        profitable_importance[feature].append(importance)
                    if len(side_unprofitable) > 0:
                        unprofitable_importance[feature].append(importance)

            except Exception as e:
                print(f"   ⚠️  Error getting importance for {model_name}: {e}")

    # Aggregate importance across all trades
    print(f"\n{'='*80}")
    print(f"📊 FEATURE IMPORTANCE COMPARISON")
    print(f"{'='*80}")

    # Calculate average importance for each feature
    all_features = set(list(profitable_importance.keys()) + list(unprofitable_importance.keys()))

    feature_comparison = []

    for feature in all_features:
        prof_imp = profitable_importance.get(feature, [])
        unprof_imp = unprofitable_importance.get(feature, [])

        avg_prof = np.mean(prof_imp) if len(prof_imp) > 0 else 0
        avg_unprof = np.mean(unprof_imp) if len(unprof_imp) > 0 else 0

        # Calculate difference
        diff = avg_unprof - avg_prof
        diff_pct = (diff / avg_prof * 100) if avg_prof > 0 else 0

        feature_comparison.append({
            'feature': feature,
            'profitable_avg': avg_prof,
            'unprofitable_avg': avg_unprof,
            'difference': diff,
            'difference_pct': diff_pct,
            'is_rsi': 'rsi' in feature.lower()
        })

    df_comparison = pd.DataFrame(feature_comparison)

    # Filter for RSI features
    rsi_features = df_comparison[df_comparison['is_rsi']].copy()
    rsi_features = rsi_features.sort_values('difference', ascending=False)

    print(f"\n🔍 TOP 20 RSI FEATURES - HIGHER IN UNPROFITABLE TRADES:")
    print(f"{'Feature':<50} {'Profitable':<15} {'Unprofitable':<15} {'Diff %'}")
    print("="*100)

    for idx, row in rsi_features.head(20).iterrows():
        arrow = "⬆️" if row['difference'] > 0 else "⬇️"
        print(f"{arrow} {row['feature']:<48} {row['profitable_avg']:<14.1f} {row['unprofitable_avg']:<14.1f} {row['difference_pct']:+.1f}%")

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"📊 RSI FEATURES SUMMARY")
    print(f"{'='*80}")

    total_rsi = len(rsi_features)
    higher_in_unprof = len(rsi_features[rsi_features['difference'] > 0])
    lower_in_unprof = len(rsi_features[rsi_features['difference'] < 0])

    print(f"\n   Total RSI features: {total_rsi}")
    print(f"   Higher importance in UNPROFITABLE trades: {higher_in_unprof} ({higher_in_unprof/total_rsi*100:.1f}%)")
    print(f"   Lower importance in UNPROFITABLE trades: {lower_in_unprof} ({lower_in_unprof/total_rsi*100:.1f}%)")

    # Calculate total RSI importance
    total_prof_rsi = rsi_features['profitable_avg'].sum()
    total_unprof_rsi = rsi_features['unprofitable_avg'].sum()

    print(f"\n   Total RSI importance in PROFITABLE: {total_prof_rsi:.1f}")
    print(f"   Total RSI importance in UNPROFITABLE: {total_unprof_rsi:.1f}")
    print(f"   Difference: {total_unprof_rsi - total_prof_rsi:+.1f} ({(total_unprof_rsi - total_prof_rsi)/total_prof_rsi*100:+.1f}%)")

    # Non-RSI features for comparison
    print(f"\n{'='*80}")
    print(f"📊 NON-RSI FEATURES SUMMARY (for comparison)")
    print(f"{'='*80}")

    non_rsi = df_comparison[~df_comparison['is_rsi']].copy()
    non_rsi = non_rsi.sort_values('difference', ascending=False)

    print(f"\n🔍 TOP 10 NON-RSI FEATURES - HIGHER IN UNPROFITABLE:")
    print(f"{'Feature':<50} {'Profitable':<15} {'Unprofitable':<15} {'Diff %'}")
    print("="*100)

    for idx, row in non_rsi.head(10).iterrows():
        arrow = "⬆️" if row['difference'] > 0 else "⬇️"
        print(f"{arrow} {row['feature']:<48} {row['profitable_avg']:<14.1f} {row['unprofitable_avg']:<14.1f} {row['difference_pct']:+.1f}%")

    # Key insight
    print(f"\n{'='*80}")
    print(f"💡 KEY INSIGHT")
    print(f"{'='*80}")

    if higher_in_unprof > lower_in_unprof and total_unprof_rsi > total_prof_rsi * 1.2:
        print(f"\n⚠️  CRITICAL: RSI features are {(total_unprof_rsi/total_prof_rsi-1)*100:.1f}% MORE important in UNPROFITABLE trades!")
        print(f"   {higher_in_unprof}/{total_rsi} RSI features have higher weight in losers")
        print(f"\n🎯 RECOMMENDATION:")
        print(f"   Model is OVER-RELYING on RSI in unprofitable scenarios")
        print(f"   RSI may be giving FALSE SIGNALS that lead to losses")
        print(f"\n   Solutions:")
        print(f"   1. Add penalty for RSI features in loss function")
        print(f"   2. Remove weakest RSI features (high in losses, low in wins)")
        print(f"   3. Add trend confirmation to override RSI signals")

    elif total_unprof_rsi < total_prof_rsi * 0.8:
        print(f"\n✅ GOOD: RSI features are LESS important in unprofitable trades")
        print(f"   RSI is not the main problem")
    else:
        print(f"\n⚠️  MODERATE: RSI importance similar in both groups")
        print(f"   RSI contributes to both wins and losses")

    # Save results
    df_comparison.to_csv('feature_importance_comparison.csv', index=False)
    print(f"\n💾 Full comparison saved to: feature_importance_comparison.csv")

def main():
    """Main execution."""
    analyze_feature_importance_by_outcome()
    print(f"\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
