#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEEP FEATURE ANALYSIS - Find root cause of losses
==================================================

For each trade from Nov 1-11:
1. Fetch OHLCV data at entry time
2. Calculate ALL features (using data_preparer_pa)
3. Record exact feature values at entry
4. Compare features between profitable vs unprofitable trades
5. Find which features gave FALSE SIGNALS for SHORTs

Goal: Identify WHY bot opened SHORTs during price increases
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from collections import defaultdict
from bybit_adapter import BybitAdapter
from data_preparer_pa import fetch_and_prepare_data
from dotenv import load_dotenv
import os
from tqdm import tqdm
import json

# Load API keys
load_dotenv('.env_luk')


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
        return None, None, None, None, None

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

        return model_long, model_short, features_long, features_short, helpers

    except Exception as e:
        print(f"Error loading model for {symbol}: {e}")
        return None, None, None, None, None


def get_features_at_entry(symbol, entry_time_str, adapter, helpers):
    """
    Get feature values at the moment of trade entry.

    Args:
        symbol: Trading pair
        entry_time_str: Entry timestamp string (e.g., '2025-11-07 10:30:00')
        adapter: BybitAdapter instance
        helpers: Helper timeframes for this symbol

    Returns:
        DataFrame with features, or None if failed
    """
    try:
        # Parse entry time
        entry_time = pd.to_datetime(entry_time_str)

        # Fetch enough historical data (500 candles should be sufficient)
        # We need to fetch data UP TO entry_time (not after)
        end_date = entry_time.strftime('%Y-%m-%d')

        # Use data_preparer_pa to calculate features
        df = fetch_and_prepare_data(
            ticker=symbol,
            timeframe='15m',
            limit=500,
            helper_timeframes=helpers,
            side='backtest',  # Doesn't matter for feature calculation
            date_from=end_date,
            version='v1.2.sol',
            skip_slow_features=True  # Fast mode for live-like calculation
        )

        if df.empty:
            return None

        # Find the candle that matches entry time (or closest before)
        df.index = pd.to_datetime(df.index)

        # Get candle at or before entry time
        mask = df.index <= entry_time
        if not mask.any():
            return None

        entry_candle = df[mask].iloc[-1]

        return entry_candle

    except Exception as e:
        print(f"   ⚠️  Error getting features: {e}")
        return None


def analyze_deep_features():
    """
    Deep feature analysis for all trades Nov 1-11.
    """
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║         DEEP FEATURE ANALYSIS - ROOT CAUSE OF LOSSES                      ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    # Load trades
    df_trades = pd.read_csv('live_trades_nov1-11_2025.csv')
    df_trades['createdTime'] = pd.to_datetime(df_trades['createdTime'])

    print(f"\n📊 DATASET:")
    print(f"   Total Trades: {len(df_trades)}")
    print(f"   Date Range: {df_trades['createdTime'].min()} → {df_trades['createdTime'].max()}")
    print(f"   Total P&L: ${df_trades['closedPnl'].sum():.2f}")

    # Separate by outcome
    profitable = df_trades[df_trades['closedPnl'] > 0]
    unprofitable = df_trades[df_trades['closedPnl'] <= 0]

    print(f"\n   Profitable: {len(profitable)} trades (${profitable['closedPnl'].sum():.2f})")
    print(f"   Unprofitable: {len(unprofitable)} trades (${unprofitable['closedPnl'].sum():.2f})")

    # Initialize API adapter
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    base_url = os.getenv("BYBIT_BASE_URL")
    adapter = BybitAdapter(api_key=api_key, api_secret=api_secret, base_url=base_url)

    # Storage for feature analysis
    feature_data = {
        'profitable': defaultdict(list),
        'unprofitable': defaultdict(list)
    }

    trade_details = []

    print(f"\n{'='*80}")
    print(f"🔍 ANALYZING FEATURE VALUES AT ENTRY (this will take a few minutes...)")
    print(f"{'='*80}\n")

    # Analyze sample of trades (to save time - 50 trades should be enough)
    sample_size = min(50, len(df_trades))
    df_sample = df_trades.sample(n=sample_size, random_state=42).sort_values('createdTime')

    for idx, trade in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Processing trades"):
        symbol = trade['symbol']
        side = trade['side']
        entry_price = float(trade['avgEntryPrice'])
        pnl = float(trade['closedPnl'])
        entry_time_str = trade['createdTime'].strftime('%Y-%m-%d %H:%M:%S')

        # Load model for this symbol
        model_long, model_short, features_long, features_short, helpers = load_model_and_features(symbol)

        if model_long is None:
            continue

        # Get features at entry
        features_at_entry = get_features_at_entry(symbol, entry_time_str, adapter, helpers)

        if features_at_entry is None:
            continue

        # Determine which model was used
        if side == 'Buy':
            model = model_long
            model_features = features_long
        else:
            model = model_short
            model_features = features_short

        # Extract feature importance
        feature_importance = dict(zip(model_features, model.feature_importances_))

        # Store feature values
        outcome = 'profitable' if pnl > 0 else 'unprofitable'

        trade_detail = {
            'symbol': symbol,
            'side': side,
            'entry_time': entry_time_str,
            'entry_price': entry_price,
            'pnl': pnl,
            'outcome': outcome,
            'features': {}
        }

        # Store all feature values
        for feature in model_features:
            if feature in features_at_entry.index:
                value = features_at_entry[feature]
                feature_data[outcome][feature].append(value)
                trade_detail['features'][feature] = float(value) if not pd.isna(value) else None

        trade_details.append(trade_detail)

    print(f"\n✅ Feature extraction complete!")
    print(f"   Analyzed {len(trade_details)} trades")

    # ========================================================================
    # ANALYSIS: Compare feature values between profitable vs unprofitable
    # ========================================================================

    print(f"\n{'='*80}")
    print(f"📊 FEATURE VALUE COMPARISON: Profitable vs Unprofitable")
    print(f"{'='*80}\n")

    # Calculate average feature values for each outcome
    comparison_results = []

    all_features = set(list(feature_data['profitable'].keys()) + list(feature_data['unprofitable'].keys()))

    for feature in all_features:
        prof_values = feature_data['profitable'].get(feature, [])
        unprof_values = feature_data['unprofitable'].get(feature, [])

        if not prof_values or not unprof_values:
            continue

        # Remove NaN values
        prof_values = [v for v in prof_values if not pd.isna(v)]
        unprof_values = [v for v in unprof_values if not pd.isna(v)]

        if not prof_values or not unprof_values:
            continue

        avg_prof = np.mean(prof_values)
        avg_unprof = np.mean(unprof_values)

        # Calculate difference
        diff = avg_unprof - avg_prof
        diff_pct = (diff / avg_prof * 100) if avg_prof != 0 else 0

        # Calculate standard deviation
        std_prof = np.std(prof_values)
        std_unprof = np.std(unprof_values)

        comparison_results.append({
            'feature': feature,
            'avg_profitable': avg_prof,
            'avg_unprofitable': avg_unprof,
            'difference': diff,
            'difference_pct': diff_pct,
            'std_profitable': std_prof,
            'std_unprofitable': std_unprof,
            'n_profitable': len(prof_values),
            'n_unprofitable': len(unprof_values)
        })

    df_comparison = pd.DataFrame(comparison_results)

    # Sort by absolute difference percentage
    df_comparison['abs_diff_pct'] = df_comparison['difference_pct'].abs()
    df_comparison = df_comparison.sort_values('abs_diff_pct', ascending=False)

    # Show top features with biggest differences
    print(f"🔍 TOP 20 FEATURES WITH BIGGEST VALUE DIFFERENCES:\n")
    print(f"{'Feature':<50} {'Profitable':<15} {'Unprofitable':<15} {'Diff %':<12} Status")
    print("="*110)

    for idx, row in df_comparison.head(20).iterrows():
        arrow = "⬆️" if row['difference'] > 0 else "⬇️"
        status = "HIGHER in losses" if row['difference'] > 0 else "LOWER in losses"

        print(f"{arrow} {row['feature']:<48} {row['avg_profitable']:<14.4f} {row['avg_unprofitable']:<14.4f} {row['difference_pct']:+11.1f}% {status}")

    # ========================================================================
    # SPECIAL ANALYSIS: SHORTs during uptrend
    # ========================================================================

    print(f"\n{'='*80}")
    print(f"🔍 SPECIAL ANALYSIS: WHY SHORTs DURING UPTREND?")
    print(f"{'='*80}\n")

    # Filter for SHORT trades
    short_trades = [t for t in trade_details if t['side'] == 'Sell']

    if short_trades:
        print(f"Total SHORTs analyzed: {len(short_trades)}")

        # Check RSI, momentum, trend indicators for SHORTs
        short_features = ['rsi_14', 'sma_20_slope', 'sma_50_slope', 'above_sma_20', 'above_sma_50',
                         'momentum_regime', 'rsi_slope_2h', 'rsi_slope_4h', 'price_vs_sma200']

        print(f"\nKey Trend/Momentum Indicators in SHORTs:")
        print(f"{'Feature':<30} {'Avg Value':<15} {'Interpretation'}")
        print("="*70)

        for feature in short_features:
            values = []
            for trade in short_trades:
                if feature in trade['features'] and trade['features'][feature] is not None:
                    values.append(trade['features'][feature])

            if values:
                avg_value = np.mean(values)

                # Interpret value
                if 'rsi' in feature.lower():
                    interp = "OVERSOLD (<30)" if avg_value < 30 else "OVERBOUGHT (>70)" if avg_value > 70 else "NEUTRAL (30-70)"
                elif 'slope' in feature.lower():
                    interp = "DOWNTREND (<0)" if avg_value < 0 else "UPTREND (>0)"
                elif 'above' in feature.lower():
                    interp = f"{avg_value*100:.1f}% above SMA"
                else:
                    interp = ""

                print(f"{feature:<30} {avg_value:<14.4f} {interp}")

    # Save detailed results
    df_comparison.to_csv('deep_feature_analysis_results.csv', index=False)

    with open('trade_details_with_features.json', 'w') as f:
        json.dump(trade_details, f, indent=2)

    print(f"\n💾 Results saved to:")
    print(f"   - deep_feature_analysis_results.csv (feature comparison)")
    print(f"   - trade_details_with_features.json (full trade details)")

    # ========================================================================
    # KEY INSIGHTS
    # ========================================================================

    print(f"\n{'='*80}")
    print(f"💡 KEY INSIGHTS")
    print(f"{'='*80}\n")

    # Find features that are significantly different
    significant_features = df_comparison[df_comparison['abs_diff_pct'] > 20]

    if len(significant_features) > 0:
        print(f"⚠️  CRITICAL: Found {len(significant_features)} features with >20% difference!\n")
        print(f"Top 5 problematic features:")
        for idx, row in significant_features.head(5).iterrows():
            print(f"   {row['feature']}: {row['difference_pct']:+.1f}% difference")
            print(f"      Profitable avg: {row['avg_profitable']:.4f}")
            print(f"      Unprofitable avg: {row['avg_unprofitable']:.4f}\n")
    else:
        print(f"✅ No features with extreme differences found")

    print(f"✓ Analysis complete!")


def main():
    """Main execution."""
    analyze_deep_features()


if __name__ == "__main__":
    main()
