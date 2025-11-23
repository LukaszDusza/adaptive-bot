#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Szybka analiza najgorszych tradów - oblicza features dla specified trades.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bybit_adapter import BybitAdapter
from data_preparer_pa import fetch_and_prepare_data
from dotenv import load_dotenv


# Load env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../.env_luk')
load_dotenv(env_path)

# Initialize adapter
adapter = BybitAdapter(
    api_key=os.getenv('BYBIT_API_KEY'),
    api_secret=os.getenv('BYBIT_API_SECRET'),
    base_url=os.getenv('BYBIT_BASE_URL', 'https://api.bybit.com'),
    category="linear"
)

# Load model info for DOGEUSDT
model_dir = "/Users/lukasz/projects/adaptive-bot/price_action/models/v1.2.doge"
strategy_dir = os.path.join(model_dir, "DOGEUSDT_15m_plus_1h_4h_6h_12h_1d_long")

model = joblib.load(os.path.join(strategy_dir, "model.joblib"))
features = joblib.load(os.path.join(strategy_dir, "features.joblib"))

import json
with open(os.path.join(strategy_dir, "training_metadata.json")) as f:
    metadata = json.load(f)

# Get feature importance
importance = model.feature_importances_
feat_imp_df = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values('importance', ascending=False)

top_features = feat_imp_df.head(20)['feature'].tolist()

# Worst trades timestamps (from previous output)
worst_trades = [
    {'timestamp': 1762530589487, 'datetime': '2025-11-07 15:49:49', 'pnl': -1.536551, 'entry': 0.170317},
    {'timestamp': 1762535475900, 'datetime': '2025-11-07 17:11:15', 'pnl': -1.253942, 'entry': 0.176767},
    {'timestamp': 1763569397099, 'datetime': '2025-11-19 16:23:17', 'pnl': -1.171101, 'entry': 0.151923},
    {'timestamp': 1763405886572, 'datetime': '2025-11-17 18:58:06', 'pnl': -1.040403, 'entry': 0.153965},
    {'timestamp': 1763656695986, 'datetime': '2025-11-20 16:38:15', 'pnl': -0.929397, 'entry': 0.149760},
]

print(f"\n{'='*80}")
print(f"🔻 ANALIZA 5 NAJGORSZYCH TRADÓW DOGEUSDT")
print(f"{'='*80}\n")

results = []

for i, trade in enumerate(worst_trades, 1):
    print(f"\n{'─'*80}")
    print(f"TRADE #{i}: {trade['datetime']} | PnL: {trade['pnl']:.2f} USDT | Entry: ${trade['entry']:.6f}")
    print(f"{'─'*80}")

    timestamp_ms = trade['timestamp']

    # Fetch data and calculate features
    # Note: fetch_and_prepare_data fetches latest data, so we need to use date_from
    # to get data around the timestamp
    from_date = datetime.fromtimestamp(timestamp_ms / 1000 - 30*24*60*60).strftime('%Y-%m-%d')  # 30 days before

    try:
        df = fetch_and_prepare_data(
            ticker="DOGEUSDT",
            timeframe="15m",
            limit=1000,
            helper_timeframes=metadata['helper_timeframes'],
            side="long",
            skip_slow_features=False,
            date_from=from_date
        )

        if df is None or len(df) == 0:
            print("  ❌ No data fetched")
            continue

        # Get last row (entry point)
        last_row = df.iloc[-1]

        # Extract top features
        print(f"\n📊 TOP 10 FEATURES at entry:")
        feature_vals = {}
        for j, feature in enumerate(top_features[:10], 1):
            val = last_row.get(feature, np.nan)
            feature_vals[feature] = val
            print(f"  {j:2d}. {feature:30s} = {val:.6f}")

        # Market context
        print(f"\n💹 MARKET CONTEXT:")
        print(f"  Close: ${last_row.get('close', 0):.6f}")
        print(f"  ATR (normalized): {last_row.get('atr_normalized', 0):.6f}")
        print(f"  ATR (1d): {last_row.get('atr_normalized_1d', 0):.6f}")
        print(f"  RSI 14: {last_row.get('rsi_14', 0):.2f}")
        print(f"  RSI Slope 6h: {last_row.get('rsi_slope_6h', 0):.6f}")
        print(f"  Volume: {last_row.get('volume', 0):.2f}")

        # Check if high volatility
        atr_norm = last_row.get('atr_normalized', 0)
        atr_1d = last_row.get('atr_normalized_1d', 0)

        if atr_norm > 0.015 or atr_1d > 0.02:
            print(f"\n  ⚠️  HIGH VOLATILITY DETECTED!")
            print(f"     ATR normalized: {atr_norm:.6f} (threshold: 0.015)")
            print(f"     ATR 1d: {atr_1d:.6f} (threshold: 0.020)")

        # Store results
        results.append({
            'datetime': trade['datetime'],
            'pnl': trade['pnl'],
            'entry_price': trade['entry'],
            'atr_normalized': atr_norm,
            'atr_1d': atr_1d,
            'rsi_slope_6h': last_row.get('rsi_slope_6h', 0),
            'rsi_14': last_row.get('rsi_14', 0),
        })

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        continue

# Summary
print(f"\n\n{'='*80}")
print(f"📊 SUMMARY - Worst Trades Analysis")
print(f"{'='*80}\n")

results_df = pd.DataFrame(results)

if len(results_df) > 0:
    print(results_df.to_string(index=False))

    print(f"\n\n🔍 KEY FINDINGS:")
    print(f"  Average ATR (normalized): {results_df['atr_normalized'].mean():.6f}")
    print(f"  Average ATR (1d): {results_df['atr_1d'].mean():.6f}")
    print(f"  Average RSI Slope 6h: {results_df['rsi_slope_6h'].mean():.6f}")

    # Compare with all trades
    print(f"\n\n📈 COMPARISON WITH ALL DOGEUSDT TRADES:")

    pnl_all = pd.read_csv('/Users/lukasz/projects/adaptive-bot/price_action/analiza/DOGEUSDT_closed_pnl.csv')

    print(f"  Total trades: {len(pnl_all)}")
    print(f"  Avg PnL (all): {pnl_all['closedPnl'].astype(float).mean():.4f} USDT")
    print(f"  Avg PnL (worst 5): {results_df['pnl'].mean():.4f} USDT")

print(f"\n{'='*80}\n")
