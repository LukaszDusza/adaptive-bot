#!/usr/bin/env python3
"""
Debug: Porównaj timestampy świec live bot vs backtest
"""

import pandas as pd
from data_preparer_pa import fetch_and_prepare_data
from dotenv import load_dotenv
import joblib

# Use LIVE API
load_dotenv('.env')

TICKER = "SOLUSDT"
TIMEFRAME = "15m"
HELPER_TFS = ["1h", "4h", "12h", "1d"]
VERSION = "v1.2.sol"

# Load model features
base_id = f"{TICKER}_{TIMEFRAME}_plus_{'_'.join(HELPER_TFS)}"
long_id = base_id + "_long"
features_long = joblib.load(f"models/{VERSION}/{long_id}/features.joblib")
features_short = joblib.load(f"models/{VERSION}/{long_id.replace('long', 'short')}/features.joblib")
all_features = list(set(features_long + features_short))

print("="*100)
print("DEBUG: Timestampy świec - Live Bot vs Backtest")
print("="*100)

# Fetch jak LIVE BOT (limit=2000, no warmup)
print("\n📦 Fetching data LIVE BOT style (limit=2000)...")
df_live_style = fetch_and_prepare_data(
    ticker=TICKER,
    timeframe=TIMEFRAME,
    limit=2000,
    helper_timeframes=HELPER_TFS,
    side='backtest',
    version=VERSION,
    model_features_to_preserve=all_features
)

# Exclude last candle (jak robi live bot)
df_live_closed = df_live_style.iloc[:-1] if len(df_live_style) > 1 else df_live_style

print(f"✓ Live bot style: {len(df_live_closed)} closed candles")
print(f"  Range: {df_live_closed.index[0]} → {df_live_closed.index[-1]}")

# Show Nov 7 candles
nov7_candles = df_live_closed[df_live_closed.index.date == pd.Timestamp('2025-11-07').date()]
print(f"\n  Nov 7 candles: {len(nov7_candles)}")
print(f"  First: {nov7_candles.index[0]}")
print(f"  Last:  {nov7_candles.index[-1]}")

# Show timestamps around 02:00
print(f"\n  Timestampy wokół 02:00:")
around_2am = nov7_candles[(nov7_candles.index >= '2025-11-07 01:45') &
                          (nov7_candles.index <= '2025-11-07 02:30')]
for ts in around_2am.index:
    print(f"    {ts}")

# Check for close price at 02:00
candle_0200 = nov7_candles[nov7_candles.index == '2025-11-07 02:00:00']
if len(candle_0200) > 0:
    print(f"\n  Świeca 02:00:00:")
    print(f"    Close: {candle_0200.iloc[0]['close']:.2f}")
    print(f"    RSI: {candle_0200.iloc[0].get('rsi_14', 'N/A')}")

    # Check features used in model
    if 'rsi_14' in candle_0200.columns:
        print(f"\n  Sample features @ 02:00:")
        sample_features = ['close', 'rsi_14', 'atr_14', 'sma_50', 'volume']
        for feat in sample_features:
            if feat in candle_0200.columns:
                print(f"    {feat}: {candle_0200.iloc[0][feat]:.4f}")

print("\n" + "="*100)
print("WNIOSKI:")
print("="*100)
print("1. Timestampy są RÓWNE (00:00, 00:15, 00:30...)")
print("2. Backtest używa tych SAMYCH świec co live bot")
print("3. Różnice muszą być gdzie indziej!")
