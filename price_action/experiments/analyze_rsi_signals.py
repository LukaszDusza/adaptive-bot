#!/usr/bin/env python3
"""
Analiza sygnałów SHORT w kontekście RSI podczas trendu wzrostowego
"""

import pandas as pd
import numpy as np
from data_preparer_pa import fetch_and_prepare_data
from datetime import datetime
import joblib

# Load models
version = "v1.2.sol"
ticker = "SOLUSDT"
timeframe = "15m"
helper_timeframes = ["1h", "4h", "12h", "1d"]

strategy_id_long = f"{ticker}_{timeframe}_plus_{'_'.join(helper_timeframes)}_long"
strategy_id_short = f"{ticker}_{timeframe}_plus_{'_'.join(helper_timeframes)}_short"

model_dir_long = f"models/{version}/{strategy_id_long}"
model_dir_short = f"models/{version}/{strategy_id_short}"

print(f"\n📊 ANALIZA RSI I SYGNAŁÓW TRADING")
print("=" * 100)

# Load models
try:
    model_long = joblib.load(f"{model_dir_long}/model.joblib")
    scaler_long = joblib.load(f"{model_dir_long}/scaler.joblib")
    features_long = joblib.load(f"{model_dir_long}/features.joblib")

    with open(f"{model_dir_long}/training_metadata.json", 'r') as f:
        metadata_long = __import__('json').load(f)
    optimal_threshold_long = metadata_long.get('optimal_threshold', 0.5)

    print(f"✓ LONG model loaded: {len(features_long)} features, threshold={optimal_threshold_long:.3f}")
except Exception as e:
    print(f"❌ Failed to load LONG model: {e}")
    model_long = None

try:
    model_short = joblib.load(f"{model_dir_short}/model.joblib")
    scaler_short = joblib.load(f"{model_dir_short}/scaler.joblib")
    features_short = joblib.load(f"{model_dir_short}/features.joblib")

    with open(f"{model_dir_short}/training_metadata.json", 'r') as f:
        metadata_short = __import__('json').load(f)
    optimal_threshold_short = metadata_short.get('optimal_threshold', 0.5)

    print(f"✓ SHORT model loaded: {len(features_short)} features, threshold={optimal_threshold_short:.3f}")
except Exception as e:
    print(f"❌ Failed to load SHORT model: {e}")
    model_short = None

# Fetch data for last 14 days (same as backtest)
print(f"\n📥 Fetching data...")
df = fetch_and_prepare_data(
    ticker=ticker,
    timeframe=timeframe,
    limit=1344,  # 14 days
    helper_timeframes=helper_timeframes,
    side='long',  # doesn't matter, just need features
    version=version
)

print(f"✓ Data fetched: {len(df)} candles")
print(f"  Date range: {df.index[0]} → {df.index[-1]}")

# Get predictions
prob_threshold = 0.5004  # from live bot
min_confidence_ratio = 1.3

results = []

for idx in range(200, len(df)):  # Skip warmup
    row = df.iloc[idx]

    # Get features for LONG
    X_long = row[features_long].values.reshape(1, -1)
    X_long_scaled = scaler_long.transform(X_long)
    proba_long = model_long.predict_proba(X_long_scaled)[0][1] if model_long else 0

    # Get features for SHORT
    X_short = row[features_short].values.reshape(1, -1)
    X_short_scaled = scaler_short.transform(X_short)
    proba_short = model_short.predict_proba(X_short_scaled)[0][1] if model_short else 0

    # Determine signal (same logic as bot)
    signal = "HOLD"

    if proba_long >= prob_threshold and proba_short >= prob_threshold:
        # Both models confident - use min_confidence_ratio
        if proba_long / proba_short >= min_confidence_ratio:
            signal = "LONG"
        elif proba_short / proba_long >= min_confidence_ratio:
            signal = "SHORT"
    elif proba_long >= prob_threshold:
        signal = "LONG"
    elif proba_short >= prob_threshold:
        signal = "SHORT"

    # Get RSI
    rsi = row.get('rsi_14', None)
    rsi_1h = row.get('1h_rsi_14', None)
    rsi_4h = row.get('4h_rsi_14', None)

    # Price info
    close = row['close']

    results.append({
        'timestamp': row.name,
        'close': close,
        'signal': signal,
        'proba_long': proba_long,
        'proba_short': proba_short,
        'confidence_ratio': proba_long / proba_short if proba_short > 0 else 0,
        'rsi_15m': rsi,
        'rsi_1h': rsi_1h,
        'rsi_4h': rsi_4h
    })

df_results = pd.DataFrame(results)

# Filter for signals only
df_signals = df_results[df_results['signal'] != 'HOLD'].copy()

print(f"\n📊 PODSUMOWANIE SYGNAŁÓW (ostatnie 14 dni):")
print(f"  Total candles: {len(df_results)}")
print(f"  LONG signals: {len(df_signals[df_signals['signal'] == 'LONG'])}")
print(f"  SHORT signals: {len(df_signals[df_signals['signal'] == 'SHORT'])}")
print(f"  HOLD: {len(df_results[df_results['signal'] == 'HOLD'])}")

# Analyze SHORT signals during uptrend (last 3 days = dzisiaj i wczoraj)
print(f"\n" + "=" * 100)
print(f"🔍 ANALIZA SHORTów PODCZAS TRENDU WZROSTOWEGO (ostatnie 3 dni)")
print("=" * 100)

# Last 3 days = 288 candles (3 * 24 * 4)
df_recent = df_results.tail(288).copy()
df_short_signals = df_recent[df_recent['signal'] == 'SHORT'].copy()

print(f"\nSHORT sygnały w ostatnich 3 dniach: {len(df_short_signals)}")

if len(df_short_signals) > 0:
    print(f"\n{'Timestamp':<20} | {'Price':>8} | {'RSI 15m':>8} | {'RSI 1h':>8} | {'RSI 4h':>8} | {'P(SHORT)':>7} | {'P(LONG)':>7} | {'Ratio':>6}")
    print("-" * 100)

    for _, row in df_short_signals.iterrows():
        ts = row['timestamp'].strftime('%Y-%m-%d %H:%M')
        rsi_15m_str = f"{row['rsi_15m']:>8.1f}" if row['rsi_15m'] is not None and not pd.isna(row['rsi_15m']) else "     N/A"
        rsi_1h_str = f"{row['rsi_1h']:>8.1f}" if row['rsi_1h'] is not None and not pd.isna(row['rsi_1h']) else "     N/A"
        rsi_4h_str = f"{row['rsi_4h']:>8.1f}" if row['rsi_4h'] is not None and not pd.isna(row['rsi_4h']) else "     N/A"
        print(f"{ts:<20} | ${row['close']:7.2f} | {rsi_15m_str} | {rsi_1h_str} | {rsi_4h_str} | {row['proba_short']:>7.3f} | {row['proba_long']:>7.3f} | {row['confidence_ratio']:>6.2f}")

    # Statistics
    print(f"\n📈 STATYSTYKI RSI DLA SHORT SYGNAŁÓW:")
    print(f"  RSI 15m - Avg: {df_short_signals['rsi_15m'].mean():.1f} | Min: {df_short_signals['rsi_15m'].min():.1f} | Max: {df_short_signals['rsi_15m'].max():.1f}")
    print(f"  RSI 1h  - Avg: {df_short_signals['rsi_1h'].mean():.1f} | Min: {df_short_signals['rsi_1h'].min():.1f} | Max: {df_short_signals['rsi_1h'].max():.1f}")
    print(f"  RSI 4h  - Avg: {df_short_signals['rsi_4h'].mean():.1f} | Min: {df_short_signals['rsi_4h'].min():.1f} | Max: {df_short_signals['rsi_4h'].max():.1f}")

    # Check how many had RSI > 70 (overbought)
    overbought_15m = (df_short_signals['rsi_15m'] > 70).sum()
    overbought_1h = (df_short_signals['rsi_1h'] > 70).sum()
    overbought_4h = (df_short_signals['rsi_4h'] > 70).sum()

    print(f"\n⚠️  OVERBOUGHT RSI (>70):")
    print(f"  15m: {overbought_15m}/{len(df_short_signals)} ({100*overbought_15m/len(df_short_signals):.1f}%)")
    print(f"  1h:  {overbought_1h}/{len(df_short_signals)} ({100*overbought_1h/len(df_short_signals):.1f}%)")
    print(f"  4h:  {overbought_4h}/{len(df_short_signals)} ({100*overbought_4h/len(df_short_signals):.1f}%)")

# Analyze price trend during last 3 days
first_price = df_recent.iloc[0]['close']
last_price = df_recent.iloc[-1]['close']
price_change_pct = ((last_price - first_price) / first_price) * 100

print(f"\n📈 TREND CENOWY (ostatnie 3 dni):")
print(f"  Start: ${first_price:.2f}")
print(f"  End:   ${last_price:.2f}")
print(f"  Change: {price_change_pct:+.2f}%")

if price_change_pct > 2:
    print(f"\n  ⚠️  SILNY TREND WZROSTOWY - SHORTy powinny być unikane!")
elif price_change_pct < -2:
    print(f"\n  ⚠️  SILNY TREND SPADKOWY - LONGi powinny być unikane!")

# Check for RSI divergence (RSI overbought but price keeps rising)
print(f"\n" + "=" * 100)
print(f"💡 WNIOSKI:")
print("=" * 100)

if len(df_short_signals) > 0 and price_change_pct > 2:
    avg_rsi = df_short_signals['rsi_15m'].mean()
    if avg_rsi > 60:
        print(f"\n  ⚠️  KLASYCZNY PROBLEM: RSI OVERBOUGHT w TRENDZIE WZROSTOWYM")
        print(f"      - Model daje SHORTy bo RSI wysoki ({avg_rsi:.1f})")
        print(f"      - Ale cena dalej rośnie (+{price_change_pct:.2f}%)")
        print(f"      - W silnym trendzie RSI może pozostać overbought przez długi czas!")
        print(f"\n  💡 ROZWIĄZANIE:")
        print(f"      1. Dodaj trend filter (ADX, EMA slope)")
        print(f"      2. Nie shortuj jeśli trend > +X% w ostatnich N dniach")
        print(f"      3. Wymagaj większego confidence ratio w trendzie")
        print(f"      4. Dodaj feature: 'trend_strength_3d' do modelu")

print("\n" + "=" * 100)
