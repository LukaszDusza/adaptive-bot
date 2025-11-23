#!/usr/bin/env python3
"""
Porównanie wyników backtest vs live bot dla 7 listopada 2025
"""

import sys
import pandas as pd
import joblib
from backtester import BacktestEngine
from data_preparer_pa import fetch_and_prepare_data
from dotenv import load_dotenv

# CRITICAL: Use LIVE API to match live bot data exactly
load_dotenv('.env')

# Config (same as live bot)
TICKER = "SOLUSDT"
TIMEFRAME = "15m"
HELPER_TFS = ["1h", "4h", "12h", "1d"]
VERSION = "v1.2.sol"

# Live bot params
PROB_THRESHOLD = 0.5004
MIN_CONFIDENCE_RATIO = 1.3
TP_PCT = 0.025
TSL_PCT = 0.008
LEVERAGE = 20
TRADE_SIZE = 100.0

# DCA params
DCA_ENABLED = True
DCA_LEVEL1_PCT = 0.006
DCA_ATR_MULTIPLIER = 1.5
DCA_MAX_SWING_DISTANCE = 0.015
LIMIT_ORDER_CANDLES = 16
LIMIT_OFFSET_PCT = 0.005

print("=" * 100)
print("PORÓWNANIE BACKTEST vs LIVE BOT (7 listopada 2025)")
print("=" * 100)

# Load models
base_id = f"{TICKER}_{TIMEFRAME}_plus_{'_'.join(HELPER_TFS)}"
long_id = base_id + "_long"
short_id = base_id + "_short"

print(f"\n📦 Loading models from version: {VERSION}")
model_long = joblib.load(f"models/{VERSION}/{long_id}/model.joblib")
scaler_long = joblib.load(f"models/{VERSION}/{long_id}/scaler.joblib")
features_long = joblib.load(f"models/{VERSION}/{long_id}/features.joblib")

model_short = joblib.load(f"models/{VERSION}/{short_id}/model.joblib")
scaler_short = joblib.load(f"models/{VERSION}/{short_id}/scaler.joblib")
features_short = joblib.load(f"models/{VERSION}/{short_id}/features.joblib")

all_model_features = list(set(features_long + features_short))

# Fetch data (EXACTLY 2000 like live bot uses)
print(f"\n📊 Fetching data...")
df = fetch_and_prepare_data(
    ticker=TICKER,
    timeframe=TIMEFRAME,
    limit=2000,  # CRITICAL: Must match live bot's limit=2000 for identical features
    helper_timeframes=HELPER_TFS,
    side='backtest',
    version=VERSION,
    model_features_to_preserve=all_model_features
)

print(f"✓ Data fetched: {len(df)} candles")
print(f"  Date range: {df.index[0]} → {df.index[-1]}")

# Show how many Nov 7 candles we have
nov7_candles = df[df.index.date == pd.Timestamp('2025-11-07').date()]
print(f"  Nov 7 candles: {len(nov7_candles)} (from {nov7_candles.index[0]} to {nov7_candles.index[-1]})")

# Run backtest
print(f"\n🚀 Running backtest with DCA mode...")
engine = BacktestEngine(
    initial_capital=10000.0,
    maker_fee=0.0002,
    taker_fee=0.00055,
    slippage_pct=0.0003,
    enable_limit_order=True,
    limit_order_candles=LIMIT_ORDER_CANDLES,
    limit_offset_pct=LIMIT_OFFSET_PCT,
    dca_enabled=DCA_ENABLED,
    dca_level1_pct=DCA_LEVEL1_PCT,
    dca_atr_multiplier=DCA_ATR_MULTIPLIER,
    dca_max_swing_distance=DCA_MAX_SWING_DISTANCE,
    dca_min_level_distance=0.003,
    min_order_qty=0.01,
    timeframe=TIMEFRAME,
    cooldown_minutes=0,  # DISABLE COOLDOWN to match live bot behavior
    warmup_candles=200,
    suppress_limit_logs=True  # Quiet logs
)

results = engine.run(
    df=df,
    model_long=model_long,
    scaler_long=scaler_long,
    features_long=features_long,
    model_short=model_short,
    scaler_short=scaler_short,
    features_short=features_short,
    prob_threshold=PROB_THRESHOLD,
    risk_pct=0.02,
    tp_pct=TP_PCT,
    sl_pct=TP_PCT * 0.5,
    tsl_pct=TSL_PCT,
    enable_partial_tp=False,
    enable_dynamic_tp=True,
    min_confidence_ratio=MIN_CONFIDENCE_RATIO
)

# Filter trades from Nov 7, 2025
trades_df = pd.DataFrame(engine.trades)
if len(trades_df) > 0:
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

    # Filter Nov 7
    nov7_trades = trades_df[trades_df['entry_time'].dt.date == pd.Timestamp('2025-11-07').date()]

    print(f"\n{'='*100}")
    print(f"BACKTEST TRADES (7 listopada 2025):")
    print(f"{'='*100}")

    if len(nov7_trades) > 0:
        for idx, trade in nov7_trades.iterrows():
            entry_time = trade['entry_time'].strftime('%H:%M:%S')
            exit_time = trade['exit_time'].strftime('%H:%M:%S')
            side_emoji = "🟢" if trade['side'] == 'Long' else "🔴"

            print(f"{entry_time} → {exit_time} | {side_emoji} {trade['side']:5s} | "
                  f"Entry=${trade['entry_price']:7.2f} Exit=${trade['exit_price']:7.2f} | "
                  f"PnL=${trade['net_pnl_usd']:+6.2f} ({trade['pnl_pct']:+6.2f}%) | "
                  f"Reason: {trade['exit_reason']}")

        total_pnl = nov7_trades['net_pnl_usd'].sum()
        wins = (nov7_trades['net_pnl_usd'] > 0).sum()
        wr = wins / len(nov7_trades) * 100

        print(f"\n📊 Backtest Summary (7.11):")
        print(f"   Total Trades: {len(nov7_trades)}")
        print(f"   Win Rate: {wr:.1f}%")
        print(f"   Total PnL: ${total_pnl:+.2f}")
    else:
        print("❌ Brak trades z 7 listopada w backteście!")
else:
    print("❌ Brak trades w backteście!")

print(f"\n{'='*100}")
print(f"LIVE BOT TRADES (7 listopada 2025 - z analyze_bybit_today.py):")
print(f"{'='*100}")

# Live bot trades (from earlier analysis)
live_trades = [
    ("02:02:30", "SHORT", 154.56, 156.71, +0.61),
    ("07:56:01", "LONG", 157.41, 156.99, +0.18),
    ("10:03:36", "SHORT", 156.21, 154.45, -0.56),
    ("11:17:41", "SHORT", 153.45, 151.37, -0.66),
    ("15:37:13", "SHORT", 150.22, 152.38, +0.20),
    ("16:10:07", "LONG", 155.55, 157.29, -0.56),
    ("17:31:48", "LONG", 158.86, 161.07, -0.70),
]

for entry_time, side, entry, exit, pnl in live_trades:
    side_emoji = "🟢" if side == "LONG" else "🔴"
    pnl_pct = ((exit - entry) / entry * 100) if side == "LONG" else ((entry - exit) / entry * 100)
    print(f"{entry_time}          | {side_emoji} {side:5s} | "
          f"Entry=${entry:7.2f} Exit=${exit:7.2f} | "
          f"PnL=${pnl:+6.2f} ({pnl_pct:+6.2f}%)")

live_total_pnl = sum([pnl for _, _, _, _, pnl in live_trades])
live_wins = sum([1 for _, _, _, _, pnl in live_trades if pnl > 0])
live_wr = live_wins / len(live_trades) * 100

print(f"\n📊 Live Bot Summary (7.11):")
print(f"   Total Trades: {len(live_trades)}")
print(f"   Win Rate: {live_wr:.1f}%")
print(f"   Total PnL: ${live_total_pnl:+.2f}")

# Check ALL decision log for Nov 7 to see when trading started
print(f"\n{'='*100}")
print(f"ALL DECISIONS (7 listopada - pierwsze 20):")
print(f"{'='*100}")

decision_df = pd.DataFrame(engine.decision_log)
if len(decision_df) > 0:
    decision_df['timestamp'] = pd.to_datetime(decision_df['timestamp'])
    nov7_decisions = decision_df[decision_df['timestamp'].dt.date == pd.Timestamp('2025-11-07').date()]

    print(f"Total Nov 7 decisions: {len(nov7_decisions)}")
    print(f"First decision: {nov7_decisions.iloc[0]['timestamp']}")
    print(f"Last decision: {nov7_decisions.iloc[-1]['timestamp']}")

    print(f"\nFirst 20 decisions:")

    # DEBUG: Print first decision keys
    if len(nov7_decisions) > 0:
        print(f"\nDEBUG: Keys in first decision: {nov7_decisions.iloc[0].keys().tolist()}\n")

    for _, dec in nov7_decisions.head(20).iterrows():
        ts = dec['timestamp'].strftime('%H:%M:%S')
        signal = dec['decision']
        prob_long = dec.get('proba_long', 0)  # Changed from prob_long to proba_long!
        prob_short = dec.get('proba_short', 0)  # Changed from prob_short to proba_short!
        cooldown = dec.get('cooldown_blocked', False)

        print(f"{ts} | Signal={signal:5s} | P(L)={prob_long:.3f} P(S)={prob_short:.3f} | "
              f"Cooldown={'YES' if cooldown else 'NO '}")

print(f"\n{'='*100}")
print(f"RÓŻNICE:")
print(f"{'='*100}")
if len(nov7_trades) > 0:
    print(f"  Liczba tradów:  Backtest={len(nov7_trades)} vs Live={len(live_trades)}")
    print(f"  Win Rate:       Backtest={wr:.1f}% vs Live={live_wr:.1f}%")
    print(f"  Total PnL:      Backtest=${total_pnl:+.2f} vs Live=${live_total_pnl:+.2f}")
    print(f"  Różnica PnL:    ${total_pnl - live_total_pnl:+.2f}")
else:
    print("  ❌ Nie można porównać - brak trades z backtestów!")
