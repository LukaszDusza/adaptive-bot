#!/usr/bin/env python3
"""
Analiza najlepszych tradów 1000PEPEUSDT - porównanie z DOGE.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bybit_adapter import BybitAdapter
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

# Best trades timestamps
best_trades = [
    {'timestamp': 1762518733911, 'datetime': '2025-11-07 12:32:13', 'pnl': 1.182717, 'side': 'Buy', 'entry': 0.005666, 'exit': 0.005569},
    {'timestamp': 1762307422690, 'datetime': '2025-11-05 01:50:22', 'pnl': 0.896349, 'side': 'Sell', 'entry': 0.005323, 'exit': 0.005445},
    {'timestamp': 1763426770615, 'datetime': '2025-11-18 00:46:10', 'pnl': 0.862978, 'side': 'Buy', 'entry': 0.004696, 'exit': 0.004613},
    {'timestamp': 1763148206195, 'datetime': '2025-11-14 19:23:26', 'pnl': 0.776013, 'side': 'Sell', 'entry': 0.005072, 'exit': 0.005188},
    {'timestamp': 1763390148759, 'datetime': '2025-11-17 14:35:48', 'pnl': 0.775357, 'side': 'Sell', 'entry': 0.004848, 'exit': 0.004930},
]

print(f"\n{'='*90}")
print(f"🔝 ANALIZA 5 NAJLEPSZYCH TRADÓW 1000PEPEUSDT - MARKET CONDITIONS")
print(f"{'='*90}\n")

results = []

for i, trade in enumerate(best_trades, 1):
    print(f"\n{'─'*90}")
    print(f"TRADE #{i}: {trade['datetime']} | {trade['side']} | PnL: +{trade['pnl']:.4f} USDT")
    print(f"Entry: ${trade['entry']:.6f} → Exit: ${trade['exit']:.6f} | Δ: {((trade['exit']/trade['entry'])-1)*100:.2f}%")
    print(f"{'─'*90}")

    timestamp_ms = trade['timestamp']

    # Fetch OHLCV data
    try:
        end_date = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d')

        klines = adapter.fetch_ohlcv(
            symbol="1000PEPEUSDT",
            timeframe="15",
            limit=200,
            end_date=end_date
        )

        if not klines or len(klines) < 50:
            print("  ❌ Insufficient data")
            continue

        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate key indicators
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_normalized'] = df['atr_14'] / df['close']

        df['rsi_14'] = ta.rsi(df['close'], length=14)

        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)

        # Volume
        df['volume_sma_20'] = ta.sma(df['volume'], length=20)
        df['rvol'] = df['volume'] / df['volume_sma_20']

        # Get values at entry (last candle)
        entry_row = df.iloc[-1]

        close_price = entry_row['close']
        atr_norm = entry_row['atr_normalized']
        rsi = entry_row['rsi_14']
        rvol = entry_row['rvol']

        # Trend analysis
        sma_50 = entry_row['sma_50']
        sma_200 = entry_row['sma_200']

        trend = "UNKNOWN"
        if not pd.isna(sma_50) and not pd.isna(sma_200):
            if sma_50 > sma_200:
                trend = "UPTREND"
            else:
                trend = "DOWNTREND"

        price_vs_sma50 = ((close_price / sma_50) - 1) * 100 if not pd.isna(sma_50) else np.nan

        print(f"\n💹 MARKET CONDITIONS AT ENTRY:")
        print(f"  Close Price: ${close_price:.6f}")
        print(f"  ATR (14): ${entry_row['atr_14']:.8f}")
        print(f"  ATR (normalized): {atr_norm:.6f} {'⚠️ HIGH VOLATILITY!' if atr_norm > 0.015 else ''}")
        print(f"  RSI (14): {rsi:.2f}")
        print(f"  Relative Volume: {rvol:.2f}x {'📈 High volume!' if rvol > 1.5 else ''}")
        print(f"  Trend: {trend}")
        print(f"  Price vs SMA50: {price_vs_sma50:+.2f}%")

        # Analyze last 10 candles
        recent_atr = df.tail(10)['atr_normalized'].mean()
        recent_rvol = df.tail(10)['rvol'].mean()

        print(f"\n📊 RECENT MARKET (last 10 candles ~2.5h):")
        print(f"  Avg ATR: {recent_atr:.6f}")
        print(f"  Avg RVol: {recent_rvol:.2f}x")

        # Check if we entered against trend
        side_vs_trend = ""
        if trade['side'] == 'Buy' and trend == 'DOWNTREND':
            side_vs_trend = "⚠️ LONG AGAINST DOWNTREND!"
        elif trade['side'] == 'Sell' and trend == 'UPTREND':
            side_vs_trend = "⚠️ SHORT AGAINST UPTREND!"

        if side_vs_trend:
            print(f"\n  {side_vs_trend}")

        # Store results
        results.append({
            'datetime': trade['datetime'],
            'side': trade['side'],
            'pnl': trade['pnl'],
            'entry_price': trade['entry'],
            'exit_price': trade['exit'],
            'atr_normalized': atr_norm,
            'rsi_14': rsi,
            'rvol': rvol,
            'trend': trend,
            'price_vs_sma50': price_vs_sma50,
            'side_vs_trend': side_vs_trend
        })

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        continue

# Summary
print(f"\n\n{'='*90}")
print(f"📊 SUMMARY - Best Trades Analysis (1000PEPEUSDT)")
print(f"{'='*90}\n")

results_df = pd.DataFrame(results)

if len(results_df) > 0:
    print(results_df[['datetime', 'side', 'pnl', 'atr_normalized', 'rsi_14', 'rvol', 'trend']].to_string(index=False))

    print(f"\n\n🔍 KEY FINDINGS:")
    print(f"  Avg ATR (normalized): {results_df['atr_normalized'].mean():.6f}")
    print(f"  Avg RSI: {results_df['rsi_14'].mean():.2f}")
    print(f"  Avg RVol: {results_df['rvol'].mean():.2f}x")
    print(f"  Trades against trend: {results_df['side_vs_trend'].str.contains('AGAINST', na=False).sum()}/{len(results_df)}")

    # ATR threshold analysis
    high_atr_count = (results_df['atr_normalized'] > 0.015).sum()
    print(f"\n  Trades with HIGH ATR (>0.015): {high_atr_count}/{len(results_df)} ({high_atr_count/len(results_df)*100:.0f}%)")

    # Volume analysis
    high_vol_count = (results_df['rvol'] > 1.0).sum()
    print(f"  Trades with HIGH Volume (>1.0x): {high_vol_count}/{len(results_df)} ({high_vol_count/len(results_df)*100:.0f}%)")

print(f"\n{'='*90}\n")

# Now compare with DOGE worst trades
print(f"\n{'='*90}")
print(f"📊 COMPARISON: 1000PEPEUSDT (BEST) vs DOGEUSDT (WORST)")
print(f"{'='*90}\n")

# DOGE worst stats (from previous analysis)
doge_worst_atr = 0.008263
doge_worst_rvol = 0.18
doge_worst_rsi = 56.98

if len(results_df) > 0:
    pepe_best_atr = results_df['atr_normalized'].mean()
    pepe_best_rvol = results_df['rvol'].mean()
    pepe_best_rsi = results_df['rsi_14'].mean()

    print(f"{'Metric':<25} | {'PEPE (Winners)':<15} | {'DOGE (Losers)':<15} | {'Diff':<15}")
    print(f"{'-'*25}-+-{'-'*15}-+-{'-'*15}-+-{'-'*15}")
    print(f"{'ATR (normalized)':<25} | {pepe_best_atr:>15.6f} | {doge_worst_atr:>15.6f} | {pepe_best_atr - doge_worst_atr:>+15.6f}")
    print(f"{'Relative Volume':<25} | {pepe_best_rvol:>15.2f}x | {doge_worst_rvol:>15.2f}x | {pepe_best_rvol - doge_worst_rvol:>+15.2f}x")
    print(f"{'RSI':<25} | {pepe_best_rsi:>15.2f} | {doge_worst_rsi:>15.2f} | {pepe_best_rsi - doge_worst_rsi:>+15.2f}")

    print(f"\n🎯 KEY DIFFERENCES:")
    if pepe_best_rvol > doge_worst_rvol * 2:
        print(f"  ✅ WINNERS have {pepe_best_rvol/doge_worst_rvol:.1f}x HIGHER VOLUME")
    if pepe_best_atr > doge_worst_atr:
        print(f"  ⚠️  WINNERS have HIGHER ATR (+{(pepe_best_atr - doge_worst_atr)*100:.2f}%)")
    else:
        print(f"  ✅ WINNERS have LOWER ATR ({(doge_worst_atr - pepe_best_atr)*100:.2f}% less volatile)")

print(f"\n{'='*90}\n")
