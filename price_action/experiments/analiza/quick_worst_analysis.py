#!/usr/bin/env python3
"""
Szybka analiza najgorszych tradów - pobiera tylko kluczowe wskaźniki bezpośrednio z API.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

# Worst trades timestamps
worst_trades = [
    {'timestamp': 1762530589487, 'datetime': '2025-11-07 15:49:49', 'pnl': -1.536551, 'side': 'Buy', 'entry': 0.170317, 'exit': 0.17410},
    {'timestamp': 1762535475900, 'datetime': '2025-11-07 17:11:15', 'pnl': -1.253942, 'side': 'Buy', 'entry': 0.176767, 'exit': 0.18069},
    {'timestamp': 1763569397099, 'datetime': '2025-11-19 16:23:17', 'pnl': -1.171101, 'side': 'Sell', 'entry': 0.151923, 'exit': 0.14855},
    {'timestamp': 1763405886572, 'datetime': '2025-11-17 18:58:06', 'pnl': -1.040403, 'side': 'Sell', 'entry': 0.153965, 'exit': 0.14941},
    {'timestamp': 1763656695986, 'datetime': '2025-11-20 16:38:15', 'pnl': -0.929397, 'side': 'Sell', 'entry': 0.149760, 'exit': 0.14635},
]

print(f"\n{'='*90}")
print(f"🔻 ANALIZA 5 NAJGORSZYCH TRADÓW DOGEUSDT - MARKET CONDITIONS")
print(f"{'='*90}\n")

results = []

for i, trade in enumerate(worst_trades, 1):
    print(f"\n{'─'*90}")
    print(f"TRADE #{i}: {trade['datetime']} | {trade['side']} | PnL: {trade['pnl']:.4f} USDT")
    print(f"Entry: ${trade['entry']:.6f} → Exit: ${trade['exit']:.6f} | Δ: {((trade['exit']/trade['entry'])-1)*100:.2f}%")
    print(f"{'─'*90}")

    timestamp_ms = trade['timestamp']

    # Fetch OHLCV data around the timestamp (200 candles before for ATR calculation)
    # Fetch 15m candles
    try:
        # Fetch OHLCV using adapter.fetch_ohlcv
        # Need to convert timestamp to date (YYYY-MM-DD format only)
        end_date = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d')

        klines = adapter.fetch_ohlcv(
            symbol="DOGEUSDT",
            timeframe="15",
            limit=200,
            end_date=end_date
        )

        if not klines or len(klines) < 50:
            print("  ❌ Insufficient data")
            continue

        # Convert to DataFrame - klines is list of lists: [timestamp, open, high, low, close, volume, turnover]
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
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
        print(f"  ATR (14): ${entry_row['atr_14']:.6f}")
        print(f"  ATR (normalized): {atr_norm:.6f} {'⚠️ HIGH VOLATILITY!' if atr_norm > 0.015 else ''}")
        print(f"  RSI (14): {rsi:.2f}")
        print(f"  Relative Volume: {rvol:.2f}x {'📈 High volume!' if rvol > 1.5 else ''}")
        print(f"  Trend: {trend}")
        print(f"  Price vs SMA50: {price_vs_sma50:+.2f}%")

        # Analyze last 10 candles (volatility trend)
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
print(f"📊 SUMMARY - Worst Trades Analysis")
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
    print(f"\n  ⚠️  Trades with HIGH ATR (>0.015): {high_atr_count}/{len(results_df)} ({high_atr_count/len(results_df)*100:.0f}%)")

print(f"\n{'='*90}\n")
