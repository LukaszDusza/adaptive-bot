#!/usr/bin/env python3
"""
Analiza rynku 7 listopada - sprawdzenie jak wyglądał chart i volume.
"""

import sys
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bybit_adapter import BybitAdapter
from dotenv import load_dotenv

# Load env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../.env_luk')
load_dotenv(env_path)

adapter = BybitAdapter(
    api_key=os.getenv('BYBIT_API_KEY'),
    api_secret=os.getenv('BYBIT_API_SECRET'),
    base_url=os.getenv('BYBIT_BASE_URL', 'https://api.bybit.com'),
    category="linear"
)

# Analyze key tickers for Nov 7
tickers = ['DOGEUSDT', 'SOLUSDT', 'BRETTUSDT', 'WIFUSDT', 'PONKEUSDT']

print(f"\n{'='*90}")
print(f"📈 ANALIZA RYNKU - 7 LISTOPADA 2025")
print(f"{'='*90}\n")

for ticker in tickers:
    print(f"\n{'─'*90}")
    print(f"📊 {ticker}")
    print(f"{'─'*90}")

    try:
        # Fetch data for Nov 6-8 (3 days around Nov 7)
        klines = adapter.fetch_ohlcv(
            symbol=ticker,
            timeframe="15",
            limit=300,  # ~3 days
            end_date="2025-11-08"
        )

        if not klines or len(klines) < 100:
            print("  ❌ Insufficient data")
            continue

        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate indicators
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_normalized'] = df['atr_14'] / df['close']
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['volume_sma_20'] = ta.sma(df['volume'], length=20)
        df['rvol'] = df['volume'] / df['volume_sma_20']

        # Filter Nov 7
        df['date'] = df.index.date
        nov7_data = df[df['date'] == pd.to_datetime('2025-11-07').date()]

        if len(nov7_data) == 0:
            print("  ❌ No data for Nov 7")
            continue

        # Calculate Nov 7 statistics
        open_price = nov7_data.iloc[0]['open']
        close_price = nov7_data.iloc[-1]['close']
        high_price = nov7_data['high'].max()
        low_price = nov7_data['low'].min()

        daily_change = ((close_price - open_price) / open_price) * 100
        intraday_range = ((high_price - low_price) / open_price) * 100

        avg_atr = nov7_data['atr_normalized'].mean()
        avg_rvol = nov7_data['rvol'].mean()
        max_rvol = nov7_data['rvol'].max()

        print(f"\n📈 PRICE ACTION:")
        print(f"  Open:  ${open_price:.6f}")
        print(f"  Close: ${close_price:.6f}")
        print(f"  High:  ${high_price:.6f}")
        print(f"  Low:   ${low_price:.6f}")
        print(f"  Daily Change: {daily_change:+.2f}%")
        print(f"  Intraday Range: {intraday_range:.2f}% {'⚠️ WIDE RANGE!' if intraday_range > 10 else ''}")

        print(f"\n💹 VOLATILITY:")
        print(f"  Avg ATR: {avg_atr:.6f} {'⚠️ HIGH!' if avg_atr > 0.015 else ''}")
        print(f"  Avg RVol: {avg_rvol:.2f}x")
        print(f"  Max RVol: {max_rvol:.2f}x {'📈 Volume spike!' if max_rvol > 2.0 else ''}")

        # Check if it was whipsaw day
        # Count how many times price crossed the open level
        df_nov7 = nov7_data.copy()
        df_nov7['above_open'] = df_nov7['close'] > open_price
        crosses = (df_nov7['above_open'] != df_nov7['above_open'].shift()).sum()

        print(f"\n🔄 WHIPSAW ANALYSIS:")
        print(f"  Price crosses above/below open: {crosses} times")
        if crosses > 10:
            print(f"  ⚠️ HIGH WHIPSAW DAY - cena oscylowała {crosses}x!")

        # Show candle by candle movement
        print(f"\n📊 MOVEMENT PATTERN (first 16 candles = 4 hours):")
        for i, (idx, row) in enumerate(nov7_data.head(16).iterrows()):
            time_str = idx.strftime('%H:%M')
            change = ((row['close'] - open_price) / open_price) * 100
            direction = '📈' if row['close'] > row['open'] else '📉'
            print(f"  {time_str}: ${row['close']:.6f} ({change:+.2f}% from open) {direction} RVol:{row['rvol']:.2f}x")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        continue

print(f"\n{'='*90}\n")
