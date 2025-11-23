#!/usr/bin/env python3
"""
Uniwersalny skrypt do analizy najgorszych tradów dla dowolnego tickera.

Usage: python analyze_ticker_worst_trades.py TICKER
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


def analyze_ticker_worst_trades(ticker, n_trades=5):
    """Analyze worst N trades for a ticker."""

    print(f"\n{'='*90}")
    print(f"🔻 ANALIZA {n_trades} NAJGORSZYCH TRADÓW: {ticker}")
    print(f"{'='*90}\n")

    # Load trades
    analiza_dir = os.path.dirname(os.path.abspath(__file__))
    pnl_file = os.path.join(analiza_dir, f"{ticker}_closed_pnl.csv")

    if not os.path.exists(pnl_file):
        print(f"❌ File not found: {pnl_file}")
        return None

    pnl_df = pd.read_csv(pnl_file)

    # Find worst trades
    worst = pnl_df.nsmallest(n_trades, 'closedPnl')

    print(f"📉 Total trades: {len(pnl_df)}")
    print(f"   Total PnL: {pnl_df['closedPnl'].astype(float).sum():.2f} USDT")
    print(f"   Win Rate: {(pnl_df['closedPnl'].astype(float) > 0).sum() / len(pnl_df) * 100:.1f}%\n")

    results = []

    for idx, (i, trade) in enumerate(worst.iterrows(), 1):
        print(f"\n{'─'*90}")
        print(f"TRADE #{idx}: {trade['createdTime_readable']} | {trade['side']} | PnL: {trade['closedPnl']:.4f} USDT")
        print(f"Entry: ${trade['avgEntryPrice']:.6f} → Exit: ${trade['avgExitPrice']:.6f}")
        print(f"{'─'*90}")

        timestamp_ms = int(trade['createdTime'])

        # Fetch OHLCV data
        try:
            end_date = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d')

            klines = adapter.fetch_ohlcv(
                symbol=ticker,
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

            # Calculate indicators
            df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['atr_normalized'] = df['atr_14'] / df['close']
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['sma_200'] = ta.sma(df['close'], length=200)
            df['volume_sma_20'] = ta.sma(df['volume'], length=20)
            df['rvol'] = df['volume'] / df['volume_sma_20']

            # Get entry values
            entry_row = df.iloc[-1]

            close_price = entry_row['close']
            atr_norm = entry_row['atr_normalized']
            rsi = entry_row['rsi_14']
            rvol = entry_row['rvol']

            # Trend
            sma_50 = entry_row['sma_50']
            sma_200 = entry_row['sma_200']

            trend = "UNKNOWN"
            if not pd.isna(sma_50) and not pd.isna(sma_200):
                trend = "UPTREND" if sma_50 > sma_200 else "DOWNTREND"

            price_vs_sma50 = ((close_price / sma_50) - 1) * 100 if not pd.isna(sma_50) else np.nan

            print(f"\n💹 MARKET CONDITIONS:")
            print(f"  ATR (norm): {atr_norm:.6f} {'⚠️ HIGH!' if atr_norm > 0.015 else ''}")
            print(f"  RSI: {rsi:.2f}")
            print(f"  RVol: {rvol:.2f}x {'⚠️ LOW VOLUME!' if rvol < 0.30 else ''}")
            print(f"  Trend: {trend}")

            # Recent market
            recent_atr = df.tail(10)['atr_normalized'].mean()
            recent_rvol = df.tail(10)['rvol'].mean()

            # Store results
            results.append({
                'datetime': trade['createdTime_readable'],
                'side': trade['side'],
                'pnl': float(trade['closedPnl']),
                'entry': float(trade['avgEntryPrice']),
                'exit': float(trade['avgExitPrice']),
                'atr_normalized': atr_norm,
                'rsi_14': rsi,
                'rvol': rvol,
                'trend': trend,
                'recent_atr': recent_atr,
                'recent_rvol': recent_rvol
            })

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    # Summary
    if results:
        print(f"\n\n{'='*90}")
        print(f"📊 SUMMARY - {ticker}")
        print(f"{'='*90}\n")

        results_df = pd.DataFrame(results)
        print(results_df[['datetime', 'side', 'pnl', 'atr_normalized', 'rsi_14', 'rvol']].to_string(index=False))

        print(f"\n\n🔍 KEY METRICS:")
        print(f"  Avg ATR: {results_df['atr_normalized'].mean():.6f}")
        print(f"  Avg RSI: {results_df['rsi_14'].mean():.2f}")
        print(f"  Avg RVol: {results_df['rvol'].mean():.2f}x")

        # Count problematic conditions
        high_atr = (results_df['atr_normalized'] > 0.015).sum()
        low_vol = (results_df['rvol'] < 0.30).sum()

        print(f"\n  ⚠️  High ATR (>0.015): {high_atr}/{len(results_df)} ({high_atr/len(results_df)*100:.0f}%)")
        print(f"  ⚠️  Low Volume (<0.30x): {low_vol}/{len(results_df)} ({low_vol/len(results_df)*100:.0f}%)")

        return results_df
    else:
        print("\n❌ No results")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_ticker_worst_trades.py TICKER")
        print("Example: python analyze_ticker_worst_trades.py PONKEUSDT")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    analyze_ticker_worst_trades(ticker)
