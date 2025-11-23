#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check if losing trades were against the main trend (SMA 200).
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bybit_adapter import BybitAdapter
import time

def load_account(env_file: str) -> BybitAdapter:
    """Load Bybit adapter."""
    load_dotenv(env_file, override=True)
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    base_url = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")

    return BybitAdapter(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        hedge_mode=False
    )

def get_thursday_worst_losses():
    """Get worst losses from Thursday."""
    df = pd.read_csv('live_trades_nov1-11_2025.csv')
    df['createdTime'] = pd.to_datetime(df['createdTime'])
    df['date'] = df['createdTime'].dt.date

    # Filter for Nov 7
    target_date = pd.to_datetime('2025-11-07').date()
    thursday = df[df['date'] == target_date].copy()

    # Get worst losses
    worst = thursday.nsmallest(20, 'closedPnl')

    return worst

def fetch_ohlcv_for_trade(adapter, symbol, entry_time, lookback_candles=300):
    """
    Fetch OHLCV data for a trade with enough history for SMA 200.

    Args:
        adapter: BybitAdapter
        symbol: Trading symbol
        entry_time: Entry timestamp
        lookback_candles: Number of candles to fetch (need 200+ for SMA 200)
    """
    try:
        # Convert entry time to timestamp
        if isinstance(entry_time, str):
            entry_time = pd.to_datetime(entry_time)

        # Format end_date as YYYY-MM-DD
        end_date_str = entry_time.strftime('%Y-%m-%d')

        # Fetch 15m candles
        ohlcv = adapter.fetch_ohlcv(
            symbol=symbol,
            timeframe='15',
            limit=lookback_candles,
            end_date=end_date_str
        )

        if not ohlcv or len(ohlcv) < 200:
            print(f"   ⚠️  Not enough data for {symbol} (got {len(ohlcv) if ohlcv else 0} candles)")
            return None

        # Convert to DataFrame
        # Bybit returns: [timestamp, open, high, low, close, volume, turnover]
        df = pd.DataFrame(ohlcv)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']

        # Convert types
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # Calculate SMA 200
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # Get last row (entry candle)
        entry_candle = df.iloc[-1]

        return entry_candle

    except Exception as e:
        print(f"   ❌ Error fetching data for {symbol}: {e}")
        return None

def analyze_trend_direction(trades, adapter):
    """Analyze if trades were with or against trend."""

    print(f"\n{'='*80}")
    print(f"🔍 TREND ANALYSIS - SMA 200 DIRECTION")
    print(f"{'='*80}")

    results = []

    for idx, trade in trades.iterrows():
        symbol = trade['symbol']
        side = trade['side']
        entry_time = trade['createdTime']
        entry_price = trade['avgEntryPrice']
        pnl = trade['closedPnl']

        print(f"\n{'❌' if pnl < 0 else '✅'} {symbol} {side.upper()} @ {entry_time.strftime('%H:%M')}")
        print(f"   Entry: ${entry_price:.4f} | P&L: ${pnl:.2f}")

        # Fetch OHLCV
        entry_candle = fetch_ohlcv_for_trade(adapter, symbol, entry_time)

        if entry_candle is None:
            results.append({
                'symbol': symbol,
                'side': side,
                'pnl': pnl,
                'entry_price': entry_price,
                'sma_200': None,
                'against_trend': None
            })
            continue

        sma_200 = entry_candle['sma_200']
        close = entry_candle['close']

        # Determine if against trend
        if side == 'Buy':
            # LONG: should trade when price > SMA 200 (uptrend)
            against_trend = close < sma_200
            trend_str = "DOWNTREND (bearish)" if against_trend else "UPTREND (bullish)"
        else:
            # SHORT: should trade when price < SMA 200 (downtrend)
            against_trend = close > sma_200
            trend_str = "UPTREND (bullish)" if against_trend else "DOWNTREND (bearish)"

        distance_pct = ((close - sma_200) / sma_200) * 100

        print(f"   Close: ${close:.4f} | SMA 200: ${sma_200:.4f}")
        print(f"   Distance from SMA: {distance_pct:+.2f}%")
        print(f"   Market Trend: {trend_str}")

        if against_trend:
            print(f"   ⚠️  COUNTER-TREND TRADE!")
            if side == 'Buy':
                print(f"      Opened LONG in DOWNTREND (price below SMA 200)")
            else:
                print(f"      Opened SHORT in UPTREND (price above SMA 200)")
        else:
            print(f"   ✅ WITH-TREND TRADE")

        results.append({
            'symbol': symbol,
            'side': side,
            'pnl': pnl,
            'entry_price': entry_price,
            'sma_200': sma_200,
            'close': close,
            'distance_pct': distance_pct,
            'against_trend': against_trend
        })

        # Sleep to avoid rate limits
        time.sleep(0.3)

    # Summary
    df_results = pd.DataFrame(results)
    df_results = df_results.dropna(subset=['against_trend'])

    if len(df_results) == 0:
        print("\n⚠️  No valid data for analysis")
        return

    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")

    total = len(df_results)
    against_trend_count = df_results['against_trend'].sum()
    with_trend_count = total - against_trend_count

    against_trend_pct = (against_trend_count / total) * 100
    with_trend_pct = (with_trend_count / total) * 100

    print(f"\n📈 TREND ALIGNMENT:")
    print(f"   Total Trades Analyzed: {total}")
    print(f"   ❌ Counter-Trend Trades: {against_trend_count} ({against_trend_pct:.1f}%)")
    print(f"   ✅ With-Trend Trades: {with_trend_count} ({with_trend_pct:.1f}%)")

    # P&L by trend direction
    counter_trend_pnl = df_results[df_results['against_trend']]['pnl'].sum()
    with_trend_pnl = df_results[~df_results['against_trend']]['pnl'].sum()

    print(f"\n💰 P&L BY TREND DIRECTION:")
    print(f"   Counter-Trend P&L: ${counter_trend_pnl:.2f}")
    print(f"   With-Trend P&L: ${with_trend_pnl:.2f}")

    if against_trend_count > 0:
        avg_counter_pnl = counter_trend_pnl / against_trend_count
        print(f"   Avg Counter-Trend Loss: ${avg_counter_pnl:.2f}")

    if with_trend_count > 0:
        avg_with_pnl = with_trend_pnl / with_trend_count
        print(f"   Avg With-Trend P&L: ${avg_with_pnl:.2f}")

    # Breakdown by side
    print(f"\n📊 BREAKDOWN BY SIDE:")

    longs = df_results[df_results['side'] == 'Buy']
    shorts = df_results[df_results['side'] == 'Sell']

    if len(longs) > 0:
        long_counter = longs['against_trend'].sum()
        long_counter_pct = (long_counter / len(longs)) * 100
        long_counter_pnl = longs[longs['against_trend']]['pnl'].sum()

        print(f"\n   LONG trades:")
        print(f"      Total: {len(longs)}")
        print(f"      Counter-Trend (price < SMA 200): {long_counter} ({long_counter_pct:.1f}%)")
        print(f"      Counter-Trend P&L: ${long_counter_pnl:.2f}")

    if len(shorts) > 0:
        short_counter = shorts['against_trend'].sum()
        short_counter_pct = (short_counter / len(shorts)) * 100
        short_counter_pnl = shorts[shorts['against_trend']]['pnl'].sum()

        print(f"\n   SHORT trades:")
        print(f"      Total: {len(shorts)}")
        print(f"      Counter-Trend (price > SMA 200): {short_counter} ({short_counter_pct:.1f}%)")
        print(f"      Counter-Trend P&L: ${short_counter_pnl:.2f}")

    # Key insight
    print(f"\n{'='*80}")
    print(f"💡 KEY INSIGHT")
    print(f"{'='*80}")

    if against_trend_pct > 60:
        print(f"\n⚠️  CRITICAL: {against_trend_pct:.1f}% tradów było PRZECIWKO głównemu trendowi!")
        print(f"   Model ignoruje trend direction (SMA 200)")
        print(f"   RSI/momentum features dominują bez trend confirmation")
        print(f"\n🎯 REKOMENDACJA:")
        print(f"   Dodaj MANDATORY trend filter:")
        print(f"   - LONG tylko gdy price > SMA_200")
        print(f"   - SHORT tylko gdy price < SMA_200")
        print(f"   Expected impact: Eliminate {against_trend_pct:.0f}% worst trades = +${abs(counter_trend_pnl):.2f}")

    elif against_trend_pct > 40:
        print(f"\n⚠️  MODERATE: {against_trend_pct:.1f}% tradów przeciwko trendowi")
        print(f"   To jest sporo - dodanie trend filter może pomóc")
    else:
        print(f"\n✅ GOOD: Tylko {against_trend_pct:.1f}% tradów przeciwko trendowi")
        print(f"   Problem leży gdzie indziej")

    # Save results
    df_results.to_csv('trend_analysis_thursday.csv', index=False)
    print(f"\n💾 Results saved to: trend_analysis_thursday.csv")

def main():
    """Main execution."""

    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║        TREND DIRECTION ANALYSIS - ARE LOSSES COUNTER-TREND?               ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    # Get worst losses
    worst_losses = get_thursday_worst_losses()

    print(f"\n📊 Analyzing {len(worst_losses)} worst losses from Thursday Nov 7")

    # Load adapter
    adapter = load_account(".env_luk")

    # Analyze
    analyze_trend_direction(worst_losses, adapter)

    print(f"\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
