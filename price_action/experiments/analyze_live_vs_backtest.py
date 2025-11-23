#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed analysis of live trades vs backtest to identify discrepancies.
Focus on SHORT positions during price increases.
"""

import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bybit_adapter import BybitAdapter
import pandas as pd
from typing import List, Dict
import time

def load_account(env_file: str) -> BybitAdapter:
    """Load Bybit adapter for specific account."""
    load_dotenv(env_file, override=True)

    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    base_url = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
    hedge_mode = os.getenv("BYBIT_HEDGE_MODE", "false").lower() == "true"

    if not api_key or not api_secret:
        raise ValueError(f"Missing API credentials in {env_file}")

    return BybitAdapter(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        hedge_mode=hedge_mode
    )

def get_detailed_trade_history(adapter: BybitAdapter, start_date_str: str, end_date_str: str):
    """
    Fetch detailed trade history including order details with SL/TP.

    Returns:
        DataFrame with complete trade information
    """
    print(f"\n{'='*80}")
    print(f"📊 FETCHING DETAILED TRADE HISTORY")
    print(f"{'='*80}")

    # Convert dates
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    start_ts = int(start_date.timestamp() * 1000)

    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    end_date = end_date.replace(hour=23, minute=59, second=59)
    end_ts = int(end_date.timestamp() * 1000)

    print(f"📅 Date range: {start_date_str} to {end_date_str}")

    # ========================================
    # 1. Get Closed P&L (has entry/exit prices)
    # ========================================
    print(f"\n🔍 Step 1: Fetching closed P&L...")
    all_closed_pnl = []

    current_end = end_date
    chunk = 1

    while current_end > start_date:
        current_start = max(current_end - timedelta(days=7), start_date)

        start_ts = int(current_start.timestamp() * 1000)
        end_ts = int(current_end.timestamp() * 1000)

        print(f"   Chunk {chunk}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}", end="")

        try:
            closed_pnl = adapter.get_closed_pnl_history(
                symbol=None,
                start_time_ms=start_ts,
                end_time_ms=end_ts,
                limit=100
            )

            if closed_pnl:
                print(f" → {len(closed_pnl)} records")
                all_closed_pnl.extend(closed_pnl)
            else:
                print(" → No data")

        except Exception as e:
            print(f" → Error: {e}")

        current_end = current_start - timedelta(seconds=1)
        chunk += 1
        time.sleep(0.2)

    print(f"\n✓ Total closed P&L records: {len(all_closed_pnl)}")

    if not all_closed_pnl:
        print("⚠️  No data found")
        return None

    # Convert to DataFrame
    df_pnl = pd.DataFrame(all_closed_pnl)
    df_pnl['createdTime'] = pd.to_datetime(df_pnl['createdTime'].astype(float), unit='ms')
    df_pnl['updatedTime'] = pd.to_datetime(df_pnl['updatedTime'].astype(float), unit='ms')

    # Convert numeric fields
    df_pnl['closedPnl'] = df_pnl['closedPnl'].astype(float)
    df_pnl['avgEntryPrice'] = df_pnl['avgEntryPrice'].astype(float)
    df_pnl['avgExitPrice'] = df_pnl['avgExitPrice'].astype(float)
    df_pnl['qty'] = df_pnl['qty'].astype(float)

    # ========================================
    # 2. Get Order History (has SL/TP info)
    # ========================================
    print(f"\n🔍 Step 2: Fetching order history with SL/TP...")

    # Get unique order IDs from closed P&L
    order_ids = df_pnl['orderId'].unique()
    print(f"   Found {len(order_ids)} unique orders")

    # Fetch order details in batches
    all_orders = []
    symbols = df_pnl['symbol'].unique()

    for symbol in symbols:
        print(f"   Fetching orders for {symbol}...", end="")

        current_end = end_date
        symbol_orders = []

        while current_end > start_date:
            current_start = max(current_end - timedelta(days=7), start_date)

            start_ts = int(current_start.timestamp() * 1000)
            end_ts = int(current_end.timestamp() * 1000)

            try:
                orders = adapter.get_order_history(
                    symbol=symbol,
                    start_time_ms=start_ts,
                    end_time_ms=end_ts,
                    limit=50
                )

                if orders:
                    symbol_orders.extend(orders)

            except Exception as e:
                print(f" Error: {e}")
                break

            current_end = current_start - timedelta(seconds=1)
            time.sleep(0.2)

        print(f" {len(symbol_orders)} orders")
        all_orders.extend(symbol_orders)

    print(f"\n✓ Total orders fetched: {len(all_orders)}")

    if not all_orders:
        print("⚠️  No order history found")
        # Return P&L data without SL/TP info
        return df_pnl

    # Convert orders to DataFrame
    df_orders = pd.DataFrame(all_orders)
    df_orders['createdTime'] = pd.to_datetime(df_orders['createdTime'].astype(float), unit='ms')

    # Convert SL/TP to float (handle empty strings)
    df_orders['stopLoss'] = pd.to_numeric(df_orders['stopLoss'].replace('', '0'), errors='coerce').fillna(0)
    df_orders['takeProfit'] = pd.to_numeric(df_orders['takeProfit'].replace('', '0'), errors='coerce').fillna(0)
    df_orders['price'] = pd.to_numeric(df_orders['price'].replace('', '0'), errors='coerce').fillna(0)
    df_orders['avgPrice'] = pd.to_numeric(df_orders['avgPrice'].replace('', '0'), errors='coerce').fillna(0)

    # ========================================
    # 3. Merge P&L with Order Details
    # ========================================
    print(f"\n🔗 Step 3: Merging data...")

    # Merge on orderId
    df_merged = df_pnl.merge(
        df_orders[['orderId', 'stopLoss', 'takeProfit', 'orderStatus', 'avgPrice', 'price']],
        on='orderId',
        how='left',
        suffixes=('', '_order')
    )

    print(f"✓ Merged {len(df_merged)} records")

    return df_merged

def analyze_shorts_during_rallies(df: pd.DataFrame):
    """
    Identify SHORT positions opened during price rallies.
    """
    print(f"\n{'='*80}")
    print(f"🔍 ANALYZING SHORTS DURING PRICE RALLIES")
    print(f"{'='*80}")

    # Filter SHORT positions
    shorts = df[df['side'] == 'Sell'].copy()
    print(f"\n📊 Total SHORT positions: {len(shorts)}")

    if len(shorts) == 0:
        print("No SHORT positions found")
        return

    # Calculate price change at exit
    shorts['price_change_pct'] = ((shorts['avgExitPrice'] - shorts['avgEntryPrice']) / shorts['avgEntryPrice'] * 100)

    # Identify losing shorts where price went UP (against the short)
    losing_shorts_up = shorts[
        (shorts['closedPnl'] < 0) &
        (shorts['price_change_pct'] > 0)
    ].copy()

    print(f"\n⚠️  Losing SHORTs during RALLIES: {len(losing_shorts_up)} trades")
    print(f"   Total loss from these trades: ${losing_shorts_up['closedPnl'].sum():.2f}")

    if len(losing_shorts_up) > 0:
        print(f"\n📋 Details of losing SHORTs during rallies:")
        print(f"{'Symbol':<15} {'Date':<20} {'Entry':<10} {'Exit':<10} {'Change %':<10} {'P&L':<10} {'SL':<10} {'TP':<10}")
        print("="*110)

        for idx, row in losing_shorts_up.iterrows():
            sl = f"${row['stopLoss']:.4f}" if row['stopLoss'] > 0 else "None"
            tp = f"${row['takeProfit']:.4f}" if row['takeProfit'] > 0 else "None"

            print(f"{row['symbol']:<15} {row['createdTime'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"${row['avgEntryPrice']:<9.4f} ${row['avgExitPrice']:<9.4f} "
                  f"{row['price_change_pct']:<9.2f}% ${row['closedPnl']:<9.2f} {sl:<10} {tp:<10}")

    # Group by symbol
    print(f"\n📊 SHORTS during rallies by symbol:")
    symbol_stats = losing_shorts_up.groupby('symbol').agg({
        'closedPnl': ['sum', 'count', 'mean']
    }).round(2)
    symbol_stats.columns = ['Total Loss', 'Count', 'Avg Loss']
    print(symbol_stats.to_string())

    return losing_shorts_up

def save_backtest_comparison_data(df: pd.DataFrame, filename: str = "backtest_comparison_data.csv"):
    """
    Save data in format suitable for backtest comparison.
    """
    print(f"\n{'='*80}")
    print(f"💾 SAVING BACKTEST COMPARISON DATA")
    print(f"{'='*80}")

    # Select relevant columns
    comparison_df = df[[
        'symbol', 'side', 'createdTime', 'updatedTime',
        'avgEntryPrice', 'avgExitPrice', 'qty', 'closedPnl',
        'stopLoss', 'takeProfit', 'leverage'
    ]].copy()

    # Sort by time
    comparison_df = comparison_df.sort_values('createdTime')

    # Save
    comparison_df.to_csv(filename, index=False)
    print(f"✓ Saved {len(comparison_df)} trades to: {filename}")
    print(f"\n📋 Date range: {comparison_df['createdTime'].min()} to {comparison_df['createdTime'].max()}")
    print(f"📊 Symbols: {', '.join(comparison_df['symbol'].unique())}")

def main():
    """Main execution."""

    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║        LIVE TRADING ANALYSIS - IDENTIFY ML PREDICTION ISSUES              ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    # Date range: Nov 1-11, 2025
    start_date = "2025-11-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Load account
    adapter = load_account(".env_luk")

    # Get detailed trade history
    df = get_detailed_trade_history(adapter, start_date, end_date)

    if df is None or len(df) == 0:
        print("\n❌ No data to analyze")
        return

    # Analyze shorts during rallies
    losing_shorts = analyze_shorts_during_rallies(df)

    # Save data for backtest comparison
    save_backtest_comparison_data(df, "live_trades_nov1-11_2025.csv")

    print(f"\n{'='*80}")
    print("✓ Analysis complete!")
    print(f"{'='*80}\n")

    # Summary
    print("\n📊 SUMMARY:")
    print(f"   Total trades: {len(df)}")
    print(f"   Total P&L: ${df['closedPnl'].sum():.2f}")
    print(f"   LONG trades: {len(df[df['side'] == 'Buy'])}")
    print(f"   SHORT trades: {len(df[df['side'] == 'Sell'])}")

    if losing_shorts is not None and len(losing_shorts) > 0:
        print(f"\n⚠️  PROBLEM IDENTIFIED:")
        print(f"   {len(losing_shorts)} SHORT positions were opened during price rallies")
        print(f"   These resulted in ${losing_shorts['closedPnl'].sum():.2f} loss")
        print(f"   This suggests ML model is too aggressive with SHORT signals")
        print(f"   Likely cause: Over-reliance on RSI/momentum indicators during extended trends")

if __name__ == "__main__":
    main()
