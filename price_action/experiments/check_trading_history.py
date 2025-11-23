#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to check Bybit trading history and calculate P&L.
Fetches closed P&L data for both accounts up to November 1, 2025.
"""

import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bybit_adapter import BybitAdapter
import pandas as pd
from typing import List, Dict

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

def get_trading_history_range(adapter: BybitAdapter, account_name: str, start_date_str: str, end_date_str: str):
    """
    Fetch and analyze trading history for an account within a date range.

    Args:
        adapter: BybitAdapter instance
        account_name: Account identifier (e.g., 'Lukasz', 'Sylwia')
        start_date_str: Start date (YYYY-MM-DD)
        end_date_str: End date (YYYY-MM-DD)
    """
    print(f"\n{'='*80}")
    print(f"📊 ACCOUNT: {account_name}")
    print(f"{'='*80}")

    # Convert dates to timestamps
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    start_date = start_date.replace(hour=0, minute=0, second=0)

    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    end_date = end_date.replace(hour=23, minute=59, second=59)

    print(f"📅 Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # ========================================
    # 1. CLOSED P&L HISTORY (7-day chunks)
    # ========================================
    print(f"\n🔍 Fetching closed P&L history in 7-day chunks...")

    all_closed_pnl = []

    # Split into 7-day chunks
    current_end = end_date
    chunk = 1

    while current_end > start_date:
        current_start = max(current_end - timedelta(days=7), start_date)

        start_ts = int(current_start.timestamp() * 1000)
        end_ts = int(current_end.timestamp() * 1000)

        print(f"   Chunk {chunk}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}", end="")

        try:
            # Get closed P&L for this 7-day window
            closed_pnl = adapter.get_closed_pnl_history(
                symbol=None,  # All symbols
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

        # Move to previous 7-day window
        current_end = current_start - timedelta(seconds=1)
        chunk += 1

    print(f"\n✓ Total closed P&L records: {len(all_closed_pnl)}")

    if not all_closed_pnl:
        print(f"\n⚠️  No trading history found for {account_name}")
        print(f"   This account may not have any closed positions in the specified period.")
        return

    # ========================================
    # 2. ANALYZE CLOSED P&L
    # ========================================
    df = pd.DataFrame(all_closed_pnl)

    # Convert timestamps to readable dates
    df['date'] = pd.to_datetime(df['createdTime'].astype(float), unit='ms')

    # Convert numeric fields
    df['closedPnl'] = df['closedPnl'].astype(float)
    df['avgEntryPrice'] = df['avgEntryPrice'].astype(float)
    df['avgExitPrice'] = df['avgExitPrice'].astype(float)
    df['qty'] = df['qty'].astype(float)
    df['leverage'] = df['leverage'].astype(float)

    # Add win/loss classification
    df['outcome'] = df['closedPnl'].apply(lambda x: 'WIN' if x > 0 else 'LOSS')

    # Sort by date
    df = df.sort_values('date')

    # ========================================
    # 3. CALCULATE STATISTICS
    # ========================================
    total_trades = len(df)
    total_pnl = df['closedPnl'].sum()

    wins = df[df['closedPnl'] > 0]
    losses = df[df['closedPnl'] <= 0]

    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0

    avg_win = wins['closedPnl'].mean() if num_wins > 0 else 0
    avg_loss = losses['closedPnl'].mean() if num_losses > 0 else 0

    largest_win = df['closedPnl'].max()
    largest_loss = df['closedPnl'].min()

    # Profit factor
    total_wins = wins['closedPnl'].sum() if num_wins > 0 else 0
    total_losses = abs(losses['closedPnl'].sum()) if num_losses > 0 else 0
    profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')

    # ========================================
    # 4. PRINT SUMMARY
    # ========================================
    print(f"\n{'='*80}")
    print(f"📈 TRADING SUMMARY - {account_name}")
    print(f"{'='*80}")

    print(f"\n💰 P&L OVERVIEW:")
    print(f"   Total P&L: ${total_pnl:,.2f} USDT")
    print(f"   Total Trades: {total_trades}")
    print(f"   Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")

    print(f"\n📊 WIN/LOSS STATISTICS:")
    print(f"   Wins: {num_wins} ({win_rate:.1f}%)")
    print(f"   Losses: {num_losses} ({100-win_rate:.1f}%)")
    print(f"   Win Rate: {win_rate:.1f}%")

    print(f"\n💵 PROFIT METRICS:")
    print(f"   Avg Win: ${avg_win:.2f}")
    print(f"   Avg Loss: ${avg_loss:.2f}")
    print(f"   Largest Win: ${largest_win:.2f}")
    print(f"   Largest Loss: ${largest_loss:.2f}")
    print(f"   Profit Factor: {profit_factor:.2f}")

    # Symbol breakdown
    print(f"\n🎯 BREAKDOWN BY SYMBOL:")
    symbol_stats = df.groupby('symbol').agg({
        'closedPnl': ['sum', 'count', 'mean'],
        'outcome': lambda x: (x == 'WIN').sum()
    }).round(2)
    symbol_stats.columns = ['Total P&L', 'Trades', 'Avg P&L', 'Wins']
    symbol_stats['Win Rate %'] = (symbol_stats['Wins'] / symbol_stats['Trades'] * 100).round(1)
    print(symbol_stats.to_string())

    # ========================================
    # 5. RECENT TRADES
    # ========================================
    print(f"\n📋 LAST 10 TRADES:")
    recent = df[['date', 'symbol', 'side', 'qty', 'avgEntryPrice', 'avgExitPrice', 'closedPnl', 'outcome']].tail(10)
    for idx, row in recent.iterrows():
        outcome_symbol = "✅" if row['outcome'] == 'WIN' else "❌"
        print(f"   {outcome_symbol} {row['date'].strftime('%Y-%m-%d %H:%M')} | {row['symbol']:12} | "
              f"{row['side']:4} | Qty: {row['qty']:7.2f} | "
              f"Entry: ${row['avgEntryPrice']:8.2f} | Exit: ${row['avgExitPrice']:8.2f} | "
              f"P&L: ${row['closedPnl']:8.2f}")

    # ========================================
    # 6. SAVE TO CSV
    # ========================================
    csv_filename = f"trading_history_{account_name.lower()}_{end_date_str}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 Full history saved to: {csv_filename}")

def main():
    """Main execution function."""

    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║           BYBIT TRADING HISTORY ANALYZER - CRYPTO TRADING BOT             ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    # From November 1, 2025 to today (November 11, 2025)
    start_date = "2025-11-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n🗓️  Date range: {start_date} to {end_date} (last ~10 days)")

    # Only check Lukasz account
    accounts = [
        (".env_luk", "Lukasz"),
    ]

    for env_file, account_name in accounts:
        try:
            adapter = load_account(env_file)
            get_trading_history_range(adapter, account_name, start_date, end_date)
        except Exception as e:
            print(f"\n❌ Error processing account {account_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("✓ Analysis complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
