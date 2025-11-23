#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt do szybkiej analizy danych handlowych z Bybit.

Użycie:
    python analyze_trading_data.py                    # Podsumowanie wszystkich tickerów
    python analyze_trading_data.py SOLUSDT            # Szczegółowa analiza dla tickera
    python analyze_trading_data.py --compare          # Porównanie tickerów
"""

import pandas as pd
import os
import sys
from glob import glob
from datetime import datetime

ANALIZA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_ticker_data(ticker):
    """Load all data for a specific ticker."""
    pnl_file = os.path.join(ANALIZA_DIR, f"{ticker}_closed_pnl.csv")
    exec_file = os.path.join(ANALIZA_DIR, f"{ticker}_executions.csv")
    orders_file = os.path.join(ANALIZA_DIR, f"{ticker}_orders.csv")

    if not os.path.exists(pnl_file):
        return None, None, None

    pnl = pd.read_csv(pnl_file)
    exec_df = pd.read_csv(exec_file) if os.path.exists(exec_file) else None
    orders = pd.read_csv(orders_file) if os.path.exists(orders_file) else None

    return pnl, exec_df, orders


def analyze_ticker(ticker):
    """Detailed analysis for a single ticker."""
    print(f"\n{'='*80}")
    print(f"📊 ANALIZA: {ticker}")
    print(f"{'='*80}\n")

    pnl, exec_df, orders = load_ticker_data(ticker)

    if pnl is None:
        print(f"❌ Brak danych dla {ticker}")
        return

    # Basic stats
    total_pnl = pnl['closedPnl'].astype(float).sum()
    profitable = (pnl['closedPnl'].astype(float) > 0).sum()
    losing = (pnl['closedPnl'].astype(float) < 0).sum()
    win_rate = profitable / len(pnl) * 100 if len(pnl) > 0 else 0

    print("📈 PODSTAWOWE STATYSTYKI:")
    print(f"  Total Trades: {len(pnl)}")
    print(f"  Total PnL: {total_pnl:.2f} USDT")
    print(f"  Win Rate: {win_rate:.1f}% ({profitable}W / {losing}L)")

    # Win/Loss stats
    wins = pnl[pnl['closedPnl'].astype(float) > 0]['closedPnl'].astype(float)
    losses = pnl[pnl['closedPnl'].astype(float) < 0]['closedPnl'].astype(float)

    if len(wins) > 0:
        print(f"  Avg Win: {wins.mean():.4f} USDT | Max Win: {wins.max():.4f} USDT")
    if len(losses) > 0:
        print(f"  Avg Loss: {losses.mean():.4f} USDT | Max Loss: {losses.min():.4f} USDT")

    if len(wins) > 0 and len(losses) > 0:
        rr_ratio = abs(wins.mean() / losses.mean())
        print(f"  Risk/Reward Ratio: {rr_ratio:.2f}")

    # Side analysis
    print("\n📊 ANALIZA PER STRONA (LONG vs SHORT):")
    side_stats = pnl.groupby('side').agg({
        'closedPnl': ['sum', 'mean', 'count']
    }).round(4)
    print(side_stats)

    # Fees analysis
    total_fees = (pnl['openFee'].astype(float) + pnl['closeFee'].astype(float)).sum()
    print(f"\n💰 OPŁATY:")
    print(f"  Total Fees: {total_fees:.4f} USDT")
    print(f"  Avg Fee per Trade: {total_fees / len(pnl):.4f} USDT")
    print(f"  PnL + Fees: {total_pnl + total_fees:.4f} USDT (gross profit)")

    # Time analysis
    if 'createdTime_readable' in pnl.columns:
        pnl['datetime'] = pd.to_datetime(pnl['createdTime_readable'])
        pnl['date'] = pnl['datetime'].dt.date
        pnl['hour'] = pnl['datetime'].dt.hour

        print("\n📅 ANALIZA CZASOWA:")
        pnl['closedPnl_float'] = pnl['closedPnl'].astype(float)
        daily_pnl = pnl.groupby('date')['closedPnl_float'].sum()
        print(f"  Najlepszy dzień: {daily_pnl.idxmax()} (+{daily_pnl.max():.2f} USDT)")
        print(f"  Najgorszy dzień: {daily_pnl.idxmin()} ({daily_pnl.min():.2f} USDT)")

    # Top/Worst trades
    print("\n🔝 TOP 5 NAJLEPSZYCH TRADÓW:")
    best = pnl.nlargest(5, 'closedPnl')[['side', 'closedPnl', 'avgEntryPrice', 'avgExitPrice', 'createdTime_readable']]
    print(best.to_string(index=False))

    print("\n⚠️  TOP 5 NAJGORSZYCH TRADÓW:")
    worst = pnl.nsmallest(5, 'closedPnl')[['side', 'closedPnl', 'avgEntryPrice', 'avgExitPrice', 'createdTime_readable']]
    print(worst.to_string(index=False))

    # Orders analysis
    if orders is not None:
        print(f"\n📋 ANALIZA ORDERÓW:")
        print(f"  Total Orders: {len(orders)}")
        status_counts = orders['orderStatus'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count} ({count/len(orders)*100:.1f}%)")


def summary_all_tickers():
    """Summary for all tickers."""
    print(f"\n{'='*80}")
    print("📊 PODSUMOWANIE WSZYSTKICH TICKERÓW (1-21 listopada 2025)")
    print(f"{'='*80}\n")

    pnl_files = sorted(glob(os.path.join(ANALIZA_DIR, '*_closed_pnl.csv')))

    summary = []

    for file in pnl_files:
        ticker = os.path.basename(file).replace('_closed_pnl.csv', '')
        pnl_df = pd.read_csv(file)

        if len(pnl_df) > 0:
            total_pnl = pnl_df['closedPnl'].astype(float).sum()
            profitable = (pnl_df['closedPnl'].astype(float) > 0).sum()
            losing = (pnl_df['closedPnl'].astype(float) < 0).sum()
            win_rate = profitable / len(pnl_df) * 100

            wins = pnl_df[pnl_df['closedPnl'].astype(float) > 0]['closedPnl'].astype(float)
            losses = pnl_df[pnl_df['closedPnl'].astype(float) < 0]['closedPnl'].astype(float)

            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0
            rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

            summary.append({
                'Ticker': ticker,
                'Trades': len(pnl_df),
                'PnL': total_pnl,
                'WR%': win_rate,
                'AvgWin': avg_win,
                'AvgLoss': avg_loss,
                'R:R': rr
            })

    df = pd.DataFrame(summary).sort_values('PnL', ascending=False)

    # Format for display
    df['PnL'] = df['PnL'].apply(lambda x: f"{x:+.2f}")
    df['WR%'] = df['WR%'].apply(lambda x: f"{x:.1f}")
    df['AvgWin'] = df['AvgWin'].apply(lambda x: f"{x:.4f}")
    df['AvgLoss'] = df['AvgLoss'].apply(lambda x: f"{x:.4f}")
    df['R:R'] = df['R:R'].apply(lambda x: f"{x:.2f}")

    print(df.to_string(index=False))

    # Overall stats
    summary_df = pd.DataFrame(summary)
    print(f"\n{'='*80}")
    print(f"TOTAL PnL: {summary_df['PnL'].sum():.2f} USDT")
    print(f"Total Trades: {summary_df['Trades'].sum()}")
    print(f"Overall Win Rate: {(summary_df['AvgWin'].sum() / (summary_df['AvgWin'].sum() + abs(summary_df['AvgLoss'].sum()))) * 100:.1f}%")

    print("\n🔝 TOP 3 Najlepsze:")
    for _, row in summary_df.nlargest(3, 'PnL').iterrows():
        print(f"  {row['Ticker']}: +{row['PnL']:.2f} USDT ({row['WR%']:.1f}% WR, R:R {row['R:R']:.2f})")

    print("\n⚠️  TOP 3 Najgorsze:")
    for _, row in summary_df.nsmallest(3, 'PnL').iterrows():
        print(f"  {row['Ticker']}: {row['PnL']:.2f} USDT ({row['WR%']:.1f}% WR, R:R {row['R:R']:.2f})")


def main():
    if len(sys.argv) == 1:
        # No args - show summary
        summary_all_tickers()
    elif sys.argv[1] == '--compare':
        summary_all_tickers()
    else:
        # Ticker specified
        ticker = sys.argv[1].upper()
        analyze_ticker(ticker)


if __name__ == "__main__":
    main()
