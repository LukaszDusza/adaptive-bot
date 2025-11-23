#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep analysis of loss sources - identify why avg loss > avg win.
"""

import pandas as pd
import numpy as np

def analyze_losses():
    """Analyze where losses come from."""

    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║           DEEP LOSS ANALYSIS - FIND THE ROOT CAUSE                        ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    # Load data
    df = pd.read_csv('live_trades_nov1-11_2025.csv')
    df['createdTime'] = pd.to_datetime(df['createdTime'])
    df['updatedTime'] = pd.to_datetime(df['updatedTime'])

    # Calculate metrics
    df['price_change_pct'] = ((df['avgExitPrice'] - df['avgEntryPrice']) / df['avgEntryPrice'] * 100)
    df['duration_minutes'] = (df['updatedTime'] - df['createdTime']).dt.total_seconds() / 60
    df['outcome'] = df['closedPnl'].apply(lambda x: 'WIN' if x > 0 else 'LOSS')

    # Calculate if SL or TP was hit
    df['sl_set'] = df['stopLoss'] > 0
    df['tp_set'] = df['takeProfit'] > 0

    # For LONG: SL hit if exit < SL (within tolerance)
    # For SHORT: SL hit if exit > SL (within tolerance)
    df['sl_likely_hit'] = False
    df.loc[(df['side'] == 'Buy') & (df['sl_set']), 'sl_likely_hit'] = \
        df.loc[(df['side'] == 'Buy') & (df['sl_set']), 'avgExitPrice'] <= df.loc[(df['side'] == 'Buy') & (df['sl_set']), 'stopLoss'] * 1.01

    df.loc[(df['side'] == 'Sell') & (df['sl_set']), 'sl_likely_hit'] = \
        df.loc[(df['side'] == 'Sell') & (df['sl_set']), 'avgExitPrice'] >= df.loc[(df['side'] == 'Sell') & (df['sl_set']), 'stopLoss'] * 0.99

    # For LONG: TP hit if exit >= TP (within tolerance)
    # For SHORT: TP hit if exit <= TP (within tolerance)
    df['tp_likely_hit'] = False
    df.loc[(df['side'] == 'Buy') & (df['tp_set']), 'tp_likely_hit'] = \
        df.loc[(df['side'] == 'Buy') & (df['tp_set']), 'avgExitPrice'] >= df.loc[(df['side'] == 'Buy') & (df['tp_set']), 'takeProfit'] * 0.99

    df.loc[(df['side'] == 'Sell') & (df['tp_set']), 'tp_likely_hit'] = \
        df.loc[(df['side'] == 'Sell') & (df['tp_set']), 'avgExitPrice'] <= df.loc[(df['side'] == 'Sell') & (df['tp_set']), 'takeProfit'] * 1.01

    print(f"\n{'='*80}")
    print(f"📊 OVERALL STATISTICS")
    print(f"{'='*80}")

    total_trades = len(df)
    total_pnl = df['closedPnl'].sum()
    wins = df[df['closedPnl'] > 0]
    losses = df[df['closedPnl'] <= 0]

    print(f"\n💰 P&L:")
    print(f"   Total: ${total_pnl:.2f}")
    print(f"   Wins: {len(wins)} trades, ${wins['closedPnl'].sum():.2f} total, ${wins['closedPnl'].mean():.2f} avg")
    print(f"   Losses: {len(losses)} trades, ${losses['closedPnl'].sum():.2f} total, ${losses['closedPnl'].mean():.2f} avg")
    print(f"   Win Rate: {len(wins)/total_trades*100:.1f}%")
    print(f"   Profit Factor: {abs(wins['closedPnl'].sum() / losses['closedPnl'].sum()):.2f}")

    print(f"\n⏱️  Duration:")
    print(f"   Avg Win Duration: {wins['duration_minutes'].mean():.1f} min")
    print(f"   Avg Loss Duration: {losses['duration_minutes'].mean():.1f} min")

    print(f"\n📊 LONG vs SHORT:")
    print(f"\n   LONG (Buy):")
    longs = df[df['side'] == 'Buy']
    long_wins = longs[longs['closedPnl'] > 0]
    long_losses = longs[longs['closedPnl'] <= 0]
    print(f"      Total: {len(longs)} trades")
    print(f"      P&L: ${longs['closedPnl'].sum():.2f}")
    print(f"      Win Rate: {len(long_wins)/len(longs)*100:.1f}%")
    print(f"      Avg Win: ${long_wins['closedPnl'].mean():.2f}" if len(long_wins) > 0 else "      Avg Win: N/A")
    print(f"      Avg Loss: ${long_losses['closedPnl'].mean():.2f}" if len(long_losses) > 0 else "      Avg Loss: N/A")

    print(f"\n   SHORT (Sell):")
    shorts = df[df['side'] == 'Sell']
    short_wins = shorts[shorts['closedPnl'] > 0]
    short_losses = shorts[shorts['closedPnl'] <= 0]
    print(f"      Total: {len(shorts)} trades")
    print(f"      P&L: ${shorts['closedPnl'].sum():.2f}")
    print(f"      Win Rate: {len(short_wins)/len(shorts)*100:.1f}%")
    print(f"      Avg Win: ${short_wins['closedPnl'].mean():.2f}" if len(short_wins) > 0 else "      Avg Win: N/A")
    print(f"      Avg Loss: ${short_losses['closedPnl'].mean():.2f}" if len(short_losses) > 0 else "      Avg Loss: N/A")

    print(f"\n{'='*80}")
    print(f"🎯 SL/TP ANALYSIS")
    print(f"{'='*80}")

    print(f"\n   SL/TP Configuration:")
    print(f"      Trades with SL set: {df['sl_set'].sum()} ({df['sl_set'].sum()/len(df)*100:.1f}%)")
    print(f"      Trades with TP set: {df['tp_set'].sum()} ({df['tp_set'].sum()/len(df)*100:.1f}%)")

    print(f"\n   Exit Reasons (estimated):")
    sl_hits = df[df['sl_likely_hit']]
    tp_hits = df[df['tp_likely_hit']]
    manual_exits = df[~df['sl_likely_hit'] & ~df['tp_likely_hit']]

    print(f"      SL hits (losses): {len(sl_hits)} trades, ${sl_hits['closedPnl'].sum():.2f}")
    print(f"      TP hits (wins): {len(tp_hits)} trades, ${tp_hits['closedPnl'].sum():.2f}")
    print(f"      Manual/Other exits: {len(manual_exits)} trades, ${manual_exits['closedPnl'].sum():.2f}")

    print(f"\n{'='*80}")
    print(f"❌ TOP 20 WORST LOSSES")
    print(f"{'='*80}")

    worst_losses = df.nsmallest(20, 'closedPnl')
    print(f"\n{'Symbol':<15} {'Side':<6} {'Date':<20} {'Entry':<10} {'Exit':<10} {'P&L':<10} {'Duration':<12} {'SL Hit':<8} {'TP Hit'}")
    print("="*120)

    for idx, row in worst_losses.iterrows():
        print(f"{row['symbol']:<15} {row['side']:<6} {row['createdTime'].strftime('%Y-%m-%d %H:%M'):<20} "
              f"${row['avgEntryPrice']:<9.4f} ${row['avgExitPrice']:<9.4f} "
              f"${row['closedPnl']:<9.2f} {row['duration_minutes']:<11.1f}m "
              f"{'Yes' if row['sl_likely_hit'] else 'No':<8} {'Yes' if row['tp_likely_hit'] else 'No'}")

    print(f"\n{'='*80}")
    print(f"🔍 LOSS ANALYSIS BY SYMBOL")
    print(f"{'='*80}")

    # Worst performing symbols
    symbol_stats = df.groupby('symbol').agg({
        'closedPnl': ['sum', 'count', 'mean'],
        'outcome': lambda x: (x == 'WIN').sum()
    }).round(2)
    symbol_stats.columns = ['Total P&L', 'Trades', 'Avg P&L', 'Wins']
    symbol_stats['Win Rate %'] = (symbol_stats['Wins'] / symbol_stats['Trades'] * 100).round(1)
    symbol_stats = symbol_stats.sort_values('Total P&L')

    print(f"\nWorst 5 symbols:")
    print(symbol_stats.head(5).to_string())

    print(f"\n{'='*80}")
    print(f"💡 KEY INSIGHTS")
    print(f"{'='*80}")

    # Calculate key metrics
    avg_win_size = wins['closedPnl'].mean()
    avg_loss_size = abs(losses['closedPnl'].mean())
    loss_to_win_ratio = avg_loss_size / avg_win_size

    print(f"\n1. RISK/REWARD IMBALANCE:")
    print(f"   Average win: ${avg_win_size:.2f}")
    print(f"   Average loss: ${avg_loss_size:.2f}")
    print(f"   Loss/Win ratio: {loss_to_win_ratio:.2f}x")
    print(f"   ⚠️  Your average LOSS is {loss_to_win_ratio:.1f}x larger than average WIN!")
    print(f"   This means you need {loss_to_win_ratio*100:.0f}% win rate just to break even.")

    print(f"\n2. SL/TP EFFECTIVENESS:")
    sl_hit_rate = len(sl_hits) / len(df) * 100
    tp_hit_rate = len(tp_hits) / len(df) * 100
    print(f"   SL hit rate: {sl_hit_rate:.1f}% of all trades")
    print(f"   TP hit rate: {tp_hit_rate:.1f}% of all trades")

    if sl_hit_rate > tp_hit_rate:
        print(f"   ⚠️  SL hits more often than TP - suggests SL too tight or wrong market timing")

    print(f"\n3. DURATION ANALYSIS:")
    if losses['duration_minutes'].mean() > wins['duration_minutes'].mean():
        duration_diff = losses['duration_minutes'].mean() - wins['duration_minutes'].mean()
        print(f"   ⚠️  Losing trades last {duration_diff:.1f} minutes LONGER than winning trades")
        print(f"   This suggests you're holding losers too long hoping for recovery")

    print(f"\n4. DIRECTION BIAS:")
    long_pnl = longs['closedPnl'].sum()
    short_pnl = shorts['closedPnl'].sum()

    if long_pnl < 0 and short_pnl < 0:
        print(f"   ⚠️  BOTH directions losing money:")
        print(f"      LONGs: ${long_pnl:.2f}")
        print(f"      SHORTs: ${short_pnl:.2f}")
        print(f"   This suggests fundamental issues with:")
        print(f"      - Model predictions (ML not working)")
        print(f"      - SL/TP positioning (risk management)")
        print(f"      - Market conditions (high volatility/chop)")
    elif long_pnl < short_pnl:
        print(f"   LONGs underperforming SHORTs:")
        print(f"      LONGs: ${long_pnl:.2f}")
        print(f"      SHORTs: ${short_pnl:.2f}")
        print(f"   Possible in downtrend market, but check if model is too bearish")

    # Save detailed analysis
    df_analysis = df[[
        'symbol', 'side', 'createdTime', 'updatedTime', 'duration_minutes',
        'avgEntryPrice', 'avgExitPrice', 'price_change_pct',
        'closedPnl', 'outcome', 'leverage',
        'stopLoss', 'takeProfit', 'sl_likely_hit', 'tp_likely_hit'
    ]].copy()

    df_analysis = df_analysis.sort_values('closedPnl')
    df_analysis.to_csv('detailed_loss_analysis.csv', index=False)

    print(f"\n💾 Detailed analysis saved to: detailed_loss_analysis.csv")

if __name__ == "__main__":
    analyze_losses()
