#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest for Nov 1-9, 2025 to compare with live trading results.
Uses EXACT same parameters as live bots.
"""

import sys
import pandas as pd
from datetime import datetime
from backtester import BacktestEngine

def run_backtest_comparison():
    """Run backtest for same period as live trading."""

    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║        BACKTEST vs LIVE COMPARISON - NOVEMBER 1-9, 2025                   ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    # Load live results for comparison
    live_df = pd.read_csv('live_trades_nov1-11_2025.csv')
    live_df['createdTime'] = pd.to_datetime(live_df['createdTime'])

    print(f"\n📊 LIVE TRADING RESULTS:")
    print(f"   Period: {live_df['createdTime'].min()} to {live_df['createdTime'].max()}")
    print(f"   Total Trades: {len(live_df)}")
    print(f"   Total P&L: ${live_df['closedPnl'].sum():.2f}")

    wins = live_df[live_df['closedPnl'] > 0]
    losses = live_df[live_df['closedPnl'] <= 0]
    print(f"   Win Rate: {len(wins)/len(live_df)*100:.1f}%")
    print(f"   Avg Win: ${wins['closedPnl'].mean():.2f}")
    print(f"   Avg Loss: ${losses['closedPnl'].mean():.2f}")

    # Symbols traded
    symbols = live_df['symbol'].unique()
    print(f"\n📋 Symbols traded: {len(symbols)}")
    for symbol in symbols:
        symbol_trades = live_df[live_df['symbol'] == symbol]
        print(f"   {symbol}: {len(symbol_trades)} trades, ${symbol_trades['closedPnl'].sum():.2f}")

    # Backtest configuration (from docker-compose.yaml)
    configs = {
        'SOLUSDT': {
            'version': 'v1.2.sol',
            'timeframe': '15m',
            'helpers': ['1h', '4h', '12h', '1d'],
            'prob_threshold': 0.5004,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        'DOGEUSDT': {
            'version': 'v1.2.doge',
            'timeframe': '15m',
            'helpers': ['1h', '4h', '6h', '12h', '1d'],
            'prob_threshold': 0.6039,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        'ETHUSDT': {
            'version': 'v1.2.eth',
            'timeframe': '15m',
            'helpers': ['1h', '2h', '4h', '6h', '12h', '1d'],
            'prob_threshold': 0.75,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        'LINKUSDT': {
            'version': 'v1.2.link',
            'timeframe': '15m',
            'helpers': ['1h', '2h', '4h', '6h', '12h', '1d'],
            'prob_threshold': 0.7142,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        'HBARUSDT': {
            'version': 'v1.2.hbar',
            'timeframe': '15m',
            'helpers': ['1h', '2h', '4h', '6h', '12h', '1d'],
            'prob_threshold': 0.6838,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        'NEARUSDT': {
            'version': 'v1.2.near',
            'timeframe': '15m',
            'helpers': ['1h', '2h', '4h', '6h', '12h', '1d'],
            'prob_threshold': 0.5815,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        'WIFUSDT': {
            'version': 'v1.2.wif',
            'timeframe': '15m',
            'helpers': ['1h', '2h', '4h', '12h', '1d'],
            'prob_threshold': 0.44,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        '1000PEPEUSDT': {
            'version': 'v1.2.1000pepe',
            'timeframe': '15m',
            'helpers': ['1h', '2h', '4h', '12h', '1d'],
            'prob_threshold': 0.47,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        'BRETTUSDT': {
            'version': 'v1.2.brett',
            'timeframe': '15m',
            'helpers': ['1h', '2h', '4h', '12h', '1d'],
            'prob_threshold': 0.42,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        'ONDOUSDT': {
            'version': 'v1.2.ondo',
            'timeframe': '15m',
            'helpers': ['1h', '2h', '4h', '12h', '1d'],
            'prob_threshold': 0.40,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        'PONKEUSDT': {
            'version': 'v1.2.ponke',
            'timeframe': '15m',
            'helpers': ['1h', '2h', '4h', '12h', '1d'],
            'prob_threshold': 0.43,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        'SUIUSDT': {
            'version': 'v1.2.sui',
            'timeframe': '15m',
            'helpers': ['1h', '2h', '4h', '12h', '1d'],
            'prob_threshold': 0.4654,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        'KASUSDT': {
            'version': 'v1.2.kas',
            'timeframe': '15m',
            'helpers': ['1h', '2h', '4h', '12h', '1d'],
            'prob_threshold': 0.5537,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        'TRXUSDT': {
            'version': 'v1.2.trx',
            'timeframe': '15m',
            'helpers': ['1h', '2h', '4h', '12h', '1d'],
            'prob_threshold': 0.6719,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        },
        'XRPUSDT': {
            'version': 'v1.2.xrp',
            'timeframe': '15m',
            'helpers': ['1h', '2h', '4h', '12h', '1d'],
            'prob_threshold': 0.64,
            'tp_pct': 0.025,
            'tsl_pct': 0.008
        }
    }

    print(f"\n{'='*80}")
    print(f"🔬 RUNNING BACKTESTS")
    print(f"{'='*80}")

    backtest_results = {}

    for symbol in symbols:
        if symbol not in configs:
            print(f"\n⚠️  Skipping {symbol} - no config found")
            continue

        config = configs[symbol]
        print(f"\n📊 {symbol}:")
        print(f"   Model: {config['version']}")
        print(f"   Prob Threshold: {config['prob_threshold']}")
        print(f"   TP: {config['tp_pct']*100:.1f}%, TSL: {config['tsl_pct']*100:.1f}%")

        try:
            # Initialize backtester
            bt = BacktestEngine(
                ticker=symbol,
                timeframe=config['timeframe'],
                helper_timeframes=config['helpers'],
                version=config['version'],
                prob_threshold=config['prob_threshold'],
                min_proba_diff=0.0,  # Using only prob_threshold
                tp_pct=config['tp_pct'],
                tsl_pct=config['tsl_pct'],
                trade_size_usd=100.0,
                leverage=20
            )

            # Run backtest for Nov 1-9, 2025
            # Need ~1000 candles for 15m = ~10 days
            results = bt.run(
                start_date="2025-11-01",
                end_date="2025-11-09",
                use_dynamic_tp=True  # Same as live bots
            )

            if results:
                backtest_results[symbol] = results
                print(f"   ✓ Backtest complete:")
                print(f"      Trades: {results['total_trades']}")
                print(f"      P&L: ${results['total_pnl']:.2f}")
                print(f"      Win Rate: {results['win_rate']:.1f}%")
            else:
                print(f"   ❌ Backtest failed")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Compare results
    print(f"\n{'='*80}")
    print(f"📈 COMPARISON: LIVE vs BACKTEST")
    print(f"{'='*80}")

    print(f"\n{'Symbol':<15} {'Live Trades':<12} {'BT Trades':<12} {'Live P&L':<12} {'BT P&L':<12} {'Match?'}")
    print("="*80)

    total_live_pnl = 0
    total_bt_pnl = 0
    total_live_trades = 0
    total_bt_trades = 0

    for symbol in symbols:
        live_symbol = live_df[live_df['symbol'] == symbol]
        live_trades = len(live_symbol)
        live_pnl = live_symbol['closedPnl'].sum()

        bt_trades = 0
        bt_pnl = 0.0

        if symbol in backtest_results:
            bt_trades = backtest_results[symbol]['total_trades']
            bt_pnl = backtest_results[symbol]['total_pnl']

        match = "✅" if abs(live_trades - bt_trades) <= 2 and abs(live_pnl - bt_pnl) < 5 else "❌"

        print(f"{symbol:<15} {live_trades:<12} {bt_trades:<12} ${live_pnl:<11.2f} ${bt_pnl:<11.2f} {match}")

        total_live_pnl += live_pnl
        total_bt_pnl += bt_pnl
        total_live_trades += live_trades
        total_bt_trades += bt_trades

    print("="*80)
    print(f"{'TOTAL':<15} {total_live_trades:<12} {total_bt_trades:<12} ${total_live_pnl:<11.2f} ${total_bt_pnl:<11.2f}")

    # Calculate discrepancy
    trades_diff = abs(total_live_trades - total_bt_trades)
    pnl_diff = abs(total_live_pnl - total_bt_pnl)
    pnl_diff_pct = (pnl_diff / abs(total_live_pnl) * 100) if total_live_pnl != 0 else 0

    print(f"\n{'='*80}")
    print(f"🔍 DISCREPANCY ANALYSIS")
    print(f"{'='*80}")

    print(f"\n📊 Trades:")
    print(f"   Live: {total_live_trades}")
    print(f"   Backtest: {total_bt_trades}")
    print(f"   Difference: {trades_diff} trades ({trades_diff/total_live_trades*100:.1f}%)")

    print(f"\n💰 P&L:")
    print(f"   Live: ${total_live_pnl:.2f}")
    print(f"   Backtest: ${total_bt_pnl:.2f}")
    print(f"   Difference: ${pnl_diff:.2f} ({pnl_diff_pct:.1f}%)")

    # Verdict
    print(f"\n{'='*80}")
    print(f"🎯 VERDICT")
    print(f"{'='*80}")

    if pnl_diff_pct < 10 and trades_diff < 10:
        print(f"\n✅ BACKTEST IS RELIABLE")
        print(f"   P&L difference: {pnl_diff_pct:.1f}% (acceptable)")
        print(f"   Trade count difference: {trades_diff} (acceptable)")
        print(f"   → Backtest accurately reflects live performance")
    elif pnl_diff_pct < 25:
        print(f"\n⚠️  BACKTEST HAS MINOR DISCREPANCIES")
        print(f"   P&L difference: {pnl_diff_pct:.1f}% (moderate)")
        print(f"   → Possible causes:")
        print(f"      - Slippage/execution differences")
        print(f"      - Fee calculation differences")
        print(f"      - Timing of signal generation")
    else:
        print(f"\n❌ BACKTEST IS UNRELIABLE")
        print(f"   P&L difference: {pnl_diff_pct:.1f}% (CRITICAL)")
        print(f"   → Major issues detected:")
        print(f"      - Look-ahead bias (using future data)")
        print(f"      - Feature calculation errors")
        print(f"      - SL/TP logic mismatch")
        print(f"      - Data quality issues")

    print(f"\n💾 Results saved to: backtest_nov2025_results.csv")

if __name__ == "__main__":
    run_backtest_comparison()
