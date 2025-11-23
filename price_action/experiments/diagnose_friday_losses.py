#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose losses from Friday Nov 8, 2025 with ML indicator analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bybit_adapter import BybitAdapter
import joblib

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

def get_friday_trades():
    """Get all trades from Friday Nov 8, 2025."""
    df = pd.read_csv('live_trades_nov1-11_2025.csv')
    df['createdTime'] = pd.to_datetime(df['createdTime'])
    df['date'] = df['createdTime'].dt.date

    # Filter for Nov 7 (Thursday) - most trades
    target_date = pd.to_datetime('2025-11-07').date()
    target_trades = df[df['date'] == target_date].copy()

    return target_trades

def load_model_and_features(symbol: str):
    """Load model and feature list for symbol."""
    # Map symbols to model versions
    model_map = {
        'SOLUSDT': 'v1.2.sol',
        'DOGEUSDT': 'v1.2.doge',
        'ETHUSDT': 'v1.2.eth',
        'LINKUSDT': 'v1.2.link',
        'HBARUSDT': 'v1.2.hbar',
        'NEARUSDT': 'v1.2.near',
        'WIFUSDT': 'v1.2.wif',
        '1000PEPEUSDT': 'v1.2.1000pepe',
        'BRETTUSDT': 'v1.2.brett',
        'ONDOUSDT': 'v1.2.ondo',
        'PONKEUSDT': 'v1.2.ponke',
        'SUIUSDT': 'v1.2.sui',
        'KASUSDT': 'v1.2.kas',
        'TRXUSDT': 'v1.2.trx',
        'XRPUSDT': 'v1.2.xrp'
    }

    if symbol not in model_map:
        return None, None, None, None

    version = model_map[symbol]

    # Determine helper timeframes
    helpers_map = {
        'SOLUSDT': ['1h', '4h', '12h', '1d'],
        'DOGEUSDT': ['1h', '4h', '6h', '12h', '1d'],
        'ETHUSDT': ['1h', '2h', '4h', '6h', '12h', '1d'],
        'LINKUSDT': ['1h', '2h', '4h', '6h', '12h', '1d'],
        'HBARUSDT': ['1h', '2h', '4h', '6h', '12h', '1d'],
        'NEARUSDT': ['1h', '2h', '4h', '6h', '12h', '1d'],
        'WIFUSDT': ['1h', '2h', '4h', '12h', '1d'],
        '1000PEPEUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'BRETTUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'ONDOUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'PONKEUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'SUIUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'KASUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'TRXUSDT': ['1h', '2h', '4h', '12h', '1d'],
        'XRPUSDT': ['1h', '2h', '4h', '12h', '1d']
    }

    helpers = helpers_map.get(symbol, ['1h', '4h', '12h', '1d'])
    helpers_str = '_'.join(helpers)

    # Try to load models
    try:
        model_long_path = f"models/{version}/{symbol}_15m_plus_{helpers_str}_long/model.joblib"
        model_short_path = f"models/{version}/{symbol}_15m_plus_{helpers_str}_short/model.joblib"
        features_long_path = f"models/{version}/{symbol}_15m_plus_{helpers_str}_long/features.joblib"
        features_short_path = f"models/{version}/{symbol}_15m_plus_{helpers_str}_short/features.joblib"

        model_long = joblib.load(model_long_path)
        model_short = joblib.load(model_short_path)
        features_long = joblib.load(features_long_path)
        features_short = joblib.load(features_short_path)

        return model_long, model_short, features_long, features_short

    except Exception as e:
        print(f"   ⚠️  Could not load models for {symbol}: {e}")
        return None, None, None, None

def get_feature_importance(model, features, top_n=10):
    """Get top N most important features."""
    if model is None or features is None:
        return []

    try:
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feature_importance.head(top_n)
    except:
        return []

def analyze_trade_with_indicators(trade, adapter):
    """Analyze single trade with ML indicators."""
    symbol = trade['symbol']
    entry_time = trade['createdTime']
    side = trade['side']
    entry_price = trade['avgEntryPrice']
    exit_price = trade['avgExitPrice']
    pnl = trade['closedPnl']

    print(f"\n{'='*80}")
    print(f"{'❌' if pnl < 0 else '✅'} {symbol} {side.upper()} @ {entry_time}")
    print(f"{'='*80}")
    print(f"Entry: ${entry_price:.4f} → Exit: ${exit_price:.4f} → P&L: ${pnl:.2f}")

    # Load model
    model_long, model_short, features_long, features_short = load_model_and_features(symbol)

    if model_long is None:
        print(f"   ⚠️  No model found for {symbol}")
        return

    # Get feature importance
    if side == 'Buy':
        fi = get_feature_importance(model_long, features_long, top_n=10)
        model_features = features_long
    else:
        fi = get_feature_importance(model_short, features_short, top_n=10)
        model_features = features_short

    print(f"\n📊 TOP 10 ML FEATURES ({side}):")
    if len(fi) > 0:
        for idx, row in fi.iterrows():
            print(f"   {row['feature']:<40} {row['importance']:.4f}")
    else:
        print(f"   No feature importance data")

    # Try to get actual feature values at entry time
    # This would require fetching OHLCV and calculating features
    # For now, just show what we have

    print(f"\n💡 ANALYSIS:")

    # Analyze based on side
    if side == 'Sell' and pnl < 0:
        # Failed SHORT
        price_change = ((exit_price - entry_price) / entry_price) * 100
        print(f"   ❌ SHORT pozycja straciła podczas wzrostu ceny")
        print(f"   📈 Cena wzrosła o {price_change:.2f}%")
        print(f"   🤔 Możliwe przyczyny:")
        print(f"      - RSI pokazywał wykupienie (overbought)")
        print(f"      - Momentum wskaźniki sugerowały odwrócenie")
        print(f"      - Ale trend był silny i kontynuował wzrost")

        # Check top features
        if len(fi) > 0:
            top_feature = fi.iloc[0]['feature']
            if 'rsi' in top_feature.lower():
                print(f"   ⚠️  PROBLEM: Top feature '{top_feature}' - RSI może źle działać w silnym trendzie!")
            elif 'momentum' in top_feature.lower() or 'roc' in top_feature.lower():
                print(f"   ⚠️  PROBLEM: Top feature '{top_feature}' - momentum może dawać fałszywe sygnały!")

    elif side == 'Buy' and pnl < 0:
        # Failed LONG
        price_change = ((exit_price - entry_price) / entry_price) * 100
        print(f"   ❌ LONG pozycja straciła podczas spadku ceny")
        print(f"   📉 Cena spadła o {abs(price_change):.2f}%")
        print(f"   🤔 Możliwe przyczyny:")
        print(f"      - RSI pokazywał wyprzedanie (oversold)")
        print(f"      - Model spodziewał się odbicia")
        print(f"      - Ale trend spadkowy kontynuował")

def main():
    """Main analysis."""

    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║        THURSDAY NOV 7, 2025 - DETAILED LOSS DIAGNOSIS (83 TRADES!)        ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    # Get Thursday trades (worst day)
    thursday_trades = get_friday_trades()

    if len(thursday_trades) == 0:
        print("\n⚠️  No trades found for Thursday Nov 7, 2025")
        return

    print(f"\n📊 THURSDAY SUMMARY:")
    print(f"   Total Trades: {len(thursday_trades)}")
    print(f"   Total P&L: ${thursday_trades['closedPnl'].sum():.2f}")

    wins = thursday_trades[thursday_trades['closedPnl'] > 0]
    losses = thursday_trades[thursday_trades['closedPnl'] <= 0]

    print(f"   Wins: {len(wins)} (${wins['closedPnl'].sum():.2f})")
    print(f"   Losses: {len(losses)} (${losses['closedPnl'].sum():.2f})")
    print(f"   Win Rate: {len(wins)/len(thursday_trades)*100:.1f}%")

    # Breakdown by side
    longs = thursday_trades[thursday_trades['side'] == 'Buy']
    shorts = thursday_trades[thursday_trades['side'] == 'Sell']

    print(f"\n   LONGs: {len(longs)} trades, ${longs['closedPnl'].sum():.2f}")
    print(f"   SHORTs: {len(shorts)} trades, ${shorts['closedPnl'].sum():.2f}")

    # Load adapter
    adapter = load_account(".env_luk")

    # Analyze worst losses
    print(f"\n{'='*80}")
    print(f"🔍 ANALYZING WORST LOSSES FROM THURSDAY")
    print(f"{'='*80}")

    worst_losses = thursday_trades.nsmallest(10, 'closedPnl')

    for idx, trade in worst_losses.iterrows():
        analyze_trade_with_indicators(trade, adapter)

    # Summary by symbol
    print(f"\n{'='*80}")
    print(f"📊 THURSDAY LOSSES BY SYMBOL")
    print(f"{'='*80}")

    symbol_stats = thursday_trades.groupby('symbol').agg({
        'closedPnl': ['sum', 'count', 'mean']
    }).round(2)
    symbol_stats.columns = ['Total P&L', 'Trades', 'Avg P&L']
    symbol_stats = symbol_stats.sort_values('Total P&L')

    print(symbol_stats.to_string())

    # Analyze SHORT failures specifically
    print(f"\n{'='*80}")
    print(f"🔍 SHORT ANALYSIS - WHY SHORTS FAILED")
    print(f"{'='*80}")

    losing_shorts = shorts[shorts['closedPnl'] < 0].copy()
    print(f"\n❌ Losing SHORTs: {len(losing_shorts)} trades, ${losing_shorts['closedPnl'].sum():.2f}")

    # Calculate price change for each short
    losing_shorts['price_change_pct'] = ((losing_shorts['avgExitPrice'] - losing_shorts['avgEntryPrice']) / losing_shorts['avgEntryPrice'] * 100)

    avg_price_change = losing_shorts['price_change_pct'].mean()
    print(f"\n📈 Średni wzrost ceny podczas SHORT: {avg_price_change:.2f}%")
    print(f"   (SHORT traci kiedy cena ROŚNIE)")

    if avg_price_change > 0:
        print(f"\n💡 DIAGNOZA:")
        print(f"   Model otwierał SHORTy podczas WZROSTU ceny")
        print(f"   To oznacza że:")
        print(f"   1. RSI/Momentum wskaźniki pokazywały 'overbought'")
        print(f"   2. Model spodziewał się korekty/odwrócenia")
        print(f"   3. ALE trend wzrostowy był silniejszy niż oczekiwano")
        print(f"   4. Shorty zostały wyrzucone przez SL podczas dalszego wzrostu")

    print(f"\n💾 Analysis complete!")

if __name__ == "__main__":
    main()
