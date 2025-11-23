#!/usr/bin/env python3
"""
Analiza ostatnich transakcji - sprawdzenie drawdown, godzin trading, profitable hours
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd

def analyze_trades():
    trades_dir = Path("logs/trades")

    if not trades_dir.exists():
        print(f"❌ Brak katalogu {trades_dir}")
        return

    # Wczytaj wszystkie JSONy
    trades = []
    for file_path in sorted(trades_dir.glob("*.json")):
        try:
            with open(file_path, 'r') as f:
                trade_data = json.load(f)
                # Dodaj nazwę pliku dla debugowania
                trade_data['_filename'] = file_path.name
                trades.append(trade_data)
        except Exception as e:
            print(f"⚠️ Błąd wczytywania {file_path.name}: {e}")

    if not trades:
        print("❌ Brak transakcji do analizy")
        return

    print(f"\n📊 ANALIZA {len(trades)} TRANSAKCJI")
    print("=" * 80)

    # Podstawowe statystyki
    total_pnl = 0
    wins = 0
    losses = 0
    shorts = 0
    longs = 0

    # Analiza po godzinach
    hourly_stats = defaultdict(lambda: {'pnl': 0, 'count': 0, 'wins': 0, 'losses': 0})

    # Analiza po tickerach
    ticker_stats = defaultdict(lambda: {'pnl': 0, 'count': 0, 'wins': 0, 'losses': 0, 'shorts': 0, 'longs': 0})

    # Szczegółowe dane dla każdej pozycji
    detailed_trades = []

    for trade in trades:
        # Podstawowe info
        side = trade.get('side', 'UNKNOWN')
        ticker = trade.get('ticker', 'UNKNOWN')
        pnl_pct = trade.get('pnl_percentage', 0)

        # Entry time
        entry_time_str = trade.get('entry_time', '')
        entry_dt = None
        if entry_time_str:
            try:
                entry_dt = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
            except:
                pass

        # PnL
        total_pnl += pnl_pct

        if pnl_pct > 0:
            wins += 1
        else:
            losses += 1

        # Side
        if side.upper() == 'SHORT':
            shorts += 1
        else:
            longs += 1

        # Hourly stats
        if entry_dt:
            hour = entry_dt.hour
            hourly_stats[hour]['pnl'] += pnl_pct
            hourly_stats[hour]['count'] += 1
            if pnl_pct > 0:
                hourly_stats[hour]['wins'] += 1
            else:
                hourly_stats[hour]['losses'] += 1

        # Ticker stats
        ticker_stats[ticker]['pnl'] += pnl_pct
        ticker_stats[ticker]['count'] += 1
        if pnl_pct > 0:
            ticker_stats[ticker]['wins'] += 1
        else:
            ticker_stats[ticker]['losses'] += 1
        if side.upper() == 'SHORT':
            ticker_stats[ticker]['shorts'] += 1
        else:
            ticker_stats[ticker]['longs'] += 1

        # Indicators at entry
        entry_indicators = trade.get('entry_indicators', {})
        rsi = entry_indicators.get('rsi_14', None)

        # Prediction probabilities
        prediction_proba = trade.get('prediction_proba', None)

        detailed_trades.append({
            'filename': trade['_filename'],
            'ticker': ticker,
            'side': side,
            'entry_time': entry_time_str,
            'hour': entry_dt.hour if entry_dt else None,
            'pnl_pct': pnl_pct,
            'rsi': rsi,
            'proba': prediction_proba,
            'entry_price': trade.get('entry_price', 0),
            'exit_price': trade.get('exit_price', 0),
            'exit_reason': trade.get('exit_reason', 'UNKNOWN')
        })

    # PODSUMOWANIE OGÓLNE
    print(f"\n📈 PODSUMOWANIE OGÓLNE:")
    print(f"  Total PnL: {total_pnl:.2f}%")
    print(f"  Win Rate: {wins}/{wins+losses} = {100*wins/(wins+losses):.1f}%")
    print(f"  Avg Win: {sum(t['pnl_pct'] for t in detailed_trades if t['pnl_pct'] > 0) / wins if wins > 0 else 0:.2f}%")
    print(f"  Avg Loss: {sum(t['pnl_pct'] for t in detailed_trades if t['pnl_pct'] < 0) / losses if losses > 0 else 0:.2f}%")
    print(f"  LONG/SHORT: {longs} / {shorts}")

    # ANALIZA PO TICKERACH
    print(f"\n🎯 ANALIZA PO TICKERACH:")
    for ticker, stats in sorted(ticker_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
        wr = 100 * stats['wins'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {ticker:12s}: PnL={stats['pnl']:+7.2f}% | Trades={stats['count']:3d} | WR={wr:5.1f}% | L/S={stats['longs']}/{stats['shorts']}")

    # ANALIZA PO GODZINACH
    print(f"\n⏰ ANALIZA PO GODZINACH (UTC):")
    for hour in sorted(hourly_stats.keys()):
        stats = hourly_stats[hour]
        wr = 100 * stats['wins'] / stats['count'] if stats['count'] > 0 else 0
        avg_pnl = stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {hour:02d}:00 - PnL={stats['pnl']:+7.2f}% | Trades={stats['count']:3d} | WR={wr:5.1f}% | Avg={avg_pnl:+6.2f}%")

    # TOP NAJGORSZE POZYCJE
    print(f"\n💥 TOP 10 NAJGORSZYCH POZYCJI:")
    worst_trades = sorted(detailed_trades, key=lambda x: x['pnl_pct'])[:10]
    for i, t in enumerate(worst_trades, 1):
        print(f"  {i:2d}. {t['filename']:40s} | {t['ticker']:8s} {t['side']:5s} | PnL={t['pnl_pct']:+6.2f}% | RSI={t['rsi']:5.1f} | Proba={t['proba']:.3f} | {t['exit_reason']}")

    # TOP NAJLEPSZE POZYCJE
    print(f"\n🎉 TOP 10 NAJLEPSZYCH POZYCJI:")
    best_trades = sorted(detailed_trades, key=lambda x: x['pnl_pct'], reverse=True)[:10]
    for i, t in enumerate(best_trades, 1):
        print(f"  {i:2d}. {t['filename']:40s} | {t['ticker']:8s} {t['side']:5s} | PnL={t['pnl_pct']:+6.2f}% | RSI={t['rsi']:5.1f} | Proba={t['proba']:.3f} | {t['exit_reason']}")

    # ANALIZA SHORTÓW vs LONGÓW
    print(f"\n📊 SHORT vs LONG:")
    short_trades = [t for t in detailed_trades if t['side'].upper() == 'SHORT']
    long_trades = [t for t in detailed_trades if t['side'].upper() == 'LONG']

    if short_trades:
        short_pnl = sum(t['pnl_pct'] for t in short_trades)
        short_wins = sum(1 for t in short_trades if t['pnl_pct'] > 0)
        short_wr = 100 * short_wins / len(short_trades)
        print(f"  SHORT: PnL={short_pnl:+7.2f}% | Trades={len(short_trades):3d} | WR={short_wr:5.1f}%")

    if long_trades:
        long_pnl = sum(t['pnl_pct'] for t in long_trades)
        long_wins = sum(1 for t in long_trades if t['pnl_pct'] > 0)
        long_wr = 100 * long_wins / len(long_trades)
        print(f"  LONG:  PnL={long_pnl:+7.2f}% | Trades={len(long_trades):3d} | WR={long_wr:5.1f}%")

    # ANALIZA RSI
    print(f"\n📉 ANALIZA RSI:")
    short_rsi_values = [t['rsi'] for t in short_trades if t['rsi'] is not None]
    long_rsi_values = [t['rsi'] for t in long_trades if t['rsi'] is not None]

    if short_rsi_values:
        print(f"  SHORT RSI - Avg={sum(short_rsi_values)/len(short_rsi_values):.1f} | Min={min(short_rsi_values):.1f} | Max={max(short_rsi_values):.1f}")
    if long_rsi_values:
        print(f"  LONG RSI  - Avg={sum(long_rsi_values)/len(long_rsi_values):.1f} | Min={min(long_rsi_values):.1f} | Max={max(long_rsi_values):.1f}")

    # ANALIZA PRAWDOPODOBIEŃSTW
    print(f"\n🎲 ANALIZA PRAWDOPODOBIEŃSTW PREDYKCJI:")
    winning_probas = [t['proba'] for t in detailed_trades if t['pnl_pct'] > 0 and t['proba'] is not None]
    losing_probas = [t['proba'] for t in detailed_trades if t['pnl_pct'] < 0 and t['proba'] is not None]

    if winning_probas:
        print(f"  WINNING trades - Avg Proba={sum(winning_probas)/len(winning_probas):.3f} | Min={min(winning_probas):.3f} | Max={max(winning_probas):.3f}")
    if losing_probas:
        print(f"  LOSING trades  - Avg Proba={sum(losing_probas)/len(losing_probas):.3f} | Min={min(losing_probas):.3f} | Max={max(losing_probas):.3f}")

    print("\n" + "=" * 80)

    # WNIOSKI I REKOMENDACJE
    print("\n💡 WNIOSKI I REKOMENDACJE:")

    # 1. Sprawdź czy shorty są problematyczne
    if short_trades and len(short_trades) > 3:
        short_pnl = sum(t['pnl_pct'] for t in short_trades)
        if short_pnl < -5:
            print(f"  ⚠️ SHORTY SĄ STRATNE (PnL={short_pnl:.2f}%) - rozważ wyłączenie SHORT modelu lub zwiększenie thresholdu")

    # 2. Sprawdź godziny
    losing_hours = [h for h, stats in hourly_stats.items() if stats['pnl'] < -2 and stats['count'] > 1]
    if losing_hours:
        print(f"  ⚠️ STRATNE GODZINY: {', '.join(f'{h:02d}:00' for h in sorted(losing_hours))} - rozważ blacklist tych godzin")

    # 3. Prawdopodobieństwa
    if losing_probas and winning_probas:
        avg_losing_proba = sum(losing_probas) / len(losing_probas)
        avg_winning_proba = sum(winning_probas) / len(winning_probas)
        if avg_losing_proba > 0.65:
            print(f"  ⚠️ TRACĄCE POZYCJE MAJĄ WYSOKIE PROBA ({avg_losing_proba:.3f}) - zwiększ threshold z 0.7 do 0.75-0.80")

    # 4. Win rate
    if wins + losses > 0:
        wr = 100 * wins / (wins + losses)
        if wr < 40:
            print(f"  ⚠️ WIN RATE PONIŻEJ 40% ({wr:.1f}%) - model wymaga retreningu lub wyższe thresholdy")

    # 5. Częstotliwość tradingu
    if len(trades) > 15:
        print(f"  ⚠️ DUŻO TRANSAKCJI ({len(trades)}) - rozważ zwiększenie min_proba_diff lub thresholdu aby filtrować słabe sygnały")

if __name__ == "__main__":
    analyze_trades()
