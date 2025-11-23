#!/usr/bin/env python3
"""
Analiza dzisiejszych pozycji bezpośrednio z Bybit API
"""

import os
from datetime import datetime, timedelta
from bybit_adapter import BybitAdapter
from dotenv import load_dotenv
from collections import defaultdict
import pandas as pd

def analyze_today_trades():
    # Wczytaj env
    load_dotenv('.env_luk')  # lub .env_sylwia

    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    base_url = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")

    if not api_key or not api_secret:
        print("❌ Brak kluczy API w .env_luk")
        # Spróbuj z .env_sylwia
        load_dotenv('.env_sylwia')
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")

        if not api_key or not api_secret:
            print("❌ Brak kluczy API w .env_sylwia")
            return

    adapter = BybitAdapter(api_key=api_key, api_secret=api_secret, base_url=base_url)

    # Dzisiaj: 7 listopada 2025
    today_start = datetime(2025, 11, 7, 0, 0, 0)
    today_end = datetime.now()

    start_ms = int(today_start.timestamp() * 1000)
    end_ms = int(today_end.timestamp() * 1000)

    print(f"\n🔍 Pobieram pozycje z {today_start.strftime('%Y-%m-%d %H:%M')} do {today_end.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 100)

    # Pobierz wszystkie closed PnL
    all_trades = []

    # Pobierz dla wszystkich symboli
    symbols = ['SOLUSDT', 'DOGEUSDT', 'ETHUSDT', 'LINKUSDT', 'PEPEUSDT']

    for symbol in symbols:
        trades = adapter.get_closed_pnl_history(symbol=symbol, start_time_ms=start_ms, end_time_ms=end_ms, limit=100)
        all_trades.extend(trades)
        print(f"  {symbol}: {len(trades)} pozycji")

    if not all_trades:
        print("\n❌ Brak pozycji z dzisiaj!")
        return

    print(f"\n📊 ZNALEZIONO {len(all_trades)} POZYCJI Z DZISIAJ")
    print("=" * 100)

    # Analiza
    total_pnl = 0
    wins = 0
    losses = 0
    shorts = 0
    longs = 0

    # Statystyki po godzinach
    hourly_stats = defaultdict(lambda: {'pnl': 0, 'count': 0, 'wins': 0, 'losses': 0})

    # Statystyki po tickerach
    ticker_stats = defaultdict(lambda: {'pnl': 0, 'count': 0, 'wins': 0, 'losses': 0, 'shorts': 0, 'longs': 0})

    # Szczegółowe dane
    detailed_trades = []

    for trade in all_trades:
        symbol = trade.get('symbol', 'UNKNOWN')
        side = trade.get('side', 'UNKNOWN')  # Buy = LONG, Sell = SHORT
        pnl = float(trade.get('closedPnl', 0))
        avg_entry = float(trade.get('avgEntryPrice', 0))
        avg_exit = float(trade.get('avgExitPrice', 0))
        qty = float(trade.get('closedSize', 0))
        leverage = trade.get('leverage', '1')
        created_time = int(trade.get('createdTime', 0))

        # Timestamp to datetime (UTC)
        dt = datetime.fromtimestamp(created_time / 1000)

        # PnL percentage (approximate)
        if avg_entry > 0:
            if side == 'Buy':  # LONG
                pnl_pct = ((avg_exit - avg_entry) / avg_entry) * 100 * float(leverage)
            else:  # SHORT
                pnl_pct = ((avg_entry - avg_exit) / avg_entry) * 100 * float(leverage)
        else:
            pnl_pct = 0

        total_pnl += pnl

        if pnl > 0:
            wins += 1
        else:
            losses += 1

        if side == 'Sell':  # SHORT
            shorts += 1
        else:
            longs += 1

        # Hourly stats
        hour = dt.hour
        hourly_stats[hour]['pnl'] += pnl
        hourly_stats[hour]['count'] += 1
        if pnl > 0:
            hourly_stats[hour]['wins'] += 1
        else:
            hourly_stats[hour]['losses'] += 1

        # Ticker stats
        ticker_stats[symbol]['pnl'] += pnl
        ticker_stats[symbol]['count'] += 1
        if pnl > 0:
            ticker_stats[symbol]['wins'] += 1
        else:
            ticker_stats[symbol]['losses'] += 1
        if side == 'Sell':
            ticker_stats[symbol]['shorts'] += 1
        else:
            ticker_stats[symbol]['longs'] += 1

        detailed_trades.append({
            'time': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'hour': hour,
            'symbol': symbol,
            'side': 'SHORT' if side == 'Sell' else 'LONG',
            'entry': avg_entry,
            'exit': avg_exit,
            'qty': qty,
            'leverage': leverage,
            'pnl_usd': pnl,
            'pnl_pct': pnl_pct
        })

    # PODSUMOWANIE
    print(f"\n💰 PODSUMOWANIE:")
    print(f"  Total PnL: ${total_pnl:.2f}")
    print(f"  Win Rate: {wins}/{wins+losses} = {100*wins/(wins+losses):.1f}%")
    if wins > 0:
        avg_win = sum(t['pnl_usd'] for t in detailed_trades if t['pnl_usd'] > 0) / wins
        print(f"  Avg Win: ${avg_win:.2f}")
    if losses > 0:
        avg_loss = sum(t['pnl_usd'] for t in detailed_trades if t['pnl_usd'] < 0) / losses
        print(f"  Avg Loss: ${avg_loss:.2f}")
    print(f"  LONG/SHORT: {longs} / {shorts}")

    # ANALIZA PO TICKERACH
    print(f"\n🎯 ANALIZA PO TICKERACH:")
    for ticker, stats in sorted(ticker_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
        wr = 100 * stats['wins'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {ticker:12s}: PnL=${stats['pnl']:+8.2f} | Trades={stats['count']:3d} | WR={wr:5.1f}% | L/S={stats['longs']}/{stats['shorts']}")

    # ANALIZA PO GODZINACH
    print(f"\n⏰ ANALIZA PO GODZINACH (UTC+1 CET):")
    for hour in sorted(hourly_stats.keys()):
        stats = hourly_stats[hour]
        wr = 100 * stats['wins'] / stats['count'] if stats['count'] > 0 else 0
        avg_pnl = stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
        # Convert UTC to CET (UTC+1)
        cet_hour = (hour + 1) % 24
        print(f"  {cet_hour:02d}:00 CET | PnL=${stats['pnl']:+8.2f} | Trades={stats['count']:3d} | WR={wr:5.1f}% | Avg=${avg_pnl:+7.2f}")

    # TOP NAJGORSZE
    print(f"\n💥 TOP 15 NAJGORSZYCH POZYCJI:")
    worst = sorted(detailed_trades, key=lambda x: x['pnl_usd'])[:15]
    for i, t in enumerate(worst, 1):
        direction = "🔴 SHORT" if t['side'] == 'SHORT' else "🟢 LONG "
        print(f"  {i:2d}. {t['time']} | {t['symbol']:8s} {direction} | Entry=${t['entry']:8.2f} Exit=${t['exit']:8.2f} | PnL=${t['pnl_usd']:+7.2f} ({t['pnl_pct']:+6.2f}%)")

    # TOP NAJLEPSZE
    print(f"\n🎉 TOP 10 NAJLEPSZYCH POZYCJI:")
    best = sorted(detailed_trades, key=lambda x: x['pnl_usd'], reverse=True)[:10]
    for i, t in enumerate(best, 1):
        direction = "🔴 SHORT" if t['side'] == 'SHORT' else "🟢 LONG "
        print(f"  {i:2d}. {t['time']} | {t['symbol']:8s} {direction} | Entry=${t['entry']:8.2f} Exit=${t['exit']:8.2f} | PnL=${t['pnl_usd']:+7.2f} ({t['pnl_pct']:+6.2f}%)")

    # ANALIZA SHORTÓW vs LONGÓW
    print(f"\n📊 SHORT vs LONG:")
    short_trades = [t for t in detailed_trades if t['side'] == 'SHORT']
    long_trades = [t for t in detailed_trades if t['side'] == 'LONG']

    if short_trades:
        short_pnl = sum(t['pnl_usd'] for t in short_trades)
        short_wins = sum(1 for t in short_trades if t['pnl_usd'] > 0)
        short_wr = 100 * short_wins / len(short_trades)
        print(f"  SHORT: PnL=${short_pnl:+8.2f} | Trades={len(short_trades):3d} | WR={short_wr:5.1f}%")

    if long_trades:
        long_pnl = sum(t['pnl_usd'] for t in long_trades)
        long_wins = sum(1 for t in long_trades if t['pnl_usd'] > 0)
        long_wr = 100 * long_wins / len(long_trades)
        print(f"  LONG:  PnL=${long_pnl:+8.2f} | Trades={len(long_trades):3d} | WR={long_wr:5.1f}%")

    # SERIA STRAT
    print(f"\n📉 ANALIZA SERII STRAT:")
    max_losing_streak = 0
    current_streak = 0
    for t in sorted(detailed_trades, key=lambda x: x['time']):
        if t['pnl_usd'] < 0:
            current_streak += 1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:
            current_streak = 0
    print(f"  Najdłuższa seria strat: {max_losing_streak}")

    # KONKLUZJE
    print(f"\n" + "=" * 100)
    print(f"\n💡 WNIOSKI I REKOMENDACJE:")

    # 1. Ogólny PnL
    if total_pnl < -10:
        print(f"  ⚠️  DUŻY DRAWDOWN (${total_pnl:.2f}) - bot wymaga natychmiastowej interwencji")

    # 2. Shorty vs Longi
    if short_trades and len(short_trades) > 3:
        short_pnl = sum(t['pnl_usd'] for t in short_trades)
        if short_pnl < -5:
            print(f"  ⚠️  SHORTY SĄ STRATNE (${short_pnl:.2f}) - prawdopodobnie trend wzrostowy, wyłącz SHORT model")

    # 3. Win rate
    if wins + losses > 0:
        wr = 100 * wins / (wins + losses)
        if wr < 35:
            print(f"  ⚠️  WIN RATE BARDZO NISKI ({wr:.1f}%) - zwiększ threshold z 0.7 do 0.80-0.85")
        elif wr < 45:
            print(f"  ⚠️  WIN RATE NISKI ({wr:.1f}%) - zwiększ threshold z 0.7 do 0.75")

    # 4. Overtrading
    if len(detailed_trades) > 20:
        print(f"  ⚠️  OVERTRADING ({len(detailed_trades)} pozycji) - zwiększ min_proba_diff lub threshold")

    # 5. Stratne godziny
    losing_hours = []
    for hour, stats in hourly_stats.items():
        if stats['pnl'] < -3 and stats['count'] > 2:
            cet_hour = (hour + 1) % 24
            losing_hours.append(cet_hour)

    if losing_hours:
        print(f"  ⚠️  STRATNE GODZINY (CET): {', '.join(f'{h:02d}:00' for h in sorted(losing_hours))} - rozważ blacklist")

    # 6. Max losing streak
    if max_losing_streak > 5:
        print(f"  ⚠️  SERIA {max_losing_streak} STRAT Z RZĘDU - model nie radzi sobie w obecnych warunkach rynkowych")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    analyze_today_trades()
