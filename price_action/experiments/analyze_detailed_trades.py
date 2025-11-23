#!/usr/bin/env python3
"""
Szczegółowa analiza transakcji z SL/TP i execution details
"""

import os
from datetime import datetime, timedelta
from bybit_adapter import BybitAdapter
from dotenv import load_dotenv
import pandas as pd

def analyze_detailed():
    # Wczytaj env
    load_dotenv('.env_luk')

    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    base_url = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")

    if not api_key or not api_secret:
        load_dotenv('.env_sylwia')
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")

        if not api_key or not api_secret:
            print("❌ Brak kluczy API")
            return

    adapter = BybitAdapter(api_key=api_key, api_secret=api_secret, base_url=base_url)

    # Dzisiaj
    today_start = datetime(2025, 11, 7, 0, 0, 0)
    today_end = datetime.now()
    start_ms = int(today_start.timestamp() * 1000)
    end_ms = int(today_end.timestamp() * 1000)

    # Wybierz ticker do szczegółowej analizy
    ticker = 'DOGEUSDT'  # Lub SOLUSDT

    print(f"\n🔍 SZCZEGÓŁOWA ANALIZA: {ticker}")
    print("=" * 120)

    # Pobierz closed PnL
    trades = adapter.get_closed_pnl_history(symbol=ticker, start_time_ms=start_ms, end_time_ms=end_ms, limit=100)

    # Pobierz executions (individual fills)
    executions = adapter.get_execution_list(symbol=ticker, start_time_ms=start_ms, end_time_ms=end_ms, limit=100)

    print(f"\nZnaleziono {len(trades)} zamkniętych pozycji i {len(executions)} executions\n")

    # Sortuj po czasie
    trades.sort(key=lambda x: int(x.get('createdTime', 0)))

    for i, trade in enumerate(trades, 1):
        side = trade.get('side', 'UNKNOWN')
        side_display = "🔴 SHORT" if side == 'Sell' else "🟢 LONG "

        avg_entry = float(trade.get('avgEntryPrice', 0))
        avg_exit = float(trade.get('avgExitPrice', 0))
        pnl = float(trade.get('closedPnl', 0))
        qty = float(trade.get('closedSize', 0))
        leverage = trade.get('leverage', '1')
        created_time = int(trade.get('createdTime', 0))

        dt = datetime.fromtimestamp(created_time / 1000)

        # PnL percentage
        if avg_entry > 0:
            if side == 'Buy':  # LONG
                pnl_pct = ((avg_exit - avg_entry) / avg_entry) * 100 * float(leverage)
                price_change_pct = ((avg_exit - avg_entry) / avg_entry) * 100
            else:  # SHORT
                pnl_pct = ((avg_entry - avg_exit) / avg_entry) * 100 * float(leverage)
                price_change_pct = ((avg_exit - avg_entry) / avg_entry) * 100
        else:
            pnl_pct = 0
            price_change_pct = 0

        # Oblicz gdzie były SL/TP (estimate based on typical bot settings)
        if side == 'Buy':  # LONG
            # TP powyżej entry, SL poniżej
            est_tp_distance_pct = 3.0  # typical tp_pct
            est_sl_distance_pct = -1.0  # typical sl distance
            est_tp = avg_entry * (1 + est_tp_distance_pct / 100)
            est_sl = avg_entry * (1 + est_sl_distance_pct / 100)

            # Check if hit SL or TP based on actual PnL (not price movement)
            if pnl < 0:
                exit_reason = "❌ SL (strata)"
            else:
                exit_reason = "✅ TP (zysk)"
        else:  # SHORT
            # TP poniżej entry, SL powyżej
            est_tp_distance_pct = -3.0
            est_sl_distance_pct = 1.0
            est_tp = avg_entry * (1 + est_tp_distance_pct / 100)
            est_sl = avg_entry * (1 + est_sl_distance_pct / 100)

            if pnl < 0:
                exit_reason = "❌ SL (strata)"
            else:
                exit_reason = "✅ TP (zysk)"

        print(f"\n{'=' * 120}")
        print(f"#{i} | {dt.strftime('%Y-%m-%d %H:%M:%S')} CET | {side_display}")
        print(f"{'=' * 120}")
        print(f"  Entry Price:    ${avg_entry:10.2f}")
        print(f"  Exit Price:     ${avg_exit:10.2f}")
        print(f"  Est. TP:        ${est_tp:10.2f}  (Target: {est_tp_distance_pct:+.1f}%)")
        print(f"  Est. SL:        ${est_sl:10.2f}  (Stop: {est_sl_distance_pct:+.1f}%)")
        print(f"  Price Change:   {price_change_pct:+6.2f}%")
        print(f"  PnL:            ${pnl:+7.2f}  ({pnl_pct:+6.2f}% leveraged)")
        print(f"  Quantity:       {qty}")
        print(f"  Leverage:       {leverage}x")
        print(f"  Exit Reason:    {exit_reason}")

    # ANALIZA CIĄGU POZYCJI
    print(f"\n\n{'=' * 120}")
    print(f"📊 ANALIZA CIĄGU POZYCJI - CZY BOT PONAWIA POZYCJE W TYM SAMYM KIERUNKU?")
    print(f"{'=' * 120}\n")

    consecutive_same_side = []
    prev_side = None
    streak = 0

    for trade in trades:
        side = trade.get('side', 'UNKNOWN')

        if side == prev_side:
            streak += 1
        else:
            if streak > 1:
                consecutive_same_side.append({
                    'side': 'SHORT' if prev_side == 'Sell' else 'LONG',
                    'streak': streak
                })
            streak = 1
            prev_side = side

    if streak > 1:
        consecutive_same_side.append({
            'side': 'SHORT' if prev_side == 'Sell' else 'LONG',
            'streak': streak
        })

    if consecutive_same_side:
        print("Znaleziono serie pozycji w tym samym kierunku:\n")
        for s in consecutive_same_side:
            print(f"  - {s['streak']} pozycji {s['side']} z rzędu")
        print("\n⚠️  BOT PONAWIA POZYCJE W TYM SAMYM KIERUNKU - to może być problem jeśli trend jest przeciwny!")
    else:
        print("✓ Brak powtarzających się serii - bot zmienia kierunek")

    # SPRAWDŹ CENĘ W CIĄGU DNIA
    print(f"\n\n{'=' * 120}")
    print(f"📈 SPRAWDZENIE TRENDU CENOWEGO W CIĄGU DNIA")
    print(f"{'=' * 120}\n")

    if trades:
        first_price = float(trades[0].get('avgEntryPrice', 0))
        last_price = float(trades[-1].get('avgExitPrice', 0))
        price_trend = ((last_price - first_price) / first_price) * 100

        print(f"  Pierwsza pozycja entry:  ${first_price:.2f}")
        print(f"  Ostatnia pozycja exit:   ${last_price:.2f}")
        print(f"  Zmiana ceny w ciągu dnia: {price_trend:+.2f}%")

        if price_trend > 2:
            print(f"\n  ⚠️  TREND WZROSTOWY ({price_trend:+.2f}%) - SHORTY będą tracić!")
            short_count = sum(1 for t in trades if t.get('side') == 'Sell')
            print(f"      Bot otworzył {short_count} SHORTów mimo wzrostowego trendu")
        elif price_trend < -2:
            print(f"\n  ⚠️  TREND SPADKOWY ({price_trend:+.2f}%) - LONGI będą tracić!")
            long_count = sum(1 for t in trades if t.get('side') == 'Buy')
            print(f"      Bot otworzył {long_count} LONGów mimo spadkowego trendu")

    print("\n" + "=" * 120)

if __name__ == "__main__":
    analyze_detailed()
