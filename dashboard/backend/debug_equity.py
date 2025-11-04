#!/usr/bin/env python3
"""
Debug script to check equity calculations from Bybit API
"""
import sys
import os

# Add price_action to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../price_action")))

from bybit_adapter import BybitAdapter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Initialize Bybit adapter
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    base_url = os.getenv("BYBIT_BASE_URL")

    if not api_key or not api_secret:
        print("❌ Missing BYBIT_API_KEY or BYBIT_API_SECRET in .env")
        return

    print(f"🔗 Connecting to Bybit API: {base_url}")
    adapter = BybitAdapter(api_key, api_secret, base_url if base_url != "https://api.bybit.com" else None)

    # 1. Get current wallet balance
    print("\n" + "="*60)
    print("1️⃣  CURRENT WALLET BALANCE")
    print("="*60)
    current_equity = adapter.get_balance(use_available=False)
    print(f"   Total Equity: ${current_equity:.2f} USDT")

    # 2. Get closed P&L history
    print("\n" + "="*60)
    print("2️⃣  CLOSED P&L HISTORY (last 100 trades)")
    print("="*60)
    pnl_records = adapter.get_closed_pnl_history(limit=100)

    if not pnl_records:
        print("   ❌ No closed P&L records found")
        return

    print(f"   Found {len(pnl_records)} closed trades")

    # Calculate total realized PnL
    total_realized_pnl = sum(float(r.get('closedPnl', 0)) for r in pnl_records)
    print(f"   Total Realized P&L (from history): ${total_realized_pnl:.2f}")

    # Show first and last few trades
    print("\n   📊 First 5 trades:")
    sorted_records = sorted(pnl_records, key=lambda x: int(x.get('createdTime', 0)))
    for i, record in enumerate(sorted_records[:5]):
        pnl = float(record.get('closedPnl', 0))
        symbol = record.get('symbol', 'N/A')
        side = record.get('side', 'N/A')
        from datetime import datetime
        ts = datetime.fromtimestamp(int(record.get('createdTime', 0)) / 1000)
        print(f"      {i+1}. {ts:%Y-%m-%d %H:%M} | {symbol:10s} | {side:5s} | P&L: ${pnl:+7.2f}")

    print("\n   📊 Last 5 trades:")
    for i, record in enumerate(sorted_records[-5:], start=len(sorted_records)-4):
        pnl = float(record.get('closedPnl', 0))
        symbol = record.get('symbol', 'N/A')
        side = record.get('side', 'N/A')
        from datetime import datetime
        ts = datetime.fromtimestamp(int(record.get('createdTime', 0)) / 1000)
        print(f"      {i}. {ts:%Y-%m-%d %H:%M} | {symbol:10s} | {side:5s} | P&L: ${pnl:+7.2f}")

    # 3. Calculate implied starting balance
    print("\n" + "="*60)
    print("3️⃣  EQUITY CURVE CALCULATION")
    print("="*60)

    # Work backwards to find starting equity
    implied_starting_equity = current_equity - total_realized_pnl

    print(f"   Current Equity:          ${current_equity:.2f}")
    print(f"   Total Realized P&L:      ${total_realized_pnl:+.2f}")
    print(f"   Implied Starting Equity: ${implied_starting_equity:.2f}")
    print(f"   ➜ Cumulative P&L:        ${current_equity - implied_starting_equity:+.2f}")

    # 4. Check for active positions (unrealized PnL)
    print("\n" + "="*60)
    print("4️⃣  ACTIVE POSITIONS (Unrealized P&L)")
    print("="*60)

    positions = adapter.get_all_positions()
    active_positions = [p for p in positions if abs(float(p.get('size', 0))) > 0]

    if active_positions:
        print(f"   Found {len(active_positions)} active position(s)")
        total_unrealized_pnl = 0.0
        for pos in active_positions:
            symbol = pos.get('symbol', 'N/A')
            side = pos.get('side', 'N/A')
            size = float(pos.get('size', 0))
            unrealized_pnl = float(pos.get('unrealisedPnl', 0))
            total_unrealized_pnl += unrealized_pnl
            print(f"      {symbol:10s} | {side:5s} | Size: {size:8.4f} | Unrealized P&L: ${unrealized_pnl:+7.2f}")

        print(f"\n   Total Unrealized P&L: ${total_unrealized_pnl:+.2f}")

        # IMPORTANT: If equity includes unrealized PnL, we need to subtract it
        print("\n   ⚠️  NOTE: Current equity may include unrealized P&L!")
        print(f"      If yes: Equity without unrealized = ${current_equity - total_unrealized_pnl:.2f}")
    else:
        print("   No active positions")

    # 5. Summary
    print("\n" + "="*60)
    print("5️⃣  SUMMARY")
    print("="*60)
    print(f"   Dashboard shows:  Final P&L = $8.12 (from screenshot)")
    print(f"   Calculated:       Cumulative P&L = ${total_realized_pnl:+.2f}")
    print("")
    if abs(total_realized_pnl - 8.12) > 0.01:
        print("   ⚠️  MISMATCH DETECTED!")
        print("   Possible causes:")
        print("      1. Unrealized P&L included in equity calculation")
        print("      2. Deposits/withdrawals not accounted for")
        print("      3. Funding fees not included in closed P&L")
        print("      4. Incomplete P&L history (only showing last 100 trades)")
    else:
        print("   ✅ Numbers match!")

if __name__ == "__main__":
    main()
