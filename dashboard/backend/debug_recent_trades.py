"""
Debug script to check recent trades from Bybit API
"""
import sys
import os
import json
from datetime import datetime

# Add price_action to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../price_action")))

from bybit_adapter import BybitAdapter

# Load credentials
api_key = "AQuYHxou38tOrfx5Lw"
api_secret = "7R63fmKf2cttDin5rfQRecawJVf86reoQYX5"

print("🔍 Debugging Recent Trades from Bybit API")
print("=" * 70)

# Initialize adapter
adapter = BybitAdapter(api_key=api_key, api_secret=api_secret)

# Get closed P&L history
print("\n📊 Fetching closed P&L history (last 50 trades)...")
pnl_records = adapter.get_closed_pnl_history(limit=50)

print(f"✓ Retrieved {len(pnl_records)} closed P&L records\n")

# Filter DOGEUSDT trades
doge_trades = [r for r in pnl_records if r.get('symbol') == 'DOGEUSDT']

print(f"🐕 Found {len(doge_trades)} DOGEUSDT trades:\n")

for idx, record in enumerate(doge_trades[:2], 1):  # Show first 2
    # Parse data
    created_time = int(record.get('createdTime', 0))
    updated_time = int(record.get('updatedTime', 0))

    created_dt = datetime.fromtimestamp(created_time / 1000)
    updated_dt = datetime.fromtimestamp(updated_time / 1000)

    closed_pnl = float(record.get('closedPnl', 0))
    avg_entry_price = float(record.get('avgEntryPrice', 0))
    avg_exit_price = float(record.get('avgExitPrice', 0))
    closed_size = float(record.get('closedSize', 0))
    side = record.get('side', 'Unknown')

    # Calculate PnL percentage (same as backend)
    pnl_percent = 0.0
    if avg_entry_price > 0 and closed_size > 0:
        position_value = avg_entry_price * closed_size
        pnl_percent = (closed_pnl / position_value) * 100

    # Determine exit reason
    if closed_pnl > 0:
        exit_reason = "TP"
    elif closed_pnl < 0:
        exit_reason = "SL"
    else:
        exit_reason = "Manual"

    print(f"#{idx} {record.get('symbol')} {side}")
    print(f"   Created:  {created_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Updated:  {updated_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Entry:    ${avg_entry_price:.4f}")
    print(f"   Exit:     ${avg_exit_price:.4f}")
    print(f"   Size:     {closed_size:.2f}")
    print(f"   PnL:      ${closed_pnl:.2f}")
    print(f"   PnL %:    {pnl_percent:+.2f}%")
    print(f"   Exit:     {exit_reason}")
    print(f"   RAW RECORD:")
    print(json.dumps(record, indent=2))

    # Calculate expected PnL for LONG position
    if side == 'Buy':
        expected_pnl = (avg_exit_price - avg_entry_price) * closed_size
        print(f"   Expected: ${expected_pnl:.2f} (if no fees)")
        print(f"   Diff:     ${closed_pnl - expected_pnl:.2f} (fees)")

    print()

print("\n" + "=" * 70)
print("🔍 Look for the trade matching your case:")
print("   DOGEUSDT LONG, Entry ~$0.1656, Exit ~$0.1649")
print("   Nov 4, 06:31 AM")
