"""
Debug script to check Bybit closedPnl sign convention
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.bybit_service import BybitService

# Initialize service
service = BybitService()

# Get recent closed P&L records
print("=" * 80)
print("CHECKING BYBIT CLOSED PNL SIGN CONVENTION")
print("=" * 80)

pnl_records = service.get_closed_pnl_history(limit=20)

print(f"\nFound {len(pnl_records)} recent trades\n")

for i, record in enumerate(pnl_records[:10], 1):
    symbol = record.get('symbol', 'N/A')
    side = record.get('side', 'N/A')
    closed_pnl = float(record.get('closedPnl', 0))
    avg_entry = float(record.get('avgEntryPrice', 0))
    avg_exit = float(record.get('avgExitPrice', 0))

    print(f"Trade {i}:")
    print(f"  Symbol: {symbol}")
    print(f"  Side: {side}")
    print(f"  Entry: ${avg_entry:.4f}")
    print(f"  Exit: ${avg_exit:.4f}")
    print(f"  closedPnl: ${closed_pnl:.6f}")

    # Calculate expected PnL direction
    if side == 'Buy':
        expected_profit = avg_exit > avg_entry
    else:
        expected_profit = avg_exit < avg_entry

    print(f"  Expected: {'PROFIT' if expected_profit else 'LOSS'}")
    print(f"  Actual sign: {'POSITIVE' if closed_pnl > 0 else 'NEGATIVE'}")

    # Check if sign matches expectation
    matches = (expected_profit and closed_pnl > 0) or (not expected_profit and closed_pnl < 0)
    print(f"  ✓ MATCH" if matches else "  ✗ MISMATCH!")
    print()

# Calculate total
total_pnl = sum([float(r.get('closedPnl', 0)) for r in pnl_records])
print(f"\nSUM of last {len(pnl_records)} closedPnl values: ${total_pnl:.4f}")
print("=" * 80)
