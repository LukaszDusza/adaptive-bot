#!/usr/bin/env python3
"""Analyze live trading performance from logs."""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def analyze_trade_log(log_file: Path):
    """Extract key metrics from a trade log."""
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)

        # Extract basic info
        ticker = data.get('ticker', 'UNKNOWN')
        side = data.get('side', 'UNKNOWN')
        entry_time = data.get('entry_time', '')
        exit_time = data.get('exit_time', '')

        # Entry details
        entry_price = float(data.get('entry_price', 0))
        quantity = float(data.get('quantity', 0))

        # Exit details
        exit_reason = data.get('exit_reason', 'UNKNOWN')
        exit_price = float(data.get('exit_price', 0))

        # PnL
        pnl_usdt = float(data.get('pnl_usdt', 0))
        pnl_pct = float(data.get('pnl_pct', 0))

        # Prediction details
        long_proba = data.get('long_proba', 0)
        short_proba = data.get('short_proba', 0)

        return {
            'log_file': log_file.name,
            'ticker': ticker,
            'side': side,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl_usdt': pnl_usdt,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'long_proba': long_proba,
            'short_proba': short_proba,
        }
    except Exception as e:
        print(f"Error processing {log_file.name}: {e}")
        return None

def main():
    logs_dir = Path("logs/trades")

    if not logs_dir.exists():
        print("❌ No logs/trades directory found")
        return

    # Load all trade logs
    trade_files = sorted(logs_dir.glob("*.json"))

    if not trade_files:
        print("❌ No trade logs found")
        return

    print(f"\n📊 Analyzing {len(trade_files)} trades...")

    trades = []
    for trade_file in trade_files:
        trade_data = analyze_trade_log(trade_file)
        if trade_data:
            trades.append(trade_data)

    df = pd.DataFrame(trades)

    # Overall stats
    print("\n" + "="*100)
    print("🔍 OVERALL LIVE TRADING STATISTICS")
    print("="*100)

    total_trades = len(df)
    winning_trades = len(df[df['pnl_usdt'] > 0])
    losing_trades = len(df[df['pnl_usdt'] < 0])
    breakeven_trades = len(df[df['pnl_usdt'] == 0])

    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    total_pnl = df['pnl_usdt'].sum()
    avg_win = df[df['pnl_usdt'] > 0]['pnl_usdt'].mean() if winning_trades > 0 else 0
    avg_loss = df[df['pnl_usdt'] < 0]['pnl_usdt'].mean() if losing_trades > 0 else 0

    print(f"\nTotal Trades: {total_trades}")
    print(f"Winning: {winning_trades} ({win_rate:.1f}%)")
    print(f"Losing: {losing_trades} ({(losing_trades/total_trades*100):.1f}%)")
    print(f"Breakeven: {breakeven_trades}")
    print(f"\nTotal PnL: ${total_pnl:.2f}")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "N/A")

    # By ticker
    print("\n" + "="*100)
    print("📈 PERFORMANCE BY TICKER")
    print("="*100)

    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker]
        ticker_trades = len(ticker_df)
        ticker_wins = len(ticker_df[ticker_df['pnl_usdt'] > 0])
        ticker_wr = ticker_wins / ticker_trades * 100 if ticker_trades > 0 else 0
        ticker_pnl = ticker_df['pnl_usdt'].sum()

        print(f"\n{ticker}:")
        print(f"  Trades: {ticker_trades}, Win Rate: {ticker_wr:.1f}%, Total PnL: ${ticker_pnl:.2f}")

        # By side
        for side in ['LONG', 'SHORT']:
            side_df = ticker_df[ticker_df['side'] == side]
            if len(side_df) > 0:
                side_wins = len(side_df[side_df['pnl_usdt'] > 0])
                side_wr = side_wins / len(side_df) * 100
                side_pnl = side_df['pnl_usdt'].sum()
                print(f"    {side:5s}: {len(side_df):2d} trades, WR: {side_wr:.1f}%, PnL: ${side_pnl:.2f}")

    # Exit reasons
    print("\n" + "="*100)
    print("🚪 EXIT REASONS")
    print("="*100)

    exit_counts = df['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        pct = count / total_trades * 100
        reason_pnl = df[df['exit_reason'] == reason]['pnl_usdt'].sum()
        print(f"{reason:20s}: {count:3d} ({pct:5.1f}%) - Total PnL: ${reason_pnl:.2f}")

    # Recent trades
    print("\n" + "="*100)
    print("📋 LAST 10 TRADES")
    print("="*100)

    recent_df = df.sort_values('entry_time', ascending=False).head(10)
    for _, trade in recent_df.iterrows():
        print(f"{trade['entry_time'][:10]} {trade['ticker']:8s} {trade['side']:5s} "
              f"${trade['pnl_usdt']:7.2f} ({trade['pnl_pct']:+6.2f}%) - {trade['exit_reason']}")

    # Save full data
    output_file = "live_trades_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Full data saved to {output_file}")

if __name__ == "__main__":
    main()
