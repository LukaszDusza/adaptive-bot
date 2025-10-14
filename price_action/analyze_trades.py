"""
Trade Analysis Tool
Analyzes logged trades from the trading bot
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from trade_logger import TradeLogger, TradeVisualizer


def print_daily_summary():
    """Print daily trade summary"""
    print("\n" + "="*70)
    print(f"{'DAILY SUMMARY':^70}")
    print("="*70)
    
    analytics_dir = Path('logs/analytics')
    
    if not analytics_dir.exists():
        print("\n⚠ No analytics directory found.")
        return
    
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    summary_file = analytics_dir / f"daily_summary_{today}.json"
    
    if not summary_file.exists():
        print(f"\n⚠ No trades today ({today}).")
        return
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print(f"\n📅 Date: {summary['date']}")
    print(f"\n📊 Performance:")
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Wins: {summary['wins']} | Losses: {summary['losses']}")
    print(f"  Win Rate: {summary['win_rate']:.2f}%")
    print(f"  Total P&L: ${summary['total_pnl']:.2f}")
    
    print(f"\n📝 Individual Trades:")
    for trade in summary['trades']:
        pnl = trade['pnl']
        duration = trade['duration'] / 3600
        emoji = "✅" if pnl > 0 else "❌"
        print(f"  {emoji} {trade['trade_id']}")
        print(f"     {trade['side']} | P&L: ${pnl:+.2f} | Duration: {duration:.1f}h")


def analyze_tsl_effectiveness():
    """Analyze how effective trailing stop loss is"""
    print("\n" + "="*70)
    print(f"{'TSL EFFECTIVENESS ANALYSIS':^70}")
    print("="*70)
    
    logger = TradeLogger()
    trades = logger.get_trade_history(days=30)
    
    if not trades:
        print("\n⚠ No trades in last 30 days.")
        return
    
    total_trades = len(trades)
    trades_with_tsl = 0
    tsl_saved_from_loss = 0
    tsl_locked_profit = 0
    avg_tsl_updates = 0
    total_tsl_updates = 0
    
    for trade in trades:
        tsl_events = [e for e in trade['events'] if e['type'] == 'TSL_UPDATE']
        exit_event = [e for e in trade['events'] if e['type'] == 'EXIT']
        
        if tsl_events:
            trades_with_tsl += 1
            total_tsl_updates += len(tsl_events)
            
            # Check if TSL saved from bigger loss
            if exit_event:
                exit_data = exit_event[0]['data']
                trigger = exit_data.get('trigger', '')
                
                if trigger == 'SL' and len(tsl_events) > 0:
                    # TSL moved, then hit - saved from bigger loss
                    tsl_saved_from_loss += 1
                
                # Check if TSL locked in profit
                if len(tsl_events) > 0:
                    last_tsl = tsl_events[-1]['data']
                    entry_event = [e for e in trade['events'] if e['type'] == 'ENTRY'][0]
                    entry_price = entry_event['data']['entry_price']
                    final_sl = last_tsl['new_sl']
                    
                    # For LONG: final_sl > entry_price means profit locked
                    # For SHORT: final_sl < entry_price means profit locked
                    if trade['side'] == 'LONG' and final_sl > entry_price:
                        tsl_locked_profit += 1
                    elif trade['side'] == 'SHORT' and final_sl < entry_price:
                        tsl_locked_profit += 1
    
    if trades_with_tsl > 0:
        avg_tsl_updates = total_tsl_updates / trades_with_tsl
    
    print(f"\n📊 TSL Statistics (Last 30 Days):")
    print(f"  Total Trades: {total_trades}")
    print(f"  Trades with TSL Updates: {trades_with_tsl} ({trades_with_tsl/total_trades*100:.1f}%)")
    print(f"  Total TSL Updates: {total_tsl_updates}")
    print(f"  Average TSL Updates per Trade: {avg_tsl_updates:.1f}")
    
    print(f"\n💰 TSL Effectiveness:")
    print(f"  Saved from Bigger Loss: {tsl_saved_from_loss} trades")
    print(f"  Locked in Profit: {tsl_locked_profit} trades")
    
    if trades_with_tsl > 0:
        effectiveness = (tsl_saved_from_loss + tsl_locked_profit) / trades_with_tsl * 100
        print(f"  Overall Effectiveness: {effectiveness:.1f}%")


def analyze_partial_tp():
    """Analyze partial take profit performance"""
    print("\n" + "="*70)
    print(f"{'PARTIAL TP ANALYSIS':^70}")
    print("="*70)
    
    logger = TradeLogger()
    trades = logger.get_trade_history(days=30)
    
    if not trades:
        print("\n⚠ No trades in last 30 days.")
        return
    
    total_trades = len(trades)
    trades_with_partial = 0
    total_partial_pnl = 0
    avg_time_to_partial = []
    
    for trade in trades:
        partial_events = [e for e in trade['events'] if e['type'] == 'PARTIAL_TP']
        
        if partial_events:
            trades_with_partial += 1
            
            # Sum up partial TP P&L
            for event in partial_events:
                pnl = event['data'].get('pnl_usd', 0)
                total_partial_pnl += pnl
            
            # Calculate time to first partial TP
            entry_event = [e for e in trade['events'] if e['type'] == 'ENTRY'][0]
            entry_time = datetime.fromisoformat(entry_event['timestamp'])
            partial_time = datetime.fromisoformat(partial_events[0]['timestamp'])
            duration_hours = (partial_time - entry_time).total_seconds() / 3600
            avg_time_to_partial.append(duration_hours)
    
    print(f"\n📊 Partial TP Statistics (Last 30 Days):")
    print(f"  Total Trades: {total_trades}")
    print(f"  Trades with Partial TP: {trades_with_partial} ({trades_with_partial/total_trades*100:.1f}%)")
    print(f"  Total Partial TP P&L: ${total_partial_pnl:.2f}")
    
    if avg_time_to_partial:
        avg_hours = sum(avg_time_to_partial) / len(avg_time_to_partial)
        print(f"  Average Time to Partial TP: {avg_hours:.2f} hours")
        print(f"  Min Time to Partial TP: {min(avg_time_to_partial):.2f} hours")
        print(f"  Max Time to Partial TP: {max(avg_time_to_partial):.2f} hours")
    
    print(f"\n💡 Impact:")
    if trades_with_partial > 0:
        avg_partial_pnl = total_partial_pnl / trades_with_partial
        print(f"  Average Partial TP P&L: ${avg_partial_pnl:.2f}")
        print(f"  Contribution to Total P&L: ${total_partial_pnl:.2f}")


def show_best_worst_trades():
    """Show best and worst trades"""
    print("\n" + "="*70)
    print(f"{'BEST & WORST TRADES':^70}")
    print("="*70)
    
    logger = TradeLogger()
    trades = logger.get_trade_history(days=30)
    
    if not trades:
        print("\n⚠ No trades in last 30 days.")
        return
    
    # Sort by P&L
    trades_sorted = sorted(trades, key=lambda t: t['summary'].get('total_pnl_usd', 0))
    
    print(f"\n🏆 Top 3 Best Trades:")
    for i, trade in enumerate(trades_sorted[-3:][::-1], 1):
        pnl = trade['summary'].get('total_pnl_usd', 0)
        pnl_pct = trade['summary'].get('total_pnl_pct', 0)
        duration = trade['summary'].get('duration_seconds', 0) / 3600
        print(f"{i}. {trade['trade_id']}")
        print(f"   {trade['side']} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Duration: {duration:.1f}h")
    
    print(f"\n💀 Top 3 Worst Trades:")
    for i, trade in enumerate(trades_sorted[:3], 1):
        pnl = trade['summary'].get('total_pnl_usd', 0)
        pnl_pct = trade['summary'].get('total_pnl_pct', 0)
        duration = trade['summary'].get('duration_seconds', 0) / 3600
        print(f"{i}. {trade['trade_id']}")
        print(f"   {trade['side']} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Duration: {duration:.1f}h")


def export_all_trades_csv():
    """Export all trades to CSV"""
    print("\n" + "="*70)
    print(f"{'EXPORT TO CSV':^70}")
    print("="*70)
    
    trades_dir = Path('logs/trades')
    
    if not trades_dir.exists():
        print("\n⚠ No trades directory found.")
        return
    
    all_trades = []
    
    for trade_file in trades_dir.glob('*.json'):
        with open(trade_file) as f:
            trade = json.load(f)
        
        # Count events
        num_tsl = sum(1 for e in trade['events'] if e['type'] == 'TSL_UPDATE')
        has_partial = any(e['type'] == 'PARTIAL_TP' for e in trade['events'])
        num_errors = sum(1 for e in trade['events'] if e['type'] == 'ERROR')
        
        all_trades.append({
            'trade_id': trade['trade_id'],
            'ticker': trade['ticker'],
            'side': trade['side'],
            'start_time': trade['start_time'],
            'end_time': trade.get('end_time', 'N/A'),
            'duration_hours': trade['summary'].get('duration_seconds', 0) / 3600,
            'pnl_usd': trade['summary'].get('total_pnl_usd', 0),
            'pnl_pct': trade['summary'].get('total_pnl_pct', 0),
            'num_events': len(trade['events']),
            'num_tsl_updates': num_tsl,
            'partial_tp_taken': has_partial,
            'num_errors': num_errors
        })
    
    if not all_trades:
        print("\n⚠ No trades found.")
        return
    
    df = pd.DataFrame(all_trades)
    output_file = 'trade_analysis.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Exported {len(df)} trades to {output_file}")
    print(f"\nSummary Statistics:")
    print(f"  Total Trades: {len(df)}")
    print(f"  Total P&L: ${df['pnl_usd'].sum():.2f}")
    print(f"  Win Rate: {(df['pnl_usd'] > 0).mean() * 100:.2f}%")
    print(f"  Avg Duration: {df['duration_hours'].mean():.2f} hours")
    print(f"  Avg TSL Updates: {df['num_tsl_updates'].mean():.1f}")


def visualize_trade(trade_id: str):
    """Visualize specific trade"""
    print("\n" + "="*70)
    print(f"{'VISUALIZE TRADE':^70}")
    print("="*70)
    
    try:
        logger = TradeLogger()
        visualizer = TradeVisualizer(logger)
        
        print(f"\nReconstructing trade: {trade_id}")
        trade = visualizer.reconstruct_trade(trade_id)
        
        if not trade:
            print(f"\n❌ Trade not found: {trade_id}")
            return
        
        print(f"✓ Trade loaded")
        print(f"\nTrade Summary:")
        print(f"  Side: {trade['side']}")
        print(f"  Duration: {trade['summary'].get('duration_seconds', 0) / 3600:.2f} hours")
        print(f"  P&L: ${trade['summary'].get('total_pnl_usd', 0):.2f}")
        print(f"  Events: {len(trade['events'])}")
        
        # Create chart
        output_file = f"{trade_id}_chart.html"
        print(f"\nCreating chart: {output_file}")
        visualizer.create_chart(trade_id, output_file)
        
        print(f"\n✓ Chart saved: {output_file}")
        print(f"  Open in browser to view")
        
    except ImportError:
        print("\n⚠ Plotly not installed. Run: pip install plotly")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def show_indicator_analysis():
    """Analyze indicator values at entry"""
    print("\n" + "="*70)
    print(f"{'INDICATOR ANALYSIS':^70}")
    print("="*70)
    
    logger = TradeLogger()
    trades = logger.get_trade_history(days=7)
    
    if not trades:
        print("\n⚠ No trades in last 7 days.")
        return
    
    # Collect winning and losing trade indicators
    winning_indicators = []
    losing_indicators = []
    
    for trade in trades:
        if 'indicators' in trade and 'entry' in trade['indicators']:
            indicators = trade['indicators']['entry'].get('indicators', {})
            pnl = trade['summary'].get('total_pnl_usd', 0)
            
            if pnl > 0:
                winning_indicators.append(indicators)
            else:
                losing_indicators.append(indicators)
    
    if not winning_indicators and not losing_indicators:
        print("\n⚠ No indicator data found.")
        return
    
    print(f"\nTrades analyzed:")
    print(f"  Winners: {len(winning_indicators)}")
    print(f"  Losers: {len(losing_indicators)}")
    
    if winning_indicators:
        df_win = pd.DataFrame(winning_indicators)
        print(f"\n{'Winning Trades - Average Indicators:':^70}")
        print(df_win.mean().to_string())
    
    if losing_indicators:
        df_lose = pd.DataFrame(losing_indicators)
        print(f"\n{'Losing Trades - Average Indicators:':^70}")
        print(df_lose.mean().to_string())


def main_menu():
    """Interactive menu"""
    while True:
        print("\n" + "="*70)
        print(f"{'TRADE ANALYSIS MENU':^70}")
        print("="*70)
        print("\n1. Daily Summary")
        print("2. TSL Effectiveness")
        print("3. Partial TP Analysis")
        print("4. Best & Worst Trades")
        print("5. Export to CSV")
        print("6. Visualize Trade (requires trade_id)")
        print("7. Indicator Analysis")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == '1':
            print_daily_summary()
        elif choice == '2':
            analyze_tsl_effectiveness()
        elif choice == '3':
            analyze_partial_tp()
        elif choice == '4':
            show_best_worst_trades()
        elif choice == '5':
            export_all_trades_csv()
        elif choice == '6':
            trade_id = input("\nEnter trade_id: ").strip()
            if trade_id:
                visualize_trade(trade_id)
            else:
                print("❌ Invalid trade_id")
        elif choice == '7':
            show_indicator_analysis()
        elif choice == '8':
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid option")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(f"{'TRADE ANALYSIS TOOL':^70}")
    print("="*70)
    print("\nThis tool helps you analyze logged trades from bot.py")
    
    # Check if logs directory exists
    if not Path('logs').exists():
        print("\n⚠ Warning: logs/ directory not found")
        print("Make sure you're running this from the correct directory")
        print("and that the bot has been running and logging trades.")
        exit(1)
    
    main_menu()
