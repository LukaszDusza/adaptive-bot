import pandas as pd
from tqdm import tqdm
import config
import asyncio

from async_data_fetcher import fetch_data_for_trainer_async
from services.analysis_service import AnalysisService
from logic.position_manager import PositionManager
from logic.fees import get_fee_calculator
from utils.data_preparer import prepare_full_feature_set
from utils.reporting import generate_full_report, save_events_log

# --- Fees (tak jak na giełdzie, w bps od nominału) ---
FEE_BPS_OPEN  = getattr(config, "FEE_BPS_OPEN", 3.0)   # np. 0.03%
FEE_BPS_CLOSE = getattr(config, "FEE_BPS_CLOSE", 3.0)   # np. 0.03%

# --- Fees (round-trip) tylko dla backtestu ---
from config import TRADE_COST_USD as ROUND_TRIP_COST
OPEN_COST  = ROUND_TRIP_COST * 0.5
CLOSE_COST = ROUND_TRIP_COST * 0.5

async def run():
    analyzer = AnalysisService(config.TICKER_NAME_FOR_MODELS)
    manager = PositionManager(config)
    fee_calculator = get_fee_calculator(config)

    df_raw = await fetch_data_for_trainer_async(
        ticker=config.TICKER,
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )

    if df_raw is None or df_raw.empty:
        return print("Nie udało się pobrać danych. Zakończono.")

    test_data = prepare_full_feature_set(df_raw)

    capital = config.INITIAL_CAPITAL
    trades, equity_curve = [], {test_data.index[0]: capital}

    print("Uruchamianie symulacji...")
    pbar = tqdm(test_data.iterrows(), total=len(test_data))
    for timestamp, current_candle in pbar:
        analysis = analyzer.get_analysis_from_row(current_candle)
        
        # Use the same API as live_trader for consistent position management
        signal = manager.get_trading_signal(current_candle, analysis, capital)
        
        if signal['action'] in ['OPEN_LONG', 'OPEN_SHORT']:
            # Simulate position opening - create position in manager
            position_data = {
                'strategy': signal['strategy'],
                'entry_date': timestamp,
                'entry_price': signal['entry_price'],
                'size': signal['size'],
                'current_sl_price': signal['stop_loss'],
                'tp_price': signal['take_profit'],
                'breakeven_trigger_price': signal['breakeven_trigger'],
                'trailing_trigger_price': signal['trailing_trigger'],
                'conf_momentum': signal['confidence']['momentum'],
                'conf_reversion': signal['confidence']['reversion'],
                'conf_pa': signal['confidence']['pa']
            }
            manager.update_position_from_live_data(position_data)
            
            # Calculate opening fee
            entry_price = signal['entry_price']
            size = signal['size']
            open_fee = fee_calculator.calculate_exchange_fees(entry_price * size, FEE_BPS_OPEN)
            capital -= open_fee

        elif signal['action'] == 'CLOSE':
            # Get position details before closing
            position_status = manager.get_position_status()
            if position_status['has_position']:
                pos = position_status['position']
                
                # Apply slippage to exit price (just like in real trading)
                raw_exit_price = signal['exit_price']
                exit_price = fee_calculator.apply_slippage(
                    exit_reason=signal['exit_reason'],
                    strategy=pos['strategy'],
                    raw_price=raw_exit_price,
                    candle=current_candle
                )
                
                # Calculate PnL using fees.py
                pnl_gross = fee_calculator.calculate_pnl(
                    strategy=pos['strategy'],
                    entry_price=pos['entry_price'],
                    exit_price=exit_price,
                    size=pos['size']
                )
                
                # Calculate fees
                close_fee = fee_calculator.calculate_exchange_fees(exit_price * pos['size'], FEE_BPS_CLOSE)
                open_fee = fee_calculator.calculate_exchange_fees(pos['entry_price'] * pos['size'], FEE_BPS_OPEN)
                total_fee = open_fee + close_fee
                
                # Net PnL after fees
                pnl_net = pnl_gross - total_fee
                capital += pnl_gross  # Add gross PnL
                capital -= close_fee  # Subtract closing fee
                
                # Create trade record
                trade_record = {
                    'entry_date': pos['entry_date'],
                    'exit_date': timestamp,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'size': pos['size'],
                    'pnl_usd': pnl_gross,
                    'exit_reason': signal['exit_reason'],
                    'strategy': pos['strategy'],
                    'conf_momentum': pos['confidence_levels']['momentum'],
                    'conf_reversion': pos['confidence_levels']['reversion'],
                    'conf_pa': pos['confidence_levels']['pa'],
                    'fees_usd': total_fee,
                    'pnl_net_usd': pnl_net
                }
                trades.append(trade_record)
                
                # Clear position from manager
                manager.clear_position()
        
        # Process position management instructions (BE, TSL, etc.)
        for instruction in signal.get('instructions', []):
            if instruction['type'] == 'MOVE_SL_TO_BREAKEVEN':
                # In backtest, this is handled automatically by position_manager
                pass
            elif instruction['type'] in ['ACTIVATE_TRAILING_STOP', 'UPDATE_TRAILING_STOP']:
                # In backtest, this is handled automatically by position_manager
                pass

        equity_curve[timestamp] = capital

        if config.DEBUG_MODE and analysis:
            confs = {e: f"{o['confidence']:.2f}" for e, o in analysis['expert_opinions'].items()}
            pbar.set_description(
                f"Kapitał: ${capital:,.2f} | Pozycja: {'TAK' if manager.active_position else 'NIE'} | Conf: {confs}")

    print("\nSymulacja zakończona. Generowanie raportów...")
    trades_df = pd.DataFrame(trades)
    generate_full_report(trades_df, equity_curve, capital, config, test_data)
    save_events_log(manager.events, config)


if __name__ == "__main__":
    asyncio.run(run())