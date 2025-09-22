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
        decision, details = manager.process_candle(current_candle, analysis, capital)

        if decision == 'OPEN':
            # details to obiekt Position; pobieramy nominał wejścia
            entry_price = details.entry_price if hasattr(details, "entry_price") else details["entry_price"]
            size = details.size if hasattr(details, "size") else details["size"]
            open_fee = fee_calculator.calculate_exchange_fees(entry_price * size, FEE_BPS_OPEN)
            capital -= open_fee

        elif decision == 'CLOSE':
            exit_price = details['exit_price']
            size = details['size']

            close_fee = fee_calculator.calculate_exchange_fees(exit_price * size, FEE_BPS_CLOSE)
            capital += details['pnl_usd']  # brutto PnL
            capital -= close_fee  # koszt zamknięcia

            # policz łączne fee (open przeliczamy deterministycznie z danych wejścia)
            open_fee = fee_calculator.calculate_exchange_fees(details['entry_price'] * size, FEE_BPS_OPEN)
            total_fee = open_fee + close_fee
            details['fees_usd'] = total_fee
            details['pnl_net_usd'] = details['pnl_usd'] - total_fee

            trades.append(details)

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