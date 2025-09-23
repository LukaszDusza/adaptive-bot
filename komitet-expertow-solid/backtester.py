import pandas as pd
from tqdm import tqdm
import config
import asyncio
import logging

from async_data_fetcher import fetch_data_for_trainer_async
from services.analysis_service import AnalysisService
from logic.position_manager import PositionManager
from logic.fees import get_fee_calculator
from utils.data_preparer import prepare_feature_set_for_timeframe
from utils.reporting import generate_full_report, save_events_log

# --- Fees (tak jak na giełdzie, w bps od nominału) ---
FEE_BPS_OPEN = getattr(config, "FEE_BPS_OPEN", 3.0)
FEE_BPS_CLOSE = getattr(config, "FEE_BPS_CLOSE", 3.0)


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
        print("Nie udało się pobrać danych. Zakończono.")
        logging.error("Pobieranie danych nie powiodło się. Pusty DataFrame.")
        return

    test_data = prepare_feature_set_for_timeframe(df_raw)

    capital = config.INITIAL_CAPITAL
    trades, equity_curve = [], {test_data.index[0]: capital}
    open_trade_fee = 0.0

    print("Uruchamianie symulacji...")
    logging.info(f"Rozpoczynanie symulacji dla {config.TICKER} od {config.START_DATE} do {config.END_DATE}")

    pbar = tqdm(test_data.iterrows(), total=len(test_data))
    for timestamp, current_candle in pbar:
        analysis = analyzer.get_analysis_from_row(current_candle)

        action, details = manager.process_candle(current_candle, analysis, capital)

        if action == 'OPEN':
            pos = details
            open_trade_fee = fee_calculator.calculate_exchange_fees(pos.entry_price * pos.size, FEE_BPS_OPEN)
            capital -= open_trade_fee

        elif action == 'CLOSE':
            closed_trade = details

            pnl_gross = closed_trade['pnl_usd']
            close_fee = fee_calculator.calculate_exchange_fees(closed_trade['exit_price'] * closed_trade['size'],
                                                               FEE_BPS_CLOSE)
            total_fee = open_trade_fee + close_fee
            pnl_net = pnl_gross - total_fee
            capital += pnl_gross
            capital -= close_fee
            trade_record = {**closed_trade, 'fees_usd': total_fee, 'pnl_net_usd': pnl_net}
            trades.append(trade_record)
            open_trade_fee = 0.0

        equity_curve[timestamp] = capital

        if config.DEBUG_MODE and analysis:
            confs = {e: f"{o['confidence']:.2f}" for e, o in analysis['expert_opinions'].items()}
            position_status = 'TAK' if manager.active_position else 'NIE'
            pbar.set_description(
                f"Kapitał: ${capital:,.2f} | Pozycja: {position_status} | Conf: {confs}")

    print("\nSymulacja zakończona. Generowanie raportów...")
    logging.info("Symulacja zakończona. Rozpoczęto generowanie raportów.")

    trades_df = pd.DataFrame(trades)
    generate_full_report(trades_df, equity_curve, capital, config, test_data)
    save_events_log(manager.events, config)

    logging.info("Raporty zostały wygenerowane pomyślnie.")


if __name__ == "__main__":
    logging.basicConfig(
        filename='backtest_run.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    try:
        asyncio.run(run())
    except Exception as e:
        logging.critical(f"Wystąpił nieobsługiwany błąd, który przerwał działanie backtestera: {e}", exc_info=True)
        print(f"\nKRYTYCZNY BŁĄD: {e}. Sprawdź plik backtest_run.log po szczegóły.")
