import pandas as pd
from tqdm import tqdm
import config
import asyncio

from utils.async_data_fetcher import fetch_data_for_trainer_async
from services.analysis_service import AnalysisService
from logic.position_manager import PositionManager
from utils.data_preparer import prepare_full_feature_set
from utils.reporting import generate_full_report, save_events_log

async def run():
    analyzer = AnalysisService(config.TICKER_NAME_FOR_MODELS)
    manager = PositionManager(config)

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
            capital -= config.TRADE_COST_USD
        elif decision == 'CLOSE':
            capital += details['pnl_usd']
            trades.append(details)
        equity_curve[timestamp] = capital
        if config.DEBUG_MODE and analysis:
            confs = {e: f"{o['confidence']:.2f}" for e, o in analysis['expert_opinions'].items()}
            pbar.set_description(
                f"Kapitał: ${capital:,.2f} | Pozycja: {'TAK' if manager.active_position else 'NIE'} | Conf: {confs}")

    print("\nSymulacja zakończona. Generowanie raportów...")
    trades_df = pd.DataFrame(trades)
    generate_full_report(trades_df, equity_curve, capital, config)
    save_events_log(manager.events, config)


if __name__ == "__main__":
    asyncio.run(run())