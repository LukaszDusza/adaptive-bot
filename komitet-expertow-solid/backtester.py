
# backtester.py
import pandas as pd
from tqdm import tqdm

import config
from services.bybit_service import BybitService
from services.analysis_service import AnalysisService
from logic.position_manager import PositionManager
from utils.data_preparer import prepare_full_feature_set
from utils.reporting import generate_full_report, save_events_log


def run():
    # 1. Inicjalizacja
    bybit = BybitService(mode='backtest')
    analyzer = AnalysisService(config.TICKER_NAME_FOR_MODELS)
    manager = PositionManager(config)

    # 2. Dane
    # POPRAWKA 2: Użycie poprawnej nazwy funkcji
    df_raw = bybit.fetch_historical_data_range(config.TICKER, config.START_DATE, config.END_DATE)
    if df_raw is None: return print("Nie udało się pobrać danych. Zakończono.")

    test_data = prepare_full_feature_set(df_raw)

    # 3. Pętla symulacji
    capital = config.INITIAL_CAPITAL
    trades, equity_curve = [], {test_data.index[0]: capital}

    print("Uruchamianie symulacji...")
    pbar = tqdm(test_data.iterrows(), total=len(test_data))
    for timestamp, current_candle in pbar:
        analysis = analyzer.get_analysis_from_row(current_candle)

        # Centralny punkt decyzyjny - przekazujemy świecę do managera
        decision, details = manager.process_candle(current_candle, analysis, capital)

        # Backtester REAGUJE na decyzję, a nie ją podejmuje
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

    # 4. Raportowanie
    print("\nSymulacja zakończona. Generowanie raportów...")
    trades_df = pd.DataFrame(trades)
    generate_full_report(trades_df, equity_curve, capital, config)
    save_events_log(manager.events, config)


if __name__ == "__main__":
    run()