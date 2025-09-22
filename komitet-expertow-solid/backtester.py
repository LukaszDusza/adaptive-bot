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
FEE_BPS_OPEN = getattr(config, "FEE_BPS_OPEN", 3.0)  # np. 0.03%
FEE_BPS_CLOSE = getattr(config, "FEE_BPS_CLOSE", 3.0)  # np. 0.03%


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
        return

    test_data = prepare_full_feature_set(df_raw)

    capital = config.INITIAL_CAPITAL
    trades, equity_curve = [], {test_data.index[0]: capital}
    open_trade_fee = 0.0

    print("Uruchamianie symulacji...")
    pbar = tqdm(test_data.iterrows(), total=len(test_data))
    for timestamp, current_candle in pbar:
        analysis = analyzer.get_analysis_from_row(current_candle)

        # --- ZMIANA GŁÓWNA: Użycie API przeznaczonego dla backtestera ---
        # Metoda process_candle zarządza stanem pozycji (w tym BE i TSL) wewnętrznie.
        action, details = manager.process_candle(current_candle, analysis, capital)

        if action == 'OPEN':
            # Pozycja została utworzona wewnątrz managera.
            # Obliczamy i odejmujemy od kapitału opłatę za otwarcie.
            pos = details
            open_trade_fee = fee_calculator.calculate_exchange_fees(pos.entry_price * pos.size, FEE_BPS_OPEN)
            capital -= open_trade_fee

        elif action == 'CLOSE':
            # process_candle zwraca słownik z zamkniętą transakcją.
            # PnL brutto ('pnl_usd') jest już obliczony z uwzględnieniem poślizgu.
            closed_trade = details

            pnl_gross = closed_trade['pnl_usd']

            # Oblicz opłatę za zamknięcie
            close_fee = fee_calculator.calculate_exchange_fees(closed_trade['exit_price'] * closed_trade['size'],
                                                               FEE_BPS_CLOSE)
            total_fee = open_trade_fee + close_fee

            # Oblicz PnL netto
            pnl_net = pnl_gross - total_fee

            # Zaktualizuj kapitał
            capital += pnl_gross  # Dodaj PnL brutto (po poślizgu)
            capital -= close_fee  # Odejmij opłatę za zamknięcie

            # Stwórz kompletny rekord transakcji i go zapisz
            trade_record = {**closed_trade, 'fees_usd': total_fee, 'pnl_net_usd': pnl_net}
            trades.append(trade_record)

            # Zresetuj opłatę za otwarcie
            open_trade_fee = 0.0

        # --- USUNIĘTO: Pętla po 'instructions' ---
        # Cała logika BE/TSL jest teraz obsługiwana wewnątrz manager.process_candle(),
        # więc nie musimy już nic tutaj robić.

        equity_curve[timestamp] = capital

        if config.DEBUG_MODE and analysis:
            confs = {e: f"{o['confidence']:.2f}" for e, o in analysis['expert_opinions'].items()}
            position_status = 'TAK' if manager.active_position else 'NIE'
            pbar.set_description(
                f"Kapitał: ${capital:,.2f} | Pozycja: {position_status} | Conf: {confs}")

    print("\nSymulacja zakończona. Generowanie raportów...")
    trades_df = pd.DataFrame(trades)
    generate_full_report(trades_df, equity_curve, capital, config, test_data)
    save_events_log(manager.events, config)


if __name__ == "__main__":
    asyncio.run(run())