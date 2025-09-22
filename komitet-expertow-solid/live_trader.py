import time
import logging

import pandas as pd

import config  # Importuje konfigurację (TICKER, etc.)

# Importy z Twojego projektu
from services.analysis_service import AnalysisService
from services.bybit_service import BybitService  # Zakładając, że BybitService jest w tym samym folderze
from logic.position_manager import PositionManager
from utils.data_preparer import prepare_full_feature_set

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LiveTrader:
    def __init__(self, config):
        """Inicjalizacja kluczowych komponentów tradera."""
        self.config = config
        self.ticker = config.TICKER

        # Inicjalizacja serwisów
        self.bybit_service = BybitService(mode='live', testnet=True)  # Zmień na False dla realnego handlu!
        self.analysis_service = AnalysisService(config.TICKER_NAME_FOR_MODELS)
        self.manager = PositionManager(config)

        self.is_running = False
        logger.info("Live Trader zainicjalizowany.")

    def _sync_position_state(self):
        """Synchronizuje stan PositionManager z rzeczywistą pozycją na giełdzie."""
        logger.info("Synchronizowanie stanu pozycji z giełdą...")

        active_positions = self.bybit_service.get_current_positions()
        position_on_exchange = next((p for p in active_positions if p['symbol'] == self.ticker), None)

        if not position_on_exchange:
            self.manager.clear_position()
            logger.info("Brak aktywnej pozycji na giełdzie. Stan wewnętrzny wyczyszczony.")
            return

        # Jeśli jest pozycja, zaktualizuj stan wewnętrzny
        pos_data = position_on_exchange['raw_data']
        strategy = 'long' if pos_data['side'] == 'Buy' else 'short'

        # UWAGA: Te dane są przybliżone. Brakuje nam triggerów BE/TSL, które musiałyby być
        # przechowywane w zewnętrznej bazie danych, aby przetrwać restarty.
        # Dla uproszczenia, zakładamy, że stan BE/TSL jest resetowany.
        position_details_for_manager = {
            'strategy': strategy,
            'entry_date': pd.to_datetime(int(pos_data['updatedTime']), unit='ms'),
            'entry_price': float(pos_data['avgPrice']),
            'size': float(pos_data['size']),
            'current_sl_price': float(pos_data.get('stopLoss', 0.0)),
            'tp_price': float(pos_data.get('takeProfit', 0.0)),
            'is_be': False,  # Stan nie jest znany, bezpiecznie założyć False
            'is_trailing': False,  # Stan nie jest znany, bezpiecznie założyć False
            # Poniższe wartości są nieznane, ustawiamy na 0
            'breakeven_trigger_price': 0,
            'breakeven_sl_price': 0,
            'trailing_trigger_price': 0,
            # Konfiguracje confidence nie są dostępne z giełdy
            'conf_momentum': 0, 'conf_reversion': 0, 'conf_pa': 0
        }
        self.manager.update_position_from_live_data(position_details_for_manager)
        logger.info(
            f"Stan wewnętrzny zsynchronizowany z pozycją na giełdzie: {strategy} {pos_data['size']} {self.ticker}")

    def _get_current_capital(self) -> float:
        """Pobiera dostępny kapitał z konta Unified Trading."""
        balance_info = self.bybit_service.get_account_balance()
        if balance_info and 'coin' in balance_info:
            for coin in balance_info['coin']:
                if coin.get('coin') == 'USDT':
                    # Używamy equity, które uwzględnia niezrealizowany PnL
                    return float(coin.get('equity', 0.0))
        logger.warning("Nie udało się pobrać kapitału. Zwracam 0.")
        return 0.0

    def _process_signal(self, signal: dict, current_candle):
        """Przetwarza sygnał z PositionManagera i wykonuje akcje na giełdzie."""
        action = signal.get('action')

        # --- 1. Obsługa instrukcji zarządzania pozycją (BE, TSL) ---
        # Wykonujemy to w pierwszej kolejności, aby zaktualizować SL przed podjęciem innych decyzji.
        for instruction in signal.get('instructions', []):
            logger.info(f"Otrzymano instrukcję: {instruction['type']}")
            if instruction['type'] == 'MOVE_SL_TO_BREAKEVEN' or instruction['type'] == 'UPDATE_TRAILING_STOP':
                new_sl = instruction['new_sl_price']
                self.bybit_service.modify_position(symbol=self.ticker, stop_loss=new_sl)
                # Aktualizujemy stan wewnętrzny po wysłaniu zlecenia
                self.manager.update_position_from_live_data({'current_sl_price': new_sl})

        # --- 2. Obsługa głównych akcji (OPEN, CLOSE, HOLD) ---
        if action in ['OPEN_LONG', 'OPEN_SHORT']:
            side = 'Buy' if action == 'OPEN_LONG' else 'Sell'
            logger.info(f"Otrzymano sygnał otwarcia: {action} dla {self.ticker}")

            # Umieść zlecenie na giełdzie
            order_response = self.bybit_service.place_order(
                symbol=self.ticker,
                side=side,
                order_type='Market',
                qty=signal['size'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit']
            )

            # Po udanym zleceniu, zaktualizuj stan wewnętrzny
            if order_response and order_response.get('ret_code') == 0:
                logger.info("Zlecenie otwarcia pozycji przyjęte przez giełdę.")
                # Czekamy chwilę na wypełnienie zlecenia
                time.sleep(5)
                # Synchronizujemy stan, aby pobrać faktyczną cenę wejścia etc.
                self._sync_position_state()
            else:
                logger.error(f"Nie udało się otworzyć pozycji: {order_response.get('error')}")

        elif action == 'CLOSE':
            logger.info(f"Otrzymano sygnał zamknięcia pozycji: {signal.get('exit_reason')}")
            close_response = self.bybit_service.close_position(self.ticker)
            if close_response and close_response.get('ret_code') == 0:
                logger.info("Zlecenie zamknięcia pozycji przyjęte przez giełdę.")
                self.manager.clear_position()  # Wyczyść stan wewnętrzny
            else:
                logger.error(f"Nie udało się zamknąć pozycji: {close_response.get('error')}")

        elif action == 'HOLD':
            # logger.info("Sygnał: HOLD. Brak akcji.")
            pass

    def run(self):
        """Główna pętla działania tradera."""
        self.is_running = True
        logger.info(f"Live Trader uruchomiony dla {self.ticker}. Rozpoczynam pętlę.")

        # Synchronizuj stan na starcie
        self._sync_position_state()

        while self.is_running:
            try:
                # 1. Pobierz dane
                candles_df = self.bybit_service.fetch_recent_candles(self.ticker, interval_minutes=5, limit=500)
                if candles_df.empty:
                    logger.warning("Nie udało się pobrać świec. Czekam na następną próbę.")
                    time.sleep(60)
                    continue

                # 2. Przygotuj wskaźniki i features
                features_df = prepare_full_feature_set(candles_df)
                current_candle = features_df.iloc[-1]

                # 3. Pobierz analizę (predykcje modeli)
                analysis = self.analysis_service.get_analysis_from_row(current_candle)

                # 4. Pobierz aktualny kapitał z konta
                current_capital = self._get_current_capital()
                if current_capital == 0:
                    logger.error("Kapitał wynosi 0. Zatrzymuję pętlę, aby uniknąć błędów.")
                    self.stop()
                    continue

                # 5. Pobierz sygnał transakcyjny
                # To jest kluczowy moment: przekazujemy aktualny kapitał do PositionManagera,
                # który obliczy wielkość pozycji na podstawie tego samego wzoru co w backtesterze.
                signal = self.manager.get_trading_signal(current_candle, analysis, current_capital)

                # 6. Przetwórz sygnał
                self._process_signal(signal, current_candle)

                # Czekaj na kolejną świecę (z małym buforem)
                logger.info("Cykl zakończony. Czekam na następną świecę...")
                time.sleep(305)  # 5 minut * 60 sekund + 5 sekund bufora

            except KeyboardInterrupt:
                self.stop()
            except Exception as e:
                logger.error(f"Wystąpił nieoczekiwany błąd w głównej pętli: {e}", exc_info=True)
                time.sleep(60)  # Czekaj minutę po błędzie

    def stop(self):
        """Zatrzymuje pętlę tradera."""
        self.is_running = False
        logger.info("Zatrzymywanie Live Tradera...")


if __name__ == '__main__':
    # Uruchomienie tradera
    trader = LiveTrader(config)
    trader.run()