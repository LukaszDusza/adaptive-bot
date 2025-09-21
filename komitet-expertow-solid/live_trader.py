# live_trader.py
import time
import config
from services.bybit_service import BybitService
from services.analysis_service import AnalysisService
from logic.position_manager import PositionManager
from utils.data_preparer import prepare_full_feature_set  # <-- UŻYWAMY TEJ SAMEJ FUNKCJI!


def run():
    print("--- URUCHAMIANIE BOTA W TRYBIE LIVE (POPRAWNA LOGIKA) ---")

    # Inicjalizacja tych samych modułów, co w backtesterze
    bybit = BybitService(mode='live')
    analyzer = AnalysisService(config.TICKER_NAME_FOR_MODELS)
    manager = PositionManager(config)

    while True:
        print("Nowa pętla live tradera...")

        # 1. Pobierz najnowszy fragment danych 5-minutowych
        # Potrzebujemy wystarczająco dużo danych do obliczenia wskaźników (np. 500 świec)
        df_raw_5m = bybit.fetch_recent_candles(config.TICKER, interval_minutes=5, limit=500)

        if df_raw_5m is None or df_raw_5m.empty:
            print("Nie udało się pobrać danych, czekam na kolejną pętlę.")
            time.sleep(60)
            continue

        # 2. Przygotuj cechy, używając DOKŁADNIE tej samej funkcji co backtester
        features_df = prepare_full_feature_set(df_raw_5m)

        # 3. Weź tylko ostatni, w pełni obliczony wiersz
        last_candle_features = features_df.iloc[-1]

        # 4. Przekaż go do tych samych modułów analizy i zarządzania
        analysis = analyzer.get_analysis_from_row(last_candle_features)
        decision, details = manager.process_candle(last_candle_features, analysis,
                                                   live_capital)  # Załóżmy, że mamy live_capital

        # 5. Wykonaj decyzję
        if decision == 'OPEN':
            print(f"DECYZJA LIVE: Otwórz pozycję! Szczegóły: {details}")
            # bybit.place_order(...)
        elif decision == 'CLOSE':
            print(f"DECYZJA LIVE: Zamknij pozycję! Szczegóły: {details}")
            # bybit.close_position(...)

        # Oczekiwanie na kolejną świecę 5m
        print("Czekam na kolejną świecę...")
        time.sleep(300)