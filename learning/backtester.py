import pandas as pd
import joblib
import json
import warnings
# --- IMPORTUJEMY NASZ CENTRALNY PROCESOR DANYCH ---
from data_processor import load_and_prepare_timeframe

warnings.filterwarnings('ignore', category=UserWarning)

# ==============================================================================
# --- LABORATORIUM STRATEGII: TUTAJ EKSPERYMENTUJESZ Z PARAMETRAMI ---
# ==============================================================================

# --- 1. Zarządzanie Kapitałem ---
INITIAL_CAPITAL = 10000.0
# Wybierz tryb: 'fixed_usd' lub 'percent_of_capital'
SIZING_MODE = 'percent_of_capital'
# Jeśli 'fixed_usd', ta kwota będzie używana na każdą pozycję
FIXED_POSITION_SIZE_USD = 100.0
# Jeśli 'percent_of_capital', taki procent kapitału będzie ryzykowany na stratę
RISK_PERCENT_OF_CAPITAL = 0.02  # Ryzykujemy 2% kapitału na transakcję

# --- 2. Zarządzanie Ryzykiem Transakcji ---
TAKE_PROFIT_PERCENT = 0.04  # Spróbujmy celować w większy zysk: 4%
STOP_LOSS_PERCENT = 0.02  # W związku z tym zwiększamy też SL, aby zachować stosunek 2:1

# --- 3. Horyzont Czasowy ---
# Ten parametr MUSI być zgodny z tym, na czym trenowany był model
EVALUATION_WINDOW_HOURS = 24

# --- 4. Pliki Danych ---
# Pełna historia (potrzebna do "rozgrzewki" wskaźników)
HISTORICAL_FILE_1H = 'historical_data_icp_1h.csv'
HISTORICAL_FILE_4H = 'historical_data_icp_4h.csv'
HISTORICAL_FILE_1D = 'historical_data_icp_1D.csv'
# Nowe dane, na których faktycznie testujemy model
BACKTEST_FILE_1H = 'september_2025_1h.csv'
BACKTEST_FILE_4H = 'september_2025_4h.csv'
BACKTEST_FILE_1D = 'september_2025_1D.csv'
# Nazwa pliku do zapisu szczegółowego logu transakcji
TRADES_LOG_FILENAME = 'strategy_lab_trades_log.csv'


# ==============================================================================


def run_strategy_lab():
    """Główna funkcja uruchamiająca symulację z konfigurowalną strategią."""
    print("--- Uruchamianie Laboratorium Strategii AI z Detalicznym Logowaniem ---")

    try:
        model = joblib.load('trading_model.joblib')
        scaler = joblib.load('scaler.joblib')
        with open('feature_names.json', 'r') as f:
            expected_features = json.load(f)
    except FileNotFoundError as e:
        print(f"BŁĄD: Nie znaleziono pliku {e.filename}. Uruchom najpierw 'model_trainer.py'.")
        return

    # Przygotowanie danych
    print("Wczytywanie i przygotowywanie danych...")
    test_1h_dates = pd.read_csv(BACKTEST_FILE_1H, header=None, usecols=[2], names=['timestamp'])
    start_date = pd.to_datetime(test_1h_dates['timestamp']).min()
    end_date = pd.to_datetime(test_1h_dates['timestamp']).max()
    full_data_1h = load_and_prepare_timeframe([HISTORICAL_FILE_1H, BACKTEST_FILE_1H], '1h', include_ohlcv=True)
    full_data_4h = load_and_prepare_timeframe([HISTORICAL_FILE_4H, BACKTEST_FILE_4H], '4h')
    full_data_1d = load_and_prepare_timeframe([HISTORICAL_FILE_1D, BACKTEST_FILE_1D], '1d')
    combined_data = pd.merge_asof(full_data_1h, full_data_4h, left_index=True, right_index=True, direction='backward')
    combined_data = pd.merge_asof(combined_data, full_data_1d, left_index=True, right_index=True, direction='backward')
    backtest_data = combined_data.loc[start_date:end_date].copy()
    backtest_data.dropna(inplace=True)
    X_test = backtest_data[expected_features]
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    y_map = {0: -1, 1: 0, 2: 1}
    backtest_data['signal'] = [y_map[p] for p in predictions]

    print("Uruchamianie symulacji ze zdefiniowaną strategią...")
    detailed_trades_log = []
    current_capital = INITIAL_CAPITAL

    i = 0
    while i < len(backtest_data) - EVALUATION_WINDOW_HOURS:
        if backtest_data['signal'].iloc[i] == 1:
            entry_price = backtest_data['close_1h'].iloc[i]
            entry_date = backtest_data.index[i]

            if SIZING_MODE == 'percent_of_capital':
                risk_amount_usd = current_capital * RISK_PERCENT_OF_CAPITAL
                position_size_usd = risk_amount_usd / STOP_LOSS_PERCENT
            else:
                position_size_usd = FIXED_POSITION_SIZE_USD

            trade_closed = False
            for j in range(1, EVALUATION_WINDOW_HOURS + 1):
                future_high = backtest_data['high_1h'].iloc[i + j]
                future_low = backtest_data['low_1h'].iloc[i + j]

                exit_reason, pnl_usd, exit_price = None, 0, None

                if future_low <= entry_price * (1 - STOP_LOSS_PERCENT):
                    pnl_usd = position_size_usd * STOP_LOSS_PERCENT * -1
                    exit_reason = 'Stop Loss'
                    exit_price = entry_price * (1 - STOP_LOSS_PERCENT)
                elif future_high >= entry_price * (1 + TAKE_PROFIT_PERCENT):
                    pnl_usd = position_size_usd * TAKE_PROFIT_PERCENT
                    exit_reason = 'Take Profit'
                    exit_price = entry_price * (1 + TAKE_PROFIT_PERCENT)

                if exit_reason:
                    current_capital += pnl_usd
                    detailed_trades_log.append({
                        'entry_date': entry_date, 'entry_price': entry_price,
                        'exit_date': backtest_data.index[i + j], 'exit_price': exit_price,
                        'pnl_usd': pnl_usd, 'exit_reason': exit_reason,
                        'capital_after_trade': current_capital
                    })
                    trade_closed = True
                    i = i + j + 1
                    break

            if not trade_closed:
                exit_price = backtest_data['close_1h'].iloc[i + EVALUATION_WINDOW_HOURS]
                pnl_percent = (exit_price - entry_price) / entry_price
                pnl_usd = position_size_usd * pnl_percent
                current_capital += pnl_usd
                detailed_trades_log.append({
                    'entry_date': entry_date, 'entry_price': entry_price,
                    'exit_date': backtest_data.index[i + EVALUATION_WINDOW_HOURS], 'exit_price': exit_price,
                    'pnl_usd': pnl_usd, 'exit_reason': 'Time Exit',
                    'capital_after_trade': current_capital
                })
                i = i + EVALUATION_WINDOW_HOURS + 1
        else:
            i += 1

    # --- Prezentacja wyników ---
    print("\n--- WYNIKI SYMULACJI STRATEGII ---")
    if not detailed_trades_log:
        print("W okresie testowym model nie wygenerował żadnej transakcji.")
        return

    trades_df = pd.DataFrame(detailed_trades_log)
    trades_df.to_csv(TRADES_LOG_FILENAME)
    print(f"Szczegółowy log transakcji został zapisany do pliku: {TRADES_LOG_FILENAME}")

    num_trades = len(trades_df)
    win_rate = (len(trades_df[trades_df['pnl_usd'] > 0]) / num_trades) * 100 if num_trades > 0 else 0

    if not trades_df.empty:
        final_capital = trades_df['capital_after_trade'].iloc[-1]
    else:
        final_capital = INITIAL_CAPITAL

    total_pnl_percent = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    buy_hold_pnl = ((backtest_data['close_1h'].iloc[-1] - backtest_data['open_1h'].iloc[0]) /
                    backtest_data['open_1h'].iloc[0]) * 100

    print(f"\nOkres testowy: od {backtest_data.index.min()} do {backtest_data.index.max()}")
    print(f"\nKapitał początkowy: ${INITIAL_CAPITAL:,.2f}")
    print(f"Kapitał końcowy: ${final_capital:,.2f}")
    print(f"Zysk/Strata (P/L): ${final_capital - INITIAL_CAPITAL:,.2f} ({total_pnl_percent:.2f}%)")
    print("-" * 30)
    print(f"Liczba zrealizowanych transakcji: {num_trades}")
    print(f"Procent transakcji zyskownych (Win Rate): {win_rate:.2f}%")
    print(f"Średni zysk na transakcję: ${trades_df['pnl_usd'].mean():.2f}")
    print(f"\nRozkład powodów zamknięcia pozycji:\n{trades_df['exit_reason'].value_counts(normalize=True).round(2)}")
    print("-" * 30)
    print(f"Dla porównania, zwrot z 'Kup i Trzymaj' w tym okresie: {buy_hold_pnl:.2f}%")

    if total_pnl_percent > buy_hold_pnl:
        print("\nGratulacje! Twoja strategia pobiła rynek w testowanym okresie.")
    else:
        print("\nTwoja strategia okazała się gorsza niż pasywne trzymanie aktywa.")

    print("\n--- Przykładowe Transakcje (do weryfikacji na wykresie) ---")
    print("Pierwsze 5 transakcji:")
    print(trades_df.head().round(4))
    print("\nOstatnie 5 transakcji:")
    print(trades_df.tail().round(4))


if __name__ == "__main__":
    run_strategy_lab()

