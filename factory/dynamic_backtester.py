import pandas as pd
import joblib
import json
import warnings
import os
import mplfinance as mpf
from data_processor import load_and_prepare_timeframe

warnings.filterwarnings('ignore', category=UserWarning)

# ==============================================================================
# --- LABORATORIUM STRATEGII: TUTAJ EKSPERYMENTUJESZ Z PARAMETRAMI ---
# ==============================================================================

# --- 0. WYBÓR STRATEGII DO TESTOWANIA ---
STRATEGY_TO_TEST = 'long'  # Zmień na 'long' lub 'short'

# --- 1. Zarządzanie Kapitałem ---
INITIAL_CAPITAL = 10000.0
SIZING_MODE = 'percent_of_capital'
FIXED_POSITION_SIZE_USD = 100.0
RISK_PERCENT_OF_CAPITAL = 0.02

# --- 2. Zarządzanie Ryzykiem Transakcji ---
TAKE_PROFIT_PERCENT = 0.04
STOP_LOSS_PERCENT = 0.02

# --- 3. Pliki Danych ---
# Horyzont czasowy (EVALUATION_WINDOW_HOURS) nie jest już potrzebny do wyjścia z pozycji,
# ale zostawiamy go, bo może być przydatny w przyszłości.
HISTORICAL_FILE_1H = 'historical_data_icp_1h.csv'
HISTORICAL_FILE_4H = 'historical_data_icp_4h.csv'
HISTORICAL_FILE_1D = 'historical_data_icp_1D.csv'
BACKTEST_FILE_1H = 'september_2025_1h.csv'
BACKTEST_FILE_4H = 'september_2025_4h.csv'
BACKTEST_FILE_1D = 'september_2025_1D.csv'
TRADES_LOG_FILENAME = f'dynamic_trades_log_{STRATEGY_TO_TEST}.csv'


# ==============================================================================
# --- FUNKCJA DO TWORZENIA WYKRESÓW (BEZ ZMIAN) ---
# ==============================================================================
def plot_trade(trade_info, ohlc_data, strategy, trade_number, chart_dir):
    entry_date = trade_info['entry_date']
    exit_date = trade_info['exit_date']
    entry_price = trade_info['entry_price']
    exit_price = trade_info['exit_price']
    pnl = trade_info['pnl_usd']
    exit_reason = trade_info['exit_reason']

    if strategy == 'long':
        tp_price = entry_price * (1 + TAKE_PROFIT_PERCENT)
        sl_price = entry_price * (1 - STOP_LOSS_PERCENT)
    else:  # short
        tp_price = entry_price * (1 - TAKE_PROFIT_PERCENT)
        sl_price = entry_price * (1 + STOP_LOSS_PERCENT)
    try:
        entry_idx = ohlc_data.index.get_loc(entry_date)
        exit_idx = ohlc_data.index.get_loc(exit_date)
        start_idx = max(0, entry_idx - 10)
        end_idx = min(len(ohlc_data), exit_idx + 10)
        plot_data = ohlc_data.iloc[start_idx:end_idx].copy()
    except KeyError:
        print(
            f"Ostrzeżenie: Nie można zlokalizować daty transakcji w danych. Pomijanie wykresu dla transakcji #{trade_number}.")
        return

    plot_data.rename(columns={
        f'open_{"1h"}': 'Open', f'high_{"1h"}': 'High',
        f'low_{"1h"}': 'Low', f'close_{"1h"}': 'Close', f'volume_{"1h"}': 'Volume'
    }, inplace=True)
    entry_marker = pd.Series(float('nan'), index=plot_data.index)
    exit_marker = pd.Series(float('nan'), index=plot_data.index)
    entry_marker[entry_date] = entry_price * 0.99
    exit_marker[exit_date] = exit_price
    entry_marker_style = '^' if strategy == 'long' else 'v'
    addplots = [
        mpf.make_addplot(entry_marker, type='scatter', marker=entry_marker_style, color='lime', markersize=150),
        mpf.make_addplot(exit_marker, type='scatter', marker='o', color='red', markersize=150)
    ]
    hlines = dict(hlines=[tp_price, sl_price], colors=['g', 'r'], linestyle='--')
    title = (f"Trade #{trade_number} | {strategy.upper()} | Exit: {exit_reason}\n"
             f"PnL: ${pnl:.2f} | Entry: {entry_price:.4f} -> Exit: {exit_price:.4f}")
    filename = os.path.join(chart_dir, f"trade_{trade_number}_{strategy}_{exit_reason.replace(' ', '_')}.png")
    mpf.plot(plot_data, type='candle', style='yahoo', title=title, addplot=addplots, hlines=hlines, figscale=1.2,
             savefig=filename)


# ==============================================================================
def run_strategy_lab(strategy: str):
    print(f"--- Uruchamianie DYNAMICZNEGO Backtestera dla: {strategy.upper()} ---")

    chart_dir = f'dynamic_trade_charts_{strategy}'
    os.makedirs(chart_dir, exist_ok=True)
    print(f"Wykresy transakcji będą zapisywane w folderze: {chart_dir}")

    # --- Wczytywanie modelu i danych (bez zmian) ---
    model_filename = f'trading_model_{strategy}.joblib'
    scaler_filename = f'scaler_{strategy}.joblib'
    features_filename = f'feature_names_{strategy}.json'
    try:
        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)
        with open(features_filename, 'r') as f:
            expected_features = json.load(f)
    except FileNotFoundError as e:
        print(f"BŁĄD: Nie znaleziono pliku {e.filename}. Uruchom najpierw 'model_trainer.py --strategy {strategy}'.")
        return

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

    print(f"Uruchamianie symulacji z DYNAMICZNYM zarządzaniem pozycją...")

    # --- NOWA LOGIKA SYMULACJI ---
    detailed_trades_log = []
    current_capital = INITIAL_CAPITAL
    trade_counter = 0
    position_open = False
    open_trade_info = {}

    for i in range(len(backtest_data)):
        current_signal = backtest_data['signal'].iloc[i]
        current_high = backtest_data['high_1h'].iloc[i]
        current_low = backtest_data['low_1h'].iloc[i]
        current_close = backtest_data['close_1h'].iloc[i]
        current_date = backtest_data.index[i]

        exit_reason, exit_price = None, None

        # --- 1. SPRAWDŹ, CZY ZAMKNĄĆ OTWARTĄ POZYCJĘ ---
        if position_open:
            # Sprawdzenie warunków TP/SL
            if strategy == 'long':
                if current_low <= open_trade_info['sl_price']:
                    exit_reason, exit_price = 'Stop Loss', open_trade_info['sl_price']
                elif current_high >= open_trade_info['tp_price']:
                    exit_reason, exit_price = 'Take Profit', open_trade_info['tp_price']
            elif strategy == 'short':
                if current_high >= open_trade_info['sl_price']:
                    exit_reason, exit_price = 'Stop Loss', open_trade_info['sl_price']
                elif current_low <= open_trade_info['tp_price']:
                    exit_reason, exit_price = 'Take Profit', open_trade_info['tp_price']

            # Sprawdzenie wyjścia na podstawie sygnału z modelu
            if not exit_reason and current_signal != 1:
                exit_reason, exit_price = 'Model Exit', current_close

            # Jeśli jest powód do zamknięcia, zamknij pozycję
            if exit_reason:
                entry_price = open_trade_info['entry_price']
                position_size_usd = open_trade_info['position_size_usd']

                pnl_percent = ((exit_price - entry_price) / entry_price) if strategy == 'long' else (
                            (entry_price - exit_price) / entry_price)
                pnl_usd = position_size_usd * pnl_percent
                current_capital += pnl_usd

                trade_info = {
                    'entry_date': open_trade_info['entry_date'], 'entry_price': entry_price,
                    'exit_date': current_date, 'exit_price': exit_price,
                    'pnl_usd': pnl_usd, 'exit_reason': exit_reason, 'capital_after_trade': current_capital
                }
                detailed_trades_log.append(trade_info)
                trade_counter += 1
                plot_trade(trade_info, backtest_data, strategy, trade_counter, chart_dir)

                position_open = False
                open_trade_info = {}

        # --- 2. SPRAWDŹ, CZY OTWORZYĆ NOWĄ POZYCJĘ ---
        if not position_open:
            if current_signal == 1:
                position_open = True
                entry_price = current_close

                if SIZING_MODE == 'percent_of_capital':
                    risk_amount_usd = current_capital * RISK_PERCENT_OF_CAPITAL
                    position_size_usd = risk_amount_usd / STOP_LOSS_PERCENT if STOP_LOSS_PERCENT > 0 else float('inf')
                else:
                    position_size_usd = FIXED_POSITION_SIZE_USD

                open_trade_info = {
                    'entry_date': current_date,
                    'entry_price': entry_price,
                    'tp_price': entry_price * (1 + TAKE_PROFIT_PERCENT) if strategy == 'long' else entry_price * (
                                1 - TAKE_PROFIT_PERCENT),
                    'sl_price': entry_price * (1 - STOP_LOSS_PERCENT) if strategy == 'long' else entry_price * (
                                1 + STOP_LOSS_PERCENT),
                    'position_size_usd': position_size_usd
                }

    # --- Prezentacja wyników (bez zmian) ---
    print(f"\n--- WYNIKI SYMULACJI STRATEGII {strategy.upper()} (DYNAMICZNY) ---")
    if not detailed_trades_log:
        print("W okresie testowym model nie wygenerował żadnej transakcji.")
        return

    trades_df = pd.DataFrame(detailed_trades_log)
    trades_df.to_csv(TRADES_LOG_FILENAME)
    print(f"Szczegółowy log transakcji został zapisany do pliku: {TRADES_LOG_FILENAME}")

    num_trades = len(trades_df)
    win_rate = (len(trades_df[trades_df['pnl_usd'] > 0]) / num_trades) * 100 if num_trades > 0 else 0
    final_capital = trades_df['capital_after_trade'].iloc[-1] if not trades_df.empty else INITIAL_CAPITAL
    total_pnl_percent = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    buy_hold_pnl = ((backtest_data['close_1h'].iloc[-1] - backtest_data['open_1h'].iloc[0]) /
                    backtest_data['open_1h'].iloc[0]) * 100

    print(f"\nOkres testowy: od {backtest_data.index.min()} do {backtest_data.index.max()}")
    print(f"Kapitał początkowy: ${INITIAL_CAPITAL:,.2f}")
    print(f"Kapitał końcowy: ${final_capital:,.2f}")
    print(f"Zysk/Strata (P/L): ${final_capital - INITIAL_CAPITAL:,.2f} ({total_pnl_percent:.2f}%)")
    print("-" * 30)
    print(f"Liczba zrealizowanych transakcji: {num_trades}")
    print(f"Procent transakcji zyskownych (Win Rate): {win_rate:.2f}%")
    print(f"Średni zysk na transakcję: ${trades_df['pnl_usd'].mean():.2f}")
    print(f"\nRozkład powodów zamknięcia pozycji:\n{trades_df['exit_reason'].value_counts(normalize=True).round(2)}")
    print("-" * 30)
    print(f"Dla porównania, zwrot z 'Kup i Trzymaj' w tym okresie: {buy_hold_pnl:.2f}%")


if __name__ == "__main__":
    run_strategy_lab(STRATEGY_TO_TEST)