import pandas as pd
import joblib
import json
import warnings
import os
import argparse
import mplfinance as mpf
from pathlib import Path
# Krok 1: Importujemy nasz centralny procesor danych
from data_processor import process_data_from_single_csv

# Ignorowanie ostrzeżeń z biblioteki mplfinance
warnings.filterwarnings('ignore', category=UserWarning)


# --- Funkcja do tworzenia wykresów (bez zmian) ---
def plot_trade(trade_info, ohlc_data, strategy, trade_number, chart_dir, ticker, tp_percent, sl_percent):
    entry_date = trade_info['entry_date']
    exit_date = trade_info['exit_date']
    entry_price = trade_info['entry_price']
    exit_price = trade_info['exit_price']
    pnl = trade_info['pnl_usd']
    exit_reason = trade_info['exit_reason']

    if strategy == 'long':
        tp_price = entry_price * (1 + tp_percent)
        sl_price = entry_price * (1 - sl_percent)
    else:  # short
        tp_price = entry_price * (1 - tp_percent)
        sl_price = entry_price * (1 + sl_percent)

    try:
        entry_idx = ohlc_data.index.get_loc(entry_date)
        if exit_date not in ohlc_data.index:
            exit_idx = ohlc_data.index.get_indexer([exit_date], method='nearest')[0]
        else:
            exit_idx = ohlc_data.index.get_loc(exit_date)

        start_idx = max(0, entry_idx - 15)
        end_idx = min(len(ohlc_data), exit_idx + 15)
        plot_data = ohlc_data.iloc[start_idx:end_idx].copy()
    except KeyError as e:
        print(f"Ostrzeżenie: Nie można zlokalizować daty {e}. Pomijanie wykresu dla transakcji #{trade_number}.")
        return

    plot_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
                     inplace=True)

    entry_marker = pd.Series(float('nan'), index=plot_data.index)
    exit_marker = pd.Series(float('nan'), index=plot_data.index)

    if entry_date in plot_data.index:
        entry_marker[entry_date] = entry_price * 0.99
    if exit_date in plot_data.index:
        exit_marker[exit_date] = exit_price

    entry_marker_style = '^' if strategy == 'long' else 'v'
    addplots = [
        mpf.make_addplot(entry_marker, type='scatter', marker=entry_marker_style, color='lime', markersize=200),
        mpf.make_addplot(exit_marker, type='scatter', marker='o', color='red', markersize=200)
    ]
    hlines = dict(hlines=[tp_price, sl_price], colors=['g', 'r'], linestyle='--')
    title = (f"Trade #{trade_number} | {ticker} | {strategy.upper()} | Exit: {exit_reason}\n"
             f"PnL: ${pnl:.2f} | Entry: {entry_price:.4f} -> Exit: {exit_price:.4f}")
    filename = os.path.join(chart_dir, f"trade_{trade_number}_{strategy}_{exit_reason.replace(' ', '_')}.png")
    mpf.plot(plot_data, type='candle', style='yahoo', title=title,
             addplot=addplots, hlines=hlines, figscale=1.5, savefig=filename)


# --- Główna funkcja backtestera ---
def run_backtest(args):
    model_path = Path(args.model_file)
    scaler_path = Path(args.scaler_file)

    # Automatyczne wykrywanie strategii ('long' lub 'short') na podstawie nazwy pliku modelu
    if 'long' in model_path.stem.lower():
        strategy = 'long'
    elif 'short' in model_path.stem.lower():
        strategy = 'short'
    else:
        strategy = 'unknown'

    # Wyprowadzenie ścieżki do pliku z cechami, jeśli nie została podana
    if args.features_file:
        features_path = Path(args.features_file)
    else:
        base_name = model_path.stem
        features_path = model_path.with_name(f"{base_name.replace('trading_model', 'feature_names')}.json")

    print(f"--- Uruchamianie Uniwersalnego Backtestera AI ---")
    print(f"Ticker: {args.ticker.upper()} | Strategia: {strategy.upper()}")

    output_dir_name = f"backtest_results_{args.ticker}_{model_path.stem}"
    chart_dir = os.path.join(output_dir_name, 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    trades_log_filename = os.path.join(output_dir_name, f'trades_log.csv')

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(features_path, 'r') as f:
            expected_features = json.load(f)
    except FileNotFoundError as e:
        print(f"BŁĄD: Nie znaleziono wymaganego pliku: {e.filename}.")
        return

    # Krok 2: Przygotowujemy dane do backtestu, używając tej samej funkcji co trener
    print("Wczytywanie i przygotowywanie danych do backtestu...")
    backtest_data = process_data_from_single_csv(args.data_file, args.start_date, args.end_date)

    if backtest_data is None or backtest_data.empty:
        print("Nie udało się przygotować danych do backtestu. Przerywanie.")
        return

    # Upewnienie się, że mamy wszystkie potrzebne kolumny
    if not all(feature in backtest_data.columns for feature in expected_features):
        print("BŁĄD: W przygotowanych danych brakuje niektórych cech wymaganych przez model.")
        return

    X_test = backtest_data[expected_features]
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    y_map = {0: -1, 1: 0, 2: 1}  # 0->Hold/Sell, 1->Neutral, 2->Buy
    backtest_data['signal'] = [y_map[p] for p in predictions]

    # --- Pętla symulacji i wyniki (bez zmian) ---
    print(f"Uruchamianie symulacji dla strategii {strategy.upper()}...")
    detailed_trades_log = []
    current_capital = args.initial_capital
    trade_counter = 0
    i = 0
    while i < len(backtest_data) - args.evaluation_window:
        if backtest_data['signal'].iloc[i] == 1:
            entry_price = backtest_data['close'].iloc[i]
            entry_date = backtest_data.index[i]
            risk_amount_usd = current_capital * args.risk_percent
            position_size_usd = risk_amount_usd / args.sl_percent if args.sl_percent > 0 else float('inf')
            trade_closed = False
            for j in range(1, args.evaluation_window + 1):
                future_high = backtest_data['high'].iloc[i + j]
                future_low = backtest_data['low'].iloc[i + j]
                exit_reason, pnl_usd, exit_price = None, 0, None
                if strategy == 'long':
                    sl_price = entry_price * (1 - args.sl_percent)
                    tp_price = entry_price * (1 + args.tp_percent)
                    if future_low <= sl_price:
                        pnl_usd = position_size_usd * args.sl_percent * -1
                        exit_reason, exit_price = 'Stop Loss', sl_price
                    elif future_high >= tp_price:
                        pnl_usd = position_size_usd * args.tp_percent
                        exit_reason, exit_price = 'Take Profit', tp_price
                else:
                    sl_price = entry_price * (1 + args.sl_percent)
                    tp_price = entry_price * (1 - args.tp_percent)
                    if future_high >= sl_price:
                        pnl_usd = position_size_usd * args.sl_percent * -1
                        exit_reason, exit_price = 'Stop Loss', sl_price
                    elif future_low <= tp_price:
                        pnl_usd = position_size_usd * args.tp_percent
                        exit_reason, exit_price = 'Take Profit', tp_price
                if exit_reason:
                    current_capital += pnl_usd
                    trade_info = {'entry_date': entry_date, 'entry_price': entry_price,
                                  'exit_date': backtest_data.index[i + j], 'exit_price': exit_price, 'pnl_usd': pnl_usd,
                                  'exit_reason': exit_reason, 'capital_after_trade': current_capital}
                    detailed_trades_log.append(trade_info)
                    trade_counter += 1
                    plot_trade(trade_info, backtest_data, strategy, trade_counter, chart_dir, args.ticker,
                               args.tp_percent, args.sl_percent)
                    trade_closed = True
                    i += j + 1
                    break
            if not trade_closed:
                exit_price = backtest_data['close'].iloc[i + args.evaluation_window]
                pnl_percent = ((exit_price - entry_price) / entry_price) if strategy == 'long' else (
                            (entry_price - exit_price) / entry_price)
                pnl_usd = position_size_usd * pnl_percent
                current_capital += pnl_usd
                trade_info = {'entry_date': entry_date, 'entry_price': entry_price,
                              'exit_date': backtest_data.index[i + args.evaluation_window], 'exit_price': exit_price,
                              'pnl_usd': pnl_usd, 'exit_reason': 'Time Exit', 'capital_after_trade': current_capital}
                detailed_trades_log.append(trade_info)
                trade_counter += 1
                plot_trade(trade_info, backtest_data, strategy, trade_counter, chart_dir, args.ticker, args.tp_percent,
                           args.sl_percent)
                i += args.evaluation_window + 1
        else:
            i += 1

    print(f"\n--- WYNIKI SYMULACJI STRATEGII {strategy.upper()} DLA {args.ticker.upper()} ---")
    if not detailed_trades_log:
        print("W okresie testowym model nie wygenerował żadnej transakcji.")
        return

    trades_df = pd.DataFrame(detailed_trades_log)
    trades_df.to_csv(trades_log_filename)
    print(f"Szczegółowy log transakcji został zapisany do pliku: {trades_log_filename}")

    num_trades = len(trades_df)
    win_rate = (len(trades_df[trades_df['pnl_usd'] > 0]) / num_trades) * 100 if num_trades > 0 else 0
    final_capital = trades_df['capital_after_trade'].iloc[-1] if not trades_df.empty else args.initial_capital
    total_pnl_percent = ((final_capital - args.initial_capital) / args.initial_capital) * 100
    buy_hold_pnl = ((backtest_data['close'].iloc[-1] - backtest_data['open'].iloc[0]) / backtest_data['open'].iloc[
        0]) * 100

    print(f"\nOkres testowy: od {backtest_data.index.min()} do {backtest_data.index.max()}")
    print(f"Kapitał początkowy: ${args.initial_capital:,.2f}")
    print(f"Kapitał końcowy: ${final_capital:,.2f}")
    print(f"Zysk/Strata (P/L): ${final_capital - args.initial_capital:,.2f} ({total_pnl_percent:.2f}%)")
    print("-" * 30)
    print(f"Liczba zrealizowanych transakcji: {num_trades}")
    print(f"Procent transakcji zyskownych (Win Rate): {win_rate:.2f}%")
    print(f"Średni zysk na transakcję: ${trades_df['pnl_usd'].mean():.2f}")
    print(f"\nRozkład powodów zamknięcia pozycji:\n{trades_df['exit_reason'].value_counts(normalize=True).round(2)}")
    print("-" * 30)
    print(f"Dla porównania, zwrot z 'Kup i Trzymaj' w tym okresie: {buy_hold_pnl:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uniwersalny backtester dla strategii tradingowych AI.")

    # Krok 3: Aktualizujemy listę argumentów
    parser.add_argument("--data-file", type=str, required=True,
                        help="Ścieżka do pliku CSV z surowymi danymi historycznymi.")
    parser.add_argument("--start-date", type=str, required=True,
                        help="Data początkowa dla danych do backtestu (format: RRRR-MM-DD).")
    parser.add_argument("--end-date", type=str, required=True,
                        help="Data końcowa dla danych do backtestu (format: RRRR-MM-DD).")

    parser.add_argument("--model-file", type=str, required=True, help="Ścieżka do wytrenowanego modelu .joblib.")
    parser.add_argument("--scaler-file", type=str, required=True, help="Ścieżka do dopasowanego scalera .joblib.")
    parser.add_argument("--features-file", type=str, help="(Opcjonalnie) Ścieżka do pliku .json z listą cech.")
    parser.add_argument("--ticker", type=str, default="ASSET", help="Nazwa tickera/aktywa do celów opisowych.")

    # Parametry strategii
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Kapitał początkowy symulacji.")
    parser.add_argument("--risk-percent", type=float, default=0.02,
                        help="Procent kapitału ryzykowany na jedną transakcję.")
    parser.add_argument("--tp-percent", type=float, default=0.04,
                        help="Poziom Take Profit jako procent od ceny wejścia.")
    parser.add_argument("--sl-percent", type=float, default=0.02, help="Poziom Stop Loss jako procent od ceny wejścia.")
    parser.add_argument("--evaluation-window", type=int, default=24,
                        help="Maksymalny czas utrzymywania pozycji w godzinach.")

    args = parser.parse_args()
    run_backtest(args)