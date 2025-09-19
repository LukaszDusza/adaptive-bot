import pandas as pd
import joblib
import json
import warnings
import os
import argparse
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from data_processor import process_data_from_single_csv

warnings.filterwarnings('ignore', category=UserWarning)


# --- Funkcja do tworzenia wykresów transakcji (bez zmian) ---
def plot_trade(trade_info, ohlc_data, strategy, trade_number, chart_dir, ticker):
    entry_date = trade_info['entry_date']
    exit_date = trade_info['exit_date']
    entry_price = trade_info['actual_entry_price']
    exit_price = trade_info['exit_price']
    pnl = trade_info['pnl_usd']
    exit_reason = trade_info['exit_reason']
    tp_price = trade_info['tp_price']
    sl_price = trade_info['sl_price']
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
    if entry_date in plot_data.index: entry_marker[entry_date] = entry_price * 0.99
    if exit_date in plot_data.index: exit_marker[exit_date] = exit_price
    entry_marker_style = '^' if strategy == 'long' else 'v'
    addplots = [
        mpf.make_addplot(entry_marker, type='scatter', marker=entry_marker_style, color='lime', markersize=200),
        mpf.make_addplot(exit_marker, type='scatter', marker='o', color='red', markersize=200)
    ]
    hlines = dict(hlines=[tp_price, sl_price], colors=['g', 'r'], linestyle='--')
    title = (f"Trade #{trade_number} | {ticker} | {strategy.upper()} | Exit: {exit_reason}\n"
             f"PnL: ${pnl:.2f} | Entry: {entry_price:.4f} -> Exit: {exit_price:.4f}")
    filename = os.path.join(chart_dir, f"trade_{trade_number}_{strategy}_{exit_reason.replace(' ', '_')}.png")
    mpf.plot(plot_data, type='candle', style='yahoo', title=title, addplot=addplots, hlines=hlines, figscale=1.5,
             savefig=filename)


# <<< NOWA FUNKCJA DO RAPORTOWANIA >>>
def generate_report_and_plots(trades_df, initial_capital, ohlc_data, output_dir, ticker):
    """Generuje końcowy raport, statystyki i wykresy."""

    # --- Obliczanie Drawdownu ---
    trades_df['capital_after_trade'] = trades_df['capital_after_trade'].ffill()
    running_max = trades_df['capital_after_trade'].cummax()
    drawdown = running_max - trades_df['capital_after_trade']
    drawdown_percent = (drawdown / running_max) * 100
    max_drawdown = drawdown_percent.max()
    max_drawdown_date = drawdown_percent.idxmax()

    # --- Wykres Krzywej Kapitału i Drawdownu ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(trades_df['exit_date'], trades_df['capital_after_trade'], marker='o', linestyle='-', markersize=3,
            label='Kapitał (Equity)')
    ax.plot(trades_df['exit_date'], running_max, color='gray', linestyle='--', linewidth=1.5,
            label='Historyczny szczyt kapitału')
    ax.fill_between(trades_df.index, trades_df['capital_after_trade'], running_max,
                    where=(trades_df['capital_after_trade'] < running_max),
                    color='red', alpha=0.2, label='Okresy obsunięcia (Drawdown)')

    ax.set_title(f'Krzywa Kapitału i Drawdown dla {ticker}', fontsize=16)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Kapitał (USD)', fontsize=12)
    formatter = mticker.FuncFormatter(lambda y, _: '${:,.0f}'.format(y))
    ax.yaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    ax.legend()

    chart_filename = os.path.join(output_dir, 'equity_curve.png')
    plt.savefig(chart_filename)
    print(f"\nWykres krzywej kapitału został zapisany do pliku: {chart_filename}")

    # --- Obliczanie dodatkowych statystyk ---
    num_trades = len(trades_df)
    profitable_trades = trades_df[trades_df['pnl_usd'] > 0]
    losing_trades = trades_df[trades_df['pnl_usd'] <= 0]
    win_rate = (len(profitable_trades) / num_trades) * 100 if num_trades > 0 else 0
    final_capital = trades_df['capital_after_trade'].iloc[-1]
    total_pnl_percent = ((final_capital - initial_capital) / initial_capital) * 100
    buy_hold_pnl = ((ohlc_data['close'].iloc[-1] - ohlc_data['open'].iloc[0]) / ohlc_data['open'].iloc[0]) * 100
    avg_win = profitable_trades['pnl_usd'].mean() if not profitable_trades.empty else 0
    avg_loss = losing_trades['pnl_usd'].mean() if not losing_trades.empty else 0
    profit_factor = avg_win / abs(avg_loss) if abs(avg_loss) > 0 else float('inf')

    # --- Drukowanie podsumowania ---
    print(f"\n--- WYNIKI SYMULACJI DLA {ticker.upper()} ---")
    print(f"\nOkres testowy: od {ohlc_data.index.min()} do {ohlc_data.index.max()}")
    print(f"Kapitał początkowy: ${initial_capital:,.2f}")
    print(f"Kapitał końcowy: ${final_capital:,.2f}")
    print(f"Zysk/Strata (P/L): ${final_capital - initial_capital:,.2f} ({total_pnl_percent:.2f}%)")
    print("-" * 40)
    print(f"Liczba zrealizowanych transakcji: {num_trades}")
    print(f"Procent transakcji zyskownych (Win Rate): {win_rate:.2f}%")
    print(f"Maksymalny Drawdown (spadek od szczytu): {max_drawdown:.2f}%")
    print(f"\nŚrednia zyskowna transakcja: ${avg_win:,.2f}")
    print(f"Średnia stratna transakcja: ${avg_loss:,.2f}")
    print(f"Współczynnik zyskowności (Profit Factor): {profit_factor:.2f}")
    print(f"\nRozkład powodów zamknięcia pozycji:\n{trades_df['exit_reason'].value_counts(normalize=True).round(2)}")
    print("-" * 40)
    print(f"Dla porównania, zwrot z 'Kup i Trzymaj': {buy_hold_pnl:.2f}%")


# --- Główna funkcja backtestera ---
def run_backtest(args):
    print(f"--- Uruchamianie Realistycznego Backtestera AI (z symulacją poślizgu) dla {args.ticker.upper()} ---")
    output_dir_name = f"backtest_results_{args.ticker}"
    chart_dir = os.path.join(output_dir_name, 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    trades_log_filename = os.path.join(output_dir_name, 'trades_log.csv')
    try:
        model_long = joblib.load(args.model_long);
        scaler_long = joblib.load(args.scaler_long)
        model_short = joblib.load(args.model_short);
        scaler_short = joblib.load(args.scaler_short)
        with open(args.features_file, 'r') as f:
            expected_features = json.load(f)
    except FileNotFoundError as e:
        print(f"BŁĄD: Nie znaleziono wymaganego pliku: {e.filename}."); return

    print("Wczytywanie i przygotowywanie danych do backtestu...")
    backtest_data = process_data_from_single_csv(args.data_file, args.start_date, args.end_date)
    if backtest_data is None or backtest_data.empty: print(
        "Nie udało się przygotować danych do backtestu. Przerywanie."); return

    print("Generowanie predykcji dla obu modeli...")
    X_test = backtest_data.reindex(columns=expected_features, fill_value=0)  # Zapewnienie tej samej kolejności kolumn
    pred_numeric_long, pred_proba_long = model_long.predict(scaler_long.transform(X_test)), model_long.predict_proba(
        scaler_long.transform(X_test))
    pred_numeric_short, pred_proba_short = model_short.predict(
        scaler_short.transform(X_test)), model_short.predict_proba(scaler_short.transform(X_test))
    actions_long, confs_long, actions_short, confs_short = [], [], [], []
    for i in range(len(pred_numeric_long)):
        num, action = pred_numeric_long[i], "DO_NOTHING"
        if num == 2:
            action = "ENTER_LONG"
        elif num == 0:
            action = "EXIT_LONG"
        actions_long.append(action);
        confs_long.append(pred_proba_long[i][num])
    for i in range(len(pred_numeric_short)):
        num, action = pred_numeric_short[i], "DO_NOTHING"
        if num == 2:
            action = "ENTER_SHORT"
        elif num == 0:
            action = "EXIT_SHORT"
        actions_short.append(action);
        confs_short.append(pred_proba_short[i][num])
    backtest_data['long_action'], backtest_data['long_confidence'] = actions_long, confs_long
    backtest_data['short_action'], backtest_data['short_confidence'] = actions_short, confs_short

    print("Uruchamianie symulacji...")
    detailed_trades_log, current_capital, trade_counter, i = [], args.initial_capital, 0, 0
    while i < len(backtest_data):
        long_signal = (backtest_data['long_action'].iloc[i] == "ENTER_LONG" and backtest_data['long_confidence'].iloc[
            i] >= args.min_confidence)
        short_signal = (
                    backtest_data['short_action'].iloc[i] == "ENTER_SHORT" and backtest_data['short_confidence'].iloc[
                i] >= args.min_confidence)
        enter_trade, strategy = False, None
        if long_signal and short_signal:
            if backtest_data['long_confidence'].iloc[i] > backtest_data['short_confidence'].iloc[i]:
                enter_trade, strategy = True, 'long'
            else:
                enter_trade, strategy = True, 'short'
        elif long_signal:
            enter_trade, strategy = True, 'long'
        elif short_signal:
            enter_trade, strategy = True, 'short'
        if not enter_trade: i += 1; continue
        ideal_entry_price, entry_date = backtest_data['close'].iloc[i], backtest_data.index[i]
        sl_dist_percent = abs(ideal_entry_price - (ideal_entry_price - backtest_data['ATRr_14_1h'].iloc[
            i] * args.atr_loss_multiplier)) / ideal_entry_price if ideal_entry_price > 0 else 0
        position_size_usd = (current_capital * args.risk_percent) / sl_dist_percent if sl_dist_percent > 0 else 0
        slippage_penalty = (position_size_usd / args.slippage_base_usd) * args.slippage_factor
        actual_entry_price = ideal_entry_price * (1 + slippage_penalty) if strategy == 'long' else ideal_entry_price * (
                    1 - slippage_penalty)
        atr_value = backtest_data['ATRr_14_1h'].iloc[i]
        tp_price, sl_price, exit_signal_action = 0, 0, ""
        if strategy == 'long':
            tp_price, sl_price = actual_entry_price + (atr_value * args.atr_profit_multiplier), actual_entry_price - (
                        atr_value * args.atr_loss_multiplier)
            exit_signal_action = "EXIT_LONG"
        else:
            tp_price, sl_price = actual_entry_price - (atr_value * args.atr_profit_multiplier), actual_entry_price + (
                        atr_value * args.atr_loss_multiplier)
            exit_signal_action = "EXIT_SHORT"
        trade_closed = False
        for j in range(1, args.evaluation_window + 1):
            if i + j >= len(backtest_data): break
            future_high, future_low = backtest_data['high'].iloc[i + j], backtest_data['low'].iloc[i + j]
            exit_date, exit_reason, pnl_usd, exit_price = backtest_data.index[i + j], None, 0, None
            if strategy == 'long':
                if future_low <= sl_price:
                    exit_reason, exit_price = 'Stop Loss', sl_price
                elif future_high >= tp_price:
                    exit_reason, exit_price = 'Take Profit', tp_price
            else:
                if future_high >= sl_price:
                    exit_reason, exit_price = 'Stop Loss', sl_price
                elif future_low <= tp_price:
                    exit_reason, exit_price = 'Take Profit', tp_price
            if not exit_reason:
                action_check, conf_check = backtest_data[f'{strategy}_action'].iloc[i + j], \
                backtest_data[f'{strategy}_confidence'].iloc[i + j]
                if action_check == exit_signal_action and conf_check >= args.min_confidence:
                    exit_reason, exit_price = "Model Exit", backtest_data['close'].iloc[i + j]
            if exit_reason:
                pnl_percent = (exit_price - actual_entry_price) / actual_entry_price if strategy == 'long' else (
                                                                                                                            actual_entry_price - exit_price) / actual_entry_price
                pnl_usd = position_size_usd * pnl_percent
                current_capital += pnl_usd
                trade_info = {'entry_date': entry_date, 'ideal_entry_price': ideal_entry_price,
                              'actual_entry_price': actual_entry_price, 'exit_date': exit_date,
                              'exit_price': exit_price, 'pnl_usd': pnl_usd, 'exit_reason': exit_reason,
                              'capital_after_trade': current_capital, 'tp_price': tp_price, 'sl_price': sl_price}
                detailed_trades_log.append(trade_info);
                trade_counter += 1
                plot_trade(trade_info, backtest_data, strategy, trade_counter, chart_dir, args.ticker)
                trade_closed = True;
                i += (j + 1);
                break
        if not trade_closed:
            exit_idx = min(i + args.evaluation_window, len(backtest_data) - 1)
            exit_price, exit_date = backtest_data['close'].iloc[exit_idx], backtest_data.index[exit_idx]
            pnl_percent = (exit_price - actual_entry_price) / actual_entry_price if strategy == 'long' else (
                                                                                                                        actual_entry_price - exit_price) / actual_entry_price
            pnl_usd = position_size_usd * pnl_percent
            current_capital += pnl_usd
            trade_info = {'entry_date': entry_date, 'ideal_entry_price': ideal_entry_price,
                          'actual_entry_price': actual_entry_price, 'exit_date': exit_date, 'exit_price': exit_price,
                          'pnl_usd': pnl_usd, 'exit_reason': 'Time Exit', 'capital_after_trade': current_capital,
                          'tp_price': tp_price, 'sl_price': sl_price}
            detailed_trades_log.append(trade_info);
            trade_counter += 1
            plot_trade(trade_info, backtest_data, strategy, trade_counter, chart_dir, args.ticker)
            i += (args.evaluation_window + 1)

    if not detailed_trades_log:
        print("W okresie testowym model nie wygenerował żadnej transakcji.");
        return
    trades_df = pd.DataFrame(detailed_trades_log)
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df.set_index('entry_date', inplace=True, drop=False)
    trades_log_filename = os.path.join(output_dir_name, 'trades_log.csv')
    trades_df.to_csv(trades_log_filename)
    print(f"Szczegółowy log transakcji został zapisany do pliku: {trades_log_filename}")

    generate_report_and_plots(trades_df, args.initial_capital, backtest_data, output_dir_name, args.ticker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realistyczny backtester AI z symulacją poślizgu cenowego.")
    parser.add_argument("--data-file", type=str, required=True, help="Ścieżka do pliku CSV z danymi historycznymi.")
    parser.add_argument("--start-date", type=str, required=True, help="Data początkowa (RRRR-MM-DD).")
    parser.add_argument("--end-date", type=str, required=True, help="Data końcowa (RRRR-MM-DD).")
    parser.add_argument("--model-long", type=str, required=True, help="Ścieżka do modelu LONG .joblib.")
    parser.add_argument("--scaler-long", type=str, required=True, help="Ścieżka do scalera LONG .joblib.")
    parser.add_argument("--model-short", type=str, required=True, help="Ścieżka do modelu SHORT .joblib.")
    parser.add_argument("--scaler-short", type=str, required=True, help="Ścieżka do scalera SHORT .joblib.")
    parser.add_argument("--features-file", type=str, required=True, help="Ścieżka do pliku .json z listą cech.")
    parser.add_argument("--ticker", type=str, default="ASSET", help="Nazwa tickera do celów opisowych.")
    parser.add_argument("--initial-capital", type=float, default=2000.0)
    parser.add_argument("--risk-percent", type=float, default=0.02)
    parser.add_argument("--atr-profit-multiplier", type=float, default=2.0)
    parser.add_argument("--atr-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.60,
                        help="Minimalna pewność modelu do otwarcia pozycji.")
    parser.add_argument("--evaluation-window", type=int, default=24,
                        help="Maksymalny czas utrzymywania pozycji (w świecach 1h).")
    parser.add_argument("--slippage-base-usd", type=float, default=20000.0,
                        help="Wartość pozycji w USD, powyżej której zaczyna się poślizg.")
    parser.add_argument("--slippage-factor", type=float, default=0.0005,
                        help="Kara do ceny za każdy 'slippage_base_usd' wielkości pozycji (0.0005 = 0.05%%).")
    args = parser.parse_args()
    run_backtest(args)