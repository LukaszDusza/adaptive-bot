import pandas as pd
import joblib
import json
import argparse
import pandas_ta as ta
import numpy as np
import os
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from datetime import datetime


# --- FUNKCJE POMOCNICZE ---
def process_data_from_single_csv(csv_path: str):
    print("--- Uruchamianie procesora danych z jednego pliku CSV ---")
    try:
        df_raw = pd.read_csv(csv_path, parse_dates=['timestamp'])
    except Exception as e:
        print(f"BŁĄD KRYTYCZNY: Nie udało się wczytać pliku CSV. Szczegóły: {e}")
        return None
    df_raw.rename(columns={'open_price': 'open', 'high_price': 'high', 'low_price': 'low', 'close_price': 'close'},
                  inplace=True)
    timeframe_map = {5: '5m', 15: '15m', 60: '1h'};
    df_raw['timeframe'] = df_raw['timeframe'].map(timeframe_map)
    df_raw.dropna(subset=['timeframe'], inplace=True)
    all_dataframes = {}
    for tf_name, tf_df in df_raw.groupby('timeframe'):
        df_processed = tf_df.copy();
        df_processed.set_index('timestamp', inplace=True);
        df_processed.sort_index(inplace=True)
        df_processed.ta.rsi(append=True);
        df_processed.ta.atr(append=True);
        df_processed.ta.macd(append=True);
        df_processed.ta.bbands(append=True);
        df_processed.ta.stoch(append=True);
        df_processed.ta.adx(append=True);
        df_processed.ta.ema(length=50, append=True);
        df_processed.ta.ema(length=200, append=True)
        all_dataframes[tf_name] = df_processed
    df_5m = all_dataframes['5m'].add_suffix('_5m').rename(
        columns={'open_5m': 'open', 'high_5m': 'high', 'low_5m': 'low', 'close_5m': 'close', 'volume_5m': 'volume'})
    df_15m = all_dataframes['15m'].drop(columns=['open', 'high', 'low', 'close', 'volume']).add_suffix('_15m')
    df_1h = all_dataframes['1h'].drop(columns=['open', 'high', 'low', 'close', 'volume']).add_suffix('_1h')
    final_df = pd.merge_asof(df_5m, df_15m, left_index=True, right_index=True, direction='backward')
    final_df = pd.merge_asof(final_df, df_1h, left_index=True, right_index=True, direction='backward')
    return final_df


def plot_trade(trade_info, ohlc_data, chart_dir):
    entry_date = trade_info['entry_date']
    exit_date = trade_info['exit_date']
    plot_start_date = entry_date - pd.Timedelta(hours=2)
    plot_end_date = exit_date + pd.Timedelta(hours=2)
    plot_data = ohlc_data.loc[plot_start_date:plot_end_date]
    entry_marker_data = pd.Series(np.nan, index=plot_data.index)
    if entry_date in plot_data.index: entry_marker_data[entry_date] = trade_info['entry_price'] * 0.998
    exit_marker_data = pd.Series(np.nan, index=plot_data.index)
    if exit_date in plot_data.index: exit_marker_data[exit_date] = trade_info['exit_price']
    entry_marker_style = '^' if trade_info['strategy'] == 'long' else 'v'
    entry_color = 'green' if trade_info['strategy'] == 'long' else 'red'
    add_plots = [
        mpf.make_addplot(entry_marker_data, type='scatter', markersize=150, marker=entry_marker_style,
                         color=entry_color),
        mpf.make_addplot(exit_marker_data, type='scatter', markersize=150, marker='o', color='blue')
    ]
    hlines = dict(hlines=[trade_info['tp_price'], trade_info['sl_price']], colors=['g', 'r'], linestyle='--')
    title = f"Trade #{trade_info['trade_number']} ({trade_info['strategy'].upper()}) | P/L: ${trade_info['pnl_usd']:.2f}"
    mpf.plot(plot_data, type='candle', style='yahoo', title=title, ylabel='Price ($)',
             addplot=add_plots, hlines=hlines, savefig=f"{chart_dir}/trade_{trade_info['trade_number']}.png",
             figsize=(15, 7))


# --- NOWA, ZINTEGROWANA FUNKCJA DO GENEROWANIA WYKRESÓW KAPITAŁU I DRAWDOWN ---
def plot_equity_and_drawdown(equity_series, ticker):
    """Generuje jeden wykres zawierający krzywą kapitału i drawdown."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Wykres 1: Krzywa Kapitału
    equity_series.plot(ax=ax1, lw=2, title=f'Krzywa Kapitału i Drawdown - {ticker}')
    ax1.set_ylabel('Kapitał ($)')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Wykres 2: Drawdown
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max * 100
    drawdown.plot(ax=ax2, kind='area', color='red', alpha=0.4)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Data')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    chart_filename = f'equity_and_drawdown_{ticker}.png'
    plt.savefig(chart_filename)
    plt.close()
    print(f"Zapisano zintegrowany wykres kapitału i drawdown do pliku: {chart_filename}")


# --- GŁÓWNA LOGIKA BACKTESTERA ---
def run_ensemble_backtest_with_model_exit(args):
    print(f"--- Uruchamianie Backtestu z DYNAMICZNYM WYJŚCIEM | Ticker: {args.ticker} ---")

    print("Wczytywanie wytrenowanych modeli ekspertów...")
    models, features = {}, {}
    for expert in ['momentum', 'reversion', 'pa']:
        models[expert] = joblib.load(f'expert_{expert}_{args.ticker}_5m.joblib')
        with open(f'features_{expert}_{args.ticker}_5m.json', 'r') as f:
            features[expert] = json.load(f)
    scaler_pa = joblib.load(f'scaler_pa_{args.ticker}_5m.joblib')

    print("Przygotowywanie danych testowych...")
    full_df = process_data_from_single_csv(args.data_file)
    if full_df is None: return

    _, test_data_full = train_test_split(full_df, test_size=0.2, shuffle=False)
    print(f"Filtrowanie danych testowych do zakresu od {args.start_date} do {args.end_date}...")
    test_data = test_data_full.loc[args.start_date:args.end_date].copy()
    if test_data.empty:
        print("BŁĄD: Brak danych w podanym zakresie dat.");
        return

    print("Tworzenie cech 'Price Action' dla danych testowych...")
    pa_df = test_data[['open', 'high', 'low', 'close', 'volume', 'ATRr_14_5m']].copy()
    pa_df['impulse_strength'] = (pa_df['close'] - pa_df['open']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    pa_df['volatility_burst'] = (pa_df['high'] - pa_df['low']) / pa_df['ATRr_14_5m'].replace(0, 1)
    pa_df['closing_position'] = (pa_df['close'] - pa_df['low']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    pa_df['volume_spike'] = pa_df['volume'] / pa_df['volume'].rolling(window=20).mean().replace(0, 1)
    for col in ['impulse_strength', 'volatility_burst', 'closing_position', 'volume_spike']:
        for n in [1, 2, 3]:
            pa_df[f'{col}_lag_{n}'] = pa_df[col].shift(n)
    pa_features_to_add = [col for col in pa_df.columns if
                          col not in ['open', 'high', 'low', 'close', 'volume', 'ATRr_14_5m']]
    test_data = pd.concat([test_data, pa_df[pa_features_to_add]], axis=1)

    print("Generowanie predykcji od wszystkich ekspertów...")
    for expert in ['momentum', 'reversion', 'pa']:
        X = test_data[features[expert]]
        if expert == 'pa':
            X.replace([np.inf, -np.inf], 0, inplace=True);
            X.fillna(0, inplace=True)
            X = scaler_pa.transform(X)
        test_data[f'pred_{expert}'] = models[expert].predict(X)
        test_data[f'conf_{expert}'] = models[expert].predict_proba(X).max(axis=1)

    capital = args.initial_capital
    equity_curve = {test_data.index[0]: capital}
    trades = []
    active_positions = {}

    print("Uruchamianie symulacji z DYNAMICZNYM WYJŚCIEM...")
    for i in range(len(test_data)):
        current_date = test_data.index[i]
        current_votes_long, current_votes_short = 0, 0
        expert_preds_current = {}
        for expert in ['momentum', 'reversion', 'pa']:
            pred = test_data[f'pred_{expert}'].iloc[i];
            conf = test_data[f'conf_{expert}'].iloc[i]
            expert_preds_current[f'pred_{expert}'] = int(pred)
            if conf >= args.min_confidence:
                if pred == 1:
                    current_votes_long += 1
                else:
                    current_votes_short += 1

        for strategy in list(active_positions.keys()):
            pos = active_positions[strategy];
            exit_reason = None
            if (strategy == 'long' and current_votes_short >= 2) or (strategy == 'short' and current_votes_long >= 2):
                exit_reason = "Model Exit Signal"
            elif strategy == 'long':
                if test_data['low'].iloc[i] <= pos['sl_price']:
                    exit_reason = "Stop Loss"
                elif test_data['high'].iloc[i] >= pos['tp_price']:
                    exit_reason = "Take Profit"
            elif strategy == 'short':
                if test_data['high'].iloc[i] >= pos['sl_price']:
                    exit_reason = "Stop Loss"
                elif test_data['low'].iloc[i] <= pos['tp_price']:
                    exit_reason = "Take Profit"

            if exit_reason:
                exit_price = test_data['close'].iloc[i] if exit_reason == "Model Exit Signal" else (
                    pos['sl_price'] if "Stop Loss" in exit_reason else pos['tp_price'])
                pnl_usd = (exit_price - pos['entry_price']) * pos['position_size_units'] if strategy == 'long' else (
                                                                                                                                pos[
                                                                                                                                    'entry_price'] - exit_price) * \
                                                                                                                    pos[
                                                                                                                        'position_size_units']
                capital += pnl_usd
                pos.update({'exit_date': current_date, 'exit_price': exit_price, 'pnl_usd': pnl_usd,
                            'exit_reason': exit_reason})
                trades.append(pos)
                del active_positions[strategy]

        strategy_to_open = None
        if current_votes_long >= 2 and 'long' not in active_positions:
            strategy_to_open = 'long'
        elif current_votes_short >= 2 and 'short' not in active_positions:
            strategy_to_open = 'short'

        if strategy_to_open:
            entry_price = test_data['close'].iloc[i]
            stop_loss_distance = test_data['ATRr_14_5m'].iloc[i] * args.atr_multiplier
            sl_price = entry_price - stop_loss_distance if strategy_to_open == 'long' else entry_price + stop_loss_distance
            tp_price = entry_price + (
                        abs(entry_price - sl_price) * args.rrr) if strategy_to_open == 'long' else entry_price - (
                        abs(entry_price - sl_price) * args.rrr)
            position_value = capital * args.risk_percent * args.leverage
            position_size = position_value / entry_price if entry_price > 0 else 0
            risk_amount_usd = position_size * abs(entry_price - sl_price)
            active_positions[strategy_to_open] = {
                'trade_number': len(trades) + 1, 'strategy': strategy_to_open, 'entry_date': current_date,
                'entry_price': entry_price,
                'position_size_units': position_size, 'risk_amount_usd': risk_amount_usd,
                'tp_price': tp_price, 'sl_price': sl_price, 'capital_before_trade': capital, 'is_be': False,
                **expert_preds_current
            }
            capital -= args.trade_cost

        equity_curve[current_date] = capital

    # --- Krok 5: Podsumowanie, logi i wykresy ---
    trades_df = pd.DataFrame(trades)
    log_filename = f'backtest_log_ensemble_{args.ticker}.csv';
    trades_df.to_csv(log_filename, index=False)
    print(f"\nZapisano szczegółowy dziennik transakcji do pliku: {log_filename}")

    chart_dir = f'trade_charts_{args.ticker}';
    if not os.path.exists(chart_dir): os.makedirs(chart_dir)
    print(f"Generowanie wykresów dla każdej transakcji do folderu: {chart_dir} (może to potrwać)...")

    # --- ZMIANA: Wywołanie nowej, zintegrowanej funkcji do rysowania wykresów ---
    equity_series = pd.Series(equity_curve)
    plot_equity_and_drawdown(equity_series, args.ticker)

    if not trades_df.empty:
        pnl_total = capital - args.initial_capital
        wins = trades_df[trades_df['pnl_usd'] > 0]

        print(f"\n--- WYNIKI REALISTYCZNEGO BACKTESTU Z DYNAMICZNYM WYJŚCIEM ---")
        print(f"Testowany okres: od {args.start_date} do {args.end_date}")
        print(f"Kapitał początkowy: ${args.initial_capital:,.2f}")
        print(f"Kapitał końcowy: ${capital:,.2f}")
        print(f"Zysk/Strata (P/L): ${pnl_total:,.2f} ({(pnl_total / args.initial_capital * 100):.2f}%)")
        print("-" * 40)
        print(
            f"Liczba transakcji: {len(trades_df)} (Long: {len(trades_df[trades_df.strategy == 'long'])}, Short: {len(trades_df[trades_df.strategy == 'short'])})")
        print(f"Procent zyskownych (Win Rate): {len(wins) / len(trades_df) * 100:.2f}%")
        if abs(trades_df[trades_df.pnl_usd <= 0]['pnl_usd'].sum()) > 0:
            print(
                f"Profit Factor: {wins['pnl_usd'].sum() / abs(trades_df[trades_df.pnl_usd <= 0]['pnl_usd'].sum()):.2f}")
        else:
            print("Profit Factor: N/A (brak stratnych transakcji)")
        trades_df['duration'] = trades_df['exit_date'] - trades_df['entry_date']
        print(f"Średni czas trwania pozycji: {trades_df['duration'].mean()}")
        print("\nRozkład powodów zamknięcia pozycji:")
        print(trades_df['exit_reason'].value_counts(normalize=True).apply("{:.2%}".format))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realistyczny backtester dla 'Komitetu Ekspertów'.")
    today_str = datetime.now().strftime('%Y-%m-%d')
    parser.add_argument("--start-date", type=str, default="2025-04-14",
                        help="Data początkowa testu (format: YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=today_str, help="Data końcowa testu (format: YYYY-MM-DD).")
    parser.add_argument("--data-file", type=str, required=True, help="Ścieżka do pliku CSV z danymi testowymi.")
    parser.add_argument("--ticker", type=str, default="ETH", help="Nazwa tickera.")
    parser.add_argument("--initial-capital", type=float, default=1000.0, help="Kapitał początkowy.")
    parser.add_argument("--risk-percent", type=float, default=0.02, help="Procent kapitału ryzykowany na transakcję.")
    parser.add_argument("--min-confidence", type=float, default=0.58,
                        help="Minimalna pewność modeli do otwarcia pozycji.")
    parser.add_argument("--rrr", type=float, default=2.0, help="Risk-Reward Ratio (TP/SL).")
    parser.add_argument("--leverage", type=float, default=10.0, help="Dźwignia finansowa.")
    parser.add_argument("--trade-cost", type=float, default=1.1, help="Koszt otwarcia pozycji w USD.")
    parser.add_argument("--atr-multiplier", type=float, default=1.5, help="Mnożnik ATR dla Stop Lossa.")


    args = parser.parse_args()
    run_ensemble_backtest_with_model_exit(args)