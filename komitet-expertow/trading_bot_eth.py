import os
import time
import pandas as pd
import sys
import json
import logging
import argparse
import pandas_ta as ta
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
from tqdm import tqdm
from pybit.unified_trading import HTTP

# --- Konfiguracja Globalna ---
DEFAULT_TICKER = "ETHUSDT"
DEFAULT_RISK = 0.02
DEFAULT_LEVERAGE = "10"
DEFAULT_ATR_MULTIPLIER = 1.5
DEFAULT_MIN_CONFIDENCE = 0.58
DEFAULT_RRR = 2.0
DEFAULT_TRADE_COST = 1.0

# --- Konfiguracja Logowania ---
logging.basicConfig(level=logging.INFO,
                    format='{\"timestamp\": \"%(asctime)s\", \"level\": \"%(levelname)s\", \"service\": \"trading_bot_eth\", \"message\": %(message)s}',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')


def log(event, details):
    logging.info(json.dumps({"event": event, "details": details}))


# --- NOWA, KOMPLETNA FUNKCJA DO PRZYGOTOWANIA DANYCH ---
def prepare_full_feature_set(df_5m_raw: pd.DataFrame):
    print("Agregowanie danych i obliczanie wszystkich wskaźników oraz cech...")

    ohlc = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    df_15m_raw = df_5m_raw.resample('15min').agg(ohlc).dropna()
    df_1h_raw = df_5m_raw.resample('1h').agg(ohlc).dropna()

    all_dataframes = {'5m': df_5m_raw, '15m': df_15m_raw, '1h': df_1h_raw}

    for tf_name, df in all_dataframes.items():
        df.ta.rsi(append=True);
        df.ta.atr(append=True);
        df.ta.macd(append=True);
        df.ta.bbands(append=True);
        df.ta.stoch(append=True);
        df.ta.adx(append=True)

    df_5m = all_dataframes['5m'].add_suffix('_5m').rename(
        columns={'open_5m': 'open', 'high_5m': 'high', 'low_5m': 'low', 'close_5m': 'close', 'volume_5m': 'volume'})
    df_15m = all_dataframes['15m'].drop(columns=['open', 'high', 'low', 'close', 'volume']).add_suffix('_15m')
    df_1h = all_dataframes['1h'].drop(columns=['open', 'high', 'low', 'close', 'volume']).add_suffix('_1h')

    final_df = pd.merge_asof(df_5m, df_15m, left_index=True, right_index=True, direction='backward')
    final_df = pd.merge_asof(final_df, df_1h, left_index=True, right_index=True, direction='backward')

    pa_df = final_df[['open', 'high', 'low', 'close', 'volume', 'ATRr_14_5m']].copy()
    pa_df['impulse_strength'] = (pa_df['close'] - pa_df['open']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    pa_df['volatility_burst'] = (pa_df['high'] - pa_df['low']) / pa_df['ATRr_14_5m'].replace(0, 1)
    pa_df['closing_position'] = (pa_df['close'] - pa_df['low']) / (pa_df['high'] - pa_df['low']).replace(0, 1)
    pa_df['volume_spike'] = pa_df['volume'] / pa_df['volume'].rolling(window=20).mean().replace(0, 1)
    for col in ['impulse_strength', 'volatility_burst', 'closing_position', 'volume_spike']:
        for n in [1, 2, 3]: pa_df[f'{col}_lag_{n}'] = pa_df[col].shift(n)

    pa_features_to_add = [col for col in pa_df.columns if
                          col not in ['open', 'high', 'low', 'close', 'volume', 'ATRr_14_5m']]
    final_df = pd.concat([final_df, pa_df[pa_features_to_add]], axis=1)

    final_df.dropna(inplace=True)
    return final_df


# --- Główna Klasa Bota ---
class TradingBot:
    def __init__(self, args):
        print("Inicjalizacja TradingBota...")
        self.args = args
        self.session = self._initialize_session()
        self.models, self.features, self.scaler_pa = {}, {}, None
        self._load_ml_artifacts()

    def _initialize_session(self):
        print(f"Inicjalizacja sesji Bybit w trybie '{self.args.mode}'...")
        if self.args.mode == 'live':
            api_key, api_secret = os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET")
            if not (api_key and api_secret): sys.exit("Brak kluczy API dla trybu LIVE")
            return HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
        return HTTP(testnet=False)

    def _load_ml_artifacts(self):
        print(f"Wczytywanie modeli i cech dla {self.args.ticker_name}...")
        try:
            for expert in ['momentum', 'reversion', 'pa']:
                self.models[expert] = joblib.load(f'expert_{expert}_{self.args.ticker_name}_5m.joblib')
                with open(f'features_{expert}_{self.args.ticker_name}_5m.json', 'r') as f:
                    self.features[expert] = json.load(f)
            self.scaler_pa = joblib.load(f'scaler_pa_{self.args.ticker_name}_5m.joblib')
            print("Wszystkie modele i artefakty ML załadowane pomyślnie.")
        except FileNotFoundError as e:
            sys.exit(f"Nie znaleziono plików modelu: {e.filename}")

    def _get_analysis_from_row(self, data_row: pd.Series) -> dict:
        expert_opinions = {}
        for expert in ['momentum', 'reversion', 'pa']:
            X_df = pd.DataFrame([data_row[self.features[expert]]])

            if expert == 'pa':
                X_df.replace([np.inf, -np.inf], 0, inplace=True);
                X_df.fillna(0, inplace=True)
                X = self.scaler_pa.transform(X_df)
            else:
                X = X_df

            prediction = int(self.models[expert].predict(X)[0])
            confidence = float(self.models[expert].predict_proba(X).max())
            expert_opinions[expert] = {"prediction": prediction, "confidence": confidence}

        return {
            "current_price": float(data_row['close']),
            "atr_value_5m": float(data_row['ATRr_14_5m']),
            "expert_opinions": expert_opinions
        }

    def run_backtest(self):
        args = self.args
        print("\n--- URUCHAMIANIE BOTA W TRYBIE BACKTESTU ---")

        print(f"Pobieranie danych historycznych dla {args.ticker} od {args.start_date} do {args.end_date}...")
        all_data = []
        start_ts = int(datetime.strptime(args.start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(args.end_date, "%Y-%m-%d").timestamp() * 1000)
        current_ts = start_ts
        while current_ts < end_ts:
            response = self.session.get_kline(category="linear", symbol=args.ticker, interval=5, start=current_ts,
                                              limit=1000)
            if response.get('retCode') == 0 and response.get('result', {}).get('list'):
                data_chunk = response['result']['list']
                if not data_chunk: break
                data_chunk.sort(key=lambda k: int(k[0]))
                all_data.extend(data_chunk)
                current_ts = int(data_chunk[-1][0]) + (5 * 60 * 1000)
            else:
                break
            time.sleep(0.2)

        if not all_data: print("BŁĄD: Nie udało się pobrać danych."); return

        df_raw = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms')
        df_raw.set_index('timestamp', inplace=True)
        df_raw = df_raw.astype(float).sort_index().drop_duplicates()

        data_filename = f"backtest_data_{args.ticker}_{args.start_date}_to_{args.end_date}.csv"
        df_raw.to_csv(data_filename)
        print(f"Zapisano surowe dane testowe do pliku: {data_filename}")

        test_data = prepare_full_feature_set(df_raw)

        capital = args.initial_capital
        trades, active_positions = [], {}
        equity_curve = {test_data.index[0]: capital}

        print("Uruchamianie symulacji z prawdziwymi modelami ML...")
        pbar = tqdm(range(len(test_data)))
        for i in pbar:
            current_candle = test_data.iloc[i]

            analysis = self._get_analysis_from_row(current_candle)
            if not analysis: continue

            if args.debug:
                confidences = {expert: f"{opinion['confidence']:.2f}" for expert, opinion in
                               analysis['expert_opinions'].items()}
                pbar.set_description(f"Confidence: {confidences}")

            votes_long, votes_short = 0, 0
            for opinion in analysis['expert_opinions'].values():
                if opinion['confidence'] >= args.min_confidence:
                    if opinion['prediction'] == 1:
                        votes_long += 1
                    else:
                        votes_short += 1

            for strategy in list(active_positions.keys()):
                pos = active_positions[strategy];
                exit_reason = None
                if (strategy == 'long' and votes_short >= 2) or (strategy == 'short' and votes_long >= 2):
                    exit_reason = "Model Exit Signal"
                elif strategy == 'long' and current_candle['low'] <= pos['sl_price']:
                    exit_reason = "Stop Loss"
                elif strategy == 'long' and current_candle['high'] >= pos['tp_price']:
                    exit_reason = "Take Profit"

                if exit_reason:
                    exit_price = current_candle['close'] if "Model Exit" in exit_reason else (
                        pos['sl_price'] if "Stop Loss" in exit_reason else pos['tp_price'])
                    pnl = (exit_price - pos['entry_price']) * pos['size']
                    capital += pnl
                    pos.update({'exit_date': current_candle.name, 'exit_price': exit_price, 'pnl_usd': pnl,
                                'exit_reason': exit_reason, 'strategy': strategy,
                                'conf_momentum': analysis['expert_opinions']['momentum']['confidence'],
                                'conf_reversion': analysis['expert_opinions']['reversion']['confidence'],
                                'conf_pa': analysis['expert_opinions']['pa']['confidence']})
                    trades.append(pos)
                    del active_positions[strategy]

            strategy_to_open = None
            if votes_long >= 2 and 'long' not in active_positions:
                strategy_to_open = 'long'
            elif votes_short >= 2 and 'short' not in active_positions:
                strategy_to_open = 'short'

            if strategy_to_open:
                entry_price = analysis['current_price']
                stop_loss_distance = analysis['atr_value_5m'] * args.atr_multiplier
                sl_price = entry_price - stop_loss_distance if strategy_to_open == 'long' else entry_price + stop_loss_distance
                tp_price = entry_price + (
                            abs(entry_price - sl_price) * args.rrr) if strategy_to_open == 'long' else entry_price - (
                            abs(entry_price - sl_price) * args.rrr)
                position_value = capital * args.risk_percent * float(args.leverage)
                position_size = position_value / entry_price if entry_price > 0 else 0

                active_positions[strategy_to_open] = {'entry_date': current_candle.name, 'entry_price': entry_price,
                                                      'size': position_size, 'sl_price': sl_price, 'tp_price': tp_price}
                capital -= args.trade_cost

            equity_curve[current_candle.name] = capital

        self._generate_backtest_report(trades, equity_curve, capital)

    def _generate_backtest_report(self, trades, equity_curve, final_capital):
        args = self.args;
        report_filename = f"backtest_report_{args.ticker}_{args.start_date}_to_{args.end_date}.csv"
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df.to_csv(report_filename, index=False)
            print(f"\nZapisano raport z transakcji do pliku: {report_filename}")
        equity_series = pd.Series(equity_curve)
        plot_equity_and_drawdown(equity_series, args.ticker)
        print("\n--- WYNIKI BACKTESTU ---")
        print(f"Testowany okres: od {args.start_date} do {args.end_date}")
        print(f"Kapitał początkowy: ${args.initial_capital:,.2f}")
        print(f"Kapitał końcowy: ${final_capital:,.2f}")
        pnl_total = final_capital - args.initial_capital
        print(f"Zysk/Strata (P/L): ${pnl_total:,.2f} ({(pnl_total / args.initial_capital * 100):.2f}%)")
        if not trades_df.empty:
            wins = trades_df[trades_df['pnl_usd'] > 0];
            losses = trades_df[trades_df['pnl_usd'] <= 0]
            print("-" * 40)
            print(f"Liczba transakcji: {len(trades_df)}")
            print(f"Procent zyskownych (Win Rate): {len(wins) / len(trades_df) * 100:.2f}%")
            if abs(losses['pnl_usd'].sum()) > 0:
                print(f"Profit Factor: {wins['pnl_usd'].sum() / abs(losses['pnl_usd'].sum()):.2f}")
            else:
                print("Profit Factor: N/A (brak stratnych transakcji)")
            trades_df['duration'] = trades_df['exit_date'] - trades_df['entry_date']
            print(f"Średni czas trwania pozycji: {trades_df['duration'].mean()}")
            print("\nRozkład powodów zamknięcia pozycji:")
            print(trades_df['exit_reason'].value_counts(normalize=True).apply("{:.2%}".format))

            # --- NOWA SEKCJA: STATYSTYKI PEWNOŚCI MODELI ---
            print("\n--- Statystyki Pewności Modeli (dla zrealizowanych transakcji) ---")
            avg_conf_momentum = trades_df['conf_momentum'].mean()
            avg_conf_reversion = trades_df['conf_reversion'].mean()
            avg_conf_pa = trades_df['conf_pa'].mean()

            all_confs = pd.concat([trades_df['conf_momentum'], trades_df['conf_reversion'], trades_df['conf_pa']])
            avg_conf_overall = all_confs.mean()

            print(f"Średnia pewność (Momentum): {avg_conf_momentum:.2%}")
            print(f"Średnia pewność (Reversion): {avg_conf_reversion:.2%}")
            print(f"Średnia pewność (Price Action): {avg_conf_pa:.2%}")
            print(f"Średnia pewność (Ogółem): {avg_conf_overall:.2%}")

    def run_live(self):
        print("Tryb LIVE nie jest jeszcze w pełni zaimplementowany w tej strukturze.")


def plot_equity_and_drawdown(equity_series, ticker):
    plt.style.use('seaborn-v0_8-darkgrid');
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Wyniki Strategii - {ticker}', fontsize=16);
    equity_series.plot(ax=ax1, lw=2, title='Krzywa Kapitału')
    ax1.set_ylabel('Kapitał ($)');
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    running_max = equity_series.cummax();
    drawdown = (equity_series - running_max) / running_max * 100
    drawdown.plot(ax=ax2, kind='area', color='red', alpha=0.4, title='Obsunięcie Kapitału (Drawdown)')
    ax2.set_ylabel('Drawdown (%)');
    ax2.set_xlabel('Data');
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
    plt.savefig(f'equity_and_drawdown_{ticker}.png');
    plt.close()
    print(f"\nZapisano zintegrowany wykres kapitału i drawdown do pliku: equity_and_drawdown_{ticker}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot handlowy z trybem live i backtest opartym o te same modele ML.")

    parser.add_argument("--mode", type=str, default="live", choices=['live', 'backtest'], help="Tryb pracy bota.")
    parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER, help="Symbol do handlu lub testowania.")
    parser.add_argument("--ticker-name", type=str, default="ETH",
                        help="Nazwa tickera używana w nazwach plików modeli (np. ETH, ICP).")
    parser.add_argument("--start-date", type=str, default="2025-09-01",
                        help="Data początkowa dla backtestu (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help="Data końcowa dla backtestu (YYYY-MM-DD).")
    parser.add_argument("--initial-capital", type=float, default=3700.0, help="Kapitał początkowy dla backtestu.")
    parser.add_argument("--risk-percent", type=float, default=DEFAULT_RISK,
                        help="Procent kapitału ryzykowany na transakcję.")
    parser.add_argument("--leverage", type=str, default=DEFAULT_LEVERAGE, help="Dźwignia finansowa.")
    parser.add_argument("--atr-multiplier", type=float, default=DEFAULT_ATR_MULTIPLIER,
                        help="Mnożnik ATR dla Stop Lossa.")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                        help="Minimalna pewność modeli do otwarcia pozycji.")
    parser.add_argument("--rrr", type=float, default=DEFAULT_RRR, help="Risk-Reward Ratio (TP/SL).")
    parser.add_argument("--trade-cost", type=float, default=DEFAULT_TRADE_COST, help="Koszt otwarcia pozycji w USD.")
    parser.add_argument("--debug", action='store_true',
                        help="Włącz tryb debugowania, aby wyświetlać confidence w czasie rzeczywistym.")

    args = parser.parse_args()

    bot = TradingBot(args)
    if args.mode == 'backtest':
        bot.run_backtest()
    else:
        bot.run_live()