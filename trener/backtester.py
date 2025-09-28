# backtester.py
import backtrader as bt
import pandas as pd
import joblib
import json
import numpy as np
import re
import sys
import datetime
import logging
import os
from tqdm import tqdm
import warnings

# Ignorujemy specyficzne ostrzeżenie o braku nazw cech, aby nie zaśmiecać konsoli.
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")


def get_data(datapath, start_date, end_date):
    """
    Wczytuje dane z pliku CSV i przechowuje je w pamięci podręcznej (cache),
    aby uniknąć wielokrotnego odczytu z dysku podczas optymalizacji.
    """
    _data_cache = {}
    if datapath not in _data_cache:
        logging.info(f"Pierwsze uruchomienie: wczytywanie danych z pliku '{datapath}' do pamięci...")
        df = pd.read_csv(datapath, index_col='timestamp', parse_dates=True)
        _data_cache[datapath] = df
        logging.info("Dane zostały załadowane i zapisane w cache.")
    else:
        logging.info("Korzystanie z danych załadowanych do pamięci podręcznej (cache).")
    return _data_cache[datapath].loc[start_date:end_date]


def sanitize_name(name):
    """Zastępuje wszystkie znaki nieprawidłowe dla identyfikatorów Pythona na podkreślniki."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


class MLStrategy(bt.Strategy):
    params = (
        ('risk_pct', 0.02),  # Ryzyko na transakcję (2% kapitału)
        ('tp_atr_multiplier', 1.5),  # Mnożnik ATR dla Take Profit
        ('sl_atr_multiplier', 1.5),  # Mnożnik ATR dla Stop Loss
        ('tsl_atr_multiplier', 2.0),  # Mnożnik ATR dla Trailing Stop
        ('confidence_threshold', 0.60),  # Minimalna pewność modelu do otwarcia pozycji
        ('all_feature_columns', []),
        ('leverage', 5.0)
    )

    def __init__(self):
        logging.info("--- Inicjalizacja Strategii ---")
        self.model = joblib.load("final_model.joblib")
        self.scaler = joblib.load("final_scaler.joblib")
        with open('best_features.json', 'r') as f:
            self.best_features = json.load(f)

        self.feature_lines = {}
        for feature in self.best_features:
            sanitized_name = sanitize_name(feature)
            self.feature_lines[feature] = getattr(self.datas[0].lines, sanitized_name)

        self.atr = bt.indicators.ATR(self.datas[0], period=14)

        self.order_entry = None
        self.order_stop_loss = None
        self.order_take_profit = None
        logging.info("--- Inicjalizacja Zakończona ---")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            if self.order_entry and order.ref == self.order_entry.ref:
                self.order_entry = None
            elif self.order_stop_loss and order.ref == self.order_stop_loss.ref:
                self.order_stop_loss = None
            elif self.order_take_profit and order.ref == self.order_take_profit.ref:
                self.order_take_profit = None

    def next(self):
        # 1. Zarządzanie otwartą pozycją (Trailing Stop Loss)
        if self.position:
            current_price = self.datas[0].close[0]
            current_atr = self.atr[0]

            if self.position.size > 0 and self.order_stop_loss:
                new_stop_price = current_price - self.p.tsl_atr_multiplier * current_atr
                if new_stop_price > self.order_stop_loss.p.price:
                    self.broker.cancel(self.order_stop_loss)
                    self.order_stop_loss = self.sell(exectype=bt.Order.Stop, price=new_stop_price,
                                                     size=self.position.size)
                    logging.info(f"Trailing Stop dla LONG przesunięty na: {new_stop_price:.5f}")

            elif self.position.size < 0 and self.order_stop_loss:
                new_stop_price = current_price + self.p.tsl_atr_multiplier * current_atr
                if new_stop_price < self.order_stop_loss.p.price:
                    self.broker.cancel(self.order_stop_loss)
                    self.order_stop_loss = self.buy(exectype=bt.Order.Stop, price=new_stop_price,
                                                    size=self.position.size)
                    logging.info(f"Trailing Stop dla SHORT przesunięty na: {new_stop_price:.5f}")
            return

        # 2. Logika wejścia w nową pozycję
        if self.order_entry:
            return

        current_features = [self.feature_lines[f][0] for f in self.best_features]
        scaled_features = self.scaler.transform(np.array([current_features]))
        prediction = self.model.predict(scaled_features)[0]
        confidence = self.model.predict_proba(scaled_features)[0][int(prediction)]
        signal = 'LONG' if prediction == 1 else 'SHORT'

        if confidence >= self.p.confidence_threshold:
            atr_val = self.atr[0]
            if atr_val <= 0: return

            entry_price = self.datas[0].close[0]

            risk_per_unit = atr_val * self.p.sl_atr_multiplier
            if risk_per_unit <= 0: return

            risk_amount = self.broker.getvalue() * self.p.risk_pct

            final_size = risk_amount / risk_per_unit

            logging.info(f"NOWY SYGNAŁ {signal} | Cena: {entry_price:.5f} | Wielkość (z 2% MM): {final_size:.2f}")

            if signal == 'LONG':
                self.order_entry = self.buy(size=final_size)
                sl_price = entry_price - risk_per_unit
                tp_price = entry_price + atr_val * self.p.tp_atr_multiplier
                self.order_stop_loss = self.sell(exectype=bt.Order.Stop, price=sl_price, size=final_size)
                self.order_take_profit = self.sell(exectype=bt.Order.Limit, price=tp_price, size=final_size)

            elif signal == 'SHORT':
                self.order_entry = self.sell(size=final_size)
                sl_price = entry_price + risk_per_unit
                tp_price = entry_price - atr_val * self.p.tp_atr_multiplier
                self.order_stop_loss = self.buy(exectype=bt.Order.Stop, price=sl_price, size=final_size)
                self.order_take_profit = self.buy(exectype=bt.Order.Limit, price=tp_price, size=final_size)


if __name__ == '__main__':
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('logs', exist_ok=True)
    log_filename = f'logs/backtest_{run_timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)],
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    BACKTEST_START_DATE = datetime.datetime(2025, 8, 15)
    BACKTEST_END_DATE = datetime.datetime(2025, 9, 15)
    LEVERAGE = 5.0

    cerebro = bt.Cerebro()
    datapath = 'data_for_backtest.csv'

    all_feature_columns = pd.read_csv(datapath, nrows=0).columns[7:].tolist()


    class PandasDataWithFeatures(bt.feeds.PandasData):
        lines = tuple(sanitize_name(name) for name in all_feature_columns)
        params = tuple([('datetime', None), ('openinterest', -1)] +
                       [(sanitize_name(name), -1) for name in all_feature_columns])


    dataframe = get_data(datapath, BACKTEST_START_DATE, BACKTEST_END_DATE)
    for col in all_feature_columns:
        if col not in dataframe.columns:
            dataframe[col] = 0

    data = PandasDataWithFeatures(dataname=dataframe)

    cerebro.adddata(data)
    cerebro.addstrategy(MLStrategy, all_feature_columns=all_feature_columns, leverage=LEVERAGE)

    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001, leverage=LEVERAGE)
    cerebro.broker.set_slippage_perc(perc=0.001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    logging.info(f'--- Uruchamianie Backtestu od {BACKTEST_START_DATE.date()} do {BACKTEST_END_DATE.date()} ---')
    results = cerebro.run()
    logging.info('--- Backtest Zakończony ---')

    strat = results[0]
    logging.info('\n--- WYNIKI STRATEGII ---')
    logging.info(f"Kapitał końcowy: {cerebro.broker.getvalue():.2f}")

    analysis = {
        "sharpe_ratio": strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0),
        "max_drawdown": strat.analyzers.drawdown.get_analysis().max.drawdown,
        "total_trades": strat.analyzers.trades.get_analysis().get('total', {}).get('total', 0),
        "won_trades": strat.analyzers.trades.get_analysis().get('won', {}).get('total', 0)
    }

    if analysis['max_drawdown'] is None:
        analysis['max_drawdown'] = 0.0

    logging.info(f"Annualized Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
    logging.info(f"Max. Drawdown: {analysis['max_drawdown']:.2f}%")
    logging.info(f"Liczba transakcji: {analysis['total_trades']}")
    if analysis['total_trades'] > 0:
        win_rate = (analysis['won_trades'] / analysis['total_trades']) * 100
        logging.info(f"Win rate: {win_rate:.2f}%")
    else:
        logging.info("Brak transakcji do analizy.")