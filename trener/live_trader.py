#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_trader.py
--------------
Ostateczna wersja live-runnera. W trybie "paper trading" wszystkie zlecenia
(otwarcie, zamknięcie, SL/TP) są faktycznie wysyłane do API demo Bybit,
zapewniając pełną synchronizację i realistyczne testowanie.

- Wykorzystuje BybitAdapter do wszystkich interakcji z API.
- Usunięto logikę, która uniemożliwiała wysyłanie zleceń w trybie papierowym.
- Logika handlowa jest w pełni spójna z pro_backtester.py.
"""

import os
import time
import json
import math
import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# === 1) Import modułów ===
try:
    from data_preparer import prepare_feature_set_for_timeframe as PREPARE_FEATS
    from bybit_adapter import BybitAdapter, BybitAPIError
except ImportError as e:
    raise RuntimeError(f"Nie udało się zaimportować wymaganych modułów: {e}")


def _norm_symbol(symbol: str) -> str:
    """Normalizuje symbol do formatu wymaganego przez API."""
    if ":" in symbol:
        symbol = symbol.split(":")[0]
    symbol = symbol.replace("/", "")
    return symbol


# === 2) Konwersja Klines na DataFrame ===
class _Klines:
    COLS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]

    @staticmethod
    def to_df(klines: List[List[float]]) -> pd.DataFrame:
        if not klines:
            raise ValueError("Brak danych klines")

        df = pd.DataFrame(klines, columns=_Klines.COLS[:len(klines[0])])
        if "turnover" not in df.columns:
            df["turnover"] = 0.0

        for c in _Klines.COLS[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True).dt.tz_localize(None)

        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["open", "high", "low", "close", "volume"]
        )
        return df


# === 3) Helper: Wybierz ostatni kompletny wiersz cech ===
def _pick_last_complete_row(df: pd.DataFrame, required_cols: List[str], lookback: int = 50) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    tail = df.tail(lookback)
    missing = [c for c in required_cols if c not in tail.columns]
    if missing:
        raise ValueError(f"[LIVE] Brak kolumn cech: {missing[:10]}")
    view = tail[required_cols]
    complete = view.dropna(how="any")
    if complete.empty:
        return None
    return complete.tail(1)


# === 4) Wrapper modelu ===
class _Model:
    def __init__(self, model, scaler, best_features: List[str]):
        self.model = model
        self.scaler = scaler
        self.feature_order = list(getattr(scaler, "feature_names_in_", best_features))

    def proba(self, feats_df: pd.DataFrame) -> Optional[Tuple[float, float]]:
        row = _pick_last_complete_row(feats_df, self.feature_order)
        if row is None:
            return None
        Xs = self.scaler.transform(row[self.feature_order])
        proba = self.model.predict_proba(Xs)[0]
        return float(proba[0]), float(proba[1])


# === 5) Zarządzanie stanem pozycji ===
@dataclass
class Position:
    side: str
    entry_price: float
    qty: float
    tsl_price: float
    tp_price: Optional[float]
    open_time: pd.Timestamp
    highest_close_since_entry: Optional[float] = None
    lowest_close_since_entry: Optional[float] = None

    def unrealized_pnl(self, last_price: float) -> float:
        delta = (last_price - self.entry_price) if self.side == "long" else (self.entry_price - last_price)
        return delta * self.qty


# === 6) Główna klasa logiki handlowej ===
class LiveTrader:
    def __init__(self, cfg, model: _Model, adapter: BybitAdapter):
        self.cfg = cfg
        self.model = model
        self.adapter = adapter
        self.position: Optional[Position] = None
        self.symbol_u = _norm_symbol(cfg.symbol)

    def run(self):
        logging.info(
            f"Start LiveTrader dla {self.cfg.symbol} @ {self.cfg.timeframe} | Paper Mode: {self.cfg.use_paper}")
        self._sync_position_from_exchange()

        while True:
            try:
                klines = self.adapter.fetch_ohlcv(self.cfg.symbol, self.cfg.timeframe, limit=self.cfg.hist_limit)
                df_raw = _Klines.to_df(klines)
                feats = PREPARE_FEATS(df_raw, base_tf=self.cfg.timeframe)

                last_price = self.adapter.latest_price(self.symbol_u)
                if last_price <= 0:
                    logging.warning("Nie udało się pobrać ostatniej ceny, pomijam cykl.")
                    time.sleep(self.cfg.poll_sec)
                    continue

                atr_val = self._calculate_atr(df_raw)
                if atr_val is None:
                    logging.warning("Nie udało się obliczyć ATR, pomijam cykl.")
                    time.sleep(self.cfg.poll_sec)
                    continue

                probas = self.model.proba(feats)
                if probas is None:
                    logging.warning("Brak kompletnego wiersza cech, pomijam cykl.")
                    time.sleep(self.cfg.poll_sec)
                    continue

                p_short, p_long = probas
                logging.info(f"Cena={last_price:.6f} | ATR={atr_val:.6f} | P(long)={p_long:.3f} P(short)={p_short:.3f}")

                self._process_cycle(p_long, p_short, last_price, atr_val, feats.index[-1], df_raw.iloc[-1]['close'])

            except KeyboardInterrupt:
                logging.info("Trader zatrzymany przez użytkownika.")
                break
            except Exception as e:
                logging.error(f"Wystąpił błąd w głównej pętli: {e}", exc_info=True)

            time.sleep(self.cfg.poll_sec)

    def _process_cycle(self, p_long: float, p_short: float, current_price: float, atr_val: float, last_ts: pd.Timestamp,
                       last_close: float):
        signal = 0
        if p_long >= self.cfg.min_conf_long:
            signal = 1
        elif p_short >= self.cfg.min_conf_short:
            signal = -1

        if self.position:
            self._update_tsl(last_close, atr_val)
            exit_reason = self._check_for_exit(current_price)
            if exit_reason:
                self._execute_close(current_price, reason=exit_reason)
                return

            if signal == 0 and self.cfg.flat_on_low_conf != "off":
                self._handle_flat_on_low_conf(current_price)
            elif (signal == 1 and self.position.side == "short") or (signal == -1 and self.position.side == "long"):
                self._execute_close(current_price, reason="reverse")
                self._execute_open("long" if signal == 1 else "short", current_price, atr_val, last_ts)
            else:
                pnl = self.position.unrealized_pnl(current_price)
                logging.info(
                    f"[W POZYCJI] Strona: {self.position.side.upper()}, Wejście: {self.position.entry_price:.6f}, "
                    f"TSL: {self.position.tsl_price:.6f}, TP: {self.position.tp_price}, Niezreal. PnL: {pnl:.4f}")
        else:
            if signal == 1:
                self._execute_open("long", current_price, atr_val, last_ts)
            elif signal == -1:
                self._execute_open("short", current_price, atr_val, last_ts)
            else:
                logging.info("[HOLD] Brak pozycji i sygnału wejścia.")

    def _execute_open(self, side: str, price: float, atr_val: float, timestamp: pd.Timestamp):
        qty, stop_price, tp_price = self._calculate_position_size(price, atr_val, side)
        if qty <= 0:
            logging.warning("Obliczona wielkość pozycji <= 0. Pomijam wejście.")
            return

        try:
            logging.info(f"Wysyłanie zlecenia otwarcia: {side.upper()} {qty} {self.symbol_u} @ Market")
            self.adapter.set_leverage(self.symbol_u, self.cfg.leverage)
            order_side = "Buy" if side == "long" else "Sell"
            self.adapter.market_open(self.symbol_u, order_side, qty)

            self.adapter.set_stop_loss(self.symbol_u, stop_price, "Sell" if side == "long" else "Buy")
            if tp_price:
                self.adapter.set_take_profit(self.symbol_u, tp_price, "Sell" if side == "long" else "Buy")

            self.position = Position(
                side=side, entry_price=price, qty=qty,
                tsl_price=stop_price, tp_price=tp_price,
                open_time=timestamp,
                highest_close_since_entry=price if side == "long" else None,
                lowest_close_since_entry=price if side == "short" else None
            )
            logging.info(
                f"[OTWARCIE POTWIERDZONE] {side.upper()} | Ilość: {qty} | Cena: {price:.6f} | TSL: {stop_price:.6f} | TP: {tp_price}")

        except BybitAPIError as e:
            logging.error(f"Błąd API podczas otwierania pozycji: {e}")
            self.position = None

    def _execute_close(self, price: float, reason: str):
        if not self.position: return
        pnl = self.position.unrealized_pnl(price)

        try:
            logging.info(f"Wysyłanie zlecenia zamknięcia dla pozycji {self.position.side.upper()} z powodu: {reason}")
            self.adapter.close_position(self.symbol_u)
            self.adapter.cancel_tpsl(self.symbol_u)

            log_msg = f"[ZAMKNIĘCIE POTWIERDZONE] Powód: {reason} | Cena: {price:.6f} | PnL: {pnl:.4f}"
            balance = self.adapter.get_balance()
            log_msg += f" | Aktualne saldo: {balance:.2f}"
            logging.info(log_msg)

            self.position = None

        except BybitAPIError as e:
            logging.error(f"Błąd API podczas zamykania pozycji: {e}")

    def _calculate_position_size(self, price: float, atr: float, side: str) -> Tuple[float, float, Optional[float]]:
        stop_dist = max(1e-9, self.cfg.atr_mult_stop * atr)
        if self.cfg.min_stop_pct_of_price > 0:
            stop_dist = max(stop_dist, price * self.cfg.min_stop_pct_of_price)

        try:
            base_equity = self.adapter.get_balance(use_available=False)
            if base_equity <= 0:
                logging.warning("Saldo <= 0. Używam domyślnej wartości 10000.")
                base_equity = 10000.0
        except BybitAPIError as e:
            logging.error(f"Nie udało się pobrać salda: {e}. Używam domyślnej wartości 10000.")
            base_equity = 10000.0

        risk_cash = base_equity * self.cfg.risk_fraction
        qty = (risk_cash * self.cfg.leverage) / stop_dist
        cap = base_equity * self.cfg.leverage * self.cfg.max_notional_frac
        if qty * price > cap:
            qty = cap / price

        qty = round(qty)

        stop_price = price - stop_dist if side == "long" else price + stop_dist
        tp_price = price + self.cfg.atr_mult_tp * atr if self.cfg.atr_mult_tp and side == "long" else \
            price - self.cfg.atr_mult_tp * atr if self.cfg.atr_mult_tp and side == "short" else None

        stop_price = round(stop_price, 6)
        if tp_price is not None:
            tp_price = round(tp_price, 6)

        return qty, stop_price, tp_price

    def _update_tsl(self, last_close: float, atr_val: float):
        if not self.position: return

        new_stop, side_to_update = None, None
        if self.position.side == "long":
            self.position.highest_close_since_entry = max(self.position.highest_close_since_entry or last_close,
                                                          last_close)
            candidate = self.position.highest_close_since_entry - self.cfg.atr_mult_stop * atr_val

            candidate = round(candidate, 6)

            if candidate > self.position.tsl_price:
                self.position.tsl_price = candidate
                new_stop, side_to_update = candidate, "Sell"
        else:  # short
            self.position.lowest_close_since_entry = min(self.position.lowest_close_since_entry or last_close,
                                                         last_close)
            candidate = self.position.lowest_close_since_entry + self.cfg.atr_mult_stop * atr_val

            # === POPRAWKA: Zaokrąglanie kandydata na nowy stop loss ===
            candidate = round(candidate, 6)

            if candidate < self.position.tsl_price:
                self.position.tsl_price = candidate
                new_stop, side_to_update = candidate, "Buy"

        if new_stop:
            logging.info(f"Aktualizacja TSL dla {self.position.side.upper()} do {new_stop:.6f}")
            try:
                self.adapter.set_stop_loss(self.symbol_u, new_stop, side_to_update)
            except BybitAPIError as e:
                # === POPRAWKA: Lepsza obsługa błędu "not modified" ===
                if "not modified" in str(e) or "34040" in str(e):
                    logging.info(f"TSL nie został zmodyfikowany (prawdopodobnie ta sama cena po zaokrągleniu).")
                else:
                    logging.warning(f"Błąd API podczas aktualizacji TSL: {e}")

    def _check_for_exit(self, price: float) -> Optional[str]:
        if not self.position: return None
        if self.position.side == "long":
            if self.position.tp_price and price >= self.position.tp_price: return "TP"
            if price <= self.position.tsl_price: return "TSL"
        else:
            if self.position.tp_price and price <= self.position.tp_price: return "TP"
            if price >= self.position.tsl_price: return "TSL"
        return None

    def _handle_flat_on_low_conf(self, price: float):
        if not self.position or self.cfg.flat_on_low_conf == "off": return
        pnl = self.position.unrealized_pnl(price)
        do_flat = False
        if self.cfg.flat_on_low_conf == "always":
            do_flat = True
        elif self.cfg.flat_on_low_conf == "loss_only" and pnl < 0:
            do_flat = True
        elif self.cfg.flat_on_low_conf == "not_protected":
            is_protected = (self.position.side == "long" and self.position.tsl_price > self.position.entry_price) or \
                           (self.position.side == "short" and self.position.tsl_price < self.position.entry_price)
            if not is_protected: do_flat = True
        if do_flat: self._execute_close(price, reason="flat_on_low_conf")

    def _calculate_atr(self, df_ohlc: pd.DataFrame, length: int = 14) -> Optional[float]:
        try:
            import pandas_ta as ta
            atr = ta.atr(high=df_ohlc["high"], low=df_ohlc["low"], close=df_ohlc["close"], length=length)
            val = float(atr.dropna().iloc[-1])
            return val if np.isfinite(val) else None
        except Exception:
            return None

    def _sync_position_from_exchange(self):
        logging.info("Synchronizuję stan pozycji z giełdą...")
        try:
            pos_data = self.adapter.get_position(self.symbol_u)
            if pos_data and float(pos_data.get('size', 0)) > 0:
                side = "long" if pos_data['side'] == 'Long' else "short"
                self.position = Position(
                    side=side, entry_price=float(pos_data['entryPrice']),
                    qty=float(pos_data['size']), tsl_price=0, tp_price=None,
                    open_time=pd.Timestamp.utcnow(),
                )
                logging.info(f"Znaleziono istniejącą pozycję na giełdzie: {self.position}")
            else:
                self.position = None
                logging.info("Brak aktywnej pozycji na giełdzie.")
        except BybitAPIError as e:
            logging.error(f"Nie udało się zsynchronizować pozycji: {e}")
            self.position = None


# === 7) Konfiguracja CLI i główne wykonanie ===
@dataclass
class Cfg:
    api_base: Optional[str]
    api_key: str
    api_secret: str
    symbol: str
    timeframe: str
    hist_limit: int
    model_path: str
    scaler_path: str
    best_features_path: str
    min_conf_long: float
    min_conf_short: float
    risk_fraction: float
    atr_mult_stop: float
    atr_mult_tp: Optional[float]
    min_stop_pct_of_price: float
    leverage: float
    max_notional_frac: float
    poll_sec: float
    flat_on_low_conf: str
    use_paper: bool


def main():
    parser = argparse.ArgumentParser(description="Live Trader z pełną synchronizacją API w trybie papierowym")
    parser.add_argument("--api_base", default=os.getenv("BYBIT_API_BASE", "https://api-demo.bybit.com"))
    parser.add_argument("--api_key", default=os.getenv("BYBIT_API_KEY"))
    parser.add_argument("--api_secret", default=os.getenv("BYBIT_API_SECRET"))
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--hist_limit", type=int, default=2000)
    parser.add_argument("--model_path", default="final_model.joblib")
    parser.add_argument("--scaler_path", default="final_scaler.joblib")
    parser.add_argument("--best_features_path", default="best_features.json")
    parser.add_argument("--min_conf_long", type=float, default=0.70)
    parser.add_argument("--min_conf_short", type=float, default=0.70)
    parser.add_argument("--risk_fraction", type=float, default=0.02)
    parser.add_argument("--atr_mult_stop", type=float, default=2.0)
    parser.add_argument("--atr_mult_tp", type=str, default="3.0", help='Mnożnik dla TP lub "none"')
    parser.add_argument("--min_stop_pct_of_price", type=float, default=0.0)
    parser.add_argument("--leverage", type=float, default=5.0)
    parser.add_argument("--max_notional_frac", type=float, default=1.0)
    parser.add_argument("--flat_on_low_conf", choices=["off", "always", "loss_only", "not_protected"], default="off")
    parser.add_argument("--poll_sec", type=float, default=10.0)
    parser.add_argument("--paper", dest="use_paper", action="store_true",
                        help="Włącz tryb handlu papierowego (na koncie demo)")
    parser.add_argument("--live", dest="use_paper", action="store_false", help="Włącz tryb handlu na żywo")
    parser.set_defaults(use_paper=True)
    args = parser.parse_args()

    if (not args.use_paper or args.api_base == "https://api.bybit.com") and (not args.api_key or not args.api_secret):
        raise ValueError(
            "Klucze API są wymagane w trybie live. Użyj --api_key i --api_secret lub ustaw zmienne środowiskowe.")

    atr_tp = None if str(args.atr_mult_tp).lower() in ("none", "null") else float(args.atr_mult_tp)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = Cfg(
        api_base=args.api_base, api_key=args.api_key, api_secret=args.api_secret,
        symbol=args.symbol, timeframe=args.timeframe, hist_limit=args.hist_limit,
        model_path=args.model_path, scaler_path=args.scaler_path, best_features_path=args.best_features_path,
        min_conf_long=args.min_conf_long, min_conf_short=args.min_conf_short,
        risk_fraction=args.risk_fraction, atr_mult_stop=args.atr_mult_stop, atr_mult_tp=atr_tp,
        min_stop_pct_of_price=args.min_stop_pct_of_price, leverage=args.leverage,
        max_notional_frac=args.max_notional_frac,
        poll_sec=args.poll_sec, flat_on_low_conf=args.flat_on_low_conf,
        use_paper=args.use_paper
    )

    adapter = BybitAdapter(api_key=cfg.api_key, api_secret=cfg.api_secret, base_url=cfg.api_base)
    model_sk = joblib.load(cfg.model_path)
    scaler = joblib.load(cfg.scaler_path)
    with open(cfg.best_features_path, "r") as f:
        best_features = json.load(f)
    model = _Model(model_sk, scaler, best_features)

    trader = LiveTrader(cfg, model, adapter)
    trader.run()


if __name__ == "__main__":
    main()