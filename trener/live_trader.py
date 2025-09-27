# live_trader.py — Bybit (pybit) LIVE/paper trading
# Cechy 1:1 z treningiem: korzystamy z data_preparer.prepare_feature_set_for_timeframe
# Naprawa KeyError('MACDh_12_26_9'): deep patch pandas_ta (moduł, akcesor, submoduły) PRZED importem data_preparer

from __future__ import annotations

import os, sys, time, math, json, argparse, logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import joblib

# ---------- Bybit adapter ----------
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
try:
    from bybit_adapter import BybitAdapter
except ModuleNotFoundError as e:
    raise SystemExit("Brak pliku bybit_pybit_adapter.py w katalogu z live_trader.py") from e

# ---------- config ----------
try:
    import config
except ModuleNotFoundError as e:
    raise SystemExit("Nie znaleziono config.py obok live_trader.py") from e

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("live")


# -------------------- narzędzia & wersje --------------------

def _log_versions():
    import pandas as pd, numpy as np
    try:
        import pandas_ta as ta
        ta_ver = getattr(ta, "version", None) or getattr(ta, "__version__", "unknown")
    except Exception:
        ta_ver = "NOT INSTALLED"
    log.info(f"ENV -> pandas={pd.__version__}, numpy={np.__version__}, pandas_ta={ta_ver}")


def _assert_best_features(full_feats: pd.DataFrame, best_features_path: str = "best_features.json"):
    with open(best_features_path, "r") as f:
        req = json.load(f)
    missing = [c for c in req if c not in full_feats.columns]
    if missing:
        preview = ", ".join(missing[:30]) + (" ..." if len(missing) > 30 else "")
        raise SystemExit(
            f"[LIVE] Brakuje {len(missing)} kolumn z best_features.json.\n"
            f"Przykład: {preview}\n"
            f"Upewnij się, że środowisko i generator cech są identyczne jak w treningu."
        )


def compute_atr_from_ohlc(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> float:
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    if len(tr) < period:
        return float(np.nan)
    return float(pd.Series(tr).rolling(period, min_periods=period).mean().iloc[-2])


def ccxt_style_to_bybit_symbol(sym: str) -> str:
    if "/" in sym:
        base, rest = sym.split("/", 1)
        quote = rest.split(":")[0]
        return (base + quote).upper()
    return sym.replace(":", "").replace("/", "").upper()


def tf_to_interval(tf: str) -> str:
    tf = tf.lower()
    if tf.endswith("m"): return str(int(tf[:-1]))
    if tf.endswith("h"): return str(int(tf[:-1])*60)
    if tf in ("1d","d"): return "D"
    if tf in ("1w","w"): return "W"
    return "5"


# -------------------- DEEP PATCH dla pandas_ta MACD --------------------

def _deep_patch_pandas_ta_macd():
    """
    Patchuje:
      - pandas_ta.macd
      - PandasTA.macd (df.ta.macd)
      - pandas_ta.momentum.macd (jeśli istnieje)
      - pandas_ta.overlap.macd  (dla starych/alternatywnych struktur)
    tak, aby ZAWSZE istniały kolumny:
      MACD_{f}_{s}_{g}, MACDs_{f}_{s}_{g}, MACDh_{f}_{s}_{g}
    """
    try:
        import pandas_ta as ta
        # próbujmy importować różne miejsca macd
        try:
            from pandas_ta.core import PandasTA
        except Exception:
            PandasTA = None
        try:
            import pandas_ta.momentum as ta_mom
        except Exception:
            ta_mom = None
        try:
            import pandas_ta.overlap as ta_ovl
        except Exception:
            ta_ovl = None
    except Exception:
        return

    def _flatten_cols(cols):
        return [
            "_".join(map(str, c)) if isinstance(c, tuple) else str(c)
            for c in cols
        ]

    def _normalize_macd_df(df, f, s, g):
        if df is None or getattr(df, "empty", True):
            return df
        df = df.copy()
        df.columns = _flatten_cols(df.columns)

        want_macd = f"MACD_{f}_{s}_{g}"
        want_sig  = f"MACDs_{f}_{s}_{g}"
        want_hist = f"MACDh_{f}_{s}_{g}"

        # MACD
        if want_macd not in df.columns:
            cand = [c for c in df.columns if c.lower().startswith("macd_") and c.endswith(f"_{f}_{s}_{g}")]
            if cand:
                df[want_macd] = df[cand[0]]

        # SIGNAL
        if want_sig not in df.columns:
            cand = [c for c in df.columns if (("sig" in c.lower() or "signal" in c.lower()) and c.endswith(f"_{f}_{s}_{g}"))]
            if cand:
                df[want_sig] = df[cand[0]]

        # HIST
        if want_hist not in df.columns:
            cand = [c for c in df.columns if (("hist" in c.lower() or "macdh" in c.lower() or "histogram" in c.lower()) and c.endswith(f"_{f}_{s}_{g}"))]
            if cand:
                df[want_hist] = df[cand[0]]
            else:
                if want_macd in df.columns and want_sig in df.columns:
                    df[want_hist] = df[want_macd] - df[want_sig]

        # Uporządkuj kolejność
        cols_out = [c for c in [want_macd, want_hist, want_sig] if c in df.columns]
        return df[cols_out] if cols_out else df

    # zachowaj oryginały
    _orig_module_fn = getattr(ta, "macd", None)
    _orig_acc_fn = getattr(PandasTA, "macd", None) if PandasTA else None
    _orig_mom_fn = getattr(ta_mom, "macd", None) if ta_mom else None
    _orig_ovl_fn = getattr(ta_ovl, "macd", None) if ta_ovl else None

    def _wrap(fn):
        def _inner(*args, **kwargs):
            f = kwargs.get("fast", kwargs.get("fastperiod", 12)) or 12
            s = kwargs.get("slow", kwargs.get("slowperiod", 26)) or 26
            g = kwargs.get("signal", kwargs.get("signalperiod", 9)) or 9
            out = fn(*args, **kwargs)
            return _normalize_macd_df(out, f, s, g)
        return _inner

    if _orig_module_fn and not getattr(ta.macd, "_patched_live", False):
        ta.macd = _wrap(_orig_module_fn)
        ta.macd._patched_live = True  # type: ignore[attr-defined]

    if _orig_mom_fn and not getattr(ta_mom.macd, "_patched_live", False):
        ta_mom.macd = _wrap(_orig_mom_fn)  # type: ignore[assignment]
        ta_mom.macd._patched_live = True  # type: ignore[attr-defined]

    if _orig_ovl_fn and not getattr(ta_ovl.macd, "_patched_live", False):
        ta_ovl.macd = _wrap(_orig_ovl_fn)  # type: ignore[assignment]
        ta_ovl.macd._patched_live = True  # type: ignore[attr-defined]

    # akcesor df.ta.macd
    if PandasTA and _orig_acc_fn and not getattr(PandasTA.macd, "_patched_live", False):
        def _acc(self, *args, **kwargs):
            out = _orig_acc_fn(self, *args, **kwargs)
            f = kwargs.get("fast", 12) or 12
            s = kwargs.get("slow", 26) or 26
            g = kwargs.get("signal", 9) or 9
            return _normalize_macd_df(out, f, s, g)
        PandasTA.macd = _acc  # type: ignore[assignment]
        PandasTA.macd._patched_live = True  # type: ignore[attr-defined]


# -------------------- konfiguracja --------------------

@dataclass
class LiveConfig:
    # Bybit
    symbol: str = "DOGE/USDT:USDT"
    timeframe: str = "5m"
    api_base: str = "https://api-demo.bybit.com"
    leverage: float = 1.0
    hist_limit: int = 3000

    # Model
    model_path: str = "final_model.joblib"
    scaler_path: str = "final_scaler.joblib"
    best_features_path: str = "best_features.json"

    # Polityka sygnału
    min_conf_long: float = 0.70
    min_conf_short: float = 0.70
    signal_delay_bars: int = 1

    # Ryzyko i stopy
    risk_fraction: float = 0.02
    atr_period: int = 14
    atr_mult_stop: float = 2.0
    atr_mult_tp: Optional[float] = 3.0
    min_stop_pct_of_price: float = 0.0

    # Limity
    max_notional_frac: float = 0.5
    point_value: float = 1.0

    # Zachowanie
    allow_same_bar_reverse: bool = True
    poll_sec: float = 5.0


# -------------------- model sygnału --------------------

class SignalModel:
    def __init__(self, model_path: str, scaler_path: str, best_features_path: str,
                 min_conf_long: float, min_conf_short: float):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(best_features_path, "r") as f:
            self.best_features = json.load(f)

        self.min_conf_long = min_conf_long / 100.0 if min_conf_long > 1 else min_conf_long
        self.min_conf_short = min_conf_short / 100.0 if min_conf_short > 1 else min_conf_short

    def predict_signal(self, feats_row_df: pd.DataFrame) -> int:
        X = feats_row_df[self.best_features].astype(float)
        Xs = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(Xs)[0]
            p_long = float(proba[1]) if len(proba) > 1 else float(proba[0])
            p_short = 1.0 - p_long
            if p_long >= self.min_conf_long:
                return 1
            if p_short >= self.min_conf_short:
                return -1
            return 0
        pred = self.model.predict(Xs)[0]
        if pred in (-1, 0, 1):
            return int(pred)
        return 1 if float(pred) > 0 else -1


# -------------------- stan pozycji --------------------

@dataclass
class PositionState:
    side: Optional[str] = None
    qty: float = 0.0
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    tp_price: Optional[float] = None
    high_close_since_entry: Optional[float] = None
    low_close_since_entry: Optional[float] = None


# -------------------- główna klasa --------------------

class LiveTrader:
    def __init__(self, cfg: LiveConfig, api_key: str, api_secret: str):
        # wymuś zgodność TF
        if cfg.timeframe != getattr(config, "BASE_TIMEFRAME", cfg.timeframe):
            raise SystemExit(
                f"timeframe={cfg.timeframe} musi równać się BASE_TIMEFRAME z config.py "
                f"({getattr(config, 'BASE_TIMEFRAME', 'nieznane')})"
            )

        self.cfg = cfg
        self.symbol_u = ccxt_style_to_bybit_symbol(cfg.symbol)
        self.interval = tf_to_interval(cfg.timeframe)

        # 1) Patch MACD zanim zaimportujemy data_preparer
        _deep_patch_pandas_ta_macd()

        # 2) Dopiero teraz import data_preparer i pobranie funkcji
        try:
            import importlib
            dp = importlib.import_module("data_preparer")
        except Exception as e:
            raise SystemExit(f"Nie udało się zaimportować data_preparer.py: {e}") from e

        try:
            self._prepare_feats = getattr(dp, "prepare_feature_set_for_timeframe")
        except AttributeError:
            raise SystemExit("W data_preparer.py nie znaleziono prepare_feature_set_for_timeframe(df, base_tf=...)")

        self._ensure_hist_limit()

        self.bybit = BybitAdapter(
            api_key=api_key, api_secret=api_secret, base_url=cfg.api_base, category="linear"
        )
        self.bybit.set_leverage(self.symbol_u, cfg.leverage)

        self.model = SignalModel(cfg.model_path, cfg.scaler_path, cfg.best_features_path,
                                 cfg.min_conf_long, cfg.min_conf_short)
        self.pos = PositionState()
        self._last_closed_ts: Optional[pd.Timestamp] = None

        _log_versions()

    # ----- autowarmup historii -----
    def _ensure_hist_limit(self):
        try:
            with open(self.cfg.best_features_path,"r") as f:
                feats = json.load(f)
        except Exception:
            feats = []

        def _tf_m(tf: str) -> int:
            tf = tf.lower()
            if tf.endswith("m"): return int(tf[:-1])
            if tf.endswith("h"): return int(tf[:-1]) * 60
            if tf in ("1d","d"): return 24*60
            return 5

        base_m = _tf_m(self.cfg.timeframe)
        tfs = []
        for s in feats:
            parts = s.split("_")
            if parts:
                tf = parts[-1]
                if isinstance(tf, str) and (tf.endswith("m") or tf.endswith("h") or tf in ("d","1d")):
                    tfs.append(tf)
        if not tfs:
            tfs = [self.cfg.timeframe]
        max_tf_m = max(_tf_m(tf) for tf in tfs)

        macd_slow = getattr(config, "MACD_SLOW", 26)
        ichimoku_b = max(52, getattr(config, "ICHIMOKU_SENKOU_B", 52))
        hurst_win  = getattr(config, "HURST_WINDOW", 256)
        rsi_len    = getattr(config, "RSI_LENGTH", 14)
        fibo_win   = getattr(config, "FIBO_WINDOW", 120)
        max_window = max(macd_slow, ichimoku_b, hurst_win, rsi_len, fibo_win, 60)

        need = int(math.ceil((max_tf_m / base_m) * (max_window + 50) * 1.7))
        if self.cfg.hist_limit < need:
            log.info(f"Auto-raising hist_limit {self.cfg.hist_limit} -> {need} (warmup)")
            self.cfg.hist_limit = need

    # ----- CECHY 1:1 z treningiem (data_preparer) -----
    def _build_features_live(self, kline_rows: List[List[str]]) -> pd.DataFrame:
        arr = np.array(kline_rows, dtype=object)
        ts_ms = arr[:, 0].astype(np.int64)
        o = arr[:, 1].astype(float)
        h = arr[:, 2].astype(float)
        l = arr[:, 3].astype(float)
        c = arr[:, 4].astype(float)
        v = arr[:, 5].astype(float)
        to = arr[:, 6].astype(float)

        df_raw = pd.DataFrame({
            "open": o, "high": h, "low": l, "close": c, "volume": v, "turnover": to
        }, index=pd.to_datetime(ts_ms, unit="ms", utc=True))

        # budowa cech 1:1
        full = self._prepare_feats(df_raw, base_tf=getattr(config, "BASE_TIMEFRAME", "5m"))

        # porządkowanie
        full = full.sort_index()
        full = full[~full.index.duplicated(keep="last")]
        full.replace([np.inf, -np.inf], np.nan, inplace=True)
        full.dropna(inplace=True)

        # walidacja best_features
        _assert_best_features(full, best_features_path=self.cfg.best_features_path)
        return full

    # ----- sizing wg ryzyka (ATR) -----
    def _qty_by_risk(self, equity: float, price: float, atr: float, side: str) -> Tuple[float,float,Optional[float],bool]:
        k = self.cfg.atr_mult_stop
        stop_dist = max(1e-9, k * float(atr))
        if self.cfg.min_stop_pct_of_price > 0:
            stop_dist = max(stop_dist, price * self.cfg.min_stop_pct_of_price)
        risk_cash = equity * self.cfg.risk_fraction
        qty = (risk_cash * self.cfg.leverage) / (stop_dist * self.cfg.point_value)

        notional = qty * price * self.cfg.point_value
        cap = equity * self.cfg.leverage * self.cfg.max_notional_frac
        capped = False
        if notional > cap and price > 0:
            qty = cap / (price * self.cfg.point_value)
            capped = True

        if side == "long":
            stop_p = price - stop_dist
            tp_p = None if self.cfg.atr_mult_tp is None else price + self.cfg.atr_mult_tp * float(atr)
        else:
            stop_p = price + stop_dist
            tp_p = None if self.cfg.atr_mult_tp is None else price - self.cfg.atr_mult_tp * float(atr)
        return qty, stop_p, tp_p, capped

    def _update_tsl(self, last_close: float, atr: float):
        if self.pos.side == "long":
            self.pos.high_close_since_entry = max(self.pos.high_close_since_entry or last_close, last_close)
            new_stop = self.pos.high_close_since_entry - self.cfg.atr_mult_stop * float(atr)
            self.pos.stop_price = max(self.pos.stop_price or new_stop, new_stop)
        elif self.pos.side == "short":
            self.pos.low_close_since_entry = min(self.pos.low_close_since_entry or last_close, last_close)
            new_stop = self.pos.low_close_since_entry + self.cfg.atr_mult_stop * float(atr)
            self.pos.stop_price = min(self.pos.stop_price or new_stop, new_stop)

    def _unrealized(self, price: float) -> float:
        if not self.pos.side or self.pos.entry_price is None:
            return 0.0
        d = (price - self.pos.entry_price) if self.pos.side == "long" else (self.pos.entry_price - price)
        return d * self.pos.qty * self.cfg.point_value

    # ----- pętla live -----
    def run(self):
        while True:
            try:
                klines = self.bybit.fetch_ohlcv(self.symbol_u, self.interval, limit=self.cfg.hist_limit)
                if not klines or len(klines) < 200:
                    time.sleep(self.cfg.poll_sec); continue

                feats = self._build_features_live(klines)
                if feats.empty or len(feats) < 10:
                    time.sleep(self.cfg.poll_sec); continue

                prev_idx, cur_idx = feats.index[-2], feats.index[-1]
                if hasattr(self, "_last_closed_ts") and self._last_closed_ts == prev_idx:
                    time.sleep(self.cfg.poll_sec); continue
                self._last_closed_ts = prev_idx

                prev_row_df = feats.loc[[prev_idx]]

                arr = np.array(klines, dtype=object)
                open_cur  = float(arr[-1][1])
                close_prev = float(arr[-2][4])

                # ATR: preferuj z cech; jeśli nie ma, policz fallbackiem
                atr_cols = [c for c in prev_row_df.columns if c.upper().startswith("ATR")]
                if atr_cols:
                    atr_prev = float(prev_row_df[atr_cols].iloc[0].astype(float).max())
                else:
                    o = arr[:,1].astype(float); h = arr[:,2].astype(float)
                    l = arr[:,3].astype(float); c = arr[:,4].astype(float)
                    atr_prev = compute_atr_from_ohlc(o, h, l, c, self.cfg.atr_period)
                    if math.isnan(atr_prev):
                        time.sleep(self.cfg.poll_sec); continue

                # update TSL po close t-1
                if self.pos.side:
                    self._update_tsl(close_prev, atr_prev)

                # sygnał z t-1
                signal = self.model.predict_signal(prev_row_df[self.model.best_features])

                if signal == 0:
                    time.sleep(self.cfg.poll_sec); continue

                # equity (USDT)
                bal = self.bybit.fetch_balance_usdt()
                equity = float(bal.get("total", 0) or bal.get("walletBalance", 0) or 0.0)
                if equity <= 0:
                    time.sleep(self.cfg.poll_sec); continue

                # ENTRY / REVERSE
                if self.pos.side is None:
                    side = "long" if signal == 1 else "short"
                    qty, stop_p, tp_p, capped = self._qty_by_risk(equity, open_cur, atr_prev, side)
                    if qty > 0:
                        if side == "long":
                            self.bybit.market_entry(self.symbol_u, "Buy", qty)
                        else:
                            self.bybit.market_entry(self.symbol_u, "Sell", qty)
                        self.pos = PositionState(
                            side=side, qty=qty, entry_price=open_cur,
                            stop_price=stop_p, tp_price=tp_p,
                            high_close_since_entry=open_cur if side=="long" else None,
                            low_close_since_entry=open_cur if side=="short" else None,
                        )
                        log.info(f"ENTRY {side.upper()} qty={qty:.6f} @ {open_cur:.6g} stop={stop_p:.6g} tp={tp_p if tp_p is not None else 'None'} capped={capped}")

                else:
                    if self.pos.side == "long" and signal == -1:
                        upnl = self._unrealized(open_cur)
                        protected = (upnl > 0) and (self.pos.stop_price is not None and self.pos.stop_price > (self.pos.entry_price or -np.inf))
                        if not protected:
                            self.bybit.market_close(self.symbol_u, "Sell", self.pos.qty)
                            log.info(f"EXIT LONG (reverse) @ {open_cur:.6g}")
                            side = "short"
                            qty, stop_p, tp_p, capped = self._qty_by_risk(equity, open_cur, atr_prev, side)
                            if self.cfg.allow_same_bar_reverse and qty > 0:
                                self.bybit.market_entry(self.symbol_u, "Sell", qty)
                                self.pos = PositionState(side=side, qty=qty, entry_price=open_cur,
                                                         stop_price=stop_p, tp_price=tp_p,
                                                         low_close_since_entry=open_cur)
                                log.info(f"ENTRY SHORT qty={qty:.6f} @ {open_cur:.6g}")

                    elif self.pos.side == "short" and signal == 1:
                        upnl = self._unrealized(open_cur)
                        protected = (upnl > 0) and (self.pos.stop_price is not None and self.pos.stop_price < (self.pos.entry_price or +np.inf))
                        if not protected:
                            self.bybit.market_close(self.symbol_u, "Buy", self.pos.qty)
                            log.info(f"EXIT SHORT (reverse) @ {open_cur:.6g}")
                            side = "long"
                            qty, stop_p, tp_p, capped = self._qty_by_risk(equity, open_cur, atr_prev, side)
                            if self.cfg.allow_same_bar_reverse and qty > 0:
                                self.bybit.market_entry(self.symbol_u, "Buy", qty)
                                self.pos = PositionState(side=side, qty=qty, entry_price=open_cur,
                                                         stop_price=stop_p, tp_price=tp_p,
                                                         high_close_since_entry=open_cur)
                                log.info(f"ENTRY LONG qty={qty:.6f} @ {open_cur:.6g}")

                # Wyjścia po close t-1 (realizacja na open t)
                if self.pos.side:
                    if self.pos.tp_price is not None:
                        if (self.pos.side=="long" and close_prev >= self.pos.tp_price) or \
                           (self.pos.side=="short" and close_prev <= self.pos.tp_price):
                            self.bybit.market_close(self.symbol_u, "Sell" if self.pos.side=="long" else "Buy", self.pos.qty)
                            log.info(f"EXIT TP {self.pos.side.upper()} @ open_next={open_cur:.6g}")
                            self.pos = PositionState(); time.sleep(self.cfg.poll_sec); continue
                    if self.pos.stop_price is not None:
                        if (self.pos.side=="long" and close_prev <= self.pos.stop_price) or \
                           (self.pos.side=="short" and close_prev >= self.pos.stop_price):
                            self.bybit.market_close(self.symbol_u, "Sell" if self.pos.side=="long" else "Buy", self.pos.qty)
                            log.info(f"EXIT TSL {self.pos.side.upper()} @ open_next={open_cur:.6g}")
                            self.pos = PositionState(); time.sleep(self.cfg.poll_sec); continue

                time.sleep(self.cfg.poll_sec)

            except KeyboardInterrupt:
                log.info("Stop requested."); break
            except Exception as e:
                if any(k in str(e).lower() for k in ["macdh", "macd", "ichimoku", "hurst", "lag", "rolling", "window"]):
                    self.cfg.hist_limit = min(self.cfg.hist_limit * 2, 20000)
                    log.info(f"Bumping hist_limit to {self.cfg.hist_limit} po błędzie cech: {e}")
                log.exception(f"Live loop error: {e}")
                time.sleep(self.cfg.poll_sec)


# -------------------- CLI --------------------

def _parse_tp(x):
    s = str(x).strip().lower()
    if s in {"none","null","","nan"}: return None
    return float(x)

def parse_args():
    p = argparse.ArgumentParser(description="Bybit (pybit) LIVE/paper trading – cechy 1:1 jak w treningu (data_preparer + deep patch MACD)")
    p.add_argument("--symbol", default="DOGE/USDT:USDT")
    p.add_argument("--timeframe", default=getattr(config, "BASE_TIMEFRAME", "5m"))
    p.add_argument("--api_base", default=os.getenv("BYBIT_API_BASE") or "https://api-demo.bybit.com",
                   help="https://api-demo.bybit.com (DEMO) / https://api.bybit.com (MAIN) / https://api-testnet.bybit.com (TESTNET)")
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--hist_limit", type=int, default=3000)

    p.add_argument("--model_path", default="final_model.joblib")
    p.add_argument("--scaler_path", default="final_scaler.joblib")
    p.add_argument("--best_features_path", default="best_features.json")

    p.add_argument("--min_conf_long", type=float, default=0.70)
    p.add_argument("--min_conf_short", type=float, default=0.70)
    p.add_argument("--risk_fraction", type=float, default=0.02)
    p.add_argument("--atr_period", type=int, default=14)
    p.add_argument("--atr_mult_stop", type=float, default=2.0)
    p.add_argument("--atr_mult_tp", type=_parse_tp, default=3.0)
    p.add_argument("--min_stop_pct_of_price", type=float, default=0.0)
    p.add_argument("--signal_delay_bars", type=int, default=1)
    p.add_argument("--max_notional_frac", type=float, default=0.5)
    p.add_argument("--point_value", type=float, default=1.0)
    p.add_argument("--poll_sec", type=float, default=5.0)
    return p.parse_args()

def main():
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    if not api_key or not api_secret:
        raise SystemExit("Brak BYBIT_API_KEY / BYBIT_API_SECRET w .env")

    cfg = LiveConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        api_base=args.api_base,
        leverage=args.leverage,
        hist_limit=args.hist_limit,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        best_features_path=args.best_features_path,
        min_conf_long=args.min_conf_long,
        min_conf_short=args.min_conf_short,
        risk_fraction=args.risk_fraction,
        atr_period=args.atr_period,
        atr_mult_stop=args.atr_mult_stop,
        atr_mult_tp=args.atr_mult_tp,
        min_stop_pct_of_price=args.min_stop_pct_of_price,
        signal_delay_bars=args.signal_delay_bars,
        max_notional_frac=args.max_notional_frac,
        point_value=args.point_value,
        poll_sec=args.poll_sec,
    )

    trader = LiveTrader(cfg, api_key, api_secret)
    trader.run()

if __name__ == "__main__":
    main()
