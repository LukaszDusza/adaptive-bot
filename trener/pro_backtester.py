# pro_backtester.py
# ATR-TSL ML backtester for single-position (long OR short) with:
# - No lookahead (signal computed at t-1, executed at next bar OPEN)
# - Confidence filters (min_conf_long/short) -> HOLD when below
# - Optional "flat on low confidence" policy (off/always/loss_only/not_protected)
# - Sizing by % equity risk on ATR stop, with:
#     * point_value (PnL multiplier per price unit) + AUTO via real_median_price
#     * notional cap (<= equity * leverage * max_notional_frac)
#     * minimum stop distance as % of price
# - Gap-aware exits at OPEN
# - Fill modes: intrabar (H/L) or close
# - Costs: commission %, slippage %, fixed fee per fill
# - Analytics NETTO (po kosztach), wykresy, HTML report
# - Per-trade charts (pierwsze N): ENTRY/EXIT, TP, TSL
# - Eksport Pine v5 overlay do TradingView (ENTRY/EXIT + TSL/TP)

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
import json
import math
import logging
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import AutoDateLocator, DateFormatter
import base64
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------- Config -----------------------------

@dataclass
class BacktestConfig:
    # Artifacts
    data_csv_path: str
    model_path: str
    scaler_path: str
    best_features_path: str

    # Date range (inclusive)
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Runtime
    signal_every_n_bars: int = 1
    signal_delay_bars: int = 1
    bars_per_year: int = 252*24*12

    # Confidence filters (0-1 or 0-100; normalized internally)
    min_conf_long: float = 0.93
    min_conf_short: float = 0.93

    # Flat-on-low-confidence policy
    # off | always | loss_only | not_protected
    flat_on_low_conf: str = "off"

    # Risk & stops
    risk_fraction: float = 0.0175
    atr_period: int = 14
    atr_mult_stop: float = 1.085
    atr_mult_tp: Optional[float] = 1.2
    min_stop_pct_of_price: float = 0.0   # e.g. 0.001 = 0.1% of price

    # Costs & leverage
    initial_equity: float = 10_000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    slippage_jitter_bps: float = 0.0
    leverage: float = 1.0
    fixed_fee_per_fill: float = 0.0          # stała opłata za każdy fill ($)
    # notional cap fraction of (equity * leverage)
    max_notional_frac: float = 1.0
    # PnL multiplier per price unit (1 if prices are in account currency; >1 if normalized)
    point_value: float = 1.0
    # Optional: auto point_value = real_median_price / csv_median_close (over test window)
    real_median_price: Optional[float] = None

    # Columns
    col_open: str = "open"
    col_high: str = "high"
    col_low: str  = "low"
    col_close: str = "close"
    col_volume: str = "volume"

    # Behavior
    allow_same_bar_reverse: bool = True
    tz_aware: bool = False
    fills: str = "intrabar"              # "intrabar" or "close"
    make_charts: bool = True
    log_equity: bool = False
    per_trade_charts: int = 15           # number of trade charts to render (0 = none)

# ----------------------------- Trade struct -----------------------------

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    side: str
    entry_price: float
    exit_price: Optional[float]
    qty: float
    atr_at_entry: float
    stop_at_entry: float
    tp_at_entry: Optional[float]
    # extras
    risk_cash: float = 0.0
    stop_distance: float = 0.0
    pnl: Optional[float] = None          # alias of net_pnl for back-compat
    pnl_pct: Optional[float] = None
    r_multiple: Optional[float] = None
    bars_held: int = 0
    notes: Optional[str] = None
    # new
    fees: float = 0.0
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None

# ----------------------------- Backtester -----------------------------

class ProBacktester:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        # normalize min-confidence thresholds to 0..1
        if self.cfg.min_conf_long > 1.0:  self.cfg.min_conf_long  /= 100.0
        if self.cfg.min_conf_short > 1.0: self.cfg.min_conf_short /= 100.0
        # normalize flat policy
        self.cfg.flat_on_low_conf = (self.cfg.flat_on_low_conf or "off").lower()
        if self.cfg.flat_on_low_conf not in ("off","always","loss_only","not_protected"):
            self.cfg.flat_on_low_conf = "off"

        self.model = None
        self.scaler = None
        self.best_features: List[str] = []
        self.df: Optional[pd.DataFrame] = None

        # state
        self.equity: float = cfg.initial_equity
        self.position_side: Optional[str] = None
        self.position_qty: float = 0.0
        self.entry_price: Optional[float] = None
        self.entry_index: Optional[int] = None
        self.stop_price: Optional[float] = None
        self.tp_price: Optional[float] = None
        self.highest_close_since_entry: Optional[float] = None
        self.lowest_close_since_entry: Optional[float] = None
        self.current_atr: Optional[float] = None

        self.pending_signal_queue: Dict[int, int] = {}
        self.equity_curve: List[float] = []
        self.position_open_bars: int = 0
        self.trades: List[Trade] = []

        # Counters
        self.capped_entries: int = 0
        self.gap_exits: int = 0

        # Cost tracking
        self.cost_commission: float = 0.0
        self.cost_slippage: float = 0.0
        self.cost_fixed: float = 0.0

    # ---------- utils ----------
    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int, col_high: str, col_low: str, col_close: str) -> pd.Series:
        high = df[col_high].astype(float)
        low = df[col_low].astype(float)
        close = df[col_close].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=period).mean()

    def _load_data_and_model(self):
        cfg = self.cfg
        logging.info("--- Inicjalizacja Strategii ---")
        df = pd.read_csv(cfg.data_csv_path, index_col=0)
        try:
            df.index = pd.to_datetime(df.index, utc=cfg.tz_aware)
        except Exception:
            pass

        if cfg.start_date:
            df = df.loc[df.index >= pd.to_datetime(cfg.start_date)]
        if cfg.end_date:
            df = df.loc[df.index <= pd.to_datetime(cfg.end_date)]

        for c in [cfg.col_open, cfg.col_high, cfg.col_low, cfg.col_close]:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in data.")

        if "ATR" not in df.columns:
            df["ATR"] = self._compute_atr(df, cfg.atr_period, cfg.col_high, cfg.col_low, cfg.col_close)

        # Auto point_value from real median price if requested
        if cfg.real_median_price is not None and cfg.real_median_price > 0:
            med_csv = float(df[cfg.col_close].astype(float).median())
            if med_csv > 0:
                cfg.point_value = float(cfg.real_median_price) / med_csv
                logging.info(f"[auto point_value] csv_median_close={med_csv:.6g}, real_median_price={cfg.real_median_price:.6g} -> point_value={cfg.point_value:.6g}")

        self.model = joblib.load(cfg.model_path)
        self.scaler = joblib.load(cfg.scaler_path)
        with open(cfg.best_features_path, "r") as f:
            self.best_features = json.load(f)
        if not self.best_features:
            raise ValueError("best_features.json must be a non-empty list of feature names.")
        missing = [c for c in self.best_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns in data: {missing}")

        self.df = df

    def _commission_and_slippage(self, price: float, qty: float) -> float:
        notional = abs(price * qty * self.cfg.point_value)
        commission = notional * self.cfg.commission_rate
        jitter = (random.uniform(-self.cfg.slippage_jitter_bps, self.cfg.slippage_jitter_bps) / 10_000.0) if self.cfg.slippage_jitter_bps > 0 else 0.0
        slippage = notional * max(0.0, self.cfg.slippage_rate + jitter)
        fixed = float(self.cfg.fixed_fee_per_fill)
        self.cost_commission += commission
        self.cost_slippage += slippage
        self.cost_fixed += fixed
        return commission + slippage + fixed

    def _calc_qty_by_risk(self, price: float, atr: float, side: str) -> Tuple[float, float, float, float, float, bool]:
        k = self.cfg.atr_mult_stop
        stop_dist = max(1e-9, k * atr)
        if self.cfg.min_stop_pct_of_price > 0:
            stop_dist = max(stop_dist, price * self.cfg.min_stop_pct_of_price)
        risk_cash = self.equity * self.cfg.risk_fraction
        # Qty so that stop_loss ≈ risk_cash: stop_dist * qty * point_value = risk_cash
        qty = (risk_cash * self.cfg.leverage) / (stop_dist * self.cfg.point_value)
        if side == "long":
            stop_price = price - stop_dist
            tp_price = None if self.cfg.atr_mult_tp is None else price + self.cfg.atr_mult_tp * atr
        else:
            stop_price = price + stop_dist
            tp_price = None if self.cfg.atr_mult_tp is None else price - self.cfg.atr_mult_tp * atr

        # Notional cap
        notional_unit = price * self.cfg.point_value  # per one qty
        cap = self.equity * self.cfg.leverage * self.cfg.max_notional_frac
        capped = False
        if qty * notional_unit > cap and notional_unit > 0:
            qty = cap / notional_unit
            capped = True
        return qty, stop_price, tp_price, risk_cash, stop_dist, capped

    def _update_trailing_stop(self, last_close: float, atr: float):
        if self.position_side == "long":
            self.highest_close_since_entry = max(self.highest_close_since_entry or last_close, last_close)
            new_stop = self.highest_close_since_entry - self.cfg.atr_mult_stop * atr
            self.stop_price = max(self.stop_price or new_stop, new_stop)
        elif self.position_side == "short":
            self.lowest_close_since_entry = min(self.lowest_close_since_entry or last_close, last_close)
            new_stop = self.lowest_close_since_entry + self.cfg.atr_mult_stop * atr
            self.stop_price = min(self.stop_price or new_stop, new_stop)

    def _pnl_unrealized(self, price: float) -> float:
        if not self.position_side or self.entry_price is None:
            return 0.0
        delta = (price - self.entry_price) if self.position_side == "long" else (self.entry_price - price)
        return delta * self.position_qty * self.cfg.point_value

    def _enter_position(self, i: int, side: str, exec_price: float, atr: float):
        qty, stop_at_entry, tp_at_entry, risk_cash, stop_dist, capped = self._calc_qty_by_risk(exec_price, atr, side)
        if qty <= 0:
            return
        costs = self._commission_and_slippage(exec_price, qty)
        self.equity -= costs
        if capped:
            self.capped_entries += 1
        self.position_side = side
        self.position_qty = qty
        self.entry_price = exec_price
        self.entry_index = i
        self.stop_price = stop_at_entry
        self.tp_price = tp_at_entry
        self.highest_close_since_entry = exec_price if side == "long" else None
        self.lowest_close_since_entry = exec_price if side == "short" else None
        self.trades.append(Trade(
            entry_time=self.df.index[i],
            exit_time=None,
            side=side,
            entry_price=exec_price,
            exit_price=None,
            qty=qty,
            atr_at_entry=atr,
            stop_at_entry=stop_at_entry,
            tp_at_entry=tp_at_entry,
            risk_cash=risk_cash,
            stop_distance=stop_dist,
            notes=("enter@open|capped" if capped else "enter@open"),
            fees=costs
        ))

    def _exit_position(self, i: int, exec_price: float, reason: str):
        if self.position_side is None:
            return
        qty = self.position_qty
        costs = self._commission_and_slippage(exec_price, qty)
        self.equity -= costs
        delta = (exec_price - self.entry_price) if self.position_side == "long" else (self.entry_price - exec_price)
        gross = delta * qty * self.cfg.point_value
        self.equity += gross
        trade = self.trades[-1]
        trade.exit_time = self.df.index[i]
        trade.exit_price = exec_price
        trade.gross_pnl = gross
        trade.fees += costs
        trade.net_pnl = gross - trade.fees
        trade.pnl = trade.net_pnl  # alias
        trade.pnl_pct = trade.net_pnl / (self.cfg.initial_equity if self.cfg.initial_equity != 0 else 1.0)
        trade.bars_held = i - (self.entry_index if self.entry_index is not None else i)
        trade.r_multiple = (trade.net_pnl / (trade.risk_cash + 1e-9)) if trade.risk_cash else None
        trade.notes = (trade.notes or "") + f"|exit:{reason}"
        self.position_side = None
        self.position_qty = 0.0
        self.entry_price = None
        self.entry_index = None
        self.stop_price = None
        self.tp_price = None
        self.highest_close_since_entry = None
        self.lowest_close_since_entry = None

    # ---------- ML ----------
    def _predict_signal(self, feats_row: pd.DataFrame) -> int:
        """
        Returns:
          1  -> long  (if P(long) >= min_conf_long)
         -1  -> short (if P(short)>= min_conf_short)
          0  -> hold  (otherwise)
        """
        Xdf = feats_row[self.best_features].astype(float)
        Xs = self.scaler.transform(Xdf)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(Xs)[0]
            if len(proba) >= 2:
                p_long = float(proba[1])
                p_short = float(proba[0]) if len(proba) >= 2 else 1.0 - p_long
                if p_long >= self.cfg.min_conf_long:
                    return 1
                elif p_short >= self.cfg.min_conf_short:
                    return -1
                else:
                    return 0
            else:
                pred = self.model.predict(Xs)[0]
                return int(pred) if pred in (-1, 0, 1) else (1 if pred > 0 else -1)
        else:
            pred = self.model.predict(Xs)[0]
            if pred in (-1, 0, 1):
                return int(pred)
            return 1 if float(pred) > 0 else -1

    # ---------- Analytics ----------
    def _calc_analytics(self, eq: pd.Series) -> Dict[str, Any]:
        rets = eq.pct_change().fillna(0.0)
        if isinstance(eq.index, pd.DatetimeIndex) and len(eq) > 1:
            elapsed_years = max(1e-9, (eq.index[-1] - eq.index[0]).total_seconds() / (365.25*24*3600))
        else:
            elapsed_years = max(1e-9, len(eq) / self.cfg.bars_per_year)

        sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(self.cfg.bars_per_year)
        downside = rets.copy(); downside[downside > 0] = 0.0
        sortino = (rets.mean() / (downside.std() + 1e-12)) * np.sqrt(self.cfg.bars_per_year)

        peak = eq.cummax()
        drawdown = (eq - peak) / peak
        max_dd = drawdown.min()

        total_return = (eq.iloc[-1] / eq.iloc[0]) - 1 if len(eq) > 1 else 0.0
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / elapsed_years) - 1 if len(eq) > 1 else 0.0
        calmar = (cagr / abs(max_dd)) if max_dd < 0 else np.inf

        wins = [t for t in self.trades if (t.net_pnl is not None and t.net_pnl > 0)]
        losses = [t for t in self.trades if (t.net_pnl is not None and t.net_pnl <= 0)]
        win_rate = len(wins) / len(self.trades) if self.trades else 0.0
        avg_win = float(np.mean([t.net_pnl for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t.net_pnl for t in losses])) if losses else 0.0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        profit_factor = (float(np.sum([t.net_pnl for t in wins])) / (abs(float(np.sum([t.net_pnl for t in losses]))) + 1e-12)) if losses else float('inf')
        avg_bars = float(np.mean([t.bars_held for t in self.trades])) if self.trades else 0.0

        r_list = [t.r_multiple for t in self.trades if t.r_multiple is not None]
        avg_r = float(np.mean(r_list)) if r_list else 0.0
        med_r = float(np.median(r_list)) if r_list else 0.0

        exposure = self.position_open_bars / len(eq) if len(eq) else 0.0

        return {
            "initial_equity": self.cfg.initial_equity,
            "final_equity": float(eq.iloc[-1]) if len(eq) else self.cfg.initial_equity,
            "total_return": float(total_return),
            "CAGR": float(cagr),
            "max_drawdown": float(max_dd),
            "calmar": float(calmar),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "num_trades": len(self.trades),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "expectancy": float(expectancy),
            "profit_factor": float(profit_factor),
            "avg_bars_held": float(avg_bars),
            "avg_R": float(avg_r),
            "median_R": float(med_r),
            "exposure": float(exposure),
            "capped_entries": int(self.capped_entries),
            "gap_exits": int(self.gap_exits),
            "point_value": float(self.cfg.point_value),
            "max_notional_frac": float(self.cfg.max_notional_frac),
            "total_commission_cost": float(self.cost_commission),
            "total_slippage_cost": float(self.cost_slippage),
            "total_fixed_fees": float(self.cost_fixed),
            "total_costs": float(self.cost_commission + self.cost_slippage + self.cost_fixed),
            "avg_fee_per_trade": float((self.cost_commission + self.cost_slippage + self.cost_fixed)/max(1,len(self.trades))),
        }

    # ---------- Charts + report ----------
    def _fmt_thousands(self, x, pos): return f"{x:,.0f}"

    def _save_charts(self, basepath: Path, eq: pd.Series) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        fmt = FuncFormatter(self._fmt_thousands)

        def save_fig(name: str):
            fpath = f"{basepath}_{name}.png"
            plt.tight_layout()
            plt.savefig(fpath)
            plt.close()
            paths[name] = fpath

        # Equity
        plt.figure()
        eq.plot()
        plt.title("Equity Curve")
        plt.xlabel("Time"); plt.ylabel("Equity")
        ax = plt.gca()
        ax.yaxis.set_major_formatter(fmt)
        if self.cfg.log_equity:
            ax.set_yscale("log")
        save_fig("equity_curve")

        # Underwater (%)
        peak = eq.cummax()
        uw = -(eq - peak) / (peak.replace(0, np.nan))
        plt.figure()
        (uw*100).plot()
        plt.title("Underwater (Drawdown %)")
        plt.xlabel("Time"); plt.ylabel("Drawdown (%)")
        save_fig("underwater")

        # Rolling Sharpe
        rets = eq.pct_change().fillna(0.0)
        window = max(10, int(self.cfg.bars_per_year/12))
        roll_sharpe = (rets.rolling(window).mean() / (rets.rolling(window).std() + 1e-12)) * np.sqrt(self.cfg.bars_per_year)
        plt.figure()
        roll_sharpe.plot()
        plt.title(f"Rolling Sharpe (window={window})")
        plt.xlabel("Time"); plt.ylabel("Sharpe")
        save_fig("rolling_sharpe")

        # PnL histogram
        pnls = pd.Series([t.net_pnl for t in self.trades if t.net_pnl is not None])
        if len(pnls):
            plt.figure()
            pnls.plot(kind="hist", bins=50)
            plt.title("Trade Net PnL Distribution")
            plt.xlabel("Net PnL"); plt.ylabel("Count")
            ax = plt.gca(); ax.xaxis.set_major_formatter(fmt)
            save_fig("pnl_hist")

        # R histogram
        r_list = [t.r_multiple for t in self.trades if t.r_multiple is not None]
        if len(r_list):
            plt.figure()
            pd.Series(r_list).plot(kind="hist", bins=50)
            plt.title("R-Multiple Distribution")
            plt.xlabel("R"); plt.ylabel("Count")
            save_fig("r_hist")

        # Monthly heatmap
        if isinstance(eq.index, pd.DatetimeIndex):
            rets = eq.pct_change().fillna(0.0)
            monthly = rets.resample("M").apply(lambda x: (1+x).prod()-1)
            if len(monthly):
                mat = monthly.to_frame("ret")
                mat["year"] = mat.index.year
                mat["month"] = mat.index.month
                pivot = mat.pivot(index="year", columns="month", values="ret").fillna(0.0)
                plt.figure()
                plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
                plt.title("Monthly Returns")
                plt.xlabel("Month"); plt.ylabel("Year")
                plt.colorbar()
                save_fig("monthly_heatmap")

        return paths

    def _save_trade_charts(self, basepath: Path, max_trades: int = 15) -> List[str]:
        if max_trades <= 0 or not self.trades:
            return []
        paths: List[str] = []
        df = self.df
        cfg = self.cfg

        for idx, t in enumerate(self.trades[:max_trades], start=1):
            try:
                # find indices
                try:
                    i_entry = df.index.get_loc(t.entry_time)
                except KeyError:
                    i_entry = int(np.searchsorted(df.index.values, np.array([t.entry_time], dtype='datetime64[ns]'))[0])
                i_exit = i_entry if t.exit_time is None else (
                    df.index.get_loc(t.exit_time) if t.exit_time in df.index else int(np.searchsorted(df.index.values, np.array([t.exit_time], dtype='datetime64[ns]'))[0])
                )
                i0 = max(0, i_entry - 20)
                i1 = min(len(df)-1, i_exit + 20)
                window = df.iloc[i0:i1+1].copy()

                # Compute TSL path in the window for this trade
                closes = window[cfg.col_close].astype(float).values
                atrs = window["ATR"].astype(float).values
                times = window.index

                tsl = np.full_like(closes, np.nan, dtype=float)
                # Track from the entry forward
                start_rel = i_entry - i0
                if start_rel < 0 or start_rel >= len(window):  # safety
                    continue
                if t.side == "long":
                    high_run = closes[start_rel]
                    for k in range(start_rel, len(closes)):
                        high_run = max(high_run, closes[k])
                        tsl[k] = high_run - cfg.atr_mult_stop * atrs[k]
                else:
                    low_run = closes[start_rel]
                    for k in range(start_rel, len(closes)):
                        low_run = min(low_run, closes[k])
                        tsl[k] = low_run + cfg.atr_mult_stop * atrs[k]

                # Plot price (close), entry/exit, TP, TSL
                fig = plt.figure()
                ax = plt.gca()
                ax.plot(times, closes, label="Close")
                ax.plot(times, tsl, label="TSL")

                # Entry & Exit markers
                entry_ts = t.entry_time; exit_ts = t.exit_time if t.exit_time is not None else times[-1]
                ax.axvline(entry_ts, linestyle="--")
                ax.annotate(f"ENTRY {t.side.upper()}\n{t.entry_price:.6f}", xy=(entry_ts, t.entry_price),
                            xytext=(10,10), textcoords="offset points")

                if t.exit_price is not None:
                    ax.axvline(exit_ts, linestyle="--")
                    ax.annotate(f"EXIT ({t.notes.split('|')[-1] if t.notes else ''})\n{t.exit_price:.6f}", xy=(exit_ts, t.exit_price),
                                xytext=(10,-20), textcoords="offset points")

                # TP line
                if t.tp_at_entry is not None and not math.isnan(t.tp_at_entry):
                    ax.axhline(t.tp_at_entry, linestyle=":", color="C0")
                    ax.text(times[max(start_rel,0)], t.tp_at_entry, "TP", va="bottom")

                # Axes formatting
                ax.set_title(f"Trade {idx:02d} | {str(entry_ts)} \u2192 {str(exit_ts)}")
                ax.set_xlabel("Time"); ax.set_ylabel("Price")
                locator = AutoDateLocator()
                formatter = DateFormatter("%Y-%m-%d\n%H:%M")
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                ax.legend(loc="best")
                fig.autofmt_xdate()

                fpath = f"{basepath}_trade_{idx:04d}.png"
                plt.tight_layout()
                plt.savefig(fpath)
                plt.close()
                paths.append(fpath)
            except Exception as e:
                logging.warning(f"Trade chart {idx} failed: {e}")
        return paths

    def _render_html_report(self, html_path: Path, stats: Dict[str, Any], chart_paths: Dict[str, str]):
        def img_tag(p: str) -> str:
            try:
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;"/>'
            except Exception:
                return f"<p>(missing image: {p})</p>"

        def fmt_pct(x): return f"{x*100:.2f}%"
        def fmt_num(x): return f"{x:,.2f}"

        rows = []
        pretty = {
            "Initial equity": fmt_num(stats["initial_equity"]),
            "Final equity": fmt_num(stats["final_equity"]),
            "Total return": fmt_pct(stats["total_return"]),
            "CAGR": fmt_pct(stats["CAGR"]),
            "Max drawdown": fmt_pct(stats["max_drawdown"]),
            "Calmar": f'{stats["calmar"]:.2f}' if np.isfinite(stats["calmar"]) else "∞",
            "Sharpe": f'{stats["sharpe"]:.2f}',
            "Sortino": f'{stats["sortino"]:.2f}',
            "# Trades": f'{int(stats["num_trades"])}',
            "Win rate": fmt_pct(stats["win_rate"]),
            "Profit factor": f'{stats["profit_factor"]:.2f}' if np.isfinite(stats["profit_factor"]) else "∞",
            "Expectancy": fmt_num(stats["expectancy"]),
            "Avg bars held": f'{stats["avg_bars_held"]:.2f}',
            "Avg R": f'{stats["avg_R"]:.3f}',
            "Median R": f'{stats["median_R"]:.3f}',
            "Exposure": fmt_pct(stats["exposure"]),
            "Capped entries": f'{int(stats["capped_entries"])}',
            "Gap exits": f'{int(stats["gap_exits"])}',
            "Point value": fmt_num(stats["point_value"]),
            "Max notional frac.": f'{stats["max_notional_frac"]:.2f}',
            "Total commission": fmt_num(stats["total_commission_cost"]),
            "Total slippage": fmt_num(stats["total_slippage_cost"]),
            "Total fixed fees": fmt_num(stats["total_fixed_fees"]),
            "All costs (net)": fmt_num(stats["total_costs"]),
            "Avg fee/trade": fmt_num(stats["avg_fee_per_trade"]),
        }
        for k,v in pretty.items():
            rows.append(f"<tr><td>{k}</td><td style='text-align:right'>{v}</td></tr>")

        html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Backtest Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }}
h1,h2 {{ margin: 0.4em 0; }}
table {{ border-collapse: collapse; width: 100%; max-width: 720px; }}
td,th {{ border-bottom: 1px solid #ddd; padding: 8px; }}
.section {{ margin: 24px 0; }}
</style>
</head>
<body>
<h1>Backtest Report</h1>
<div class="section">
  <h2>Summary</h2>
  <table>{''.join(rows)}</table>
</div>
<div class="section">
  <h2>Equity</h2>
  {img_tag(chart_paths.get('equity_curve',''))}
</div>
<div class="section">
  <h2>Drawdown</h2>
  {img_tag(chart_paths.get('underwater',''))}
</div>
<div class="section">
  <h2>Rolling Sharpe</h2>
  {img_tag(chart_paths.get('rolling_sharpe',''))}
</div>
<div class="section">
  <h2>Distributions</h2>
  {img_tag(chart_paths.get('pnl_hist',''))}
  {img_tag(chart_paths.get('r_hist',''))}
</div>
<div class="section">
  <h2>Monthly Returns</h2>
  {img_tag(chart_paths.get('monthly_heatmap',''))}
</div>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

    # ---------- TradingView Pine overlay export ----------
    def _export_tradingview_overlay(self, basepath: Path):
        '''Exports a Pine v5 overlay that plots entries/exits as labels and reconstructs TSL on-chart.
        File: <base>_tv_overlay.pine'''
        trades = [t for t in self.trades if t.exit_time is not None]
        if not trades:
            return None

        def ts_parts(ts: pd.Timestamp):
            ts = ts.to_pydatetime()
            return ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second

        # Build arrays as pine code
        entry_lines, exit_lines, entry_price_lines, exit_price_lines, side_lines, tp_lines = [], [], [], [], [], []
        for t in trades:
            y, m, d, hh, mm, ss = ts_parts(t.entry_time)
            ye, me, de, hhe, mme, sse = ts_parts(t.exit_time)
            entry_lines.append(f'    array.push(entry_t, timestamp("UTC", {y}, {m}, {d}, {hh}, {mm}, {ss}))')
            exit_lines.append (f'    array.push(exit_t,  timestamp("UTC", {ye}, {me}, {de}, {hhe}, {mme}, {sse}))')
            entry_price_lines.append(f'    array.push(entry_p, {t.entry_price:.10f})')
            exit_price_lines.append (f'    array.push(exit_p,  {t.exit_price if t.exit_price is not None else float("nan"):.10f})')
            side_val = 1 if t.side == "long" else -1
            side_lines.append       (f'    array.push(side_s,  {side_val})')
            tp_val = 'na' if (t.tp_at_entry is None or (isinstance(t.tp_at_entry, float) and (t.tp_at_entry!=t.tp_at_entry))) else f'{t.tp_at_entry:.10f}'
            tp_lines.append         (f'    array.push(tp_lvl,  {tp_val})')

        atr_p = self.cfg.atr_period
        atr_k = self.cfg.atr_mult_stop

        pine = f"""//@version=5
indicator("Backtest Trades Overlay (generated)", overlay=true, max_labels_count=20000, max_lines_count=20000)

// Parameters mirrored from backtest (for TSL reconstruction)
atrPeriod  = input.int({atr_p}, "ATR Period")
atrMult    = input.float({atr_k}, "ATR Mult for TSL")

// Arrays with trade events (times in UTC)
var entry_t = array.new_int()
var exit_t  = array.new_int()
var entry_p = array.new_float()
var exit_p  = array.new_float()
var side_s  = array.new_int()     // 1=long, -1=short
var tp_lvl  = array.new_float()

if barstate.isfirst
{chr(10).join(entry_lines)}
{chr(10).join(exit_lines)}
{chr(10).join(entry_price_lines)}
{chr(10).join(exit_price_lines)}
{chr(10).join(side_lines)}
{chr(10).join(tp_lines)}

// Helpers
isBetween(ts) =>
    // true on the bar that "captures" timestamp ts
    nz(time[1], time - 1) < ts and time >= ts

// State
var int idx_e = 0  // processed entries
var int idx_x = 0  // processed exits
var bool inPos = false
var int  side  = 0     // 1 long, -1 short
var float entryPrice = na
var float tpAtEntry  = na

// Reconstructed TSL series (using chart ATR)
atr = ta.atr(atrPeriod)
var float tsl = na
var float runHi = na
var float runLo = na

// Process entries on-bar
if idx_e < array.size(entry_t)
    nt = array.get(entry_t, idx_e)
    if isBetween(nt)
        inPos := true
        side  := array.get(side_s, idx_e)
        entryPrice := array.get(entry_p, idx_e)
        tpAtEntry  := array.get(tp_lvl, idx_e)
        label.new(bar_index, close, text="ENTRY " + (side==1?"LONG":"SHORT") + "\\n" + str.tostring(entryPrice, format.mintick),
            style=label.style_label_up, textcolor=color.white, size=size.tiny, color=(side==1?color.new(color.green, 0):color.new(color.red, 0)))
        // init TSL tracker
        runHi := close
        runLo := close
        tsl := na
        idx_e += 1

// Update trailing stop each bar while in position
if inPos
    if side == 1
        runHi := math.max(runHi, close)
        tsl := runHi - atrMult * atr
    else
        runLo := math.min(runLo, close)
        tsl := runLo + atrMult * atr

// Plot TP level (static from entry)
plot(inPos ? tpAtEntry : na, title="TP (at entry)", linewidth=1, style=plot.style_linebr, color=color.new(color.blue, 0))
// Plot TSL
plot(inPos ? tsl : na, title="TSL", linewidth=2, style=plot.style_linebr, color=color.new(color.orange, 0))

// Process exits on-bar
if idx_x < array.size(exit_t)
    xt = array.get(exit_t, idx_x)
    if isBetween(xt)
        px = array.get(exit_p, idx_x)
        reason = "EXIT"
        label.new(bar_index, close, text=reason + "\\n" + str.tostring(px, format.mintick),
            style=label.style_label_down, textcolor=color.white, size=size.tiny, color=color.new(color.gray, 0))
        inPos := false
        side  := 0
        entryPrice := na
        tpAtEntry  := na
        tsl := na
        runHi := na
        runLo := na
        idx_x += 1
"""
        pine_path = f"{basepath}_tv_overlay.pine"
        with open(pine_path, "w", encoding="utf-8") as f:
            f.write(pine)
        return pine_path

    # -------------------------- Main loop --------------------------
    def run(self) -> Dict[str, Any]:
        self._load_data_and_model()
        df = self.df
        cfg = self.cfg

        df["_ml_signal_runtime"] = np.nan

        # queue signals (compute on previous bar; execute later)
        for i in range(len(df)):
            if i % cfg.signal_every_n_bars == 0 and i >= 1:
                sig = self._predict_signal(df.iloc[[i-1]])
                exec_i = i + (cfg.signal_delay_bars - 1)
                self.pending_signal_queue[exec_i] = int(sig)
                df.iat[i, df.columns.get_loc("_ml_signal_runtime")] = int(sig)

        for i in range(len(df)):
            row = df.iloc[i]
            o = float(row[cfg.col_open]); h = float(row[cfg.col_high])
            l = float(row[cfg.col_low]);  c = float(row[cfg.col_close])
            atr_val = float(row["ATR"]) if not math.isnan(row["ATR"]) else None
            self.current_atr = atr_val if atr_val is not None else self.current_atr

            # 0) GAP-AWARE exits at OPEN before new signals
            if self.position_side is not None:
                if self.position_side == "long":
                    if self.stop_price is not None and o <= self.stop_price:
                        self._exit_position(i, o, reason="gap_stop_open"); self.gap_exits += 1
                    elif self.tp_price is not None and o >= self.tp_price:
                        self._exit_position(i, o, reason="gap_tp_open"); self.gap_exits += 1
                elif self.position_side == "short":
                    if self.stop_price is not None and o >= self.stop_price:
                        self._exit_position(i, o, reason="gap_stop_open"); self.gap_exits += 1
                    elif self.tp_price is not None and o <= self.tp_price:
                        self._exit_position(i, o, reason="gap_tp_open"); self.gap_exits += 1

            # 1) Execute pending signal at OPEN (including flat_on_low_conf when sig==0)
            if i in self.pending_signal_queue and self.current_atr is not None:
                signal = self.pending_signal_queue[i]
                # flat policy
                if signal == 0 and self.position_side is not None and cfg.flat_on_low_conf != "off":
                    do_flat = False
                    upnl_at_open = self._pnl_unrealized(o)
                    if cfg.flat_on_low_conf == "always":
                        do_flat = True
                    elif cfg.flat_on_low_conf == "loss_only":
                        do_flat = (upnl_at_open < 0)
                    elif cfg.flat_on_low_conf == "not_protected":
                        if self.position_side == "long":
                            do_flat = (self.stop_price is not None and self.entry_price is not None and self.stop_price <= self.entry_price)
                        else:
                            do_flat = (self.stop_price is not None and self.entry_price is not None and self.stop_price >= self.entry_price)
                    if do_flat:
                        self._exit_position(i, o, reason="flat_on_low_conf@open")

                # non-zero signals
                if signal != 0:
                    if self.position_side is None:
                        if signal == 1: self._enter_position(i, "long", o, self.current_atr)
                        elif signal == -1: self._enter_position(i, "short", o, self.current_atr)
                    else:
                        if self.position_side == "long" and signal == -1:
                            upnl = self._pnl_unrealized(o)
                            if not (upnl > 0 and (self.stop_price or -np.inf) > (self.entry_price or np.inf)):
                                self._exit_position(i, o, reason="reverse_from_long@open")
                                if cfg.allow_same_bar_reverse: self._enter_position(i, "short", o, self.current_atr)
                        elif self.position_side == "short" and signal == 1:
                            upnl = self._pnl_unrealized(o)
                            if not (upnl > 0 and (self.stop_price or np.inf) < (self.entry_price or -np.inf)):
                                self._exit_position(i, o, reason="reverse_from_short@open")
                                if cfg.allow_same_bar_reverse: self._enter_position(i, "long", o, self.current_atr)

            # 2) Update trailing stop on close
            if self.position_side is not None and self.current_atr is not None:
                self._update_trailing_stop(c, self.current_atr)
                self.position_open_bars += 1

            # 3) Exits
            if self.position_side is not None:
                if cfg.fills == "intrabar":
                    if self.position_side == "long":
                        if self.stop_price is not None and l <= self.stop_price <= h:
                            self._exit_position(i, self.stop_price, reason="tsl_stop")
                        elif self.tp_price is not None and l <= self.tp_price <= h:
                            self._exit_position(i, self.tp_price, reason="take_profit")
                    else:
                        if self.stop_price is not None and l <= self.stop_price <= h:
                            self._exit_position(i, self.stop_price, reason="tsl_stop")
                        elif self.tp_price is not None and l <= self.tp_price <= h:
                            self._exit_position(i, self.tp_price, reason="take_profit")
                else:
                    if self.position_side == "long":
                        if self.stop_price is not None and c <= self.stop_price:
                            self._exit_position(i, c, reason="tsl_stop_close")
                        elif self.tp_price is not None and c >= self.tp_price:
                            self._exit_position(i, c, reason="take_profit_close")
                    else:
                        if self.stop_price is not None and c >= self.stop_price:
                            self._exit_position(i, c, reason="tsl_stop_close")
                        elif self.tp_price is not None and c <= self.tp_price:
                            self._exit_position(i, c, reason="take_profit_close")

            # 4) MTM
            mtm = self._pnl_unrealized(c)
            self.equity_curve.append(self.equity + mtm)

        if self.position_side is not None:
            last_close = float(df.iloc[-1][cfg.col_close])
            self._exit_position(len(df)-1, last_close, reason="eod")

        eq = pd.Series(self.equity_curve, index=df.index[:len(self.equity_curve)])
        stats = self._calc_analytics(eq)

        return {"report": stats, "equity_curve": eq, "trades": self.trades, "runtime_config": asdict(self.cfg)}

# ----------------------------- Runner + CLI -----------------------------

def run_backtest(
    data_csv_path: str,
    model_path: str,
    scaler_path: str,
    best_features_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    signal_every_n_bars: int = 1,
    signal_delay_bars: int = 1,
    bars_per_year: int = 252*24*12,
    min_conf_long: float = 0.7,
    min_conf_short: float = 0.7,
    flat_on_low_conf: str = "off",
    risk_fraction: float = 0.02,
    atr_period: int = 14,
    atr_mult_stop: float = 2.0,
    atr_mult_tp: Optional[float] = 3.0,
    min_stop_pct_of_price: float = 0.0,
    initial_equity: float = 10_000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    slippage_jitter_bps: float = 0.0,
    leverage: float = 1.0,
    fixed_fee_per_fill: float = 0.0,
    max_notional_frac: float = 1.0,
    point_value: float = 1.0,
    real_median_price: Optional[float] = None,
    fills: str = "intrabar",
    make_charts: bool = True,
    log_equity: bool = False,
    per_trade_charts: int = 15,
) -> Dict[str, Any]:
    cfg = BacktestConfig(
        data_csv_path=data_csv_path,
        model_path=model_path,
        scaler_path=scaler_path,
        best_features_path=best_features_path,
        start_date=start_date,
        end_date=end_date,
        signal_every_n_bars=signal_every_n_bars,
        signal_delay_bars=signal_delay_bars,
        bars_per_year=bars_per_year,
        min_conf_long=min_conf_long,
        min_conf_short=min_conf_short,
        flat_on_low_conf=flat_on_low_conf,
        risk_fraction=risk_fraction,
        atr_period=atr_period,
        atr_mult_stop=atr_mult_stop,
        atr_mult_tp=atr_mult_tp,
        min_stop_pct_of_price=min_stop_pct_of_price,
        initial_equity=initial_equity,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        slippage_jitter_bps=slippage_jitter_bps,
        leverage=leverage,
        fixed_fee_per_fill=fixed_fee_per_fill,
        max_notional_frac=max_notional_frac,
        point_value=point_value,
        real_median_price=real_median_price,
        fills=fills,
        make_charts=make_charts,
        log_equity=log_equity,
        per_trade_charts=per_trade_charts,
    )
    engine = ProBacktester(cfg)
    out = engine.run()

    base = Path(data_csv_path).with_suffix("")
    eq_path = f"{base}_equity_curve.csv"
    trades_path = f"{base}_trades.csv"
    stats_path = f"{base}_stats.json"
    html_path = f"{base}_report.html"

    # Save artifacts
    pd.DataFrame({"equity": out["equity_curve"].values}, index=out["equity_curve"].index).to_csv(eq_path)
    trades_records = [asdict(t) for t in out["trades"]]
    pd.DataFrame(trades_records).to_csv(trades_path, index=False)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(out["report"], f, indent=2)

    chart_paths: Dict[str, str] = {}
    if make_charts:
        chart_paths = engine._save_charts(Path(data_csv_path).with_suffix(""), out["equity_curve"])
        trade_pngs = engine._save_trade_charts(Path(data_csv_path).with_suffix(""), max_trades=per_trade_charts)
        if trade_pngs:
            with open(f"{base}_trade_plots.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(trade_pngs))
    # Export TradingView overlay Pine script
    tv_path = engine._export_tradingview_overlay(Path(data_csv_path).with_suffix(""))
    if tv_path:
        print(f"Saved TradingView overlay: {tv_path}")

    engine._render_html_report(Path(html_path), out["report"], chart_paths)

    print("=== Backtest Report ===")
    for k, v in out["report"].items():
        print(f"{k:>22}: {v}")
    print(f"Saved equity curve CSV: {eq_path}")
    print(f"Saved trades CSV     : {trades_path}")
    print(f"Saved stats JSON     : {stats_path}")
    print(f"Saved HTML report    : {html_path}")
    if make_charts:
        print(f"Saved charts prefix  : {base}_*")
        print(f"Saved per-trade charts (up to {per_trade_charts}): see {base}_trade_plots.txt")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATR-TSL ML Backtester (no lookahead, confidence filters, caps) + trade charts + HTML + TV overlay")
    parser.add_argument("--data_csv_path", default="data_for_backtest.csv")
    parser.add_argument("--model_path", default="final_model.joblib")
    parser.add_argument("--scaler_path", default="final_scaler.joblib")
    parser.add_argument("--best_features_path", default="best_features.json")
    parser.add_argument("--start_date", default=None)
    parser.add_argument("--end_date", default=None)
    parser.add_argument("--signal_every_n_bars", type=int, default=1)
    parser.add_argument("--signal_delay_bars", type=int, default=1)
    parser.add_argument("--bars_per_year", type=int, default=252*24*12)
    parser.add_argument("--min_conf_long", type=float, default=0.95, help="70 or 0.7 mean 70%")
    parser.add_argument("--min_conf_short", type=float, default=0.95, help="70 or 0.7 mean 70%")
    parser.add_argument("--flat_on_low_conf", choices=["off","always","loss_only","not_protected"], default="off")
    parser.add_argument("--risk_fraction", type=float, default=0.02)
    parser.add_argument("--atr_period", type=int, default=14)
    parser.add_argument("--atr_mult_stop", type=float, default=2.0)
    parser.add_argument("--atr_mult_tp", type=float, default=3.5)
    parser.add_argument("--min_stop_pct_of_price", type=float, default=0.0)
    parser.add_argument("--initial_equity", type=float, default=1_000.0)
    parser.add_argument("--commission_rate", type=float, default=0.001)
    parser.add_argument("--slippage_rate", type=float, default=0.0005)
    parser.add_argument("--slippage_jitter_bps", type=float, default=0.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--fixed_fee_per_fill", type=float, default=1.0)
    parser.add_argument("--max_notional_frac", type=float, default=1.0)
    parser.add_argument("--point_value", type=float, default=1.0)
    parser.add_argument("--real_median_price", type=float, default=None, help="If set, auto point_value = real_median_price / csv_median_close (over test window)")
    parser.add_argument("--fills", choices=["intrabar","close"], default="close")
    parser.add_argument("--no_charts", action="store_true")
    parser.add_argument("--log_equity", action="store_true")
    parser.add_argument("--per_trade_charts", type=int, default=50)
    args = parser.parse_args()

    run_backtest(
        data_csv_path=args.data_csv_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        best_features_path=args.best_features_path,
        start_date=args.start_date,
        end_date=args.end_date,
        signal_every_n_bars=args.signal_every_n_bars,
        signal_delay_bars=args.signal_delay_bars,
        bars_per_year=args.bars_per_year,
        min_conf_long=args.min_conf_long,
        min_conf_short=args.min_conf_short,
        flat_on_low_conf=args.flat_on_low_conf,
        risk_fraction=args.risk_fraction,
        atr_period=args.atr_period,
        atr_mult_stop=args.atr_mult_stop,
        atr_mult_tp=args.atr_mult_tp,
        min_stop_pct_of_price=args.min_stop_pct_of_price,
        initial_equity=args.initial_equity,
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
        slippage_jitter_bps=args.slippage_jitter_bps,
        leverage=args.leverage,
        fixed_fee_per_fill=args.fixed_fee_per_fill,
        max_notional_frac=args.max_notional_frac,
        point_value=args.point_value,
        real_median_price=args.real_median_price,
        fills=args.fills,
        make_charts=(not args.no_charts),
        log_equity=args.log_equity,
        per_trade_charts=args.per_trade_charts,
    )
