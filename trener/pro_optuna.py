# pro_optuna.py
# Hyperparameter search for pro_backtester.run_backtest using Optuna.
# Tunes: risk_fraction, atr_mult_stop (TSL), atr_mult_tp (ratio or None).
#
# Example:
# python pro_optuna.py \
#   --data_csv_path data_for_backtest.csv \
#   --model_path final_model.joblib \
#   --scaler_path final_scaler.joblib \
#   --best_features_path best_features.json \
#   --start_date "2025-08-01 00:00" --end_date "2025-08-31 23:59" \
#   --min_conf_long 0.70 --min_conf_short 0.70 \
#   --flat_on_low_conf off --fills close \
#   --initial_equity 10000 --fixed_fee_per_fill 1.0 \
#   --max_notional_frac 0.5 --real_median_price 0.221 \
#   --min_stop_pct_of_price 0.002 \
#   --metric calmar --allow_no_tp \
#   --n_trials 80 --study_name DOGE_Aug2025_calmar \
#   --storage sqlite:///optuna_doge.db \
#   --n_jobs -1

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from pro_backtester import run_backtest


def parse_args():
    p = argparse.ArgumentParser(description="Optuna tuner for pro_backtester (ATR-TSL + risk)")
    # artifacts & data
    p.add_argument("--data_csv_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--scaler_path", required=True)
    p.add_argument("--best_features_path", required=True)
    p.add_argument("--start_date", required=True)
    p.add_argument("--end_date", required=True)
    # strategy fixed knobs
    p.add_argument("--min_conf_long", type=float, default=0.70)
    p.add_argument("--min_conf_short", type=float, default=0.70)
    p.add_argument("--flat_on_low_conf", choices=["off","always","loss_only","not_protected"], default="off")
    p.add_argument("--fills", choices=["intrabar","close"], default="close")
    p.add_argument("--initial_equity", type=float, default=10_000.0)
    p.add_argument("--commission_rate", type=float, default=0.0)
    p.add_argument("--slippage_rate", type=float, default=0.0)
    p.add_argument("--fixed_fee_per_fill", type=float, default=1.0)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--max_notional_frac", type=float, default=0.5)
    p.add_argument("--real_median_price", type=float, default=None)
    p.add_argument("--min_stop_pct_of_price", type=float, default=0.002)
    p.add_argument("--signal_delay_bars", type=int, default=1)
    p.add_argument("--bars_per_year", type=int, default=252*24*12)
    p.add_argument("--per_trade_charts", type=int, default=0)  # off for speed
    # optimization ranges
    p.add_argument("--risk_min", type=float, default=0.005)   # 0.5%
    p.add_argument("--risk_max", type=float, default=0.04)    # 4%
    p.add_argument("--stop_min", type=float, default=1.0)     # ATR mult for TSL
    p.add_argument("--stop_max", type=float, default=4.0)
    p.add_argument("--tp_ratio_min", type=float, default=1.2) # TP = stop * ratio
    p.add_argument("--tp_ratio_max", type=float, default=4.0)
    p.add_argument("--allow_no_tp", action="store_true", help="Try also TP=None during search")
    # objective
    p.add_argument("--metric", choices=["calmar","sharpe","sortino","final_equity","profit_factor","expectancy"], default="calmar")
    p.add_argument("--exposure_max", type=float, default=1.0, help="Penalty if exposure exceeds this (0..1). 1.0 = no penalty.")
    p.add_argument("--dd_penalty", type=float, default=0.0, help="Subtract dd_penalty*|max_drawdown| from objective.")
    # CV / robustness
    p.add_argument("--n_splits", type=int, default=1, help="Walk-forward splits over [start,end]. 1 = single run.")
    # optuna infra
    p.add_argument("--n_trials", type=int, default=60)
    p.add_argument("--study_name", type=str, default="backtest_tuning")
    p.add_argument("--storage", type=str, default=None, help="e.g. sqlite:///optuna.db")
    p.add_argument("--seed", type=int, default=42)
    # parallelism
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel Optuna jobs. -1/0/None => use all CPUs.")
    # save prefix
    p.add_argument("--save_prefix", type=str, default="data_for_backtest")
    # convergence plot/diagnostics
    p.add_argument("--convergence_tail_k", type=int, default=20, help="Tail window for plateau diagnostics.")
    p.add_argument("--no_convergence_plot", action="store_true", help="Disable saving convergence plot PNG.")
    return p.parse_args()


def date_linspace(start: str, end: str, n: int) -> List[Tuple[str,str]]:
    s = datetime.fromisoformat(start.replace("Z",""))
    e = datetime.fromisoformat(end.replace("Z",""))
    if n <= 1:
        return [(start, end)]
    dur = (e - s) / n
    spans = []
    left = s
    for i in range(n):
        right = s + (i+1)*dur
        if i == n-1: right = e
        spans.append((left.isoformat(sep=" "), right.isoformat(sep=" ")))
        left = right
    return spans


def metric_value(stats: Dict[str, Any], metric: str, exposure_max: float, dd_penalty: float) -> float:
    val = {
        "calmar": stats.get("calmar", -np.inf),
        "sharpe": stats.get("sharpe", -np.inf),
        "sortino": stats.get("sortino", -np.inf),
        "final_equity": stats.get("final_equity", -np.inf),
        "profit_factor": stats.get("profit_factor", -np.inf),
        "expectancy": stats.get("expectancy", -np.inf),
    }[metric]
    # exposure cap penalty
    exp = float(stats.get("exposure", 1.0))
    if exp > exposure_max:
        val -= 10.0 * (exp - exposure_max)
    # drawdown penalty
    dd = abs(float(stats.get("max_drawdown", 0.0)))
    val -= dd_penalty * dd
    return float(val)


def build_params(trial: optuna.trial.Trial, args) -> Dict[str, Any]:
    risk_fraction = trial.suggest_float("risk_fraction", args.risk_min, args.risk_max, log=True)
    atr_mult_stop = trial.suggest_float("atr_mult_stop", args.stop_min, args.stop_max)
    tp_mode = "ratio"
    if args.allow_no_tp:
        tp_mode = trial.suggest_categorical("tp_mode", ["ratio","none"])
    if tp_mode == "none":
        atr_mult_tp = None
    else:
        ratio = trial.suggest_float("tp_ratio", args.tp_ratio_min, args.tp_ratio_max)
        atr_mult_tp = atr_mult_stop * ratio
    return dict(risk_fraction=risk_fraction, atr_mult_stop=atr_mult_stop, atr_mult_tp=atr_mult_tp)


def run_span(args, params: Dict[str, Any], s: str, e: str) -> Dict[str, Any]:
    out = run_backtest(
        data_csv_path=args.data_csv_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        best_features_path=args.best_features_path,
        start_date=s,
        end_date=e,
        signal_every_n_bars=1,
        signal_delay_bars=args.signal_delay_bars,
        bars_per_year=args.bars_per_year,
        min_conf_long=args.min_conf_long,
        min_conf_short=args.min_conf_short,
        flat_on_low_conf=args.flat_on_low_conf,
        risk_fraction=params["risk_fraction"],
        atr_period=14,
        atr_mult_stop=params["atr_mult_stop"],
        atr_mult_tp=params["atr_mult_tp"],
        min_stop_pct_of_price=args.min_stop_pct_of_price,
        initial_equity=args.initial_equity,
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
        slippage_jitter_bps=0.0,
        leverage=args.leverage,
        max_notional_frac=args.max_notional_frac,
        point_value=1.0,
        real_median_price=args.real_median_price,
        fills=args.fills,
        make_charts=False,
        log_equity=False,
        per_trade_charts=0,
        fixed_fee_per_fill=args.fixed_fee_per_fill,
    )
    return out["report"]


def objective(trial: optuna.trial.Trial, args) -> float:
    params = build_params(trial, args)
    spans = date_linspace(args.start_date, args.end_date, args.n_splits)
    vals = []
    for (s,e) in spans:
        stats = run_span(args, params, s, e)
        val = metric_value(stats, args.metric, args.exposure_max, args.dd_penalty)
        vals.append(val)
        trial.report(float(np.mean(vals)), step=len(vals))
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(vals))


def save_artifacts_and_convergence(study: optuna.Study, args, prefix: Path):
    # Trials dataframe
    rows = []
    for t in study.trials:
        rows.append({**t.params, "value": t.value, "number": t.number, "state": str(t.state)})
    df = pd.DataFrame(rows).sort_values("number", ascending=True)
    csv_path = f"{prefix}_optuna_trials.csv"
    df.to_csv(csv_path, index=False)

    # Convergence plot (value & running best vs trial number)
    if not args.no_convergence_plot and len(df):
        x = df["number"].to_numpy()
        y = df["value"].to_numpy(dtype=float)
        # running best (max so far)
        run_best = np.maximum.accumulate(np.nan_to_num(y, nan=-np.inf))

        plt.figure()
        plt.scatter(x, y, s=12)        # wszystkie wartości
        plt.plot(x, run_best)          # linia najlepszego dotychczas
        plt.title(f"Optuna convergence: {args.metric}")
        plt.xlabel("Trial number")
        plt.ylabel(args.metric)
        plt.tight_layout()
        png_path = f"{prefix}_optuna_convergence.png"
        plt.savefig(png_path)
        plt.close()
        print(f"Saved convergence plot: {png_path}")

        # Plateau diagnostics (ostatnie K prób)
        k = int(args.convergence_tail_k)
        if k > 0 and len(run_best) >= k:
            tail = run_best[-k:]
            tail_improvement = float(tail[-1] - tail[0])
            per_trial = tail_improvement / (k - 1 if k > 1 else 1)
            print(f"[Convergence] Last {k} trials: best improved by {tail_improvement:.6g} ({per_trial:.6g} per trial).")
            if abs(per_trial) < 1e-4:
                print("[Convergence] Plateau detected (avg improvement < 1e-4 per trial). Consider stopping.")

    # Best params JSON
    best = study.best_trial
    best_params = best.params.copy()
    if ("tp_ratio" in best_params) and ("atr_mult_stop" in best_params) and ("atr_mult_tp" not in best_params):
        best_params["atr_mult_tp"] = best_params["atr_mult_stop"] * best_params["tp_ratio"]
    best_json = f"{prefix}_best_params.json"
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    # Ready-to-run command
    tp_arg = f"--atr_mult_tp {best_params['atr_mult_tp']}" if best_params.get("atr_mult_tp", None) is not None else "--atr_mult_tp None"
    cmd = f"""
python pro_backtester.py \
  --data_csv_path {args.data_csv_path} \
  --model_path {args.model_path} \
  --scaler_path {args.scaler_path} \
  --best_features_path {args.best_features_path} \
  --start_date "{args.start_date}" \
  --end_date   "{args.end_date}" \
  --signal_delay_bars {args.signal_delay_bars} \
  --min_conf_long {args.min_conf_long} \
  --min_conf_short {args.min_conf_short} \
  --flat_on_low_conf {args.flat_on_low_conf} \
  --risk_fraction {best_params['risk_fraction']} \
  --atr_period 14 \
  --atr_mult_stop {best_params['atr_mult_stop']} \
  {tp_arg} \
  --min_stop_pct_of_price {args.min_stop_pct_of_price} \
  --initial_equity {args.initial_equity} \
  --commission_rate {args.commission_rate} \
  --slippage_rate {args.slippage_rate} \
  --fixed_fee_per_fill {args.fixed_fee_per_fill} \
  --leverage {args.leverage} \
  --max_notional_frac {args.max_notional_frac} \
  --real_median_price {args.real_median_price if args.real_median_price is not None else ''} \
  --fills {args.fills} \
  --per_trade_charts 15 \
  --log_equity
"""
    print("\n=== Best trial ===")
    print(f"number={best.number}  value={best.value:.6f}")
    print("params:", json.dumps(best_params, indent=2))
    print("\nRun best backtest with:")
    print(cmd)


def main():
    args = parse_args()

    # Resolve n_jobs: -1 / 0 / None => all CPUs
    if args.n_jobs in (-1, 0, None):
        args.n_jobs = os.cpu_count() or 1

    sampler = TPESampler(seed=args.seed, multivariate=True, group=True)
    pruner = MedianPruner(n_warmup_steps=max(1, args.n_splits//2))
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True if args.storage else False,
    )
    study.optimize(lambda tr: objective(tr, args),
                   n_trials=args.n_trials,
                   n_jobs=args.n_jobs,
                   gc_after_trial=True)

    prefix = Path(args.save_prefix).with_suffix("")
    save_artifacts_and_convergence(study, args, prefix)


if __name__ == "__main__":
    main()
