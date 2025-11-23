"""
PROFIT-AWARE OPTIMIZATION MODULE - FIXED VERSION
=================================================

FIX: Removed look-ahead bias from trade simulation.

Original problem:
- simulate_trades_vectorized() used pre-computed labels (y_true)
- Labels from triple-barrier LOOK 24 candles forward
- Threshold optimization "knew" which trades would be profitable

Fixed version:
- simulate_trades_forward() ACTUALLY simulates forward from entry
- No pre-computed labels used
- Checks TP/SL timing candle-by-candle
- TRUE backtesting without look-ahead bias

Author: Łukasz + Claude
Created: 2024-11-23 (original)
Fixed: 2024-11-23 (look-ahead bias removal)
Version: 2.1 (fixed)
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple, Dict, List
import logging
from sklearn.metrics import precision_score, recall_score

logger = logging.getLogger(__name__)


@njit
def _simulate_single_trade(entry_idx: int,
                          prices: np.ndarray,
                          side: int,  # 1=LONG, -1=SHORT
                          tp_pct: float,
                          sl_pct: float,
                          max_holding_period: int) -> float:
    """
    Simulate single trade forward WITHOUT knowing outcome.

    Args:
        entry_idx: Entry candle index
        prices: Full price array
        side: 1 for LONG, -1 for SHORT
        tp_pct: Take profit %
        sl_pct: Stop loss %
        max_holding_period: Max candles to hold

    Returns:
        Trade return (%, negative if loss)
    """
    entry_price = prices[entry_idx]

    # Calculate TP/SL levels
    if side == 1:  # LONG
        tp_level = entry_price * (1 + tp_pct)
        sl_level = entry_price * (1 - sl_pct)
    else:  # SHORT
        tp_level = entry_price * (1 - tp_pct)
        sl_level = entry_price * (1 + sl_pct)

    # Simulate forward
    for i in range(1, max_holding_period + 1):
        current_idx = entry_idx + i

        # Check if reached end of data
        if current_idx >= len(prices):
            # Exit at last available price (timeout)
            exit_price = prices[-1]
            break

        current_price = prices[current_idx]

        # Check TP/SL hits
        if side == 1:  # LONG
            if current_price >= tp_level:
                # TP hit
                return tp_pct
            elif current_price <= sl_level:
                # SL hit
                return -sl_pct
        else:  # SHORT
            if current_price <= tp_level:
                # TP hit
                return tp_pct
            elif current_price >= sl_level:
                # SL hit
                return -sl_pct

    # Timeout - exit at market
    exit_price = prices[min(entry_idx + max_holding_period, len(prices) - 1)]
    if side == 1:
        return_pct = (exit_price - entry_price) / entry_price
    else:
        return_pct = (entry_price - exit_price) / entry_price

    return return_pct


def simulate_trades_forward(y_pred: np.ndarray,
                            prices: pd.Series,
                            side: int = 1,
                            tp_pct: float = 0.03,
                            sl_pct: float = 0.01,
                            fee_pct: float = 0.0006,
                            max_holding_period: int = None) -> pd.Series:
    """
    Forward-looking trade simulation WITHOUT look-ahead bias.

    CRITICAL FIX: Does NOT use pre-computed labels (y_true).
    Instead, simulates forward from each entry point.

    Args:
        y_pred: Predicted signals (0=HOLD, 1=SIGNAL)
        prices: Price series
        side: 1 for LONG predictions, -1 for SHORT predictions
        tp_pct: Take profit %
        sl_pct: Stop loss %
        fee_pct: Round-trip fee %
        max_holding_period: Max candles to hold trade

    Returns:
        Series of trade returns (% PnL per trade)

    Example:
        # LONG model predictions
        returns = simulate_trades_forward(
            y_pred=predictions,
            prices=df['close'],
            side=1,  # LONG
            tp_pct=0.03,
            sl_pct=0.01
        )

        # SHORT model predictions
        returns = simulate_trades_forward(
            y_pred=predictions,
            prices=df['close'],
            side=-1,  # SHORT
            tp_pct=0.03,
            sl_pct=0.01
        )
    """
    # FIX #5: Auto-detect max_holding_period if not specified
    if max_holding_period is None:
        # Default: 50 candles (conservative)
        # In practice, this should match label_params['time_limit']
        # which will be passed from find_optimal_threshold_for_profit()
        max_holding_period = 50

    prices_arr = prices.values
    returns = np.zeros(len(y_pred))

    # Find all entry points
    entry_indices = np.where(y_pred == 1)[0]

    # Simulate each trade
    for entry_idx in entry_indices:
        # Skip if too close to end (can't hold full period)
        if entry_idx >= len(prices_arr) - 2:
            returns[entry_idx] = 0.0
            continue

        # Simulate trade forward (NO look-ahead!)
        gross_return = _simulate_single_trade(
            entry_idx,
            prices_arr,
            side,
            tp_pct,
            sl_pct,
            max_holding_period
        )

        # Subtract fees
        net_return = gross_return - fee_pct

        returns[entry_idx] = net_return

    return pd.Series(returns, index=prices.index)


def calculate_sharpe_ratio(returns: pd.Series,
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 35040) -> float:
    """Calculate annualized Sharpe Ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = (mean_return - risk_free_rate / periods_per_year) / std_return * np.sqrt(periods_per_year)

    return sharpe


def calculate_trading_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive trading metrics."""
    if len(returns) == 0:
        return {
            'sharpe_ratio': 0.0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'n_trades': 0
        }

    # Filter out HOLD periods (0 returns)
    trade_returns = returns[returns != 0]

    if len(trade_returns) == 0:
        return {
            'sharpe_ratio': 0.0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'n_trades': 0
        }

    # Basic metrics
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0.0
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Cumulative metrics
    cumulative_returns = trade_returns.cumsum()
    total_pnl = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0.0

    # Max drawdown
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns - running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

    # Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(trade_returns)

    return {
        'sharpe_ratio': sharpe_ratio,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'n_trades': len(trade_returns)
    }


def profit_aware_objective_score(y_pred: np.ndarray,
                                 prices: pd.Series,
                                 side: int = 1,
                                 tp_pct: float = 0.03,
                                 sl_pct: float = 0.01,
                                 fee_pct: float = 0.0006,
                                 min_trades: int = 10) -> float:
    """
    Profit-aware objective function (FIXED - no look-ahead bias).

    CRITICAL FIX: Does NOT use y_true (pre-computed labels).
    Uses forward simulation instead.

    Args:
        y_pred: Predicted labels (0=HOLD, 1=SIGNAL)
        prices: Price series
        side: 1 for LONG, -1 for SHORT
        tp_pct: Take profit %
        sl_pct: Stop loss %
        fee_pct: Round-trip fee %
        min_trades: Minimum trades required

    Returns:
        Composite score (0-1 range, higher = better)
    """
    # Simulate trades (NO look-ahead!)
    returns = simulate_trades_forward(y_pred, prices, side, tp_pct, sl_pct, fee_pct)

    # Calculate metrics
    metrics = calculate_trading_metrics(returns)

    # Penalize too few trades
    if metrics['n_trades'] < min_trades:
        penalty = metrics['n_trades'] / min_trades
    else:
        penalty = 1.0

    # Penalize negative Sharpe heavily
    if metrics['sharpe_ratio'] < 0:
        return 0.0

    # Normalize Sharpe (clip to 0-3 range)
    normalized_sharpe = np.clip(metrics['sharpe_ratio'] / 3.0, 0, 1)

    # Normalize win rate (already 0-1)
    normalized_wr = metrics['win_rate']

    # Normalize PnL (assuming +50% total PnL is excellent)
    normalized_pnl = np.clip(metrics['total_pnl'] / 0.5, 0, 1)

    # Composite score
    score = (
        0.50 * normalized_sharpe +
        0.30 * normalized_wr +
        0.20 * normalized_pnl
    ) * penalty

    return score


def find_optimal_threshold_for_profit(model,
                                       X_calib: pd.DataFrame,
                                       prices: pd.Series,
                                       side: str = 'long',
                                       tp_pct: float = 0.03,
                                       sl_pct: float = 0.01,
                                       fee_pct: float = 0.0006,
                                       threshold_range: Tuple[float, float] = (0.30, 0.85),
                                       threshold_step: float = 0.01,
                                       min_trades: int = 10,
                                       max_holding_period: int = None) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold for profit (FIXED - no look-ahead bias).

    CRITICAL FIX: Does NOT use y_calib (pre-computed labels).
    Uses forward simulation instead.

    Args:
        model: Trained classifier
        X_calib: Calibration features
        prices: Price series (aligned with X_calib)
        side: 'long' or 'short'
        tp_pct: Take profit %
        sl_pct: Stop loss %
        fee_pct: Trading fees %
        threshold_range: (min, max) thresholds
        threshold_step: Granularity
        min_trades: Minimum trades required

    Returns:
        Tuple of (optimal_threshold, metrics_at_optimal)
    """
    logger.info("\n" + "="*60)
    logger.info("PROFIT-AWARE THRESHOLD OPTIMIZATION (FIXED - NO BIAS)")
    logger.info("="*60)
    logger.info(f"Side: {side.upper()}")
    logger.info(f"Searching thresholds: {threshold_range[0]:.2f} - {threshold_range[1]:.2f} (step: {threshold_step})")

    # Get model probabilities
    probas = model.predict_proba(X_calib)[:, 1]

    # Determine side multiplier
    side_multiplier = 1 if side == 'long' else -1

    best_score = -np.inf
    best_threshold = 0.5
    best_metrics = {}

    results = []

    # Search over threshold range
    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)

    for threshold in thresholds:
        # Generate predictions at this threshold
        y_pred = (probas >= threshold).astype(int)

        # Simulate trades (NO look-ahead!)
        # FIX #5: Pass max_holding_period to match label params
        returns = simulate_trades_forward(
            y_pred, prices, side_multiplier, tp_pct, sl_pct, fee_pct,
            max_holding_period=max_holding_period
        )

        # Calculate metrics
        metrics = calculate_trading_metrics(returns)

        # Calculate composite score
        score = profit_aware_objective_score(
            y_pred, prices, side_multiplier, tp_pct, sl_pct, fee_pct, min_trades
        )

        results.append({
            'threshold': threshold,
            'score': score,
            'sharpe': metrics['sharpe_ratio'],
            'total_pnl': metrics['total_pnl'],
            'win_rate': metrics['win_rate'],
            'n_trades': metrics['n_trades']
        })

        # Update best
        if score > best_score and metrics['n_trades'] >= min_trades:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Log top 5 thresholds
    logger.info(f"\nTop 5 thresholds by profit score:")
    top_5 = results_df.nlargest(5, 'score')
    for idx, row in top_5.iterrows():
        logger.info(f"  Threshold={row['threshold']:.2f}: "
                   f"Score={row['score']:.3f}, "
                   f"Sharpe={row['sharpe']:.2f}, "
                   f"PnL={row['total_pnl']*100:.1f}%, "
                   f"WR={row['win_rate']*100:.1f}%, "
                   f"Trades={int(row['n_trades'])}")

    logger.info(f"\n✅ OPTIMAL THRESHOLD: {best_threshold:.2f}")
    logger.info(f"   Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
    logger.info(f"   Total PnL: {best_metrics['total_pnl']*100:.1f}%")
    logger.info(f"   Win Rate: {best_metrics['win_rate']*100:.1f}%")
    logger.info(f"   Profit Factor: {best_metrics['profit_factor']:.2f}")
    logger.info(f"   Max Drawdown: {best_metrics['max_drawdown']*100:.1f}%")
    logger.info(f"   Number of Trades: {best_metrics['n_trades']}")
    logger.info("="*60)

    return best_threshold, best_metrics


def evaluate_on_holdout(model,
                        X_holdout: pd.DataFrame,
                        prices: pd.Series,
                        optimal_threshold: float,
                        side: str = 'long',
                        tp_pct: float = 0.03,
                        sl_pct: float = 0.01,
                        fee_pct: float = 0.0006) -> Dict[str, float]:
    """
    Final evaluation on holdout (FIXED - no look-ahead bias).

    CRITICAL FIX: Does NOT use y_holdout (pre-computed labels).
    Uses forward simulation instead.
    """
    logger.info("\n" + "="*60)
    logger.info("HOLDOUT SET EVALUATION (FIXED - NO BIAS)")
    logger.info("="*60)

    # Predict with optimal threshold
    probas = model.predict_proba(X_holdout)[:, 1]
    y_pred = (probas >= optimal_threshold).astype(int)

    # Determine side
    side_multiplier = 1 if side == 'long' else -1

    # Simulate trades (NO look-ahead!)
    returns = simulate_trades_forward(
        y_pred, prices, side_multiplier, tp_pct, sl_pct, fee_pct
    )

    # Calculate metrics
    metrics = calculate_trading_metrics(returns)

    logger.info(f"Threshold: {optimal_threshold:.2f}")
    logger.info(f"\nTrading Metrics:")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Total PnL: {metrics['total_pnl']*100:.1f}%")
    logger.info(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
    logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
    logger.info(f"  Avg Win: {metrics['avg_win']*100:.2f}%")
    logger.info(f"  Avg Loss: {metrics['avg_loss']*100:.2f}%")
    logger.info(f"  Number of Trades: {metrics['n_trades']}")

    # Deployment recommendation
    if metrics['sharpe_ratio'] > 1.0 and metrics['win_rate'] > 0.45 and metrics['n_trades'] >= 20:
        recommendation = "✅ READY FOR DEPLOYMENT"
    elif metrics['sharpe_ratio'] > 0.5 and metrics['win_rate'] > 0.40:
        recommendation = "⚠️  CAUTIOUS DEPLOYMENT (paper trading first)"
    else:
        recommendation = "❌ NOT READY (retrain with different parameters)"

    logger.info(f"\n{recommendation}")
    logger.info("="*60)

    metrics['threshold'] = optimal_threshold

    return metrics
