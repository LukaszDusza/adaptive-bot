#!/usr/bin/env python3
"""
Professional Backtester V2 - Without Look-Ahead Bias
Fixes all critical issues from the original backtester.
"""

import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import timedelta

# Import data preparation module
from data_preparer_pa import fetch_and_prepare_data

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class Position:
    """Data structure for open position"""
    side: str  # 'Long' or 'Short'
    entry_price: float
    entry_time: pd.Timestamp
    current_size: float
    initial_size: float
    stop_loss: float
    take_profit: float
    highest_price: float  # For TSL (Long)
    lowest_price: float  # For TSL (Short)
    partial_tp_taken: bool = False
    tp_pct: float = 0.0
    entry_probability: float = 0.0


@dataclass
class Trade:
    """Data structure for closed trade"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl_pct: float
    pnl_usd: float
    fees_usd: float
    net_pnl_usd: float
    exit_reason: str
    duration: timedelta
    mae_pct: float  # Maximum Adverse Excursion
    mfe_pct: float  # Maximum Favorable Excursion
    partial_tp_hit: bool = False
    partial_tp_time: Optional[pd.Timestamp] = None
    partial_tp_price: Optional[float] = None
    tsl_history: Optional[str] = None  # JSON string of TSL updates
    chart_ohlcv: Optional[str] = None  # JSON string of OHLCV data for chart


class BacktestEngine:
    """Backtest engine without look-ahead bias"""
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 maker_fee: float = 0.0002,
                 taker_fee: float = 0.00055,
                 slippage_pct: float = 0.0001):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_pct = slippage_pct
        
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.decision_log: List[Dict] = []
        
        # Track current trade data
        self.current_trade_ohlcv: List[Dict] = []
        self.current_tsl_history: List[Tuple] = []
        self.partial_tp_hit: bool = False
        self.partial_tp_time: Optional[pd.Timestamp] = None
        self.partial_tp_price: Optional[float] = None
        
    def calculate_fees(self, price: float, size: float, is_maker: bool = False) -> float:
        """Calculate transaction fees"""
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return price * size * fee_rate / price  # Returns fee in USD
    
    def apply_slippage(self, price: float, side: str, is_entry: bool = True) -> float:
        """Apply slippage to price"""
        if is_entry:
            if side == 'Long':
                return price * (1 + self.slippage_pct)
            else:
                return price * (1 - self.slippage_pct)
        else:
            if side == 'Long':
                return price * (1 - self.slippage_pct)
            else:
                return price * (1 + self.slippage_pct)
    
    def update_position_extremes(self, candle: pd.Series):
        """Update highest/lowest prices for TSL"""
        if not self.position:
            return
            
        if self.position.side == 'Long':
            self.position.highest_price = max(self.position.highest_price, candle['high'])
        else:
            self.position.lowest_price = min(self.position.lowest_price, candle['low'])
    
    def calculate_mae_mfe(self, position: Position, candle: pd.Series) -> Tuple[float, float]:
        """Calculate MAE and MFE for position"""
        if position.side == 'Long':
            mae_pct = (candle['low'] / position.entry_price - 1) * 100
            mfe_pct = (candle['high'] / position.entry_price - 1) * 100
        else:
            mae_pct = (position.entry_price / candle['high'] - 1) * 100
            mfe_pct = (position.entry_price / candle['low'] - 1) * 100
        
        return min(mae_pct, 0), max(mfe_pct, 0)
    
    def check_exit_conditions(self, candle: pd.Series) -> Tuple[bool, Optional[float], Optional[str]]:
        """Check position exit conditions"""
        if not self.position:
            return False, None, None
        
        pos = self.position
        
        if pos.side == 'Long':
            if candle['low'] <= pos.stop_loss:
                return True, pos.stop_loss, 'SL'
            if candle['high'] >= pos.take_profit and pos.take_profit != np.inf:
                return True, pos.take_profit, 'TP'
        else:
            if candle['high'] >= pos.stop_loss:
                return True, pos.stop_loss, 'SL'
            if candle['low'] <= pos.take_profit and pos.take_profit != 0:
                return True, pos.take_profit, 'TP'
        
        return False, None, None
    
    def check_partial_tp(self, candle: pd.Series) -> Tuple[bool, Optional[float]]:
        """Check if partial TP level reached"""
        if not self.position or self.position.partial_tp_taken:
            return False, None
        
        pos = self.position
        
        if pos.take_profit == np.inf or pos.take_profit == 0:
            return False, None
            
        partial_tp_price = pos.entry_price + (pos.take_profit - pos.entry_price) / 2 if pos.side == 'Long' \
                          else pos.entry_price - (pos.entry_price - pos.take_profit) / 2
        
        if pos.side == 'Long' and candle['high'] >= partial_tp_price:
            return True, partial_tp_price
        elif pos.side == 'Short' and candle['low'] <= partial_tp_price:
            return True, partial_tp_price
        
        return False, None
    
    def close_position(self, exit_price: float, exit_time: pd.Timestamp, 
                       exit_reason: str, mae_pct: float, mfe_pct: float,
                       size_to_close: Optional[float] = None):
        """Close position (fully or partially)"""
        if not self.position:
            return
        
        pos = self.position
        close_size = size_to_close if size_to_close else pos.current_size
        
        # Apply slippage
        exit_price_with_slippage = self.apply_slippage(exit_price, pos.side, is_entry=False)
        
        # Calculate P&L
        if pos.side == 'Long':
            pnl_pct = (exit_price_with_slippage / pos.entry_price - 1)
        else:
            pnl_pct = (pos.entry_price / exit_price_with_slippage - 1)
        
        pnl_usd = pnl_pct * close_size
        
        # Calculate fees
        entry_fee = self.calculate_fees(pos.entry_price, close_size, is_maker=True)
        exit_fee = self.calculate_fees(exit_price_with_slippage, close_size, is_maker=False)
        total_fees = entry_fee + exit_fee
        
        net_pnl_usd = pnl_usd - total_fees
        
        # Prepare chart OHLCV data
        chart_ohlcv_json = None
        if self.current_trade_ohlcv:
            # Convert timestamp to milliseconds for JSON serialization
            ohlcv_data = []
            for candle in self.current_trade_ohlcv:
                candle_data = candle.copy()
                if isinstance(candle_data['timestamp'], pd.Timestamp):
                    candle_data['timestamp'] = int(candle_data['timestamp'].timestamp() * 1000)
                ohlcv_data.append(candle_data)
            
            df_chart = pd.DataFrame(ohlcv_data)
            chart_ohlcv_json = json.dumps({
                'index': df_chart['timestamp'].tolist(),
                'columns': [col for col in df_chart.columns if col != 'timestamp'],
                'data': df_chart[[col for col in df_chart.columns if col != 'timestamp']].values.tolist()
            })
        
        # Prepare TSL history
        tsl_history_json = None
        if self.current_tsl_history:
            tsl_history_json = json.dumps([[str(t), float(sl)] for t, sl in self.current_tsl_history])
        
        # Save trade
        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=exit_time,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price_with_slippage,
            size=close_size,
            pnl_pct=pnl_pct * 100,
            pnl_usd=pnl_usd,
            fees_usd=total_fees,
            net_pnl_usd=net_pnl_usd,
            exit_reason=exit_reason,
            duration=exit_time - pos.entry_time,
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
            partial_tp_hit=self.partial_tp_hit,
            partial_tp_time=self.partial_tp_time,
            partial_tp_price=self.partial_tp_price,
            tsl_history=tsl_history_json,
            chart_ohlcv=chart_ohlcv_json
        )
        
        self.trades.append(trade)
        self.current_capital += net_pnl_usd
        self.equity_curve.append(self.current_capital)
        
        # Update position or close
        if size_to_close:
            pos.current_size -= close_size
            pos.partial_tp_taken = True
            pos.stop_loss = pos.entry_price
            logging.info(f"Partial close: {exit_reason}, Net P&L: ${net_pnl_usd:.2f}")
        else:
            # Reset tracking for next trade
            self.current_trade_ohlcv = []
            self.current_tsl_history = []
            self.partial_tp_hit = False
            self.partial_tp_time = None
            self.partial_tp_price = None
            self.position = None
            logging.info(f"Position closed: {exit_reason}, Net P&L: ${net_pnl_usd:.2f}")
    
    def update_trailing_stop(self, candle: pd.Series, tsl_pct: float):
        """Update trailing stop loss"""
        if not self.position:
            return
        
        pos = self.position
        old_sl = pos.stop_loss
        
        if pos.side == 'Long':
            current_profit_pct = (candle['close'] / pos.entry_price - 1)
            if current_profit_pct > 0:
                new_sl = pos.highest_price * (1 - tsl_pct)
                if new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl
                    # Record TSL update
                    self.current_tsl_history.append((candle.name, new_sl))
        else:
            current_profit_pct = (pos.entry_price / candle['close'] - 1)
            if current_profit_pct > 0:
                new_sl = pos.lowest_price * (1 + tsl_pct)
                if new_sl < pos.stop_loss:
                    pos.stop_loss = new_sl
                    # Record TSL update
                    self.current_tsl_history.append((candle.name, new_sl))
    
    def open_position(self, candle: pd.Series, side: str, probability: float,
                     risk_pct: float, tp_pct: float, sl_pct: float):
        """Open new position"""
        if self.position:
            return
        
        position_size = self.current_capital * risk_pct
        entry_price = candle['open']
        entry_price_with_slippage = self.apply_slippage(entry_price, side, is_entry=True)
        
        if side == 'Long':
            stop_loss = entry_price_with_slippage * (1 - sl_pct)
            take_profit = entry_price_with_slippage * (1 + tp_pct) if tp_pct > 0 else np.inf
        else:
            stop_loss = entry_price_with_slippage * (1 + sl_pct)
            take_profit = entry_price_with_slippage * (1 - tp_pct) if tp_pct > 0 else 0
        
        self.position = Position(
            side=side,
            entry_price=entry_price_with_slippage,
            entry_time=candle.name,
            current_size=position_size,
            initial_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            highest_price=entry_price_with_slippage if side == 'Long' else np.inf,
            lowest_price=entry_price_with_slippage if side == 'Short' else 0,
            partial_tp_taken=False,
            tp_pct=tp_pct,
            entry_probability=probability
        )
        
        # Capture entry candle OHLCV data immediately
        self.current_trade_ohlcv.append({
            'timestamp': candle.name,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle.get('volume', 0)
        })
        
        logging.info(f"NEW POSITION: {side} @ {entry_price_with_slippage:.4f}, "
                    f"SL: {stop_loss:.4f}, TP: {take_profit:.4f}, Prob: {probability:.3f}")
    
    def run(self, df: pd.DataFrame, 
            model_long, scaler_long, features_long,
            model_short, scaler_short, features_short,
            prob_threshold: float,
            risk_pct: float,
            tp_pct: float,
            sl_pct: float,
            tsl_pct: float,
            enable_partial_tp: bool) -> Dict:
        """Main backtest loop - NO LOOK-AHEAD BIAS"""
        
        logging.info(f"Starting backtest. Candles: {len(df)}, Capital: ${self.initial_capital}")
        
        current_mae_pct = 0.0
        current_mfe_pct = 0.0
        
        for i in range(1, len(df)):
            current_candle = df.iloc[i]
            
            # STEP 1: MANAGE OPEN POSITION
            if self.position:
                # Collect OHLCV data for trade chart
                self.current_trade_ohlcv.append({
                    'timestamp': current_candle.name,
                    'open': current_candle['open'],
                    'high': current_candle['high'],
                    'low': current_candle['low'],
                    'close': current_candle['close'],
                    'volume': current_candle.get('volume', 0)
                })
                
                self.update_position_extremes(current_candle)
                
                mae, mfe = self.calculate_mae_mfe(self.position, current_candle)
                current_mae_pct = min(current_mae_pct, mae)
                current_mfe_pct = max(current_mfe_pct, mfe)
                
                # Check partial TP
                if enable_partial_tp:
                    partial_hit, partial_price = self.check_partial_tp(current_candle)
                    if partial_hit:
                        # Track partial TP hit
                        self.partial_tp_hit = True
                        self.partial_tp_time = current_candle.name
                        self.partial_tp_price = partial_price
                        
                        self.close_position(
                            exit_price=partial_price,
                            exit_time=current_candle.name,
                            exit_reason='Partial TP',
                            mae_pct=current_mae_pct,
                            mfe_pct=current_mfe_pct,
                            size_to_close=self.position.current_size / 2
                        )
                
                # Check exit conditions
                should_exit, exit_price, exit_reason = self.check_exit_conditions(current_candle)
                
                if should_exit:
                    self.close_position(
                        exit_price=exit_price,
                        exit_time=current_candle.name,
                        exit_reason=exit_reason,
                        mae_pct=current_mae_pct,
                        mfe_pct=current_mfe_pct
                    )
                    current_mae_pct = 0.0
                    current_mfe_pct = 0.0
                else:
                    self.update_trailing_stop(current_candle, tsl_pct)
            
            # STEP 2: NEW POSITION DECISION
            if not self.position:
                # CRITICAL: Use PREVIOUS closed candle for prediction
                last_closed_candle = df.iloc[[i - 1]]
                
                missing_long = set(features_long) - set(last_closed_candle.columns)
                missing_short = set(features_short) - set(last_closed_candle.columns)
                
                if missing_long or missing_short:
                    if i == 1:  # Log only once at start
                        logging.warning(f"Missing features detected - skipping predictions!")
                        if missing_long:
                            logging.warning(f"Missing LONG features ({len(missing_long)}): {list(missing_long)[:10]}")
                        if missing_short:
                            logging.warning(f"Missing SHORT features ({len(missing_short)}): {list(missing_short)[:10]}")
                        logging.info(f"Available features in data: {len(last_closed_candle.columns)}")
                    continue
                
                X_long = last_closed_candle[features_long]
                X_short = last_closed_candle[features_short]
                
                X_long_scaled = scaler_long.transform(X_long)
                X_short_scaled = scaler_short.transform(X_short)
                
                proba_long = model_long.predict_proba(X_long_scaled)[0][1]
                proba_short = model_short.predict_proba(X_short_scaled)[0][1]
                
                decision = "HOLD"
                chosen_proba = 0.0
                
                if proba_long > prob_threshold and proba_long > proba_short:
                    decision = "LONG"
                    chosen_proba = proba_long
                elif proba_short > prob_threshold and proba_short > proba_long:
                    decision = "SHORT"
                    chosen_proba = proba_short
                
                # DEBUG: Log first 10 predictions to diagnose issue
                if i <= 10:
                    logging.info(f"Candle {i} @ {current_candle.name}: proba_long={proba_long:.4f}, proba_short={proba_short:.4f}, decision={decision}")
                
                self.decision_log.append({
                    'timestamp': current_candle.name,
                    'proba_long': proba_long,
                    'proba_short': proba_short,
                    'decision': decision
                })
                
                if decision != "HOLD":
                    self.open_position(
                        candle=current_candle,
                        side=decision.capitalize(),
                        probability=chosen_proba,
                        risk_pct=risk_pct,
                        tp_pct=tp_pct,
                        sl_pct=sl_pct
                    )
        
        # Close any open position at end
        if self.position:
            last_candle = df.iloc[-1]
            self.close_position(
                exit_price=last_candle['close'],
                exit_time=last_candle.name,
                exit_reason='End of Backtest',
                mae_pct=current_mae_pct,
                mfe_pct=current_mfe_pct
            )
        
        return {
            'trades': self.trades,
            'decision_log': self.decision_log,
            'equity_curve': self.equity_curve,
            'final_capital': self.current_capital
        }


def calculate_metrics(trades: List[Trade], equity_curve: List[float], 
                     initial_capital: float) -> Dict:
    """Calculate comprehensive performance metrics"""
    if not trades:
        return {'total_trades': 0}
    
    df_trades = pd.DataFrame([asdict(t) for t in trades])
    
    total_trades = len(df_trades)
    wins = df_trades[df_trades['net_pnl_usd'] > 0]
    losses = df_trades[df_trades['net_pnl_usd'] <= 0]
    
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win = wins['net_pnl_usd'].mean() if not wins.empty else 0
    avg_loss = losses['net_pnl_usd'].mean() if not losses.empty else 0
    
    gross_profit = wins['net_pnl_usd'].sum()
    gross_loss = abs(losses['net_pnl_usd'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    total_return_pct = (equity_curve[-1] / initial_capital - 1) * 100
    
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max * 100
    max_drawdown_pct = drawdown.min()
    
    returns = equity_series.pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0
    
    downside_returns = returns[returns < 0]
    sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 and downside_returns.std() > 0 else 0
    
    recovery_factor = total_return_pct / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    
    # Calculate total PnL in USD
    total_pnl_usd = equity_curve[-1] - initial_capital
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_return_pct': total_return_pct,
        'total_pnl_usd': total_pnl_usd,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'recovery_factor': recovery_factor,
        'expectancy': expectancy,
        'total_fees': df_trades['fees_usd'].sum(),
        'avg_mae_pct': df_trades['mae_pct'].mean(),
        'avg_mfe_pct': df_trades['mfe_pct'].mean()
    }


def print_results(results: Dict, metrics: Dict, strategy_id: str):
    """Print and save backtest results"""
    
    os.makedirs("backtests", exist_ok=True)
    
    if results['trades']:
        df_trades = pd.DataFrame([asdict(t) for t in results['trades']])
        df_trades.to_csv(f"backtests/{strategy_id}_trades.csv", index=False)
    
    pd.DataFrame({'equity': results['equity_curve']}).to_csv(
        f"backtests/{strategy_id}_equity.csv", index=False)
    
    if results['decision_log']:
        pd.DataFrame(results['decision_log']).to_csv(
            f"backtests/{strategy_id}_decisions.csv", index=False)
    
    with open(f"backtests/{strategy_id}_metrics.json", 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                  for k, v in metrics.items()}, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print(f"{'BACKTEST RESULTS':^70}")
    print("="*70)
    print(f"\nStrategy: {strategy_id}")
    print(f"\nPERFORMANCE:")
    print(f"  Total PnL:        ${metrics.get('total_pnl_usd', 0):+.2f}")
    print(f"  Total Return:     {metrics.get('total_return_pct', 0):.2f}%")
    print(f"  Max Drawdown:     {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Sortino Ratio:    {metrics.get('sortino_ratio', 0):.3f}")
    print(f"\nTRADES:")
    print(f"  Total:            {metrics.get('total_trades', 0)}")
    print(f"  Win Rate:         {metrics.get('win_rate', 0):.2f}%")
    print(f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
    print(f"  Expectancy:       ${metrics.get('expectancy', 0):.2f}")
    print(f"\nFEES:")
    print(f"  Total:            ${metrics.get('total_fees', 0):.2f}")
    print("="*70 + "\n")


def _get_strategy_id(ticker, timeframe, helper_timeframes):
    helpers = '_plus_' + '_'.join(helper_timeframes) if helper_timeframes else ""
    return f"{ticker}_{timeframe.replace(' ', '')}{helpers}_long_short_combined"


def main(args):
    base_id = _get_strategy_id(args.ticker, args.timeframe, args.helper_timeframes)
    long_id = base_id.replace('_long_short_combined', '_long')
    short_id = base_id.replace('_long_short_combined', '_short')
    
    try:
        model_long = joblib.load(f"models/{long_id}_model.joblib")
        scaler_long = joblib.load(f"models/{long_id}_scaler.joblib")
        features_long = joblib.load(f"models/{long_id}_features.joblib")
        
        model_short = joblib.load(f"models/{short_id}_model.joblib")
        scaler_short = joblib.load(f"models/{short_id}_scaler.joblib")
        features_short = joblib.load(f"models/{short_id}_features.joblib")
    except FileNotFoundError as e:
        logging.error(f"Model files not found: {e}")
        return
    
    df = fetch_and_prepare_data(
        ticker=args.ticker,
        timeframe=args.timeframe,
        limit=args.limit,
        helper_timeframes=args.helper_timeframes,
        side='backtest'
    )
    
    if df.empty:
        logging.error("Failed to prepare data")
        return
    
    engine = BacktestEngine(
        initial_capital=args.initial_capital,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        slippage_pct=args.slippage_pct
    )
    
    results = engine.run(
        df=df,
        model_long=model_long,
        scaler_long=scaler_long,
        features_long=features_long,
        model_short=model_short,
        scaler_short=scaler_short,
        features_short=features_short,
        prob_threshold=args.prob_threshold,
        risk_pct=args.risk_pct,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        tsl_pct=args.tsl_pct,
        enable_partial_tp=args.partial_tp
    )
    
    metrics = calculate_metrics(results['trades'], results['equity_curve'], args.initial_capital)
    metrics['initial_capital'] = args.initial_capital
    
    print_results(results, metrics, base_id)


def run_backtester_with_args(args):
    """
    Wrapper function to run backtester with args from main.py
    
    Args:
        args: Argument namespace from main.py with backtest parameters
    """
    # Add default values for parameters not passed from main.py
    if not hasattr(args, 'initial_capital'):
        args.initial_capital = 10000.0
    if not hasattr(args, 'risk_pct'):
        args.risk_pct = 0.02
    if not hasattr(args, 'sl_pct'):
        args.sl_pct = args.tp_pct * 0.5  # Default SL at 50% of TP
    if not hasattr(args, 'maker_fee'):
        args.maker_fee = 0.0002
    if not hasattr(args, 'taker_fee'):
        args.taker_fee = 0.00055
    if not hasattr(args, 'slippage_pct'):
        args.slippage_pct = 0.0001
    
    # Run the main backtest function
    main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--timeframe', required=True)
    parser.add_argument('--helper-timeframes', nargs='*', default=None)
    parser.add_argument('--limit', type=int, default=10000)
    parser.add_argument('--initial-capital', type=float, default=10000.0)
    parser.add_argument('--risk-pct', type=float, default=0.02)
    parser.add_argument('--tp-pct', type=float, required=True)
    parser.add_argument('--sl-pct', type=float, required=True)
    parser.add_argument('--tsl-pct', type=float, required=True)
    parser.add_argument('--prob-threshold', type=float, required=True)
    parser.add_argument('--partial-tp', action='store_true')
    parser.add_argument('--maker-fee', type=float, default=0.0002)
    parser.add_argument('--taker-fee', type=float, default=0.00055)
    parser.add_argument('--slippage-pct', type=float, default=0.0001)
    
    main(parser.parse_args())
