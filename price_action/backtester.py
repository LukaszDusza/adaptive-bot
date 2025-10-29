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
    dynamic_tp_levels_taken: int = 0  # Tracks how many of 4 dynamic TP levels have been taken


@dataclass
class PendingLimitOrder:
    """Data structure for pending limit order"""
    side: str  # 'Long' or 'Short'
    limit_price: float
    size: float
    placed_time: pd.Timestamp
    placed_candle_index: int
    probability: float
    tp_pct: float
    sl_pct: float


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
    # Trade categorization for analysis
    entry_hour: int = 0  # Hour of day (0-23)
    exit_hour: int = 0   # Hour of day (0-23)
    entry_day_of_week: int = 0  # Monday=0, Sunday=6
    exit_day_of_week: int = 0


class BacktestEngine:
    """Backtest engine without look-ahead bias"""

    def __init__(self,
                 initial_capital: float = 10000.0,
                 maker_fee: float = 0.0002,
                 taker_fee: float = 0.00055,
                 slippage_pct: float = 0.0003,
                 enable_limit_order: bool = False,
                 limit_order_candles: int = 5,
                 limit_offset_pct: float = 0.005,
                 timeframe: str = '15m',
                 suppress_limit_logs: bool = False,
                 cooldown_minutes: int = 30,
                 warmup_candles: int = 200):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_pct = slippage_pct

        # Limit order settings
        self.enable_limit_order = enable_limit_order
        self.limit_order_candles = limit_order_candles
        self.limit_offset_pct = limit_offset_pct
        self.timeframe = timeframe
        self.suppress_limit_logs = suppress_limit_logs

        # Cooldown settings (matches live bot behavior)
        self.cooldown_minutes = cooldown_minutes
        self.last_position_closed_at: Optional[pd.Timestamp] = None

        # Warmup period (skip first N candles for indicator stability)
        self.warmup_candles = warmup_candles

        self.position: Optional[Position] = None
        self.pending_limit_order: Optional[PendingLimitOrder] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.decision_log: List[Dict] = []

        # Track current trade data
        self.current_trade_ohlcv: List[Dict] = []
        self.current_tsl_history: List[Tuple] = []
        self.partial_tp_hit: bool = False
        self.partial_tp_time: Optional[pd.Timestamp] = None
        self.partial_tp_price: Optional[float] = None

        # Limit order statistics
        self.limit_orders_filled = 0
        self.limit_orders_cancelled = 0
        
    def calculate_fees(self, price: float, size: float, is_maker: bool = False) -> float:
        """Calculate transaction fees in USD"""
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return size * fee_rate  # Fee in USD (size is already in USD)
    
    def apply_slippage(self, price: float, side: str, is_entry: bool = True,
                      volatility_multiplier: float = 1.0) -> float:
        """
        Apply slippage to price with volatility adjustment.

        Args:
            price: Base price
            side: 'Long' or 'Short'
            is_entry: True for entry, False for exit
            volatility_multiplier: Multiplier for slippage (1.0 = normal, >1.0 = high volatility)
        """
        adjusted_slippage = self.slippage_pct * volatility_multiplier

        if is_entry:
            if side == 'Long':
                return price * (1 + adjusted_slippage)
            else:
                return price * (1 - adjusted_slippage)
        else:
            if side == 'Long':
                return price * (1 - adjusted_slippage)
            else:
                return price * (1 + adjusted_slippage)
    
    def calculate_volatility_multiplier(self, candle: pd.Series) -> float:
        """
        Calculate volatility multiplier for slippage adjustment.
        Uses ATR normalized if available, otherwise returns 1.0.

        Returns:
            Multiplier between 0.5 and 2.0
        """
        atr_normalized = candle.get('atr_normalized', 1.0)

        # ATR normalized:
        # - 1.0 = normal volatility → multiplier 1.0
        # - > 1.5 = high volatility → multiplier up to 2.0
        # - < 0.5 = low volatility → multiplier down to 0.5
        if atr_normalized > 1.5:
            multiplier = min(1.0 + (atr_normalized - 1.0) * 0.5, 2.0)
        elif atr_normalized < 0.5:
            multiplier = max(0.5, atr_normalized)
        else:
            multiplier = 1.0

        return multiplier

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
    
    def check_dynamic_tp(self, candle: pd.Series) -> Tuple[bool, Optional[float], Optional[int]]:
        """Check if next dynamic TP level reached (4 levels: 25%, 50%, 75%, 100%)"""
        if not self.position:
            return False, None, None
        
        pos = self.position
        
        if pos.take_profit == np.inf or pos.take_profit == 0:
            return False, None, None
        
        # Check if all 4 levels already taken
        if pos.dynamic_tp_levels_taken >= 4:
            return False, None, None
        
        # Calculate distance from entry to TP
        if pos.side == 'Long':
            distance = pos.take_profit - pos.entry_price
        else:
            distance = pos.entry_price - pos.take_profit
        
        # Calculate the next level to check (1-based: 1, 2, 3, 4)
        next_level = pos.dynamic_tp_levels_taken + 1
        
        # Calculate the price for this level
        level_pct = next_level * 0.25  # 0.25, 0.50, 0.75, 1.0
        
        if pos.side == 'Long':
            level_price = pos.entry_price + (distance * level_pct)
            hit = candle['high'] >= level_price
        else:
            level_price = pos.entry_price - (distance * level_pct)
            hit = candle['low'] <= level_price
        
        if hit:
            return True, level_price, next_level
        
        return False, None, None
    
    def close_position(self, exit_price: float, exit_time: pd.Timestamp,
                       exit_reason: str, mae_pct: float, mfe_pct: float,
                       size_to_close: Optional[float] = None,
                       candle: Optional[pd.Series] = None):
        """Close position (fully or partially)"""
        if not self.position:
            return

        pos = self.position
        close_size = size_to_close if size_to_close else pos.current_size

        # Track position close time for cooldown (only for full closes)
        is_full_close = size_to_close is None or close_size >= pos.current_size
        if is_full_close:
            self.last_position_closed_at = exit_time

        # Calculate volatility multiplier for slippage adjustment
        volatility_multiplier = 1.0
        if candle is not None:
            volatility_multiplier = self.calculate_volatility_multiplier(candle)

        # Apply slippage with volatility adjustment
        exit_price_with_slippage = self.apply_slippage(
            exit_price, pos.side, is_entry=False, volatility_multiplier=volatility_multiplier
        )
        
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
            chart_ohlcv=chart_ohlcv_json,
            # Trade categorization
            entry_hour=pos.entry_time.hour,
            exit_hour=exit_time.hour,
            entry_day_of_week=pos.entry_time.dayofweek,
            exit_day_of_week=exit_time.dayofweek
        )
        
        self.trades.append(trade)
        self.current_capital += net_pnl_usd
        self.equity_curve.append(self.current_capital)
        
        # Update position or close
        if size_to_close:
            pos.current_size -= close_size
            pos.partial_tp_taken = True
            
            # Check if position is fully closed after partial close
            if pos.current_size <= 0:
                # Reset tracking for next trade
                self.current_trade_ohlcv = []
                self.current_tsl_history = []
                self.partial_tp_hit = False
                self.partial_tp_time = None
                self.partial_tp_price = None
                self.position = None
                logging.info(f"Position fully closed via partial closes: {exit_reason}, Net P&L: ${net_pnl_usd:.2f}")
            else:
                # Move SL to breakeven if it was below breakeven
                if pos.side == 'Long':
                    if pos.stop_loss < pos.entry_price:
                        pos.stop_loss = pos.entry_price
                        logging.info(f"TSL moved to breakeven after partial TP: {pos.entry_price:.4f}")
                else:  # Short
                    if pos.stop_loss > pos.entry_price:
                        pos.stop_loss = pos.entry_price
                        logging.info(f"TSL moved to breakeven after partial TP: {pos.entry_price:.4f}")
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
    
    def is_in_cooldown_period(self, current_time: pd.Timestamp) -> bool:
        """
        Check if backtest is in cooldown period after position close.
        Matches live bot behavior (30 min cooldown = 2x 15m candles).

        Args:
            current_time: Current candle timestamp

        Returns:
            True if cooldown is active, False otherwise
        """
        if self.last_position_closed_at is None:
            return False

        if self.cooldown_minutes <= 0:
            return False

        elapsed_minutes = (current_time - self.last_position_closed_at).total_seconds() / 60

        if elapsed_minutes < self.cooldown_minutes:
            return True

        return False

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

    def place_limit_order(self, candle: pd.Series, candle_index: int, side: str,
                         probability: float, risk_pct: float, tp_pct: float, sl_pct: float):
        """Place a limit order (like bot.py)"""
        if self.pending_limit_order or self.position:
            return

        current_price = candle['close']  # Use close price for reference
        position_size = self.current_capital * risk_pct

        # Calculate limit price with offset (matches bot.py logic)
        if side == 'Long':
            # LONG: Buy cheaper (limit below current price)
            limit_price = current_price * (1 - self.limit_offset_pct)
        else:
            # SHORT: Sell higher (limit above current price)
            limit_price = current_price * (1 + self.limit_offset_pct)

        self.pending_limit_order = PendingLimitOrder(
            side=side,
            limit_price=limit_price,
            size=position_size,
            placed_time=candle.name,
            placed_candle_index=candle_index,
            probability=probability,
            tp_pct=tp_pct,
            sl_pct=sl_pct
        )

        if not self.suppress_limit_logs:
            logging.info(f"📋 LIMIT ORDER PLACED: {side} @ {limit_price:.4f} | "
                        f"Current: {current_price:.4f} | Offset: {self.limit_offset_pct*100:.2f}% | "
                        f"Max wait: {self.limit_order_candles} candles")

    def check_limit_order_fill(self, candle: pd.Series, candle_index: int) -> bool:
        """
        Check if pending limit order should be filled (matches bot.py logic).
        Returns True if filled, False otherwise.
        """
        if not self.pending_limit_order:
            return False

        order = self.pending_limit_order

        # Check if price reached limit (using candle's high/low for accuracy)
        filled = False
        if order.side == 'Long':
            # LONG: filled if candle low <= limit_price
            filled = candle['low'] <= order.limit_price
        else:
            # SHORT: filled if candle high >= limit_price
            filled = candle['high'] >= order.limit_price

        if filled:
            self.fill_limit_order(candle)
            return True

        # Check timeout (based on candles elapsed)
        candles_elapsed = candle_index - order.placed_candle_index
        if candles_elapsed >= self.limit_order_candles:
            self.cancel_limit_order(candle)
            return False

        return False

    def fill_limit_order(self, candle: pd.Series):
        """Execute limit order fill (matches bot.py entry logic)"""
        if not self.pending_limit_order:
            return

        order = self.pending_limit_order

        # Use limit price as entry (no additional slippage - already maker order)
        entry_price = order.limit_price

        # Calculate SL and TP
        if order.side == 'Long':
            stop_loss = entry_price * (1 - order.sl_pct)
            take_profit = entry_price * (1 + order.tp_pct) if order.tp_pct > 0 else np.inf
        else:
            stop_loss = entry_price * (1 + order.sl_pct)
            take_profit = entry_price * (1 - order.tp_pct) if order.tp_pct > 0 else 0

        self.position = Position(
            side=order.side,
            entry_price=entry_price,
            entry_time=candle.name,
            current_size=order.size,
            initial_size=order.size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            highest_price=entry_price if order.side == 'Long' else np.inf,
            lowest_price=entry_price if order.side == 'Short' else 0,
            partial_tp_taken=False,
            tp_pct=order.tp_pct,
            entry_probability=order.probability
        )

        # Capture entry candle OHLCV data
        self.current_trade_ohlcv.append({
            'timestamp': candle.name,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle.get('volume', 0)
        })

        self.limit_orders_filled += 1
        tp_display = f"{take_profit:.4f}" if take_profit not in [0, np.inf] else "OFF"

        if not self.suppress_limit_logs:
            logging.warning(f"✅ LIMIT ORDER FILLED: {order.side} @ {entry_price:.4f} | "
                           f"SL: {stop_loss:.4f} | TP: {tp_display} | "
                           f"Prob: {order.probability:.3f}")

        # Clear pending order
        self.pending_limit_order = None

    def cancel_limit_order(self, candle: pd.Series):
        """Cancel limit order due to timeout"""
        if not self.pending_limit_order:
            return

        order = self.pending_limit_order
        self.limit_orders_cancelled += 1

        if not self.suppress_limit_logs:
            logging.warning(f"⏰ LIMIT ORDER CANCELLED: {order.side} @ {order.limit_price:.4f} | "
                           f"Timeout: {self.limit_order_candles} candles elapsed")

        # Clear pending order
        self.pending_limit_order = None

    def open_position(self, candle: pd.Series, side: str, probability: float,
                     risk_pct: float, tp_pct: float, sl_pct: float):
        """Open new position (uses close price to match live bot market order execution)"""
        if self.position:
            return

        position_size = self.current_capital * risk_pct
        entry_price = candle['close']  # Use close to simulate market order execution

        # Calculate volatility multiplier for slippage adjustment
        volatility_multiplier = self.calculate_volatility_multiplier(candle)
        entry_price_with_slippage = self.apply_slippage(
            entry_price, side, is_entry=True, volatility_multiplier=volatility_multiplier
        )
        
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
            enable_partial_tp: bool,
            enable_dynamic_tp: bool = False,
            enable_profit_protection: bool = False,
            min_proba_diff: float = 0.0,
            ict_context_mode: bool = False,
            ict_min_strength: float = 0.3,
            ict_max_reduction: float = 0.20,
            regime_sensitivity: float = 1.0) -> Dict:
        """Main backtest loop - NO LOOK-AHEAD BIAS"""

        # Validate mutual exclusivity
        if enable_partial_tp and enable_dynamic_tp:
            raise ValueError("Cannot enable both --partial-tp and --dynamic-tp. Choose only one.")
        
        logging.info(f"Starting backtest. Candles: {len(df)}, Capital: ${self.initial_capital}")
        if enable_dynamic_tp:
            logging.info("Dynamic TP enabled: 4 levels at 25%, 50%, 75%, 100% from BE to TP")
        elif enable_partial_tp:
            logging.info("Partial TP enabled: 50% at halfway from BE to TP")
        if enable_profit_protection:
            logging.info("Profit Protection enabled: move SL to BE if profit peaks >0.25% then declines")

        current_mae_pct = 0.0
        current_mfe_pct = 0.0
        highest_profit_pct = 0.0  # Track peak profit for profit protection

        # Log limit order mode if enabled
        if self.enable_limit_order:
            logging.info(f"Limit Order Mode ENABLED: offset={self.limit_offset_pct*100:.2f}%, max_wait={self.limit_order_candles} candles")

        # Apply warmup period (skip first N candles for indicator stability)
        start_index = max(1, self.warmup_candles)
        if self.warmup_candles > 0:
            logging.info(f"Warmup period: skipping first {self.warmup_candles} candles for indicator stability")
            logging.info(f"Trading will start from candle {start_index} / {len(df)}")

        for i in range(start_index, len(df)):
            current_candle = df.iloc[i]

            # STEP 0.5: CHECK PENDING LIMIT ORDERS
            if self.pending_limit_order:
                self.check_limit_order_fill(current_candle, i)

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

                # ========== PROFIT PROTECTION ==========
                if enable_profit_protection:
                    # Only activate if partial TP not yet taken
                    partial_tp_not_taken = (enable_partial_tp and not self.position.partial_tp_taken) or \
                                          (enable_dynamic_tp and self.position.dynamic_tp_levels_taken == 0) or \
                                          (not enable_partial_tp and not enable_dynamic_tp)

                    if partial_tp_not_taken:
                        # Calculate current profit percentage
                        if self.position.side == 'Long':
                            current_profit_pct = (current_candle['close'] / self.position.entry_price - 1) * 100
                        else:
                            current_profit_pct = (self.position.entry_price / current_candle['close'] - 1) * 100

                        # Update highest profit if current is higher
                        highest_profit_pct = max(highest_profit_pct, current_profit_pct)

                        # Protection thresholds
                        PROFIT_THRESHOLD = 0.25  # Minimum profit to activate protection (0.25%)
                        PROFIT_DECLINE_TRIGGER = 0.25  # If profit drops to this level, move SL to BE

                        # If profit peaked above threshold but now declined to trigger level
                        if (highest_profit_pct > PROFIT_THRESHOLD and
                            current_profit_pct <= PROFIT_DECLINE_TRIGGER):

                            # Move SL to breakeven if not already there
                            if self.position.side == 'Long':
                                if self.position.stop_loss < self.position.entry_price:
                                    logging.warning(
                                        f"🛡️ PROFIT PROTECTION @ {current_candle.name}: "
                                        f"Moving SL to BE (profit peaked at {highest_profit_pct:.2f}%, "
                                        f"now at {current_profit_pct:.2f}%)"
                                    )
                                    self.position.stop_loss = self.position.entry_price
                            else:  # Short
                                if self.position.stop_loss > self.position.entry_price:
                                    logging.warning(
                                        f"🛡️ PROFIT PROTECTION @ {current_candle.name}: "
                                        f"Moving SL to BE (profit peaked at {highest_profit_pct:.2f}%, "
                                        f"now at {current_profit_pct:.2f}%)"
                                    )
                                    self.position.stop_loss = self.position.entry_price

                # Check partial TP (old mechanism)
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
                            size_to_close=self.position.current_size / 2,
                            candle=current_candle
                        )
                
                # Check dynamic TP (new mechanism)
                if enable_dynamic_tp:
                    dynamic_hit, dynamic_price, level = self.check_dynamic_tp(current_candle)
                    if dynamic_hit:
                        # Track dynamic TP hit
                        self.partial_tp_hit = True
                        self.partial_tp_time = current_candle.name
                        self.partial_tp_price = dynamic_price

                        # CRITICAL FIX: Match live bot behavior
                        # Level 4 closes ALL remaining to avoid rounding error accumulation
                        if level == 4:
                            size_to_close = self.position.current_size
                            logging.info(f"Dynamic TP Level 4: closing ALL remaining {size_to_close:.4f}")
                        else:
                            # Levels 1-3: close 25% of initial with rounding (matches live bot)
                            size_to_close = round(self.position.initial_size * 0.25, 2)

                            # Validate: don't close more than current size
                            if size_to_close > self.position.current_size:
                                size_to_close = self.position.current_size
                                logging.info(f"Dynamic TP L{level}: adjusted qty to current size {size_to_close:.4f}")

                        self.close_position(
                            exit_price=dynamic_price,
                            exit_time=current_candle.name,
                            exit_reason=f'Dynamic TP Level {level} ({level*25}%)',
                            mae_pct=current_mae_pct,
                            mfe_pct=current_mfe_pct,
                            size_to_close=size_to_close,
                            candle=current_candle
                        )
                        
                        # Increment the level counter and move SL to BE after first level
                        if self.position:  # Position still exists (partial close)
                            self.position.dynamic_tp_levels_taken = level
                            
                            # Move SL to breakeven only after first level
                            if level == 1:
                                if self.position.side == 'Long':
                                    if self.position.stop_loss < self.position.entry_price:
                                        self.position.stop_loss = self.position.entry_price
                                        logging.info(f"SL moved to breakeven after Dynamic TP Level 1: {self.position.entry_price:.4f}")
                                else:  # Short
                                    if self.position.stop_loss > self.position.entry_price:
                                        self.position.stop_loss = self.position.entry_price
                                        logging.info(f"SL moved to breakeven after Dynamic TP Level 1: {self.position.entry_price:.4f}")
                
                # Check exit conditions
                should_exit, exit_price, exit_reason = self.check_exit_conditions(current_candle)
                
                if should_exit:
                    self.close_position(
                        exit_price=exit_price,
                        exit_time=current_candle.name,
                        exit_reason=exit_reason,
                        mae_pct=current_mae_pct,
                        mfe_pct=current_mfe_pct,
                        candle=current_candle
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
                
                # Calculate probability difference (confidence gap)
                proba_diff = abs(proba_long - proba_short)

                # ========== ICT CONTEXT MODE (OPCJA 3) - Backtest Version ==========
                # Dynamic threshold adjustment based on Smart Money signals

                if ict_context_mode:
                    # Extract ICT features from current candle
                    ict_composite = current_candle.get('ict_composite_score', 0.5)
                    liquidity_sweep = current_candle.get('liquidity_sweep', 0.0)
                    fvg_signal = current_candle.get('fvg_signal', 0.0)
                    order_block = current_candle.get('order_block', 0.0)
                    market_structure_shift = current_candle.get('market_structure_shift', 0.0)

                    # Extract market regime features
                    rsi = current_candle.get('rsi_14', 50)
                    atr_norm = current_candle.get('atr_normalized', 1.0)
                    volume_ratio = current_candle.get('volume_vs_ma_20', 1.0)

                    # 1. Calculate ICT Strength (0-1 scale)
                    ict_strength = 0.0
                    if ict_composite > 0.6:
                        ict_strength += 0.3
                    if liquidity_sweep > 0:
                        ict_strength += 0.2
                    if abs(fvg_signal) > 0:
                        ict_strength += 0.2
                    if abs(order_block) > 0:
                        ict_strength += 0.2
                    if abs(market_structure_shift) > 0:
                        ict_strength += 0.1
                    ict_strength = min(ict_strength, 1.0)

                    # 2. Calculate Market Regime Multiplier
                    # FIX: Odwrócona logika - niska zmienność NIE PODNOSI progów
                    regime_multiplier = 1.0

                    # Trending market (easier to trade) → lower thresholds
                    if (rsi > 60 and proba_long > proba_short) or (rsi < 40 and proba_short > proba_long):
                        regime_multiplier *= (1.0 - 0.1 * regime_sensitivity)  # -10% threshold

                    # High volatility with volume confirmation → lower thresholds
                    if atr_norm > 1.2 and volume_ratio > 1.3:
                        regime_multiplier *= (1.0 - 0.15 * regime_sensitivity)  # -15% threshold

                    # Low volatility OR low volume → LEKKO OBNIŻ (więcej okazji w spokojnym rynku)
                    # FIX: Było 1.0 + 0.15 (podnosiło), teraz 1.0 - 0.05 (lekko obniża)
                    if atr_norm < 0.7 or volume_ratio < 0.6:
                        regime_multiplier *= (1.0 - 0.05 * regime_sensitivity)  # -5% threshold (neutral/slightly lower)

                    # Extreme RSI without volume → LEKKO PODNOSI (ale mniej niż wcześniej)
                    # FIX: Było 1.0 + 0.25 (mocno podnosiło), teraz 1.0 + 0.10 (lekko podnosi)
                    if (rsi > 75 or rsi < 25) and volume_ratio < 0.8:
                        regime_multiplier *= (1.0 + 0.10 * regime_sensitivity)  # +10% threshold (was +25%)

                    # 3. Dynamic Threshold Adjustment
                    base_prob_threshold = prob_threshold
                    base_diff_threshold = min_proba_diff

                    if ict_strength >= ict_min_strength:
                        ict_adjustment = 1.0 - (ict_strength * ict_max_reduction)
                    else:
                        ict_adjustment = 1.0

                    dynamic_prob_threshold = base_prob_threshold * regime_multiplier * ict_adjustment
                    dynamic_diff_threshold = base_diff_threshold * regime_multiplier * ict_adjustment

                    # Safety bounds
                    dynamic_prob_threshold = np.clip(dynamic_prob_threshold, 0.45, 0.85)
                    dynamic_diff_threshold = np.clip(dynamic_diff_threshold, 0.08, 0.35)
                else:
                    # Use static thresholds (original behavior)
                    dynamic_prob_threshold = prob_threshold
                    dynamic_diff_threshold = min_proba_diff
                    ict_strength = 0.0
                    regime_multiplier = 1.0

                # Determine decision with dynamic thresholds
                decision = "HOLD"
                chosen_proba = 0.0

                if proba_long > dynamic_prob_threshold and proba_long > proba_short:
                    # Check if confidence gap is sufficient
                    if proba_diff >= dynamic_diff_threshold:
                        decision = "LONG"
                        chosen_proba = proba_long
                elif proba_short > dynamic_prob_threshold and proba_short > proba_long:
                    # Check if confidence gap is sufficient
                    if proba_diff >= dynamic_diff_threshold:
                        decision = "SHORT"
                        chosen_proba = proba_short

                # DEBUG: Log first 10 predictions to diagnose issue
                if i <= 10:
                    if ict_context_mode:
                        logging.info(f"Candle {i} @ {current_candle.name}: proba_long={proba_long:.4f}, proba_short={proba_short:.4f}, "
                                   f"ICT_str={ict_strength:.2f}, threshold={dynamic_prob_threshold:.3f}, decision={decision}")
                    else:
                        logging.info(f"Candle {i} @ {current_candle.name}: proba_long={proba_long:.4f}, proba_short={proba_short:.4f}, "
                                   f"diff={proba_diff:.4f}, decision={decision}")

                # Extended decision log with ICT context info
                decision_entry = {
                    'timestamp': current_candle.name,
                    'proba_long': proba_long,
                    'proba_short': proba_short,
                    'proba_diff': proba_diff,
                    'decision': decision,
                    'base_prob_threshold': prob_threshold,
                    'base_diff_threshold': min_proba_diff,
                }

                if ict_context_mode:
                    decision_entry.update({
                        'ict_mode': True,
                        'ict_strength': ict_strength,
                        'regime_multiplier': regime_multiplier,
                        'dynamic_prob_threshold': dynamic_prob_threshold,
                        'dynamic_diff_threshold': dynamic_diff_threshold,
                        'threshold_reduction_pct': (prob_threshold - dynamic_prob_threshold) / prob_threshold * 100
                    })
                else:
                    decision_entry['ict_mode'] = False

                self.decision_log.append(decision_entry)

                if decision != "HOLD" and not self.pending_limit_order:
                    # COOLDOWN CHECK: Matches live bot behavior
                    # Prevents immediate re-entry after position close
                    if self.is_in_cooldown_period(current_candle.name):
                        elapsed_minutes = (current_candle.name - self.last_position_closed_at).total_seconds() / 60
                        remaining_minutes = self.cooldown_minutes - elapsed_minutes
                        logging.info(f"⏰ COOLDOWN: Skipping {decision} signal - {remaining_minutes:.1f} min remaining "
                                    f"(last close: {self.last_position_closed_at})")
                        decision_entry['cooldown_blocked'] = True
                        decision_entry['cooldown_remaining_minutes'] = remaining_minutes
                    else:
                        # Use limit order or market order depending on mode
                        if self.enable_limit_order:
                            self.place_limit_order(
                                candle=current_candle,
                                candle_index=i,
                                side=decision.capitalize(),
                                probability=chosen_proba,
                                risk_pct=risk_pct,
                                tp_pct=tp_pct,
                                sl_pct=sl_pct
                            )
                        else:
                            self.open_position(
                                candle=current_candle,
                                side=decision.capitalize(),
                                probability=chosen_proba,
                                risk_pct=risk_pct,
                                tp_pct=tp_pct,
                                sl_pct=sl_pct
                            )
                        # Reset profit protection tracking for new position
                        highest_profit_pct = 0.0
        
        # Close any open position at end
        if self.position:
            last_candle = df.iloc[-1]
            self.close_position(
                exit_price=last_candle['close'],
                exit_time=last_candle.name,
                exit_reason='End of Backtest',
                mae_pct=current_mae_pct,
                mfe_pct=current_mfe_pct,
                candle=last_candle
            )

        # Log limit order statistics if enabled
        if self.enable_limit_order:
            logging.info(f"Limit Order Stats: {self.limit_orders_filled} filled, {self.limit_orders_cancelled} cancelled")

        return {
            'trades': self.trades,
            'decision_log': self.decision_log,
            'equity_curve': self.equity_curve,
            'final_capital': self.current_capital,
            'limit_orders_filled': self.limit_orders_filled,
            'limit_orders_cancelled': self.limit_orders_cancelled
        }


def calculate_metrics(trades: List[Trade], equity_curve: List[float],
                     initial_capital: float) -> Dict:
    """Calculate comprehensive performance metrics"""
    if not trades:
        # Return full dict with zeros/defaults when no trades
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'total_return_pct': 0,
            'total_pnl_usd': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'recovery_factor': 0,
            'expectancy': 0
        }
    
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

    # Extended statistics: consecutive wins/losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_streak = 0
    for _, trade in df_trades.iterrows():
        if trade['net_pnl_usd'] > 0:
            if current_streak >= 0:
                current_streak += 1
            else:
                current_streak = 1
            max_consecutive_wins = max(max_consecutive_wins, current_streak)
        else:
            if current_streak <= 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))

    # Extended statistics: best/worst trade
    best_trade = df_trades['net_pnl_usd'].max() if not df_trades.empty else 0
    worst_trade = df_trades['net_pnl_usd'].min() if not df_trades.empty else 0

    # Extended statistics: average trade duration
    avg_trade_duration_hours = df_trades['duration'].apply(lambda x: x.total_seconds() / 3600).mean() if not df_trades.empty else 0

    # Extended statistics: win rate by side
    long_trades = df_trades[df_trades['side'] == 'Long']
    short_trades = df_trades[df_trades['side'] == 'Short']

    long_wins = long_trades[long_trades['net_pnl_usd'] > 0]
    short_wins = short_trades[short_trades['net_pnl_usd'] > 0]

    win_rate_long = len(long_wins) / len(long_trades) * 100 if len(long_trades) > 0 else 0
    win_rate_short = len(short_wins) / len(short_trades) * 100 if len(short_trades) > 0 else 0

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
        'avg_mfe_pct': df_trades['mfe_pct'].mean(),
        # Extended statistics
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'avg_trade_duration_hours': avg_trade_duration_hours,
        'win_rate_long': win_rate_long,
        'win_rate_short': win_rate_short,
        'total_long_trades': len(long_trades),
        'total_short_trades': len(short_trades)
    }


def print_results(results: Dict, metrics: Dict, strategy_id: str, version: str = 'v1.0'):
    """Print and save backtest results"""
    
    backtest_dir = os.path.join("models", version, "backtests")
    os.makedirs(backtest_dir, exist_ok=True)
    
    if results['trades']:
        df_trades = pd.DataFrame([asdict(t) for t in results['trades']])
        df_trades.to_csv(f"{backtest_dir}/{strategy_id}_trades.csv", index=False)
    
    pd.DataFrame({'equity': results['equity_curve']}).to_csv(
        f"{backtest_dir}/{strategy_id}_equity.csv", index=False)
    
    if results['decision_log']:
        pd.DataFrame(results['decision_log']).to_csv(
            f"{backtest_dir}/{strategy_id}_decisions.csv", index=False)
    
    with open(f"{backtest_dir}/{strategy_id}_metrics.json", 'w') as f:
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
    print(f"  Best Trade:       ${metrics.get('best_trade', 0):+.2f}")
    print(f"  Worst Trade:      ${metrics.get('worst_trade', 0):+.2f}")
    print(f"  Avg Duration:     {metrics.get('avg_trade_duration_hours', 0):.2f} hours")
    print(f"  Max Win Streak:   {metrics.get('max_consecutive_wins', 0)}")
    print(f"  Max Loss Streak:  {metrics.get('max_consecutive_losses', 0)}")
    print(f"\nTRADES BY SIDE:")
    print(f"  LONG:  {metrics.get('total_long_trades', 0)} trades, {metrics.get('win_rate_long', 0):.2f}% WR")
    print(f"  SHORT: {metrics.get('total_short_trades', 0)} trades, {metrics.get('win_rate_short', 0):.2f}% WR")
    print(f"\nFEES:")
    print(f"  Total:            ${metrics.get('total_fees', 0):.2f}")

    # Display limit order stats if they exist
    limit_filled = results.get('limit_orders_filled', 0)
    limit_cancelled = results.get('limit_orders_cancelled', 0)
    if limit_filled > 0 or limit_cancelled > 0:
        print(f"\nLIMIT ORDERS:")
        print(f"  Filled:           {limit_filled}")
        print(f"  Cancelled:        {limit_cancelled}")
        if limit_filled + limit_cancelled > 0:
            fill_rate = (limit_filled / (limit_filled + limit_cancelled)) * 100
            print(f"  Fill Rate:        {fill_rate:.2f}%")

    print("="*70 + "\n")


def _get_strategy_id(ticker, timeframe, helper_timeframes):
    helpers = '_plus_' + '_'.join(helper_timeframes) if helper_timeframes else ""
    return f"{ticker}_{timeframe.replace(' ', '')}{helpers}_long_short_combined"


def validate_date_range(backtest_df: pd.DataFrame, training_metadata_path: str, version: str):
    """
    Validate that backtest period doesn't overlap with training period.
    Prevents data leakage from testing on training data.

    Args:
        backtest_df: DataFrame with backtest data
        training_metadata_path: Path to training_metadata.json
        version: Model version

    Raises:
        ValueError: If backtest overlaps with training period
    """
    metadata_file = f"{training_metadata_path}/training_metadata.json"

    if not os.path.exists(metadata_file):
        logging.warning(f"Training metadata not found: {metadata_file}. Skipping date validation.")
        return

    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Extract training end date
        training_end_str = metadata.get('training_end_date')
        if not training_end_str:
            logging.warning("training_end_date not found in metadata. Skipping validation.")
            return

        # Parse training end date (handle various formats)
        try:
            training_end = pd.to_datetime(training_end_str)
        except:
            logging.warning(f"Could not parse training_end_date: {training_end_str}. Skipping validation.")
            return

        # Get backtest start date
        backtest_start = backtest_df.index[0]

        # Check for overlap
        if backtest_start <= training_end:
            overlap_days = (training_end - backtest_start).days
            raise ValueError(
                f"\n{'='*70}\n"
                f"⚠️  DATA LEAKAGE WARNING ⚠️\n"
                f"{'='*70}\n"
                f"Backtest period OVERLAPS with training period!\n"
                f"\n"
                f"Training ended:    {training_end.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Backtest starts:   {backtest_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Overlap:           {overlap_days} days\n"
                f"\n"
                f"This invalidates backtest results (data leakage).\n"
                f"Use --date-from parameter to start backtest AFTER training end date.\n"
                f"{'='*70}\n"
            )

        # Log success
        gap_days = (backtest_start - training_end).days
        logging.info(f"✓ Date validation passed: {gap_days} days gap between training and backtest")

    except FileNotFoundError:
        logging.warning(f"Training metadata file not found. Skipping date validation.")
    except json.JSONDecodeError:
        logging.warning(f"Could not parse training metadata JSON. Skipping validation.")


def main(args):
    version = getattr(args, 'version', 'v1.0')
    
    base_id = _get_strategy_id(args.ticker, args.timeframe, args.helper_timeframes)
    long_id = base_id.replace('_long_short_combined', '_long')
    short_id = base_id.replace('_long_short_combined', '_short')
    
    try:
        model_long = joblib.load(f"models/{version}/{long_id}/model.joblib")
        scaler_long = joblib.load(f"models/{version}/{long_id}/scaler.joblib")
        features_long = joblib.load(f"models/{version}/{long_id}/features.joblib")
        
        model_short = joblib.load(f"models/{version}/{short_id}/model.joblib")
        scaler_short = joblib.load(f"models/{version}/{short_id}/scaler.joblib")
        features_short = joblib.load(f"models/{version}/{short_id}/features.joblib")
        
        logging.info(f"✓ Models loaded from version: {version}")
    except FileNotFoundError as e:
        logging.error(f"Model files not found for version {version}: {e}")
        logging.error(f"Expected path: models/{version}/{long_id}/")
        return
    
    # Combine all model features for preservation
    all_model_features = list(set(features_long + features_short))
    
    print(f"\n🔍 DEBUG: Loaded {len(features_long)} LONG features, {len(features_short)} SHORT features")
    print(f"🔍 DEBUG: Total unique model features: {len(all_model_features)}")
    print(f"🔍 DEBUG: hl_spread in features_long: {'hl_spread' in features_long}")
    print(f"🔍 DEBUG: dist_from_swing_high_50_4h in features_long: {'dist_from_swing_high_50_4h' in features_long}")
    print(f"🔍 DEBUG: Sample features: {features_long[:5]}\n")
    
    df = fetch_and_prepare_data(
        ticker=args.ticker,
        timeframe=args.timeframe,
        limit=args.limit,
        helper_timeframes=args.helper_timeframes,
        side='backtest',
        date_from=getattr(args, 'date_from', None),
        version=version,
        model_features_to_preserve=all_model_features
    )
    
    if df.empty:
        logging.error("Failed to prepare data")
        return

    # Validate backtest date range (prevent data leakage)
    validate_date_range(df, f"models/{version}/{long_id}", version)

    engine = BacktestEngine(
        initial_capital=args.initial_capital,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        slippage_pct=args.slippage_pct,
        enable_limit_order=getattr(args, 'limit_order', False),
        limit_order_candles=getattr(args, 'limit_order_candles', 5),
        limit_offset_pct=getattr(args, 'limit_offset_pct', 0.005),
        timeframe=args.timeframe,
        cooldown_minutes=getattr(args, 'cooldown_minutes', 30),  # Matches live bot default
        warmup_candles=getattr(args, 'warmup_candles', 200)  # Skip first N candles for indicator stability
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
        enable_partial_tp=args.partial_tp,
        enable_dynamic_tp=args.dynamic_tp,
        enable_profit_protection=getattr(args, 'protect_profit', False),
        min_proba_diff=args.min_proba_diff,
        ict_context_mode=getattr(args, 'ict_context', False),
        ict_min_strength=getattr(args, 'ict_min_strength', 0.3),
        ict_max_reduction=getattr(args, 'ict_max_reduction', 0.20),
        regime_sensitivity=getattr(args, 'regime_sensitivity', 1.0)
    )
    
    metrics = calculate_metrics(results['trades'], results['equity_curve'], args.initial_capital)
    metrics['initial_capital'] = args.initial_capital
    
    print_results(results, metrics, base_id, version)


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
        args.slippage_pct = 0.0003  # Includes funding rate equivalent
    if not hasattr(args, 'min_proba_diff'):
        args.min_proba_diff = 0.0
    if not hasattr(args, 'dynamic_tp'):
        args.dynamic_tp = False
    # Limit order defaults
    if not hasattr(args, 'limit_order'):
        args.limit_order = False
    if not hasattr(args, 'limit_order_candles'):
        args.limit_order_candles = 5
    if not hasattr(args, 'limit_offset_pct'):
        args.limit_offset_pct = 0.005
    # ICT Context Mode defaults
    if not hasattr(args, 'ict_context'):
        args.ict_context = False
    if not hasattr(args, 'ict_min_strength'):
        args.ict_min_strength = 0.3
    if not hasattr(args, 'ict_max_reduction'):
        args.ict_max_reduction = 0.20
    if not hasattr(args, 'regime_sensitivity'):
        args.regime_sensitivity = 1.0
    # Cooldown defaults
    if not hasattr(args, 'cooldown_minutes'):
        args.cooldown_minutes = 30  # Matches live bot default
    # Warmup defaults
    if not hasattr(args, 'warmup_candles'):
        args.warmup_candles = 200  # Skip first N candles for indicator stability

    # Run the main backtest function
    main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--timeframe', required=True)
    parser.add_argument('--helper-timeframes', nargs='*', default=None)
    parser.add_argument('--version', type=str, default='v1.0',
                        help='Model version to use for backtesting')
    parser.add_argument('--limit', type=int, default=10000)
    parser.add_argument('--initial-capital', type=float, default=10000.0)
    parser.add_argument('--risk-pct', type=float, default=0.02)
    parser.add_argument('--tp-pct', type=float, required=True)
    parser.add_argument('--sl-pct', type=float, required=True)
    parser.add_argument('--tsl-pct', type=float, required=True)
    parser.add_argument('--prob-threshold', type=float, required=True)
    parser.add_argument('--min-proba-diff', type=float, default=0.0,
                        help='Minimum probability difference between BUY and SELL (confidence gap)')
    parser.add_argument('--partial-tp', action='store_true',
                        help='Enable old partial TP mechanism (50%% at halfway to TP)')
    parser.add_argument('--dynamic-tp', action='store_true',
                        help='Enable new dynamic TP mechanism (25%% at each of 4 levels: 25%%, 50%%, 75%%, 100%%)')
    parser.add_argument('--protect-profit', action='store_true',
                        help='Enable profit protection: move SL to breakeven if profit peaks >0.25%% but declines before hitting partial TP')
    parser.add_argument('--maker-fee', type=float, default=0.0002)
    parser.add_argument('--taker-fee', type=float, default=0.00055)
    parser.add_argument('--slippage-pct', type=float, default=0.0003)  # Includes funding rate equivalent
    parser.add_argument('--limit-order', action='store_true',
                        help='Enable limit order mode (place limit orders instead of market orders)')
    parser.add_argument('--limit-order-candles', type=int, default=5,
                        help='Maximum candles to wait for limit order fill before cancelling')
    parser.add_argument('--limit-offset-pct', type=float, default=0.005,
                        help='Limit order price offset percentage (default 0.5%%)')
    parser.add_argument('--cooldown-minutes', type=int, default=30,
                        help='Minutes to wait after position close before allowing new entry (default: 30, matches live bot)')
    parser.add_argument('--warmup-candles', type=int, default=200,
                        help='Number of candles to skip at start for indicator stability (default: 200)')

    main(parser.parse_args())
