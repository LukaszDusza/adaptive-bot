#!/usr/bin/env python3
"""
Trading Bot V2 - With Advanced Trade Logging

FEATURES:
1. Full trade reconstruction capability
2. Candle data storage (Parquet)
3. Indicator snapshots at entry
4. Event logging (Entry, TSL, Partial TP, Exit)
5. Performance analytics
6. Visualization support
"""

import argparse
import os
import joblib
import time
import logging
import json
import numpy as np
from datetime import datetime
from typing import Optional, Dict
from dotenv import load_dotenv

from bybit_adapter import BybitAdapter
from data_preparer_pa import fetch_and_prepare_data
from trade_logger import TradeLogger
from prediction_logger import PredictionLogger
from feature_cache import FeatureCache
from trade_metrics import TradeMetricsCalculator

# Logging setup with rotation and compact formatting
from logging_config import setup_logging

# Setup logger for bot module
logger = setup_logging('bot', enable_file_logging=True)


class BotConfig:
    """Bot configuration"""
    TICKER: str = "ETHUSDT"
    TIMEFRAME: str = "15m"
    HELPER_TIMEFRAMES: list = ["1h", "4h"]
    TRADE_SIZE_USD: float = 100.0
    TRADE_SIZE_PCT: float = 0.0  # If > 0, use % of available balance (e.g., 0.01 = 1%). Overrides TRADE_SIZE_USD
    LEVERAGE: int = 10
    MAX_POSITIONS: int = 0  # Maximum number of concurrent positions (0 = no limit). Checked via API before opening position.
    TP_PCT: float = 0.03
    TSL_PCT: float = 0.01
    PROBABILITY_THRESHOLD: float = 0.7
    MIN_CONFIDENCE_RATIO: float = 1.5  # Minimum confidence ratio: stronger_signal / weaker_signal (1.5 = 50% more confident)
    LOOP_SLEEP_SECONDS: int = 300  # 5 minutes - optimized for 15m timeframe (reduces cache hits from 93% to 67%)
    CANDLES_FOR_FEATURES: int = 500  # Reduced from 5000 for live bot (5000 is too heavy for Docker 2GB RAM limit)
    PARTIAL_TP_ENABLED: bool = True
    DYNAMIC_TP_ENABLED: bool = False  # New dynamic TP: 25% at each of 4 levels (25%, 50%, 75%, 100%)
    HEDGE_MODE: bool = False  # Hedge Mode: positionIdx 1=Long, 2=Short. One-Way Mode: positionIdx 0
    LIMIT_ORDER_MODE: bool = False  # Use limit orders instead of market orders
    MAX_WAITING_LIMIT_ORDER: int = 300  # Seconds to wait for limit order execution before cancelling (will be auto-calculated if LIMIT_ORDER_CANDLES is set)
    LIMIT_ORDER_CANDLES: int = 5  # How many candles to wait for limit order (default: 5 candles). Auto-calculates MAX_WAITING_LIMIT_ORDER based on TIMEFRAME
    LIMIT_OFFSET_PCT: float = 0.005  # Price offset for limit orders (0.5% default)
    PROTECT_PROFIT_ENABLED: bool = False  # Enable profit protection: move SL to BE if profit peaks >0.25% but declines
    COOLDOWN_AFTER_LOSS_MINUTES: int = 30  # Minutes to wait before opening new position after ANY closed position (30min = 2x 15m candles, prevents re-entry on same candle)
    RETRY_DELAY: int = 3
    # DCA (Dollar Cost Averaging) Mode - Single ATR-based limit order
    # Based on historical analysis (Nov 2025): Avg ATR = 0.785%, 1.5x multiplier = 1.177% distance
    DCA_ENABLED: bool = False  # Enable DCA mode (requires LIMIT_ORDER_MODE=True)
    DCA_ATR_MULTIPLIER: float = 1.5  # ATR-based distance (ATR * 1.5 = ~1.18% avg, recommended)
    # FIX #5: Magic numbers moved to config
    FEE_PCT: float = 0.0006  # 0.06% round-trip fees (maker+taker conservative estimate)
    API_RETRY_INITIAL_SLEEP: float = 0.5  # Initial sleep for API retries (exponential backoff)
    API_RETRY_MAX_ATTEMPTS: int = 5  # Max retry attempts for API calls

    # ICT Context Mode - Dynamic threshold adjustment based on Smart Money signals
    ICT_CONTEXT_MODE: bool = False  # Enable ICT context-aware decision making (OPCJA 3)
    ICT_MIN_STRENGTH: float = 0.3  # Minimum ICT strength to apply threshold reduction (0.0-1.0)
    ICT_MAX_THRESHOLD_REDUCTION: float = 0.20  # Max % reduction in thresholds when ICT strong (e.g., 0.20 = 20%)
    REGIME_SENSITIVITY: float = 1.0  # Multiplier for regime adjustments (1.0 = normal, 0.5 = less sensitive, 2.0 = more sensitive)

    # RSI Reversal Filter - Enter on reversal FROM extreme, not AT extreme
    RSI_REVERSAL_FILTER_ENABLED: bool = True  # Enable RSI reversal filter (crossing 70/30 levels)
    RSI_HIGH_THRESHOLD: float = 65.0  # Block LONG if RSI above this without 30-cross (default: 65)
    RSI_LOW_THRESHOLD: float = 35.0   # Block SHORT if RSI below this without 70-cross (default: 35)
    RSI_OVERBOUGHT_LEVEL: float = 70.0  # RSI level for overbought (crossing down = SHORT signal)
    RSI_OVERSOLD_LEVEL: float = 30.0   # RSI level for oversold (crossing up = LONG signal)

    def __repr__(self):
        """
        Safe representation without exposing sensitive data.

        SECURITY: This method intentionally excludes API keys, secrets, and
        any other sensitive configuration that should not appear in logs.
        """
        return (
            f"BotConfig(ticker={self.TICKER}, timeframe={self.TIMEFRAME}, "
            f"leverage={self.LEVERAGE}, trade_size={self.TRADE_SIZE_USD}, "
            f"tp={self.TP_PCT}, tsl={self.TSL_PCT}, threshold={self.PROBABILITY_THRESHOLD})"
        )


class TradingBot:
    """Trading bot with advanced logging"""
    
    def __init__(self, config: BotConfig, version: str = 'v1.0'):
        self.config = config
        self.version = version

        # Auto-calculate MAX_WAITING_LIMIT_ORDER based on LIMIT_ORDER_CANDLES and TIMEFRAME
        if self.config.LIMIT_ORDER_CANDLES > 0:
            timeframe_minutes = self._parse_timeframe_to_minutes(self.config.TIMEFRAME)
            self.config.MAX_WAITING_LIMIT_ORDER = self.config.LIMIT_ORDER_CANDLES * timeframe_minutes * 60
            logger.info(f"🕐 Auto-calculated limit order timeout: {self.config.LIMIT_ORDER_CANDLES} candles × {timeframe_minutes}min = {self.config.MAX_WAITING_LIMIT_ORDER}s ({self.config.MAX_WAITING_LIMIT_ORDER/60:.1f} minutes)")

        self.adapter = self._init_adapter()
        self.base_id = self._get_strategy_id()

        # Load models
        (self.model_long, self.scaler_long, self.features_long,
         self.model_short, self.scaler_short, self.features_short) = self._load_models()
        
        # State management
        self.state = {}
        self.state_dir = "bot_state"
        os.makedirs(self.state_dir, exist_ok=True)
        self.state_file = os.path.join(self.state_dir, f"{self.base_id}_state.json")
        self._load_state()
        
        # Initialize trade logger
        self.trade_logger = TradeLogger(base_dir="logs")

        # Initialize prediction logger
        self.prediction_logger = PredictionLogger(
            strategy_id=self.base_id,
            log_base_dir="/app/prediction_logs",
            container_name=os.getenv('CONTAINER_NAME', 'unknown')
        )

        # Restore trade logging if position was active before restart
        print("📋 Step 1/4: Restoring trade logging if needed...")
        self._restore_trade_logging_if_needed()
        print("✓ Trade logging restoration complete")

        # Cache for last decision data
        self.last_decision_data = {}
        self.last_candle_data = {}

        # Limit order tracking
        # Single mode: 1 order, DCA mode: 3 orders
        # IMPORTANT: Initialize BEFORE _check_existing_position_on_startup() so recovery can populate it
        self.active_limit_orders = []  # List of {'order_id': str, 'timestamp': float, 'side': str, 'price': float, 'qty': float, 'level': int}

        # Store df for DCA level calculations
        self.last_df = None

        # EPHEMERAL MODE: Auto-recovery from Bybit API
        # If container restarted without persistent state, detect existing position and restore
        # IMPORTANT: This must come AFTER self.active_limit_orders initialization
        print("📋 Step 2/4: Checking existing positions on Bybit...")
        self._check_existing_position_on_startup()
        print("✓ Position check complete")

        # PERFORMANCE: Feature cache for incremental updates
        print("📋 Step 3/4: Initializing feature cache...")
        self.feature_cache = FeatureCache(window_size=self.config.CANDLES_FOR_FEATURES)
        print("✓ Feature cache initialized")

        logger.info("="*60)
        logger.info(f"Bot V2 initialized with Trade Logger: {self.base_id}")
        logger.info(f"TP: {config.TP_PCT*100:.2f}% | TSL: {config.TSL_PCT*100:.2f}%")
        logger.info("="*60)

    def _parse_timeframe_to_minutes(self, timeframe: str) -> int:
        """
        Parse timeframe string to minutes.
        Examples: '15m' → 15, '1h' → 60, '4h' → 240, '1D' → 1440
        """
        timeframe = timeframe.strip().replace(' ', '')
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('D') or timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            raise ValueError(f"Unsupported timeframe format: {timeframe}. Use formats like '15m', '1h', '4h', '1D'")

    # ========== FIX #9: Helper methods for API retry with exponential backoff ==========

    def _retry_with_backoff(self, func, *args, max_attempts=None, **kwargs):
        """
        Execute function with exponential backoff retry.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            max_attempts: Max retry attempts (default: from config)
            **kwargs: Keyword arguments for func

        Returns:
            Function result or None if all retries failed
        """
        if max_attempts is None:
            max_attempts = self.config.API_RETRY_MAX_ATTEMPTS

        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:
                    # Last attempt failed
                    logger.error(f"All {max_attempts} retry attempts failed for {func.__name__}: {e}")
                    return None

                # Calculate exponential backoff: 0.5s, 0.75s, 1.12s, 1.69s, 2.53s
                sleep_time = self.config.API_RETRY_INITIAL_SLEEP * (1.5 ** attempt)
                logger.warning(f"Retry {attempt + 1}/{max_attempts} for {func.__name__} after {sleep_time:.2f}s: {e}")
                time.sleep(sleep_time)

        return None

    def _init_adapter(self):
        # Load .env_demo as fallback (Docker containers use env_file in docker-compose.yaml)
        load_dotenv('.env_demo')
        key = os.getenv("BYBIT_API_KEY")
        secret = os.getenv("BYBIT_API_SECRET")
        base_url = os.getenv("BYBIT_BASE_URL")
        if not key or not secret:
            raise ValueError("Missing API keys - ensure .env file is configured or env vars are set")
        return BybitAdapter(api_key=key, api_secret=secret, base_url=base_url, hedge_mode=self.config.HEDGE_MODE)
    
    def _get_strategy_id(self):
        helpers = '_plus_' + '_'.join(self.config.HELPER_TIMEFRAMES) if self.config.HELPER_TIMEFRAMES else ""
        return f"{self.config.TICKER}_{self.config.TIMEFRAME.replace(' ', '')}{helpers}"
    
    def _load_models(self):
        version = getattr(self, 'version', 'v1.0')
        long_id = f"{self.base_id}_long"
        short_id = f"{self.base_id}_short"
        
        try:
            m_long = joblib.load(f"models/{version}/{long_id}/model.joblib")
            s_long = joblib.load(f"models/{version}/{long_id}/scaler.joblib")
            f_long = joblib.load(f"models/{version}/{long_id}/features.joblib")
            
            m_short = joblib.load(f"models/{version}/{short_id}/model.joblib")
            s_short = joblib.load(f"models/{version}/{short_id}/scaler.joblib")
            f_short = joblib.load(f"models/{version}/{short_id}/features.joblib")
            
            logger.info(f"✓ Models loaded from version {version} (Long: {len(f_long)} features, Short: {len(f_short)} features)")
            return m_long, s_long, f_long, m_short, s_short, f_short
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found for version {version}: {e}")
    
    def _save_state(self):
        """
        Save bot state to JSON file with deep copy for thread safety.

        FIX #3: Changed from shallow copy to deep copy to prevent race conditions
        when state is modified during JSON serialization.
        """
        try:
            import copy
            # Deep copy for thread safety (prevents RuntimeError if dict changes during iteration)
            state_copy = copy.deepcopy(self.state)
            state_copy['last_updated'] = datetime.now().isoformat()

            # Write with atomic rename to prevent corruption
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(state_copy, f, indent=2)

            # Atomic rename (prevents partial writes if crash during save)
            os.replace(temp_file, self.state_file)

            # Set restricted permissions (owner read/write only)
            os.chmod(self.state_file, 0o600)

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
                logger.info(f"✓ State loaded: {self.state}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
    
    def _restore_trade_logging_if_needed(self):
        """
        Restore trade logging if bot restarts with an active position.
        This prevents the 'Attempted to log TSL_UPDATE but no active trade' warning.
        """
        if not self.state:
            return
        
        # Check if state indicates an active position
        if 'side' in self.state and 'entry_price' in self.state:
            side = self.state.get('side')
            entry_price = self.state.get('entry_price')
            
            if side and entry_price:
                logger.info(f"🔄 Restoring trade logging for existing {side} position @ {entry_price}")
                
                # Create minimal decision data for restored trade
                decision_data = {
                    "decision": "BUY" if side == "Long" else "SELL",
                    "proba_buy": 0.0,
                    "proba_sell": 0.0,
                    "threshold": self.config.PROBABILITY_THRESHOLD,
                    "candle_close_time": self.state.get('last_updated', datetime.now().isoformat()),
                    "restored": True  # Flag to indicate this is a restored trade
                }
                
                # Restart trade logging
                self.trade_logger.start_trade(
                    ticker=self.config.TICKER,
                    side=side.upper(),
                    decision_data=decision_data
                )
                
                logger.info("✓ Trade logging restored")

    def _check_existing_position_on_startup(self):
        """
        EPHEMERAL MODE: Auto-recovery from Bybit API.

        If container restarted without persistent state (ephemeral volumes),
        this method detects existing positions AND open limit orders from Bybit API
        and reconstructs minimal state to prevent duplicate position errors.
        """
        try:
            # Skip if state already has position info (normal restoration path)
            if self.state.get('side'):
                logger.info("✓ State already loaded, skipping API recovery")
                # Still check for open orders even if position state exists
                self._restore_open_orders_from_bybit()
                return

            # Check Bybit API for existing position
            position = self.adapter.get_position(self.config.TICKER)

            if not position:
                logger.info("✓ No existing position detected on Bybit")
                # Check for open limit orders even without position
                self._restore_open_orders_from_bybit()
                return

            # Position exists but state is empty - RECOVERY NEEDED
            side = position['side']  # "Long" or "Short"
            entry_price = position['entryPrice']
            size = position['size']
            stop_loss = position.get('stopLoss', 0.0)
            take_profit = position.get('takeProfit', 0.0)

            logger.warning("="*60)
            logger.warning(f"⚠️  EPHEMERAL RECOVERY: Detected {side} position on Bybit")
            logger.warning(f"    Entry: {entry_price:.4f} | Size: {size}")
            logger.warning(f"    SL: {stop_loss:.4f} | TP: {take_profit:.4f}")
            logger.warning(f"    Reconstructing state from API...")
            logger.warning("="*60)

            # ========== CRITICAL FIX: Set SL/TP if missing ==========
            # This happens when limit order fills but bot restarts before _finalize_limit_order_entry()
            # can set SL/TP. Position is unprotected - MUST set SL/TP immediately.
            position_side = "Buy" if side == "Long" else "Sell"
            sl_tp_set = False

            # Check if SL is missing (0 or very close to 0)
            if stop_loss == 0.0:
                # Calculate SL based on entry price and config
                if side == "Long":
                    calculated_sl = entry_price * (1 - self.config.TSL_PCT)
                else:
                    calculated_sl = entry_price * (1 + self.config.TSL_PCT)

                try:
                    logger.warning(f"    ⚠️  MISSING SL DETECTED - Setting SL: {calculated_sl:.4f}")
                    self.adapter.set_stop_loss(self.config.TICKER, calculated_sl, position_side)
                    stop_loss = calculated_sl
                    sl_tp_set = True
                    logger.warning(f"    ✓ SL set successfully: {calculated_sl:.4f}")
                except Exception as e:
                    logger.error(f"    ❌ Failed to set SL: {e}")

            # Check if TP is missing (0 or very close to 0)
            if take_profit == 0.0 and self.config.TP_PCT > 0:
                # Calculate TP based on entry price and config
                if side == "Long":
                    calculated_tp = entry_price * (1 + self.config.TP_PCT)
                else:
                    calculated_tp = entry_price * (1 - self.config.TP_PCT)

                try:
                    logger.warning(f"    ⚠️  MISSING TP DETECTED - Setting TP: {calculated_tp:.4f}")
                    self.adapter.set_take_profit(self.config.TICKER, calculated_tp, position_side)
                    take_profit = calculated_tp
                    sl_tp_set = True
                    logger.warning(f"    ✓ TP set successfully: {calculated_tp:.4f}")
                except Exception as e:
                    logger.error(f"    ❌ Failed to set TP: {e}")

            if sl_tp_set:
                logger.warning("    ✅ Position protection restored (SL/TP set)")
            else:
                logger.info("    ✓ SL/TP already set on Bybit")

            # Reconstruct minimal state to prevent duplicate positions
            self.state = {
                'side': side,
                'entry_price': entry_price,
                'initial_tp': take_profit if take_profit > 0 else (entry_price * (1 + self.config.TP_PCT) if side == "Long" else entry_price * (1 - self.config.TP_PCT)),
                'last_sl': stop_loss,  # Use actual SL from API
                'partial_tp_taken': False,
                'dynamic_tp_levels_taken': 0,
                'highest_price': entry_price if side == "Long" else 0.0,
                'lowest_price': entry_price if side == "Short" else float('inf')
            }
            self._save_state()

            # Start trade logger with recovered position
            if not self.trade_logger.current_trade:
                self.trade_logger.start_trade(
                    ticker=self.config.TICKER,
                    side=side,
                    decision_data={
                        'recovered_from_api': True,
                        'entry_price': entry_price,
                        'size': size
                    }
                )
                logger.info(f"✓ Trade logger started for recovered {side} position")

            # Also check for any open limit orders
            self._restore_open_orders_from_bybit()

        except Exception as e:
            logger.error(f"Error during startup position recovery: {e}", exc_info=True)
            # Non-fatal - bot will continue, but may have duplicate position risk
            logger.warning("⚠️  Recovery failed - monitor for duplicate positions!")

    def _restore_open_orders_from_bybit(self):
        """
        Restore open limit orders from Bybit API after restart.

        This ensures that if bot restarts while limit orders are pending,
        it will continue monitoring them and process fills correctly.

        Critical for ensuring SL/TP is set after order fills post-restart.
        """
        if not self.config.LIMIT_ORDER_MODE:
            return  # Not in limit order mode

        try:
            # Get open orders from Bybit
            existing_orders = self.adapter.get_open_orders(self.config.TICKER)

            if not existing_orders:
                logger.info("✓ No open limit orders detected on Bybit")
                return

            # We have open orders - restore them to tracking
            logger.warning("="*60)
            logger.warning(f"🔄 LIMIT ORDER RECOVERY: Found {len(existing_orders)} open orders on Bybit")

            # Clear any stale in-memory state
            self.active_limit_orders = []

            # Populate self.active_limit_orders from Bybit
            for idx, bybit_order in enumerate(existing_orders, start=1):
                order_id = bybit_order.get('orderId')
                side = bybit_order.get('side')  # "Buy" or "Sell"
                price = float(bybit_order.get('price', 0))
                qty = float(bybit_order.get('qty', 0))
                created_time = bybit_order.get('createdTime', 0)

                # Convert to timestamp (Bybit uses milliseconds)
                timestamp = int(created_time) / 1000 if created_time else time.time()

                # Reconstruct order tracking
                self.active_limit_orders.append({
                    'order_id': order_id,
                    'timestamp': timestamp,
                    'side': 'LONG' if side == 'Buy' else 'SHORT',
                    'price': price,
                    'qty': qty,
                    'decision': 'BUY' if side == 'Buy' else 'SELL',
                    'level': idx  # Assign sequential level (original level unknown)
                })

                logger.warning(f"   ✓ Restored: {order_id} | {side} @ {price:.4f} | Qty: {qty:.4f}")

            logger.warning(f"   Total: {len(self.active_limit_orders)} orders restored to monitoring")
            logger.warning(f"   Bot will now track these orders and process fills")
            logger.warning("="*60)

            # Start trade logger if not already started
            if not self.trade_logger.current_trade and self.active_limit_orders:
                side = self.active_limit_orders[0]['side']
                self.trade_logger.start_trade(
                    ticker=self.config.TICKER,
                    side=side,
                    decision_data={
                        'recovered_orders': True,
                        'order_count': len(self.active_limit_orders),
                        'timestamp': datetime.now().isoformat()
                    }
                )
                logger.info(f"✓ Trade logger started for recovered limit orders")

        except Exception as e:
            logger.error(f"Failed to restore open orders from Bybit: {e}", exc_info=True)
            logger.warning("⚠️  Order recovery failed - monitor for missing SL/TP after fills!")

    def _extract_top_indicators(self, df_row, features_list, n=10) -> Dict[str, float]:
        """Extract top N indicator values from dataframe row"""
        indicators = {}
        for feat in features_list[:n]:
            if feat in df_row.index:
                value = df_row[feat]
                # Convert numpy types to python types
                if hasattr(value, 'item'):
                    value = value.item()
                indicators[feat] = float(value) if value is not None else None
        return indicators

    def _calculate_dca_levels(self, decision: str, current_price: float, df=None) -> list:
        """
        Calculate single ATR-based DCA limit order level.

        Based on historical analysis (Nov 2025):
        - Average ATR: 0.785%
        - Recommended multiplier: 1.5x
        - Average distance: 1.177%

        Returns list with 1 price: [dca_level]
        """
        try:
            # Get ATR from last closed candle
            atr = None
            if df is not None and len(df) > 1:
                df_closed = df.iloc[:-1]
                last_row = df_closed.iloc[-1]
                if 'atr_14' in last_row.index:
                    atr = last_row['atr_14']

            # Calculate ATR-based DCA level
            if atr and atr > 0:
                atr_distance = atr * self.config.DCA_ATR_MULTIPLIER

                if decision == "BUY":
                    dca_level = current_price - atr_distance
                else:  # SELL
                    dca_level = current_price + atr_distance

                distance_pct = (atr_distance / current_price) * 100

                logger.info(f"📊 DCA Level calculated for {decision}:")
                logger.info(f"   Current price: ${current_price:.4f}")
                logger.info(f"   ATR (14):      ${atr:.6f}")
                logger.info(f"   Multiplier:    {self.config.DCA_ATR_MULTIPLIER}x")
                logger.info(f"   Distance:      ${atr_distance:.6f} ({distance_pct:.3f}%)")
                logger.info(f"   DCA Level:     ${dca_level:.4f}")

                return [dca_level]

            else:
                # Fallback if ATR not available: use fixed 1.2% offset
                fallback_pct = 0.012  # 1.2% (close to 1.177% average)

                if decision == "BUY":
                    dca_level = current_price * (1 - fallback_pct)
                else:
                    dca_level = current_price * (1 + fallback_pct)

                logger.warning(f"⚠️  ATR not available, using fallback {fallback_pct*100:.1f}% offset")
                logger.info(f"   DCA Level: ${dca_level:.4f}")

                return [dca_level]

        except Exception as e:
            logger.error(f"❌ Error calculating DCA level: {e}", exc_info=True)
            # Fallback: fixed 1.2% offset
            fallback_pct = 0.012
            if decision == "BUY":
                return [current_price * (1 - fallback_pct)]
            else:
                return [current_price * (1 + fallback_pct)]

    def _cancel_unfilled_limit_orders_safe(self, orders_to_cancel: list, reason: str = "position_closed"):
        """
        FIX #1: Safe cancellation of specific unfilled limit orders.

        This method only cancels orders that have been verified to be open on Bybit,
        preventing race conditions where orders fill between check and cancel.

        Args:
            orders_to_cancel: List of order dicts to cancel (pre-filtered)
            reason: Reason for cancellation (for logging)
        """
        if not orders_to_cancel:
            return

        cancelled_count = 0
        total_count = len(orders_to_cancel)
        logger.warning(f"🚫 Cancelling {total_count} verified unfilled order(s) (reason: {reason})")

        for order_info in orders_to_cancel:
            try:
                order_id = order_info['order_id']
                resp = self.adapter.cancel_order(self.config.TICKER, order_id)

                logger.info(f"   ✓ Cancelled order {order_id} (Level {order_info.get('level', 'N/A')})")
                cancelled_count += 1

                # Log cancellation event
                if self.trade_logger.current_trade:
                    self.trade_logger.log_event(f"LIMIT_ORDER_CANCELLED", {
                        "order_id": order_id,
                        "level": order_info.get('level'),
                        "price": float(order_info.get('price', 0)),
                        "reason": reason
                    })

            except Exception as e:
                # This can still fail if order fills between our check and this cancel
                # But it's much less likely than without the pre-check
                logger.warning(f"   ⚠️  Failed to cancel order {order_id}: {e} (may have filled concurrently)")

        # Clear active limit orders list
        self.active_limit_orders.clear()
        logger.warning(f"   ✅ Cancelled {cancelled_count}/{total_count} orders")

    def _cancel_unfilled_limit_orders(self, reason: str = "position_closed"):
        """
        Cancel all unfilled limit orders (DCA mode).

        CRITICAL: This prevents unfilled DCA orders from executing AFTER
        position is closed (e.g., after hitting TP), which would open
        a new unwanted position.

        Args:
            reason: Reason for cancellation (for logging)
        """
        if not self.active_limit_orders:
            return

        cancelled_count = 0
        total_count = len(self.active_limit_orders)
        logger.warning(f"🚫 Cancelling {total_count} unfilled limit order(s) (reason: {reason})")

        for order_info in self.active_limit_orders[:]:  # Use slice copy to avoid modification during iteration
            try:
                order_id = order_info['order_id']
                resp = self.adapter.cancel_order(self.config.TICKER, order_id)

                logger.info(f"   ✓ Cancelled order {order_id} (Level {order_info.get('level', 'N/A')})")
                cancelled_count += 1

                # Log cancellation event
                if self.trade_logger.current_trade:
                    self.trade_logger.log_event(f"LIMIT_ORDER_CANCELLED", {
                        "order_id": order_id,
                        "level": order_info.get('level'),
                        "price": float(order_info.get('price', 0)),
                        "reason": reason
                    })

            except Exception as e:
                logger.error(f"   ✗ Failed to cancel order {order_id}: {e}")

        # Clear active limit orders list
        self.active_limit_orders.clear()
        logger.warning(f"   ✅ Cancelled {cancelled_count}/{total_count} orders")

    def _is_in_cooldown_period(self) -> bool:
        """
        Check if bot is in cooldown period after ANY closed position.

        This prevents bot from immediately re-entering on the same 15m candle
        when it checks every 60s (e.g., after manual close or TP/SL hit).

        Returns True if cooldown is active, False otherwise.
        """
        if not self.state:
            return False

        last_close_time_str = self.state.get('last_position_closed_at')
        if not last_close_time_str:
            return False

        try:
            last_close_time = datetime.fromisoformat(last_close_time_str)
            cooldown_minutes = self.state.get('cooldown_minutes', self.config.COOLDOWN_AFTER_LOSS_MINUTES)
            elapsed_minutes = (datetime.now() - last_close_time).total_seconds() / 60

            if elapsed_minutes < cooldown_minutes:
                remaining_minutes = cooldown_minutes - elapsed_minutes
                last_pnl = self.state.get('last_pnl_usd', 0)
                pnl_info = f" (last PnL: {last_pnl:+.2f} USD)" if last_pnl != 0 else ""
                logger.info(f"⏰ COOLDOWN ACTIVE: {remaining_minutes:.1f} min remaining{pnl_info}")
                return True
            else:
                # Cooldown expired - clear state
                logger.info(f"✅ Cooldown expired ({elapsed_minutes:.1f} min). Ready for new positions.")
                self.state = {}
                self._save_state()
                return False

        except Exception as e:
            logger.error(f"Error checking cooldown: {e}")
            return False
    
    def get_decision(self, df=None) -> str:
        """Get ML model decision with full logging"""
        try:
            # Fetch data if not provided
            if df is None:
                logger.info(f"📥 Fetching market data for {self.config.TICKER}...")
                # Combine all model features for preservation
                all_model_features = list(set(self.features_long + self.features_short))

                # PERFORMANCE: Use feature cache for incremental updates
                df, is_new_candle = self.feature_cache.get_features(
                    ticker=self.config.TICKER,
                    timeframe=self.config.TIMEFRAME,
                    helper_timeframes=self.config.HELPER_TIMEFRAMES,
                    side='backtest',
                    version=self.version,
                    model_features_to_preserve=all_model_features
                )
                logger.info(f"✓ Data fetched successfully ({len(df)} candles)")

                # OPTIMIZATION: Skip ML prediction if same candle (cache hit)
                if not is_new_candle:
                    logger.info("⏭️  Skipping ML prediction - no new closed candle (cache hit)")
                    return "HOLD"
            
            if df.empty:
                logger.error("❌ Empty dataframe received!")
                return "ERROR"
            
            # CRITICAL: Use ONLY closed candles (exclude the last incomplete candle)
            # The last candle (iloc[-1]) is still forming and will be excluded
            df_closed = df.iloc[:-1]
            
            if df_closed.empty:
                logger.error("❌ No closed candles available!")
                return "ERROR"
            
            # Use the last CLOSED candle for ML prediction
            last_row = df_closed.iloc[-1]
            last_row_df = df_closed.iloc[[-1]]
            
            # Extract features with validation
            logger.info("🤖 Running ML model predictions...")

            # VALIDATION: Check if all required features are present
            missing_long = set(self.features_long) - set(last_row_df.columns)
            missing_short = set(self.features_short) - set(last_row_df.columns)

            if missing_long:
                logger.error(f"❌ Missing {len(missing_long)} LONG features!")
                logger.error(f"   Examples: {list(missing_long)[:5]}")
                return "ERROR"

            if missing_short:
                logger.error(f"❌ Missing {len(missing_short)} SHORT features!")
                logger.error(f"   Examples: {list(missing_short)[:5]}")
                return "ERROR"

            X_long = last_row_df[self.features_long]
            X_short = last_row_df[self.features_short]
            
            X_long_scaled = self.scaler_long.transform(X_long)
            X_short_scaled = self.scaler_short.transform(X_short)
            
            proba_buy = self.model_long.predict_proba(X_long_scaled)[0][1]
            proba_sell = self.model_short.predict_proba(X_short_scaled)[0][1]
            
            # Calculate confidence ratio (relative strength)
            confidence_ratio = max(proba_buy, proba_sell) / min(proba_buy, proba_sell) if min(proba_buy, proba_sell) > 0 else float('inf')
            
            logger.info(f"📊 Model Probabilities: BUY={proba_buy:.3f}, SELL={proba_sell:.3f} (threshold={self.config.PROBABILITY_THRESHOLD:.3f})")
            logger.info(f"📏 Confidence Ratio: {confidence_ratio:.3f} (min_required={self.config.MIN_CONFIDENCE_RATIO:.3f})")

            # ========== ICT CONTEXT MODE (OPCJA 3) ==========
            # Dynamic threshold adjustment based on Smart Money signals
            # When ICT signals are strong → lower thresholds (catch more opportunities)
            # When market regime is unstable → higher thresholds (filter noise)

            if self.config.ICT_CONTEXT_MODE:
                # Extract ICT features from last closed candle
                ict_composite = last_row.get('ict_composite_score', 0.5)  # Default: neutral
                liquidity_sweep = last_row.get('liquidity_sweep', 0.0)
                fvg_signal = last_row.get('fvg_signal', 0.0)
                order_block = last_row.get('order_block', 0.0)
                market_structure_shift = last_row.get('market_structure_shift', 0.0)

                # Extract market regime features
                rsi = last_row.get('rsi_14', 50)
                atr_norm = last_row.get('atr_normalized', 1.0)
                volume_ratio = last_row.get('volume_vs_ma_20', 1.0)

                # 1. Calculate ICT Strength (0-1 scale)
                ict_strength = 0.0

                # High-value ICT signals (weighted sum)
                if ict_composite > 0.6:  # Strong composite score
                    ict_strength += 0.3
                if liquidity_sweep > 0:  # Liquidity grab detected
                    ict_strength += 0.2
                if abs(fvg_signal) > 0:  # Fair Value Gap present
                    ict_strength += 0.2
                if abs(order_block) > 0:  # Order Block identified
                    ict_strength += 0.2
                if abs(market_structure_shift) > 0:  # Structure break
                    ict_strength += 0.1

                ict_strength = min(ict_strength, 1.0)  # Cap at 1.0

                # 2. Calculate Market Regime Multiplier
                regime_multiplier = 1.0

                # Trending market (easier to trade) → lower thresholds
                if (rsi > 60 and proba_buy > proba_sell) or (rsi < 40 and proba_sell > proba_buy):
                    regime_multiplier *= (1.0 - 0.1 * self.config.REGIME_SENSITIVITY)  # -10% threshold

                # High volatility with volume confirmation → lower thresholds
                if atr_norm > 1.2 and volume_ratio > 1.3:
                    regime_multiplier *= (1.0 - 0.15 * self.config.REGIME_SENSITIVITY)  # -15% threshold

                # Low volatility OR low volume → LEKKO OBNIŻ (więcej okazji w spokojnym rynku)
                # FIX: Było 1.0 + 0.15 (podnosiło), teraz 1.0 - 0.05 (lekko obniża)
                if atr_norm < 0.7 or volume_ratio < 0.6:
                    regime_multiplier *= (1.0 - 0.05 * self.config.REGIME_SENSITIVITY)  # -5% threshold (neutral/slightly lower)

                # Extreme RSI without volume → LEKKO PODNOSI (ale mniej niż wcześniej)
                # FIX: Było 1.0 + 0.25 (mocno podnosiło), teraz 1.0 + 0.10 (lekko podnosi)
                if (rsi > 75 or rsi < 25) and volume_ratio < 0.8:
                    regime_multiplier *= (1.0 + 0.10 * self.config.REGIME_SENSITIVITY)  # +10% threshold (was +25%)

                # 3. Dynamic Threshold Adjustment
                base_prob_threshold = self.config.PROBABILITY_THRESHOLD
                base_ratio_threshold = self.config.MIN_CONFIDENCE_RATIO

                # Apply ICT strength reduction (only if above minimum strength)
                if ict_strength >= self.config.ICT_MIN_STRENGTH:
                    ict_adjustment = 1.0 - (ict_strength * self.config.ICT_MAX_THRESHOLD_REDUCTION)
                else:
                    ict_adjustment = 1.0  # No reduction if ICT too weak

                # Apply adjustments
                dynamic_prob_threshold = base_prob_threshold * regime_multiplier * ict_adjustment
                dynamic_ratio_threshold = base_ratio_threshold / (regime_multiplier * ict_adjustment)  # Note: inverse - lower thresholds = easier to meet

                # Safety bounds (never go too extreme)
                dynamic_prob_threshold = np.clip(dynamic_prob_threshold, 0.45, 0.85)
                dynamic_ratio_threshold = np.clip(dynamic_ratio_threshold, 1.10, 2.50)  # Range: 10% to 150% more confident

                # Enhanced logging
                logger.info(f"🎯 ICT Context: composite={ict_composite:.2f}, strength={ict_strength:.2f}")
                logger.info(f"📈 Regime: RSI={rsi:.1f}, ATR={atr_norm:.2f}, Vol={volume_ratio:.2f}, mult={regime_multiplier:.2f}")
                logger.info(f"🔧 Dynamic Thresholds: prob={dynamic_prob_threshold:.3f} (base={base_prob_threshold:.2f}), ratio={dynamic_ratio_threshold:.3f} (base={base_ratio_threshold:.2f})")
            else:
                # Use static thresholds (original behavior)
                dynamic_prob_threshold = self.config.PROBABILITY_THRESHOLD
                dynamic_ratio_threshold = self.config.MIN_CONFIDENCE_RATIO
                logger.info(f"📊 Static Thresholds: prob={dynamic_prob_threshold:.3f}, ratio={dynamic_ratio_threshold:.3f}")

            # Determine decision with dynamic thresholds
            decision = "HOLD"
            if proba_buy > dynamic_prob_threshold and proba_buy > proba_sell:
                # Check if confidence gap is sufficient
                if confidence_ratio >= dynamic_ratio_threshold:
                    decision = "BUY"
                    logger.info(f"✅ BUY signal accepted: proba={proba_buy:.3f} > threshold={dynamic_prob_threshold:.3f}, ratio={confidence_ratio:.3f} >= {dynamic_ratio_threshold:.3f}")
                else:
                    logger.info(f"⚠️  BUY signal rejected: insufficient confidence ratio ({confidence_ratio:.3f} < {dynamic_ratio_threshold:.3f})")
            elif proba_sell > dynamic_prob_threshold and proba_sell > proba_buy:
                # Check if confidence gap is sufficient
                if confidence_ratio >= dynamic_ratio_threshold:
                    decision = "SELL"
                    logger.info(f"✅ SELL signal accepted: proba={proba_sell:.3f} > threshold={dynamic_prob_threshold:.3f}, ratio={confidence_ratio:.3f} >= {dynamic_ratio_threshold:.3f}")
                else:
                    logger.info(f"⚠️  SELL signal rejected: insufficient confidence ratio ({confidence_ratio:.3f} < {dynamic_ratio_threshold:.3f})")
            else:
                # HOLD - log the reason
                logger.info(f"⏸️  HOLD: Neither condition met (BUY={proba_buy:.3f}, SELL={proba_sell:.3f}, threshold={dynamic_prob_threshold:.3f}, ratio={confidence_ratio:.3f})")

            logger.info(f"🎯 ML Decision: {decision}")

            # ========== RSI REVERSAL FILTER ==========
            # Override ML decision based on RSI crossing levels (reversal signals)
            # Logic: Enter on reversal FROM extreme, not AT extreme
            # - SHORT: When RSI crosses OVERBOUGHT_LEVEL DOWN (reversal from overbought)
            # - LONG: When RSI crosses OVERSOLD_LEVEL UP (reversal from oversold)
            # - HOLD: If high/low RSI without recent crossing

            if self.config.RSI_REVERSAL_FILTER_ENABLED:
                rsi_current = last_row.get('rsi_14', 50)  # Default to neutral if missing
                rsi_prev = df_closed.iloc[-2].get('rsi_14', 50) if len(df_closed) > 1 else rsi_current

                original_decision = decision  # Store original for logging

                # Check for RSI crossings (reversal signals)
                rsi_crossed_ob_down = (rsi_prev > self.config.RSI_OVERBOUGHT_LEVEL and rsi_current <= self.config.RSI_OVERBOUGHT_LEVEL)
                rsi_crossed_os_up = (rsi_prev < self.config.RSI_OVERSOLD_LEVEL and rsi_current >= self.config.RSI_OVERSOLD_LEVEL)

                # OVERRIDE 1: RSI crossed OVERBOUGHT DOWN → SHORT signal (reversal from overbought)
                if rsi_crossed_ob_down:
                    if decision != "SELL":
                        logger.info(f"🔴 RSI REVERSAL: Crossed {self.config.RSI_OVERBOUGHT_LEVEL} down ({rsi_prev:.1f} → {rsi_current:.1f}) - OVERRIDE to SHORT")
                        decision = "SELL"
                    else:
                        logger.info(f"🔴 RSI REVERSAL: Crossed {self.config.RSI_OVERBOUGHT_LEVEL} down ({rsi_prev:.1f} → {rsi_current:.1f}) - Confirming SHORT signal")

                # OVERRIDE 2: RSI crossed OVERSOLD UP → LONG signal (reversal from oversold)
                elif rsi_crossed_os_up:
                    if decision != "BUY":
                        logger.info(f"🟢 RSI REVERSAL: Crossed {self.config.RSI_OVERSOLD_LEVEL} up ({rsi_prev:.1f} → {rsi_current:.1f}) - OVERRIDE to LONG")
                        decision = "BUY"
                    else:
                        logger.info(f"🟢 RSI REVERSAL: Crossed {self.config.RSI_OVERSOLD_LEVEL} up ({rsi_prev:.1f} → {rsi_current:.1f}) - Confirming LONG signal")

                # FILTER 1: Block LONG if RSI high without reversal signal
                elif decision == "BUY" and rsi_current > self.config.RSI_HIGH_THRESHOLD:
                    logger.warning(f"⚠️ RSI HIGH FILTER: RSI={rsi_current:.1f} >{self.config.RSI_HIGH_THRESHOLD} without {self.config.RSI_OVERSOLD_LEVEL}-cross - BLOCKING LONG (wait for reversal)")
                    decision = "HOLD"

                # FILTER 2: Block SHORT if RSI low without reversal signal
                elif decision == "SELL" and rsi_current < self.config.RSI_LOW_THRESHOLD:
                    logger.warning(f"⚠️ RSI LOW FILTER: RSI={rsi_current:.1f} <{self.config.RSI_LOW_THRESHOLD} without {self.config.RSI_OVERBOUGHT_LEVEL}-cross - BLOCKING SHORT (wait for reversal)")
                    decision = "HOLD"

                # Log filter summary
                if decision != original_decision:
                    logger.info(f"🔄 RSI Filter Applied: {original_decision} → {decision} (RSI: {rsi_current:.1f})")
                else:
                    logger.info(f"✓ RSI Filter: Passed (RSI: {rsi_current:.1f}, no override needed)")

                logger.info(f"🎯 Final Decision (after RSI filter): {decision}")
            else:
                logger.info(f"⏭️  RSI Reversal Filter: DISABLED")

            # Cache decision data for later use
            self.last_decision_data = {
                "decision": decision,
                "proba_buy": float(proba_buy),
                "proba_sell": float(proba_sell),
                "threshold": self.config.PROBABILITY_THRESHOLD,
                "candle_close_time": last_row.name.isoformat() if hasattr(last_row.name, 'isoformat') else str(last_row.name)
            }

            # Cache candle data
            self.last_candle_data = {
                'timestamp': last_row.name.isoformat() if hasattr(last_row.name, 'isoformat') else str(last_row.name),
                'open': float(last_row['open']),
                'high': float(last_row['high']),
                'low': float(last_row['low']),
                'close': float(last_row['close']),
                'volume': float(last_row['volume']),
                'turnover': float(last_row.get('turnover', 0))
            }

            # Log prediction to CSV (for visualization)
            try:
                self.prediction_logger.log_prediction(
                    candle=last_row,
                    buy_prob=proba_buy,
                    sell_prob=proba_sell,
                    threshold=self.config.PROBABILITY_THRESHOLD,
                    min_confidence_ratio=self.config.MIN_CONFIDENCE_RATIO,
                    decision=decision
                )
            except Exception as e:
                logger.warning(f"⚠️  Failed to log prediction: {e}")
            
            # Log candle data (even without position)
            position_data = None
            if self.state:
                position_data = {
                    'side': self.state.get('side'),
                    'entry_price': self.state.get('entry_price'),
                    'sl': self.state.get('last_sl'),
                    'tp': self.state.get('initial_tp'),
                    'pnl_pct': 0,  # Will be calculated in _manage_position
                    'highest_price': self.state.get('highest_price'),
                    'lowest_price': self.state.get('lowest_price')
                }

            self.trade_logger.log_candle(self.last_candle_data, position_data)

            # Extract top indicators if signal
            if decision in ["BUY", "SELL"]:
                features_to_use = self.features_long if decision == "BUY" else self.features_short
                top_indicators = self._extract_top_indicators(last_row, features_to_use, n=15)

                self.last_decision_data['top_indicators'] = top_indicators

            # Cache df for DCA level calculations
            self.last_df = df

            return decision
            
        except Exception as e:
            logger.error(f"Error in get_decision: {e}", exc_info=True)
            return "ERROR"
    
    def _update_tsl(self, position: Dict, current_price: float):
        """Update TSL - ONLY when in profit AND all DCA limit orders filled/expired"""
        side = position.get('side')
        entry = position.get('entryPrice', 0)

        if entry == 0:
            return

        # DCA MODE: TSL completely disabled
        # Reason: Partial TP moves SL to BE+fees, which is safer and simpler
        # than tracking TSL with multiple unfilled orders
        if self.config.LIMIT_ORDER_MODE and self.config.DCA_ENABLED:
            logger.debug(f"TSL disabled for DCA mode (SL managed by partial TP)")
            return

        # SINGLE LIMIT ORDER MODE: Block TSL if waiting for limit order to fill
        if self.config.LIMIT_ORDER_MODE and self.active_limit_orders:
            logger.debug(f"TSL blocked: waiting for limit order to fill/expire")
            return

        # Track highest/lowest
        if side == 'Long':
            highest = self.state.get('highest_price', entry)
            if current_price > highest:
                highest = current_price
                self.state['highest_price'] = highest
                self._save_state()

            # TSL ONLY if in profit
            if highest > entry:
                new_sl = highest * (1 - self.config.TSL_PCT)
                last_sl = self.state.get('last_sl', 0)

                if new_sl > last_sl:
                    # FIX #4: Retry TSL update with exponential backoff
                    logger.info(f"📈 TSL update: {last_sl:.4f} → {new_sl:.4f}")

                    result = self._retry_with_backoff(
                        self.adapter.set_stop_loss,
                        self.config.TICKER, new_sl, "Buy",
                        max_attempts=3  # 3 attempts should be enough for SL update
                    )

                    if result is not None:
                        # Success
                        self.trade_logger.log_event("TSL_UPDATE", {
                            "old_sl": float(last_sl),
                            "new_sl": float(new_sl),
                            "current_price": float(current_price),
                            "highest_price": float(highest),
                            "reason": "in_profit",
                            "candle": self.last_candle_data
                        })

                        self.state['last_sl'] = new_sl
                        self._save_state()
                    else:
                        # All retries failed - CRITICAL
                        logger.error(f"❌ CRITICAL: TSL update failed after 3 retries - position may be unprotected!")
                        self.trade_logger.log_event("ERROR", {
                            "type": "TSL_UPDATE_FAILED_ALL_RETRIES",
                            "old_sl": float(last_sl),
                            "intended_new_sl": float(new_sl),
                            "error": "All retry attempts exhausted"
                        })
        
        elif side == 'Short':
            lowest = self.state.get('lowest_price', entry)
            if current_price < lowest:
                lowest = current_price
                self.state['lowest_price'] = lowest
                self._save_state()
            
            # TSL ONLY if in profit
            if lowest < entry:
                new_sl = lowest * (1 + self.config.TSL_PCT)
                last_sl = self.state.get('last_sl', 999999)
                
                if new_sl < last_sl:
                    # FIX #4: Retry TSL update with exponential backoff
                    logger.info(f"📉 TSL update: {last_sl:.4f} → {new_sl:.4f}")

                    result = self._retry_with_backoff(
                        self.adapter.set_stop_loss,
                        self.config.TICKER, new_sl, "Sell",
                        max_attempts=3
                    )

                    if result is not None:
                        # Success
                        self.trade_logger.log_event("TSL_UPDATE", {
                            "old_sl": float(last_sl),
                            "new_sl": float(new_sl),
                            "current_price": float(current_price),
                            "lowest_price": float(lowest),
                            "reason": "in_profit",
                            "candle": self.last_candle_data
                        })

                        self.state['last_sl'] = new_sl
                        self._save_state()
                    else:
                        # All retries failed - CRITICAL
                        logger.error(f"❌ CRITICAL: TSL update failed after 3 retries - position may be unprotected!")
                        self.trade_logger.log_event("ERROR", {
                            "type": "TSL_UPDATE_FAILED_ALL_RETRIES",
                            "old_sl": float(last_sl),
                            "intended_new_sl": float(new_sl),
                            "error": "All retry attempts exhausted"
                        })

    def _check_profit_protection(self, position: Dict, current_price: float):
        """
        Protect profit by moving SL to breakeven if profit peaks above threshold
        but starts declining before hitting partial TP.

        This prevents situations where price reaches good profit (e.g., 0.32%)
        but reverses before hitting partial TP, potentially resulting in a loss.
        """
        if not self.config.PROTECT_PROFIT_ENABLED:
            return

        # Only activate if partial TP not yet taken (old mechanism)
        if self.config.PARTIAL_TP_ENABLED and self.state.get('partial_tp_taken', False):
            return

        # Only activate if no dynamic TP levels taken yet (new mechanism)
        if self.config.DYNAMIC_TP_ENABLED and self.state.get('dynamic_tp_levels_taken', 0) > 0:
            return

        side = position.get('side')
        entry = position.get('entryPrice', 0)

        if entry == 0:
            return

        # Calculate current profit percentage
        if side == 'Long':
            profit_pct = (current_price / entry - 1) * 100
        else:
            profit_pct = (entry / current_price - 1) * 100

        # Track highest profit achieved (initialize if not exists)
        if 'highest_profit_pct' not in self.state:
            self.state['highest_profit_pct'] = profit_pct

        # Update highest profit if current is higher
        if profit_pct > self.state['highest_profit_pct']:
            self.state['highest_profit_pct'] = profit_pct
            self._save_state()

        # Protection thresholds
        PROFIT_THRESHOLD = 0.25  # Minimum profit to activate protection (0.25%)
        PROFIT_DECLINE_TRIGGER = 0.25  # If profit drops to this level, move SL to BE (0.25%)

        # If highest profit was above threshold (e.g., 0.32%) but now declined to trigger level (0.25%)
        if (self.state['highest_profit_pct'] > PROFIT_THRESHOLD and
            profit_pct <= PROFIT_DECLINE_TRIGGER):

            # Move SL to breakeven if not already there
            last_sl = self.state.get('last_sl', 0)

            if side == 'Long':
                if last_sl < entry:
                    logger.warning(
                        f"🛡️ PROFIT PROTECTION: Moving SL to breakeven "
                        f"(profit peaked at {self.state['highest_profit_pct']:.2f}%, now at {profit_pct:.2f}%)"
                    )

                    try:
                        self.adapter.set_stop_loss(self.config.TICKER, entry, "Buy")

                        # Log event
                        self.trade_logger.log_event("PROFIT_PROTECTION", {
                            "peak_profit_pct": float(self.state['highest_profit_pct']),
                            "current_profit_pct": float(profit_pct),
                            "old_sl": float(last_sl),
                            "new_sl": float(entry),
                            "current_price": float(current_price),
                            "reason": "profit_declined_before_partial_tp",
                            "candle": self.last_candle_data
                        })

                        self.state['last_sl'] = entry
                        self._save_state()

                        logger.info(f"✓ Profit protection activated: SL moved to {entry:.4f}")

                    except Exception as e:
                        logger.error(f"Failed to activate profit protection: {e}")
                        self.trade_logger.log_event("ERROR", {
                            "type": "PROFIT_PROTECTION_FAILED",
                            "error": str(e)
                        })

            else:  # Short
                if last_sl > entry or last_sl == 0:
                    logger.warning(
                        f"🛡️ PROFIT PROTECTION: Moving SL to breakeven "
                        f"(profit peaked at {self.state['highest_profit_pct']:.2f}%, now at {profit_pct:.2f}%)"
                    )

                    try:
                        self.adapter.set_stop_loss(self.config.TICKER, entry, "Sell")

                        # Log event
                        self.trade_logger.log_event("PROFIT_PROTECTION", {
                            "peak_profit_pct": float(self.state['highest_profit_pct']),
                            "current_profit_pct": float(profit_pct),
                            "old_sl": float(last_sl),
                            "new_sl": float(entry),
                            "current_price": float(current_price),
                            "reason": "profit_declined_before_partial_tp",
                            "candle": self.last_candle_data
                        })

                        self.state['last_sl'] = entry
                        self._save_state()

                        logger.info(f"✓ Profit protection activated: SL moved to {entry:.4f}")

                    except Exception as e:
                        logger.error(f"Failed to activate profit protection: {e}")
                        self.trade_logger.log_event("ERROR", {
                            "type": "PROFIT_PROTECTION_FAILED",
                            "error": str(e)
                        })

    def _handle_partial_tp(self, position: Dict, current_price: float):
        """Handle partial TP with breakeven (old mechanism) or dynamic TP (new mechanism)"""
        # Validate mutual exclusivity
        if self.config.PARTIAL_TP_ENABLED and self.config.DYNAMIC_TP_ENABLED:
            logger.error("Cannot enable both PARTIAL_TP and DYNAMIC_TP. Choose only one.")
            return
        
        if self.config.TP_PCT <= 0:
            return
        
        # Handle old partial TP mechanism
        if self.config.PARTIAL_TP_ENABLED:
            if self.state.get('partial_tp_taken', False):
                return
            self._handle_old_partial_tp(position, current_price)
        
        # Handle new dynamic TP mechanism
        elif self.config.DYNAMIC_TP_ENABLED:
            self._handle_dynamic_tp(position, current_price)
    
    def _handle_old_partial_tp(self, position: Dict, current_price: float):
        """Handle old partial TP mechanism (50% at halfway to TP)"""
        side = position.get('side')
        entry = position.get('entryPrice', 0)
        size = position.get('size', 0)
        
        if entry == 0 or size == 0:
            return
        
        # Calculate partial TP price
        initial_tp = self.state.get('initial_tp', 0)
        if initial_tp == 0:
            initial_tp = entry * (1 + self.config.TP_PCT) if side == 'Long' else entry * (1 - self.config.TP_PCT)
        
        if side == 'Long':
            partial_tp_price = entry + (initial_tp - entry) / 2
            hit = current_price >= partial_tp_price
        else:
            partial_tp_price = entry - (entry - initial_tp) / 2
            hit = current_price <= partial_tp_price
        
        if not hit:
            return

        # Execute partial close
        logger.warning(f"🎯 PARTIAL TP HIT at {current_price:.4f}")

        # Use adapter's round_qty for ticker-specific precision
        qty = self.adapter.round_qty(self.config.TICKER, size / 2)

        # Validate against minimum order quantity
        min_qty = self.adapter.get_min_order_qty(self.config.TICKER)
        if qty < min_qty:
            logger.warning(f"Calculated qty {qty:.4f} < minOrderQty {min_qty:.4f}, closing entire position instead")
            qty = size

        if qty <= 0:
            return

        reduce_side = "Sell" if side == 'Long' else "Buy"

        try:
            # CRITICAL: Cancel any unfilled limit orders (DCA mode)
            # This prevents unfilled orders from executing after partial TP hit
            self._cancel_unfilled_limit_orders(reason="partial_tp_hit")

            # Pass position_side for correct positionIdx in hedge mode
            self.adapter.market_close(self.config.TICKER, reduce_side, qty, position_side=side)
            time.sleep(2)

            # Get actual remaining size after partial close
            updated_position = self.adapter.get_position(self.config.TICKER)
            actual_remaining_size = updated_position.get('size', 0) if updated_position else 0
            actual_qty_closed = size - actual_remaining_size

            logger.info(f"Partial close: requested={qty}, actual_closed={actual_qty_closed}, remaining={actual_remaining_size}")

            # Calculate partial P&L based on actual closed quantity
            if side == 'Long':
                pnl_pct = (current_price / entry - 1) * 100 if entry > 0 else 0
            else:
                pnl_pct = (entry / current_price - 1) * 100 if (entry > 0 and current_price > 0) else 0

            # FIX #6: Guard against division by zero
            # FIX #7: Use initial_size from state (works with both fixed and percentage-based sizing)
            initial_size = self.state.get('initial_size', self.config.TRADE_SIZE_USD)
            if size > 0 and initial_size > 0:
                pnl_usd = (initial_size * pnl_pct / 100) * (actual_qty_closed / size)
            else:
                logger.error(f"❌ CRITICAL: size=0 or initial_size=0 during PnL calculation (partial TP)")
                pnl_usd = 0

            # Calculate SL at BE + fees (0.06% for maker+taker)
            # This ensures we don't lose money on remaining position
            FEE_PCT = 0.0006  # 0.06% (conservative estimate for round-trip)
            if side == 'Long':
                new_sl = entry * (1 + FEE_PCT)
            else:
                new_sl = entry * (1 - FEE_PCT)

            # Log partial TP event with ACTUAL quantities
            self.trade_logger.log_event("PARTIAL_TP", {
                "trigger_price": float(partial_tp_price),
                "exit_price": float(current_price),
                "quantity_requested": float(qty),
                "quantity_closed": float(actual_qty_closed),
                "quantity_remaining": float(actual_remaining_size),
                "pnl_pct": float(pnl_pct),
                "pnl_usd": float(pnl_usd),
                "new_sl": float(new_sl),
                "reason": "breakeven_plus_fees",
                "candle": self.last_candle_data
            })

            # Move SL to BE + fees
            last_sl = self.state.get('last_sl', 0)
            should_update_sl = False

            if side == 'Long':
                # For Long: move SL to BE+fees if it was below
                if last_sl < new_sl:
                    should_update_sl = True
                    logger.info(f"TSL was below BE+fees ({last_sl:.4f}), moving to BE+fees: {new_sl:.4f}")
            else:  # Short
                # For Short: move SL to BE+fees if it was above
                if last_sl > new_sl or last_sl == 0:
                    should_update_sl = True
                    logger.info(f"TSL was above BE+fees ({last_sl:.4f}), moving to BE+fees: {new_sl:.4f}")

            if should_update_sl:
                logger.info(f"Moving SL to BE + fees: {new_sl:.4f} (entry: {entry:.4f} + {FEE_PCT*100:.2f}%)")
                if side == 'Long':
                    self.adapter.set_stop_loss(self.config.TICKER, new_sl, "Buy")
                else:
                    self.adapter.set_stop_loss(self.config.TICKER, new_sl, "Sell")

                self.state['last_sl'] = new_sl

            self.state['partial_tp_taken'] = True
            self._save_state()

            logger.info(f"✓ Partial TP executed. Remaining: {actual_remaining_size} (requested close: {qty}, actual close: {actual_qty_closed})")
            
        except Exception as e:
            logger.error(f"Partial TP failed: {e}")
            self.trade_logger.log_event("ERROR", {
                "type": "PARTIAL_TP_FAILED",
                "error": str(e)
            })
    
    def _handle_dynamic_tp(self, position: Dict, current_price: float):
        """Handle new dynamic TP mechanism (25% at each of 4 levels: 25%, 50%, 75%, 100%)"""
        side = position.get('side')
        entry = position.get('entryPrice', 0)
        size = position.get('size', 0)
        
        if entry == 0 or size == 0:
            return
        
        # Get current level (0-3, representing levels 1-4)
        dynamic_tp_levels_taken = self.state.get('dynamic_tp_levels_taken', 0)
        
        if dynamic_tp_levels_taken >= 4:
            return  # All levels taken
        
        # Calculate initial TP
        initial_tp = self.state.get('initial_tp', 0)
        if initial_tp == 0:
            initial_tp = entry * (1 + self.config.TP_PCT) if side == 'Long' else entry * (1 - self.config.TP_PCT)
        
        # Calculate distance from entry to TP
        if side == 'Long':
            distance = initial_tp - entry
        else:
            distance = entry - initial_tp
        
        # Calculate next level (1-based: 1, 2, 3, 4)
        next_level = dynamic_tp_levels_taken + 1
        level_pct = next_level * 0.25  # 0.25, 0.50, 0.75, 1.0
        
        # Calculate price for this level
        if side == 'Long':
            level_price = entry + (distance * level_pct)
            hit = current_price >= level_price
        else:
            level_price = entry - (distance * level_pct)
            hit = current_price <= level_price
        
        if not hit:
            return

        # Execute partial close
        initial_size = self.state.get('initial_size', size)

        # CRITICAL FIX: Level 4 must close ALL remaining to avoid rounding errors
        if next_level == 4:
            # CRITICAL: Cancel unfilled limit orders before closing position
            self._cancel_unfilled_limit_orders(reason="dynamic_tp_level_4_closing_position")

            # Close entire remaining position
            qty = size
            logger.warning(f"🎯 DYNAMIC TP LEVEL 4 HIT (100%) - CLOSING ALL REMAINING at {current_price:.4f}")
        else:
            # Levels 1-3: close 25% of initial size with proper rounding
            qty = self.adapter.round_qty(self.config.TICKER, initial_size * 0.25)

            # Validate against minimum order quantity
            min_qty = self.adapter.get_min_order_qty(self.config.TICKER)
            if qty < min_qty:
                logger.warning(f"Calculated qty {qty:.4f} < minOrderQty {min_qty:.4f}, closing entire position instead")
                qty = size

            # Ensure we don't try to close more than current position
            if qty > size:
                logger.warning(f"Calculated qty {qty:.4f} > current size {size:.4f}, adjusting to {size:.4f}")
                qty = size

            logger.warning(f"🎯 DYNAMIC TP LEVEL {next_level} HIT ({int(level_pct*100)}%) at {current_price:.4f}")

        if qty <= 0:
            logger.error(f"Invalid qty={qty} for Dynamic TP Level {next_level}, skipping")
            return

        min_qty = self.adapter.get_min_order_qty(self.config.TICKER)
        if next_level < 4 and qty < size and qty < min_qty * 2:
            logger.warning(f"⚠️ Skipping Dynamic TP Level {next_level}: qty={qty:.4f} too small for partial close (< {min_qty * 2:.4f}). Will wait for higher levels.")
            return

        reduce_side = "Sell" if side == 'Long' else "Buy"

        try:
            # Pass position_side for correct positionIdx in hedge mode
            self.adapter.market_close(self.config.TICKER, reduce_side, qty, position_side=side)
            time.sleep(2)

            # Get actual remaining size after partial close
            updated_position = self.adapter.get_position(self.config.TICKER)
            actual_remaining_size = updated_position.get('size', 0) if updated_position else 0
            actual_qty_closed = size - actual_remaining_size

            logger.info(f"Dynamic TP L{next_level}: requested={qty}, actual_closed={actual_qty_closed}, remaining={actual_remaining_size}")

            # Calculate partial P&L based on actual closed quantity
            if side == 'Long':
                pnl_pct = (current_price / entry - 1) * 100 if entry > 0 else 0
            else:
                pnl_pct = (entry / current_price - 1) * 100 if (entry > 0 and current_price > 0) else 0

            # FIX #6: Guard against division by zero
            # FIX #7: Use initial_size from state (already retrieved above)
            trade_size_usd = self.state.get('initial_size', self.config.TRADE_SIZE_USD)
            if initial_size > 0 and trade_size_usd > 0:
                pnl_usd = (trade_size_usd * pnl_pct / 100) * (actual_qty_closed / initial_size)
            else:
                logger.error(f"❌ CRITICAL: initial_size=0 during PnL calculation (dynamic TP L{next_level})")
                pnl_usd = 0

            # Log dynamic TP event with ACTUAL quantities
            self.trade_logger.log_event(f"DYNAMIC_TP_L{next_level}", {
                "level": next_level,
                "level_percentage": int(level_pct * 100),
                "trigger_price": float(level_price),
                "exit_price": float(current_price),
                "quantity_requested": float(qty),
                "quantity_closed": float(actual_qty_closed),
                "quantity_remaining": float(actual_remaining_size),
                "pnl_pct": float(pnl_pct),
                "pnl_usd": float(pnl_usd),
                "candle": self.last_candle_data
            })
            
            # Update state
            self.state['dynamic_tp_levels_taken'] = next_level
            
            # Move SL to breakeven only after first level
            if next_level == 1:
                last_sl = self.state.get('last_sl', 0)
                should_update_sl = False
                
                if side == 'Long':
                    if last_sl < entry:
                        should_update_sl = True
                        logger.info(f"TSL was below breakeven ({last_sl:.4f}), moving to breakeven: {entry:.4f}")
                else:  # Short
                    if last_sl > entry or last_sl == 0:
                        should_update_sl = True
                        logger.info(f"TSL was above breakeven ({last_sl:.4f}), moving to breakeven: {entry:.4f}")
                
                if should_update_sl:
                    logger.info(f"Moving SL to breakeven: {entry:.4f}")
                    if side == 'Long':
                        self.adapter.set_stop_loss(self.config.TICKER, entry, "Buy")
                    else:
                        self.adapter.set_stop_loss(self.config.TICKER, entry, "Sell")
                    
                    self.state['last_sl'] = entry
            
            self._save_state()

            logger.info(f"✓ Dynamic TP Level {next_level} executed. Remaining: {actual_remaining_size} (requested close: {qty}, actual close: {actual_qty_closed})")
            
        except Exception as e:
            logger.error(f"Dynamic TP Level {next_level} failed: {e}")
            self.trade_logger.log_event("ERROR", {
                "type": f"DYNAMIC_TP_L{next_level}_FAILED",
                "level": next_level,
                "error": str(e)
            })
    
    def _manage_position(self) -> bool:
        """Manage open position. Returns True if position exists."""
        try:
            position = self.adapter.get_position(self.config.TICKER)
            
            if not position or position.get('size', 0) == 0:
                # FIX #1: CRITICAL RACE CONDITION FIX
                # Cancel ONLY unfilled orders that are ACTUALLY open on Bybit
                # This prevents attempting to cancel already-filled orders (race condition)
                if self.active_limit_orders:
                    try:
                        # Get current open orders from Bybit (source of truth)
                        open_orders_on_bybit = self.adapter.get_open_orders(self.config.TICKER)
                        open_order_ids = {o['orderId'] for o in open_orders_on_bybit}

                        # Filter: only cancel orders that are BOTH in our tracking AND open on Bybit
                        orders_to_cancel = [
                            o for o in self.active_limit_orders
                            if o['order_id'] in open_order_ids
                        ]

                        if orders_to_cancel:
                            logger.info(f"Position closed - cancelling {len(orders_to_cancel)}/{len(self.active_limit_orders)} unfilled orders")
                            self._cancel_unfilled_limit_orders_safe(orders_to_cancel, reason="position_closed_sl_or_tp")
                        else:
                            logger.info(f"Position closed - no unfilled orders to cancel (all {len(self.active_limit_orders)} already filled/cancelled)")
                            # Clear tracking since all orders processed
                            self.active_limit_orders.clear()

                    except Exception as e:
                        logger.error(f"Failed to check open orders before cancellation: {e}")
                        # Fallback: try to cancel all tracked orders (may fail for already-filled)
                        self._cancel_unfilled_limit_orders(reason="position_closed_sl_or_tp")

                # Position was closed - end trade logging if active
                if self.state and self.trade_logger.current_trade:
                    entry = self.state.get('entry_price', 0)
                    current_price = self.adapter.latest_price(self.config.TICKER)

                    # Calculate final P&L
                    side = self.state.get('side')
                    if side == 'Long':
                        pnl_pct = (current_price / entry - 1) * 100 if entry > 0 else 0
                    else:
                        pnl_pct = (entry / current_price - 1) * 100 if entry > 0 else 0

                    # FIX #7: Use initial_size from state (works with both fixed and percentage-based sizing)
                    trade_size_usd = self.state.get('initial_size', self.config.TRADE_SIZE_USD)
                    pnl_usd = trade_size_usd * pnl_pct / 100

                    # Log exit
                    self.trade_logger.log_event("EXIT", {
                        "trigger": "SL_or_TP",
                        "exit_price": float(current_price),
                        "pnl_pct": float(pnl_pct),
                        "pnl_usd": float(pnl_usd),
                        "candle": self.last_candle_data
                    })

                    # Calculate duration
                    duration = (datetime.now() - datetime.fromisoformat(
                        self.trade_logger.current_trade['start_time']
                    )).total_seconds()

                    # Determine exit reason
                    exit_reason = TradeMetricsCalculator.determine_exit_reason(
                        position_closed=True,
                        sl_hit=True,  # Position closed by SL or TP
                        tp_hit=True,
                        partial_tp_taken=self.state.get('partial_tp_taken', False),
                        dynamic_tp_levels_taken=self.state.get('dynamic_tp_levels_taken', 0)
                    )

                    # Calculate complete trade summary with all metrics
                    trade_summary = TradeMetricsCalculator.calculate_trade_summary(
                        side=side,
                        entry_price=entry,
                        exit_price=current_price,
                        quantity=self.state.get('original_qty', 0),
                        highest_price=self.state.get('highest_price', entry),
                        lowest_price=self.state.get('lowest_price', entry),
                        leverage=self.config.LEVERAGE,
                        entry_is_limit=self.config.LIMIT_ORDER_MODE,
                        exit_is_limit=False,  # Exit always market order
                        partial_tp_value_usd=trade_size_usd * 0.5 if self.state.get('partial_tp_taken') else 0.0,
                        duration_seconds=duration,
                        exit_reason=exit_reason,
                        initial_sl=self.state.get('initial_sl'),
                        initial_tp=self.state.get('initial_tp'),
                        final_sl=self.state.get('last_sl'),
                        final_tp=self.state.get('initial_tp')
                    )

                    # End trade with complete metrics
                    self.trade_logger.end_trade(trade_summary)

                    # COOLDOWN: Always activate cooldown after ANY closed position
                    # This prevents bot from immediately re-entering based on same 15m candle
                    # (e.g., if user manually closes position or bot closes at TP/SL)
                    cooldown_until = datetime.now()
                    result_emoji = "❌" if pnl_usd < 0 else "✅"
                    logger.warning(f"⏰ {result_emoji} POSITION CLOSED: PnL={pnl_usd:.2f} USD ({pnl_pct:.2f}%)")
                    logger.warning(f"   Activating {self.config.COOLDOWN_AFTER_LOSS_MINUTES}min cooldown before next position")

                    # Save cooldown timestamp in a new state (separate from position state)
                    self.state = {
                        'last_position_closed_at': cooldown_until.isoformat(),
                        'cooldown_minutes': self.config.COOLDOWN_AFTER_LOSS_MINUTES,
                        'last_pnl_usd': pnl_usd  # For logging purposes
                    }
                    self._save_state()
                    return False

                if self.state:
                    logger.info("Position closed. Resetting state.")
                    self.state = {}
                    self._save_state()
                return False
            
            side = position.get('side')
            entry = position.get('entryPrice', 0)
            size = position.get('size', 0)
            current_price = self.adapter.latest_price(self.config.TICKER)

            # FIX: Recover state from API if position exists but state is empty
            # This handles cases where network errors prevented state from being saved during position opening
            if not self.state or 'entry_price' not in self.state:
                logger.warning("="*60)
                logger.warning(f"⚠️  STATE RECOVERY: Position exists but state is empty!")
                logger.warning(f"    {side} position | Entry: {entry:.4f} | Size: {size}")
                logger.warning(f"    Recovering state from Bybit API...")
                logger.warning("="*60)

                stop_loss = position.get('stopLoss', 0.0)
                take_profit = position.get('takeProfit', 0.0)

                # Reconstruct minimal state
                self.state = {
                    'side': side,
                    'entry_price': entry,
                    'initial_tp': take_profit if take_profit > 0 else (entry * (1 + self.config.TP_PCT) if side == "Long" else entry * (1 - self.config.TP_PCT)),
                    'last_sl': stop_loss if stop_loss > 0 else (entry * (1 - self.config.TSL_PCT) if side == "Long" else entry * (1 + self.config.TSL_PCT)),
                    'partial_tp_taken': False,
                    'dynamic_tp_levels_taken': 0,
                    'highest_price': entry if side == "Long" else current_price,
                    'lowest_price': entry if side == "Short" else current_price,
                    'original_qty': size,
                    'initial_size': entry * size  # Approximate trade size
                }
                self._save_state()

                # Start trade logger if not active
                if not self.trade_logger.current_trade:
                    self.trade_logger.start_trade(
                        ticker=self.config.TICKER,
                        side=side,
                        decision_data={
                            'recovered_from_api_during_manage': True,
                            'entry_price': entry,
                            'size': size,
                            'note': 'State recovered due to network error during position opening'
                        }
                    )
                    logger.warning(f"    ✓ Trade logger started for recovered {side} position")

                logger.warning("    ✓ State recovery complete")
                logger.warning("="*60)

            # DUST POSITION CLEANUP: Check if position is too small to close normally
            min_qty = self.adapter.get_min_order_qty(self.config.TICKER)
            if size > 0 and size < min_qty:
                logger.warning(
                    f"DUST POSITION DETECTED: {size:.4f} < minOrderQty {min_qty:.4f}. "
                    f"Attempting to close entire position."
                )
                try:
                    reduce_side = "Sell" if side == 'Long' else "Buy"
                    # Try to close with actual position size
                    # Bybit may accept closing existing position even if size < minOrderQty
                    self.adapter.market_close(self.config.TICKER, reduce_side, size, position_side=side)
                    time.sleep(2)

                    # Log dust cleanup
                    if self.trade_logger.current_trade:
                        self.trade_logger.log_event("DUST_CLEANUP", {
                            "size": float(size),
                            "min_qty": float(min_qty),
                            "side": side,
                            "price": float(current_price),
                            "reason": "position_too_small_after_partial_closes"
                        })

                    logger.info(f"Dust position cleanup attempted for {size:.4f} {self.config.TICKER}")
                except Exception as e:
                    logger.error(f"Failed to close dust position: {e}")
                    # Continue execution - don't crash bot

            if current_price == 0:
                return True
            
            # Calculate P&L
            pnl_pct = ((current_price / entry - 1) if side == 'Long' else (entry / current_price - 1)) * 100
            
            logger.info(
                f"📊 {side} | Size: {size} | Entry: {entry:.4f} | "
                f"Current: {current_price:.4f} | P&L: {pnl_pct:+.2f}%"
            )
            
            # Update candle log with position data
            position_data = {
                'side': side,
                'entry_price': entry,
                'sl': self.state.get('last_sl'),
                'tp': self.state.get('initial_tp'),
                'pnl_pct': pnl_pct,
                'highest_price': self.state.get('highest_price'),
                'lowest_price': self.state.get('lowest_price')
            }
            
            self.trade_logger.log_candle(self.last_candle_data, position_data)

            # Check profit protection (BEFORE partial TP)
            self._check_profit_protection(position, current_price)

            # Handle partial TP
            self._handle_partial_tp(position, current_price)

            # Update TSL
            self._update_tsl(position, current_price)
            
            return True
            
        except Exception as e:
            logger.error(f"Error managing position: {e}", exc_info=True)
            self.trade_logger.log_event("ERROR", {
                "type": "MANAGE_POSITION_ERROR",
                "error": str(e)
            })

            # FIX #7: Don't assume position exists on exception
            # Re-check position to return correct value
            try:
                position = self.adapter.get_position(self.config.TICKER)
                has_position = position and position.get('size', 0) > 0
                logger.info(f"Exception recovery: position exists = {has_position}")
                return has_position
            except (ConnectionError, TimeoutError, KeyError, ValueError) as e:
                # If we can't even check, safer to assume no position
                logger.warning(f"Cannot verify position after exception ({type(e).__name__}: {e}) - assuming no position")
                return False
    
    def _open_position(self, decision: str):
        """Open new position with verification and logging"""
        try:
            current_price = self.adapter.latest_price(self.config.TICKER)
            if current_price == 0:
                return

            side_str = "Buy" if decision == "BUY" else "Sell"
            position_type = "LONG" if decision == "BUY" else "SHORT"

            # ========== CHECK MAXIMUM POSITION LIMIT (if enabled) ==========
            # Verify that opening this position won't exceed configured limit
            if self.config.MAX_POSITIONS > 0:
                try:
                    current_positions = self.adapter.get_active_positions_count()
                    if current_positions >= self.config.MAX_POSITIONS:
                        logger.error("="*60)
                        logger.error(f"❌ POSITION LIMIT REACHED: Cannot open new {position_type} position")
                        logger.error(f"   Current Active Positions: {current_positions}")
                        logger.error(f"   Maximum Allowed: {self.config.MAX_POSITIONS}")
                        logger.error(f"   Signal: {decision} @ ${current_price:.2f}")
                        logger.error(f"   Wait for existing positions to close before opening new ones")
                        logger.error("="*60)
                        return
                    else:
                        logger.info("="*60)
                        logger.info(f"✓ POSITION LIMIT CHECK PASSED")
                        logger.info(f"   Current Positions: {current_positions}/{self.config.MAX_POSITIONS}")
                        logger.info(f"   Opening new {position_type} position...")
                        logger.info("="*60)
                except Exception as e:
                    logger.warning("="*60)
                    logger.warning(f"⚠️  Position limit check failed (API error): {e}")
                    logger.warning("   Continuing with order placement anyway (use with caution)")
                    logger.warning("="*60)

            # ========== CALCULATE POSITION SIZE (Percentage-based or Fixed USD) ==========
            # Determine trade size based on configuration:
            # - If TRADE_SIZE_PCT > 0: use percentage of available balance (dynamic sizing)
            # - Otherwise: use fixed TRADE_SIZE_USD (legacy mode)
            try:
                available_balance_usd = self.adapter.get_available_balance_usd()

                if self.config.TRADE_SIZE_PCT > 0:
                    # PERCENTAGE-BASED SIZING: Calculate trade size as % of available balance WITH leverage
                    buying_power = available_balance_usd * self.config.LEVERAGE
                    trade_size_usd = buying_power * self.config.TRADE_SIZE_PCT
                    qty = round(trade_size_usd / current_price, 2)

                    logger.info("="*60)
                    logger.info(f"📊 PERCENTAGE-BASED POSITION SIZING")
                    logger.info(f"   Available Balance: ${available_balance_usd:.2f}")
                    logger.info(f"   Leverage: {self.config.LEVERAGE}x")
                    logger.info(f"   Buying Power: ${buying_power:.2f}")
                    logger.info(f"   Risk Percentage: {self.config.TRADE_SIZE_PCT * 100:.2f}%")
                    logger.info(f"   Calculated Trade Size: ${trade_size_usd:.2f}")
                    logger.info(f"   Position Quantity: {qty:.4f} @ ${current_price:.2f}")
                    logger.info("="*60)
                else:
                    # FIXED USD SIZING: Use legacy fixed trade size
                    trade_size_usd = self.config.TRADE_SIZE_USD
                    qty = round(trade_size_usd / current_price, 2)

                    logger.info(f"💵 Fixed trade size: ${trade_size_usd:.2f} → {qty:.4f} units @ ${current_price:.2f}")

                if qty <= 0:
                    logger.error(f"❌ Invalid quantity: {qty} (trade_size=${trade_size_usd:.2f}, price=${current_price:.2f})")
                    return

                # Balance check: verify sufficient margin
                required_margin = trade_size_usd / self.config.LEVERAGE

                if available_balance_usd < required_margin:
                    logger.error("="*60)
                    logger.error(f"❌ INSUFFICIENT BALANCE: Cannot open {position_type} position")
                    logger.error(f"   Required Margin: ${required_margin:.2f} (${trade_size_usd:.2f} / {self.config.LEVERAGE}x leverage)")
                    logger.error(f"   Available Balance: ${available_balance_usd:.2f}")
                    logger.error(f"   Shortfall: ${required_margin - available_balance_usd:.2f}")
                    logger.error("   Please deposit funds or reduce trade size")
                    logger.error("="*60)
                    return
                else:
                    logger.info(f"✓ Balance check passed: ${available_balance_usd:.2f} available (need ${required_margin:.2f})")

            except Exception as e:
                logger.warning(f"⚠️  Balance check/sizing failed (API error): {e}")
                logger.warning("   Falling back to fixed TRADE_SIZE_USD mode")
                # Fallback to fixed sizing if balance API fails
                trade_size_usd = self.config.TRADE_SIZE_USD
                qty = round(trade_size_usd / current_price, 2)
                if qty <= 0:
                    return

            # ========== LIMIT ORDER MODE ==========
            if self.config.LIMIT_ORDER_MODE:
                # CRITICAL FIX: Check for existing open orders on Bybit BEFORE placing new ones
                # This prevents duplicate orders if bot restarts and loses self.active_limit_orders from memory
                try:
                    existing_orders = self.adapter.get_open_orders(self.config.TICKER)
                    if existing_orders:
                        logger.warning("="*60)
                        logger.warning(f"⚠️  DUPLICATE ORDER PREVENTION: Found {len(existing_orders)} existing open orders on Bybit")
                        for order in existing_orders[:3]:  # Show first 3
                            logger.warning(f"   Order: {order.get('orderId')} | {order.get('side')} @ {order.get('price')}")
                        logger.warning("   Skipping new order placement - please cancel existing orders first")
                        logger.warning("="*60)
                        return
                except Exception as e:
                    logger.error(f"Failed to check existing orders: {e}")
                    # Continue anyway - don't block order placement on API error

                # Start trade logging (will finalize after order fills)
                self.trade_logger.start_trade(
                    ticker=self.config.TICKER,
                    side=position_type,
                    decision_data=self.last_decision_data
                )

                # ========== DCA MODE: Single ATR-based limit order ==========
                if self.config.DCA_ENABLED:
                    # Calculate single ATR-based DCA level
                    dca_levels = self._calculate_dca_levels(decision, current_price, self.last_df)

                    # Validate minimum order quantity
                    min_qty = self.adapter.get_min_order_qty(self.config.TICKER)
                    qty_per_level = self.adapter.round_qty(self.config.TICKER, qty)

                    if qty_per_level < min_qty:
                        logger.error(f"Total qty {qty_per_level:.4f} < minOrderQty {min_qty:.4f}")
                        logger.error("Balance too low for DCA mode. Use regular limit order instead.")
                        return

                    # Single level DCA
                    num_levels = 1
                    dca_price = dca_levels[0]

                    logger.info(f"📊 DCA Mode: Single ATR-based limit order")
                    logger.info(f"   Level: ${dca_price:.4f} ({abs((dca_price/current_price - 1)*100):.2f}%)")
                    logger.info(f"   Quantity: {qty_per_level:.4f}")

                    # Calculate SL/TP for DCA level
                    # SL based on DCA entry price (better than current price)
                    furthest_level = dca_price  # Single level
                    expected_avg_entry = dca_price  # Single level

                    if decision == "BUY":
                        planned_sl = furthest_level * (1 - self.config.TSL_PCT)
                        planned_tp = expected_avg_entry * (1 + self.config.TP_PCT) if self.config.TP_PCT > 0 else 0
                    else:
                        planned_sl = furthest_level * (1 + self.config.TSL_PCT)
                        planned_tp = expected_avg_entry * (1 - self.config.TP_PCT) if self.config.TP_PCT > 0 else 0

                    logger.warning(f"📋 Placing DCA order (ATR-based): {position_type}")
                    logger.warning(f"   Current: {current_price:.4f}")
                    logger.warning(f"   Limit:   {dca_price:.4f} ({abs((dca_price/current_price - 1)*100):.2f}%)")
                    logger.warning(f"   Qty:     {qty_per_level:.4f}")
                    tp_display = f"{planned_tp:.4f}" if planned_tp > 0 else "OFF"
                    logger.warning(f"   🛡️  SL: {planned_sl:.4f} | TP: {tp_display}")

                    # Place single ATR-based DCA limit order with SL/TP protection
                    try:
                        resp = self.adapter.limit_open(
                            self.config.TICKER, side_str, qty_per_level, dca_price,
                            stop_loss=planned_sl,
                            take_profit=planned_tp if planned_tp > 0 else None
                        )

                        order_id = (resp.get("result") or {}).get("orderId")

                        if not order_id:
                            logger.error("Failed to place DCA order - no orderId returned")
                            return

                        # Track active limit order
                        self.active_limit_orders.append({
                            'order_id': order_id,
                            'timestamp': time.time(),
                            'side': position_type,
                            'price': dca_price,
                            'qty': qty_per_level,
                            'decision': decision,
                            'level': 1  # Single level
                        })

                        # Log event
                        self.trade_logger.log_event("DCA_LIMIT_ORDER_PLACED", {
                            "order_id": order_id,
                            "order_type": "LIMIT",
                            "limit_price": float(dca_price),
                            "current_price": float(current_price),
                            "quantity": float(qty_per_level),
                            "side": position_type,
                            "max_wait_seconds": self.config.MAX_WAITING_LIMIT_ORDER,
                            "offset_pct": abs((dca_price / current_price - 1) * 100),
                            "atr_multiplier": self.config.DCA_ATR_MULTIPLIER
                        })

                        logger.info(f"✓ DCA order placed: {order_id} @ {dca_price:.4f} ({abs((dca_price/current_price - 1)*100):.2f}%)")

                    except Exception as e:
                        logger.error(f"Failed to place DCA order: {e}")
                        return

                    # ========== DCA: Store state for tracking ==========
                    # SL/TP embedded in order for immediate protection when filled
                    self.state = {
                        'side': position_type.capitalize(),
                        'dca_mode': True,
                        'dca_level': dca_price,  # Single level
                        'planned_sl': planned_sl,
                        'planned_tp': planned_tp,
                        'decision': decision,
                        'sl_tp_embedded': True
                    }
                    self._save_state()

                    logger.warning(f"✓ DCA order placed | Wait max {self.config.MAX_WAITING_LIMIT_ORDER}s")
                    logger.warning(f"   Entry: {dca_price:.4f}")
                    logger.warning(f"   🛡️  SL/TP protection EMBEDDED in order:")
                    logger.warning(f"      SL: {planned_sl:.4f}")
                    tp_str = f"{planned_tp:.4f}" if planned_tp > 0 else "OFF"
                    logger.warning(f"      TP: {tp_str}")
                    logger.warning(f"   ⚠️  Position will be PROTECTED immediately when order fills!")
                    return  # Exit - will check order status in run_cycle()

                # ========== SINGLE LIMIT ORDER MODE (original logic) ==========
                else:
                    # Calculate limit price with offset
                    if decision == "BUY":
                        # LONG: Buy cheaper (limit below current price)
                        limit_price = current_price * (1 - self.config.LIMIT_OFFSET_PCT)
                    else:
                        # SHORT: Sell higher (limit above current price)
                        limit_price = current_price * (1 + self.config.LIMIT_OFFSET_PCT)

                    # CRITICAL FIX: Calculate SL/TP BEFORE placing order for immediate protection
                    # SL/TP will be embedded in the limit order and activate when order fills
                    if decision == "BUY":
                        planned_sl = limit_price * (1 - self.config.TSL_PCT)
                        planned_tp = limit_price * (1 + self.config.TP_PCT) if self.config.TP_PCT > 0 else 0
                    else:
                        planned_sl = limit_price * (1 + self.config.TSL_PCT)
                        planned_tp = limit_price * (1 - self.config.TP_PCT) if self.config.TP_PCT > 0 else 0

                    logger.warning(f"📋 Placing LIMIT order: {position_type}")
                    logger.warning(f"   Current: {current_price:.4f} | Limit: {limit_price:.4f} | Offset: {self.config.LIMIT_OFFSET_PCT*100:.2f}%")
                    tp_display = f"{planned_tp:.4f}" if planned_tp > 0 else "OFF"
                    logger.warning(f"   🛡️  Order will have SL: {planned_sl:.4f} | TP: {tp_display}")

                    # Place limit order WITH SL/TP protection
                    resp = self.adapter.limit_open(
                        self.config.TICKER, side_str, qty, limit_price,
                        stop_loss=planned_sl,
                        take_profit=planned_tp if planned_tp > 0 else None
                    )
                    order_id = (resp.get("result") or {}).get("orderId")

                    if not order_id:
                        logger.error("Failed to place limit order - no orderId returned")
                        return

                    # Track active limit order
                    self.active_limit_orders.append({
                        'order_id': order_id,
                        'timestamp': time.time(),
                        'side': position_type,
                        'price': limit_price,
                        'qty': qty,
                        'decision': decision,
                        'level': 1  # Single order = level 1
                    })

                    # Log event
                    self.trade_logger.log_event("LIMIT_ORDER_PLACED", {
                        "order_id": order_id,
                        "order_type": "LIMIT",
                        "limit_price": float(limit_price),
                        "current_price": float(current_price),
                        "quantity": float(qty),
                        "side": position_type,
                        "max_wait_seconds": self.config.MAX_WAITING_LIMIT_ORDER,
                        "sl_price": float(planned_sl),
                        "tp_price": float(planned_tp) if planned_tp > 0 else None
                    })

                    logger.info(f"✓ Limit order placed: {order_id} | Wait max {self.config.MAX_WAITING_LIMIT_ORDER}s")
                    logger.info(f"   🛡️  Position will be PROTECTED when order fills: SL={planned_sl:.4f} | TP={tp_display}")
                    return  # Exit - will check order status in run_cycle()

            # ========== MARKET ORDER MODE (original logic) ==========
            logger.warning(f"🚀 Opening {position_type}: {qty} @ ~{current_price:.4f}")

            # Start trade logging
            self.trade_logger.start_trade(
                ticker=self.config.TICKER,
                side=position_type,
                decision_data=self.last_decision_data
            )

            # Execute
            self.adapter.market_open(self.config.TICKER, side_str, qty)
            time.sleep(self.config.RETRY_DELAY)
            
            # Verify and get actual entry
            position = self.adapter.get_position(self.config.TICKER)
            actual_entry = position.get('entryPrice', current_price) if position else current_price
            
            logger.info(f"✓ Entry verified: {actual_entry:.4f}")
            
            # Calculate SL/TP
            if decision == "BUY":
                sl = actual_entry * (1 - self.config.TSL_PCT)
                tp = actual_entry * (1 + self.config.TP_PCT) if self.config.TP_PCT > 0 else 0
                highest = actual_entry
                lowest = 999999
            else:
                sl = actual_entry * (1 + self.config.TSL_PCT)
                tp = actual_entry * (1 - self.config.TP_PCT) if self.config.TP_PCT > 0 else 0
                highest = 999999
                lowest = actual_entry
            
            # Set SL - use position side (not SL order side) for correct positionIdx in hedge mode
            position_side = side_str  # "Buy" for LONG, "Sell" for SHORT
            self.adapter.set_stop_loss(self.config.TICKER, sl, position_side)
            
            # Set TP - use position side (not TP order side) for correct positionIdx in hedge mode
            if tp > 0:
                self.adapter.set_take_profit(self.config.TICKER, tp, position_side)
            
            # Log entry event
            self.trade_logger.log_event("ENTRY", {
                "order_type": "MARKET",
                "entry_price": float(actual_entry),
                "quantity": float(qty),
                "sl_price": float(sl),
                "tp_price": float(tp) if tp > 0 else None,
                "leverage": self.config.LEVERAGE,
                "candle": self.last_candle_data
            })
            
            # Log indicators
            if 'top_indicators' in self.last_decision_data:
                self.trade_logger.log_indicators(
                    indicators=self.last_decision_data['top_indicators'],
                    model_probas={
                        'proba_buy': self.last_decision_data['proba_buy'],
                        'proba_sell': self.last_decision_data['proba_sell']
                    }
                )
            
            # Save state
            self.state = {
                'side': position_type.capitalize(),
                'entry_price': actual_entry,
                'initial_tp': tp,
                'initial_sl': sl,  # Store initial SL for trade metrics
                'original_qty': qty,
                'initial_size': trade_size_usd,  # Actual trade size (USD) used for this position
                'partial_tp_taken': False,
                'dynamic_tp_levels_taken': 0,  # For dynamic TP mechanism
                'last_sl': sl,
                'highest_price': highest,
                'lowest_price': lowest
            }
            self._save_state()
            
            tp_display = f"{tp:.4f}" if tp > 0 else "OFF"
            logger.warning(f"✓ Position opened | SL: {sl:.4f} | TP: {tp_display}")
            
        except Exception as e:
            logger.error(f"Failed to open position: {e}", exc_info=True)
            if self.trade_logger.current_trade:
                self.trade_logger.log_event("ERROR", {
                    "type": "ENTRY_FAILED",
                    "error": str(e)
                })

    def _check_limit_order_status(self) -> bool:
        """
        Check status of active limit orders (supports both single and DCA mode).

        Returns:
            True if orders are being processed (still waiting), False if all completed/cancelled
        """
        if not self.active_limit_orders:
            return False

        try:
            # Get list of open (unfilled) orders from Bybit
            open_orders = self.adapter.get_open_orders(self.config.TICKER)
            open_order_ids = {order['orderId'] for order in open_orders}

            # Check which orders have been filled
            filled_orders = []
            remaining_orders = []

            for order in self.active_limit_orders:
                order_id = order['order_id']
                elapsed = time.time() - order['timestamp']

                # Check if order is still open on Bybit
                if order_id not in open_order_ids:
                    # Order was filled!
                    filled_orders.append(order)
                    logger.warning("="*60)
                    logger.warning(f"✅ LIMIT ORDER FILLED: {order_id}")
                    if self.config.DCA_ENABLED:
                        logger.warning(f"   Level: {order['level']}/3 | Price: {order['price']:.4f} | Qty: {order['qty']:.4f}")
                    else:
                        logger.warning(f"   Price: {order['price']:.4f} | Qty: {order['qty']:.4f}")
                    logger.warning("="*60)
                else:
                    # Order still open - check timeout
                    if elapsed > self.config.MAX_WAITING_LIMIT_ORDER:
                        logger.warning("="*60)
                        logger.warning(f"⏰ LIMIT ORDER TIMEOUT: {order_id}")
                        logger.warning(f"   Elapsed: {elapsed:.0f}s / {self.config.MAX_WAITING_LIMIT_ORDER}s")
                        logger.warning(f"   Cancelling order...")
                        logger.warning("="*60)

                        # Cancel order
                        try:
                            self.adapter.cancel_order(self.config.TICKER, order_id)
                            logger.info(f"✓ Order cancelled: {order_id}")

                            # Log event
                            event_name = f"DCA_LIMIT_ORDER_L{order['level']}_CANCELLED" if self.config.DCA_ENABLED else "LIMIT_ORDER_CANCELLED"
                            self.trade_logger.log_event(event_name, {
                                "order_id": order_id,
                                "reason": "TIMEOUT",
                                "elapsed_seconds": elapsed,
                                "max_wait_seconds": self.config.MAX_WAITING_LIMIT_ORDER,
                                "level": order.get('level', 1)
                            })

                        except Exception as e:
                            # Error 110001 = order already cancelled/filled - this is OK
                            if "110001" in str(e) or "not exists" in str(e).lower():
                                logger.info(f"✓ Order {order_id} already processed (filled or cancelled by exchange)")
                            else:
                                logger.error(f"Failed to cancel order {order_id}: {e}")
                    else:
                        # Still waiting
                        remaining_orders.append(order)
                        if int(elapsed) % 30 == 0:  # Log every 30s
                            if self.config.DCA_ENABLED:
                                logger.info(f"⏳ Waiting for DCA Level {order['level']}: {elapsed:.0f}s / {self.config.MAX_WAITING_LIMIT_ORDER}s")
                            else:
                                logger.info(f"⏳ Waiting for limit order: {elapsed:.0f}s / {self.config.MAX_WAITING_LIMIT_ORDER}s")

            # Update active orders list (remove filled orders)
            self.active_limit_orders = remaining_orders

            # Process filled orders AFTER updating active_limit_orders
            # This allows _finalize_limit_order_entry to know if this is the last fill
            if filled_orders:
                # Get current position with retry (Bybit may need time to update)
                position = None
                for retry in range(3):
                    time.sleep(0.5 if retry > 0 else 0)  # Wait 0.5s on retries
                    position = self.adapter.get_position(self.config.TICKER)
                    if position:
                        break
                    logger.warning(f"Position not found yet, retry {retry+1}/3...")

                if position:
                    # Finalize entry for filled orders
                    # Pass remaining_orders count so it knows if this is the last fill
                    self._finalize_limit_order_entry(position, filled_orders, len(remaining_orders))
                else:
                    logger.error("❌ CRITICAL: Position not found after 3 retries - order filled but position not detected!")
                    logger.error(f"   Filled orders: {[o['order_id'] for o in filled_orders]}")
                    logger.error(f"   This may cause duplicate orders. Manual intervention required.")

            # If all orders completed/cancelled
            if not self.active_limit_orders:
                # Check if position exists
                position = self.adapter.get_position(self.config.TICKER)

                if position:
                    # ========== SAFETY NET: Verify position has SL/TP protection ==========
                    # This should NEVER trigger (SL/TP are set in _finalize_limit_order_entry)
                    # But kept as emergency fallback if something goes wrong

                    # FIX: Convert to float to handle both string "0.0" and float 0.0
                    stop_loss = float(position.get('stopLoss', 0))
                    take_profit = float(position.get('takeProfit', 0))

                    # DEBUG: Log raw position data for diagnostics
                    logger.debug(f"Safety net check: position data = {position}")

                    if stop_loss > 0:
                        # OK - Position has SL/TP protection (expected path)
                        logger.info(f"✓ Position has SL/TP protection (SL: {stop_loss}, TP: {take_profit})")
                    else:
                        # ⚠️ EMERGENCY: Position without SL/TP (should not happen!)
                        logger.error("="*60)
                        logger.error("❌ EMERGENCY: Position WITHOUT SL/TP detected!")
                        logger.error("   This should NOT happen - indicates a bug in _finalize_limit_order_entry")
                        logger.error(f"   Raw position data: {position}")
                        logger.error("   Setting emergency SL/TP now...")
                        logger.error("="*60)

                        # Get position details
                        actual_entry = position['entryPrice']
                        side = position['side']  # "Buy" or "Sell"

                        # Determine decision from state or position side
                        if self.state and 'side' in self.state:
                            position_type = self.state['side']  # "Long" or "Short"
                        else:
                            position_type = "Long" if side == "Buy" else "Short"

                        decision = "BUY" if position_type == "Long" else "SELL"
                        side_str = "Buy" if decision == "BUY" else "Sell"

                        # Use planned SL/TP if DCA mode and available in state
                        if self.config.DCA_ENABLED and self.state and 'planned_sl' in self.state:
                            sl = self.state['planned_sl']
                            tp = self.state['planned_tp']
                            logger.info("Using planned SL/TP from state (DCA mode)")
                        else:
                            # Fallback: Calculate from actual entry
                            if decision == "BUY":
                                sl = actual_entry * (1 - self.config.TSL_PCT)
                                tp = actual_entry * (1 + self.config.TP_PCT) if self.config.TP_PCT > 0 else 0
                            else:
                                sl = actual_entry * (1 + self.config.TSL_PCT)
                                tp = actual_entry * (1 - self.config.TP_PCT) if self.config.TP_PCT > 0 else 0

                        # Set SL/TP on Bybit
                        try:
                            position_side = side_str
                            self.adapter.set_stop_loss(self.config.TICKER, sl, position_side)
                            logger.warning(f"✓ EMERGENCY Stop Loss set: {sl:.4f}")

                            if tp > 0:
                                self.adapter.set_take_profit(self.config.TICKER, tp, position_side)
                                logger.warning(f"✓ EMERGENCY Take Profit set: {tp:.4f}")

                            # Update state if it exists
                            if self.state:
                                self.state['last_sl'] = sl
                                if tp > 0:
                                    self.state['initial_tp'] = tp
                                self._save_state()

                            # Log event
                            self.trade_logger.log_event("EMERGENCY_SL_TP_SET", {
                                "entry_price": float(actual_entry),
                                "sl_price": float(sl),
                                "tp_price": float(tp) if tp > 0 else None,
                                "reason": "Position without SL/TP protection detected"
                            })

                            tp_display = f"{tp:.4f}" if tp > 0 else "OFF"
                            logger.warning(f"✓ EMERGENCY SL/TP protection enabled | SL: {sl:.4f} | TP: {tp_display}")

                        except Exception as e:
                            logger.error(f"❌ CRITICAL: Failed to set emergency SL/TP: {e}")
                            logger.error("   Position is UNPROTECTED - MANUAL INTERVENTION REQUIRED!")

                elif self.trade_logger.current_trade:
                    # No position exists - all orders cancelled without fills
                    duration = (datetime.now() - datetime.fromisoformat(
                        self.trade_logger.current_trade['start_time']
                    )).total_seconds()

                    self.trade_logger.end_trade({
                        "exit_reason": "LIMIT_ORDERS_TIMEOUT_ALL_CANCELLED",
                        "pnl_usd": 0.0,
                        "pnl_percent": 0.0,
                        "net_pnl_usd": 0.0,
                        "duration_seconds": int(duration),
                        "max_favorable_excursion": 0.0,
                        "max_adverse_excursion": 0.0,
                        "fees_paid": 0.0
                    })

                return False  # All orders completed/cancelled

            return True  # Still processing

        except Exception as e:
            logger.error(f"Error checking limit order status: {e}", exc_info=True)
            return False

    def _finalize_limit_order_entry(self, position: dict, filled_orders: list = None, remaining_orders_count: int = 0):
        """
        Finalize limit order entry after position is filled.
        Supports both single order and DCA mode (multiple fills).

        Args:
            position: Current position from Bybit API
            filled_orders: List of filled orders (for DCA mode)
            remaining_orders_count: Number of orders still pending (for DCA mode)
        """
        try:
            actual_entry = position['entryPrice']  # Average entry from Bybit (handles DCA automatically)
            side = position['side']  # "Long" or "Short"
            size = position['size']

            # Get decision from first filled order
            if filled_orders:
                decision = filled_orders[0]['decision']
            else:
                logger.error("No filled_orders provided to _finalize_limit_order_entry")
                return

            side_str = "Buy" if decision == "BUY" else "Sell"
            position_type = "LONG" if decision == "BUY" else "SHORT"

            # Check if this is first fill or subsequent DCA fill
            is_first_fill = not self.state or 'entry_price' not in self.state

            if is_first_fill:
                # ========== FIRST FILL: Initialize position ==========
                logger.info(f"Finalizing {position_type} entry @ {actual_entry:.4f}")

                # ========== NEW LOGIC: TP calculated ONCE from first filled order price ==========
                # TP NEVER changes after this point (based on best entry = first fill)
                first_fill_price = filled_orders[0]['price']

                if decision == "BUY":
                    tp = first_fill_price * (1 + self.config.TP_PCT) if self.config.TP_PCT > 0 else 0
                else:
                    tp = first_fill_price * (1 - self.config.TP_PCT) if self.config.TP_PCT > 0 else 0

                # ========== NEW LOGIC: SL based on furthest ACTIVE limit order OR BE ==========
                # If we still have pending orders → protect furthest level
                # If all filled/cancelled → go to breakeven
                sl_tp_already_set = False

                if self.config.DCA_ENABLED:
                    if self.active_limit_orders:
                        # Case: Still have pending orders - SL protects furthest level
                        # Use pre-calculated planned_sl (based on Level 3)
                        sl = self.state.get('planned_sl')
                        sl_tp_already_set = self.state.get('sl_tp_embedded', False)

                        logger.warning(f"✅ DCA MODE: FIRST FILL with {len(self.active_limit_orders)} pending orders")
                        logger.warning(f"   SL: {sl:.4f} (protecting furthest level)")
                        tp_display = f"{tp:.4f}" if tp > 0 else "OFF"
                        logger.warning(f"   TP: {tp_display} (FIXED - based on first fill @ {first_fill_price:.4f})")
                    else:
                        # Case: All orders filled/cancelled on first check - go to BE
                        if decision == "BUY":
                            sl = actual_entry * (1 - 0.0006)  # BE - fees
                        else:
                            sl = actual_entry * (1 + 0.0006)  # BE + fees

                        logger.warning(f"✅ DCA MODE: All orders filled/cancelled - SL → BREAKEVEN")
                        logger.warning(f"   SL: {sl:.4f} (BE + fees)")
                        tp_display = f"{tp:.4f}" if tp > 0 else "OFF"
                        logger.warning(f"   TP: {tp_display} (FIXED - based on first fill @ {first_fill_price:.4f})")
                else:
                    # SINGLE LIMIT ORDER: Calculate SL from actual entry
                    if decision == "BUY":
                        sl = actual_entry * (1 - self.config.TSL_PCT)
                    else:
                        sl = actual_entry * (1 + self.config.TSL_PCT)

                # Set highest/lowest for TSL tracking
                if decision == "BUY":
                    highest = actual_entry
                    lowest = 999999
                else:
                    highest = 999999
                    lowest = actual_entry

                # ========== SET SL/TP if not already embedded ==========
                if not sl_tp_already_set:
                    position_side = side_str
                    self.adapter.set_stop_loss(self.config.TICKER, sl, position_side)
                    logger.info(f"✓ Stop Loss set: {sl:.4f}")

                    if tp > 0:
                        self.adapter.set_take_profit(self.config.TICKER, tp, position_side)
                        logger.info(f"✓ Take Profit set: {tp:.4f}")

                    tp_str = f"{tp:.4f}" if tp > 0 else "OFF"
                    logger.warning(f"✓ SL/TP PROTECTION ENABLED | SL: {sl:.4f} | TP: {tp_str}")

                    # ========== VERIFY SL/TP WERE ACTUALLY SET ==========
                    # Wait for Bybit to process the SL/TP update
                    time.sleep(2.0)

                    # Verify with retries
                    verified = False
                    for retry in range(3):
                        verify_position = self.adapter.get_position(self.config.TICKER)
                        if verify_position:
                            verify_sl = verify_position.get('stopLoss', 0)
                            verify_tp = verify_position.get('takeProfit', 0)

                            # Check if SL is set (TP is optional)
                            if verify_sl > 0:
                                verified = True
                                logger.info(f"✅ SL/TP verification PASSED (retry {retry+1}/3): SL={verify_sl:.4f}, TP={verify_tp:.4f}")
                                break
                            else:
                                logger.warning(f"⚠️  SL/TP verification FAILED (retry {retry+1}/3): SL={verify_sl}, TP={verify_tp}")
                                if retry < 2:
                                    time.sleep(1.5)  # Wait before retry

                    if not verified:
                        logger.error("❌ SL/TP VERIFICATION FAILED after 3 retries!")
                        logger.error("   Forcing emergency SL/TP re-set NOW...")
                        # EMERGENCY: Force set SL/TP again with longer wait
                        self.adapter.set_stop_loss(self.config.TICKER, sl, position_side)
                        if tp > 0:
                            self.adapter.set_take_profit(self.config.TICKER, tp, position_side)
                        time.sleep(3.0)  # Longer wait
                        # Final verification
                        final_check = self.adapter.get_position(self.config.TICKER)
                        if final_check and final_check.get('stopLoss', 0) > 0:
                            logger.warning("✅ Emergency SL/TP re-set SUCCEEDED!")
                        else:
                            logger.error("❌ CRITICAL: Emergency SL/TP re-set FAILED! Manual intervention required!")

                # Calculate trade size USD for state tracking
                trade_size_usd = actual_entry * size

                # Log entry event
                event_name = "DCA_ENTRY_FIRST" if self.config.DCA_ENABLED else "ENTRY"
                entry_details = {
                    "order_type": "LIMIT",
                    "entry_price": float(actual_entry),
                    "quantity": float(size),
                    "sl_price": float(sl),
                    "tp_price": float(tp) if tp > 0 else None,
                    "leverage": self.config.LEVERAGE,
                    "candle": self.last_candle_data
                }

                if self.config.DCA_ENABLED:
                    entry_details['dca_levels_filled'] = [order['level'] for order in filled_orders]
                    entry_details['dca_levels_pending'] = len(self.active_limit_orders)

                self.trade_logger.log_event(event_name, entry_details)

                # Log indicators (only on first fill)
                if 'top_indicators' in self.last_decision_data:
                    self.trade_logger.log_indicators(
                        indicators=self.last_decision_data['top_indicators'],
                        model_probas={
                            'proba_buy': self.last_decision_data['proba_buy'],
                            'proba_sell': self.last_decision_data['proba_sell']
                        }
                    )

                # Save state
                total_qty = sum(order['qty'] for order in filled_orders)
                self.state = {
                    'side': position_type.capitalize(),
                    'entry_price': actual_entry,
                    'first_fill_price': first_fill_price,  # Store first fill for reference
                    'fixed_tp': tp,  # TP NEVER changes (based on first fill)
                    'initial_tp': tp,  # Legacy compatibility
                    'initial_sl': sl,  # Store initial SL for trade metrics
                    'original_qty': total_qty,
                    'initial_size': trade_size_usd,  # Actual trade size (USD) used for this position
                    'partial_tp_taken': False,
                    'dynamic_tp_levels_taken': 0,
                    'last_sl': sl,
                    'highest_price': highest,
                    'lowest_price': lowest,
                    'dca_fills': len(filled_orders) if self.config.DCA_ENABLED else 1
                }
                self._save_state()

                tp_display = f"{tp:.4f}" if tp > 0 else "OFF"

                # ========== CALCULATE AND LOG RRR (Risk/Reward Ratio) ==========
                if decision == "BUY":
                    risk_distance = abs(actual_entry - sl)
                    reward_distance = abs(tp - actual_entry) if tp > 0 else 0
                else:
                    risk_distance = abs(sl - actual_entry)
                    reward_distance = abs(actual_entry - tp) if tp > 0 else 0

                rrr = reward_distance / risk_distance if risk_distance > 0 else 0
                risk_pct = (risk_distance / actual_entry) * 100
                reward_pct = (reward_distance / actual_entry) * 100 if tp > 0 else 0

                if self.config.DCA_ENABLED:
                    logger.warning(f"✓ DCA Position initialized | Entry: {actual_entry:.4f} | SL: {sl:.4f} | TP: {tp_display}")
                    logger.warning(f"   Filled levels: {[order['level'] for order in filled_orders]} | Pending: {len(self.active_limit_orders)}")
                    logger.warning(f"   📊 RRR: 1:{rrr:.2f} | Risk: {risk_pct:.2f}% | Reward: {reward_pct:.2f}%")
                else:
                    logger.warning(f"✓ Position finalized | SL: {sl:.4f} | TP: {tp_display}")
                    logger.warning(f"   📊 RRR: 1:{rrr:.2f} | Risk: {risk_pct:.2f}% | Reward: {reward_pct:.2f}%")

            else:
                # ========== SUBSEQUENT DCA FILL: Update position ==========
                old_entry = self.state['entry_price']
                logger.info(f"DCA fill: updating entry {old_entry:.4f} → {actual_entry:.4f}")

                # ========== NEW LOGIC: SL based on active limit orders OR safe BE ==========
                decision = self.state['side'].upper()  # "LONG" or "SHORT"
                position_side = "Buy" if decision == "LONG" else "Sell"

                # TP NEVER changes (fixed at first fill price)
                tp = self.state.get('fixed_tp', self.state.get('initial_tp'))

                # Get current price for safety check
                current_price = self.adapter.latest_price(self.config.TICKER)
                old_sl = self.state.get('last_sl', 0)

                # Calculate target BE SL
                if decision == "LONG":
                    target_be_sl = actual_entry * (1 - 0.0006)  # BE - fees
                else:
                    target_be_sl = actual_entry * (1 + 0.0006)  # BE + fees

                # Determine SL based on pending orders and safety check
                if self.active_limit_orders:
                    # Case: Still have pending orders - keep SL at furthest level
                    sl = old_sl  # Don't change SL while waiting for more fills
                    logger.warning(f"🔄 DCA subsequent fill: {len(self.active_limit_orders)} orders still pending")
                    logger.warning(f"   SL: {sl:.4f} (UNCHANGED - protecting furthest level)")
                    logger.warning(f"   TP: {tp:.4f} (FIXED - never changes)")
                else:
                    # Case: All orders filled/cancelled - try to move SL to BE
                    # CRITICAL SAFETY CHECK: Don't set SL that would trigger immediately!

                    if decision == "LONG":
                        # For LONG: new SL must be BELOW current price
                        if target_be_sl < current_price:
                            sl = target_be_sl
                            logger.warning(f"🔄 DCA fill: All orders done - SL → BREAKEVEN (safe)")
                            logger.warning(f"   Old SL: {old_sl:.4f} → New SL: {sl:.4f} (BE - fees)")
                            logger.warning(f"   Current price: {current_price:.4f} ✅ (above SL)")
                        else:
                            # Dangerous! Keep old SL
                            sl = old_sl
                            logger.warning(f"⚠️  DCA fill: Cannot move SL to BE - would trigger instantly!")
                            logger.warning(f"   Target BE SL: {target_be_sl:.4f}")
                            logger.warning(f"   Current price: {current_price:.4f} ⚠️  (BELOW target SL!)")
                            logger.warning(f"   Keeping old SL: {sl:.4f} (will retry when price recovers)")

                            # Store pending BE target for next cycle check
                            self.state['pending_be_sl'] = target_be_sl

                    else:  # SHORT
                        # For SHORT: new SL must be ABOVE current price
                        if target_be_sl > current_price:
                            sl = target_be_sl
                            logger.warning(f"🔄 DCA fill: All orders done - SL → BREAKEVEN (safe)")
                            logger.warning(f"   Old SL: {old_sl:.4f} → New SL: {sl:.4f} (BE + fees)")
                            logger.warning(f"   Current price: {current_price:.4f} ✅ (below SL)")
                        else:
                            sl = old_sl
                            logger.warning(f"⚠️  DCA fill: Cannot move SL to BE - would trigger instantly!")
                            logger.warning(f"   Target BE SL: {target_be_sl:.4f}")
                            logger.warning(f"   Current price: {current_price:.4f} ⚠️  (ABOVE target SL!)")
                            logger.warning(f"   Keeping old SL: {sl:.4f} (will retry when price recovers)")

                            self.state['pending_be_sl'] = target_be_sl

                    logger.warning(f"   TP: {tp:.4f} (FIXED - never changes)")

                tp_display = f"{tp:.4f}" if tp > 0 else "OFF"

                # Update SL on Bybit (TP never changes - was set at first fill)
                try:
                    # Only update SL if it actually changed
                    if sl != old_sl:
                        self.adapter.set_stop_loss(self.config.TICKER, sl, position_side)
                        logger.info(f"✓ Stop Loss updated: {sl:.4f}")

                        # Verify update with single retry
                        time.sleep(1.5)
                        verify_position = self.adapter.get_position(self.config.TICKER)
                        if verify_position:
                            verify_sl = verify_position.get('stopLoss', 0)
                            if verify_sl > 0:
                                logger.info(f"✅ SL update verified: {verify_sl:.4f}")
                            else:
                                logger.warning(f"⚠️  SL verification warning: {verify_sl}")
                    else:
                        logger.info(f"✓ SL unchanged: {sl:.4f} (no update needed)")

                except Exception as e:
                    logger.error(f"Failed to update SL after DCA fill: {e}")

                # Log DCA add event
                self.trade_logger.log_event("DCA_ENTRY_ADD", {
                    "old_entry_price": float(old_entry),
                    "new_entry_price": float(actual_entry),
                    "new_quantity": float(size),
                    "added_quantity": float(sum(order['qty'] for order in filled_orders)),
                    "dca_levels_filled": [order['level'] for order in filled_orders],
                    "dca_levels_pending": len(self.active_limit_orders),
                    "old_sl": float(self.state.get('last_sl', 0)),
                    "new_sl": float(sl),
                    "old_tp": float(self.state.get('initial_tp', 0)),
                    "new_tp": float(tp) if tp > 0 else None,
                    "candle": self.last_candle_data
                })

                # Update state (only entry and SL - TP never changes!)
                self.state['entry_price'] = actual_entry
                self.state['last_sl'] = sl  # Update SL
                # TP remains fixed at first fill price - don't update!
                self.state['dca_fills'] = self.state.get('dca_fills', 1) + len(filled_orders)

                # Update highest/lowest for TSL tracking based on new breakeven
                if decision == "LONG":
                    self.state['highest_price'] = max(actual_entry, self.state.get('highest_price', actual_entry))
                    self.state['lowest_price'] = self.state.get('lowest_price', 999999)
                else:
                    self.state['highest_price'] = self.state.get('highest_price', 999999)
                    self.state['lowest_price'] = min(actual_entry, self.state.get('lowest_price', actual_entry))

                self._save_state()

                # ========== CALCULATE AND LOG RRR (Risk/Reward Ratio) ==========
                if decision == "LONG":
                    risk_distance = abs(actual_entry - sl)
                    reward_distance = abs(tp - actual_entry) if tp > 0 else 0
                else:
                    risk_distance = abs(sl - actual_entry)
                    reward_distance = abs(actual_entry - tp) if tp > 0 else 0

                rrr = reward_distance / risk_distance if risk_distance > 0 else 0
                risk_pct = (risk_distance / actual_entry) * 100
                reward_pct = (reward_distance / actual_entry) * 100 if tp > 0 else 0

                logger.warning(f"✓ DCA Position updated | New avg entry: {actual_entry:.4f} | SL: {sl:.4f} | TP: {tp_display}")
                logger.warning(f"   Total fills: {self.state['dca_fills']} | Pending: {remaining_orders_count}")
                logger.warning(f"   📊 RRR: 1:{rrr:.2f} | Risk: {risk_pct:.2f}% | Reward: {reward_pct:.2f}%")

                # RRR Warning if imbalanced (risk > reward)
                if rrr < 1.0 and tp > 0:
                    logger.warning(f"   ⚠️  WARNING: Risk > Reward! RRR = 1:{rrr:.2f} (should be >= 1:1)")

        except Exception as e:
            logger.error(f"Failed to finalize limit order entry: {e}", exc_info=True)

    def run_cycle(self):
        """Execute one cycle"""
        logger.info("="*60)
        logger.info(f"Cycle start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)

        try:
            # Always get decision to log model probabilities AND populate last_candle_data
            # This is needed even when waiting for limit orders (for _manage_position logging)
            decision = self.get_decision()

            # ========== LIMIT ORDER MODE: Check pending orders first ==========
            if self.config.LIMIT_ORDER_MODE and self.active_limit_orders:
                # Check if limit orders filled or timeout
                still_waiting = self._check_limit_order_status()

                if still_waiting:
                    # Orders still pending - but still manage position if it exists
                    # This is critical for TSL and monitoring (especially DCA mode)
                    position = self.adapter.get_position(self.config.TICKER)
                    if position and position.get('size', 0) > 0:
                        logger.info("Limit order(s) pending - managing existing position")
                        self._manage_position()
                    else:
                        logger.info("Limit order(s) pending - no position yet, waiting...")
                    return
                # If not still_waiting, orders were filled or cancelled - continue normally

            # Manage existing position
            if self._manage_position():
                return

            # Open new position if signal and no position (and no active limit orders)
            if decision in ["BUY", "SELL"]:
                # Safety check: Don't place new limit orders if some already exist
                if self.config.LIMIT_ORDER_MODE and self.active_limit_orders:
                    logger.warning("⚠️ Skipping new signal - limit order(s) already active")
                    return

                # COOLDOWN CHECK: Don't open new position if in cooldown period
                # This prevents re-entry on same 15m candle (bot checks every 60s)
                if self._is_in_cooldown_period():
                    logger.warning(f"⏰ Skipping {decision} signal - cooldown active after last position close")
                    return

                self._open_position(decision)
            elif decision == "HOLD":
                logger.info("No signal. Holding...")

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
    
    def start(self):
        """Start bot main loop"""
        logger.info("🤖 BOT V2 STARTED WITH ADVANCED LOGGING")
        
        try:
            self.adapter.set_leverage(self.config.TICKER, self.config.LEVERAGE)
            
            cycle = 0
            while True:
                cycle += 1
                logger.info(f"\nCYCLE #{cycle}")
                self.run_cycle()

                # Adaptive sleep: faster checks when waiting for limit orders
                if self.active_limit_orders:
                    sleep_time = 15  # Check every 15 seconds when orders pending
                    logger.info(f"⏰ Fast mode: Checking limit orders in {sleep_time}s")
                else:
                    sleep_time = self.config.LOOP_SLEEP_SECONDS  # Normal 5-minute interval

                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("\n" + "="*60)
            logger.info("Bot stopped by user")
            logger.info("="*60)
            
            # Close any open trade logging
            if self.trade_logger.current_trade:
                duration = (datetime.now() - datetime.fromisoformat(
                    self.trade_logger.current_trade['start_time']
                )).total_seconds()

                self.trade_logger.end_trade({
                    "exit_reason": "BOT_STOPPED_BY_USER",
                    "pnl_usd": 0.0,
                    "pnl_percent": 0.0,
                    "net_pnl_usd": 0.0,
                    "duration_seconds": int(duration),
                    "max_favorable_excursion": 0.0,
                    "max_adverse_excursion": 0.0,
                    "fees_paid": 0.0
                })
        except Exception as e:
            logger.critical(f"Critical error: {e}", exc_info=True)
            
            # Close any open trade logging
            if self.trade_logger.current_trade:
                duration = (datetime.now() - datetime.fromisoformat(
                    self.trade_logger.current_trade['start_time']
                )).total_seconds()

                self.trade_logger.end_trade({
                    "exit_reason": "BOT_CRASHED",
                    "pnl_usd": 0.0,
                    "pnl_percent": 0.0,
                    "net_pnl_usd": 0.0,
                    "duration_seconds": int(duration),
                    "max_favorable_excursion": 0.0,
                    "max_adverse_excursion": 0.0,
                    "fees_paid": 0.0,
                    "error_message": str(e)
                })


def launch_bot(args):
    """Launch bot with args"""
    config = BotConfig()
    config.TICKER = args.ticker
    config.TIMEFRAME = args.timeframe
    config.HELPER_TIMEFRAMES = args.helper_timeframes
    config.TRADE_SIZE_USD = args.trade_size
    config.TRADE_SIZE_PCT = getattr(args, 'trade_size_pct', 0.0)
    config.MAX_POSITIONS = getattr(args, 'max_positions', 0)
    config.LEVERAGE = args.leverage
    config.TP_PCT = args.tp_pct
    config.TSL_PCT = args.tsl_pct
    config.PROBABILITY_THRESHOLD = args.prob_threshold
    config.MIN_CONFIDENCE_RATIO = getattr(args, 'min_confidence_ratio', 1.5)
    # ICT Context Mode
    config.ICT_CONTEXT_MODE = getattr(args, 'ict_context', False)
    config.ICT_MIN_STRENGTH = getattr(args, 'ict_min_strength', 0.3)
    config.ICT_MAX_THRESHOLD_REDUCTION = getattr(args, 'ict_max_reduction', 0.20)
    config.REGIME_SENSITIVITY = getattr(args, 'regime_sensitivity', 1.0)
    config.PARTIAL_TP_ENABLED = args.partial_tp
    config.DYNAMIC_TP_ENABLED = getattr(args, 'dynamic_tp', False)
    config.HEDGE_MODE = getattr(args, 'hedge_mode', False)
    config.LIMIT_ORDER_MODE = getattr(args, 'limit_order', False)
    config.MAX_WAITING_LIMIT_ORDER = getattr(args, 'max_waiting_limit_order', 300)
    config.LIMIT_ORDER_CANDLES = getattr(args, 'limit_order_candles', 5)
    config.LIMIT_OFFSET_PCT = getattr(args, 'limit_offset_pct', 0.005)
    config.PROTECT_PROFIT_ENABLED = getattr(args, 'protect_profit', False)
    config.COOLDOWN_AFTER_LOSS_MINUTES = getattr(args, 'cooldown_after_loss', 30)
    # DCA Mode
    config.DCA_ENABLED = getattr(args, 'dca_mode', False)
    config.DCA_ATR_MULTIPLIER = getattr(args, 'dca_atr_multiplier', 1.5)  # Default 1.5x ATR (historical avg: 1.177%)

    # RSI Reversal Filter
    config.RSI_REVERSAL_FILTER_ENABLED = not getattr(args, 'no_rsi_reversal_filter', False)
    config.RSI_HIGH_THRESHOLD = getattr(args, 'rsi_high_threshold', 65.0)
    config.RSI_LOW_THRESHOLD = getattr(args, 'rsi_low_threshold', 35.0)
    config.RSI_OVERBOUGHT_LEVEL = getattr(args, 'rsi_overbought_level', 70.0)
    config.RSI_OVERSOLD_LEVEL = getattr(args, 'rsi_oversold_level', 30.0)

    version = getattr(args, 'version', 'v1.0')

    # Validation
    if config.PARTIAL_TP_ENABLED and config.DYNAMIC_TP_ENABLED:
        print("\n❌ ERROR: Cannot enable both --partial-tp and --dynamic-tp. Choose only one.\n")
        return

    if config.DCA_ENABLED and not config.LIMIT_ORDER_MODE:
        print("\n❌ ERROR: DCA mode requires --limit-order to be enabled.\n")
        return
    
    print("\n" + "="*70)
    print(f"{'BOT V2 WITH ADVANCED LOGGING':^70}")
    print("="*70)
    print(f"Ticker:            {config.TICKER}")
    print(f"Timeframe:         {config.TIMEFRAME}")
    print(f"Model Version:     {version}")
    if config.TRADE_SIZE_PCT > 0:
        print(f"Trade Size:        {config.TRADE_SIZE_PCT * 100:.2f}% of available balance (dynamic sizing)")
    else:
        print(f"Trade Size:        ${config.TRADE_SIZE_USD:.2f} (fixed)")
    print(f"TP/TSL:            {config.TP_PCT*100:.2f}% / {config.TSL_PCT*100:.2f}%")
    print(f"Threshold:         {config.PROBABILITY_THRESHOLD:.3f}")
    print(f"Min Conf Ratio:    {config.MIN_CONFIDENCE_RATIO:.3f}")

    # ICT Context Mode info
    if config.ICT_CONTEXT_MODE:
        print(f"ICT Context:       🎯 ENABLED (OPCJA 3)")
        print(f"  Min Strength:    {config.ICT_MIN_STRENGTH:.2f}")
        print(f"  Max Reduction:   {config.ICT_MAX_THRESHOLD_REDUCTION*100:.0f}%")
        print(f"  Regime Sens:     {config.REGIME_SENSITIVITY:.1f}x")
    else:
        print(f"ICT Context:       OFF (static thresholds)")

    print(f"Max Positions:     {config.MAX_POSITIONS if config.MAX_POSITIONS > 0 else 'UNLIMITED'}")

    if config.PARTIAL_TP_ENABLED:
        print(f"Partial TP:        ON (50% at halfway to TP)")
    elif config.DYNAMIC_TP_ENABLED:
        print(f"Dynamic TP:        ON (25% at 25%, 50%, 75%, 100%)")
    else:
        print(f"Partial/Dynamic:   OFF")
    
    print(f"Position Mode:     {'HEDGE (Long=1, Short=2)' if config.HEDGE_MODE else 'ONE-WAY (Idx=0)'}")

    if config.LIMIT_ORDER_MODE:
        timeout_info = f"{config.MAX_WAITING_LIMIT_ORDER}s"
        if config.LIMIT_ORDER_CANDLES > 0:
            timeout_info += f" ({config.LIMIT_ORDER_CANDLES} candles × {config.TIMEFRAME})"

        if config.DCA_ENABLED:
            print(f"Order Type:        DCA LIMIT (ATR-based, timeout: {timeout_info})")
            print(f"  Level:           {config.DCA_ATR_MULTIPLIER:.2f}x ATR (~{config.DCA_ATR_MULTIPLIER*0.785:.2f}% avg)")
        else:
            print(f"Order Type:        LIMIT (offset: {config.LIMIT_OFFSET_PCT*100:.2f}%, timeout: {timeout_info})")
    else:
        print(f"Order Type:        MARKET")

    print(f"Profit Protection: {'ON (BE if profit peaks >0.25% then declines)' if config.PROTECT_PROFIT_ENABLED else 'OFF'}")
    print(f"Cooldown Period:   {config.COOLDOWN_AFTER_LOSS_MINUTES} minutes after ANY closed position (prevents re-entry on same candle)")

    # RSI Reversal Filter info
    if config.RSI_REVERSAL_FILTER_ENABLED:
        print(f"RSI Filter:        ON (reversal-based entry)")
        print(f"  Crossings:       SHORT @ RSI crosses {config.RSI_OVERBOUGHT_LEVEL} down, LONG @ RSI crosses {config.RSI_OVERSOLD_LEVEL} up")
        print(f"  Blocks:          LONG if RSI >{config.RSI_HIGH_THRESHOLD} w/o 30-cross, SHORT if RSI <{config.RSI_LOW_THRESHOLD} w/o 70-cross")
    else:
        print(f"RSI Filter:        OFF")

    print(f"")
    print(f"Logs directory:    logs/")
    print(f"  - Trade JSONs:   logs/trades/")
    print(f"  - Candles:       logs/candles/")
    print(f"  - Indicators:    logs/indicators/")
    print(f"  - Analytics:     logs/analytics/")
    print("="*70 + "\n")
    
    bot = TradingBot(config, version)
    bot.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Bot V2 with Advanced Logging")
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--timeframe', required=True)
    parser.add_argument('--helper-timeframes', nargs='*', default=None)
    parser.add_argument('--version', type=str, default='v1.0',
                        help='Model version to use (e.g., v1.0, v1.1)')
    parser.add_argument('--trade-size', type=float, default=100.0,
                        help='Fixed trade size in USD (default: 100.0). Ignored if --trade-size-pct is set.')
    parser.add_argument('--trade-size-pct', type=float, default=0.0,
                        help='Trade size as percentage of available balance (e.g., 0.01 = 1%%). Overrides --trade-size if > 0.')
    parser.add_argument('--leverage', type=int, default=10)
    parser.add_argument('--tp-pct', type=float, required=True)
    parser.add_argument('--tsl-pct', type=float, required=True)
    parser.add_argument('--prob-threshold', type=float, required=True)
    parser.add_argument('--min-confidence-ratio', type=float, default=1.5,
                        help='Minimum confidence ratio: stronger_signal/weaker_signal (1.5 = 50%% more confident, 2.0 = 100%% more)')
    # ICT Context Mode arguments
    parser.add_argument('--ict-context', action='store_true',
                        help='Enable ICT context mode: dynamic thresholds based on Smart Money signals (OPCJA 3)')
    parser.add_argument('--ict-min-strength', type=float, default=0.3,
                        help='Minimum ICT strength to apply threshold reduction (default: 0.3, range: 0.0-1.0)')
    parser.add_argument('--ict-max-reduction', type=float, default=0.20,
                        help='Maximum threshold reduction when ICT strong (default: 0.20 = 20%%)')
    parser.add_argument('--regime-sensitivity', type=float, default=1.0,
                        help='Regime adjustment sensitivity multiplier (default: 1.0, range: 0.5-2.0)')
    parser.add_argument('--partial-tp', action='store_true',
                        help='Enable old partial TP mechanism (50%% at halfway to TP)')
    parser.add_argument('--dynamic-tp', action='store_true',
                        help='Enable new dynamic TP mechanism (25%% at each of 4 levels: 25%%, 50%%, 75%%, 100%%)')
    parser.add_argument('--hedge-mode', action='store_true',
                        help='Enable Hedge Mode (positionIdx: 1=Long, 2=Short). Default is One-Way Mode (positionIdx: 0).')
    parser.add_argument('--limit-order', action='store_true',
                        help='Use limit orders instead of market orders')
    parser.add_argument('--limit-offset-pct', type=float, default=0.005,
                        help='Price offset for limit orders (default: 0.005 = 0.5%%)')
    parser.add_argument('--max-waiting-limit-order', type=int, default=300,
                        help='Maximum seconds to wait for limit order execution before cancelling (default: 300). Auto-calculated if --limit-order-candles is set.')
    parser.add_argument('--limit-order-candles', type=int, default=5,
                        help='Number of candles to wait for limit order execution (default: 5). Auto-calculates timeout based on timeframe.')
    parser.add_argument('--protect-profit', action='store_true',
                        help='Enable profit protection: move SL to breakeven if profit peaks >0.25%% but declines before hitting partial TP')
    parser.add_argument('--cooldown-after-loss', type=int, default=30,
                        help='Minutes to wait before opening new position after ANY closed position (default: 30 = 2x 15m candles, prevents re-entry on same candle)')
    # DCA Mode arguments
    parser.add_argument('--dca-mode', action='store_true',
                        help='Enable DCA mode: place single ATR-based limit order (requires --limit-order)')
    parser.add_argument('--dca-atr-multiplier', type=float, default=1.5,
                        help='DCA ATR multiplier for limit order distance (default: 1.5x = ~1.18%% avg based on historical analysis)')

    # RSI Reversal Filter arguments
    parser.add_argument('--no-rsi-reversal-filter', action='store_true',
                        help='Disable RSI reversal filter (enabled by default)')
    parser.add_argument('--rsi-high-threshold', type=float, default=65.0,
                        help='Block LONG if RSI above this without 30-cross (default: 65.0)')
    parser.add_argument('--rsi-low-threshold', type=float, default=35.0,
                        help='Block SHORT if RSI below this without 70-cross (default: 35.0)')
    parser.add_argument('--rsi-overbought-level', type=float, default=70.0,
                        help='RSI overbought level - crossing down triggers SHORT signal (default: 70.0)')
    parser.add_argument('--rsi-oversold-level', type=float, default=30.0,
                        help='RSI oversold level - crossing up triggers LONG signal (default: 30.0)')

    launch_bot(parser.parse_args())
