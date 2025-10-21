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
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from bybit_adapter import BybitAdapter, BybitAPIError
from data_preparer_pa import fetch_and_prepare_data
from trade_logger import TradeLogger
from prediction_logger import PredictionLogger
from feature_cache import FeatureCache

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_v2.log'),
        logging.StreamHandler()
    ]
)


class BotConfig:
    """Bot configuration"""
    TICKER: str = "ETHUSDT"
    TIMEFRAME: str = "15m"
    HELPER_TIMEFRAMES: list = ["1h", "4h"]
    TRADE_SIZE_USD: float = 100.0
    LEVERAGE: int = 10
    TP_PCT: float = 0.03
    TSL_PCT: float = 0.01
    PROBABILITY_THRESHOLD: float = 0.7
    MIN_PROBA_DIFF: float = 0.2  # Minimum difference between BUY and SELL probabilities
    LOOP_SLEEP_SECONDS: int = 60
    CANDLES_FOR_FEATURES: int = 5000  # Reduced from 10000 to avoid Bybit API data availability issues
    PARTIAL_TP_ENABLED: bool = True
    DYNAMIC_TP_ENABLED: bool = False  # New dynamic TP: 25% at each of 4 levels (25%, 50%, 75%, 100%)
    HEDGE_MODE: bool = False  # Hedge Mode: positionIdx 1=Long, 2=Short. One-Way Mode: positionIdx 0
    LIMIT_ORDER_MODE: bool = False  # Use limit orders instead of market orders
    MAX_WAITING_LIMIT_ORDER: int = 300  # Seconds to wait for limit order execution before cancelling
    LIMIT_OFFSET_PCT: float = 0.005  # Price offset for limit orders (0.5% default)
    PROTECT_PROFIT_ENABLED: bool = False  # Enable profit protection: move SL to BE if profit peaks >0.25% but declines
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 3


class TradingBot:
    """Trading bot with advanced logging"""
    
    def __init__(self, config: BotConfig, version: str = 'v1.0'):
        self.config = config
        self.version = version
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
        self._restore_trade_logging_if_needed()

        # EPHEMERAL MODE: Auto-recovery from Bybit API
        # If container restarted without persistent state, detect existing position and restore
        self._check_existing_position_on_startup()

        # Cache for last decision data
        self.last_decision_data = {}
        self.last_candle_data = {}

        # Limit order tracking (only 1 active order per ticker)
        self.active_limit_order = None  # {'order_id': str, 'timestamp': float, 'side': str, 'price': float}

        # PERFORMANCE: Feature cache for incremental updates
        self.feature_cache = FeatureCache(window_size=self.config.CANDLES_FOR_FEATURES)

        logging.info("="*60)
        logging.info(f"Bot V2 initialized with Trade Logger: {self.base_id}")
        logging.info(f"TP: {config.TP_PCT*100:.2f}% | TSL: {config.TSL_PCT*100:.2f}%")
        logging.info("="*60)
    
    def _init_adapter(self):
        load_dotenv()
        key = os.getenv("BYBIT_API_KEY")
        secret = os.getenv("BYBIT_API_SECRET")
        base_url = os.getenv("BYBIT_BASE_URL")
        if not key or not secret:
            raise ValueError("Missing API keys in .env")
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
            
            logging.info(f"✓ Models loaded from version {version} (Long: {len(f_long)} features, Short: {len(f_short)} features)")
            return m_long, s_long, f_long, m_short, s_short, f_short
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found for version {version}: {e}")
    
    def _save_state(self):
        try:
            state_copy = self.state.copy()
            state_copy['last_updated'] = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(state_copy, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
                logging.info(f"✓ State loaded: {self.state}")
            except Exception as e:
                logging.error(f"Failed to load state: {e}")
    
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
                logging.info(f"🔄 Restoring trade logging for existing {side} position @ {entry_price}")
                
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
                
                logging.info("✓ Trade logging restored")

    def _check_existing_position_on_startup(self):
        """
        EPHEMERAL MODE: Auto-recovery from Bybit API.

        If container restarted without persistent state (ephemeral volumes),
        this method detects existing positions from Bybit API and reconstructs
        minimal state to prevent duplicate position errors.
        """
        try:
            # Skip if state already has position info (normal restoration path)
            if self.state.get('side'):
                logging.info("✓ State already loaded, skipping API recovery")
                return

            # Check Bybit API for existing position
            position = self.adapter.get_position(self.config.TICKER)

            if not position:
                logging.info("✓ No existing position detected on Bybit")
                return

            # Position exists but state is empty - RECOVERY NEEDED
            side = position['side']  # "Long" or "Short"
            entry_price = position['entryPrice']
            size = position['size']
            stop_loss = position.get('stopLoss', 0.0)
            take_profit = position.get('takeProfit', 0.0)

            logging.warning("="*60)
            logging.warning(f"⚠️  EPHEMERAL RECOVERY: Detected {side} position on Bybit")
            logging.warning(f"    Entry: {entry_price:.4f} | Size: {size}")
            logging.warning(f"    SL: {stop_loss:.4f} | TP: {take_profit:.4f}")
            logging.warning(f"    Reconstructing state from API...")
            logging.warning("="*60)

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
                logging.info(f"✓ Trade logger started for recovered {side} position")

        except Exception as e:
            logging.error(f"Error during startup position recovery: {e}", exc_info=True)
            # Non-fatal - bot will continue, but may have duplicate position risk
            logging.warning("⚠️  Recovery failed - monitor for duplicate positions!")

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
    
    def get_decision(self, df=None) -> str:
        """Get ML model decision with full logging"""
        try:
            # Fetch data if not provided
            if df is None:
                logging.info(f"📥 Fetching market data for {self.config.TICKER}...")
                # Combine all model features for preservation
                all_model_features = list(set(self.features_long + self.features_short))

                # PERFORMANCE: Use feature cache for incremental updates
                df = self.feature_cache.get_features(
                    ticker=self.config.TICKER,
                    timeframe=self.config.TIMEFRAME,
                    helper_timeframes=self.config.HELPER_TIMEFRAMES,
                    side='backtest',
                    version=self.version,
                    model_features_to_preserve=all_model_features,
                    skip_slow_features=False  # v1.2 models need pivot S/R features (nearest_resistance_strength, etc.)
                )
                logging.info(f"✓ Data fetched successfully ({len(df)} candles)")
            
            if df.empty:
                logging.error("❌ Empty dataframe received!")
                return "ERROR"
            
            # CRITICAL: Use ONLY closed candles (exclude the last incomplete candle)
            # The last candle (iloc[-1]) is still forming and will be excluded
            df_closed = df.iloc[:-1]
            
            if df_closed.empty:
                logging.error("❌ No closed candles available!")
                return "ERROR"
            
            # Use the last CLOSED candle for ML prediction
            last_row = df_closed.iloc[-1]
            last_row_df = df_closed.iloc[[-1]]
            
            # Extract features with validation
            logging.info("🤖 Running ML model predictions...")

            # VALIDATION: Check if all required features are present
            missing_long = set(self.features_long) - set(last_row_df.columns)
            missing_short = set(self.features_short) - set(last_row_df.columns)

            if missing_long:
                logging.error(f"❌ Missing {len(missing_long)} LONG features!")
                logging.error(f"   Examples: {list(missing_long)[:5]}")
                return "ERROR"

            if missing_short:
                logging.error(f"❌ Missing {len(missing_short)} SHORT features!")
                logging.error(f"   Examples: {list(missing_short)[:5]}")
                return "ERROR"

            X_long = last_row_df[self.features_long]
            X_short = last_row_df[self.features_short]
            
            X_long_scaled = self.scaler_long.transform(X_long)
            X_short_scaled = self.scaler_short.transform(X_short)
            
            proba_buy = self.model_long.predict_proba(X_long_scaled)[0][1]
            proba_sell = self.model_short.predict_proba(X_short_scaled)[0][1]
            
            # Calculate probability difference (confidence gap)
            proba_diff = abs(proba_buy - proba_sell)
            
            logging.info(f"📊 Model Probabilities: BUY={proba_buy:.3f}, SELL={proba_sell:.3f} (threshold={self.config.PROBABILITY_THRESHOLD:.3f})")
            logging.info(f"📏 Probability Difference: {proba_diff:.3f} (min_required={self.config.MIN_PROBA_DIFF:.3f})")
            
            # Determine decision
            decision = "HOLD"
            if proba_buy > self.config.PROBABILITY_THRESHOLD and proba_buy > proba_sell:
                # Check if confidence gap is sufficient
                if proba_diff >= self.config.MIN_PROBA_DIFF:
                    decision = "BUY"
                else:
                    logging.warning(f"⚠️  BUY signal rejected: insufficient confidence gap ({proba_diff:.3f} < {self.config.MIN_PROBA_DIFF:.3f})")
            elif proba_sell > self.config.PROBABILITY_THRESHOLD and proba_sell > proba_buy:
                # Check if confidence gap is sufficient
                if proba_diff >= self.config.MIN_PROBA_DIFF:
                    decision = "SELL"
                else:
                    logging.warning(f"⚠️  SELL signal rejected: insufficient confidence gap ({proba_diff:.3f} < {self.config.MIN_PROBA_DIFF:.3f})")
            
            logging.info(f"🎯 Model Decision: {decision}")

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
                    min_proba_diff=self.config.MIN_PROBA_DIFF,
                    decision=decision
                )
            except Exception as e:
                logging.warning(f"⚠️  Failed to log prediction: {e}")
            
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
            
            return decision
            
        except Exception as e:
            logging.error(f"Error in get_decision: {e}", exc_info=True)
            return "ERROR"
    
    def _update_tsl(self, position: Dict, current_price: float):
        """Update TSL - ONLY when in profit"""
        side = position.get('side')
        entry = position.get('entryPrice', 0)
        
        if entry == 0:
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
                    try:
                        logging.info(f"📈 TSL update: {last_sl:.4f} → {new_sl:.4f}")
                        self.adapter.set_stop_loss(self.config.TICKER, new_sl, "Buy")
                        
                        # Log TSL event
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
                    except Exception as e:
                        logging.error(f"TSL update failed: {e}")
                        self.trade_logger.log_event("ERROR", {
                            "type": "TSL_UPDATE_FAILED",
                            "error": str(e)
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
                    try:
                        logging.info(f"📉 TSL update: {last_sl:.4f} → {new_sl:.4f}")
                        self.adapter.set_stop_loss(self.config.TICKER, new_sl, "Sell")
                        
                        # Log TSL event
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
                    except Exception as e:
                        logging.error(f"TSL update failed: {e}")
                        self.trade_logger.log_event("ERROR", {
                            "type": "TSL_UPDATE_FAILED",
                            "error": str(e)
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
                    logging.warning(
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

                        logging.info(f"✓ Profit protection activated: SL moved to {entry:.4f}")

                    except Exception as e:
                        logging.error(f"Failed to activate profit protection: {e}")
                        self.trade_logger.log_event("ERROR", {
                            "type": "PROFIT_PROTECTION_FAILED",
                            "error": str(e)
                        })

            else:  # Short
                if last_sl > entry or last_sl == 0:
                    logging.warning(
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

                        logging.info(f"✓ Profit protection activated: SL moved to {entry:.4f}")

                    except Exception as e:
                        logging.error(f"Failed to activate profit protection: {e}")
                        self.trade_logger.log_event("ERROR", {
                            "type": "PROFIT_PROTECTION_FAILED",
                            "error": str(e)
                        })

    def _handle_partial_tp(self, position: Dict, current_price: float):
        """Handle partial TP with breakeven (old mechanism) or dynamic TP (new mechanism)"""
        # Validate mutual exclusivity
        if self.config.PARTIAL_TP_ENABLED and self.config.DYNAMIC_TP_ENABLED:
            logging.error("Cannot enable both PARTIAL_TP and DYNAMIC_TP. Choose only one.")
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
        logging.warning(f"🎯 PARTIAL TP HIT at {current_price:.4f}")

        # Use adapter's round_qty for ticker-specific precision
        qty = self.adapter.round_qty(self.config.TICKER, size / 2)

        # Validate against minimum order quantity
        min_qty = self.adapter.get_min_order_qty(self.config.TICKER)
        if qty < min_qty:
            logging.warning(f"Calculated qty {qty:.4f} < minOrderQty {min_qty:.4f}, closing entire position instead")
            qty = size

        if qty <= 0:
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

            logging.info(f"Partial close: requested={qty}, actual_closed={actual_qty_closed}, remaining={actual_remaining_size}")

            # Calculate partial P&L based on actual closed quantity
            if side == 'Long':
                pnl_pct = (current_price / entry - 1) * 100
            else:
                pnl_pct = (entry / current_price - 1) * 100

            pnl_usd = (self.config.TRADE_SIZE_USD * pnl_pct / 100) * (actual_qty_closed / size)

            # Log partial TP event with ACTUAL quantities
            self.trade_logger.log_event("PARTIAL_TP", {
                "trigger_price": float(partial_tp_price),
                "exit_price": float(current_price),
                "quantity_requested": float(qty),
                "quantity_closed": float(actual_qty_closed),
                "quantity_remaining": float(actual_remaining_size),
                "pnl_pct": float(pnl_pct),
                "pnl_usd": float(pnl_usd),
                "new_sl": float(entry),
                "reason": "breakeven",
                "candle": self.last_candle_data
            })
            
            # Move SL to breakeven if it was below breakeven
            last_sl = self.state.get('last_sl', 0)
            should_update_sl = False
            
            if side == 'Long':
                # For Long: move SL to breakeven if it was below entry
                if last_sl < entry:
                    should_update_sl = True
                    logging.info(f"TSL was below breakeven ({last_sl:.4f}), moving to breakeven: {entry:.4f}")
            else:  # Short
                # For Short: move SL to breakeven if it was above entry
                if last_sl > entry or last_sl == 0:
                    should_update_sl = True
                    logging.info(f"TSL was above breakeven ({last_sl:.4f}), moving to breakeven: {entry:.4f}")
            
            if should_update_sl:
                logging.info(f"Moving SL to breakeven: {entry:.4f}")
                if side == 'Long':
                    self.adapter.set_stop_loss(self.config.TICKER, entry, "Buy")
                else:
                    self.adapter.set_stop_loss(self.config.TICKER, entry, "Sell")
                
                self.state['last_sl'] = entry
            
            self.state['partial_tp_taken'] = True
            self._save_state()

            logging.info(f"✓ Partial TP executed. Remaining: {actual_remaining_size} (requested close: {qty}, actual close: {actual_qty_closed})")
            
        except Exception as e:
            logging.error(f"Partial TP failed: {e}")
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
            # Close entire remaining position
            qty = size
            logging.warning(f"🎯 DYNAMIC TP LEVEL 4 HIT (100%) - CLOSING ALL REMAINING at {current_price:.4f}")
        else:
            # Levels 1-3: close 25% of initial size with proper rounding
            qty = self.adapter.round_qty(self.config.TICKER, initial_size * 0.25)

            # Validate against minimum order quantity
            min_qty = self.adapter.get_min_order_qty(self.config.TICKER)
            if qty < min_qty:
                logging.warning(f"Calculated qty {qty:.4f} < minOrderQty {min_qty:.4f}, closing entire position instead")
                qty = size

            # Ensure we don't try to close more than current position
            if qty > size:
                logging.warning(f"Calculated qty {qty:.4f} > current size {size:.4f}, adjusting to {size:.4f}")
                qty = size

            logging.warning(f"🎯 DYNAMIC TP LEVEL {next_level} HIT ({int(level_pct*100)}%) at {current_price:.4f}")

        if qty <= 0:
            logging.error(f"Invalid qty={qty} for Dynamic TP Level {next_level}, skipping")
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

            logging.info(f"Dynamic TP L{next_level}: requested={qty}, actual_closed={actual_qty_closed}, remaining={actual_remaining_size}")

            # Calculate partial P&L based on actual closed quantity
            if side == 'Long':
                pnl_pct = (current_price / entry - 1) * 100
            else:
                pnl_pct = (entry / current_price - 1) * 100

            pnl_usd = (self.config.TRADE_SIZE_USD * pnl_pct / 100) * (actual_qty_closed / initial_size)

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
                        logging.info(f"TSL was below breakeven ({last_sl:.4f}), moving to breakeven: {entry:.4f}")
                else:  # Short
                    if last_sl > entry or last_sl == 0:
                        should_update_sl = True
                        logging.info(f"TSL was above breakeven ({last_sl:.4f}), moving to breakeven: {entry:.4f}")
                
                if should_update_sl:
                    logging.info(f"Moving SL to breakeven: {entry:.4f}")
                    if side == 'Long':
                        self.adapter.set_stop_loss(self.config.TICKER, entry, "Buy")
                    else:
                        self.adapter.set_stop_loss(self.config.TICKER, entry, "Sell")
                    
                    self.state['last_sl'] = entry
            
            self._save_state()

            logging.info(f"✓ Dynamic TP Level {next_level} executed. Remaining: {actual_remaining_size} (requested close: {qty}, actual close: {actual_qty_closed})")
            
        except Exception as e:
            logging.error(f"Dynamic TP Level {next_level} failed: {e}")
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
                    
                    pnl_usd = self.config.TRADE_SIZE_USD * pnl_pct / 100
                    
                    # Log exit
                    self.trade_logger.log_event("EXIT", {
                        "trigger": "SL_or_TP",
                        "exit_price": float(current_price),
                        "pnl_pct": float(pnl_pct),
                        "pnl_usd": float(pnl_usd),
                        "candle": self.last_candle_data
                    })
                    
                    # End trade
                    duration = (datetime.now() - datetime.fromisoformat(
                        self.trade_logger.current_trade['start_time']
                    )).total_seconds()
                    
                    self.trade_logger.end_trade({
                        "duration_seconds": int(duration),
                        "total_pnl_usd": float(pnl_usd),
                        "total_pnl_pct": float(pnl_pct)
                    })
                
                if self.state:
                    logging.info("Position closed. Resetting state.")
                    self.state = {}
                    self._save_state()
                return False
            
            side = position.get('side')
            entry = position.get('entryPrice', 0)
            size = position.get('size', 0)
            current_price = self.adapter.latest_price(self.config.TICKER)

            # DUST POSITION CLEANUP: Check if position is too small to close normally
            min_qty = self.adapter.get_min_order_qty(self.config.TICKER)
            if size > 0 and size < min_qty:
                logging.warning(
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

                    logging.info(f"Dust position cleanup attempted for {size:.4f} {self.config.TICKER}")
                except Exception as e:
                    logging.error(f"Failed to close dust position: {e}")
                    # Continue execution - don't crash bot

            if current_price == 0:
                return True
            
            # Calculate P&L
            pnl_pct = ((current_price / entry - 1) if side == 'Long' else (entry / current_price - 1)) * 100
            
            logging.info(
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
            logging.error(f"Error managing position: {e}", exc_info=True)
            self.trade_logger.log_event("ERROR", {
                "type": "MANAGE_POSITION_ERROR",
                "error": str(e)
            })
            return True
    
    def _open_position(self, decision: str):
        """Open new position with verification and logging"""
        try:
            current_price = self.adapter.latest_price(self.config.TICKER)
            if current_price == 0:
                return

            qty = round(self.config.TRADE_SIZE_USD / current_price, 2)
            if qty <= 0:
                return

            side_str = "Buy" if decision == "BUY" else "Sell"
            position_type = "LONG" if decision == "BUY" else "SHORT"

            # ========== LIMIT ORDER MODE ==========
            if self.config.LIMIT_ORDER_MODE:
                # Calculate limit price with offset
                if decision == "BUY":
                    # LONG: Buy cheaper (limit below current price)
                    limit_price = current_price * (1 - self.config.LIMIT_OFFSET_PCT)
                else:
                    # SHORT: Sell higher (limit above current price)
                    limit_price = current_price * (1 + self.config.LIMIT_OFFSET_PCT)

                logging.warning(f"📋 Placing LIMIT order: {position_type}")
                logging.warning(f"   Current: {current_price:.4f} | Limit: {limit_price:.4f} | Offset: {self.config.LIMIT_OFFSET_PCT*100:.2f}%")

                # Start trade logging (will finalize after order fills)
                self.trade_logger.start_trade(
                    ticker=self.config.TICKER,
                    side=position_type,
                    decision_data=self.last_decision_data
                )

                # Place limit order
                resp = self.adapter.limit_open(self.config.TICKER, side_str, qty, limit_price)
                order_id = (resp.get("result") or {}).get("orderId")

                if not order_id:
                    logging.error("Failed to place limit order - no orderId returned")
                    return

                # Track active limit order
                self.active_limit_order = {
                    'order_id': order_id,
                    'timestamp': time.time(),
                    'side': position_type,
                    'price': limit_price,
                    'qty': qty,
                    'decision': decision
                }

                # Log event
                self.trade_logger.log_event("LIMIT_ORDER_PLACED", {
                    "order_id": order_id,
                    "order_type": "LIMIT",
                    "limit_price": float(limit_price),
                    "current_price": float(current_price),
                    "quantity": float(qty),
                    "side": position_type,
                    "max_wait_seconds": self.config.MAX_WAITING_LIMIT_ORDER
                })

                logging.info(f"✓ Limit order placed: {order_id} | Wait max {self.config.MAX_WAITING_LIMIT_ORDER}s")
                return  # Exit - will check order status in run_cycle()

            # ========== MARKET ORDER MODE (original logic) ==========
            logging.warning(f"🚀 Opening {position_type}: {qty} @ ~{current_price:.4f}")

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
            
            logging.info(f"✓ Entry verified: {actual_entry:.4f}")
            
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
                'original_qty': qty,
                'initial_size': self.config.TRADE_SIZE_USD,  # For dynamic TP calculations
                'partial_tp_taken': False,
                'dynamic_tp_levels_taken': 0,  # For dynamic TP mechanism
                'last_sl': sl,
                'highest_price': highest,
                'lowest_price': lowest
            }
            self._save_state()
            
            tp_display = f"{tp:.4f}" if tp > 0 else "OFF"
            logging.warning(f"✓ Position opened | SL: {sl:.4f} | TP: {tp_display}")
            
        except Exception as e:
            logging.error(f"Failed to open position: {e}", exc_info=True)
            if self.trade_logger.current_trade:
                self.trade_logger.log_event("ERROR", {
                    "type": "ENTRY_FAILED",
                    "error": str(e)
                })

    def _check_limit_order_status(self) -> bool:
        """
        Check status of active limit order.

        Returns:
            True if order is being processed (still waiting), False if completed/cancelled
        """
        if not self.active_limit_order:
            return False

        try:
            order_id = self.active_limit_order['order_id']
            order_timestamp = self.active_limit_order['timestamp']
            elapsed = time.time() - order_timestamp
            max_wait = self.config.MAX_WAITING_LIMIT_ORDER

            # Check if order filled by querying position
            position = self.adapter.get_position(self.config.TICKER)
            if position:
                # Position exists - order was filled!
                logging.warning("="*60)
                logging.warning(f"✅ LIMIT ORDER FILLED: {order_id}")
                logging.warning(f"   Entry: {position['entryPrice']:.4f} | Size: {position['size']}")
                logging.warning("="*60)

                # Finalize entry - set SL/TP and save state
                self._finalize_limit_order_entry(position)

                # Clear limit order tracking
                self.active_limit_order = None
                return False  # Order completed

            # Position doesn't exist - check timeout
            if elapsed > max_wait:
                logging.warning("="*60)
                logging.warning(f"⏰ LIMIT ORDER TIMEOUT: {order_id}")
                logging.warning(f"   Elapsed: {elapsed:.0f}s / {max_wait}s")
                logging.warning(f"   Cancelling order...")
                logging.warning("="*60)

                # Cancel order
                try:
                    self.adapter.cancel_order(self.config.TICKER, order_id)
                    logging.info(f"✓ Order cancelled: {order_id}")

                    # Log event
                    self.trade_logger.log_event("LIMIT_ORDER_CANCELLED", {
                        "order_id": order_id,
                        "reason": "TIMEOUT",
                        "elapsed_seconds": elapsed,
                        "max_wait_seconds": max_wait
                    })

                except Exception as e:
                    logging.error(f"Failed to cancel order {order_id}: {e}")

                # Clear limit order tracking
                self.active_limit_order = None

                # End trade logging (order didn't fill)
                if self.trade_logger.current_trade:
                    self.trade_logger.end_trade({
                        "exit_reason": "LIMIT_ORDER_TIMEOUT",
                        "final_balance": 0.0,
                        "pnl": 0.0
                    })

                return False  # Order cancelled

            # Still waiting
            if int(elapsed) % 30 == 0:  # Log every 30s
                logging.info(f"⏳ Waiting for limit order: {elapsed:.0f}s / {max_wait}s")

            return True  # Still processing

        except Exception as e:
            logging.error(f"Error checking limit order status: {e}", exc_info=True)
            return False

    def _finalize_limit_order_entry(self, position: dict):
        """
        Finalize limit order entry after position is filled.
        Sets SL/TP, saves state, logs entry event.
        """
        try:
            actual_entry = position['entryPrice']
            side = position['side']  # "Long" or "Short"
            size = position['size']

            decision = self.active_limit_order['decision']
            qty = self.active_limit_order['qty']
            side_str = "Buy" if decision == "BUY" else "Sell"
            position_type = "LONG" if decision == "BUY" else "SHORT"

            logging.info(f"Finalizing {position_type} entry @ {actual_entry:.4f}")

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

            # Set SL
            position_side = side_str
            self.adapter.set_stop_loss(self.config.TICKER, sl, position_side)

            # Set TP
            if tp > 0:
                self.adapter.set_take_profit(self.config.TICKER, tp, position_side)

            # Log entry event
            self.trade_logger.log_event("ENTRY", {
                "order_type": "LIMIT",
                "entry_price": float(actual_entry),
                "quantity": float(size),
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
                'original_qty': qty,
                'initial_size': self.config.TRADE_SIZE_USD,
                'partial_tp_taken': False,
                'dynamic_tp_levels_taken': 0,
                'last_sl': sl,
                'highest_price': highest,
                'lowest_price': lowest
            }
            self._save_state()

            tp_display = f"{tp:.4f}" if tp > 0 else "OFF"
            logging.warning(f"✓ Position finalized | SL: {sl:.4f} | TP: {tp_display}")

        except Exception as e:
            logging.error(f"Failed to finalize limit order entry: {e}", exc_info=True)

    def run_cycle(self):
        """Execute one cycle"""
        logging.info("="*60)
        logging.info(f"Cycle start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("="*60)

        try:
            # ========== LIMIT ORDER MODE: Check pending order first ==========
            if self.config.LIMIT_ORDER_MODE and self.active_limit_order:
                # Check if limit order filled or timeout
                still_waiting = self._check_limit_order_status()

                if still_waiting:
                    # Order still pending - skip new signals
                    logging.info("Limit order pending - skipping new signals this cycle")
                    return
                # If not still_waiting, order was filled or cancelled - continue normally

            # Always get decision to log model probabilities
            decision = self.get_decision()

            # Manage existing position
            if self._manage_position():
                return

            # Open new position if signal and no position (and no active limit order)
            if decision in ["BUY", "SELL"]:
                # Safety check: Don't place new limit order if one already exists
                if self.config.LIMIT_ORDER_MODE and self.active_limit_order:
                    logging.warning("⚠️ Skipping new signal - limit order already active")
                    return

                self._open_position(decision)
            elif decision == "HOLD":
                logging.info("No signal. Holding...")

        except Exception as e:
            logging.error(f"Cycle error: {e}", exc_info=True)
    
    def start(self):
        """Start bot main loop"""
        logging.info("🤖 BOT V2 STARTED WITH ADVANCED LOGGING")
        
        try:
            self.adapter.set_leverage(self.config.TICKER, self.config.LEVERAGE)
            
            cycle = 0
            while True:
                cycle += 1
                logging.info(f"\nCYCLE #{cycle}")
                self.run_cycle()
                time.sleep(self.config.LOOP_SLEEP_SECONDS)
                
        except KeyboardInterrupt:
            logging.info("\n" + "="*60)
            logging.info("Bot stopped by user")
            logging.info("="*60)
            
            # Close any open trade logging
            if self.trade_logger.current_trade:
                self.trade_logger.end_trade({
                    "reason": "bot_stopped_by_user"
                })
        except Exception as e:
            logging.critical(f"Critical error: {e}", exc_info=True)
            
            # Close any open trade logging
            if self.trade_logger.current_trade:
                self.trade_logger.end_trade({
                    "reason": "bot_crashed",
                    "error": str(e)
                })


def launch_bot(args):
    """Launch bot with args"""
    config = BotConfig()
    config.TICKER = args.ticker
    config.TIMEFRAME = args.timeframe
    config.HELPER_TIMEFRAMES = args.helper_timeframes
    config.TRADE_SIZE_USD = args.trade_size
    config.LEVERAGE = args.leverage
    config.TP_PCT = args.tp_pct
    config.TSL_PCT = args.tsl_pct
    config.PROBABILITY_THRESHOLD = args.prob_threshold
    config.MIN_PROBA_DIFF = args.min_proba_diff
    config.PARTIAL_TP_ENABLED = args.partial_tp
    config.DYNAMIC_TP_ENABLED = getattr(args, 'dynamic_tp', False)
    config.HEDGE_MODE = getattr(args, 'hedge_mode', False)
    config.LIMIT_ORDER_MODE = getattr(args, 'limit_order', False)
    config.MAX_WAITING_LIMIT_ORDER = getattr(args, 'max_waiting_limit_order', 300)
    config.LIMIT_OFFSET_PCT = getattr(args, 'limit_offset_pct', 0.005)
    config.PROTECT_PROFIT_ENABLED = getattr(args, 'protect_profit', False)

    version = getattr(args, 'version', 'v1.0')
    
    if config.PARTIAL_TP_ENABLED and config.DYNAMIC_TP_ENABLED:
        print("\n❌ ERROR: Cannot enable both --partial-tp and --dynamic-tp. Choose only one.\n")
        return
    
    print("\n" + "="*70)
    print(f"{'BOT V2 WITH ADVANCED LOGGING':^70}")
    print("="*70)
    print(f"Ticker:            {config.TICKER}")
    print(f"Timeframe:         {config.TIMEFRAME}")
    print(f"Model Version:     {version}")
    print(f"Trade Size:        ${config.TRADE_SIZE_USD}")
    print(f"TP/TSL:            {config.TP_PCT*100:.2f}% / {config.TSL_PCT*100:.2f}%")
    print(f"Threshold:         {config.PROBABILITY_THRESHOLD:.3f}")
    print(f"Min Proba Diff:    {config.MIN_PROBA_DIFF:.3f}")
    
    if config.PARTIAL_TP_ENABLED:
        print(f"Partial TP:        ON (50% at halfway to TP)")
    elif config.DYNAMIC_TP_ENABLED:
        print(f"Dynamic TP:        ON (25% at 25%, 50%, 75%, 100%)")
    else:
        print(f"Partial/Dynamic:   OFF")
    
    print(f"Position Mode:     {'HEDGE (Long=1, Short=2)' if config.HEDGE_MODE else 'ONE-WAY (Idx=0)'}")

    if config.LIMIT_ORDER_MODE:
        print(f"Order Type:        LIMIT (offset: {config.LIMIT_OFFSET_PCT*100:.2f}%, timeout: {config.MAX_WAITING_LIMIT_ORDER}s)")
    else:
        print(f"Order Type:        MARKET")

    print(f"Profit Protection: {'ON (BE if profit peaks >0.25% then declines)' if config.PROTECT_PROFIT_ENABLED else 'OFF'}")

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
    parser.add_argument('--trade-size', type=float, default=100.0)
    parser.add_argument('--leverage', type=int, default=10)
    parser.add_argument('--tp-pct', type=float, required=True)
    parser.add_argument('--tsl-pct', type=float, required=True)
    parser.add_argument('--prob-threshold', type=float, required=True)
    parser.add_argument('--min-proba-diff', type=float, default=0.0,
                        help='Minimum probability difference between BUY and SELL (confidence gap)')
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
                        help='Maximum seconds to wait for limit order execution before cancelling (default: 300)')
    parser.add_argument('--protect-profit', action='store_true',
                        help='Enable profit protection: move SL to breakeven if profit peaks >0.25%% but declines before hitting partial TP')

    launch_bot(parser.parse_args())
