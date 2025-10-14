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
    TICKER: str = "SOLUSDT"
    TIMEFRAME: str = "1h"
    HELPER_TIMEFRAMES: list = ["4h", "12h", "1D"]
    TRADE_SIZE_USD: float = 100.0
    LEVERAGE: int = 10
    TP_PCT: float = 0.06
    TSL_PCT: float = 0.022
    PROBABILITY_THRESHOLD: float = 0.80
    LOOP_SLEEP_SECONDS: int = 60
    CANDLES_FOR_FEATURES: int = 200
    PARTIAL_TP_ENABLED: bool = True
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 3


class TradingBot:
    """Trading bot with advanced logging"""
    
    def __init__(self, config: BotConfig):
        self.config = config
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
        
        # Cache for last decision data
        self.last_decision_data = {}
        self.last_candle_data = {}
        
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
        return BybitAdapter(api_key=key, api_secret=secret, base_url=base_url)
    
    def _get_strategy_id(self):
        helpers = '_plus_' + '_'.join(self.config.HELPER_TIMEFRAMES) if self.config.HELPER_TIMEFRAMES else ""
        return f"{self.config.TICKER}_{self.config.TIMEFRAME.replace(' ', '')}{helpers}"
    
    def _load_models(self):
        long_id = f"{self.base_id}_long"
        short_id = f"{self.base_id}_short"
        
        try:
            m_long = joblib.load(f"models/{long_id}_model.joblib")
            s_long = joblib.load(f"models/{long_id}_scaler.joblib")
            f_long = joblib.load(f"models/{long_id}_features.joblib")
            
            m_short = joblib.load(f"models/{short_id}_model.joblib")
            s_short = joblib.load(f"models/{short_id}_scaler.joblib")
            f_short = joblib.load(f"models/{short_id}_features.joblib")
            
            logging.info(f"✓ Models loaded (Long: {len(f_long)} features, Short: {len(f_short)} features)")
            return m_long, s_long, f_long, m_short, s_short, f_short
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found: {e}")
    
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
                df = fetch_and_prepare_data(
                    ticker=self.config.TICKER,
                    timeframe=self.config.TIMEFRAME,
                    limit=self.config.CANDLES_FOR_FEATURES,
                    helper_timeframes=self.config.HELPER_TIMEFRAMES,
                    side='live'
                )
            
            if df.empty:
                return "ERROR"
            
            last_row = df.iloc[-1]
            last_row_df = df.iloc[[-1]]
            
            # Extract features
            X_long = last_row_df[self.features_long]
            X_short = last_row_df[self.features_short]
            
            X_long_scaled = self.scaler_long.transform(X_long)
            X_short_scaled = self.scaler_short.transform(X_short)
            
            proba_buy = self.model_long.predict_proba(X_long_scaled)[0][1]
            proba_sell = self.model_short.predict_proba(X_short_scaled)[0][1]
            
            logging.info(f"Probabilities: BUY={proba_buy:.3f}, SELL={proba_sell:.3f}")
            
            # Determine decision
            decision = "HOLD"
            if proba_buy > self.config.PROBABILITY_THRESHOLD and proba_buy > proba_sell:
                decision = "BUY"
            elif proba_sell > self.config.PROBABILITY_THRESHOLD and proba_sell > proba_buy:
                decision = "SELL"
            
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
                        self.adapter.set_stop_loss(self.config.TICKER, new_sl, "Sell")
                        
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
                        self.adapter.set_stop_loss(self.config.TICKER, new_sl, "Buy")
                        
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
    
    def _handle_partial_tp(self, position: Dict, current_price: float):
        """Handle partial TP with breakeven"""
        if not self.config.PARTIAL_TP_ENABLED or self.config.TP_PCT <= 0:
            return
        
        if self.state.get('partial_tp_taken', False):
            return
        
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
        
        qty = round(size / 2, 3)
        if qty <= 0:
            return
        
        reduce_side = "Sell" if side == 'Long' else "Buy"
        
        try:
            self.adapter.market_close(self.config.TICKER, reduce_side, qty)
            time.sleep(2)
            
            # Calculate partial P&L
            if side == 'Long':
                pnl_pct = (current_price / entry - 1) * 100
            else:
                pnl_pct = (entry / current_price - 1) * 100
            
            pnl_usd = (self.config.TRADE_SIZE_USD * pnl_pct / 100) / 2  # Half position
            
            # Log partial TP event
            self.trade_logger.log_event("PARTIAL_TP", {
                "trigger_price": float(partial_tp_price),
                "exit_price": float(current_price),
                "quantity_closed": float(qty),
                "quantity_remaining": float(size - qty),
                "pnl_pct": float(pnl_pct),
                "pnl_usd": float(pnl_usd),
                "new_sl": float(entry),
                "reason": "breakeven",
                "candle": self.last_candle_data
            })
            
            # Move SL to breakeven
            logging.info(f"Moving SL to breakeven: {entry:.4f}")
            if side == 'Long':
                self.adapter.set_stop_loss(self.config.TICKER, entry, "Sell")
            else:
                self.adapter.set_stop_loss(self.config.TICKER, entry, "Buy")
            
            self.state['partial_tp_taken'] = True
            self.state['last_sl'] = entry
            self._save_state()
            
            logging.info(f"✓ Partial TP executed. Remaining: {size - qty}")
            
        except Exception as e:
            logging.error(f"Partial TP failed: {e}")
            self.trade_logger.log_event("ERROR", {
                "type": "PARTIAL_TP_FAILED",
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
            
            qty = round(self.config.TRADE_SIZE_USD / current_price, 3)
            if qty <= 0:
                return
            
            side_str = "Buy" if decision == "BUY" else "Sell"
            position_type = "LONG" if decision == "BUY" else "SHORT"
            
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
            
            # Set SL
            sl_side = "Sell" if decision == "BUY" else "Buy"
            self.adapter.set_stop_loss(self.config.TICKER, sl, sl_side)
            
            # Set TP
            if tp > 0:
                tp_side = "Sell" if decision == "BUY" else "Buy"
                self.adapter.set_take_profit(self.config.TICKER, tp, tp_side)
            
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
                'partial_tp_taken': False,
                'last_sl': sl,
                'highest_price': highest,
                'lowest_price': lowest
            }
            self._save_state()
            
            logging.warning(f"✓ Position opened | SL: {sl:.4f} | TP: {tp:.4f if tp > 0 else 'OFF'}")
            
        except Exception as e:
            logging.error(f"Failed to open position: {e}", exc_info=True)
            if self.trade_logger.current_trade:
                self.trade_logger.log_event("ERROR", {
                    "type": "ENTRY_FAILED",
                    "error": str(e)
                })
    
    def run_cycle(self):
        """Execute one cycle"""
        logging.info("="*60)
        logging.info(f"Cycle start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("="*60)
        
        try:
            # Manage existing position
            if self._manage_position():
                return
            
            # Get decision
            decision = self.get_decision()
            
            if decision in ["BUY", "SELL"]:
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
    config.PARTIAL_TP_ENABLED = args.partial_tp
    
    print("\n" + "="*70)
    print(f"{'BOT V2 WITH ADVANCED LOGGING':^70}")
    print("="*70)
    print(f"Ticker:            {config.TICKER}")
    print(f"Timeframe:         {config.TIMEFRAME}")
    print(f"Trade Size:        ${config.TRADE_SIZE_USD}")
    print(f"TP/TSL:            {config.TP_PCT*100:.2f}% / {config.TSL_PCT*100:.2f}%")
    print(f"Threshold:         {config.PROBABILITY_THRESHOLD:.3f}")
    print(f"Partial TP:        {'ON' if config.PARTIAL_TP_ENABLED else 'OFF'}")
    print(f"")
    print(f"Logs directory:    logs/")
    print(f"  - Trade JSONs:   logs/trades/")
    print(f"  - Candles:       logs/candles/")
    print(f"  - Indicators:    logs/indicators/")
    print(f"  - Analytics:     logs/analytics/")
    print("="*70 + "\n")
    
    bot = TradingBot(config)
    bot.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Bot V2 with Advanced Logging")
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--timeframe', required=True)
    parser.add_argument('--helper-timeframes', nargs='*', default=None)
    parser.add_argument('--trade-size', type=float, default=100.0)
    parser.add_argument('--leverage', type=int, default=10)
    parser.add_argument('--tp-pct', type=float, required=True)
    parser.add_argument('--tsl-pct', type=float, required=True)
    parser.add_argument('--prob-threshold', type=float, required=True)
    parser.add_argument('--partial-tp', action='store_true')
    
    launch_bot(parser.parse_args())
