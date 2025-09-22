# live_trader.py
"""
Live Trading Bot - Komitet Ekspertów
Connects to Bybit exchange and executes ML-based trading decisions on 5-minute intervals.
"""

import time
import asyncio
import schedule
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

import config
from services.bybit_service import BybitService
from services.analysis_service import AnalysisService
from logic.position_manager import PositionManager
from utils.data_preparer import prepare_full_feature_set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Live trading bot that uses ML predictions for automated trading.
    """
    
    def __init__(self, trading_mode='paper'):
        """
        Initialize live trader.
        
        Args:
            trading_mode: 'paper' for paper trading, 'live' for real trading
        """
        self.trading_mode = trading_mode
        
        # Initialize services
        self.bybit = BybitService(mode=trading_mode)
        self.analyzer = AnalysisService(config.TICKER_NAME_FOR_MODELS)
        self.position_manager = PositionManager(config)
        
        # Trading state
        self.current_capital = config.INITIAL_CAPITAL
        self.last_signal_time = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        
        logger.info(f"LiveTrader initialized in {trading_mode.upper()} mode")
        logger.info(f"Trading pair: {config.TICKER}")
        logger.info(f"Initial capital: ${self.current_capital}")
    
    def run(self):
        """Main execution loop with 5-minute scheduling."""
        logger.info("=== STARTING LIVE TRADER ===")
        
        # Schedule the trading function every 5 minutes
        schedule.every(5).minutes.do(self._execute_trading_cycle)
        
        # Run initial sync to get current positions
        self._sync_positions()
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except KeyboardInterrupt:
                logger.info("Shutdown signal received")
                self._shutdown()
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                self.consecutive_errors += 1
                
                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.error("Too many consecutive errors, shutting down")
                    self._shutdown()
                    break
                
                time.sleep(60)  # Wait before retrying
    
    def _execute_trading_cycle(self):
        """Execute one complete trading cycle."""
        try:
            logger.info("--- EXECUTING TRADING CYCLE ---")
            
            # 1. Fetch market data
            market_data = self._fetch_market_data()
            if market_data is None:
                return
            
            # 2. Prepare features and get analysis
            analysis = self._get_market_analysis(market_data)
            if analysis is None:
                return
            
            # 3. Get trading signal from position manager
            current_candle = market_data.iloc[-1]
            signal = self.position_manager.get_trading_signal(current_candle, analysis, self.current_capital)
            
            # 4. Log ML predictions
            predictions = self.position_manager.get_ml_predictions(analysis)
            self._log_predictions(predictions, signal)
            
            # 5. Execute trading decision
            self._execute_signal(signal)
            
            # 6. Process position management instructions
            self._process_position_instructions(signal.get('instructions', []))
            
            # 7. Update capital and sync positions
            self._update_capital()
            self._sync_positions()
            
            # Reset error counter on success
            self.consecutive_errors = 0
            self.last_signal_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self.consecutive_errors += 1
    
    def _fetch_market_data(self) -> Optional[pd.DataFrame]:
        """Fetch and prepare market data."""
        try:
            # Fetch raw 5-minute data
            df_raw = self.bybit.fetch_recent_candles(
                symbol=config.TICKER,
                interval_minutes=5,
                limit=500
            )
            
            if df_raw is None or df_raw.empty:
                logger.warning("Failed to fetch market data")
                return None
            
            # Prepare features using the same function as backtester
            features_df = prepare_full_feature_set(df_raw)
            
            if features_df.empty:
                logger.warning("Failed to prepare features from market data")
                return None
            
            logger.info(f"Fetched {len(features_df)} candles with features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
    
    def _get_market_analysis(self, market_data) -> Optional[Dict[str, Any]]:
        """Get ML analysis from market data."""
        try:
            # Use the last complete candle
            last_candle = market_data.iloc[-1]
            analysis = self.analyzer.get_analysis_from_row(last_candle)
            
            if not analysis:
                logger.warning("Failed to get market analysis")
                return None
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting market analysis: {e}")
            return None
    
    def _log_predictions(self, predictions: Dict[str, Any], signal: Dict[str, Any]):
        """Log ML predictions and signal details."""
        logger.info("=== ML PREDICTIONS ===")
        logger.info(f"Votes: Long={predictions['votes_long']}, Short={predictions['votes_short']}")
        logger.info(f"Recommendation: {predictions['recommendation']['signal']} "
                   f"(strength: {predictions['recommendation']['strength']})")
        
        for expert, pred in predictions['predictions'].items():
            logger.info(f"{expert.upper()}: {pred['prediction']} "
                       f"(conf: {pred['confidence']:.3f}, threshold: {pred['threshold']}, "
                       f"eligible: {pred['vote_eligible']})")
        
        logger.info(f"SIGNAL: {signal['action']}")
        if 'confidence' in signal:
            conf = signal['confidence']
            logger.info(f"Confidence levels - Momentum: {conf['momentum']:.3f}, "
                       f"Reversion: {conf['reversion']:.3f}, PA: {conf['pa']:.3f}")
    
    def _execute_signal(self, signal: Dict[str, Any]):
        """Execute trading signal."""
        action = signal['action']
        
        if action == 'OPEN_LONG':
            self._execute_long_entry(signal)
        elif action == 'OPEN_SHORT':
            self._execute_short_entry(signal)
        elif action == 'CLOSE':
            self._execute_position_close(signal)
        elif action == 'HOLD':
            logger.info("Signal: HOLD - No action taken")
        else:
            logger.warning(f"Unknown signal action: {action}")
    
    def _execute_long_entry(self, signal: Dict[str, Any]):
        """Execute long position entry."""
        try:
            logger.info("=== EXECUTING LONG ENTRY ===")
            
            response = self.bybit.place_order(
                symbol=config.TICKER,
                side='Buy',
                order_type='Market',
                qty=signal['size'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit']
            )
            
            if response.get('ret_code') == 0:
                logger.info(f"Long position opened: {signal['size']} @ {signal['entry_price']}")
                
                # Update position manager
                position_data = {
                    'strategy': 'long',
                    'entry_date': datetime.now(),
                    'entry_price': signal['entry_price'],
                    'size': signal['size'],
                    'current_sl_price': signal['stop_loss'],
                    'tp_price': signal['take_profit'],
                    'breakeven_trigger_price': signal['breakeven_trigger'],
                    'trailing_trigger_price': signal['trailing_trigger'],
                    'conf_momentum': signal['confidence']['momentum'],
                    'conf_reversion': signal['confidence']['reversion'],
                    'conf_pa': signal['confidence']['pa']
                }
                self.position_manager.update_position_from_live_data(position_data)
            else:
                logger.error(f"Failed to open long position: {response}")
                
        except Exception as e:
            logger.error(f"Error executing long entry: {e}")
    
    def _execute_short_entry(self, signal: Dict[str, Any]):
        """Execute short position entry."""
        try:
            logger.info("=== EXECUTING SHORT ENTRY ===")
            
            response = self.bybit.place_order(
                symbol=config.TICKER,
                side='Sell',
                order_type='Market',
                qty=signal['size'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit']
            )
            
            if response.get('ret_code') == 0:
                logger.info(f"Short position opened: {signal['size']} @ {signal['entry_price']}")
                
                # Update position manager
                position_data = {
                    'strategy': 'short',
                    'entry_date': datetime.now(),
                    'entry_price': signal['entry_price'],
                    'size': signal['size'],
                    'current_sl_price': signal['stop_loss'],
                    'tp_price': signal['take_profit'],
                    'breakeven_trigger_price': signal['breakeven_trigger'],
                    'trailing_trigger_price': signal['trailing_trigger'],
                    'conf_momentum': signal['confidence']['momentum'],
                    'conf_reversion': signal['confidence']['reversion'],
                    'conf_pa': signal['confidence']['pa']
                }
                self.position_manager.update_position_from_live_data(position_data)
            else:
                logger.error(f"Failed to open short position: {response}")
                
        except Exception as e:
            logger.error(f"Error executing short entry: {e}")
    
    def _execute_position_close(self, signal: Dict[str, Any]):
        """Execute position close."""
        try:
            logger.info(f"=== CLOSING POSITION - {signal['exit_reason']} ===")
            
            response = self.bybit.close_position(config.TICKER)
            
            if response.get('ret_code') == 0:
                logger.info(f"Position closed due to: {signal['exit_reason']}")
                self.position_manager.clear_position()
            else:
                logger.error(f"Failed to close position: {response}")
                
        except Exception as e:
            logger.error(f"Error executing position close: {e}")
    
    def _process_position_instructions(self, instructions: list):
        """Process position management instructions."""
        for instruction in instructions:
            try:
                inst_type = instruction['type']
                
                if inst_type == 'MOVE_SL_TO_BREAKEVEN':
                    self._move_sl_to_breakeven(instruction)
                elif inst_type == 'ACTIVATE_TRAILING_STOP':
                    self._activate_trailing_stop(instruction)
                elif inst_type == 'UPDATE_TRAILING_STOP':
                    self._update_trailing_stop(instruction)
                else:
                    logger.warning(f"Unknown instruction type: {inst_type}")
                    
            except Exception as e:
                logger.error(f"Error processing instruction {instruction}: {e}")
    
    def _move_sl_to_breakeven(self, instruction: Dict[str, Any]):
        """Move stop loss to break-even."""
        logger.info(f"INSTRUCTION: Move SL to break-even @ {instruction['new_sl_price']}")
        
        response = self.bybit.modify_position(
            symbol=config.TICKER,
            stop_loss=instruction['new_sl_price']
        )
        
        if response.get('ret_code') == 0:
            logger.info("Stop loss moved to break-even")
        else:
            logger.error(f"Failed to move SL to break-even: {response}")
    
    def _activate_trailing_stop(self, instruction: Dict[str, Any]):
        """Activate trailing stop."""
        logger.info("INSTRUCTION: Trailing stop activated")
        # Trailing stop logic is handled in position manager
        # This is mainly for logging
    
    def _update_trailing_stop(self, instruction: Dict[str, Any]):
        """Update trailing stop level."""
        logger.info(f"INSTRUCTION: Update trailing stop to {instruction['new_sl_price']}")
        
        response = self.bybit.modify_position(
            symbol=config.TICKER,
            stop_loss=instruction['new_sl_price']
        )
        
        if response.get('ret_code') == 0:
            logger.info("Trailing stop updated")
        else:
            logger.error(f"Failed to update trailing stop: {response}")
    
    def _update_capital(self):
        """Update current capital from account balance."""
        try:
            balance = self.bybit.get_account_balance()
            if balance.get('ret_code') == 0:
                usdt_balance = balance['result']['USDT']['available_balance']
                self.current_capital = float(usdt_balance)
                logger.info(f"Capital updated: ${self.current_capital:.2f}")
        except Exception as e:
            logger.error(f"Error updating capital: {e}")
    
    def _sync_positions(self):
        """Sync positions with exchange."""
        try:
            positions = self.bybit.get_current_positions()
            # TODO: Implement position synchronization logic
            position_status = self.position_manager.get_position_status()
            logger.info(f"Position sync - Has position: {position_status['has_position']}")
            
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
    
    def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down live trader...")
        # TODO: Close any open positions if needed
        # TODO: Save state/logs
        logger.info("Live trader shutdown complete")


def run():
    """Main entry point for live trading."""
    try:
        # Determine trading mode from config or default to paper trading
        trading_mode = getattr(config, 'TRADING_MODE', 'paper')
        
        trader = LiveTrader(trading_mode=trading_mode)
        trader.run()
        
    except Exception as e:
        logger.error(f"Fatal error in live trader: {e}")
        raise


if __name__ == "__main__":
    run()