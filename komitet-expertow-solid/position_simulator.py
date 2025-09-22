#!/usr/bin/env python3
"""
Position Management Simulator

This script simulates all position management events to test functionality on demo account.
It will:
1. Open a position on ETHUSDT with SL/TP
2. Simulate break-even (BE) scenario
3. Simulate trailing stop scenarios
4. Test position closing mechanisms

Usage: python position_simulator.py
"""

import time
import logging
import pandas as pd
from typing import Dict, Any
from datetime import datetime

import config  # Import configuration

# Import services
from services.analysis_service import AnalysisService
from services.bybit_service import BybitService
from logic.position_manager import PositionManager
from utils.data_preparer import prepare_full_feature_set

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PositionSimulator:
    """Simulates all position management events for testing purposes."""
    
    def __init__(self):
        """Initialize simulator with demo account services."""
        logger.info("🚀 INITIALIZING POSITION SIMULATOR")
        
        # Override ticker for simulation
        self.ticker = "ETHUSDT"  # Use ETHUSDT as requested
        self.ticker_name = "ETHUSDT"
        
        # Initialize services with demo mode
        self.bybit_service = BybitService(mode='live', demo=True)
        self.analysis_service = AnalysisService(config.TICKER_NAME_FOR_MODELS)  # Use existing models
        self.manager = PositionManager(config)
        
        # Simulation state
        self.simulation_step = 0
        self.position_opened = False
        self.be_triggered = False
        self.trailing_activated = False
        
        logger.info(f"✅ Position Simulator initialized for {self.ticker}")
        logger.info(f"📊 Using demo environment: {self.bybit_service.demo}")

    def log_step(self, step_name: str, details: str = ""):
        """Log simulation step with formatting."""
        self.simulation_step += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 SIMULATION STEP {self.simulation_step}: {step_name}")
        if details:
            logger.info(f"📝 {details}")
        logger.info(f"{'='*60}")

    def get_current_price_and_analysis(self) -> tuple:
        """Get current market data and analysis for ETHUSDT."""
        logger.info(f"📈 Fetching market data for {self.ticker}...")
        
        # Get recent candles
        candles_df = self.bybit_service.fetch_recent_candles(self.ticker, interval_minutes=5, limit=500)
        if candles_df.empty:
            raise Exception(f"Failed to fetch candles for {self.ticker}")
        
        # Prepare features
        features_df = prepare_full_feature_set(candles_df)
        current_candle = features_df.iloc[-1]
        
        # Get ML analysis
        analysis = self.analysis_service.get_analysis_from_row(current_candle)
        
        logger.info(f"💰 Current {self.ticker} price: {analysis['current_price']:.2f}")
        logger.info(f"📊 ATR (5m): {analysis['atr_value_5m']:.6f}")
        
        return current_candle, analysis

    def get_account_capital(self) -> float:
        """Get available capital for position sizing."""
        balance_info = self.bybit_service.get_account_balance()
        if balance_info and 'coin' in balance_info:
            for coin in balance_info['coin']:
                if coin.get('coin') == 'USDT':
                    capital = float(coin.get('equity', 0.0))
                    logger.info(f"💰 Available capital: {capital:.2f} USDT")
                    return capital
        
        logger.warning("⚠️ Could not retrieve capital, using default 1000 USDT")
        return 1000.0

    def simulate_position_opening(self) -> bool:
        """Simulate opening a test position."""
        self.log_step("POSITION OPENING", "Opening test position with SL/TP")
        
        try:
            # Get current market data
            current_candle, analysis = self.get_current_price_and_analysis()
            capital = self.get_account_capital()
            
            # Force a LONG position for simulation (override ML predictions)
            logger.info("🎭 FORCING LONG POSITION for simulation purposes")
            
            # Calculate position parameters manually
            entry_price = analysis['current_price']
            atr_value = analysis['atr_value_5m']
            
            # Calculate SL and TP using config values
            sl_distance = atr_value * config.ATR_MULTIPLIER
            sl_price = entry_price - sl_distance  # LONG position
            
            tp_distance = abs(entry_price - sl_price) * config.RRR
            tp_price = entry_price + tp_distance
            
            # Calculate position size
            risk_usd = capital * config.RISK_PERCENT
            position_size = (risk_usd / sl_distance) if sl_distance > 0 else 0.01
            
            # Ensure reasonable position size
            position_size = max(0.001, min(position_size, 0.1))  # Between 0.001 and 0.1 BTC
            
            logger.info(f"📊 POSITION PARAMETERS:")
            logger.info(f"   Entry Price: {entry_price:.2f}")
            logger.info(f"   Stop Loss: {sl_price:.2f}")
            logger.info(f"   Take Profit: {tp_price:.2f}")
            logger.info(f"   Position Size: {position_size:.6f} BTC")
            logger.info(f"   Risk Amount: {risk_usd:.2f} USDT")
            
            # Place order on exchange
            logger.info("📤 Placing LONG order on exchange...")
            order_response = self.bybit_service.place_order(
                symbol=self.ticker,
                side='Buy',
                order_type='Market',
                qty=position_size,
                stop_loss=sl_price,
                take_profit=tp_price
            )
            
            if order_response and order_response.get('ret_code') == 0:
                logger.info("✅ Order placed successfully!")
                logger.info(f"📋 Order ID: {order_response.get('order_id')}")
                
                # Wait for order to fill
                logger.info("⏳ Waiting for order to fill...")
                time.sleep(5)
                
                # Update position manager with simulated position data
                position_details = {
                    'strategy': 'long',
                    'entry_date': pd.Timestamp.now(),
                    'entry_price': entry_price,
                    'size': position_size,
                    'current_sl_price': sl_price,
                    'tp_price': tp_price,
                    'is_be': False,
                    'is_trailing': False,
                    'breakeven_trigger_price': entry_price + (tp_distance * config.BREAKEVEN_TRIGGER_PERCENT),
                    'breakeven_sl_price': entry_price,
                    'trailing_trigger_price': entry_price + (sl_distance * config.TRAILING_SL_TRIGGER_R),
                    'conf_momentum': 0.75,
                    'conf_reversion': 0.60,
                    'conf_pa': 0.80
                }
                
                self.manager.update_position_from_live_data(position_details)
                self.position_opened = True
                
                logger.info("✅ Position opened and registered in PositionManager!")
                return True
            else:
                logger.error(f"❌ Failed to place order: {order_response}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error opening position: {e}")
            return False

    def simulate_breakeven_scenario(self) -> bool:
        """Simulate break-even trigger and application."""
        self.log_step("BREAK-EVEN SIMULATION", "Simulating break-even trigger scenario")
        
        if not self.position_opened:
            logger.error("❌ No position opened, cannot simulate break-even")
            return False
        
        try:
            # Get current position info
            position = self.manager.active_position
            if not position:
                logger.error("❌ No active position in PositionManager")
                return False
            
            logger.info(f"📊 Current position entry: {position.entry_price:.2f}")
            logger.info(f"🎯 Break-even trigger price: {position.breakeven_trigger_price:.2f}")
            
            # Simulate price movement to trigger break-even
            logger.info("🎭 SIMULATING PRICE MOVEMENT to trigger break-even...")
            
            # Create simulated candle that triggers BE
            simulated_high = position.breakeven_trigger_price + 10  # Price above BE trigger
            simulated_candle = pd.Series({
                'open': position.entry_price + 5,
                'high': simulated_high,
                'low': position.entry_price + 2,
                'close': position.entry_price + 8,
                'volume': 1000,
                'ATRr_14_5m': 15.0
            })
            
            logger.info(f"🕯️ Simulated candle: High={simulated_high:.2f} (triggers BE at {position.breakeven_trigger_price:.2f})")
            
            # Check if this triggers break-even
            current_candle, analysis = self.get_current_price_and_analysis()
            
            # Get trading signal with simulated high price
            signal = self.manager.get_trading_signal(simulated_candle, analysis, 1000)
            
            logger.info(f"📡 Trading signal received: {signal.get('action')}")
            
            # Check for BE instructions
            be_instructions = [inst for inst in signal.get('instructions', []) 
                             if inst['type'] == 'MOVE_SL_TO_BREAKEVEN']
            
            if be_instructions:
                logger.info("🎯 BREAK-EVEN TRIGGERED!")
                for instruction in be_instructions:
                    new_sl = instruction['new_sl_price']
                    logger.info(f"📤 Moving SL to break-even: {new_sl:.2f}")
                    
                    # Apply on exchange
                    modify_response = self.bybit_service.modify_position(
                        symbol=self.ticker,
                        stop_loss=new_sl
                    )
                    
                    if modify_response.get('ret_code') == 0:
                        logger.info("✅ Break-even applied on exchange!")
                        self.be_triggered = True
                        
                        # Update internal state
                        position.is_be = True
                        position.current_sl_price = new_sl
                        
                        return True
                    else:
                        logger.error(f"❌ Failed to apply BE on exchange: {modify_response}")
            else:
                logger.info("ℹ️ Break-even conditions not met with simulated price")
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error in break-even simulation: {e}")
            return False

    def simulate_trailing_stop_scenario(self) -> bool:
        """Simulate trailing stop activation and updates."""
        self.log_step("TRAILING STOP SIMULATION", "Simulating trailing stop activation and updates")
        
        if not self.position_opened:
            logger.error("❌ No position opened, cannot simulate trailing stop")
            return False
        
        try:
            position = self.manager.active_position
            if not position:
                logger.error("❌ No active position in PositionManager")
                return False
            
            logger.info(f"📊 Trailing trigger price: {position.trailing_trigger_price:.2f}")
            
            # Simulate price movement to activate trailing stop
            logger.info("🎭 SIMULATING PRICE MOVEMENT to activate trailing stop...")
            
            # Create simulated candle that triggers trailing stop
            trigger_high = position.trailing_trigger_price + 20
            simulated_candle = pd.Series({
                'open': position.entry_price + 10,
                'high': trigger_high,
                'low': position.entry_price + 8,
                'close': position.entry_price + 15,
                'volume': 1000,
                'ATRr_14_5m': 15.0
            })
            
            logger.info(f"🕯️ Simulated candle: High={trigger_high:.2f} (triggers trailing at {position.trailing_trigger_price:.2f})")
            
            # Get analysis for the signal
            current_candle, analysis = self.get_current_price_and_analysis()
            
            # Process the simulated candle
            signal = self.manager.get_trading_signal(simulated_candle, analysis, 1000)
            
            # Check for trailing stop activation
            trailing_instructions = [inst for inst in signal.get('instructions', []) 
                                   if inst['type'] == 'ACTIVATE_TRAILING_STOP']
            
            if trailing_instructions:
                logger.info("📈 TRAILING STOP ACTIVATED!")
                position.is_trailing = True
                self.trailing_activated = True
            
            # Now simulate trailing stop updates
            if position.is_trailing or self.trailing_activated:
                logger.info("🔄 SIMULATING TRAILING STOP UPDATES...")
                
                # Create multiple candles showing price advancement
                for i, close_price in enumerate([position.entry_price + 25, position.entry_price + 30, position.entry_price + 35]):
                    logger.info(f"📊 Simulation step {i+1}: Price at {close_price:.2f}")
                    
                    update_candle = pd.Series({
                        'open': close_price - 2,
                        'high': close_price + 3,
                        'low': close_price - 5,
                        'close': close_price,
                        'volume': 1000,
                        'ATRr_14_5m': 15.0
                    })
                    
                    # Calculate expected trailing SL
                    expected_new_sl = close_price - (15.0 * config.TRAILING_SL_DISTANCE_ATR)
                    logger.info(f"🎯 Expected new trailing SL: {expected_new_sl:.2f}")
                    
                    # Get trading signal
                    signal = self.manager.get_trading_signal(update_candle, analysis, 1000)
                    
                    # Check for trailing stop updates
                    update_instructions = [inst for inst in signal.get('instructions', []) 
                                         if inst['type'] == 'UPDATE_TRAILING_STOP']
                    
                    if update_instructions:
                        for instruction in update_instructions:
                            new_sl = instruction['new_sl_price']
                            logger.info(f"📤 Updating trailing SL to: {new_sl:.2f}")
                            
                            # Apply on exchange
                            modify_response = self.bybit_service.modify_position(
                                symbol=self.ticker,
                                stop_loss=new_sl
                            )
                            
                            if modify_response.get('ret_code') == 0:
                                logger.info("✅ Trailing stop updated on exchange!")
                                position.current_sl_price = new_sl
                            else:
                                logger.error(f"❌ Failed to update trailing SL: {modify_response}")
                    
                    time.sleep(2)  # Brief pause between updates
                
                logger.info("✅ Trailing stop simulation completed!")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error in trailing stop simulation: {e}")
            return False

    def simulate_position_closing(self) -> bool:
        """Simulate different position closing scenarios."""
        self.log_step("POSITION CLOSING SIMULATION", "Testing position closing mechanisms")
        
        if not self.position_opened:
            logger.error("❌ No position to close")
            return False
        
        try:
            # First, show current position status
            position = self.manager.active_position
            if position:
                logger.info("📊 FINAL POSITION STATUS:")
                logger.info(f"   Strategy: {position.strategy}")
                logger.info(f"   Entry Price: {position.entry_price:.2f}")
                logger.info(f"   Current SL: {position.current_sl_price:.2f}")
                logger.info(f"   Take Profit: {position.tp_price:.2f}")
                logger.info(f"   Break-Even Active: {position.is_be}")
                logger.info(f"   Trailing Active: {position.is_trailing}")
            
            # Close position via exchange
            logger.info("📤 Closing position on exchange...")
            close_response = self.bybit_service.close_position(self.ticker)
            
            if close_response.get('ret_code') == 0:
                logger.info("✅ Position closed successfully on exchange!")
                logger.info(f"📋 Closed {close_response.get('closed_side')} position of {close_response.get('closed_size')} {self.ticker}")
                
                # Clear internal position
                self.manager.clear_position()
                self.position_opened = False
                
                logger.info("🧹 Internal position state cleared")
                return True
            else:
                logger.error(f"❌ Failed to close position: {close_response}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")
            return False

    def run_full_simulation(self):
        """Run the complete position management simulation."""
        logger.info("🚀 STARTING COMPREHENSIVE POSITION MANAGEMENT SIMULATION")
        logger.info(f"🎯 Target Symbol: {self.ticker}")
        logger.info(f"🔧 Demo Mode: {self.bybit_service.demo}")
        
        success_steps = 0
        total_steps = 4
        
        try:
            # Step 1: Open position
            if self.simulate_position_opening():
                success_steps += 1
                logger.info("✅ Step 1/4: Position opening - SUCCESS")
            else:
                logger.error("❌ Step 1/4: Position opening - FAILED")
            
            time.sleep(3)
            
            # Step 2: Break-even simulation
            if self.simulate_breakeven_scenario():
                success_steps += 1
                logger.info("✅ Step 2/4: Break-even simulation - SUCCESS")
            else:
                logger.error("❌ Step 2/4: Break-even simulation - FAILED")
            
            time.sleep(3)
            
            # Step 3: Trailing stop simulation
            if self.simulate_trailing_stop_scenario():
                success_steps += 1
                logger.info("✅ Step 3/4: Trailing stop simulation - SUCCESS")
            else:
                logger.error("❌ Step 3/4: Trailing stop simulation - FAILED")
            
            time.sleep(3)
            
            # Step 4: Position closing
            if self.simulate_position_closing():
                success_steps += 1
                logger.info("✅ Step 4/4: Position closing - SUCCESS")
            else:
                logger.error("❌ Step 4/4: Position closing - FAILED")
            
        except Exception as e:
            logger.error(f"❌ Critical error during simulation: {e}")
        
        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info("🏁 SIMULATION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"✅ Successful steps: {success_steps}/{total_steps}")
        logger.info(f"📊 Success rate: {(success_steps/total_steps)*100:.1f}%")
        
        if success_steps == total_steps:
            logger.info("🎉 ALL SIMULATION STEPS COMPLETED SUCCESSFULLY!")
            logger.info("✅ Position management system is working correctly on demo account")
        else:
            logger.warning("⚠️ Some simulation steps failed - review logs above")
        
        logger.info(f"{'='*80}")

def main():
    """Main entry point for the position simulator."""
    try:
        simulator = PositionSimulator()
        simulator.run_full_simulation()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Simulation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in simulation: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())