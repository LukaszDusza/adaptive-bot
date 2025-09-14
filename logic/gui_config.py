"""
GUI Configuration Manager

Handles configuration loading, session state management, and database integration
for the Streamlit GUI application.
"""

import os
import logging
import streamlit as st
import pandas as pd
import concurrent.futures
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class GUIConfigManager:
    """Manages GUI configuration and session state"""
    
    @staticmethod
    def initialize_session_state():
        """Initialize Streamlit session state variables"""
        if 'config' not in st.session_state:
            st.session_state.config = GUIConfigManager.load_default_config()
            # Automatically load database configuration on startup
            try:
                GUIConfigManager.load_database_config()
            except Exception as e:
                logger.warning(f"Failed to load database configuration during initialization: {e}")
        
        if 'bot_status' not in st.session_state:
            st.session_state.bot_status = 'stopped'
        
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
        
        if 'live_data' not in st.session_state:
            st.session_state.live_data = pd.DataFrame()
        
        if 'regime_history' not in st.session_state:
            st.session_state.regime_history = []
    
    @staticmethod
    def load_default_config() -> Dict[str, Any]:
        """Load default configuration from environment variables"""
        api_key = os.getenv('BYBIT_API_KEY', '')
        api_secret = os.getenv('BYBIT_API_SECRET', '')
        testnet = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
        
        # Default trading configuration
        symbols = ['BTC/USDT', 'ETH/USDT']
        timeframe = '15m'
        initial_capital = 10000.0
        risk_per_trade = 0.02
        max_positions = 1
        max_daily_loss = 0.05
        max_drawdown = 0.15
        
        return {
            'bybit_api_key': api_key,
            'bybit_api_secret': api_secret,
            'bybit_testnet': testnet,
            'symbols': symbols,
            'timeframe': timeframe,
            'initial_capital': initial_capital,
            'risk_per_trade': risk_per_trade,
            'backtest_days': 365,
            'max_positions': max_positions,
            'max_daily_loss': max_daily_loss,
            'max_drawdown': max_drawdown,
            'database_loaded': False
        }

    @staticmethod
    def load_database_config():
        """Load configuration from database with improved error handling"""
        if st.session_state.config.get('database_loaded', False):
            return  # Already loaded
            
        try:
            results = GUIConfigManager._load_database_data_with_timeout()
            GUIConfigManager._apply_database_config(results)
            
            st.session_state.config['database_loaded'] = True
            logger.info("Database configuration loading completed")
            
        except ImportError as e:
            logger.info(f"Database models not available: {e}")
            st.session_state.config['database_loaded'] = True
        except Exception as e:
            logger.warning(f"Unexpected error loading database configuration: {e}")
            st.session_state.config['database_loaded'] = True

    @staticmethod
    def _load_database_data_with_timeout() -> Dict[str, Any]:
        """Load database data with timeout"""
        def load_database_data():
            """Database loading function to be executed with timeout"""
            results = {}
            
            # Load API keys
            try:
                from database.models import get_api_key_repository
                api_repo = get_api_key_repository()
                active_key = api_repo.get_active_api_key()
                
                if active_key:
                    results['api_key'] = active_key.api_key or ''
                    results['api_secret'] = active_key.api_secret or ''
                    results['testnet'] = bool(active_key.testnet)
                    logger.info(f"GUI loaded database API key: {active_key.name}")
                else:
                    logger.info("No active API key found in database")
                    results['api_key'] = None
            except Exception as e:
                logger.warning(f"Failed to load API keys from database: {e}")
                results['api_key'] = None
            
            # Load trading preferences
            try:
                from database.models import get_trading_preferences_repository
                prefs_repo = get_trading_preferences_repository()
                active_prefs = prefs_repo.get_active_preferences()
                
                if active_prefs:
                    prefs_dict = active_prefs.to_dict()
                    results['preferences'] = prefs_dict
                    logger.info(f"GUI loaded database trading preferences: {len(prefs_dict.get('selected_symbols', []))} symbols")
                else:
                    logger.info("No active trading preferences found in database, using defaults")
                    results['preferences'] = None
            except Exception as e:
                logger.warning(f"Failed to load trading preferences from database: {e}")
                results['preferences'] = None
            
            return results

        # Execute with 5-second timeout
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(load_database_data)
                return future.result(timeout=5.0)
        except concurrent.futures.TimeoutError:
            logger.warning("Database configuration loading timed out after 5 seconds")
            return {}

    @staticmethod
    def _apply_database_config(results: Dict[str, Any]):
        """Apply loaded database configuration to session state"""
        # Apply API key data
        if results.get('api_key') is not None:
            st.session_state.config['bybit_api_key'] = results['api_key']
            st.session_state.config['bybit_api_secret'] = results['api_secret']
            st.session_state.config['bybit_testnet'] = results['testnet']
        
        # Apply trading preferences
        if results.get('preferences') is not None:
            prefs_dict = results['preferences']
            config_updates = {
                'symbols': prefs_dict.get('selected_symbols', st.session_state.config.get('symbols', ['BTC/USDT', 'ETH/USDT'])),
                'timeframe': prefs_dict.get('timeframe', st.session_state.config.get('timeframe', '15m')),
                'initial_capital': float(prefs_dict.get('initial_capital', st.session_state.config.get('initial_capital', 10000.0))),
                'risk_per_trade': float(prefs_dict.get('risk_per_trade', st.session_state.config.get('risk_per_trade', 0.02))),
                'max_positions': int(prefs_dict.get('max_positions', st.session_state.config.get('max_positions', 1))),
                'max_daily_loss': float(prefs_dict.get('max_daily_loss', st.session_state.config.get('max_daily_loss', 0.05))),
                'max_drawdown': float(prefs_dict.get('max_drawdown', st.session_state.config.get('max_drawdown', 0.15)))
            }
            
            for key, value in config_updates.items():
                st.session_state.config[key] = value

    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get current configuration"""
        return st.session_state.config if 'config' in st.session_state else {}

    @staticmethod
    def update_config(updates: Dict[str, Any]):
        """Update configuration with new values"""
        if 'config' in st.session_state:
            st.session_state.config.update(updates)

    @staticmethod
    def generate_env_file():
        """Generate .env file from current configuration"""
        config = GUIConfigManager.get_config()
        env_content = f"""# Bybit API Configuration
BYBIT_API_KEY={config.get('bybit_api_key', '')}
BYBIT_API_SECRET={config.get('bybit_api_secret', '')}
BYBIT_TESTNET={str(config.get('bybit_testnet', True)).lower()}

# Trading Configuration
SYMBOLS={','.join(config.get('symbols', ['BTC/USDT', 'ETH/USDT']))}
TIMEFRAME={config.get('timeframe', '15m')}
INITIAL_CAPITAL={config.get('initial_capital', 10000.0)}
RISK_PER_TRADE={config.get('risk_per_trade', 0.02)}
MAX_POSITIONS={config.get('max_positions', 1)}
MAX_DAILY_LOSS={config.get('max_daily_loss', 0.05)}
MAX_DRAWDOWN={config.get('max_drawdown', 0.15)}
"""
        with open('.env', 'w') as f:
            f.write(env_content)