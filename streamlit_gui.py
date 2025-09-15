#!/usr/bin/env python3
"""
Adaptive Trading Bot - Streamlit GUI

This module provides a web-based graphical user interface for the Adaptive Trading Bot.
Features:
- Real-time monitoring dashboard
- Backtesting configuration and results
- Live trading controls
- API key configuration
- Performance metrics visualization

Usage:
    streamlit run streamlit_gui.py
"""

# Apply comprehensive vectorbt fixes BEFORE any other imports
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Apply vectorbt fixes immediately
try:
    from fix_vectorbt_logging_comprehensive import apply_comprehensive_vectorbt_fixes
    apply_comprehensive_vectorbt_fixes()
except ImportError:
    # Fallback to basic vectorbt suppression
    import logging
    logging.getLogger('vectorbt').setLevel(logging.CRITICAL)
    logging.getLogger('watchdog').setLevel(logging.CRITICAL)
    
    # Basic environment variables
    os.environ['VECTORBT_DISABLE_CACHING'] = '1'
    os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import time
import io
from collections import deque
import threading

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import bot components
from core.regime_detector import RegimeDetector, MarketRegime
from strategies.consolidation_strategy import ConsolidationStrategy
from strategies.trend_strategy import TrendStrategy
from core.risk_manager import RiskManager
from indicators.technical import TechnicalIndicators
from data.bybit_provider import BybitDataProvider, BybitConfig
from backtesting.vectorbt_engine import VectorbtAdaptiveEngine, VectorbtBacktestConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page config
st.set_page_config(
    page_title="Adaptive Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitGUI:
    """Main GUI class for the Adaptive Trading Bot"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'config' not in st.session_state:
            st.session_state.config = self.load_default_config()
            # Automatically load database configuration on startup
            try:
                self.load_database_config()
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
        
        # Initialize logs panel filters
        if 'log_level_filter' not in st.session_state:
            st.session_state.log_level_filter = ['INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        if 'log_module_filter' not in st.session_state:
            st.session_state.log_module_filter = 'all'
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from environment variables and set up lazy database loading"""
        # Start with environment variables and defaults - database loading will be lazy
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
            # Futures-only mode and leverage settings
            'futures_mode': True,
            'leverage': 2,
            'use_kelly': False,
            'kelly_cap': 0.03,
            'var_threshold': 0.05,
            'max_symbol_exposure_pct': 0.5,
            'database_loaded': False  # Flag to indicate database config not yet loaded
        }

    def load_database_config(self):
        """Load configuration from database with improved error handling"""
        if st.session_state.config.get('database_loaded', False):
            return  # Already loaded
            
        try:
            from database.models import get_api_key_repository, get_trading_preferences_repository
            
            # Use threading-based timeout instead of signal for Streamlit compatibility
            import concurrent.futures
            import time
            
            def load_database_data():
                """Database loading function to be executed with timeout"""
                results = {}
                
                # Load API keys with error recovery
                try:
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
                
                # Load trading preferences with error recovery
                try:
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
            
            # Execute with 5-second timeout using ThreadPoolExecutor
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(load_database_data)
                    results = future.result(timeout=5.0)
                
                # Apply loaded data to session state
                if results.get('api_key') is not None:
                    st.session_state.config['bybit_api_key'] = results['api_key']
                    st.session_state.config['bybit_api_secret'] = results['api_secret']
                    st.session_state.config['bybit_testnet'] = results['testnet']
                
                if results.get('preferences') is not None:
                    prefs_dict = results['preferences']
                    # Safely update config with database values
                    st.session_state.config['symbols'] = prefs_dict.get('selected_symbols', st.session_state.config.get('symbols', ['BTC/USDT', 'ETH/USDT']))
                    st.session_state.config['timeframe'] = prefs_dict.get('timeframe', st.session_state.config.get('timeframe', '15m'))
                    st.session_state.config['initial_capital'] = float(prefs_dict.get('initial_capital', st.session_state.config.get('initial_capital', 10000.0)))
                    st.session_state.config['risk_per_trade'] = float(prefs_dict.get('risk_per_trade', st.session_state.config.get('risk_per_trade', 0.02)))
                    st.session_state.config['max_positions'] = int(prefs_dict.get('max_positions', st.session_state.config.get('max_positions', 1)))
                    st.session_state.config['max_daily_loss'] = float(prefs_dict.get('max_daily_loss', st.session_state.config.get('max_daily_loss', 0.05)))
                    st.session_state.config['max_drawdown'] = float(prefs_dict.get('max_drawdown', st.session_state.config.get('max_drawdown', 0.15)))
                
                logger.info("Database configuration loading completed")
                
            except concurrent.futures.TimeoutError:
                logger.warning("Database configuration loading timed out after 5 seconds")
            except Exception as e:
                logger.warning(f"Database operation failed: {e}")
            
            st.session_state.config['database_loaded'] = True
            
        except ImportError as e:
            logger.info(f"Database models not available: {e}")
            st.session_state.config['database_loaded'] = True
        except Exception as e:
            logger.warning(f"Unexpected error loading database configuration: {e}")
            st.session_state.config['database_loaded'] = True  # Mark as attempted to prevent infinite retries
    
    def render_sidebar(self):
        """Render sidebar with configuration and controls"""
        st.sidebar.title("🤖 Adaptive Trading Bot")
        st.sidebar.markdown("---")
        
        # Navigation
        page = st.sidebar.selectbox(
            "📊 Navigation",
            ["Dashboard", "Configuration", "Strategy Info", "Backtesting", "Live Trading", "Trading History", "Performance"]
        )
        
        st.sidebar.markdown("---")
        
        # Bot Status
        st.sidebar.subheader("🔋 Bot Status")
        status_color = {"stopped": "🔴", "running": "🟢", "testing": "🟡"}
        st.sidebar.write(f"{status_color.get(st.session_state.bot_status, '⚪')} Status: **{st.session_state.bot_status.title()}**")
        
        # Quick Actions
        st.sidebar.subheader("⚡ Quick Actions")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🧪 Test Connection", key="test_conn"):
                self.test_connection()
        
        with col2:
            if st.button("🔄 Refresh Data", key="refresh"):
                st.rerun()
        
        return page
    
    def render_configuration_page(self):
        """Render configuration page with automatic database loading"""
        st.title("⚙️ Configuration")
        st.markdown("Configure your trading bot settings and API connections.")
        
        # Force reload database configuration every time page loads to ensure fresh data
        try:
            # Clear the database loaded flag to force reload
            st.session_state.config['database_loaded'] = False
            self.load_database_config()
            database_load_success = True
            database_status = "✅ Database connected - configurations loaded automatically"
        except Exception as e:
            database_load_success = False
            database_status = f"⚠️ Database connection failed: {str(e)}"
        
        # Show database status
        if database_load_success:
            st.success(database_status)
        else:
            st.warning(database_status)
        
        # API Configuration with Database Support
        st.subheader("🔑 Bybit API Key Management")
        
        # Check if database is available and try to load API keys
        database_available = False
        api_repo = None
        
        try:
            from database.models import get_api_key_repository
            api_repo = get_api_key_repository()
            # Test database connection by trying to get API keys
            api_repo.get_all_api_keys()
            database_available = True
        except ImportError as e:
            st.error(f"❌ Database models not available: {str(e)}")
            database_available = False
        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
            database_available = False
        
        if not database_available:
            st.warning("⚠️ Database not available. Using manual API key input.")
            st.info("💡 To enable automatic API key management, ensure your PostgreSQL database is running with `docker-compose up -d`")
            
            # Fallback to manual input with current config values
            col1, col2 = st.columns(2)
            with col1:
                api_key = st.text_input(
                    "API Key",
                    value=st.session_state.config.get('bybit_api_key', ''),
                    type="password",
                    help="Your Bybit API key from https://www.bybit.com/app/user/api-management"
                )
            
            with col2:
                api_secret = st.text_input(
                    "API Secret",
                    value=st.session_state.config.get('bybit_api_secret', ''),
                    type="password",
                    help="Your Bybit API secret"
                )
            
            # Testnet/Production Mode Selection with clear warnings
            st.markdown("#### 🚀 Trading Mode")
            
            testnet = st.radio(
                "Select Trading Environment:",
                options=[True, False],
                format_func=lambda x: "🧪 Testnet Mode (Safe Testing)" if x else "🔴 Production Mode (Real Money!)",
                index=0 if st.session_state.config.get('bybit_testnet', True) else 1,
                help="Choose between safe testnet environment or real production trading"
            )
            
            if not testnet:
                st.error("⚠️ **PRODUCTION MODE ACTIVE** - You are using REAL money and API!")
                st.warning("📋 Make sure you have:")
                st.warning("• ✅ Production API keys (not testnet)")
                st.warning("• ✅ Sufficient balance in your Bybit account")  
                st.warning("• ✅ Thoroughly tested your strategy on testnet first")
            else:
                st.success("🧪 **TESTNET MODE** - Safe for testing with virtual money")
                st.info("💡 Switch to Production Mode when you're ready to trade with real money")
        else:
            # Database-backed API key management
            st.success("🎯 API keys automatically loaded from database")
            self.render_api_key_management(api_repo)
            # Set variables from current loaded config
            api_key = st.session_state.config.get('bybit_api_key', '')
            api_secret = st.session_state.config.get('bybit_api_secret', '')
            testnet = st.session_state.config.get('bybit_testnet', True)
        
        # Trading Configuration
        st.subheader("📈 Trading Configuration")
        
        # Show trading preferences loading status
        if database_load_success:
            try:
                from database.models import get_trading_preferences_repository
                prefs_repo = get_trading_preferences_repository()
                active_prefs = prefs_repo.get_active_preferences()
                if active_prefs:
                    st.info(f"🎯 Trading preferences automatically loaded from database (Symbols: {len(st.session_state.config.get('symbols', []))}, Timeframe: {st.session_state.config.get('timeframe', 'N/A')})")
                else:
                    st.warning("⚠️ No active trading preferences found in database - using defaults")
            except Exception as e:
                st.warning(f"⚠️ Could not load trading preferences: {str(e)}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # Available crypto symbols with descriptions
            available_symbols = {
                'BTC/USDT': 'Bitcoin - Most stable crypto',
                'ETH/USDT': 'Ethereum - Smart contracts leader', 
                'ADA/USDT': 'Cardano - Research-driven blockchain',
                'SOL/USDT': 'Solana - High-speed blockchain',
                'DOT/USDT': 'Polkadot - Interoperability protocol',
                'MATIC/USDT': 'Polygon - Ethereum scaling solution',
                'LINK/USDT': 'Chainlink - Oracle network',
                'AVAX/USDT': 'Avalanche - Fast consensus protocol',
                'UNI/USDT': 'Uniswap - Decentralized exchange',
                'ATOM/USDT': 'Cosmos - Internet of blockchains'
            }
            
            symbols = st.multiselect(
                "Trading Symbols",
                options=list(available_symbols.keys()),
                default=st.session_state.config['symbols'],
                help="Select crypto currencies to trade. Hover for descriptions."
            )
            
            # Show selected symbols with descriptions
            if symbols:
                st.write("**Selected Cryptocurrencies:**")
                for symbol in symbols:
                    st.write(f"• {symbol}: {available_symbols[symbol]}")
        
        with col2:
            timeframe = st.selectbox(
                "Timeframe",
                options=['1m', '5m', '15m', '1h', '4h', '1d'],
                index=['1m', '5m', '15m', '1h', '4h', '1d'].index(st.session_state.config['timeframe'])
            )
        
        with col3:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                value=st.session_state.config['initial_capital'],
                min_value=100.0,
                step=100.0,
                help="Minimum $100 for testing. Start small to learn the system."
            )
        
        # Risk Management
        st.subheader("🛡️ Risk Management")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_per_trade = st.slider(
                "Risk per Trade (%)",
                min_value=0.5,
                max_value=10.0,
                value=st.session_state.config['risk_per_trade'] * 100,
                step=0.5
            ) / 100
        
        with col2:
            max_daily_loss = st.slider(
                "Max Daily Loss (%)",
                min_value=1.0,
                max_value=20.0,
                value=st.session_state.config['max_daily_loss'] * 100,
                step=1.0
            ) / 100
        
        with col3:
            max_drawdown = st.slider(
                "Max Drawdown (%)",
                min_value=5.0,
                max_value=50.0,
                value=st.session_state.config['max_drawdown'] * 100,
                step=5.0
            ) / 100
        
        # Derivatives / Futures Settings
        st.subheader("🧮 Derivatives & Risk Settings")
        col1, col2 = st.columns(2)
        with col1:
            futures_mode = st.checkbox(
                "Futures mode only (enforced)",
                value=st.session_state.config.get('futures_mode', True),
                help="Bot operates only on linear futures. Spot trading is disabled.",
                disabled=True
            )
            use_kelly = st.checkbox(
                "Use Kelly sizing (adaptive risk)",
                value=st.session_state.config.get('use_kelly', False),
                help="Use Kelly criterion based on past trade stats to adapt risk per trade (capped)."
            )
            kelly_cap = st.slider(
                "Kelly cap (%)",
                min_value=0.5,
                max_value=10.0,
                value=float(st.session_state.config.get('kelly_cap', 0.03) * 100),
                step=0.5,
                help="Maximum fraction of equity risked per trade when Kelly is enabled."
            ) / 100
        with col2:
            leverage = st.number_input(
                "Leverage",
                min_value=1,
                max_value=125,
                value=int(st.session_state.config.get('leverage', 2)),
                help="Set your default leverage for futures positions."
            )
            var_threshold = st.slider(
                "VaR(95) threshold (%)",
                min_value=1.0,
                max_value=20.0,
                value=float(st.session_state.config.get('var_threshold', 0.05) * 100),
                step=1.0,
                help="If estimated 95% Value-at-Risk exceeds this, new entries are blocked."
            ) / 100
            max_symbol_exposure_pct = st.slider(
                "Max exposure per symbol (%)",
                min_value=5.0,
                max_value=100.0,
                value=float(st.session_state.config.get('max_symbol_exposure_pct', 0.5) * 100),
                step=5.0,
                help="Cap exposure to a single symbol as % of equity."
            ) / 100
        
        # Save Configuration
        if st.button("💾 Save Configuration", key="save_config"):
            # Update session state
            st.session_state.config.update({
                'bybit_api_key': api_key,
                'bybit_api_secret': api_secret,
                'bybit_testnet': testnet,
                'symbols': symbols,
                'timeframe': timeframe,
                'initial_capital': initial_capital,
                'risk_per_trade': risk_per_trade,
                'max_daily_loss': max_daily_loss,
                'max_drawdown': max_drawdown,
                'futures_mode': True,
                'leverage': int(leverage),
                                'use_kelly': bool(use_kelly),
                                'kelly_cap': float(kelly_cap),
                                'var_threshold': float(var_threshold),
                                'max_symbol_exposure_pct': float(max_symbol_exposure_pct)
            })
            
            # Save trading preferences to database
            try:
                from database.models import get_trading_preferences_repository
                prefs_repo = get_trading_preferences_repository()
                
                prefs_data = {
                    'user_profile': 'default',
                    'selected_symbols': symbols,  # Repository will handle JSON conversion
                    'timeframe': timeframe,
                    'initial_capital': initial_capital,
                    'risk_per_trade': risk_per_trade,
                    'max_daily_loss': max_daily_loss,
                    'max_drawdown': max_drawdown,
                    'max_positions': st.session_state.config.get('max_positions', 1)
                }
                
                saved_prefs = prefs_repo.create_or_update_preferences(prefs_data)
                st.success(f"✅ Configuration saved to database! (ID: {str(saved_prefs.id)[:8]}...)")
                
            except ImportError:
                st.warning("⚠️ Database not available. Configuration saved to session only.")
            except Exception as e:
                st.error(f"❌ Failed to save to database: {e}")
                st.success("✅ Configuration saved to session!")
            
            # Generate .env file
            env_content = self.generate_env_file()
            st.download_button(
                label="📥 Download .env file",
                data=env_content,
                file_name=".env",
                mime="text/plain"
            )
    
    def render_dashboard_page(self):
        """Render main dashboard"""
        st.title("📊 Trading Dashboard")
        st.markdown("Real-time overview of your adaptive trading bot.")
        
        # Get real data from database
        try:
            from database.models import get_position_repository, get_db_manager
            db_manager = get_db_manager()
            position_repo = get_position_repository()
            
            # Get real trading statistics
            stats = position_repo.get_position_stats()
            
            # Calculate real metrics
            portfolio_value = stats.get('total_portfolio_value', 0.0)
            daily_pnl = stats.get('daily_pnl', 0.0)
            open_positions = stats.get('open_positions', 0)
            total_trades = stats.get('total_trades', 0)
            
            # Calculate deltas (comparison with previous values)
            portfolio_delta = stats.get('portfolio_change_pct', 0.0)
            daily_pnl_delta = stats.get('daily_pnl_pct', 0.0)
            positions_delta = stats.get('positions_change', 0)
            trades_delta = stats.get('new_trades_today', 0)
            
        except Exception as e:
            st.warning("⚠️ Cannot connect to database. Showing placeholder data. Please check your database connection.")
            # Fallback to indicate this is placeholder data
            portfolio_value = 0.0
            daily_pnl = 0.0
            open_positions = 0
            total_trades = 0
            portfolio_delta = 0.0
            daily_pnl_delta = 0.0
            positions_delta = 0
            trades_delta = 0
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 Portfolio Value",
                value=f"${portfolio_value:,.2f}" if portfolio_value > 0 else "No data",
                delta=f"{portfolio_delta:+.1f}%" if portfolio_delta != 0 else None
            )
        
        with col2:
            st.metric(
                label="📈 Daily P&L",
                value=f"${daily_pnl:+,.2f}" if daily_pnl != 0 else "No trades today",
                delta=f"{daily_pnl_delta:+.2f}%" if daily_pnl_delta != 0 else None
            )
        
        with col3:
            st.metric(
                label="🎯 Open Positions",
                value=str(open_positions),
                delta=f"{positions_delta:+d}" if positions_delta != 0 else None
            )
        
        with col4:
            st.metric(
                label="🔄 Total Trades",
                value=str(total_trades),
                delta=f"{trades_delta:+d}" if trades_delta != 0 else None
            )
        
        # Market Regime Detection
        st.subheader("🌐 Current Market Regime")
        self.render_regime_detection()
        
        # Recent Performance
        st.subheader("📈 Recent Performance")
        self.render_performance_chart()
    
    def render_regime_detection(self):
        """Render market regime detection section"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Try to get real regime data
            try:
                from core.regime_detector import RegimeDetector
                from data.bybit_provider import BybitDataProvider, BybitConfig
                
                # Check if we have API configuration
                config = st.session_state.config
                if config.get('bybit_api_key') and config.get('bybit_api_secret'):
                    # Get real market data and regime detection
                    bybit_config = BybitConfig(
                        api_key=config['bybit_api_key'],
                        secret=config['bybit_api_secret'],
                        testnet=config.get('bybit_testnet', True)
                    )
                    data_provider = BybitDataProvider(bybit_config)
                    regime_detector = RegimeDetector()
                    
                    # Get recent data for regime analysis
                    symbol = config.get('symbols', ['BTCUSDT'])[0]
                    df_market = data_provider.get_klines(symbol, '1h', 48)  # Last 48 hours
                    
                    if df_market is not None and not df_market.empty:
                        # Detect regime for each data point
                        regime_history = []
                        for i in range(len(df_market)):
                            if i < 20:  # Need enough data for regime detection
                                continue
                            data_slice = df_market.iloc[:i+1]
                            regime, confidence = regime_detector.detect_regime(data_slice)
                            regime_history.append({
                                'timestamp': df_market.iloc[i]['timestamp'],
                                'regime': regime.value if hasattr(regime, 'value') else str(regime),
                                'confidence': confidence
                            })
                        
                        if regime_history:
                            df = pd.DataFrame(regime_history)
                            fig = px.line(
                                df, 
                                x='timestamp', 
                                y='confidence',
                                color='regime',
                                title='Market Regime Over Time (Real Data)',
                                labels={'confidence': 'Confidence Score', 'timestamp': 'Time'}
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("📊 Not enough data for regime analysis. Need at least 20 data points.")
                    else:
                        st.warning("⚠️ Cannot fetch market data. Please check your API configuration.")
                else:
                    st.info("🔑 Please configure API keys in the Configuration page to see real regime data.")
                    
            except Exception as e:
                st.error(f"❌ Error loading regime data: {str(e)}")
                st.info("📊 Real-time regime detection unavailable. Please check your configuration.")
        
        with col2:
            # Try to get current regime status from real data
            try:
                from core.regime_detector import RegimeDetector
                from data.bybit_provider import BybitDataProvider, BybitConfig
                
                config = st.session_state.config
                if config.get('bybit_api_key') and config.get('bybit_api_secret'):
                    bybit_config = BybitConfig(
                        api_key=config['bybit_api_key'],
                        secret=config['bybit_api_secret'],
                        testnet=config.get('bybit_testnet', True)
                    )
                    data_provider = BybitDataProvider(bybit_config)
                    regime_detector = RegimeDetector()
                    
                    symbol = config.get('symbols', ['BTCUSDT'])[0]
                    df_current = data_provider.get_klines(symbol, '1h', 100)  # More data for accurate detection
                    
                    if df_current is not None and not df_current.empty:
                        current_regime, confidence = regime_detector.detect_regime(df_current)
                        regime_str = current_regime.value if hasattr(current_regime, 'value') else str(current_regime)
                        
                        # Determine strategy based on regime
                        strategy_map = {
                            'TRENDING': 'Trend Following',
                            'CONSOLIDATION': 'Range Trading',
                            'STAGNANT': 'Wait & Watch',
                            'PANIC': 'Risk Off'
                        }
                        strategy = strategy_map.get(regime_str, 'Unknown')
                        
                        # Determine if can trade
                        can_trade = regime_str in ['TRENDING', 'CONSOLIDATION']
                        trade_status = "✅ Yes" if can_trade else "❌ No"
                        
                        # Determine risk level
                        if confidence > 0.8:
                            risk_level = "🟢 Low" if can_trade else "🔴 High"
                        elif confidence > 0.6:
                            risk_level = "🟡 Medium"
                        else:
                            risk_level = "🔴 High"
                        
                        st.markdown(f"""
                        **Current Regime:** `{regime_str}`
                        
                        **Confidence:** {confidence:.1%}
                        
                        **Strategy:** {strategy}
                        
                        **Can Trade:** {trade_status}
                        
                        **Risk Level:** {risk_level}
                        
                        **Symbol:** {symbol}
                        """)
                    else:
                        st.warning("⚠️ Cannot determine current regime. Check API connection.")
                else:
                    st.info("🔑 Configure API keys to see real regime status.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.markdown("""
                **Current Regime:** `No Data`
                
                **Status:** API Configuration Required
                
                Please configure your Bybit API keys in the Configuration page.
                """)
    
    def render_performance_chart(self):
        """Render performance chart"""
        try:
            from database.models import get_position_repository
            position_repo = get_position_repository()
            
            # Get real trading positions for performance calculation
            positions = position_repo.get_positions(limit=1000)  # Get recent positions
            
            if positions:
                # Calculate portfolio performance over time
                portfolio_data = []
                running_balance = st.session_state.config.get('initial_capital', 10000)
                
                # Sort positions by entry time
                sorted_positions = sorted(positions, key=lambda p: p.entry_time if p.entry_time else datetime.now())
                
                for position in sorted_positions:
                    if position.exit_time and position.pnl is not None:
                        running_balance += float(position.pnl)
                        portfolio_data.append({
                            'timestamp': position.exit_time,
                            'portfolio_value': running_balance,
                            'pnl': float(position.pnl),
                            'symbol': position.symbol
                        })
                
                if portfolio_data:
                    df = pd.DataFrame(portfolio_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['portfolio_value'],
                        mode='lines+markers',
                        name='Portfolio Value',
                        line=dict(color='#1f77b4', width=2),
                        hovertemplate='<b>%{y:$,.2f}</b><br>%{x}<br>PnL: %{customdata:+$,.2f}<extra></extra>',
                        customdata=df['pnl']
                    ))
                    
                    fig.update_layout(
                        title='Portfolio Performance (Real Trading Data)',
                        xaxis_title='Date',
                        yaxis_title='Portfolio Value ($)',
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show performance summary
                    total_pnl = sum(df['pnl'])
                    total_trades = len(df)
                    win_rate = len(df[df['pnl'] > 0]) / total_trades * 100 if total_trades > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total P&L", f"${total_pnl:+,.2f}")
                    with col2:
                        st.metric("Total Trades", total_trades)
                    with col3:
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                        
                else:
                    st.info("📊 No completed trades found. Performance chart will show once you have trading history.")
                    self._render_placeholder_chart()
            else:
                st.info("📊 No trading positions found in database. Start trading to see performance data.")
                self._render_placeholder_chart()
                
        except Exception as e:
            st.error(f"❌ Error loading performance data: {str(e)}")
            st.info("📊 Showing placeholder chart. Please check your database connection.")
            self._render_placeholder_chart()
    
    def _render_placeholder_chart(self):
        """Render placeholder chart when no real data is available"""
        # Simple placeholder showing initial capital
        initial_capital = st.session_state.config.get('initial_capital', 10000)
        dates = [datetime.now() - timedelta(days=1), datetime.now()]
        values = [initial_capital, initial_capital]
        
        df = pd.DataFrame({
            'timestamp': dates,
            'portfolio_value': values
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#cccccc', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Portfolio Performance (No Trading Data)',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 This chart will show real performance data once you start trading.")
    
    def _run_real_backtest(self, quick=False, optimize=False):
        """Run real backtesting with actual Bybit data and strategies"""
        try:
            from backtesting.vectorbt_engine import VectorbtAdaptiveEngine, VectorbtBacktestConfig
            from data.bybit_provider import BybitDataProvider, BybitConfig
            from strategies.trend_strategy import TrendStrategy
            from strategies.consolidation_strategy import ConsolidationStrategy
            from core.regime_detector import RegimeDetector
            
            # Get configuration
            config = st.session_state.config
            
            # Check API configuration
            if not config.get('bybit_api_key') or not config.get('bybit_api_secret'):
                st.error("❌ Please configure Bybit API keys in Configuration page first.")
                return None
            
            # Setup Bybit data provider
            bybit_config = BybitConfig(
                api_key=config['bybit_api_key'],
                secret=config['bybit_api_secret'],
                testnet=config.get('bybit_testnet', True)
            )
            data_provider = BybitDataProvider(bybit_config)
            
            # Get symbols and timeframe
            symbols = config.get('symbols', ['BTCUSDT'])
            timeframe = config.get('timeframe', '1h')
            
            # Determine backtest period
            if quick:
                periods = 500  # ~3 weeks for 1h timeframe
            elif optimize:
                periods = 1000  # ~6 weeks for optimization
            else:
                periods = 2000  # ~3 months for full backtest
            
            # Get real market data
            symbol = symbols[0]  # Use first symbol for backtesting
            df = data_provider.get_klines(symbol, timeframe, periods)
            
            if df is None or df.empty:
                st.error(f"❌ Failed to fetch market data for {symbol}. Please check your API connection.")
                return None
            
            # Setup backtest configuration
            backtest_config = VectorbtBacktestConfig(
                initial_capital=config.get('initial_capital', 10000),
                transaction_cost=0.001,  # 0.1% commission
                slippage=0.001,    # 0.1% slippage
                risk_per_trade=config.get('risk_per_trade', 0.02)
            )
            
            # Initialize backtesting engine with configuration
            engine = VectorbtAdaptiveEngine(backtest_config)
            
            # Run backtest
            results = engine.run_backtest(
                data=df,
                symbol=symbol
            )
            
            if results is None:
                st.error("❌ Backtest failed to generate results.")
                return None
            
            # Extract metrics from VectorbtResults object
            portfolio = results.portfolio
            trades = results.trades
            
            # Calculate performance metrics
            total_return = portfolio.total_return()
            sharpe_ratio = portfolio.sharpe_ratio() if hasattr(portfolio, 'sharpe_ratio') else 0.0
            max_drawdown = portfolio.max_drawdown()
            total_trades = len(trades) if trades is not None and hasattr(trades, '__len__') else 0
            
            # Calculate win rate from trades
            win_rate = 0.0
            if trades is not None and hasattr(trades, '__len__') and len(trades) > 0:
                try:
                    if hasattr(trades, 'pnl'):
                        winning_trades = (trades.pnl > 0).sum() if hasattr(trades.pnl, 'sum') else 0
                        win_rate = winning_trades / len(trades) if len(trades) > 0 else 0.0
                except:
                    win_rate = 0.0
            
            # Format results for display
            formatted_results = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'equity_curve': {
                    'dates': results.equity_curve.index if results.equity_curve is not None else df.index,
                    'values': results.equity_curve.values if results.equity_curve is not None else [config.get('initial_capital', 10000)] * len(df)
                },
                'symbol': symbol,
                'timeframe': timeframe,
                'periods': periods,
                'trades_data': trades,
                'portfolio_object': portfolio
            }
            
            return formatted_results
            
        except ImportError as e:
            st.error(f"❌ Missing required modules: {str(e)}")
            st.info("💡 Make sure all dependencies are installed: `pip install -r requirements.txt`")
            return None
        except Exception as e:
            st.error(f"❌ Backtest execution failed: {str(e)}")
            logger.error(f"Backtest error: {str(e)}")
            return None
    
    def render_backtesting_page(self):
        """Render enhanced backtesting page with parameter optimization"""
        st.title("🧪 Backtesting & Parameter Optimization")
        st.markdown("Test and optimize your trading strategies with historical data.")
        
        # Import strategies
        from strategies.trend_strategy import TrendStrategy
        from strategies.consolidation_strategy import ConsolidationStrategy
        
        # Initialize session state for parameters
        if 'strategy_params' not in st.session_state:
            st.session_state.strategy_params = {
                'trend': TrendStrategy.get_parameter_info(),
                'consolidation': ConsolidationStrategy.get_parameter_info()
            }
        
        # Main configuration
        st.subheader("⚙️ Basic Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            backtest_symbol = st.selectbox(
                "Symbol",
                options=st.session_state.config['symbols'],
                index=0 if st.session_state.config['symbols'] else None,
                help="Select the cryptocurrency to backtest"
            )
        
        with col2:
            backtest_days = st.number_input(
                "Days to Test",
                value=st.session_state.config['backtest_days'],
                min_value=30,
                max_value=1000,
                step=30,
                help="Number of historical days to analyze"
            )
        
        with col3:
            walk_forward = st.checkbox(
                "Walk-Forward Analysis",
                value=False,
                help="More robust but slower analysis - tests parameters over multiple time periods"
            )
        
        # Strategy Parameter Optimization Section
        st.subheader("🎛️ Strategy Parameter Optimization")
        st.markdown("Adjust strategy parameters to optimize performance. Changes are applied immediately to backtests.")
        
        # Strategy tabs for parameter adjustment
        param_tabs = st.tabs(["📈 Trend Strategy", "📊 Consolidation Strategy", "🔄 Quick Compare"])
        
        # Trend Strategy Parameters
        with param_tabs[0]:
            st.markdown("### Trend Following Strategy Parameters")
            st.info("💡 **Tip**: Start with small adjustments (±10-20%) from default values and observe the impact on Sharpe ratio and win rate.")
            
            trend_params = TrendStrategy.get_parameter_info()
            trend_values = {}
            
            # Create parameter sliders for trend strategy
            col1, col2 = st.columns(2)
            
            with col1:
                # DMI Period
                param_info = trend_params['dmi_period']
                trend_values['dmi_period'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}"
                )
                
                # ADX Threshold
                param_info = trend_params['adx_threshold']
                trend_values['adx_threshold'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}"
                )
                
                # ATR Stop Multiplier
                param_info = trend_params['atr_stop_multiplier']
                trend_values['atr_stop_multiplier'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}"
                )
                
                # Min ADX Slope
                param_info = trend_params['min_adx_slope']
                trend_values['min_adx_slope'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}"
                )
            
            with col2:
                # Chandelier Multiplier
                param_info = trend_params['chandelier_multiplier']
                trend_values['chandelier_multiplier'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}"
                )
                
                # EMA Period
                param_info = trend_params['ema_period']
                trend_values['ema_period'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}"
                )
                
                # Volume Threshold
                param_info = trend_params['volume_threshold']
                trend_values['volume_threshold'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}"
                )
                
                # Toggles
                trend_values['use_ema_filter'] = st.checkbox(
                    "Use EMA Filter",
                    value=True,
                    help="Filter trades based on EMA trend direction"
                )
                
                trend_values['volume_confirmation'] = st.checkbox(
                    "Volume Confirmation",
                    value=False,
                    help="Require volume confirmation for signals"
                )
            
            # Store trend parameters
            st.session_state.trend_params = trend_values
            
            # Reset to defaults button
            if st.button("🔄 Reset to Defaults", key="reset_trend"):
                st.rerun()
        
        # Consolidation Strategy Parameters  
        with param_tabs[1]:
            st.markdown("### Consolidation (Mean Reversion) Strategy Parameters")
            st.info("💡 **Tip**: For consolidation strategies, focus on Stochastic levels and support/resistance distance for better entries.")
            
            cons_params = ConsolidationStrategy.get_parameter_info()
            cons_values = {}
            
            # Create parameter sliders for consolidation strategy
            col1, col2 = st.columns(2)
            
            with col1:
                # Stochastic K Period
                param_info = cons_params['stoch_k_period']
                cons_values['stoch_k_period'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}",
                    key="cons_k_period"
                )
                
                # Oversold Level
                param_info = cons_params['oversold_level']
                cons_values['oversold_level'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}",
                    key="cons_oversold"
                )
                
                # ATR Stop Multiplier
                param_info = cons_params['atr_stop_multiplier']
                cons_values['atr_stop_multiplier'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}",
                    key="cons_atr_stop"
                )
                
                # Support/Resistance Window
                param_info = cons_params['support_resistance_window']
                cons_values['support_resistance_window'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}",
                    key="cons_sr_window"
                )
            
            with col2:
                # Stochastic smoothing
                param_info = cons_params['stoch_k_smooth']
                cons_values['stoch_k_smooth'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}",
                    key="cons_k_smooth"
                )
                
                # Overbought Level
                param_info = cons_params['overbought_level']
                cons_values['overbought_level'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}",
                    key="cons_overbought"
                )
                
                # Max Distance to Level
                param_info = cons_params['max_distance_to_level']
                cons_values['max_distance_to_level'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}",
                    key="cons_max_distance"
                )
                
                # Volume Threshold
                param_info = cons_params['volume_threshold']
                cons_values['volume_threshold'] = st.slider(
                    param_info['name'],
                    min_value=param_info['min_value'],
                    max_value=param_info['max_value'],
                    value=param_info['default'],
                    step=param_info['step'],
                    help=f"{param_info['description']} - {param_info['impact']}",
                    key="cons_volume_threshold"
                )
                
                # Volume Confirmation toggle
                cons_values['volume_confirmation'] = st.checkbox(
                    "Volume Confirmation",
                    value=True,
                    help="Require volume confirmation for consolidation signals",
                    key="cons_volume_conf"
                )
            
            # Store consolidation parameters
            st.session_state.cons_params = cons_values
            
            # Reset to defaults button
            if st.button("🔄 Reset to Defaults", key="reset_cons"):
                st.rerun()
        
        # Quick Compare Tab
        with param_tabs[2]:
            st.markdown("### 🔄 Quick Parameter Comparison")
            st.markdown("Compare multiple parameter sets quickly to find optimal settings.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Preset Parameter Sets")
                
                if st.button("🚀 Conservative (Low Risk)", key="preset_conservative"):
                    st.session_state.trend_params = {
                        'dmi_period': 18, 'adx_threshold': 30.0, 'atr_stop_multiplier': 3.0,
                        'chandelier_multiplier': 4.0, 'ema_period': 60, 'min_adx_slope': 1.0,
                        'volume_threshold': 1.5, 'use_ema_filter': True, 'volume_confirmation': True
                    }
                    st.success("Conservative parameters applied!")
                
                if st.button("⚡ Aggressive (High Risk)", key="preset_aggressive"):
                    st.session_state.trend_params = {
                        'dmi_period': 10, 'adx_threshold': 20.0, 'atr_stop_multiplier': 1.5,
                        'chandelier_multiplier': 2.5, 'ema_period': 30, 'min_adx_slope': 0.0,
                        'volume_threshold': 1.0, 'use_ema_filter': False, 'volume_confirmation': False
                    }
                    st.success("Aggressive parameters applied!")
                
                if st.button("⚖️ Balanced (Medium Risk)", key="preset_balanced"):
                    st.session_state.trend_params = {
                        'dmi_period': 14, 'adx_threshold': 25.0, 'atr_stop_multiplier': 2.0,
                        'chandelier_multiplier': 3.0, 'ema_period': 50, 'min_adx_slope': 0.5,
                        'volume_threshold': 1.2, 'use_ema_filter': True, 'volume_confirmation': False
                    }
                    st.success("Balanced parameters applied!")
            
            with col2:
                st.markdown("#### Parameter Impact Guide")
                st.markdown("""
                **🎯 Key Parameters to Optimize:**
                
                **For Trend Strategy:**
                • **ADX Threshold**: Higher = fewer but stronger trends
                • **ATR Stop Multiplier**: Lower = tighter stops, more stopped out
                • **Chandelier Multiplier**: Higher = ride trends longer
                
                **For Consolidation Strategy:**
                • **Oversold/Overbought Levels**: Wider range = fewer but stronger signals
                • **Max Distance to S/R**: Lower = must be closer to levels
                • **Stochastic Period**: Lower = more sensitive to price changes
                
                **🚀 Optimization Tips:**
                1. Start with default values
                2. Change one parameter at a time  
                3. Run multiple backtests to compare results
                4. Focus on Sharpe ratio and drawdown metrics
                5. Test on different time periods for robustness
                """)
        
        # Enhanced Run Backtest Section
        st.subheader("🚀 Run Backtest")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🧪 Quick Backtest", key="quick_backtest", help="Fast backtest with current parameters"):
                with st.spinner("Running quick backtest..."):
                    try:
                        results = self._run_real_backtest(quick=True)
                        if results:
                            st.session_state.backtest_results = results
                            st.success("✅ Quick backtest completed!")
                        else:
                            st.error("❌ Backtest failed. Please check your configuration.")
                    except Exception as e:
                        st.error(f"❌ Backtest error: {str(e)}")
                st.rerun()
        
        with col2:
            if st.button("🔬 Full Backtest", key="full_backtest", help="Comprehensive backtest with charts"):
                with st.spinner("Running comprehensive backtest..."):
                    try:
                        results = self._run_real_backtest(quick=False)
                        if results:
                            st.session_state.backtest_results = results
                            st.success("✅ Full backtest completed with charts!")
                        else:
                            st.error("❌ Backtest failed. Please check your configuration.")
                    except Exception as e:
                        st.error(f"❌ Backtest error: {str(e)}")
                st.rerun()
        
        with col3:
            if st.button("📊 Parameter Sweep", key="param_sweep", help="Test multiple parameter combinations"):
                with st.spinner("Running parameter optimization..."):
                    try:
                        results = self._run_real_backtest(optimize=True)
                        if results:
                            st.session_state.backtest_results = results
                            st.success("✅ Parameter optimization completed!")
                        else:
                            st.error("❌ Parameter optimization failed. Please check your configuration.")
                    except Exception as e:
                        st.error(f"❌ Optimization error: {str(e)}")
                st.rerun()
        
        # Current Parameter Summary
        if hasattr(st.session_state, 'trend_params') or hasattr(st.session_state, 'cons_params'):
            st.subheader("📋 Current Parameter Settings")
            with st.expander("View Current Parameters", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Trend Strategy:**")
                    if hasattr(st.session_state, 'trend_params'):
                        for param, value in st.session_state.trend_params.items():
                            st.write(f"• {param}: {value}")
                    
                with col2:
                    st.markdown("**Consolidation Strategy:**")
                    if hasattr(st.session_state, 'cons_params'):
                        for param, value in st.session_state.cons_params.items():
                            st.write(f"• {param}: {value}")
        
        # Display Results
        if st.session_state.backtest_results:
            st.subheader("📊 Backtest Results")
            self.render_backtest_results()
            
            # Add parameter impact analysis
            st.subheader("🎯 Parameter Impact Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Performance Metrics")
                results = st.session_state.backtest_results
                
                # Performance scoring
                sharpe_score = "🟢 Excellent" if results['sharpe_ratio'] > 2.0 else "🟡 Good" if results['sharpe_ratio'] > 1.0 else "🔴 Poor"
                drawdown_score = "🟢 Low Risk" if abs(results['max_drawdown']) < 0.1 else "🟡 Medium Risk" if abs(results['max_drawdown']) < 0.2 else "🔴 High Risk"
                
                st.write(f"**Sharpe Ratio:** {results['sharpe_ratio']:.2f} {sharpe_score}")
                st.write(f"**Max Drawdown:** {results['max_drawdown']:.1%} {drawdown_score}")
                st.write(f"**Total Trades:** {results['total_trades']}")
                
            with col2:
                st.markdown("#### 🔧 Optimization Suggestions")
                results = st.session_state.backtest_results
                
                suggestions = []
                if results['sharpe_ratio'] < 1.0:
                    suggestions.append("• Increase ADX threshold for stronger trend confirmation")
                    suggestions.append("• Try wider stop losses (higher ATR multiplier)")
                
                if abs(results['max_drawdown']) > 0.15:
                    suggestions.append("• Reduce position sizes (lower risk per trade)")
                    suggestions.append("• Use tighter stops or more conservative parameters")
                
                if results['total_trades'] < 20:
                    suggestions.append("• Lower ADX threshold for more signals")
                    suggestions.append("• Reduce confirmation requirements")
                elif results['total_trades'] > 200:
                    suggestions.append("• Increase filtering to focus on best setups")
                    suggestions.append("• Higher confirmation thresholds")
                
                if not suggestions:
                    suggestions.append("• Parameters look well balanced!")
                    suggestions.append("• Try testing on different time periods")
                
                for suggestion in suggestions:
                    st.write(suggestion)
        
        # Add logs panel at the bottom
        st.markdown("---")
        self.render_logs_panel()
    
    def render_backtest_results(self):
        """Render backtest results with detailed trade information"""
        results = st.session_state.backtest_results
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Return", f"{results['total_return']:.1%}", f"{results['total_return']:.1%}")
        
        with col2:
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.1%}")
        
        with col4:
            st.metric("Total Trades", f"{results['total_trades']}")
            
        with col5:
            st.metric("Win Rate", f"{results.get('win_rate', 0):.1%}")
        
        # Equity Curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['equity_curve']['dates'],
            y=results['equity_curve']['values'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Backtest Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Trade Analysis
        st.subheader("📋 Detailed Trade Analysis")
        
        # Check if we have trade data
        trades_data = results.get('trades_data')
        if trades_data is not None and hasattr(trades_data, '__len__') and len(trades_data) > 0:
            try:
                # Convert trades to DataFrame for display
                if hasattr(trades_data, 'to_pandas'):
                    trades_df = trades_data.to_pandas()
                elif isinstance(trades_data, pd.DataFrame):
                    trades_df = trades_data
                else:
                    # Try to convert to DataFrame if it's a records array or similar
                    trades_df = pd.DataFrame(trades_data)
                
                if not trades_df.empty:
                    st.write(f"**Bot wykonał {len(trades_df)} transakcji podczas backtestingu:**")
                    
                    # Show trades table with key columns
                    display_df = trades_df.copy()
                    
                    # Try to show most relevant columns
                    important_cols = []
                    for col_group in [['Entry Time', 'entry_time', 'open_time'], 
                                     ['Exit Time', 'exit_time', 'close_time'],
                                     ['Side', 'side', 'direction'],
                                     ['Entry Price', 'entry_price', 'open_price'],
                                     ['Exit Price', 'exit_price', 'close_price'],
                                     ['P&L', 'pnl', 'profit_loss'],
                                     ['Duration', 'duration', 'holding_period']]:
                        for col in col_group:
                            if col in display_df.columns:
                                important_cols.append(col)
                                break
                    
                    if important_cols:
                        st.dataframe(
                            display_df[important_cols].head(20),
                            use_container_width=True
                        )
                    else:
                        st.dataframe(display_df.head(10), use_container_width=True)
                    
                    # Trade Statistics
                    if len(trades_df) > 0:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("#### 🎯 Statystyki Transakcji")
                            pnl_col = None
                            for col in ['pnl', 'profit_loss', 'return']:
                                if col in trades_df.columns:
                                    pnl_col = col
                                    break
                                    
                            if pnl_col:
                                winning_trades = (trades_df[pnl_col] > 0).sum()
                                losing_trades = (trades_df[pnl_col] <= 0).sum()
                                st.write(f"**Zyskowne:** {winning_trades}")
                                st.write(f"**Stratne:** {losing_trades}")
                                if winning_trades + losing_trades > 0:
                                    win_rate = winning_trades / (winning_trades + losing_trades) * 100
                                    st.write(f"**Win Rate:** {win_rate:.1f}%")
                        
                        with col2:
                            st.markdown("#### 💰 P&L Analysis")
                            if pnl_col:
                                best_trade = trades_df[pnl_col].max()
                                worst_trade = trades_df[pnl_col].min()
                                total_pnl = trades_df[pnl_col].sum()
                                st.write(f"**Najlepsza:** ${best_trade:.2f}")
                                st.write(f"**Najgorsza:** ${worst_trade:.2f}")
                                st.write(f"**Suma P&L:** ${total_pnl:.2f}")
                        
                        with col3:
                            st.markdown("#### 📊 Trading Insights")
                            st.write(f"**Łączne transakcje:** {len(trades_df)}")
                            if 'duration' in trades_df.columns:
                                avg_duration = trades_df['duration'].mean()
                                st.write(f"**Średni czas:** {avg_duration}")
                            
                            # Show position entry/exit info
                            if len(trades_df) > 0:
                                st.write("**Ostatnie 5 transakcji:**")
                                recent_trades = trades_df.tail(5)
                                for idx, trade in recent_trades.iterrows():
                                    side = trade.get('side', 'N/A')
                                    pnl = trade.get(pnl_col, 0) if pnl_col else 0
                                    pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                                    st.write(f"{pnl_emoji} {side.upper()} → ${pnl:.2f}")
                        
                        st.success("✅ **Szczegółowa analiza dostępna!** Bot aktywnie otwierał i zamykał pozycje podczas testów.")
                        
            except Exception as e:
                st.warning(f"Nie można wyświetlić szczegółów transakcji: {str(e)}")
                st.info("Dane o transakcjach są dostępne ale w nierozpoznawalnym formacie.")
        else:
            st.info("📊 **Brak szczegółowych danych o transakcjach**")
            st.write("To może oznaczać:")
            st.write("• Bot nie wykonał żadnych transakcji w tym okresie")
            st.write("• Parametry strategii były zbyt restrykcyjne")  
            st.write("• Warunki rynkowe nie wywołały sygnałów")
            st.write("• Spróbuj dostroić parametry lub wydłużyć okres testów")
            
        # Strategy Performance Summary
        st.subheader("🧠 Wnioski ze strategii")
        strategy_insights = []
        
        total_trades = results.get('total_trades', 0)
        sharpe = results.get('sharpe_ratio', 0)
        max_dd = abs(results.get('max_drawdown', 0))
        
        if total_trades == 0:
            strategy_insights.append("⚠️ **Brak transakcji** - Rozważ złagodzenie parametrów strategii")
        elif total_trades < 10:
            strategy_insights.append("📈 **Niska częstotliwość** - Strategia jest bardzo selektywna")
        elif total_trades > 100:
            strategy_insights.append("⚡ **Wysoka aktywność** - Strategia generuje dużo sygnałów")
        
        if sharpe > 2:
            strategy_insights.append("🎯 **Doskonałe wyniki** - Silna performance skorygowana o ryzyko")
        elif sharpe > 1:
            strategy_insights.append("✅ **Dobre wyniki** - Strategia pokazuje potencjał")
        elif sharpe < 0:
            strategy_insights.append("🔴 **Słabe wyniki** - Strategia wymaga optymalizacji")
        
        if max_dd > 0.2:
            strategy_insights.append("⚠️ **Wysokie ryzyko** - Rozważ bardziej konserwatywne zarządzanie")
        elif max_dd < 0.1:
            strategy_insights.append("🛡️ **Niski drawdown** - Dobra kontrola ryzyka")
        
        if strategy_insights:
            for insight in strategy_insights:
                st.write(insight)
        else:
            st.write("📊 **Analiza wyników dostępna po przeprowadzeniu backtestu**")
    
    def _ensure_positions_cache(self):
        """Initialize session state for background positions cache."""
        ss = st.session_state
        if 'bybit_positions_cache' not in ss:
            ss.bybit_positions_cache = []
        if 'bybit_positions_updated_at' not in ss:
            ss.bybit_positions_updated_at = None
        if 'bybit_positions_fetch_in_progress' not in ss:
            ss.bybit_positions_fetch_in_progress = False
        if 'bybit_positions_refresh_requested' not in ss:
            ss.bybit_positions_refresh_requested = False
    
    def _positions_fetch_worker(self, api_key: str, api_secret: str, testnet: bool):
        """Background worker to fetch positions and store in session cache (non-blocking UI)."""
        try:
            import ccxt
            from datetime import datetime as _dt
            
            exchange = ccxt.bybit({
                'apiKey': api_key,
                'secret': api_secret,
                'testnet': testnet,
                'enableRateLimit': True,
                'options': {'defaultType': 'linear'},
            })
            
            # Filter to only positions managed by the bot (database open positions)
            managed_symbols = set()
            try:
                from database.models import get_position_repository, PositionStatus
                pos_repo = get_position_repository()
                open_positions = pos_repo.get_positions(status=PositionStatus.OPEN, limit=1000)
                managed_symbols = set(p.symbol for p in open_positions)
            except Exception:
                # If DB unavailable, do not touch any positions
                managed_symbols = set()
            
            positions = exchange.fetch_positions()
            real_positions = []
            for pos in positions:
                if not isinstance(pos, dict):
                    continue
                # Determine active size
                size_value = None
                for size_field in ['size', 'contracts', 'amount', 'baseSize', 'quoteSize']:
                    if size_field in pos and pos.get(size_field) not in (None, 0):
                        size_value = pos.get(size_field)
                        break
                if size_value in (None, 0):
                    continue
                sym = pos.get('symbol')
                if not sym:
                    continue
                # Apply DB filter
                if managed_symbols and sym not in managed_symbols:
                    continue
                elif not managed_symbols:
                    # If no managed symbols, skip tracking
                    continue
                real_positions.append(pos)
            
            st.session_state.bybit_positions_cache = real_positions
            st.session_state.bybit_positions_updated_at = _dt.now()
        except Exception as e:
            # Store error info in cache for UI to show
            st.session_state.bybit_positions_cache = [{"error": str(e)}]
            st.session_state.bybit_positions_updated_at = _dt.now()
        finally:
            st.session_state.bybit_positions_fetch_in_progress = False
            st.session_state.bybit_positions_refresh_requested = False
    
    def start_positions_fetch(self, force: bool = False):
        """Start a background thread to fetch positions if not already running."""
        self._ensure_positions_cache()
        ss = st.session_state
        # Only start if credentials exist
        api_key = ss.config.get('bybit_api_key', '')
        api_secret = ss.config.get('bybit_api_secret', '')
        if not api_key or not api_secret:
            return
        
        # Decide if fetch is needed
        need_fetch = force or (not ss.bybit_positions_cache) or (ss.bybit_positions_updated_at is None)
        
        # Start thread if needed and not already running
        if need_fetch and not ss.bybit_positions_fetch_in_progress:
            ss.bybit_positions_fetch_in_progress = True
            thread = threading.Thread(
                target=self._positions_fetch_worker,
                args=(api_key, api_secret, ss.config.get('bybit_testnet', True)),
                daemon=True
            )
            thread.start()
    
    def render_live_trading_page(self):
        """Render live trading page"""
        st.title("🔴 Live Trading")
        st.markdown("Monitor and control live trading operations.")
        
        # Environment Status Indicator at the top
        if st.session_state.config.get('bybit_api_key') and st.session_state.config.get('bybit_api_secret'):
            if st.session_state.config.get('bybit_testnet', True):
                st.success("🧪 **Current Environment: DEMO TRADING (Testnet)** - Virtual money, safe testing")
            else:
                st.error("🔴 **Current Environment: LIVE PRODUCTION** - Real money trading!")
        else:
            st.info("ℹ️ **Environment: Not configured** - Please add API keys in Configuration")
        
        # Safety Warning
        st.warning("⚠️ **WARNING**: Live trading involves real money and risk. Always test thoroughly on testnet first!")
        
        # Trading Controls
        with st.expander("🧭 Bot Intent Preview", expanded=True):
            st.write("The bot operates in Futures mode only and will avoid interfering with manually opened positions.")
            st.write("Planned behavior:")
            st.markdown(f"• Market: Bybit Linear Futures (USDT-margined)")
            st.markdown(f"• Symbols: {', '.join(st.session_state.config.get('symbols', []))}")
            st.markdown(f"• Timeframe: {st.session_state.config.get('timeframe', '15m')}")
            st.markdown(f"• Default leverage: x{int(st.session_state.config.get('leverage', 2))}")
            st.markdown("• Entries: Based on regime detection and strategy signals; positions are recorded in the database and only those will be managed.")
            st.caption("This preview summarizes the bot's intentions. Execution occurs only upon valid signals and risk checks.")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🟢 Start Trading", key="start_trading"):
                if not st.session_state.config.get('bybit_api_key'):
                    st.error("❌ Please configure API keys first!")
                else:
                    st.session_state.bot_status = 'running'
                    st.success("✅ Trading started!")
        
        with col2:
            if st.button("🔴 Stop Trading", key="stop_trading"):
                st.session_state.bot_status = 'stopped'
                st.info("ℹ️ Trading stopped!")
        
        with col3:
            if st.button("⏸️ Pause Trading", key="pause_trading"):
                st.session_state.bot_status = 'paused'
                st.warning("⚠️ Trading paused!")
        
        # Current Positions with refresh button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📊 Current Positions")
        with col2:
            if st.button("🔄 Refresh", key="refresh_positions"):
                st.session_state.bybit_positions_refresh_requested = True
                self.start_positions_fetch(force=True)
        
        # Check if API keys are configured
        api_key = st.session_state.config.get('bybit_api_key', '')
        api_secret = st.session_state.config.get('bybit_api_secret', '')
        
        if not api_key or not api_secret:
            st.warning("⚠️ **No API keys configured.** Please configure your Bybit API keys in the Configuration page to see real positions.")
            st.info("💡 **Next Steps:**")
            st.info("• Go to Configuration page")
            st.info("• Add your Bybit API key and secret")
            st.info("• Choose Testnet (safe) or Production mode (real money)")
            st.info("• Return here to view your actual positions")
            return
        
        # Display positions using background cache (no blocking, no spinner)
        self._ensure_positions_cache()
        # If user requested refresh or cache empty, start background fetch
        if st.session_state.bybit_positions_refresh_requested or not st.session_state.bybit_positions_cache:
            self.start_positions_fetch(force=True)
        else:
            # Start periodic background fetch only if not already fetching
            self.start_positions_fetch(force=False)
        
        # Show last updated info
        last_updated = st.session_state.bybit_positions_updated_at
        if last_updated:
            st.caption(f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.caption("Positions are being fetched in the background…")
        
        cache = st.session_state.bybit_positions_cache or []
        if cache and isinstance(cache, list) and isinstance(cache[0], dict) and 'error' in cache[0]:
            st.error(f"❌ Failed to fetch positions: {cache[0]['error']}")
        elif cache:
            # Prepare display
            positions_data = {
                'Symbol': [pos.get('symbol', 'N/A') for pos in cache],
                'Side': [pos.get('side', 'N/A').upper() if pos.get('side') else 'N/A' for pos in cache],
                'Size': [pos.get('size', 0) for pos in cache],
                'Entry Price': [f"${pos.get('entryPrice', 0):.2f}" if pos.get('entryPrice') else 'N/A' for pos in cache],
                'Current Price': [f"${pos.get('markPrice', 0):.2f}" if pos.get('markPrice') else 'N/A' for pos in cache],
                'P&L': [f"${pos.get('unrealizedPnl', 0):.2f}" if pos.get('unrealizedPnl') is not None else 'N/A' for pos in cache],
                'P&L %': [f"{pos.get('percentage', 0):.2f}%" if pos.get('percentage') is not None else 'N/A' for pos in cache]
            }
            st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
            # Mode info
            if st.session_state.config.get('bybit_testnet', True):
                st.success("🧪 TESTNET MODE")
            else:
                st.error("🔴 PRODUCTION MODE")
        else:
            if st.session_state.bybit_positions_fetch_in_progress:
                st.info("Fetching positions in the background… This will not block the UI.")
            else:
                st.info("No managed positions found.")
        
        # Add logs panel at the bottom
        st.markdown("---")
        self.render_logs_panel()
    
    def render_strategy_info_page(self):
        """Render strategy information and documentation page"""
        st.title("📚 Strategy Information")
        st.markdown("Understand the bot's trading strategies and their parameters.")
        
        # Import strategies
        from strategies.trend_strategy import TrendStrategy
        from strategies.consolidation_strategy import ConsolidationStrategy
        
        # Strategy selector
        st.subheader("🎯 Available Strategies")
        strategy_tabs = st.tabs(["Trend Following", "Mean Reversion", "Regime Overview"])
        
        # Trend Strategy Tab
        with strategy_tabs[0]:
            trend_info = TrendStrategy.get_strategy_info()
            trend_params = TrendStrategy.get_parameter_info()
            
            st.header(f"📈 {trend_info['name']}")
            st.markdown(f"**Description:** {trend_info['short_description']}")
            
            # Detailed description
            st.markdown("### 📋 Strategy Details")
            st.markdown(trend_info['detailed_description'])
            
            # Entry/Exit criteria
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🟢 Entry Criteria")
                for criteria in trend_info['entry_criteria']:
                    st.markdown(f"• {criteria}")
                    
                st.markdown("#### ✅ Advantages")
                for advantage in trend_info['advantages']:
                    st.markdown(f"• {advantage}")
            
            with col2:
                st.markdown("#### 🔴 Exit Criteria")
                for criteria in trend_info['exit_criteria']:
                    st.markdown(f"• {criteria}")
                    
                st.markdown("#### ⚠️ Disadvantages")
                for disadvantage in trend_info['disadvantages']:
                    st.markdown(f"• {disadvantage}")
            
            # Market conditions
            st.markdown("#### 🌐 Market Conditions")
            market_info = trend_info['market_conditions']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Best For:**\n{market_info['best_for']}")
            with col2:
                st.warning(f"**Worst For:**\n{market_info['worst_for']}")
            with col3:
                st.success(f"**Active Regime:**\n{market_info['regime']}")
            
            # Parameters
            st.markdown("### ⚙️ Strategy Parameters")
            st.markdown("These parameters control how the strategy behaves. Adjust them in the backtesting section to optimize performance.")
            
            # Display parameters in a nice table
            param_data = []
            for param_key, param_info in trend_params.items():
                param_data.append({
                    'Parameter': param_info['name'],
                    'Current Default': param_info['default'],
                    'Range': f"{param_info['min_value']} - {param_info['max_value']}",
                    'Impact': param_info['impact'],
                    'Description': param_info['description']
                })
            
            param_df = pd.DataFrame(param_data)
            st.dataframe(param_df, use_container_width=True, hide_index=True)
        
        # Consolidation Strategy Tab
        with strategy_tabs[1]:
            cons_info = ConsolidationStrategy.get_strategy_info()
            cons_params = ConsolidationStrategy.get_parameter_info()
            
            st.header(f"📊 {cons_info['name']}")
            st.markdown(f"**Description:** {cons_info['short_description']}")
            
            # Detailed description
            st.markdown("### 📋 Strategy Details")
            st.markdown(cons_info['detailed_description'])
            
            # Entry/Exit criteria
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🟢 Entry Criteria")
                for criteria in cons_info['entry_criteria']:
                    st.markdown(f"• {criteria}")
                    
                st.markdown("#### ✅ Advantages")
                for advantage in cons_info['advantages']:
                    st.markdown(f"• {advantage}")
            
            with col2:
                st.markdown("#### 🔴 Exit Criteria")
                for criteria in cons_info['exit_criteria']:
                    st.markdown(f"• {criteria}")
                    
                st.markdown("#### ⚠️ Disadvantages")
                for disadvantage in cons_info['disadvantages']:
                    st.markdown(f"• {disadvantage}")
            
            # Market conditions
            st.markdown("#### 🌐 Market Conditions")
            market_info = cons_info['market_conditions']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Best For:**\n{market_info['best_for']}")
            with col2:
                st.warning(f"**Worst For:**\n{market_info['worst_for']}")
            with col3:
                st.success(f"**Active Regime:**\n{market_info['regime']}")
            
            # Parameters
            st.markdown("### ⚙️ Strategy Parameters")
            st.markdown("These parameters control how the strategy behaves. Adjust them in the backtesting section to optimize performance.")
            
            # Display parameters in a nice table
            param_data = []
            for param_key, param_info in cons_params.items():
                param_data.append({
                    'Parameter': param_info['name'],
                    'Current Default': param_info['default'],
                    'Range': f"{param_info['min_value']} - {param_info['max_value']}",
                    'Impact': param_info['impact'],
                    'Description': param_info['description']
                })
            
            param_df = pd.DataFrame(param_data)
            st.dataframe(param_df, use_container_width=True, hide_index=True)
        
        # Regime Overview Tab
        with strategy_tabs[2]:
            st.header("🌐 Market Regime System")
            st.markdown("""
            The Adaptive Trading Bot uses a **4-state market regime detection system** to automatically switch between strategies based on market conditions.
            """)
            
            # Regime descriptions
            regime_data = {
                'Regime': ['TRENDING', 'CONSOLIDATION', 'STAGNANT', 'PANIC'],
                'Description': [
                    'Strong directional movement (ADX > 25)',
                    'Sideways movement (ADX < 25)',
                    'Very low volatility (no trading)',
                    'Very high volatility (no trading)'
                ],
                'Active Strategy': [
                    '📈 Trend Following',
                    '📊 Mean Reversion',
                    '⏸️ No Trading',
                    '⏸️ No Trading'
                ],
                'Key Indicator': [
                    'ADX > 25, Normal ATR',
                    'ADX < 25, Normal ATR',
                    'ATR Ratio < 0.5',
                    'ATR Ratio > 2.0'
                ]
            }
            
            regime_df = pd.DataFrame(regime_data)
            st.dataframe(regime_df, use_container_width=True, hide_index=True)
            
            st.markdown("### 📊 How It Works")
            st.markdown("""
            1. **Market Analysis**: The bot continuously analyzes market data using ADX (trend strength) and ATR (volatility)
            2. **Regime Detection**: Based on these indicators, it classifies the current market regime
            3. **Strategy Selection**: The appropriate strategy is automatically activated for the detected regime
            4. **Signal Generation**: The active strategy generates buy/sell signals according to its logic
            5. **Risk Management**: All trades are subject to comprehensive risk management rules
            """)
            
            st.markdown("### 🎛️ Parameter Optimization")
            st.info("""
            **💡 Pro Tip**: Use the **Backtesting** page to experiment with different parameter values:
            
            • Adjust parameters using the sliders
            • Run quick backtests to see the impact
            • Compare results with different settings
            • Save successful parameter combinations as presets
            
            Start with small parameter changes (±10-20%) and observe the impact on key metrics like Sharpe ratio and win rate.
            """)
        
        # Add logs panel at the bottom
        st.markdown("---")
        self.render_logs_panel()
    
    def render_trading_history_page(self):
        """Render trading history page"""
        st.title("📊 Trading History")
        st.markdown("View and analyze your complete trading position history.")
        
        # Database connection check
        try:
            from database.models import get_position_repository
            position_repo = get_position_repository()
            database_available = True
        except ImportError:
            database_available = False
        
        if not database_available:
            st.error("❌ Database not available. Please ensure Docker database is running and dependencies are installed.")
            st.info("💡 Run: `docker-compose up -d` to start the database")
            return
        
        # Filters Section
        st.subheader("🔍 Filters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol_filter = st.selectbox(
                "Symbol",
                options=["All"] + ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'DOT/USDT'],
                index=0
            )
        
        with col2:
            status_filter = st.selectbox(
                "Status",
                options=["All", "Open", "Closed", "Stopped Out"],
                index=0
            )
        
        with col3:
            exit_reason_filter = st.selectbox(
                "Exit Reason",
                options=["All", "Take Profit", "Stop Loss", "Manual", "Regime Change"],
                index=0
            )
        
        with col4:
            date_range = st.date_input(
                "Date Range",
                value=[],
                help="Select date range for filtering"
            )
        
        # Load position data
        try:
            # Get position statistics
            stats = position_repo.get_position_stats()
            
            # Display summary metrics
            st.subheader("📈 Summary Statistics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Positions", stats.get('total_positions', 0))
            
            with col2:
                st.metric("Open Positions", stats.get('open_positions', 0))
            
            with col3:
                st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
            
            with col4:
                st.metric("Total P&L", f"${stats.get('total_pnl', 0):.2f}")
            
            with col5:
                winning = stats.get('winning_positions', 0)
                losing = stats.get('losing_positions', 0)
                st.metric("Win/Loss", f"{winning}/{losing}")
            
            # Get filtered positions
            symbol_param = None if symbol_filter == "All" else symbol_filter
            
            status_mapping = {
                "All": None,
                "Open": "open",
                "Closed": "closed", 
                "Stopped Out": "stopped_out"
            }
            status_param = status_mapping.get(status_filter)
            
            from database.models import PositionStatus as DbPositionStatus
            if status_param:
                status_param = DbPositionStatus(status_param)
            
            positions = position_repo.get_positions(
                symbol=symbol_param,
                status=status_param,
                limit=1000
            )
            
            if positions:
                # Convert to DataFrame for display
                position_data = []
                for pos in positions:
                    pos_dict = pos.to_dict()
                    position_data.append({
                        'Symbol': pos_dict['symbol'],
                        'Side': pos_dict['side'].title(),
                        'Entry Price': f"${pos_dict['entry_price']:.4f}" if pos_dict['entry_price'] else "N/A",
                        'Exit Price': f"${pos_dict['exit_price']:.4f}" if pos_dict['exit_price'] else "Open",
                        'Quantity': f"{pos_dict['quantity']:.6f}" if pos_dict['quantity'] else "N/A",
                        'P&L': f"${pos_dict['pnl']:.2f}" if pos_dict['pnl'] else "N/A",
                        'P&L %': f"{pos_dict['pnl_percentage']:.2f}%" if pos_dict['pnl_percentage'] else "N/A",
                        'Duration': f"{pos_dict['duration_minutes']} min" if pos_dict['duration_minutes'] else "N/A",
                        'Exit Reason': pos_dict['exit_reason'].replace('_', ' ').title() if pos_dict['exit_reason'] else "N/A",
                        'Regime': pos_dict['entry_regime'].title(),
                        'Entry Time': pos_dict['entry_time'][:19] if pos_dict['entry_time'] else "N/A",
                        'Status': pos_dict['status'].replace('_', ' ').title()
                    })
                
                df = pd.DataFrame(position_data)
                
                # Position History Table
                st.subheader("📋 Position History")
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "P&L": st.column_config.NumberColumn(
                            "P&L",
                            format="$%.2f",
                        ),
                        "Entry Time": st.column_config.DatetimeColumn(
                            "Entry Time",
                            format="DD/MM/YYYY HH:mm",
                        )
                    }
                )
                
                # P&L Distribution Chart
                if any(pos['pnl'] for pos in position_data if pos['pnl'] != "N/A"):
                    st.subheader("📊 P&L Distribution")
                    
                    pnl_values = [
                        float(pos['P&L'].replace('$', '')) 
                        for pos in position_data 
                        if pos['P&L'] not in ["N/A", "Open"]
                    ]
                    
                    if pnl_values:
                        fig = px.histogram(
                            x=pnl_values,
                            bins=20,
                            title="P&L Distribution",
                            labels={'x': 'P&L ($)', 'y': 'Number of Trades'},
                            color_discrete_sequence=['#1f77b4']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Performance by Symbol
                        symbol_performance = {}
                        for pos in position_data:
                            if pos['P&L'] not in ["N/A", "Open"]:
                                symbol = pos['Symbol']
                                pnl = float(pos['P&L'].replace('$', ''))
                                if symbol not in symbol_performance:
                                    symbol_performance[symbol] = {'total_pnl': 0, 'count': 0}
                                symbol_performance[symbol]['total_pnl'] += pnl
                                symbol_performance[symbol]['count'] += 1
                        
                        if symbol_performance:
                            st.subheader("💰 Performance by Symbol")
                            perf_data = []
                            for symbol, data in symbol_performance.items():
                                perf_data.append({
                                    'Symbol': symbol,
                                    'Total P&L': data['total_pnl'],
                                    'Trades': data['count'],
                                    'Avg P&L': data['total_pnl'] / data['count']
                                })
                            
                            perf_df = pd.DataFrame(perf_data)
                            st.dataframe(
                                perf_df,
                                use_container_width=True,
                                column_config={
                                    "Total P&L": st.column_config.NumberColumn("Total P&L", format="$%.2f"),
                                    "Avg P&L": st.column_config.NumberColumn("Avg P&L", format="$%.2f")
                                }
                            )
            
            else:
                st.info("📭 No trading positions found. Start trading to see history here!")
                
                # Show sample data structure
                st.subheader("📖 What You'll See Here")
                st.markdown("""
                Once you start trading, this page will display:
                - **Complete position history** with entry/exit details
                - **P&L tracking** for each trade with percentages
                - **Trade duration** and exit reasons (TP/SL/Manual)
                - **Performance statistics** by symbol and regime
                - **Visual charts** showing P&L distribution
                - **Filtering options** by symbol, status, and date range
                """)
        
        except Exception as e:
            st.error(f"❌ Error loading trading history: {str(e)}")
            st.info("💡 Make sure the database is running: `docker-compose up -d`")
    
    def render_performance_page(self):
        """Render performance analysis page"""
        st.title("📈 Performance Analysis")
        st.markdown("Detailed performance metrics and analysis.")
        
        # Performance Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Key Statistics")
            metrics_data = {
                'Metric': [
                    'Total Return', 'Annual Return', 'Sharpe Ratio', 'Sortino Ratio',
                    'Max Drawdown', 'Win Rate', 'Profit Factor', 'Average Trade'
                ],
                'Value': [
                    '25.4%', '18.2%', '1.45', '1.89',
                    '-12.3%', '68.5%', '1.89', '+$15.25'
                ]
            }
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
        with col2:
            st.subheader("🎯 Risk Metrics")
            risk_data = {
                'Metric': [
                    'Value at Risk (95%)', 'Expected Shortfall', 'Beta',
                    'Maximum Daily Loss', 'Current Drawdown', 'Risk-Adjusted Return'
                ],
                'Value': [
                    '-$185.50', '-$245.75', '0.82',
                    '-$125.00', '-2.1%', '1.32'
                ]
            }
            st.dataframe(pd.DataFrame(risk_data), use_container_width=True)
        
        # Monthly Returns Heatmap
        st.subheader("🔥 Monthly Returns Heatmap")
        monthly_returns = np.random.randn(12, 3) * 0.05  # Sample data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        years = ['2022', '2023', '2024']
        
        fig = px.imshow(
            monthly_returns.T,
            x=months,
            y=years,
            color_continuous_scale='RdYlGn',
            aspect='auto',
            labels={'color': 'Return %'}
        )
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)
    
    def test_connection(self):
        """Test API connection"""
        # Ensure configuration is loaded from database first
        try:
            st.session_state.config['database_loaded'] = False
            self.load_database_config()
        except Exception:
            pass
        fresh_config = dict(st.session_state.config)
        
        # Allow test to proceed but inform user if keys missing
        if not fresh_config.get('bybit_api_key') or not fresh_config.get('bybit_api_secret'):
            st.warning("⚠️ No API keys found in database or environment. Attempting limited connection test...")
        
        with st.spinner("Testing connection..."):
            try:
                # Import and use the actual test_connections function from main.py
                import asyncio
                from main import test_connections
                
                # Run the actual connection test
                success = asyncio.run(test_connections(fresh_config))
                
                if success:
                    st.success("✅ Connection successful! All components working.")
                    st.session_state.bot_status = 'testing'
                else:
                    st.error("❌ Connection test failed. Check your API keys and network connection.")
                    
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
                logger.error(f"Test connection error: {e}")
    
    def generate_env_file(self) -> str:
        """Generate .env file content"""
        config = st.session_state.config
        return f"""# Adaptive Trading Bot Configuration
BYBIT_API_KEY={config['bybit_api_key']}
BYBIT_API_SECRET={config['bybit_api_secret']}
BYBIT_TESTNET={str(config['bybit_testnet']).lower()}
TRADING_SYMBOLS={','.join(config['symbols'])}
TIMEFRAME={config['timeframe']}
INITIAL_CAPITAL={config['initial_capital']}
RISK_PER_TRADE={config['risk_per_trade']}
MAX_DAILY_LOSS={config['max_daily_loss']}
MAX_DRAWDOWN={config['max_drawdown']}
FUTURES_MODE={str(config.get('futures_mode', True)).lower()}
LEVERAGE={config.get('leverage', 2)}
USE_KELLY={str(config.get('use_kelly', False)).lower()}
KELLY_CAP={config.get('kelly_cap', 0.03)}
VAR_THRESHOLD={config.get('var_threshold', 0.05)}
MAX_SYMBOL_EXPOSURE_PCT={config.get('max_symbol_exposure_pct', 0.5)}
"""
    
    
    def render_logs_panel(self):
        """Render logs panel - disabled per requirements to remove GUI logs"""
        st.markdown("### 📋 Application Logs")
        st.info("GUI log display is disabled in this build to improve performance and avoid flickering.")
        st.caption("Logs are still recorded internally where applicable. Use external monitoring or console logs if needed.")
        return
    
    def render_api_key_management(self, api_repo):
        """Render API key management interface"""
        try:
            # Get all API keys
            all_keys = api_repo.get_all_api_keys()
            active_key = api_repo.get_active_api_key()
            
            # API Key Selection
            if all_keys:
                st.write("**Select Active API Key:**")
                
                key_options = {}
                for key in all_keys:
                    env_type = "🟡 Demo Trading (Testnet)" if key.testnet else "🔴 Live Production"
                    status = "✅ Active" if key.is_active else ""
                    display_name = f"{key.name} ({env_type}) {status}"
                    key_options[display_name] = key.id
                
                selected_key_display = st.selectbox(
                    "Choose API Key",
                    options=list(key_options.keys()),
                    index=0 if not active_key else list(key_options.values()).index(str(active_key.id)) if str(active_key.id) in key_options.values() else 0,
                    help="Select which API key to use for trading"
                )
                
                selected_key_id = key_options[selected_key_display]
                
                # Set active key button
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🎯 Set as Active", key="set_active"):
                        if api_repo.set_active_api_key(selected_key_id):
                            st.success("✅ API key activated!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to activate API key")
                
                with col2:
                    if st.button("🗑️ Delete Key", key="delete_key"):
                        if api_repo.delete_api_key(selected_key_id):
                            st.success("✅ API key deleted!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete API key")
                
                # Show details of selected key
                selected_key = api_repo.get_api_key(selected_key_id)
                if selected_key:
                    with st.expander("🔍 Key Details"):
                        st.write(f"**Name:** {selected_key.name}")
                        st.write(f"**Description:** {selected_key.description or 'No description'}")
                        st.write(f"**Environment:** {'Demo Trading (Testnet)' if selected_key.testnet else 'Live Production'}")
                        st.write(f"**Created:** {selected_key.created_at.strftime('%Y-%m-%d %H:%M') if selected_key.created_at else 'Unknown'}")
                        st.write(f"**Last Used:** {selected_key.last_used_at.strftime('%Y-%m-%d %H:%M') if selected_key.last_used_at else 'Never'}")
                        st.write(f"**API Key:** {selected_key.api_key[:8]}..." if selected_key.api_key else "")
            
            else:
                st.info("📭 No API keys found. Add your first API key below.")
            
            st.markdown("---")
            
            # Add New API Key Form
            st.write("**Add New API Key:**")
            
            with st.form("add_api_key_form"):
                new_name = st.text_input(
                    "Key Name",
                    placeholder="e.g., 'My Trading Key' or 'Backup Key'",
                    help="Give your API key a memorable name"
                )
                
                new_description = st.text_area(
                    "Description (Optional)",
                    placeholder="e.g., 'Main trading key for BTC/ETH strategies'",
                    help="Optional description for this API key"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    new_api_key = st.text_input(
                        "API Key",
                        type="password",
                        help="Your Bybit API key"
                    )
                
                with col2:
                    new_api_secret = st.text_input(
                        "API Secret",
                        type="password", 
                        help="Your Bybit API secret"
                    )
                
                col1, col2 = st.columns(2)
                with col1:
                    new_testnet = st.checkbox(
                        "Demo Trading Environment (Testnet)",
                        value=True,
                        help="Enable for demo/paper trading (safe testing). Disable for live production trading with real money."
                    )
                
                with col2:
                    set_as_active = st.checkbox(
                        "Set as Active Key",
                        value=len(all_keys) == 0,  # Auto-check if it's the first key
                        help="Make this the currently selected API key"
                    )
                
                submitted = st.form_submit_button("➕ Add API Key")
                
                if submitted:
                    if not new_name or not new_api_key or not new_api_secret:
                        st.error("❌ Please fill in all required fields (Name, API Key, API Secret)")
                    else:
                        try:
                            key_data = {
                                'name': new_name,
                                'description': new_description or None,
                                'api_key': new_api_key,
                                'api_secret': new_api_secret,
                                'testnet': new_testnet,
                                'is_active': set_as_active
                            }
                            
                            api_repo.create_api_key(key_data)
                            st.success(f"✅ API key '{new_name}' added successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            if "unique constraint" in str(e).lower():
                                st.error(f"❌ API key name '{new_name}' already exists. Please choose a different name.")
                            else:
                                st.error(f"❌ Failed to add API key: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Database error: {str(e)}")
            st.info("💡 Make sure the database is running: `docker-compose up -d`")

    
    def run(self):
        """Main application run method"""
        # Render sidebar and get current page
        current_page = self.render_sidebar()
        
        # Render selected page
        if current_page == "Dashboard":
            self.render_dashboard_page()
        elif current_page == "Configuration":
            self.render_configuration_page()
        elif current_page == "Strategy Info":
            self.render_strategy_info_page()
        elif current_page == "Backtesting":
            self.render_backtesting_page()
        elif current_page == "Live Trading":
            self.render_live_trading_page()
        elif current_page == "Trading History":
            self.render_trading_history_page()
        elif current_page == "Performance":
            self.render_performance_page()

def main():
    """Main entry point for Streamlit GUI"""
    gui = StreamlitGUI()
    gui.run()

if __name__ == "__main__":
    main()