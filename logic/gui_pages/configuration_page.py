"""
Configuration Page Module

Handles the configuration page rendering for the Streamlit GUI,
including API key management, trading settings, and risk management.
"""

import logging
import streamlit as st
from typing import Dict, Any, Optional

from logic.gui_config import GUIConfigManager

logger = logging.getLogger(__name__)


class ConfigurationPage:
    """Configuration page renderer"""

    @staticmethod
    def render():
        """Main configuration page rendering"""
        st.title("⚙️ Configuration")
        st.markdown("Configure your trading bot settings and API connections.")
        
        ConfigurationPage._render_database_status()
        ConfigurationPage._render_api_configuration()
        ConfigurationPage._render_trading_configuration()
        ConfigurationPage._render_risk_management()
        ConfigurationPage._render_save_configuration()

    @staticmethod
    def _render_database_status():
        """Render database connection status"""
        try:
            # Force reload database configuration
            st.session_state.config['database_loaded'] = False
            GUIConfigManager.load_database_config()
            st.success("✅ Database connected - configurations loaded automatically")
        except Exception as e:
            st.warning(f"⚠️ Database connection failed: {str(e)}")

    @staticmethod
    def _render_api_configuration():
        """Render API key configuration section"""
        st.subheader("🔑 Bybit API Key Management")
        
        database_available, api_repo = ConfigurationPage._check_database_availability()
        
        if not database_available:
            ConfigurationPage._render_manual_api_input()
        else:
            st.success("🎯 API keys automatically loaded from database")
            ConfigurationPage._render_database_api_management(api_repo)

    @staticmethod
    def _check_database_availability():
        """Check if database is available and return API repository"""
        try:
            from database.models import get_api_key_repository
            api_repo = get_api_key_repository()
            # Test database connection
            api_repo.get_all_api_keys()
            return True, api_repo
        except ImportError as e:
            st.error(f"❌ Database models not available: {str(e)}")
            return False, None
        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
            return False, None

    @staticmethod
    def _render_manual_api_input():
        """Render manual API key input fallback"""
        st.warning("⚠️ Database not available. Using manual API key input.")
        st.info("💡 To enable automatic API key management, ensure your PostgreSQL database is running with `docker-compose up -d`")
        
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
        
        testnet = st.checkbox(
            "Use Testnet",
            value=st.session_state.config.get('bybit_testnet', True),
            help="Use Bybit testnet for safe testing"
        )
        
        # Store in session state
        st.session_state.manual_api_key = api_key
        st.session_state.manual_api_secret = api_secret
        st.session_state.manual_testnet = testnet

    @staticmethod
    def _render_database_api_management(api_repo):
        """Render database-backed API key management"""
        try:
            all_keys = api_repo.get_all_api_keys()
            
            if all_keys:
                st.write("**Available API Keys:**")
                for key in all_keys:
                    status = "🟢 Active" if key.is_active else "⚪ Inactive"
                    network = "🧪 Testnet" if key.testnet else "🏦 Mainnet"
                    st.write(f"• **{key.name}** - {status} ({network})")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"Created: {key.created_at.strftime('%Y-%m-%d %H:%M')}")
                    with col2:
                        if st.button(f"Set Active", key=f"activate_{key.id}"):
                            try:
                                api_repo.set_active_api_key(key.id)
                                st.success(f"✅ {key.name} set as active")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error setting active API key: {e}")
            else:
                st.info("No API keys found in database. Add your first API key:")
                ConfigurationPage._render_add_api_key_form(api_repo)
                
        except Exception as e:
            st.error(f"❌ Error loading API keys: {e}")

    @staticmethod
    def _render_add_api_key_form(api_repo):
        """Render form to add new API key"""
        with st.expander("➕ Add New API Key"):
            key_name = st.text_input("Key Name", placeholder="e.g., 'Trading Account 1'")
            new_api_key = st.text_input("API Key", type="password")
            new_api_secret = st.text_input("API Secret", type="password")
            new_testnet = st.checkbox("Testnet", value=True)
            
            if st.button("💾 Save API Key"):
                if key_name and new_api_key and new_api_secret:
                    try:
                        saved_key = api_repo.create_api_key({
                            'name': key_name,
                            'api_key': new_api_key,
                            'api_secret': new_api_secret,
                            'testnet': new_testnet,
                            'is_active': True  # Make new key active by default
                        })
                        st.success(f"✅ API key '{key_name}' saved successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error saving API key: {e}")
                else:
                    st.error("❌ Please fill in all fields")

    @staticmethod
    def _render_trading_configuration():
        """Render trading configuration section"""
        st.subheader("📈 Trading Configuration")
        
        ConfigurationPage._show_trading_preferences_status()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbols = ConfigurationPage._render_symbol_selection()
        
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
        
        # Store in session state
        st.session_state.trading_symbols = symbols
        st.session_state.trading_timeframe = timeframe
        st.session_state.trading_initial_capital = initial_capital

    @staticmethod
    def _show_trading_preferences_status():
        """Show trading preferences loading status"""
        try:
            from database.models import get_trading_preferences_repository
            prefs_repo = get_trading_preferences_repository()
            active_prefs = prefs_repo.get_active_preferences()
            
            if active_prefs:
                symbols_count = len(st.session_state.config.get('symbols', []))
                timeframe = st.session_state.config.get('timeframe', 'N/A')
                st.info(f"🎯 Trading preferences automatically loaded from database (Symbols: {symbols_count}, Timeframe: {timeframe})")
            else:
                st.warning("⚠️ No active trading preferences found in database - using defaults")
        except Exception as e:
            st.warning(f"⚠️ Could not load trading preferences: {str(e)}")

    @staticmethod
    def _render_symbol_selection():
        """Render symbol selection with descriptions"""
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
        
        return symbols

    @staticmethod
    def _render_risk_management():
        """Render risk management section"""
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
        
        # Store in session state
        st.session_state.risk_per_trade = risk_per_trade
        st.session_state.max_daily_loss = max_daily_loss
        st.session_state.max_drawdown = max_drawdown

    @staticmethod
    def _render_save_configuration():
        """Render save configuration section"""
        if st.button("💾 Save Configuration", key="save_config"):
            ConfigurationPage._save_configuration()

    @staticmethod
    def _save_configuration():
        """Save configuration to session state and database"""
        # Collect values from session state or manual inputs
        config_updates = {}
        
        # API configuration
        if hasattr(st.session_state, 'manual_api_key'):
            config_updates.update({
                'bybit_api_key': st.session_state.manual_api_key,
                'bybit_api_secret': st.session_state.manual_api_secret,
                'bybit_testnet': st.session_state.manual_testnet,
            })
        
        # Trading configuration
        if hasattr(st.session_state, 'trading_symbols'):
            config_updates.update({
                'symbols': st.session_state.trading_symbols,
                'timeframe': st.session_state.trading_timeframe,
                'initial_capital': st.session_state.trading_initial_capital,
            })
        
        # Risk management
        if hasattr(st.session_state, 'risk_per_trade'):
            config_updates.update({
                'risk_per_trade': st.session_state.risk_per_trade,
                'max_daily_loss': st.session_state.max_daily_loss,
                'max_drawdown': st.session_state.max_drawdown,
            })
        
        # Update session state
        st.session_state.config.update(config_updates)
        
        # Save to database
        ConfigurationPage._save_to_database(config_updates)
        
        # Generate environment file
        ConfigurationPage._generate_env_download()

    @staticmethod
    def _save_to_database(config_updates: Dict[str, Any]):
        """Save trading preferences to database"""
        try:
            from database.models import get_trading_preferences_repository
            prefs_repo = get_trading_preferences_repository()
            
            prefs_data = {
                'user_profile': 'default',
                'selected_symbols': config_updates.get('symbols', st.session_state.config.get('symbols', [])),
                'timeframe': config_updates.get('timeframe', st.session_state.config.get('timeframe', '15m')),
                'initial_capital': config_updates.get('initial_capital', st.session_state.config.get('initial_capital', 10000)),
                'risk_per_trade': config_updates.get('risk_per_trade', st.session_state.config.get('risk_per_trade', 0.02)),
                'max_daily_loss': config_updates.get('max_daily_loss', st.session_state.config.get('max_daily_loss', 0.05)),
                'max_drawdown': config_updates.get('max_drawdown', st.session_state.config.get('max_drawdown', 0.15)),
                'max_positions': st.session_state.config.get('max_positions', 1)
            }
            
            saved_prefs = prefs_repo.create_or_update_preferences(prefs_data)
            st.success(f"✅ Configuration saved to database! (ID: {str(saved_prefs.id)[:8]}...)")
            
        except ImportError:
            st.warning("⚠️ Database not available. Configuration saved to session only.")
        except Exception as e:
            st.error(f"❌ Failed to save to database: {e}")
            st.success("✅ Configuration saved to session!")

    @staticmethod
    def _generate_env_download():
        """Generate and offer .env file download"""
        GUIConfigManager.generate_env_file()
        
        # Create download content
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
        
        st.download_button(
            label="📥 Download .env file",
            data=env_content,
            file_name=".env",
            mime="text/plain"
        )