# config.py
from datetime import datetime

# --- Konfiguracja Środowiska ---
TICKER = "ETHUSDT"
TICKER_NAME_FOR_MODELS = "ETH"
MODE = "backtest"  # 'backtest' lub 'live'

# --- Konfiguracja Backtestu ---
START_DATE = "2025-04-01"
END_DATE = "2025-04-14"
INITIAL_CAPITAL = 1000.0
DEBUG_MODE = True
# END_DATE = datetime.now().strftime('%Y-%m-%d')

# --- Zarządzanie Ryzykiem ---
RISK_PERCENT = 0.02
LEVERAGE = 1
ATR_MULTIPLIER = 2.0
RRR = 2.5
TRADE_COST_USD = 1.0

# --- Logika Wejścia/Wyjścia ---
ENTRY_VOTES = 2
EXIT_SIGNAL_PERSISTENCE = 1
MIN_CONF_MOMENTUM = 0.71
MIN_CONF_REVERSION = 0.73
MIN_CONF_PA = 0.59

# --- Mechaniki Prowadzenia Pozycji ---
BREAKEVEN_TRIGGER_PERCENT = 0.1
TRAILING_SL_TRIGGER_R = 1.0
TRAILING_SL_DISTANCE_ATR = 1.0

# Slippage (w bps = 1/10000)
SLIPPAGE_BPS_STOP = 5  # 0.05% dla SL/TSL/BE
SLIPPAGE_BPS_TP   = 2  # 0.02% dla TP
