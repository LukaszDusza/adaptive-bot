# config.py
from sklearn.preprocessing import StandardScaler

# --- 1. Konfiguracja Danych i Cache ---
TICKER = "DOGEUSDT"
TRAIN_START_DATE = "2022-01-01"
TRAIN_END_DATE = "2025-04-30"
BASE_TIMEFRAME = '5m'
FEATURE_TIMEFRAMES = [
    # Co minutę od 5m do 15m
    '5m', '6m', '7m', '8m', '9m', '10m', '11m', '12m', '13m', '14m', '15m',
    # Co 5 minut do 1h
    '20m', '25m', '30m', '35m', '40m', '45m', '50m', '55m',
    # Co godzinę do 8h
    '1h', '2h', '3h', '4h', '5h', '6h', '7h', '8h'
]
RAW_DATA_CACHE_DIR = "data_cache/raw"
FEATURES_CACHE_DIR = "data_cache/features"

# === ZMIANA 1: Rozbudowana konfiguracja celu ===
# --- 2. Konfiguracja Celu (Target) ---
TARGET_TYPE = 'DYNAMIC_ATR'  # Wybierz: 'FIXED' lub 'DYNAMIC_ATR'
HORIZON_BARS = 20

# Ustawienia dla TARGET_TYPE = 'FIXED'
PRICE_TARGET_PCT = 0.01

# Ustawienia dla TARGET_TYPE = 'DYNAMIC_ATR'
# Mnożniki określają, jak szerokie będą bariery w stosunku do ATR
# Np. 2.0 / 1.0 oznacza celowanie w zysk 2x większy od ryzyka (Risk/Reward Ratio = 2)
ATR_TP_MULTIPLIER = 1.5  # Mnożnik dla Take Profit
ATR_SL_MULTIPLIER = 1.5  # Mnożnik dla Stop Loss

# --- 3. Konfiguracja Procesu Treningu i Walidacji ---
TOP_N_FEATURES = 25
CV_SPLITS = 5
HOLDOUT_SIZE = 0.2
RANDOM_STATE = 42
OPTUNA_TRIALS = 200


# --- 4. Konfiguracja Inżynierii Cech (Feature Engineering) ---
class FeatureConfig:
    # Ogólne
    ADX_TREND_THRESHOLD = 25

    # Fibonacci
    FIBO_WINDOW = 100

    # Dywergencje
    DIVERGENCE_WINDOW = 28

    # Price Action
    PA_VOLUME_WINDOW = 20
    PA_LAG_STEPS = [1, 2, 3]

    # Parametry wskaźników
    RSI_LENGTH = 14
    ATR_LENGTH = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BBANDS_LENGTH = 20
    STOCH_K = 14
    ADX_LENGTH = 14
    CCI_LENGTH = 20
    MFI_LENGTH = 14
    AROON_LENGTH = 25
    EMA_FAST_LEN = 20
    EMA_SLOW_LEN = 50
    EMA_TREND_LEN = 200
    SKEW_LENGTH = 30
    KURTOSIS_LENGTH = 30

    # === NOWA SEKCJA: Konfiguracja cech stacjonarnych ===
    STATIONARY_WINDOW = 100
    # Lista "bazowych" nazw wskaźników do przekształcenia w wersje stacjonarne
    STATIONARITY_TARGET_INDICATORS = ['RSI', 'MACDh', 'ADX']

# --- 5. Konfiguracja Narzędzi ---
SCALER = StandardScaler()
