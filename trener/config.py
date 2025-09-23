# config.py
from sklearn.preprocessing import StandardScaler

# --- 1. Konfiguracja Danych i API ---
TICKER = "DOGEUSDT"
TRAIN_START_DATE = "2023-01-01"
TRAIN_END_DATE = "2023-02-28"
BASE_TIMEFRAME = '5m'  # Główny interwał analizy ('5m', '15m', '1h')

# --- 2. Konfiguracja Procesu Treningu ---
# Ile najlepszych cech wybrać do finalnego modelu
TOP_N_FEATURES = 25
CV_SPLITS = 5
HOLDOUT_SIZE = 0.2  # 20% danych jako zbiór testowy (holdout)
RANDOM_STATE = 42

# --- 3. Konfiguracja Narzędzi ---
# Skaler używany do normalizacji cech.
SCALER = StandardScaler()
