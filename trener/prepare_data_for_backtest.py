import pandas as pd
import config
import os
import glob
import sys
from tqdm import tqdm

print("--- Przygotowywanie danych dla backtestera ---")

try:
    # KROK 1: Jawnie znajdź i wczytaj najnowszy plik z CECHAMI
    print("KROK 1/7: Wyszukiwanie pliku z cechami...")
    feature_files = glob.glob(f'{config.FEATURES_CACHE_DIR}/features_*.parquet')
    if not feature_files:
        raise FileNotFoundError("Nie znaleziono żadnych plików z cechami (features_*.parquet).")
    latest_feature_file = max(feature_files, key=os.path.getmtime)
    print(f"-> Używanie pliku z cechami: {os.path.basename(latest_feature_file)}")
    df_features = pd.read_parquet(latest_feature_file)

    # KROK 2: Jawnie znajdź i wczytaj najnowszy plik z CELAMI
    print("KROK 2/7: Wyszukiwanie pliku z celami...")
    target_files = glob.glob(f'{config.FEATURES_CACHE_DIR}/targets_*.parquet')
    if not target_files:
        raise FileNotFoundError("Nie znaleziono żadnych plików z celami (targets_*.parquet).")
    latest_target_file = max(target_files, key=os.path.getmtime)
    print(f"-> Używanie pliku z celami: {os.path.basename(latest_target_file)}")
    df_targets = pd.read_parquet(latest_target_file)

    # KROK 3: Połącz cechy i cele w jedną ramkę danych
    print("KROK 3/7: Łączenie cech i celów...")
    df_full_features = df_features.join(df_targets)

except (ValueError, FileNotFoundError) as e:
    print(f"\nBŁĄD KRYTYCZNY: {e}")
    print("Upewnij się, że skrypt 'model_trainer.py' został uruchomiony i pomyślnie wygenerował pliki w cache.")
    sys.exit(1)


# KROK 4: Wczytaj surowe dane OHLCV
print("KROK 4/7: Wczytywanie surowych danych OHLCV...")
raw_file = f"{config.RAW_DATA_CACHE_DIR}/{config.TICKER}_{config.TRAIN_START_DATE}_{config.TRAIN_END_DATE}.csv"
df_raw = pd.read_csv(raw_file, index_col='timestamp', parse_dates=True)

# KROK 5: Połącz surowe dane z pełnym zestawem cech i celów
print("KROK 5/7: Łączenie danych i usuwanie wierszy z brakami...")
cols_to_drop = ['open', 'high', 'low', 'close', 'volume', 'turnover']
df_backtest = df_raw.join(df_full_features.drop(columns=cols_to_drop, errors='ignore'), how='inner')
df_backtest.dropna(inplace=True)

# KROK 6 (NOWY): Weryfikacja i usunięcie zduplikowanych znaczników czasu
print("KROK 6/7: Weryfikacja unikalności znaczników czasu (timestamp)...")
if df_backtest.index.duplicated().any():
    print(f"-> UWAGA: Znaleziono {df_backtest.index.duplicated().sum()} zduplikowanych indeksów. Usuwanie...")
    df_backtest = df_backtest[~df_backtest.index.duplicated(keep='first')]
    print("-> Duplikaty zostały usunięte. Pozostawiono tylko pierwsze wystąpienia.")
else:
    print("-> Weryfikacja zakończona pomyślnie. Brak duplikatów.")


# KROK 7: Zapisz finalny, kompletny plik z paskiem postępu
print("KROK 7/7: Zapisywanie finalnego pliku CSV...")
output_filename = "data_for_backtest.csv"

chunk_size = 50000
with tqdm(total=len(df_backtest), desc="Zapisywanie do CSV") as pbar:
    for i in range(0, len(df_backtest), chunk_size):
        chunk = df_backtest.iloc[i:i + chunk_size]
        if i == 0:
            chunk.to_csv(output_filename, mode='w', index=True, header=True)
        else:
            chunk.to_csv(output_filename, mode='a', index=True, header=False)
        pbar.update(len(chunk))


print(f"\n\nPomyślnie utworzono plik '{output_filename}' gotowy do użycia w backtesterze.")
print(f"Liczba wierszy: {len(df_backtest)}")
print(f"Liczba kolumn: {len(df_backtest.columns)}")
