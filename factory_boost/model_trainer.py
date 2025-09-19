import pandas as pd
import numpy as np
import joblib
import json
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from data_processor import process_data_from_single_csv
import matplotlib.pyplot as plt  # <--- NOWY IMPORT
import seaborn as sns  # <--- NOWY IMPORT


# --- NOWA FUNKCJA DO RYSOWANIA WYKRESU ---
def plot_feature_importance(model, feature_names, strategy):
    """
    Tworzy i zapisuje wykres ważności cech dla wytrenowanego modelu.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 16))
    plt.title(f"Ważność Cech (Feature Importance) - Strategia: {strategy.upper()}")
    # Użyjemy seaborn dla ładniejszego wykresu
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
    plt.tight_layout()

    # Zapisz wykres do pliku
    filename = f'feature_importance_{strategy}.png'
    plt.savefig(filename)
    print(f"\nZapisano wykres ważności cech do pliku: {filename}")
    plt.close()


# --- Definicje celów dla modelu (bez zmian) ---
def define_long_target(df, in_x_bars=12, prof_tresh=0.02, loss_tresh=0.01):
    df['y'] = 0
    df.reset_index(inplace=True)
    for i in range(len(df) - in_x_bars):
        for j in range(1, in_x_bars + 1):
            if df['high'].iloc[i + j] / df['close'].iloc[i] - 1 >= prof_tresh:
                df.loc[i, 'y'] = 2
                break
            if df['low'].iloc[i + j] / df['close'].iloc[i] - 1 <= -loss_tresh:
                df.loc[i, 'y'] = 0
                break
    df.set_index('timestamp', inplace=True)
    return df


def define_short_target(df, in_x_bars=12, prof_tresh=0.02, loss_tresh=0.01):
    df['y'] = 0
    df.reset_index(inplace=True)
    for i in range(len(df) - in_x_bars):
        for j in range(1, in_x_bars + 1):
            if df['low'].iloc[i + j] / df['close'].iloc[i] - 1 <= -prof_tresh:
                df.loc[i, 'y'] = 2
                break
            if df['high'].iloc[i + j] / df['close'].iloc[i] - 1 >= loss_tresh:
                df.loc[i, 'y'] = 0
                break
    df.set_index('timestamp', inplace=True)
    return df


# --- Główna funkcja trenująca ---
def train_model(args):
    print(f"--- Rozpoczynanie trenowania modelu dla strategii: {args.strategy.upper()} ---")

    data = process_data_from_single_csv(args.data_file, args.start_date, args.end_date)

    if data is None or data.empty:
        print("Nie udało się przygotować danych. Przerywanie trenowania.")
        return

    print("Definiowanie zmiennej docelowej (y)...")
    if args.strategy == 'long':
        data = define_long_target(data)
    else:
        data = define_short_target(data)

    data.dropna(inplace=True)

    cols_to_drop = list(data.filter(regex='^(id|timeframe|created_at|open|high|low|close|volume).*').columns)
    cols_to_drop.append('y')

    X = data.drop(columns=cols_to_drop)
    y = data['y']

    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    if X.empty or y.value_counts().min() < 5:
        print("BŁĄD: Niewystarczająca ilość danych lub klas do przeprowadzenia treningu po czyszczeniu.")
        return

    feature_names = X.columns.tolist()
    print(f"Model będzie trenowany na {len(feature_names)} cechach.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Trenowanie modelu RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_scaled, y)
    print("Trenowanie zakończone.")

    # --- WYWOŁANIE NOWEJ FUNKCJI PO TRENINGU ---
    plot_feature_importance(model, feature_names, args.strategy)

    # Zapisywanie artefaktów (bez zmian)
    model_filename = f'trading_model_{args.strategy}.joblib'
    scaler_filename = f'scaler_{args.strategy}.joblib'
    features_filename = f'feature_names_{args.strategy}.json'

    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    with open(features_filename, 'w') as f:
        json.dump(feature_names, f)

    print(f"\n--- Zakończono! Zapisano pliki dla strategii '{args.strategy}'. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trener modeli AI do strategii tradingowych.")
    parser.add_argument("--strategy", type=str, required=True, choices=['long', 'short'],
                        help="Strategia do trenowania ('long' lub 'short').")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Ścieżka do JEDNEGO pliku CSV z surowymi danymi historycznymi.")
    parser.add_argument("--start-date", type=str, required=True,
                        help="Data początkowa dla danych treningowych (format: RRRR-MM-DD).")
    parser.add_argument("--end-date", type=str, required=True,
                        help="Data końcowa dla danych treningowych (format: RRRR-MM-DD).")

    args = parser.parse_args()
    train_model(args)