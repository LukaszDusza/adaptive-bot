import pandas as pd
import numpy as np
import joblib
import json
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV  # <--- NOWY IMPORT
from data_processor import process_data_from_single_csv
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(model, feature_names, strategy):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 16))
    plt.title(f"Ważność Cech (Feature Importance) - Strategia: {strategy.upper()}")
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
    plt.tight_layout()
    filename = f'feature_importance_{strategy}.png'
    plt.savefig(filename)
    print(f"\nZapisano wykres ważności cech do pliku: {filename}")
    plt.close()


def define_long_target_dynamic(df, in_x_bars=12, atr_profit_multiplier=2.0, atr_loss_multiplier=1.0):
    df['y'] = 1
    atr_col_name = 'ATRr_14_1h'
    if atr_col_name not in df.columns:
        print(f"BŁĄD: Brak kolumny ATR '{atr_col_name}' w danych.")
        return df

    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    atr_values = df[atr_col_name].values
    y_values = df['y'].values.copy()

    for i in range(len(df) - in_x_bars):
        entry_price = close_prices[i]
        atr_value = atr_values[i]
        if atr_value <= 0 or np.isnan(atr_value): continue

        take_profit_price = entry_price + (atr_value * atr_profit_multiplier)
        stop_loss_price = entry_price - (atr_value * atr_loss_multiplier)

        for j in range(1, in_x_bars + 1):
            future_high, future_low = high_prices[i + j], low_prices[i + j]
            # Jeśli trafiono TP, oznacz jako 2 (wygrana) i przerwij
            if future_high >= take_profit_price:
                y_values[i] = 2
                break
            # Jeśli trafiono SL, oznacz jako 0 (przegrana) i przerwij
            if future_low <= stop_loss_price:
                y_values[i] = 0
                break

    df['y'] = y_values
    return df

def define_short_target_dynamic(df, in_x_bars=12, atr_profit_multiplier=2.0, atr_loss_multiplier=1.0):
    df['y'] = 1
    atr_col_name = 'ATRr_14_1h'
    if atr_col_name not in df.columns:
        print(f"BŁĄD: Brak kolumny ATR '{atr_col_name}' w danych.")
        return df

    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    atr_values = df[atr_col_name].values
    y_values = df['y'].values.copy()

    for i in range(len(df) - in_x_bars):
        entry_price = close_prices[i]
        atr_value = atr_values[i]
        if atr_value <= 0 or np.isnan(atr_value): continue

        take_profit_price = entry_price - (atr_value * atr_profit_multiplier)
        stop_loss_price = entry_price + (atr_value * atr_loss_multiplier)

        for j in range(1, in_x_bars + 1):
            future_high, future_low = high_prices[i + j], low_prices[i + j]
            # Jeśli trafiono TP, oznacz jako 2 (wygrana) i przerwij
            if future_low <= take_profit_price:
                y_values[i] = 2
                break
            # Jeśli trafiono SL, oznacz jako 0 (przegrana) i przerwij
            if future_high >= stop_loss_price:
                y_values[i] = 0
                break

    df['y'] = y_values
    return df


def train_model(args):
    print(f"--- Rozpoczynanie trenowania i OPTYMALIZACJI modelu dla strategii: {args.strategy.upper()} ---")
    data = process_data_from_single_csv(args.data_file, args.start_date, args.end_date)

    if data is None or data.empty:
        print("Nie udało się przygotować danych.");
        return

    print("Definiowanie DYNAMICZNEJ zmiennej docelowej (y) w oparciu o ATR...")
    if args.strategy == 'long':
        data = define_long_target_dynamic(data)
    else:
        data = define_short_target_dynamic(data)

    data.dropna(inplace=True)

    cols_to_drop = list(data.filter(regex='^(id|timeframe|created_at|open|high|low|close|volume).*').columns)
    cols_to_drop.append('y')
    X = data.drop(columns=cols_to_drop)
    y = data['y']
    print("Rozkład klas w danych treningowych:")
    print(y.value_counts())

    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    if X.empty or y.value_counts().min() < 5:
        print("BŁĄD: Niewystarczająca ilość danych.");
        return

    feature_names = X.columns.tolist()
    print(f"Model będzie trenowany na {len(feature_names)} cechach.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n--- Rozpoczynanie poszukiwania najlepszych hiperparametrów (Grid Search) ---")

    # Krok 1: Zdefiniuj siatkę parametrów do przetestowania
    param_grid = {
        'n_estimators': [100, 150],  # Liczba drzew w lesie
        'max_depth': [10, 20, None],  # Maksymalna głębokość drzewa
        'min_samples_leaf': [1, 2, 4],  # Minimalna liczba próbek w liściu
        'class_weight': ['balanced']
    }

    # Krok 2: Zainicjuj GridSearchCV
    # cv=3 -> 3-krotna walidacja krzyżowa
    # scoring='f1_weighted' -> metryka oceny, dobra dla niezbalansowanych klas
    # n_jobs=-1 -> użyj wszystkich dostępnych rdzeni CPU, aby przyspieszyć
    # verbose=2 -> pokazuj szczegółowe logi z postępu
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    # Krok 3: Uruchom poszukiwanie
    grid_search.fit(X_scaled, y)

    # Krok 4: Wybierz najlepszy znaleziony model i jego parametry
    print("\n--- Poszukiwanie zakończone ---")
    print(f"Najlepsze znalezione parametry: {grid_search.best_params_}")

    model = grid_search.best_estimator_

    plot_feature_importance(model, feature_names, args.strategy)

    model_filename = f'trading_model_{args.strategy}.joblib'
    scaler_filename = f'scaler_{args.strategy}.joblib'
    features_filename = f'feature_names_{args.strategy}.json'
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    with open(features_filename, 'w') as f:
        json.dump(feature_names, f)
    print(f"\n--- Zakończono! Zapisano ZOPTYMALIZOWANE pliki dla strategii '{args.strategy}'. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trener i optymalizator modeli AI.")
    parser.add_argument("--strategy", type=str, required=True, choices=['long', 'short'])
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)

    args = parser.parse_args()
    train_model(args)