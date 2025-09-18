import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import joblib
import json
import warnings
import argparse  # Nowy import
from data_processor import load_and_prepare_timeframe

warnings.filterwarnings('ignore', category=UserWarning)

# --- KONFIGURACJA TRENINGU ---
DATA_FILE_1H = 'historical_data_icp_1h.csv'
DATA_FILE_4H = 'historical_data_icp_4h.csv'
DATA_FILE_1D = 'historical_data_icp_1D.csv'
TAKE_PROFIT_PERCENT = 0.02
STOP_LOSS_PERCENT = 0.01
EVALUATION_WINDOW_HOURS = 24


# ---------------------------------

def create_triple_barrier_target_long(data, tp_perc=TAKE_PROFIT_PERCENT, sl_perc=STOP_LOSS_PERCENT,
                                      window=EVALUATION_WINDOW_HOURS):
    """
    Tworzy cel dla strategii LONG.
    Target = 1, jeśli cena osiągnie TP.
    Target = -1, jeśli cena osiągnie SL.
    """
    df = data.copy()
    df['target'] = 0
    for i in range(len(df) - window):
        entry_price = df['close_1h'].iloc[i]
        tp_price = entry_price * (1 + tp_perc)
        sl_price = entry_price * (1 - sl_perc)
        for j in range(1, window + 1):
            future_high = df['high_1h'].iloc[i + j]
            future_low = df['low_1h'].iloc[i + j]
            if future_low <= sl_price:
                df.loc[df.index[i], 'target'] = -1  # Przegrana (Stop Loss)
                break
            if future_high >= tp_price:
                df.loc[df.index[i], 'target'] = 1  # Wygrana (Take Profit)
                break
    return df['target']


def create_triple_barrier_target_short(data, tp_perc=TAKE_PROFIT_PERCENT, sl_perc=STOP_LOSS_PERCENT,
                                       window=EVALUATION_WINDOW_HOURS):
    """
    Tworzy cel dla strategii SHORT. Logika jest odwrócona.
    Target = 1 (sukces), jeśli cena osiągnie dolny próg (TP dla short).
    Target = -1 (porażka), jeśli cena osiągnie górny próg (SL dla short).
    """
    df = data.copy()
    df['target'] = 0
    for i in range(len(df) - window):
        entry_price = df['close_1h'].iloc[i]
        tp_price = entry_price * (1 - tp_perc)  # TP jest poniżej ceny wejścia
        sl_price = entry_price * (1 + sl_perc)  # SL jest powyżej ceny wejścia
        for j in range(1, window + 1):
            future_high = df['high_1h'].iloc[i + j]
            future_low = df['low_1h'].iloc[i + j]
            if future_high >= sl_price:
                df.loc[df.index[i], 'target'] = -1  # Przegrana (Stop Loss)
                break
            if future_low <= tp_price:
                df.loc[df.index[i], 'target'] = 1  # Wygrana (Take Profit)
                break
    return df['target']


def main(strategy: str):
    print(f"--- Rozpoczęcie procesu trenowania i zapisu modelu dla strategii: {strategy.upper()} ---")

    print("Przetwarzanie danych...")
    df_1h_full = load_and_prepare_timeframe(DATA_FILE_1H, '1h', include_ohlcv=True)
    df_4h_features = load_and_prepare_timeframe(DATA_FILE_4H, '4h')
    df_1d_features = load_and_prepare_timeframe(DATA_FILE_1D, '1d')

    print("Łączenie danych wielointerwałowych...")
    final_dataset = pd.merge_asof(df_1h_full, df_4h_features, left_index=True, right_index=True, direction='backward')
    final_dataset = pd.merge_asof(final_dataset, df_1d_features, left_index=True, right_index=True,
                                  direction='backward')

    print(
        f"Tworzenie celu (Triple Barrier Method) dla {strategy.upper()}: TP={TAKE_PROFIT_PERCENT * 100:.2f}%, SL={STOP_LOSS_PERCENT * 100:.2f}%, Okno={EVALUATION_WINDOW_HOURS}h")

    # --- KLUCZOWA ZMIANA: WYBÓR FUNKCJI CELU ---
    if strategy == 'long':
        final_dataset['target'] = create_triple_barrier_target_long(final_dataset)
    elif strategy == 'short':
        final_dataset['target'] = create_triple_barrier_target_short(final_dataset)
    else:
        raise ValueError("Strategia musi być 'long' lub 'short'")

    final_dataset.dropna(inplace=True)

    y = final_dataset['target']
    ohlcv_cols = [col for col in final_dataset.columns if
                  any(sub in col for sub in ['open', 'high', 'low', 'close', 'volume'])]
    X = final_dataset.drop(columns=ohlcv_cols + ['target'])

    print(f"\n--- Trenowanie finalnego modelu ({strategy.upper()}) na wszystkich dostępnych danych ---")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    best_params = {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}
    model = XGBClassifier(**best_params, objective='multi:softmax', num_class=3, use_label_encoder=False,
                          eval_metric='mlogloss', random_state=42)

    y_mapped = y.map({-1: 0, 0: 1, 1: 2})
    model.fit(X_scaled, y_mapped)
    print("Trenowanie zakończone.")

    # --- ZMIANA: ZAPIS PLIKÓW Z PRZYROSTKIEM STRATEGII ---
    model_filename = f'trading_model_{strategy}.joblib'
    scaler_filename = f'scaler_{strategy}.joblib'
    features_filename = f'feature_names_{strategy}.json'

    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    with open(features_filename, 'w') as f:
        json.dump(X.columns.tolist(), f)

    print(
        f"\nModel ({model_filename}), skaler ({scaler_filename}) i nazwy cech ({features_filename}) zostały pomyślnie zapisane.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trenowanie modelu tradingowego dla strategii LONG lub SHORT.')
    parser.add_argument('--strategy', type=str, required=True, choices=['long', 'short'],
                        help="Wybierz strategię do trenowania: 'long' lub 'short'")
    args = parser.parse_args()
    main(args.strategy)