import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


# ... (Funkcje load_and_prepare_timeframe i create_final_dataset pozostają bez zmian) ...
def load_and_prepare_timeframe(filepath: str, interval_suffix: str) -> pd.DataFrame:
    """Wczytuje dane, oblicza podstawowe i zaawansowane wskaźniki."""
    print(f"Przetwarzanie interwału: {interval_suffix} z pliku {filepath}...")

    column_names = ['id', 'interval', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'updated_at']
    df = pd.read_csv(
        filepath, header=None, names=column_names,
        usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'], parse_dates=['timestamp']
    )
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.stoch(length=14, append=True)
    df.ta.ao(append=True)
    df['candle_size'] = df['high'] - df['low']
    df['body_size'] = (df['close'] - df['open']).abs()

    df = df.drop(columns=['open', 'high', 'low', 'close', 'volume'])
    df = df.add_suffix(f'_{interval_suffix}')

    return df


def create_final_dataset(df_base: pd.DataFrame, higher_timeframes: list):
    """Łączy wskaźniki z wyższych interwałów z bazowym DataFrame."""
    print("\nŁączenie danych z różnych interwałów...")
    final_df = df_base.copy()

    for df_htf in higher_timeframes:
        final_df = pd.merge_asof(
            left=final_df.sort_index(), right=df_htf.sort_index(),
            on='timestamp', direction='backward'
        )

    final_df.set_index('timestamp', inplace=True)
    print("Dane połączone pomyślnie.")
    return final_df


def create_triple_barrier_target(df, look_forward_periods=24, tp_threshold=0.02, sl_threshold=0.01):
    """
    Tworzy zmienną docelową w oparciu o Metodę Potrójnej Bariery.
    Returns:
        1: Take Profit trafiony jako pierwszy
       -1: Stop Loss trafiony jako pierwszy
        0: Żadna z barier nie została trafiona w oknie czasowym
    """
    print(
        f"\nTworzenie celu (Triple Barrier Method): TP={tp_threshold:.2%}, SL={sl_threshold:.2%}, Okno={look_forward_periods}h")
    prices = df['close']
    outcomes = pd.Series(0, index=df.index)

    take_profit_barrier = prices * (1 + tp_threshold)
    stop_loss_barrier = prices * (1 - sl_threshold)

    for i in range(len(prices) - look_forward_periods):
        window = prices.iloc[i + 1: i + 1 + look_forward_periods]

        # Sprawdzamy, czy cena dotknęła TP
        tp_hit_time = window[window >= take_profit_barrier.iloc[i]].first_valid_index()
        # Sprawdzamy, czy cena dotknęła SL
        sl_hit_time = window[window <= stop_loss_barrier.iloc[i]].first_valid_index()

        if tp_hit_time and sl_hit_time:
            # Jeśli obie bariery zostały trafione, wybieramy tę, która była pierwsza
            outcomes.iloc[i] = 1 if tp_hit_time < sl_hit_time else -1
        elif tp_hit_time:
            outcomes.iloc[i] = 1
        elif sl_hit_time:
            outcomes.iloc[i] = -1

    return outcomes


def train_and_evaluate(X, y):
    """
    Dzieli dane, trenuje i ocenia model przewidujący 3 klasy (KUP/SPRZEDAJ/CZEKAJ).
    """
    print("\n--- Przygotowanie do trenowania finalnego systemu transakcyjnego ---")
    split_percentage = 0.8
    split_index = int(len(X) * split_percentage)

    X_train_raw, X_test_raw = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Dla klasyfikacji wieloklasowej nie używamy już scale_pos_weight
    best_params = {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}

    # Zmieniamy cel na 'multi:softmax' i podajemy liczbę klas
    model = XGBClassifier(
        **best_params,
        objective='multi:softmax',
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )

    # Mapujemy nasze wartości (-1, 0, 1) na (0, 1, 2) dla XGBoost
    y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
    y_test_mapped = y_test.map({-1: 0, 0: 1, 1: 2})

    print("Trenowanie modelu wieloklasowego...")
    model.fit(X_train_scaled, y_train_mapped)
    print("Trenowanie zakończone.")

    print("\n--- Ocena finalnego systemu na danych testowych ---")
    y_pred_mapped = model.predict(X_test_scaled)

    print("\nRaport klasyfikacji:")
    print(classification_report(y_test_mapped, y_pred_mapped,
                                target_names=['STRATA/SL (-1)', 'NEUTRALNIE (0)', 'ZYSK/TP (1)']))


def main():
    df_1h_features = load_and_prepare_timeframe('historical_data_icp_1h.csv', '1h')
    df_4h_features = load_and_prepare_timeframe('historical_data_icp_4h.csv', '4h')
    df_1d_features = load_and_prepare_timeframe('historical_data_icp_1D.csv', '1d')

    base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df_1h_base = pd.read_csv('historical_data_icp_1h.csv', header=None, names=[
        'id', 'interval', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'updated_at'
    ], usecols=base_columns, parse_dates=['timestamp'])

    df_merged = pd.merge(df_1h_base, df_1h_features, on='timestamp')
    final_dataset = create_final_dataset(df_merged, [df_4h_features, df_1d_features])

    # Tworzymy nową, zaawansowaną zmienną docelową
    final_dataset['target'] = create_triple_barrier_target(final_dataset, tp_threshold=0.02, sl_threshold=0.01)
    final_dataset.dropna(inplace=True)

    y = final_dataset['target']
    X = final_dataset.drop(columns=['open', 'high', 'low', 'close', 'volume', 'target'])

    train_and_evaluate(X, y)


if __name__ == "__main__":
    main()
