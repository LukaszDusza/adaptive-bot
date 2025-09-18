import pandas as pd
import pandas_ta as ta
import joblib
import json


def load_and_prepare_timeframe(df: pd.DataFrame, interval_suffix: str) -> pd.DataFrame:
    """Oblicza wskaźniki dla DataFrame (dane na żywo)."""
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # Obliczanie wskaźników - biblioteka pandas-ta sama obsłuży NaN na początku
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.stoch(length=14, append=True)
    df.ta.ao(append=True)
    df['candle_size'] = df['high'] - df['low']
    df['body_size'] = (df['close'] - df['open']).abs()

    # Usuwamy oryginalne kolumny OHLCV
    df = df.drop(columns=['open', 'high', 'low', 'close', 'volume'])
    # Dodajemy sufiks do nazw kolumn ze wskaźnikami
    df = df.add_suffix(f'_{interval_suffix}')
    return df


def create_live_features(recent_history_1h, recent_history_4h, recent_history_1d):
    """Tworzy pełen zestaw cech dla ostatniej, aktualnej świecy."""
    df_1h = pd.DataFrame(recent_history_1h)
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])

    df_4h = pd.DataFrame(recent_history_4h)
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])

    df_1d = pd.DataFrame(recent_history_1d)
    df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'])

    # Oblicz wskaźniki dla każdego interwału na podstawie dostarczonej historii
    features_1h = load_and_prepare_timeframe(df_1h.copy(), '1h')
    features_4h = load_and_prepare_timeframe(df_4h.copy(), '4h')
    features_1d = load_and_prepare_timeframe(df_1d.copy(), '1d')

    # Weź tylko ostatni wiersz z każdego zbioru (czyli cechy dla najnowszej świecy)
    current_features_1h = features_1h.iloc[[-1]]
    current_features_4h = features_4h.iloc[[-1]]
    current_features_1d = features_1d.iloc[[-1]]

    # Połącz cechy w jeden wektor
    final_feature_vector = pd.concat([
        current_features_1h.reset_index(drop=True),
        current_features_4h.reset_index(drop=True),
        current_features_1d.reset_index(drop=True)
    ], axis=1)

    return final_feature_vector


def get_prediction(feature_vector):
    """
    Wczytuje model, skaler i listę cech, a następnie zwraca predykcję.
    """
    try:
        model = joblib.load('trading_model.joblib')
        scaler = joblib.load('scaler.joblib')
        with open('feature_names.json', 'r') as f:
            expected_features = json.load(f)

    except FileNotFoundError as e:
        return {"error": f"Nie znaleziono pliku: {e.filename}. Uruchom najpierw skrypt treningowy."}

    # PORZĄDKOWANIE KOLEJNOŚCI KOLUMN
    try:
        feature_vector = feature_vector[expected_features]
    except KeyError as e:
        return {
            "error": f"Brakująca kolumna w danych wejściowych: {e}. Sprawdź, czy proces tworzenia cech jest identyczny."}

    # Skalowanie danych wejściowych
    scaled_features = scaler.transform(feature_vector)

    # Predykcja
    prediction_mapped = model.predict(scaled_features)[0]

    # Mapowanie z powrotem na nasze klasy
    decision_map = {-1: "STRATA/SL", 0: "NEUTRALNIE", 1: "ZYSK/TP"}
    y_map = {0: -1, 1: 0, 2: 1}
    prediction_class = y_map[prediction_mapped]

    return {"prediction_code": int(prediction_class), "decision": decision_map[prediction_class]}


def main():
    """Główna funkcja symulująca zapytanie z API."""
    print("--- Symulacja Serwisu Predykcyjnego ---")

    # W realnym systemie, Twoje API z Javy dostarczyłoby te dane, pobierając je z giełdy
    # Symulujemy to, wczytując 50 ostatnich świec z naszych plików historycznych
    # Ta liczba (50) jest bezpiecznym buforem dla wskaźników.
    CANDLE_BUFFER = 50

    column_names = ['id', 'interval', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'updated_at']
    ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    recent_1h = pd.read_csv('historical_data_icp_1h.csv', header=None, names=column_names, usecols=ohlcv_cols).tail(
        CANDLE_BUFFER).to_dict('records')
    recent_4h = pd.read_csv('historical_data_icp_4h.csv', header=None, names=column_names, usecols=ohlcv_cols).tail(
        CANDLE_BUFFER).to_dict('records')
    recent_1d = pd.read_csv('historical_data_icp_1D.csv', header=None, names=column_names, usecols=ohlcv_cols).tail(
        CANDLE_BUFFER).to_dict('records')

    print("Tworzenie wektora cech dla aktualnych danych...")
    feature_vector = create_live_features(recent_1h, recent_4h, recent_1d)

    print("Pobieranie predykcji z wytrenowanego modelu...")
    result = get_prediction(feature_vector)

    print("\n--- Wynik Predykcji ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

