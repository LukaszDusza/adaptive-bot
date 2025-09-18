import pandas as pd
import pandas_ta as ta
import joblib
import json

# Funkcja `load_and_prepare_timeframe` pozostaje bez zmian
def load_and_prepare_timeframe(df: pd.DataFrame, interval_suffix: str) -> pd.DataFrame:
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

# Funkcja `create_live_features` pozostaje bez zmian
def create_live_features(recent_history_1h, recent_history_4h, recent_history_1d):
    df_1h = pd.DataFrame(recent_history_1h)
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
    df_4h = pd.DataFrame(recent_history_4h)
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'])
    df_1d = pd.DataFrame(recent_history_1d)
    df_1d['timestamp'] = pd.to_datetime(df_1d['timestamp'])
    features_1h = load_and_prepare_timeframe(df_1h.copy(), '1h')
    features_4h = load_and_prepare_timeframe(df_4h.copy(), '4h')
    features_1d = load_and_prepare_timeframe(df_1d.copy(), '1d')
    current_features_1h = features_1h.iloc[[-1]]
    current_features_4h = features_4h.iloc[[-1]]
    current_features_1d = features_1d.iloc[[-1]]
    final_feature_vector = pd.concat([
        current_features_1h.reset_index(drop=True),
        current_features_4h.reset_index(drop=True),
        current_features_1d.reset_index(drop=True)
    ], axis=1)
    return final_feature_vector

# --- KLUCZOWA ZMIANA: FUNKCJA PRZYJMUJE STRATEGIĘ ---
def get_prediction(feature_vector, strategy: str):
    """
    Wczytuje model, skaler i listę cech dla podanej strategii, a następnie zwraca predykcję.
    """
    model_filename = f'trading_model_{strategy}.joblib'
    scaler_filename = f'scaler_{strategy}.joblib'
    features_filename = f'feature_names_{strategy}.json'

    try:
        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)
        with open(features_filename, 'r') as f:
            expected_features = json.load(f)
    except FileNotFoundError as e:
        return {"error": f"Nie znaleziono pliku: {e.filename}. Upewnij się, że model dla strategii '{strategy}' jest wytrenowany."}

    try:
        feature_vector = feature_vector[expected_features]
    except KeyError as e:
        return {"error": f"Brakująca kolumna w danych wejściowych: {e}."}

    scaled_features = scaler.transform(feature_vector)
    prediction_mapped = model.predict(scaled_features)[0]

    decision_map_long = {-1: "STRATA/SL (LONG)", 0: "NEUTRALNIE", 1: "ZYSK/TP (LONG)"}
    decision_map_short = {-1: "STRATA/SL (SHORT)", 0: "NEUTRALNIE", 1: "ZYSK/TP (SHORT)"}
    decision_map = decision_map_long if strategy == 'long' else decision_map_short

    y_map = {0: -1, 1: 0, 2: 1}
    prediction_class = y_map[prediction_mapped]

    return {"prediction_code": int(prediction_class), "decision": decision_map[prediction_class], "strategy": strategy}

# Funkcja main nie jest już potrzebna, ponieważ logikę testową przenosimy do serwera API
# Możesz ją zostawić do celów debugowania, ale nie jest kluczowa.