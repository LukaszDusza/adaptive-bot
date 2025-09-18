from flask import Flask, request, jsonify
from prediction_service import create_live_features, get_prediction
import logging

# --- KONFIGURACJA LOGOWANIA ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

def handle_prediction_request(strategy: str):
    """Wspólna logika dla obu endpointów."""
    try:
        # Pobierz dane JSON z zapytania
        data = request.get_json()

        # --- LOGOWANIE ŻĄDANIA ---
        logging.info(f"Otrzymano zapytanie dla strategii '{strategy}'.")
        if data:
            logging.info(f"Klucze w body: {list(data.keys())}")
            logging.info(f"Rozmiary danych: 1h -> {len(data.get('recent_1h', []))} świec, "
                         f"4h -> {len(data.get('recent_4h', []))} świec, "
                         f"1d -> {len(data.get('recent_1d', []))} świec.")
        else:
            logging.warning("Otrzymano puste body w zapytaniu.")
            return jsonify({"error": "Puste body zapytania."}), 400

        if 'recent_1h' not in data or 'recent_4h' not in data or 'recent_1d' not in data:
            return jsonify({"error": "Brakujące dane. Oczekiwano kluczy: recent_1h, recent_4h, recent_1d."}), 400

        feature_vector = create_live_features(
            data['recent_1h'],
            data['recent_4h'],
            data['recent_1d']
        )

        result = get_prediction(feature_vector, strategy)

        # --- LOGOWANIE ODPOWIEDZI ---
        logging.info(f"Zwracam odpowiedź dla strategii '{strategy}': {result}")

        return jsonify(result)

    except Exception as e:
        # --- LOGOWANIE BŁĘDU ---
        logging.error(f"Wystąpił wewnętrzny błąd serwera podczas przetwarzania strategii '{strategy}': {str(e)}", exc_info=True)
        return jsonify({"error": f"Wystąpił wewnętrzny błąd serwera: {str(e)}"}), 500

@app.route('/predict/long', methods=['POST'])
def predict_long():
    """Endpoint API dla predykcji strategii LONG."""
    return handle_prediction_request('long')

@app.route('/predict/short', methods=['POST'])
def predict_short():
    """Endpoint API dla predykcji strategii SHORT."""
    return handle_prediction_request('short')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)