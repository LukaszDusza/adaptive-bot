from flask import Flask, request, jsonify
import pandas as pd
# Importujemy logikę z naszego poprzedniego skryptu
from prediction_service import create_live_features, get_prediction

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint API, który przyjmuje dane rynkowe i zwraca predykcję modelu.
    """
    try:
        # Pobierz dane JSON z zapytania
        data = request.get_json()

        # Sprawdź, czy dane zawierają wymagane klucze
        if 'recent_1h' not in data or 'recent_4h' not in data or 'recent_1d' not in data:
            return jsonify({"error": "Brakujące dane. Oczekiwano kluczy: recent_1h, recent_4h, recent_1d."}), 400

        # Przekaż dane do naszej logiki tworzenia cech
        feature_vector = create_live_features(
            data['recent_1h'],
            data['recent_4h'],
            data['recent_1d']
        )

        # Uzyskaj predykcję
        result = get_prediction(feature_vector)

        # Zwróć wynik jako JSON
        return jsonify(result)

    except Exception as e:
        # Zwróć błąd, jeśli coś poszło nie tak
        return jsonify({"error": f"Wystąpił wewnętrzny błąd serwera: {str(e)}"}), 500


if __name__ == '__main__':
    # Uruchom serwer na porcie 5000, dostępny z zewnątrz kontenera
    app.run(host='0.0.0.0', port=5000)
