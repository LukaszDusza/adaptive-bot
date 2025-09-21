# services/analysis_service.py
import joblib
import json
import pandas as pd
import numpy as np
import sys

class AnalysisService:
    def __init__(self, ticker_name_for_models):
        self.models, self.features, self.scaler_pa = {}, {}, None
        self._load_ml_artifacts(ticker_name_for_models)
        print("Analysis Service zainicjalizowany z modelami.")

    def _load_ml_artifacts(self, ticker_name):
        try:
            for expert in ['momentum', 'reversion', 'pa']:
                self.models[expert] = joblib.load(f'expert_{expert}_{ticker_name}_5m.joblib')
                with open(f'features_{expert}_{ticker_name}_5m.json', 'r') as f: self.features[expert] = json.load(f)
            self.scaler_pa = joblib.load(f'scaler_pa_{ticker_name}_5m.joblib')
        except FileNotFoundError as e:
            sys.exit(f"Nie znaleziono plików modelu: {e.filename}")

    def get_analysis_from_row(self, data_row: pd.Series) -> dict:
        expert_opinions = {}
        for expert in ['momentum', 'reversion', 'pa']:
            X_df = pd.DataFrame([data_row[self.features[expert]]])
            if expert == 'pa':
                X_df.replace([np.inf, -np.inf], 0, inplace=True); X_df.fillna(0, inplace=True)
                X = self.scaler_pa.transform(X_df)
            else: X = X_df
            prediction = int(self.models[expert].predict(X)[0])
            confidence = float(self.models[expert].predict_proba(X).max())
            expert_opinions[expert] = {"prediction": prediction, "confidence": confidence}
        return {"current_price": float(data_row['close']), "atr_value_5m": float(data_row['ATRr_14_5m']),
                "expert_opinions": expert_opinions}