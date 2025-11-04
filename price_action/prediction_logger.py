#!/usr/bin/env python3
"""
Prediction Logger - zapisuje predykcje modelu wraz z OHLCV do CSV
"""

import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd


class PredictionLogger:
    """
    Logger który zapisuje każdą predykcję modelu wraz z danymi świecowymi.
    Format: timestamp, open, high, low, close, volume, buy_prob, sell_prob,
            max_prob, threshold, decision, proba_diff
    """

    def __init__(self,
                 strategy_id: str,
                 log_base_dir: str = "/app/prediction_logs",
                 container_name: Optional[str] = None):
        """
        Args:
            strategy_id: ID strategii (np. SOLUSDT_15m_plus_1h_4h_1D)
            log_base_dir: Bazowy katalog dla logów (montowany jako volume)
            container_name: Nazwa kontenera (np. syl-sol-dynamic-tp)
        """
        self.strategy_id = strategy_id
        self.log_base_dir = Path(log_base_dir)

        # Nazwa kontenera z env variable lub argument
        self.container_name = container_name or os.getenv('CONTAINER_NAME', 'unknown')

        # Timestamp startu kontenera
        self.start_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Katalog dla tego run: logs/{container_name}_{timestamp}/
        self.run_dir = self.log_base_dir / f"{self.container_name}_{self.start_timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Ścieżka do pliku CSV z predykcjami
        self.predictions_file = self.run_dir / f"predictions_{strategy_id}.csv"

        # Ścieżka do pliku z metadanymi
        self.metadata_file = self.run_dir / "metadata.json"

        # Inicjalizuj pliki
        self._init_files()

        print(f"✅ PredictionLogger initialized:")
        print(f"   Log directory: {self.run_dir}")
        print(f"   Predictions file: {self.predictions_file}")

    def _init_files(self):
        """Inicjalizuje pliki CSV z nagłówkami"""

        # CSV z predykcjami
        if not self.predictions_file.exists():
            with open(self.predictions_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',           # Timestamp predykcji
                    'candle_time',        # Timestamp świecy (close time)
                    'open',               # OHLCV
                    'high',
                    'low',
                    'close',
                    'volume',
                    'buy_prob',           # Prawdopodobieństwo BUY
                    'sell_prob',          # Prawdopodobieństwo SELL
                    'max_prob',           # max(buy_prob, sell_prob)
                    'threshold',          # Threshold używany
                    'proba_diff',         # abs(buy_prob - sell_prob)
                    'min_confidence_ratio',  # Minimalny wymagany confidence ratio
                    'decision',           # BUY / SELL / HOLD
                    'above_threshold',    # True/False - czy przekroczył threshold
                    'meets_criteria'      # True/False - threshold + proba_diff
                ])

        # Metadata JSON
        if not self.metadata_file.exists():
            import json
            metadata = {
                'container_name': self.container_name,
                'strategy_id': self.strategy_id,
                'start_time': datetime.now().isoformat(),
                'log_directory': str(self.run_dir),
                'predictions_file': str(self.predictions_file)
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

    def log_prediction(self,
                      candle: pd.Series,
                      buy_prob: float,
                      sell_prob: float,
                      threshold: float,
                      min_confidence_ratio: float,
                      decision: str) -> None:
        """
        Loguje pojedynczą predykcję do CSV

        Args:
            candle: Seria pandas z OHLCV (z kolumnami: open, high, low, close, volume)
            buy_prob: Prawdopodobieństwo BUY
            sell_prob: Prawdopodobieństwo SELL
            threshold: Threshold używany do decyzji
            min_confidence_ratio: Minimalny wymagany confidence ratio
            decision: Decyzja modelu (BUY/SELL/HOLD)
        """

        timestamp = datetime.now().isoformat()

        # Wyciągnij timestamp świecy (name to index)
        candle_time = candle.name if hasattr(candle, 'name') else timestamp

        # Oblicz metryki
        max_prob = max(buy_prob, sell_prob)
        proba_diff = abs(buy_prob - sell_prob)
        confidence_ratio = max_prob / min(buy_prob, sell_prob) if min(buy_prob, sell_prob) > 0 else float('inf')
        above_threshold = max_prob >= threshold
        meets_criteria = above_threshold and confidence_ratio >= min_confidence_ratio

        # Zapisz do CSV
        with open(self.predictions_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                candle_time,
                candle['open'],
                candle['high'],
                candle['low'],
                candle['close'],
                candle['volume'],
                round(buy_prob, 4),
                round(sell_prob, 4),
                round(max_prob, 4),
                round(threshold, 4),
                round(proba_diff, 4),
                round(min_confidence_ratio, 4),
                decision,
                above_threshold,
                meets_criteria
            ])

    def get_predictions_dataframe(self) -> pd.DataFrame:
        """Wczytuje zapisane predykcje jako DataFrame"""
        if self.predictions_file.exists():
            df = pd.read_csv(self.predictions_file, parse_dates=['timestamp', 'candle_time'])
            return df
        return pd.DataFrame()

    def get_log_summary(self) -> dict:
        """Zwraca podsumowanie logowanych predykcji"""
        df = self.get_predictions_dataframe()

        if len(df) == 0:
            return {
                'total_predictions': 0,
                'log_file': str(self.predictions_file)
            }

        summary = {
            'total_predictions': len(df),
            'predictions_above_threshold': df['above_threshold'].sum(),
            'predictions_meeting_criteria': df['meets_criteria'].sum(),
            'buy_decisions': (df['decision'] == 'BUY').sum(),
            'sell_decisions': (df['decision'] == 'SELL').sum(),
            'hold_decisions': (df['decision'] == 'HOLD').sum(),
            'avg_buy_prob': df['buy_prob'].mean(),
            'avg_sell_prob': df['sell_prob'].mean(),
            'avg_max_prob': df['max_prob'].mean(),
            'max_buy_prob': df['buy_prob'].max(),
            'max_sell_prob': df['sell_prob'].max(),
            'log_file': str(self.predictions_file),
            'log_directory': str(self.run_dir),
            'time_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            }
        }

        return summary

    def print_summary(self):
        """Wyświetla podsumowanie logów"""
        summary = self.get_log_summary()

        print("\n" + "="*70)
        print("📊 PREDICTION LOGGER SUMMARY")
        print("="*70)
        print(f"Log directory: {summary.get('log_directory', 'N/A')}")
        print(f"Total predictions: {summary.get('total_predictions', 0)}")

        if summary.get('total_predictions', 0) > 0:
            print(f"\n🎯 Decisions:")
            print(f"   BUY:  {summary['buy_decisions']}")
            print(f"   SELL: {summary['sell_decisions']}")
            print(f"   HOLD: {summary['hold_decisions']}")

            print(f"\n📈 Probabilities:")
            print(f"   Avg BUY:  {summary['avg_buy_prob']:.3f} (max: {summary['max_buy_prob']:.3f})")
            print(f"   Avg SELL: {summary['avg_sell_prob']:.3f} (max: {summary['max_sell_prob']:.3f})")
            print(f"   Avg MAX:  {summary['avg_max_prob']:.3f}")

            print(f"\n🚦 Threshold Analysis:")
            print(f"   Above threshold: {summary['predictions_above_threshold']}/{summary['total_predictions']} "
                  f"({summary['predictions_above_threshold']/summary['total_predictions']*100:.1f}%)")
            print(f"   Meeting criteria: {summary['predictions_meeting_criteria']}/{summary['total_predictions']} "
                  f"({summary['predictions_meeting_criteria']/summary['total_predictions']*100:.1f}%)")

            print(f"\n📅 Time Range:")
            print(f"   Start: {summary['time_range']['start']}")
            print(f"   End:   {summary['time_range']['end']}")

        print("="*70 + "\n")


# Przykładowe użycie
if __name__ == '__main__':
    # Test
    logger = PredictionLogger(
        strategy_id='SOLUSDT_15m_plus_1h_4h_1D',
        container_name='test-bot'
    )

    # Przykładowa świeca
    test_candle = pd.Series({
        'open': 150.5,
        'high': 151.2,
        'low': 150.0,
        'close': 150.8,
        'volume': 1000000
    })

    # Loguj predykcję
    logger.log_prediction(
        candle=test_candle,
        buy_prob=0.45,
        sell_prob=0.52,
        threshold=0.54,
        min_confidence_ratio=1.5,
        decision='HOLD'
    )

    # Podsumowanie
    logger.print_summary()
