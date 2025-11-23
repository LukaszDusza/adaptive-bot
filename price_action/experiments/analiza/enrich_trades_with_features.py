#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt do wzbogacenia danych o pozycjach o wartości wskaźników (features) w momencie entry.

Dla każdej pozycji z closed_pnl.csv:
- Pobiera dane OHLCV z Bybit dla czasu entry
- Oblicza wszystkie features używając data_preparer_pa.py
- Dodaje wartości kluczowych features do CSV

Użycie:
    python enrich_trades_with_features.py DOGEUSDT
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bybit_adapter import BybitAdapter
from data_preparer_pa import fetch_and_prepare_data
from dotenv import load_dotenv


def load_model_info(ticker, version="v1.2"):
    """Load model, features, and metadata."""
    # Find model directory
    models_dir = "/Users/lukasz/projects/adaptive-bot/price_action/models"

    # Try different naming patterns
    possible_names = [
        f"{version}.{ticker.replace('USDT', '').lower()}",
        f"{version}.{ticker.lower()}",
        ticker
    ]

    model_dir = None
    for name in possible_names:
        test_dir = os.path.join(models_dir, name)
        if os.path.exists(test_dir):
            model_dir = test_dir
            break

    if not model_dir:
        raise FileNotFoundError(f"Model directory not found for {ticker}")

    # Find strategy directory (LONG or SHORT)
    strategy_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and 'long' in d.lower()]

    if not strategy_dirs:
        raise FileNotFoundError(f"No LONG strategy found in {model_dir}")

    strategy_dir = os.path.join(model_dir, strategy_dirs[0])

    print(f"✓ Found model directory: {strategy_dir}")

    # Load model components
    model = joblib.load(os.path.join(strategy_dir, "model.joblib"))
    features = joblib.load(os.path.join(strategy_dir, "features.joblib"))

    # Load metadata
    import json
    with open(os.path.join(strategy_dir, "training_metadata.json")) as f:
        metadata = json.load(f)

    return model, features, metadata, strategy_dir


def get_features_for_timestamp(adapter, ticker, timestamp_ms, helper_timeframes, features_list):
    """
    Get feature values at a specific timestamp by fetching OHLCV and calculating features.

    Args:
        adapter: BybitAdapter instance
        ticker: Trading pair
        timestamp_ms: Timestamp in milliseconds
        helper_timeframes: List of helper timeframes
        features_list: List of feature names to calculate

    Returns:
        Dictionary with feature values
    """
    # Convert timestamp to datetime
    dt = datetime.fromtimestamp(timestamp_ms / 1000)

    print(f"  Fetching data for {dt.strftime('%Y-%m-%d %H:%M:%S')}...")

    # Fetch OHLCV data (need historical context for feature calculation)
    # Fetch ~200 candles before timestamp (enough for all indicators)
    try:
        # Use data_preparer to get features (this handles all the complexity)
        # We need to fetch data UP TO the timestamp

        # Calculate how many 15m candles = ~7 days (for ATR, SMA, etc.)
        limit = 1000  # ~10 days of 15m candles

        # Fetch data ending at timestamp
        df = fetch_and_prepare_data(
            ticker=ticker,
            timeframe="15m",
            limit=limit,
            helper_timeframes=helper_timeframes,
            side="long",
            skip_slow_features=False,
            adapter=adapter,
            end_timestamp_ms=timestamp_ms
        )

        if df is None or len(df) == 0:
            print(f"    ❌ No data fetched for {dt}")
            return None

        # Get the last row (closest to timestamp)
        last_row = df.iloc[-1]

        # Extract feature values
        feature_values = {}
        for feature in features_list:
            if feature in last_row.index:
                feature_values[feature] = last_row[feature]
            else:
                feature_values[feature] = np.nan

        # Also add some key market data
        feature_values['close_price'] = last_row.get('close', np.nan)
        feature_values['volume'] = last_row.get('volume', np.nan)

        return feature_values

    except Exception as e:
        print(f"    ❌ Error calculating features: {e}")
        import traceback
        traceback.print_exc()
        return None


def enrich_trades(ticker):
    """Enrich trades with feature values at entry time."""

    print(f"\n{'='*80}")
    print(f"🔧 ENRICHING TRADES WITH FEATURES: {ticker}")
    print(f"{'='*80}\n")

    # Load model info
    print("1. Loading model and features...")
    model, features, metadata, strategy_dir = load_model_info(ticker)

    print(f"   ✓ Model loaded from: {strategy_dir}")
    print(f"   ✓ Features: {len(features)} selected")
    print(f"   ✓ Helper timeframes: {metadata['helper_timeframes']}")

    # Get feature importance
    importance = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Select TOP 20 most important features to add
    top_features = feat_imp_df.head(20)['feature'].tolist()

    print(f"\n   📊 TOP 20 features to add:")
    for i, f in enumerate(top_features, 1):
        imp = feat_imp_df[feat_imp_df['feature'] == f]['importance'].values[0]
        print(f"      {i:2d}. {f:40s} (importance: {imp:.0f})")

    # Load closed PnL data
    print(f"\n2. Loading closed P&L data...")
    analiza_dir = os.path.dirname(os.path.abspath(__file__))
    pnl_file = os.path.join(analiza_dir, f"{ticker}_closed_pnl.csv")

    if not os.path.exists(pnl_file):
        print(f"   ❌ File not found: {pnl_file}")
        return

    pnl_df = pd.read_csv(pnl_file)
    print(f"   ✓ Loaded {len(pnl_df)} trades")

    # Initialize Bybit adapter
    print(f"\n3. Initializing Bybit adapter...")
    load_dotenv(os.path.join(os.path.dirname(strategy_dir), '../../.env_luk'))

    adapter = BybitAdapter(
        api_key=os.getenv('BYBIT_API_KEY'),
        api_secret=os.getenv('BYBIT_API_SECRET'),
        base_url=os.getenv('BYBIT_BASE_URL', 'https://api.bybit.com'),
        category="linear"
    )
    print(f"   ✓ Connected to Bybit API")

    # Process each trade
    print(f"\n4. Calculating features for each trade...")
    print(f"   (This will take ~{len(pnl_df) * 2} seconds due to API rate limits)")

    # Create columns for features
    for feature in top_features:
        pnl_df[f'feat_{feature}'] = np.nan

    # Add close_price and volume
    pnl_df['entry_close_price'] = np.nan
    pnl_df['entry_volume'] = np.nan

    # Process each trade
    for idx, row in pnl_df.iterrows():
        trade_num = idx + 1
        timestamp_ms = int(row['createdTime'])

        print(f"\n   [{trade_num}/{len(pnl_df)}] Trade at {row['createdTime_readable']}:")

        # Get features
        feature_values = get_features_for_timestamp(
            adapter=adapter,
            ticker=ticker,
            timestamp_ms=timestamp_ms,
            helper_timeframes=metadata['helper_timeframes'],
            features_list=top_features
        )

        if feature_values is None:
            print(f"      ⚠️  Skipping (no data)")
            continue

        # Add feature values to dataframe
        for feature in top_features:
            pnl_df.at[idx, f'feat_{feature}'] = feature_values.get(feature, np.nan)

        pnl_df.at[idx, 'entry_close_price'] = feature_values.get('close_price', np.nan)
        pnl_df.at[idx, 'entry_volume'] = feature_values.get('volume', np.nan)

        # Show key values
        atr_val = feature_values.get('atr_normalized', np.nan)
        rsi_slope = feature_values.get('rsi_slope_6h', np.nan)

        print(f"      ✓ Features calculated:")
        print(f"         ATR (normalized): {atr_val:.6f}" if not np.isnan(atr_val) else "         ATR: N/A")
        print(f"         RSI Slope 6h: {rsi_slope:.6f}" if not np.isnan(rsi_slope) else "         RSI Slope 6h: N/A")

    # Save enriched data
    output_file = os.path.join(analiza_dir, f"{ticker}_closed_pnl_enriched.csv")
    pnl_df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"✅ SUCCESS - Enriched data saved to:")
    print(f"   {output_file}")
    print(f"{'='*80}\n")

    # Quick analysis
    print(f"📊 QUICK ANALYSIS:")

    # Correlation between ATR and PnL
    if 'feat_atr_normalized' in pnl_df.columns:
        pnl_numeric = pd.to_numeric(pnl_df['closedPnl'], errors='coerce')
        atr_numeric = pd.to_numeric(pnl_df['feat_atr_normalized'], errors='coerce')

        correlation = pnl_numeric.corr(atr_numeric)
        print(f"\n   Correlation ATR vs PnL: {correlation:.4f}")

        if correlation < -0.2:
            print(f"   ⚠️  NEGATIVE correlation - wysokie ATR = większe straty!")

        # Group by ATR quartiles
        pnl_df['atr_quartile'] = pd.qcut(atr_numeric, q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')

        print(f"\n   PnL by ATR quartile:")
        quartile_stats = pnl_df.groupby('atr_quartile').agg({
            'closedPnl': ['count', 'sum', 'mean']
        })
        print(quartile_stats)

    return output_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python enrich_trades_with_features.py TICKER")
        print("Example: python enrich_trades_with_features.py DOGEUSDT")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    enrich_trades(ticker)


if __name__ == "__main__":
    main()
