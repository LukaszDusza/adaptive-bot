# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **ML-powered cryptocurrency trading bot** for Bybit that uses LightGBM models to predict LONG and SHORT opportunities on crypto pairs. The bot operates in real-time on live markets with features like trailing stop-loss (TSL), partial/dynamic take-profit (TP), and advanced trade logging.

**Key characteristics:**
- Dual-model approach: separate models for LONG and SHORT predictions
- Triple-barrier labeling method for training data
- Feature engineering with 30+ ICT (Inner Circle Trader) and Smart Money concepts
- Walk-forward validation with time-series cross-validation
- Multi-objective Optuna optimization for hyperparameters
- Dockerized deployment with hedge mode support

## Architecture

### Core Pipeline Flow

```
1. Data Preparation (data_preparer_pa.py)
   ↓ Fetch OHLCV from Bybit + calculate 100+ features

2. Training Pipeline (model_pipeline.py)
   ↓ Optimize labels → Feature selection → Train LightGBM → Threshold tuning

3. Backtesting (backtester.py)
   ↓ Test strategy on historical data with realistic slippage/fees

4. Live Trading (bot.py)
   ↓ Run ML predictions every cycle → Manage positions with TSL/TP
```

### Key Components

**Main Entry Point:**
- `main.py` - CLI interface with 4 modes: `--train`, `--backtest`, `--report`, `--run-bot`

**ML Training:**
- `model_pipeline.py` - Two-stage Optuna optimization:
  1. Label parameters (profit-take, stop-loss, time-limit for triple-barrier)
  2. Model hyperparameters (LightGBM with SMOTE oversampling)
- `data_preparer_pa.py` - Feature engineering (Price Action + ICT indicators)
- Models saved to: `models/{version}/{ticker}_{timeframe}_plus_{helpers}_{side}/`

**Trading Bot:**
- `bot.py` - Main trading loop with `TradingBot` class
- `bybit_adapter.py` - Wrapper around pybit for Bybit API (linear perpetuals)
- `trade_logger.py` - Advanced logging system (JSON trades, Parquet candles, indicators)

**Analysis Tools:**
- `backtester.py` - Simulation engine with Position/Trade dataclasses
- `report_generator.py` - HTML reports with equity curves and trade statistics
- `optuna_optimizer.py` - Multi-objective optimization for live parameters
- `analyze_trades.py` - Post-trade analysis from logged data

### Data Flow

**Training Phase:**
```
fetch_and_prepare_data()
  → Triple-barrier labeling (get_triple_barrier_labels)
  → Feature selection (feature importance threshold=0.85)
  → Walk-forward CV (3 splits, expanding window)
  → SMOTE resampling for class imbalance
  → Final model + scaler + feature list saved
  → Automatic analysis.py execution
```

**Live Bot Phase:**
```
get_decision() - Fetch latest candles (ONLY closed candles, exclude last forming candle)
  → Calculate all features for last CLOSED candle
  → Run prediction with both models (LONG + SHORT)
  → Check: proba > threshold AND proba_diff > min_proba_diff
  → Return: BUY / SELL / HOLD

_manage_position() - If position exists:
  → Update TSL (only when in profit)
  → Handle partial/dynamic TP levels
  → Log candles + events to trade_logger

_open_position() - If signal:
  → Market order + set SL/TP
  → Log entry with indicators snapshot
  → Save state to JSON (survives restarts)
```

## Common Development Tasks

### Training a New Model

```bash
# Navigate to price_action directory
cd price_action

# Interactive workflow script (recommended)
./run_solusdt_workflow.sh

# Or use main.py directly:
python main.py --train --side long \
  --ticker SOLUSDT --timeframe 15m \
  --helper-timeframes 1h 4h \
  --version v1.0 \
  --label-trials 100 --model-trials 100

# Train SHORT model:
python main.py --train --side short \
  --ticker SOLUSDT --timeframe 15m \
  --helper-timeframes 1h 4h \
  --version v1.0 \
  --label-trials 100 --model-trials 100
```

**Training produces:**
- `models/{version}/{strategy_id}/model.joblib` - LightGBM classifier
- `models/{version}/{strategy_id}/scaler.joblib` - StandardScaler
- `models/{version}/{strategy_id}/features.joblib` - Feature list
- `models/{version}/{strategy_id}/label_params.json` - Triple-barrier params
- `models/{version}/{strategy_id}/training_metadata.json` - Full config
- `models/{version}/{strategy_id}/analysis/` - Automatic analysis outputs

### Running Backtest

```bash
python main.py --backtest \
  --ticker SOLUSDT --timeframe 15m \
  --helper-timeframes 1h 4h \
  --version v1.0 \
  --prob-threshold 0.7 \
  --min-proba-diff 0.4 \
  --tp-pct 0.03 \
  --tsl-pct 0.01 \
  --trade-size 100.0 \
  --partial-tp  # or --dynamic-tp

# Generate HTML report:
python main.py --report \
  --ticker SOLUSDT --timeframe 15m \
  --helper-timeframes 1h 4h \
  --version v1.0
```

### Parameter Optimization with Optuna

```bash
# Multi-objective optimization (PnL, Drawdown, Trade Count)
python optuna_optimizer.py \
  --ticker SOLUSDT --timeframe 15m \
  --helper-timeframes 1h 4h \
  --trials 100 \
  --partial-tp

# Results saved to: models/{version}/optuna/optimization_results_*.txt
# Includes: PROB_THRESHOLD, MIN_PROBA_DIFF, TP_PCT, TSL_PCT
```

### Running Live Bot

```bash
# Standalone:
python main.py --run-bot \
  --ticker SOLUSDT --timeframe 15m \
  --helper-timeframes 1h 4h \
  --version v1.0 \
  --prob-threshold 0.7 \
  --min-proba-diff 0.4 \
  --tp-pct 0.03 \
  --tsl-pct 0.01 \
  --trade-size 100.0 \
  --leverage 10 \
  --partial-tp  # or --dynamic-tp
  --hedge-mode  # optional: enables LONG=positionIdx:1, SHORT=positionIdx:2

# Docker (recommended):
cd price_action
docker compose up --build -d

# View logs:
docker compose logs -f bot-syl-sol-dynamic-tp
```

**Bot state management:**
- State persisted to `bot_state/{strategy_id}_state.json`
- Survives restarts (auto-restores trade logging if position exists)
- Logs saved to `logs/trades/`, `logs/candles/`, `logs/indicators/`

### Analyzing Logged Trades

```bash
python analyze_trades.py
# Interactive tool to load and analyze trade JSONs from logs/trades/
```

## Model Architecture Details

### Feature Engineering (data_preparer_pa.py)

**5 Tiers of Features:**
1. **TIER 0:** OHLCV (open, high, low, close, volume, turnover)
2. **TIER 1:** Basic TA (RSI, SMA, VWAP, ATR, Bollinger Bands, candlestick patterns)
3. **TIER 2:** Advanced PA (Order Flow proxies, Volume Profile, Market Regime)
4. **TIER 3:** Composite indicators (MSI, Momentum Regime, Volume Confirmation)
5. **TIER 4:** ICT & Smart Money (30+ features):
   - Fair Value Gaps (FVG)
   - Liquidity Sweeps
   - Order Blocks
   - Breaker Blocks
   - Market Structure Shifts (MSS)
   - **`ict_composite_score`** - Master ICT score (highest importance)

**Multi-timeframe support:**
- Main timeframe (e.g., 15m) + helper timeframes (e.g., 1h, 4h)
- Helper features prefixed with `{timeframe}_` (e.g., `1h_rsi_14`)

### Training Optimizations (model_pipeline.py)

**Recent improvements (see code comments):**
- **POPRAWKA #1:** Recall-focused optimization for LONG (60% recall + 40% precision)
- **POPRAWKA #4:** Threshold tuning targets 55% recall (was 70%)
- **POPRAWKA #5:** Increased regularization (reg_alpha/lambda: 10-100)
- **Speed optimizations:**
  - Reduced CV splits from 6 to 3 (2x faster)
  - Added MedianPruner for early stopping of weak trials
  - Parallel trial execution (`n_jobs=-1`)

### Backtester Architecture (backtester.py)

**No look-ahead bias:**
- Uses ONLY closed candles (excludes last forming candle with `df.iloc[:-1]`)
- Realistic slippage (default: 0.01%)
- Maker/taker fees (default: 0.02%/0.055%)

**Position management:**
- Tracks MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion)
- Supports both partial TP (50% at halfway) and dynamic TP (25% at 4 levels)
- TSL only updates when in profit (prevents premature exits)

## Important Configuration

### Model Versioning

Models are organized by version string (e.g., `v1.0`, `v1.1`):
```
models/
  v1.0/
    SOLUSDT_15m_plus_1h_4h_long/
      model.joblib
      scaler.joblib
      features.joblib
      label_params.json
      training_metadata.json
      analysis/
    SOLUSDT_15m_plus_1h_4h_short/
      ...
  optuna/
    {strategy_id}_labels_study.db
    {strategy_id}_model_study.db
    optimization_results_*.txt
```

### Environment Variables (.env files)

Required for live trading:
```bash
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_BASE_URL=https://api.bybit.com  # or demo URL
```

**Multiple accounts:**
- `.env_luk` - Account 1
- `.env_sylwia` - Account 2
- Specified in `docker-compose.yaml` via `env_file:`

### Hedge Mode vs One-Way Mode

**One-Way Mode (default):**
- positionIdx: 0
- Can only have LONG or SHORT, not both simultaneously

**Hedge Mode (`--hedge-mode`):**
- positionIdx: 1 for LONG, 2 for SHORT
- Can hold both LONG and SHORT positions simultaneously
- Must be enabled in Bybit account settings

## Critical Code Patterns

### Always Exclude Last Candle for Predictions

```python
# CORRECT (bot.py:218):
df_closed = df.iloc[:-1]  # Exclude last forming candle
last_row = df_closed.iloc[-1]  # Use last CLOSED candle

# WRONG:
last_row = df.iloc[-1]  # This is still forming!
```

### Position Management with positionIdx

```python
# CORRECT - Use position side for SL/TP in hedge mode:
position_side = "Buy"  # for LONG
self.adapter.set_stop_loss(ticker, sl, position_side)

# positionIdx is internally resolved by adapter based on hedge_mode flag
```

### State Persistence Pattern

```python
# Bot saves state to JSON:
self.state = {
    'side': 'Long',
    'entry_price': 123.45,
    'initial_tp': 126.78,
    'last_sl': 120.00,
    'partial_tp_taken': False,
    'dynamic_tp_levels_taken': 0,
    'highest_price': 125.00,
    'lowest_price': 123.00
}
self._save_state()

# On restart, _restore_trade_logging_if_needed() checks state
# and resumes trade logger to prevent "no active trade" warnings
```

## Testing and Validation

### Before Deploying to Live

1. **Train models** with sufficient data (default: 138,240 candles for 15m = ~4 years)
2. **Run backtest** on recent 3 months (8,640 candles)
3. **Optimize parameters** with Optuna (100+ trials)
4. **Verify metrics:**
   - Total PnL > 0
   - Win rate > 45%
   - Max drawdown < 20%
   - Sufficient trades (not just 1-2 lucky trades)
5. **Test on paper trading** (Bybit demo account)
6. **Start with small position sizes** on live

### Key Files to Check After Training

- `models/{version}/{strategy_id}/holdout_predictions.csv` - Out-of-sample performance
- `models/{version}/{strategy_id}/analysis/*.png` - Feature importance, confusion matrix
- `models/{version}/{strategy_id}/training_metadata.json` - Verify params

## Workflow Script (run_solusdt_workflow.sh)

Interactive menu with options:
1. Train LONG model
2. Train SHORT model
3. Train BOTH models
4-5. Run Analysis (LONG/SHORT)
6. Run Backtest
7. Generate Report
8. Complete Workflow (all of the above)
9. Analyze Logged Trades
10. Optimize Parameters (Optuna)
11. **Optimize & Auto-Deploy** - Full pipeline: Optuna → Update docker-compose.yaml → Restart containers

**Auto-deploy feature (#11):**
- Runs optimization
- Extracts best params from results file
- Updates `docker-compose.yaml` with new values
- Restarts Docker containers with optimized parameters
- Creates backup before modifying files

## Docker Deployment

### docker-compose.yaml Structure

Defines multiple bot instances (one per account):
```yaml
services:
  bot-syl-sol-dynamic-tp:
    container_name: syl-sol-dynamic-tp
    env_file: .env_sylwia
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./bot_state:/app/bot_state
    command: >
      python main.py --run-bot
      --version v.1.0
      --ticker SOLUSDT
      --timeframe 15m
      --helper-timeframes 1h 4h
      --prob-threshold 0.7
      --min-proba-diff 0.4
      --tsl-pct 0.01
      --tp-pct 0.03
      --trade-size 100.0
      --leverage 10
      --partial-tp
```

### Managing Containers

```bash
# Build and start all bots:
docker compose up --build -d

# Stop all:
docker compose down

# View logs:
docker compose logs -f bot-syl-sol-dynamic-tp

# Restart single bot:
docker compose restart bot-syl-sol-dynamic-tp

# Update after code changes:
docker compose down && docker compose up --build -d
```

## Important Gotchas

1. **Candle timing:** Bot fetches data every cycle, but ONLY uses closed candles for predictions (last candle excluded)
2. **TSL only updates in profit:** `_update_tsl()` checks if position is profitable before adjusting SL
3. **Partial vs Dynamic TP:** Mutually exclusive flags - cannot enable both simultaneously
4. **Hedge mode requires Bybit config:** Must enable "Hedge Mode" in Bybit account settings before using `--hedge-mode`
5. **Model features must match:** When backtesting/live trading, ensure `--helper-timeframes` matches training
6. **Optuna studies are persistent:** Stored in SQLite DBs (`models/{version}/optuna/*.db`), can resume optimization
7. **Trade logger state:** If bot restarts mid-trade, `_restore_trade_logging_if_needed()` prevents logging errors
8. **Feature correlation removal:** `remove_correlated_features()` preserves ICT features (threshold: 0.95)

## Git Workflow

Current branch: `nowe-cechy-z-claude`

Recent commits focused on:
- Dynamic TP implementation
- Breakeven SL after partial TP
- Hedge mode support
- Fix for dropping last incomplete candle

**When committing:**
- Use Polish commit messages (existing pattern)
- Include descriptive messages (e.g., "dodany dynamic tp", "sl przestawia sie na be jesli osiagniety partial tp")
