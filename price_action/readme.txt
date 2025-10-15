# ========================================
# SOLUSDT COMPLETE WORKFLOW
# ========================================
# This is a ready-to-use set of commands for training, analyzing, and backtesting SOLUSDT models

## ========================================
## 1. TRAINING (LONG & SHORT MODELS)
## ========================================

# Train LONG model for SOLUSDT 1h with helper timeframes
python main.py \
--train \
--side long \
--ticker "SOLUSDT" \
--timeframe "1h" \
--helper-timeframes 2h 4h 6h 12h 1D \
--limit 25000 \
--label-trials 200 \
--model-trials 200

# Train SHORT model for SOLUSDT 1h with helper timeframes
python main.py \
--train \
--side short \
--ticker "SOLUSDT" \
--timeframe "1h" \
--helper-timeframes 2h 4h 6h 12h 1D \
--limit 25000 \
--label-trials 200 \
--model-trials 200

# Note: Training automatically runs analysis and saves results to model folder


## ========================================
## 2. ANALYSIS (if needed separately)
## ========================================

# Analyze LONG model
python analysis.py \
--ticker "SOLUSDT" \
--timeframe 1h \
--side long \
--helper-timeframes 2h 4h 6h 12h 1D

# Analyze SHORT model
python analysis.py \
--ticker "SOLUSDT" \
--timeframe 1h \
--side short \
--helper-timeframes 2h 4h 6h 12h 1D


## ========================================
## 3. BACKTESTING
## ========================================

# Backtest combined LONG/SHORT strategy
python main.py \
--backtest \
--ticker "ETHUSDT" \
--timeframe "1h" \
--helper-timeframes 2h 4h 6h 12h 1D \
--limit 3000 \
--prob-threshold 0.8 \
--tp-pct 0.06 \
--tsl-pct 0.03 \
--trade-size 1000 \
--partial-tp

# Backtest with different parameters (more conservative)
python main.py \
--backtest \
--ticker "SOLUSDT" \
--timeframe "1h" \
--helper-timeframes 2h 4h 6h 12h 1D \
--limit 3000 \
--prob-threshold 0.85 \
--tp-pct 0.06 \
--tsl-pct 0.03 \
--trade-size 1000 \
--partial-tp


## ========================================
## 4. GENERATE REPORT
## ========================================

# Generate HTML report from backtest results
python main.py \
--report \
--ticker SOLUSDT \
--timeframe 1h \
--helper-timeframes 2h 4h 6h 12h 1D


## ========================================
## 5. RUN LIVE BOT (PAPER TRADING)
## ========================================

# Run the bot with paper trading (testnet)
python main.py \
--run-bot \
--ticker "SOLUSDT" \
--timeframe "1h" \
--helper-timeframes 2h 4h 6h 12h 1D \
--prob-threshold 0.8 \
--tp-pct 0.07 \
--tsl-pct 0.04 \
--trade-size 100 \
--leverage 10 \
--partial-tp


## ========================================
## 6. ANALYZE LOGGED TRADES
## ========================================

# Interactive analysis of logged trades from the bot
python analyze_trades.py


## ========================================
## PARAMETER EXPLANATIONS
## ========================================

# --ticker: Trading pair (e.g., SOLUSDT, BTCUSDT)
# --timeframe: Main timeframe (1m, 5m, 15m, 1h, 4h, etc.)
# --helper-timeframes: Additional timeframes for context
# --limit: Number of candles to fetch (training: 25000+, backtest: 3000+)
# --label-trials: Optuna trials for label optimization (default: 50, recommended: 200)
# --model-trials: Optuna trials for model optimization (default: 100, recommended: 200)
# --prob-threshold: Minimum probability to enter trade (0.7-0.9)
# --tp-pct: Take profit percentage (0.05-0.10)
# --tsl-pct: Trailing stop loss percentage (0.02-0.05)
# --trade-size: Trade size in USD (paper: 100-1000, live: adjust accordingly)
# --leverage: Leverage multiplier (1-20, be careful!)
# --partial-tp: Enable partial take profit at 50% position


## ========================================
## RECOMMENDED WORKFLOW
## ========================================

# STEP 1: Train both LONG and SHORT models (takes 2-4 hours each)
#         Analysis runs automatically after training

# STEP 2: Check analysis results in model folders:
#         - SOLUSDT_1h_plus_2h_4h_6h_12h_1D_long/
#         - SOLUSDT_1h_plus_2h_4h_6h_12h_1D_short/

# STEP 3: Run backtest to see strategy performance

# STEP 4: Generate HTML report for detailed analysis

# STEP 5: If results are good, run bot in paper trading mode

# STEP 6: Use analyze_trades.py to review bot performance


## ========================================
## QUICK START FOR OTHER TIMEFRAMES
## ========================================

# For 15-minute trading (more frequent signals)
python main.py --train --side long --ticker "SOLUSDT" --timeframe "15m" \
--helper-timeframes 1h 2h 4h --limit 30000 --label-trials 200 --model-trials 200

python main.py --train --side short --ticker "SOLUSDT" --timeframe "15m" \
--helper-timeframes 1h 2h 4h --limit 30000 --label-trials 200 --model-trials 200

python main.py --backtest --ticker "SOLUSDT" --timeframe "15m" \
--helper-timeframes 1h 2h 4h --limit 5000 --prob-threshold 0.8 \
--tp-pct 0.05 --tsl-pct 0.025 --trade-size 1000 --partial-tp


# For 4-hour trading (less frequent, larger moves)
python main.py --train --side long --ticker "SOLUSDT" --timeframe "4h" \
--helper-timeframes 12h 1D --limit 15000 --label-trials 200 --model-trials 200

python main.py --train --side short --ticker "SOLUSDT" --timeframe "4h" \
--helper-timeframes 12h 1D --limit 15000 --label-trials 200 --model-trials 200

python main.py --backtest --ticker "SOLUSDT" --timeframe "4h" \
--helper-timeframes 12h 1D --limit 2000 --prob-threshold 0.8 \
--tp-pct 0.08 --tsl-pct 0.05 --trade-size 1000 --partial-tp


## ========================================
## ARCHIVED COMMANDS (OTHER TICKERS)
## ========================================
# These are kept for reference but not part of the main SOLUSDT workflow

# BNB
# python main.py --train --side long --ticker "BNBUSDT" --timeframe 1h --helper-timeframes 4h 12h 1D --limit 25000 --label-trials 200 --model-trials 200
# python main.py --train --side short --ticker "BNBUSDT" --timeframe 1h --helper-timeframes 4h 12h 1D --limit 25000 --label-trials 200 --model-trials 200

# ETH
# python main.py --train --side long --ticker "ETHUSDT" --timeframe 1h --helper-timeframes 4h 12h 1D --limit 30000 --label-trials 200 --model-trials 200
# python main.py --train --side short --ticker "ETHUSDT" --timeframe 1h --helper-timeframes 4h 12h 1D --limit 30000 --label-trials 200 --model-trials 200

# BTC
# python main.py --train --side long --ticker "BTCUSDT" --timeframe 1h --helper-timeframes 2h 4h --limit 50000 --label-trials 200 --model-trials 200
# python main.py --train --side short --ticker "BTCUSDT" --timeframe 1h --helper-timeframes 2h 4h --limit 50000 --label-trials 200 --model-trials 200
