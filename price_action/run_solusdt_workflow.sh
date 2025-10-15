#!/bin/bash

# ========================================
# SOLUSDT Complete Workflow Script
# ========================================
# This script helps you easily train, analyze, and backtest SOLUSDT models

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters for SOLUSDT 1h
TICKER="ETHUSDT"
TIMEFRAME="1h"
HELPER_TIMEFRAMES="2h 4h 6h 12h 1D"
LIMIT_TRAIN=60000
LIMIT_BACKTEST=5000
DATE_FROM="2025-05-31"  # Training data end date (prevents data leakage)
LABEL_TRIALS=100
MODEL_TRIALS=200
PROB_THRESHOLD=0.6
TP_PCT=0.06
TSL_PCT=0.03
TRADE_SIZE=1000
OPTUNA_TRIALS=100  # Number of Optuna optimization trials

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  SOLUSDT ML Trading Workflow${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Current Parameters:${NC}"
echo "  PROB_THRESHOLD: $PROB_THRESHOLD"
echo "  TP_PCT: $TP_PCT"
echo "  TSL_PCT: $TSL_PCT"
echo ""

# Function to display menu
show_menu() {
    echo -e "${GREEN}Select an action:${NC}"
    echo "1) Train LONG model"
    echo "2) Train SHORT model"
    echo "3) Train BOTH models (LONG + SHORT)"
    echo "4) Run Analysis (LONG)"
    echo "5) Run Analysis (SHORT)"
    echo "6) Run Backtest"
    echo "7) Generate Report"
    echo "8) Complete Workflow (Train Both + Backtest + Report)"
    echo "9) Analyze Logged Trades"
    echo "10) Optimize Parameters (Optuna) 🔥"
    echo "0) Exit"
    echo ""
}

# Function to prompt for ticker
prompt_ticker() {
    local skip_prompt=${1:-0}
    
    if [ "$skip_prompt" -eq 0 ]; then
        echo -e "${YELLOW}Enter ticker symbol (default: $TICKER):${NC}"
        read -p "Ticker: " input_ticker
        
        if [[ ! -z "$input_ticker" ]]; then
            TICKER=$(echo "$input_ticker" | tr '[:lower:]' '[:upper:]')
            echo -e "${GREEN}Using ticker: $TICKER${NC}"
            
            # Check if models already exist
            local strategy_id="${TICKER}_${TIMEFRAME}"
            if [ -f "models/${strategy_id}_plus_*_long_model.joblib" ] || [ -f "models/${strategy_id}_plus_*_short_model.joblib" ]; then
                echo -e "${YELLOW}⚠️  Warning: Models for $TICKER already exist and will be replaced!${NC}"
                read -p "Continue? (y/n) " -n 1 -r
                echo ""
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo "Cancelled."
                    return 1
                fi
            fi
        else
            echo -e "${GREEN}Using default ticker: $TICKER${NC}"
        fi
        echo ""
    fi
    return 0
}

# Function to train LONG model
train_long() {
    local skip_prompt=${1:-0}
    
    # Prompt for ticker selection
    if ! prompt_ticker $skip_prompt; then
        return
    fi
    
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Training LONG model for $TICKER ${TIMEFRAME}${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo "This will take approximately 2-4 hours..."
    echo ""
    
    python main.py \
        --train \
        --side long \
        --ticker "$TICKER" \
        --timeframe "$TIMEFRAME" \
        --helper-timeframes $HELPER_TIMEFRAMES \
        --limit $LIMIT_TRAIN \
        --date-from "$DATE_FROM" \
        --label-trials $LABEL_TRIALS \
        --model-trials $MODEL_TRIALS
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ LONG model training completed successfully!${NC}"
        echo -e "${GREEN}  Analysis results saved to: ${TICKER}_${TIMEFRAME}_plus_*_long/${NC}"
    else
        echo -e "${RED}✗ LONG model training failed!${NC}"
        exit 1
    fi
}

# Function to train SHORT model
train_short() {
    local skip_prompt=${1:-0}
    
    # Prompt for ticker selection
    if ! prompt_ticker $skip_prompt; then
        return
    fi
    
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Training SHORT model for $TICKER ${TIMEFRAME}${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo "This will take approximately 2-4 hours..."
    echo ""
    
    python main.py \
        --train \
        --side short \
        --ticker "$TICKER" \
        --timeframe "$TIMEFRAME" \
        --helper-timeframes $HELPER_TIMEFRAMES \
        --limit $LIMIT_TRAIN \
        --date-from "$DATE_FROM" \
        --label-trials $LABEL_TRIALS \
        --model-trials $MODEL_TRIALS
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ SHORT model training completed successfully!${NC}"
        echo -e "${GREEN}  Analysis results saved to: ${TICKER}_${TIMEFRAME}_plus_*_short/${NC}"
    else
        echo -e "${RED}✗ SHORT model training failed!${NC}"
        exit 1
    fi
}

# Function to run analysis
run_analysis() {
    local side=$1
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Running Analysis for $side model${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    python analysis.py \
        --ticker "$TICKER" \
        --timeframe $TIMEFRAME \
        --side $side \
        --helper-timeframes $HELPER_TIMEFRAMES
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Analysis completed successfully!${NC}"
    else
        echo -e "${RED}✗ Analysis failed!${NC}"
    fi
}

# Function to run backtest
run_backtest() {
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Running Backtest for $TICKER ${TIMEFRAME}${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo "Parameters:"
    echo "  - Probability Threshold: $PROB_THRESHOLD"
    echo "  - Take Profit: ${TP_PCT}%"
    echo "  - Trailing Stop Loss: ${TSL_PCT}%"
    echo "  - Trade Size: \$${TRADE_SIZE}"
    echo ""
    
    python main.py \
        --backtest \
        --ticker "$TICKER" \
        --timeframe "$TIMEFRAME" \
        --helper-timeframes $HELPER_TIMEFRAMES \
        --limit $LIMIT_BACKTEST \
        --prob-threshold $PROB_THRESHOLD \
        --tp-pct $TP_PCT \
        --tsl-pct $TSL_PCT \
        --trade-size $TRADE_SIZE \
        --partial-tp
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Backtest completed successfully!${NC}"
        echo -e "${GREEN}  Results saved to: backtests/${NC}"
    else
        echo -e "${RED}✗ Backtest failed!${NC}"
    fi
}

# Function to generate report
generate_report() {
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Generating Report${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    python main.py \
        --report \
        --ticker $TICKER \
        --timeframe $TIMEFRAME \
        --helper-timeframes $HELPER_TIMEFRAMES
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Report generated successfully!${NC}"
        echo -e "${GREEN}  Check the reports/ directory${NC}"
    else
        echo -e "${RED}✗ Report generation failed!${NC}"
    fi
}

# Function to optimize parameters with Optuna
optimize_parameters() {
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Optimizing Parameters with Optuna${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo "This will find the optimal values for:"
    echo "  - PROB_THRESHOLD (probability threshold)"
    echo "  - TP_PCT (take profit percentage)"
    echo "  - TSL_PCT (trailing stop loss percentage)"
    echo ""
    echo "Current parameters:"
    echo "  - PROB_THRESHOLD: $PROB_THRESHOLD"
    echo "  - TP_PCT: $TP_PCT"
    echo "  - TSL_PCT: $TSL_PCT"
    echo ""
    echo "Number of trials: $OPTUNA_TRIALS"
    echo ""
    
    read -p "Enter number of trials (default: $OPTUNA_TRIALS): " input_trials
    if [[ ! -z "$input_trials" ]]; then
        OPTUNA_TRIALS=$input_trials
    fi
    
    echo ""
    echo -e "${YELLOW}Starting optimization with $OPTUNA_TRIALS trials...${NC}"
    echo "This may take 30-90 minutes depending on number of trials."
    echo ""
    
    python optuna_optimizer.py \
        --ticker "$TICKER" \
        --timeframe "$TIMEFRAME" \
        --helper-timeframes $HELPER_TIMEFRAMES \
        --limit $LIMIT_BACKTEST \
        --trials $OPTUNA_TRIALS \
        --partial-tp
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}✓ Optimization completed successfully!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "Results saved to: optuna/optimization_results_${TICKER}_${TIMEFRAME}.txt"
        echo ""
        echo -e "${YELLOW}Next steps:${NC}"
        echo "1. Review the optimization results file"
        echo "2. Update the parameters at the top of this script with optimal values"
        echo "3. Run backtest (option 6) with new parameters"
        echo "4. If satisfied, deploy to live/paper trading"
        echo ""
    else
        echo -e "${RED}✗ Optimization failed!${NC}"
    fi
}

# Function to run complete workflow
complete_workflow() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Starting Complete Workflow${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "This will:"
    echo "1. Train LONG model (~2-4 hours)"
    echo "2. Train SHORT model (~2-4 hours)"
    echo "3. Run backtest"
    echo "4. Generate report"
    echo ""
    echo -e "${YELLOW}Total estimated time: 4-8 hours${NC}"
    echo ""
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        return
    fi
    
    # Prompt for ticker once
    if ! prompt_ticker; then
        return
    fi
    
    # Train LONG
    train_long 1
    echo ""
    
    # Train SHORT
    train_short 1
    echo ""
    
    # Run backtest
    run_backtest
    echo ""
    
    # Generate report
    generate_report
    echo ""
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Complete Workflow Finished!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ LONG model trained${NC}"
    echo -e "${GREEN}✓ SHORT model trained${NC}"
    echo -e "${GREEN}✓ Backtest completed${NC}"
    echo -e "${GREEN}✓ Report generated${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review analysis results in model folders"
    echo "2. Check backtest results in backtests/ folder"
    echo "3. Open HTML report for detailed performance analysis"
    echo "4. Consider running parameter optimization (option 10)"
}

# Function to analyze trades
analyze_trades() {
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Launching Trade Analysis Tool${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    python analyze_trades.py
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice [0-10]: " choice
    echo ""
    
    case $choice in
        1)
            train_long
            ;;
        2)
            train_short
            ;;
        3)
            echo -e "${YELLOW}Training BOTH models (LONG + SHORT)${NC}"
            echo "This will take approximately 4-8 hours..."
            echo ""
            # Prompt for ticker once
            if prompt_ticker; then
                train_long 1
                echo ""
                train_short 1
            fi
            ;;
        4)
            run_analysis "long"
            ;;
        5)
            run_analysis "short"
            ;;
        6)
            run_backtest
            ;;
        7)
            generate_report
            ;;
        8)
            complete_workflow
            ;;
        9)
            analyze_trades
            ;;
        10)
            optimize_parameters
            ;;
        0)
            echo -e "${BLUE}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            ;;
    esac
    
    echo ""
    echo "Press Enter to continue..."
    read
    clear
done
