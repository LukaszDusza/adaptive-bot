#!/bin/bash

# ========================================
# Complete Workflow Script
# ========================================
# This script helps you easily train, analyze, and backtest models

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TICKER="SOLUSDT"
TIMEFRAME="15m"
HELPER_TIMEFRAMES="1h 4h 1D"
VERSION=""
LIMIT_TRAIN=138240
LIMIT_BACKTEST=8640 # trzy miesiace do tylu
DATE_FROM="2025-05-31"
FETCH_MAX_HISTORY="--fetch-max-history"  # Set to "--fetch-max-history" to fetch ALL available data, or "" to use LIMIT_TRAIN
LABEL_TRIALS=100
MODEL_TRIALS=100
PROB_THRESHOLD=0.54
MIN_PROBA_DIFF=0.3
TP_PCT=0.02
TSL_PCT=0.014
TRADE_SIZE=1000
OPTUNA_TRIALS=100
TP_MECHANISM="partial-tp"  # Options: "partial-tp" (50% at halfway) or "dynamic-tp" (25% at 4 levels)
PROTECT_PROFIT=""  # Options: "" (disabled) or "--protect-profit" (enabled)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ML Trading Workflow${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to auto-detect model parameters from directory structure
auto_detect_model_params() {
    local version=$1

    if [ -z "$version" ]; then
        echo -e "${RED}Version not specified for auto-detection${NC}"
        return 1
    fi

    # Check if version directory exists
    if [ ! -d "models/$version" ]; then
        echo -e "${RED}Version directory does not exist: models/$version${NC}"
        return 1
    fi

    # Find first model directory (either _long or _short)
    local model_dir=$(ls -d models/$version/*_long 2>/dev/null | head -1)
    if [ -z "$model_dir" ]; then
        model_dir=$(ls -d models/$version/*_short 2>/dev/null | head -1)
    fi

    if [ -z "$model_dir" ]; then
        echo -e "${YELLOW}No model directories found in models/$version/${NC}"
        return 1
    fi

    # Extract directory name (e.g., SOLUSDT_15m_plus_1h_4h_1D_long)
    local dir_name=$(basename "$model_dir")

    # Parse: TICKER_TIMEFRAME_plus_HELPERS_SIDE
    # Remove _long or _short suffix
    local base_name="${dir_name%_long}"
    base_name="${base_name%_short}"

    # Extract TICKER (everything before first underscore followed by timeframe)
    # Match pattern: TICKER_TF where TF is like 15m, 1h, 4h, 1D
    if [[ $base_name =~ ^([A-Z]+)_([0-9]+[mhDW])(.*)$ ]]; then
        TICKER="${BASH_REMATCH[1]}"
        TIMEFRAME="${BASH_REMATCH[2]}"
        local remainder="${BASH_REMATCH[3]}"

        # Extract helper timeframes from "_plus_1h_4h_1D" part
        if [[ $remainder =~ _plus_(.+)$ ]]; then
            local helpers_str="${BASH_REMATCH[1]}"
            # Replace underscores with spaces (1h_4h_1D -> 1h 4h 1D)
            HELPER_TIMEFRAMES="${helpers_str//_/ }"
        else
            HELPER_TIMEFRAMES=""
        fi

        echo -e "${GREEN}✓ Auto-detected from model directory:${NC}"
        echo "  Ticker: $TICKER"
        echo "  Timeframe: $TIMEFRAME"
        echo "  Helper Timeframes: ${HELPER_TIMEFRAMES:-none}"
        echo ""
        return 0
    else
        echo -e "${YELLOW}Could not parse model directory name: $dir_name${NC}"
        return 1
    fi
}

# Prompt for version at the beginning
prompt_version() {
    echo -e "${YELLOW}Enter model version (e.g., v1.0, v1.1, v2.0):${NC}"
    read -p "Version: " input_version

    if [[ -z "$input_version" ]]; then
        echo -e "${RED}Version is required!${NC}"
        return 1
    fi

    VERSION="$input_version"
    echo -e "${GREEN}Using version: $VERSION${NC}"

    # Check if version directory exists
    if [ -d "models/$VERSION" ]; then
        echo -e "${YELLOW}⚠️  Warning: Version $VERSION already exists!${NC}"
        echo "Existing files may be overwritten."
        read -p "Continue? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            return 1
        fi
    fi
    echo ""
    return 0
}

echo -e "${YELLOW}Current Parameters:${NC}"
echo "  VERSION: $VERSION (will be prompted)"
echo "  DATE_FROM: $DATE_FROM"
echo "  FETCH_MAX_HISTORY: ${FETCH_MAX_HISTORY:-disabled}"
echo "  LIMIT_TRAIN: $LIMIT_TRAIN (ignored if FETCH_MAX_HISTORY enabled)"
echo "  LABEL_TRIALS: $LABEL_TRIALS"
echo "  MODEL_TRIALS: $MODEL_TRIALS"
echo "  PROB_THRESHOLD: $PROB_THRESHOLD"
echo "  MIN_PROBA_DIFF: $MIN_PROBA_DIFF"
echo "  TP_PCT: $TP_PCT"
echo "  TSL_PCT: $TSL_PCT"
echo "  TP_MECHANISM: $TP_MECHANISM"
echo "  PROTECT_PROFIT: ${PROTECT_PROFIT:-disabled}"
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
    echo "11) Optimize & Auto-Deploy 🚀 (Optuna → Update Docker → Restart)"
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
        else
            echo -e "${GREEN}Using default ticker: $TICKER${NC}"
        fi
        echo ""
    fi
    return 0
}

# Function to prompt for timeframe
prompt_timeframe() {
    local skip_prompt=${1:-0}
    
    if [ "$skip_prompt" -eq 0 ]; then
        echo -e "${YELLOW}Enter timeframe (e.g., 5m, 15m, 1h, 4h, default: $TIMEFRAME):${NC}"
        read -p "Timeframe: " input_timeframe
        
        if [[ ! -z "$input_timeframe" ]]; then
            TIMEFRAME="$input_timeframe"
            echo -e "${GREEN}Using timeframe: $TIMEFRAME${NC}"
        else
            echo -e "${GREEN}Using default timeframe: $TIMEFRAME${NC}"
        fi
        echo ""
    fi
    return 0
}

# Function to prompt for helper timeframes
prompt_helper_timeframes() {
    local skip_prompt=${1:-0}
    
    if [ "$skip_prompt" -eq 0 ]; then
        echo -e "${YELLOW}Enter helper timeframes (space-separated, e.g., '1h 4h', default: $HELPER_TIMEFRAMES):${NC}"
        read -p "Helper timeframes: " input_helper_timeframes
        
        if [[ ! -z "$input_helper_timeframes" ]]; then
            HELPER_TIMEFRAMES="$input_helper_timeframes"
            echo -e "${GREEN}Using helper timeframes: $HELPER_TIMEFRAMES${NC}"
        else
            echo -e "${GREEN}Using default helper timeframes: $HELPER_TIMEFRAMES${NC}"
        fi
        echo ""
    fi
    return 0
}

# Function to prompt for training parameters
prompt_training_params() {
    local skip_prompt=${1:-0}

    if [ "$skip_prompt" -eq 0 ]; then
        echo -e "${YELLOW}Training Parameters:${NC}"
        echo ""

        read -p "Label trials (default: $LABEL_TRIALS): " input_label_trials
        if [[ ! -z "$input_label_trials" ]]; then
            LABEL_TRIALS=$input_label_trials
        fi

        read -p "Model trials (default: $MODEL_TRIALS): " input_model_trials
        if [[ ! -z "$input_model_trials" ]]; then
            MODEL_TRIALS=$input_model_trials
        fi

        echo ""
        echo -e "${GREEN}Using: LABEL_TRIALS=$LABEL_TRIALS, MODEL_TRIALS=$MODEL_TRIALS${NC}"
        echo ""
    fi
    return 0
}

# Function to train LONG model
train_long() {
    local skip_prompt=${1:-0}
    
    # Prompt for version if not set
    if [[ -z "$VERSION" ]]; then
        if ! prompt_version; then
            return
        fi
    fi

    # Prompt for ticker selection
    if ! prompt_ticker $skip_prompt; then
        return
    fi

    # Prompt for timeframe selection
    if ! prompt_timeframe $skip_prompt; then
        return
    fi

    # Prompt for helper timeframes selection
    if ! prompt_helper_timeframes $skip_prompt; then
        return
    fi

    # Prompt for training parameters
    if ! prompt_training_params $skip_prompt; then
        return
    fi

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Training LONG model for $TICKER ${TIMEFRAME} (version: $VERSION)${NC}"
    echo -e "${YELLOW}Label Trials: $LABEL_TRIALS | Model Trials: $MODEL_TRIALS${NC}"
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
        --model-trials $MODEL_TRIALS \
        --version "$VERSION" \
        $FETCH_MAX_HISTORY

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ LONG model training completed successfully!${NC}"
        echo -e "${GREEN}  Model saved to: models/$VERSION/${TICKER}_${TIMEFRAME}_plus_*_long/${NC}"
    else
        echo -e "${RED}✗ LONG model training failed!${NC}"
        exit 1
    fi
}

# Function to train SHORT model
train_short() {
    local skip_prompt=${1:-0}
    
    # Prompt for version if not set
    if [[ -z "$VERSION" ]]; then
        if ! prompt_version; then
            return
        fi
    fi

    # Prompt for ticker selection
    if ! prompt_ticker $skip_prompt; then
        return
    fi

    # Prompt for timeframe selection
    if ! prompt_timeframe $skip_prompt; then
        return
    fi

    # Prompt for helper timeframes selection
    if ! prompt_helper_timeframes $skip_prompt; then
        return
    fi

    # Prompt for training parameters
    if ! prompt_training_params $skip_prompt; then
        return
    fi

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Training SHORT model for $TICKER ${TIMEFRAME} (version: $VERSION)${NC}"
    echo -e "${YELLOW}Label Trials: $LABEL_TRIALS | Model Trials: $MODEL_TRIALS${NC}"
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
        --model-trials $MODEL_TRIALS \
        --version "$VERSION" \
        $FETCH_MAX_HISTORY

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ SHORT model training completed successfully!${NC}"
        echo -e "${GREEN}  Model saved to: models/$VERSION/${TICKER}_${TIMEFRAME}_plus_*_short/${NC}"
    else
        echo -e "${RED}✗ SHORT model training failed!${NC}"
        exit 1
    fi
}

# Function to run analysis
run_analysis() {
    local side=$1

    # Prompt for version if not set
    if [[ -z "$VERSION" ]]; then
        if ! prompt_version; then
            return
        fi
    fi

    # Auto-detect model parameters from version directory
    if ! auto_detect_model_params "$VERSION"; then
        echo -e "${YELLOW}Auto-detection failed. Please enter parameters manually.${NC}"

        # Fallback to manual prompts
        if ! prompt_ticker; then
            return
        fi
        if ! prompt_timeframe; then
            return
        fi
        if ! prompt_helper_timeframes; then
            return
        fi
    fi

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Running Analysis for $side model (version: $VERSION)${NC}"
    echo -e "${YELLOW}========================================${NC}"

    python analysis.py \
        --ticker "$TICKER" \
        --timeframe $TIMEFRAME \
        --side $side \
        --helper-timeframes $HELPER_TIMEFRAMES \
        --version "$VERSION"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Analysis completed successfully!${NC}"
    else
        echo -e "${RED}✗ Analysis failed!${NC}"
    fi
}

# Function to run backtest
run_backtest() {
    # Prompt for version if not set
    if [[ -z "$VERSION" ]]; then
        if ! prompt_version; then
            return
        fi
    fi

    # Auto-detect model parameters from version directory
    if ! auto_detect_model_params "$VERSION"; then
        echo -e "${YELLOW}Auto-detection failed. Please enter parameters manually.${NC}"

        # Fallback to manual prompts
        if ! prompt_ticker; then
            return
        fi
        if ! prompt_timeframe; then
            return
        fi
        if ! prompt_helper_timeframes; then
            return
        fi
    fi

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Running Backtest for $TICKER ${TIMEFRAME} (version: $VERSION)${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo -e "${GREEN}Select TP Mechanism:${NC}"
    echo "1) Partial TP (50% at halfway to TP)"
    echo "2) Dynamic TP (25% at each of 4 levels: 25%, 50%, 75%, 100%)"
    echo ""
    read -p "Enter your choice [1-2]: " tp_choice
    echo ""

    case $tp_choice in
        1)
            SELECTED_TP_MECHANISM="partial-tp"
            echo -e "${GREEN}Selected: Partial TP (50% at halfway)${NC}"
            ;;
        2)
            SELECTED_TP_MECHANISM="dynamic-tp"
            echo -e "${GREEN}Selected: Dynamic TP (25% at 4 levels)${NC}"
            ;;
        *)
            echo -e "${YELLOW}Invalid choice. Using default: partial-tp${NC}"
            SELECTED_TP_MECHANISM="partial-tp"
            ;;
    esac
    echo ""

    # Prompt for profit protection
    echo -e "${GREEN}Enable Profit Protection?${NC}"
    echo "Moves SL to breakeven if profit peaks >0.25% but declines before hitting partial TP"
    echo ""
    echo "1) No (disabled)"
    echo "2) Yes (enabled)"
    echo ""
    read -p "Enter your choice [1-2]: " protect_choice
    echo ""

    case $protect_choice in
        2)
            PROTECT_PROFIT="--protect-profit"
            echo -e "${GREEN}Selected: Profit Protection ENABLED${NC}"
            ;;
        *)
            PROTECT_PROFIT=""
            echo -e "${YELLOW}Selected: Profit Protection DISABLED${NC}"
            ;;
    esac
    echo ""

    echo "Parameters:"
    echo "  - Version: $VERSION"
    echo "  - Probability Threshold: $PROB_THRESHOLD"
    echo "  - Min Proba Diff: $MIN_PROBA_DIFF"
    echo "  - Take Profit: ${TP_PCT}%"
    echo "  - Trailing Stop Loss: ${TSL_PCT}%"
    echo "  - Trade Size: \$${TRADE_SIZE}"
    echo "  - TP Mechanism: $SELECTED_TP_MECHANISM"
    echo "  - Profit Protection: ${PROTECT_PROFIT:-disabled}"
    echo ""

    python main.py \
        --backtest \
        --ticker "$TICKER" \
        --timeframe "$TIMEFRAME" \
        --helper-timeframes $HELPER_TIMEFRAMES \
        --limit $LIMIT_BACKTEST \
        --prob-threshold $PROB_THRESHOLD \
        --min-proba-diff $MIN_PROBA_DIFF \
        --tp-pct $TP_PCT \
        --tsl-pct $TSL_PCT \
        --trade-size $TRADE_SIZE \
        --version "$VERSION" \
        --$SELECTED_TP_MECHANISM \
        $PROTECT_PROFIT
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Backtest completed successfully!${NC}"
        echo -e "${GREEN}  Results saved to: models/$VERSION/backtests/${NC}"
    else
        echo -e "${RED}✗ Backtest failed!${NC}"
    fi
}

# Function to generate report
generate_report() {
    # Prompt for version if not set
    if [[ -z "$VERSION" ]]; then
        if ! prompt_version; then
            return
        fi
    fi

    # Auto-detect model parameters from version directory
    if ! auto_detect_model_params "$VERSION"; then
        echo -e "${YELLOW}Auto-detection failed. Please enter parameters manually.${NC}"

        # Fallback to manual prompts
        if ! prompt_ticker; then
            return
        fi
        if ! prompt_timeframe; then
            return
        fi
        if ! prompt_helper_timeframes; then
            return
        fi
    fi

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Generating Report (version: $VERSION)${NC}"
    echo -e "${YELLOW}========================================${NC}"

    python main.py \
        --report \
        --ticker $TICKER \
        --timeframe $TIMEFRAME \
        --helper-timeframes $HELPER_TIMEFRAMES \
        --version "$VERSION"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Report generated successfully!${NC}"
        echo -e "${GREEN}  Check the reports/ directory${NC}"
    else
        echo -e "${RED}✗ Report generation failed!${NC}"
    fi
}

# Function to optimize parameters with Optuna
optimize_parameters() {
    # Prompt for version if not set
    if [[ -z "$VERSION" ]]; then
        if ! prompt_version; then
            return
        fi
    fi

    # Auto-detect model parameters from version directory
    if ! auto_detect_model_params "$VERSION"; then
        echo -e "${YELLOW}Auto-detection failed. Please enter parameters manually.${NC}"

        # Fallback to manual prompts
        if ! prompt_ticker; then
            return
        fi
        if ! prompt_timeframe; then
            return
        fi
        if ! prompt_helper_timeframes; then
            return
        fi
    fi

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Optimizing Parameters with Optuna${NC}"
    echo -e "${YELLOW}Multi-Objective Optimization${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo "This will find the optimal values for:"
    echo "  - PROB_THRESHOLD (probability threshold)"
    echo "  - TP_PCT (take profit percentage)"
    echo "  - TSL_PCT (trailing stop loss percentage)"
    echo ""
    echo -e "${GREEN}Optimization Objectives (Priority Order):${NC}"
    echo "  1️⃣  PRIMARY: Maximize PnL (USD profit)"
    echo "  2️⃣  SECONDARY: Minimize Drawdown (risk control)"
    echo "  3️⃣  TERTIARY: Optimize Trade Count (balanced activity)"
    echo ""
    echo "The optimizer finds Pareto-optimal solutions that balance"
    echo "these three objectives and selects the one with highest PnL."
    echo "This prevents strategies with too few trades (e.g., 1 trade"
    echo "over 3 months) while maximizing profit and controlling risk."
    echo ""
    echo -e "${GREEN}Database Persistence:${NC}"
    echo "  • Results are saved to SQLite database in optuna/ directory"
    echo "  • Unique database per ticker + timeframe + helper timeframes"
    echo "  • You can resume optimization later by running this again"
    echo "  • Previous trials are preserved and new trials are added"
    echo ""
    echo "Current parameters:"
    echo "  - PROB_THRESHOLD: $PROB_THRESHOLD"
    echo "  - MIN_PROBA_DIFF: $MIN_PROBA_DIFF"
    echo "  - TP_PCT: $TP_PCT"
    echo "  - TSL_PCT: $TSL_PCT"
    echo ""
    
    echo -e "${GREEN}Select TP Mechanism for Optimization:${NC}"
    echo "1) Partial TP (50% at halfway to TP)"
    echo "2) Dynamic TP (25% at each of 4 levels: 25%, 50%, 75%, 100%)"
    echo ""
    read -p "Enter your choice [1-2]: " tp_choice
    echo ""
    
    case $tp_choice in
        1)
            SELECTED_TP_MECHANISM="partial-tp"
            echo -e "${GREEN}Selected: Partial TP (50% at halfway)${NC}"
            ;;
        2)
            SELECTED_TP_MECHANISM="dynamic-tp"
            echo -e "${GREEN}Selected: Dynamic TP (25% at 4 levels)${NC}"
            ;;
        *)
            echo -e "${YELLOW}Invalid choice. Using default: partial-tp${NC}"
            SELECTED_TP_MECHANISM="partial-tp"
            ;;
    esac
    echo ""

    # Prompt for profit protection
    echo -e "${GREEN}Enable Profit Protection?${NC}"
    echo "Moves SL to breakeven if profit peaks >0.25% but declines before hitting partial TP"
    echo ""
    echo "1) No (disabled)"
    echo "2) Yes (enabled)"
    echo ""
    read -p "Enter your choice [1-2]: " protect_choice
    echo ""

    case $protect_choice in
        2)
            PROTECT_PROFIT="--protect-profit"
            echo -e "${GREEN}Selected: Profit Protection ENABLED${NC}"
            ;;
        *)
            PROTECT_PROFIT=""
            echo -e "${YELLOW}Selected: Profit Protection DISABLED${NC}"
            ;;
    esac
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
        --version "$VERSION" \
        --limit $LIMIT_BACKTEST \
        --trials $OPTUNA_TRIALS \
        --$SELECTED_TP_MECHANISM \
        $PROTECT_PROFIT
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}✓ Optimization completed successfully!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "Results saved to: models/$VERSION/optuna/optimization_results_${TICKER}_${TIMEFRAME}_${SELECTED_TP_MECHANISM}_limit${LIMIT_BACKTEST}.txt"
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

# Function to automatically deploy optimized parameters
auto_deploy_optimized_params() {
    # Prompt for version if not set
    if [[ -z "$VERSION" ]]; then
        if ! prompt_version; then
            return
        fi
    fi

    # Auto-detect model parameters from version directory
    if ! auto_detect_model_params "$VERSION"; then
        echo -e "${YELLOW}Auto-detection failed. Please enter parameters manually.${NC}"

        # Fallback to manual prompts
        if ! prompt_ticker; then
            return
        fi
        if ! prompt_timeframe; then
            return
        fi
        if ! prompt_helper_timeframes; then
            return
        fi
    fi

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}🚀 Optimize & Auto-Deploy${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "This will:"
    echo "  1. Run Optuna optimization for $TICKER $TIMEFRAME (version: $VERSION)"
    echo "  2. Extract best parameters from results"
    echo "  3. Update docker-compose.yaml"
    echo "  4. Restart Docker container with new parameters"
    echo ""
    echo -e "${YELLOW}⚠️  Warning: This will modify docker-compose.yaml and restart containers!${NC}"
    echo ""

    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        return
    fi
    
    # Step 1: Run Optuna optimization
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}STEP 1/4: Running Optuna Optimization${NC}"
    echo -e "${BLUE}========================================${NC}"
    optimize_parameters
    
    # Check if optimization succeeded
    # Note: SELECTED_TP_MECHANISM is set by optimize_parameters function
    RESULTS_FILE="models/$VERSION/optuna/optimization_results_${TICKER}_${TIMEFRAME}_${SELECTED_TP_MECHANISM}_limit${LIMIT_BACKTEST}.txt"
    if [ ! -f "$RESULTS_FILE" ]; then
        echo -e "${RED}✗ Optimization results file not found: $RESULTS_FILE${NC}"
        echo -e "${RED}Cannot proceed with auto-deployment.${NC}"
        return
    fi
    
    # Step 2: Extract best parameters from results file
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}STEP 2/4: Extracting Best Parameters${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Parse parameters from results file
    OPT_PROB_THRESHOLD=$(grep "^PROB_THRESHOLD=" "$RESULTS_FILE" | cut -d'=' -f2)
    OPT_MIN_PROBA_DIFF=$(grep "^MIN_PROBA_DIFF=" "$RESULTS_FILE" | cut -d'=' -f2)
    OPT_TP_PCT=$(grep "^TP_PCT=" "$RESULTS_FILE" | cut -d'=' -f2)
    OPT_TSL_PCT=$(grep "^TSL_PCT=" "$RESULTS_FILE" | cut -d'=' -f2)
    
    if [ -z "$OPT_PROB_THRESHOLD" ] || [ -z "$OPT_MIN_PROBA_DIFF" ] || [ -z "$OPT_TP_PCT" ] || [ -z "$OPT_TSL_PCT" ]; then
        echo -e "${RED}✗ Failed to extract parameters from results file${NC}"
        echo "Expected format: PROB_THRESHOLD=X.XX"
        return
    fi
    
    echo -e "${GREEN}✓ Extracted optimal parameters:${NC}"
    echo "  PROB_THRESHOLD = $OPT_PROB_THRESHOLD"
    echo "  MIN_PROBA_DIFF = $OPT_MIN_PROBA_DIFF"
    echo "  TP_PCT         = $OPT_TP_PCT"
    echo "  TSL_PCT        = $OPT_TSL_PCT"
    
    # Step 3: Update docker-compose.yaml
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}STEP 3/4: Updating docker-compose.yaml${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    DOCKER_COMPOSE_FILE="docker-compose.yaml"
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        echo -e "${RED}✗ docker-compose.yaml not found${NC}"
        return
    fi
    
    # Create backup
    BACKUP_FILE="${DOCKER_COMPOSE_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$DOCKER_COMPOSE_FILE" "$BACKUP_FILE"
    echo -e "${GREEN}✓ Created backup: $BACKUP_FILE${NC}"
    
    # Update parameters in docker-compose.yaml for the specific ticker
    # Find the service block for this ticker and update parameters
    python3 << EOF
import re
import sys

ticker = "$TICKER"
prob_threshold = "$OPT_PROB_THRESHOLD"
min_proba_diff = "$OPT_MIN_PROBA_DIFF"
tp_pct = "$OPT_TP_PCT"
tsl_pct = "$OPT_TSL_PCT"

try:
    with open("$DOCKER_COMPOSE_FILE", 'r') as f:
        content = f.read()
    
    # Find the service block containing this ticker
    # Pattern: --ticker TICKER ... --prob-threshold X.XX --min-proba-diff X.XX --tsl-pct X.XX --tp-pct X.XX
    
    # Update prob-threshold
    content = re.sub(
        r'(--ticker\s+' + ticker + r'.*?--prob-threshold\s+)[\d.]+',
        r'\g<1>' + prob_threshold,
        content,
        flags=re.DOTALL
    )
    
    # Update min-proba-diff
    content = re.sub(
        r'(--ticker\s+' + ticker + r'.*?--min-proba-diff\s+)[\d.]+',
        r'\g<1>' + min_proba_diff,
        content,
        flags=re.DOTALL
    )
    
    # Update tsl-pct
    content = re.sub(
        r'(--ticker\s+' + ticker + r'.*?--tsl-pct\s+)[\d.]+',
        r'\g<1>' + tsl_pct,
        content,
        flags=re.DOTALL
    )
    
    # Update tp-pct
    content = re.sub(
        r'(--ticker\s+' + ticker + r'.*?--tp-pct\s+)[\d.]+',
        r'\g<1>' + tp_pct,
        content,
        flags=re.DOTALL
    )
    
    with open("$DOCKER_COMPOSE_FILE", 'w') as f:
        f.write(content)
    
    print("✓ Successfully updated docker-compose.yaml")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Error updating docker-compose.yaml: {e}")
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to update docker-compose.yaml${NC}"
        echo -e "${YELLOW}Restoring backup...${NC}"
        cp "$BACKUP_FILE" "$DOCKER_COMPOSE_FILE"
        return
    fi
    
    echo -e "${GREEN}✓ docker-compose.yaml updated successfully${NC}"
    echo ""
    echo "Updated parameters for ticker: $TICKER"
    echo "  --prob-threshold $OPT_PROB_THRESHOLD"
    echo "  --min-proba-diff $OPT_MIN_PROBA_DIFF"
    echo "  --tp-pct $OPT_TP_PCT"
    echo "  --tsl-pct $OPT_TSL_PCT"
    
    # Step 4: Restart Docker containers
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}STEP 4/4: Restarting Docker Containers${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    echo "Running: docker compose down && docker compose up --build -d"
    echo ""
    
    docker compose down
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to stop containers${NC}"
        return
    fi
    
    echo ""
    docker compose up --build -d
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to start containers${NC}"
        return
    fi
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Auto-Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Summary:"
    echo "  ✓ Optimization completed"
    echo "  ✓ Parameters extracted"
    echo "  ✓ docker-compose.yaml updated"
    echo "  ✓ Containers restarted with new parameters"
    echo ""
    echo "Optimal parameters now deployed for $TICKER:"
    echo "  PROB_THRESHOLD: $OPT_PROB_THRESHOLD"
    echo "  MIN_PROBA_DIFF: $OPT_MIN_PROBA_DIFF"
    echo "  TP_PCT:         $OPT_TP_PCT"
    echo "  TSL_PCT:        $OPT_TSL_PCT"
    echo ""
    echo "Backup saved to: $BACKUP_FILE"
    echo ""
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
    
    # Prompt for version once
    if ! prompt_version; then
        return
    fi

    # Prompt for ticker once
    if ! prompt_ticker; then
        return
    fi

    # Prompt for timeframe once
    if ! prompt_timeframe; then
        return
    fi

    # Prompt for helper timeframes once
    if ! prompt_helper_timeframes; then
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
    echo -e "${GREEN}✓ LONG model trained (version: $VERSION)${NC}"
    echo -e "${GREEN}✓ SHORT model trained (version: $VERSION)${NC}"
    echo -e "${GREEN}✓ Backtest completed${NC}"
    echo -e "${GREEN}✓ Report generated${NC}"
    echo ""
    echo "All results saved in: models/$VERSION/"
    echo ""
    echo "Next steps:"
    echo "1. Review analysis results in model folders"
    echo "2. Check backtest results in models/$VERSION/backtests/ folder"
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
    read -p "Enter your choice [0-11]: " choice
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
            # Prompt for version once
            if ! prompt_version; then
                continue
            fi
            # Prompt for ticker once
            if ! prompt_ticker; then
                continue
            fi
            # Prompt for timeframe once
            if ! prompt_timeframe; then
                continue
            fi
            # Prompt for helper timeframes once
            if ! prompt_helper_timeframes; then
                continue
            fi
            # Prompt for training parameters once
            if ! prompt_training_params; then
                continue
            fi
            train_long 1
            echo ""
            train_short 1
            ;;
        4)
            run_analysis "long"
            VERSION=""
            ;;
        5)
            run_analysis "short"
            VERSION=""
            ;;
        6)
            run_backtest
            VERSION=""
            ;;
        7)
            generate_report
            VERSION=""
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
        11)
            auto_deploy_optimized_params
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
