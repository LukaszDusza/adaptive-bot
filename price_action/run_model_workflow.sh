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
DATE_FROM=""  # Leave empty to fetch from current date backwards (RECOMMENDED for latest market data)
FETCH_MAX_HISTORY="--fetch-max-history"  # Set to "--fetch-max-history" to fetch ALL available data, or "" to use LIMIT_TRAIN
LABEL_TRIALS=100
MODEL_TRIALS=100
PROB_THRESHOLD=0.54
MIN_PROBA_DIFF=0.3
TP_PCT=0.02
TSL_PCT=0.014
TRADE_SIZE=1000
TP_MECHANISM="partial-tp"  # Using only partial-tp (50% at halfway)

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
    echo "7) Regenerate ALL Analyses for Version"
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

# Function to regenerate all analyses for a version
regenerate_all_analyses() {
    # Prompt for version if not set
    if [[ -z "$VERSION" ]]; then
        if ! prompt_version; then
            return
        fi
    fi

    # Check if version directory exists
    if [ ! -d "models/$VERSION" ]; then
        echo -e "${RED}Version directory does not exist: models/$VERSION${NC}"
        return
    fi

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Regenerating ALL analyses for version: $VERSION${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""

    # Find all model directories (_long and _short)
    local model_dirs=$(find models/$VERSION -maxdepth 1 -type d \( -name "*_long" -o -name "*_short" \) | sort)

    if [ -z "$model_dirs" ]; then
        echo -e "${RED}No model directories found in models/$VERSION/${NC}"
        return
    fi

    local count=0
    local success=0
    local failed=0

    echo -e "${BLUE}Found models:${NC}"
    echo "$model_dirs" | while read model_dir; do
        echo "  - $(basename $model_dir)"
    done
    echo ""

    # Process each model directory
    while IFS= read -r model_dir; do
        local dir_name=$(basename "$model_dir")

        # Parse directory name: TICKER_TIMEFRAME_plus_HELPERS_SIDE
        # Remove _long or _short suffix
        local base_name="${dir_name%_long}"
        base_name="${base_name%_short}"

        # Extract side
        local side="long"
        if [[ $dir_name == *_short ]]; then
            side="short"
        fi

        # Extract TICKER and TIMEFRAME
        if [[ $base_name =~ ^([A-Z]+)_([0-9]+[mhDW])(.*)$ ]]; then
            local ticker="${BASH_REMATCH[1]}"
            local timeframe="${BASH_REMATCH[2]}"
            local remainder="${BASH_REMATCH[3]}"

            # Extract helper timeframes from "_plus_1h_4h_1D" part
            local helpers=""
            if [[ $remainder =~ _plus_(.+)$ ]]; then
                local helpers_str="${BASH_REMATCH[1]}"
                # Replace underscores with spaces (1h_4h_1D -> 1h 4h 1D)
                helpers="${helpers_str//_/ }"
            fi

            count=$((count + 1))

            echo -e "${YELLOW}[$count] Processing: ${ticker} ${timeframe} ${side}${NC}"
            echo "    Helpers: ${helpers:-none}"

            # Check if model files exist
            if [ ! -f "$model_dir/model.joblib" ] || [ ! -f "$model_dir/holdout_predictions.csv" ]; then
                echo -e "${RED}    ✗ Missing required files (model.joblib or holdout_predictions.csv)${NC}"
                failed=$((failed + 1))
                echo ""
                continue
            fi

            # Run analysis
            python analysis.py \
                --ticker "$ticker" \
                --timeframe "$timeframe" \
                --side "$side" \
                --helper-timeframes $helpers \
                --version "$VERSION" 2>&1 | grep -E "(✓|✗|Error|Failed|Zapisano)"

            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                echo -e "${GREEN}    ✓ Analysis completed${NC}"
                success=$((success + 1))
            else
                echo -e "${RED}    ✗ Analysis failed${NC}"
                failed=$((failed + 1))
            fi
            echo ""
        else
            echo -e "${YELLOW}Could not parse model directory name: $dir_name${NC}"
            failed=$((failed + 1))
            echo ""
        fi
    done <<< "$model_dirs"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Summary:${NC}"
    echo -e "  Total models: $count"
    echo -e "  ${GREEN}Success: $success${NC}"
    echo -e "  ${RED}Failed: $failed${NC}"
    echo -e "${BLUE}========================================${NC}"
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
    echo "Backtest Parameters:"
    echo "  - Version: $VERSION"
    echo "  - Probability Threshold: $PROB_THRESHOLD"
    echo "  - Min Proba Diff: $MIN_PROBA_DIFF"
    echo "  - Take Profit: ${TP_PCT}%"
    echo "  - Trailing Stop Loss: ${TSL_PCT}%"
    echo "  - Trade Size: \$${TRADE_SIZE}"
    echo "  - TP Mechanism: partial-tp"
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
        --partial-tp
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Backtest completed successfully!${NC}"
        echo -e "${GREEN}  Results saved to: models/$VERSION/backtests/${NC}"
    else
        echo -e "${RED}✗ Backtest failed!${NC}"
    fi
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice [0-7]: " choice
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
            regenerate_all_analyses
            VERSION=""
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
