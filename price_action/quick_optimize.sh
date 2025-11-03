#!/bin/bash
# Quick Optuna Optimization Script
# Optimizes one ticker at a time

set -e

echo "========================================"
echo "  Quick Optuna Optimization"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if in correct directory
if [ ! -f "optuna_optimizer.py" ]; then
    echo -e "${RED}Error: optuna_optimizer.py not found${NC}"
    echo "Please run this script from price_action/ directory"
    exit 1
fi

# Function to optimize a ticker
optimize_ticker() {
    local TICKER=$1
    local VERSION=$2
    local TIMEFRAME=$3
    shift 3
    local HELPERS="$@"

    echo -e "${BLUE}============================================${NC}"
    echo -e "${GREEN}Optimizing: $TICKER${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo "Version: $VERSION"
    echo "Timeframe: $TIMEFRAME"
    echo "Helpers: $HELPERS"
    echo "Trials: 100"
    echo ""

    python optuna_optimizer.py \
        --ticker "$TICKER" \
        --timeframe "$TIMEFRAME" \
        --helper-timeframes $HELPERS \
        --version "$VERSION" \
        --trials 100 \
        --partial-tp

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Optimization completed for $TICKER${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ Optimization failed for $TICKER${NC}"
        echo ""
        return 1
    fi
}

# Menu
echo "Select optimization mode:"
echo "1) Optimize single ticker (interactive)"
echo "2) Optimize ALL tickers (automated - ~18 hours)"
echo "3) Optimize specific tickers (batch)"
echo "0) Exit"
echo ""
read -p "Enter choice [0-3]: " choice

case $choice in
    1)
        # Single ticker optimization
        echo ""
        read -p "Enter ticker (e.g., SOLUSDT): " ticker
        ticker=$(echo "$ticker" | tr '[:lower:]' '[:upper:]')

        # Auto-detect version and parameters from training_metadata.json
        model_dir=$(find models -name "${ticker}*_long" -type d | head -1)

        if [ -z "$model_dir" ]; then
            echo -e "${RED}Error: Model not found for $ticker${NC}"
            exit 1
        fi

        metadata_file="$model_dir/training_metadata.json"

        if [ ! -f "$metadata_file" ]; then
            echo -e "${RED}Error: training_metadata.json not found${NC}"
            exit 1
        fi

        # Extract parameters using Python
        params=$(python3 -c "
import json
with open('$metadata_file', 'r') as f:
    data = json.load(f)
    print(data['version'])
    print(data['timeframe'])
    print(' '.join(data.get('helper_timeframes', [])))
")

        version=$(echo "$params" | sed -n '1p')
        timeframe=$(echo "$params" | sed -n '2p')
        helpers=$(echo "$params" | sed -n '3p')

        echo -e "${YELLOW}Auto-detected:${NC}"
        echo "  Version: $version"
        echo "  Timeframe: $timeframe"
        echo "  Helpers: $helpers"
        echo ""

        read -p "Continue? (y/n): " confirm
        if [ "$confirm" != "y" ]; then
            echo "Cancelled."
            exit 0
        fi

        optimize_ticker "$ticker" "$version" "$timeframe" $helpers
        ;;

    2)
        # Optimize all tickers
        echo ""
        echo -e "${YELLOW}Warning: This will optimize ALL 9 tickers${NC}"
        echo "Estimated time: ~18 hours (2 hours per ticker)"
        echo ""
        read -p "Continue? (y/n): " confirm

        if [ "$confirm" != "y" ]; then
            echo "Cancelled."
            exit 0
        fi

        # Run Python script
        python optimize_all_models.py --trials 100 --timeout-hours 2
        ;;

    3)
        # Batch optimization
        echo ""
        echo "Available tickers:"
        echo "  1) SOLUSDT"
        echo "  2) DOGEUSDT"
        echo "  3) ETHUSDT"
        echo "  4) LINKUSDT"
        echo "  5) HBARUSDT"
        echo "  6) SUIUSDT"
        echo "  7) NEARUSDT"
        echo "  8) TRXUSDT"
        echo "  9) KASUSDT"
        echo ""
        read -p "Enter ticker numbers (space-separated, e.g., 1 2 3): " selections

        # Map selections to tickers
        declare -A ticker_map=(
            [1]="SOLUSDT v1.2.sol 15m 1h 4h 12h 1d"
            [2]="DOGEUSDT v1.2.doge 15m 1h 4h 6h 12h 1d"
            [3]="ETHUSDT v1.2.eth 15m 1h 2h 4h 6h 12h 1d"
            [4]="LINKUSDT v1.2.link 15m 1h 2h 4h 6h 12h 1d"
            [5]="HBARUSDT v1.2.hbar 15m 1h 2h 4h 6h 12h 1d"
            [6]="SUIUSDT v1.2.sui 15m 1h 2h 4h 6h 12h 1d"
            [7]="NEARUSDT v1.2.near 15m 1h 2h 4h 6h 12h 1d"
            [8]="TRXUSDT v1.2.trx 15m 1h 2h 4h 12h 1d"
            [9]="KASUSDT v1.2.kas 15m 1h 2h 4h 12h 1d"
        )

        for num in $selections; do
            if [ -n "${ticker_map[$num]}" ]; then
                params=(${ticker_map[$num]})
                optimize_ticker "${params[@]}"
            else
                echo -e "${RED}Invalid selection: $num${NC}"
            fi
        done
        ;;

    0)
        echo "Exiting."
        exit 0
        ;;

    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Optimization completed!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Next steps:"
echo "1. Check results in models/*/optuna/"
echo "2. Run: python analyze_all_models.py"
echo "3. Update docker-compose.yaml with new parameters"
echo "4. Restart: docker compose down && docker compose up -d"
