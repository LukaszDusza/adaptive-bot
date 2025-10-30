#!/bin/bash

# Test script for regenerate_all_analyses function
# Usage: ./test_regenerate_analyses.sh v1.2.doge

set -e

VERSION=${1:-"v1.2.doge"}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Regenerating ALL analyses for version: $VERSION${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Check if version directory exists
if [ ! -d "models/$VERSION" ]; then
    echo -e "${RED}Version directory does not exist: models/$VERSION${NC}"
    exit 1
fi

# Find all model directories (_long and _short)
model_dirs=$(find models/$VERSION -maxdepth 1 -type d \( -name "*_long" -o -name "*_short" \) | sort)

if [ -z "$model_dirs" ]; then
    echo -e "${RED}No model directories found in models/$VERSION/${NC}"
    exit 1
fi

count=0
success=0
failed=0

echo -e "${BLUE}Found models:${NC}"
echo "$model_dirs" | while read model_dir; do
    echo "  - $(basename $model_dir)"
done
echo ""

# Process each model directory
while IFS= read -r model_dir; do
    dir_name=$(basename "$model_dir")

    # Parse directory name: TICKER_TIMEFRAME_plus_HELPERS_SIDE
    # Remove _long or _short suffix
    base_name="${dir_name%_long}"
    base_name="${base_name%_short}"

    # Extract side
    side="long"
    if [[ $dir_name == *_short ]]; then
        side="short"
    fi

    # Extract TICKER and TIMEFRAME
    if [[ $base_name =~ ^([A-Z]+)_([0-9]+[mhDW])(.*)$ ]]; then
        ticker="${BASH_REMATCH[1]}"
        timeframe="${BASH_REMATCH[2]}"
        remainder="${BASH_REMATCH[3]}"

        # Extract helper timeframes from "_plus_1h_4h_1D" part
        helpers=""
        if [[ $remainder =~ _plus_(.+)$ ]]; then
            helpers_str="${BASH_REMATCH[1]}"
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
            --version "$VERSION" 2>&1 | grep -E "(✓|✗|Error|Failed|Zapisano|wygenerowane)"

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
