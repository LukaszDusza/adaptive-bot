#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete ML Trading Workflow Script
Cross-platform (Windows, Linux, Mac)

This script helps you easily train, analyze, and backtest models.
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import Optional, List, Tuple

# ========================================
# ANSI Colors (works on Windows 10+, Linux, Mac)
# ========================================
try:
    # Enable ANSI colors on Windows
    os.system('')
except:
    pass

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

    @staticmethod
    def strip_ansi():
        """Disable colors if terminal doesn't support them"""
        Colors.RED = Colors.GREEN = Colors.YELLOW = Colors.BLUE = Colors.NC = ''


# Check if terminal supports colors
if not sys.stdout.isatty() or os.getenv('NO_COLOR'):
    Colors.strip_ansi()


# ========================================
# Default Configuration
# ========================================
CONFIG = {
    'TICKER': 'SOLUSDT',
    'TIMEFRAME': '15m',
    'HELPER_TIMEFRAMES': ['1h', '4h', '1D'],
    'VERSION': '',
    'LIMIT_TRAIN': 138240,
    'LIMIT_BACKTEST': 8640,  # 3 miesiące
    'DATE_FROM': '',  # Empty = fetch from current date backwards
    'FETCH_MAX_HISTORY': True,  # True = fetch ALL available data
    'LABEL_TRIALS': 100,
    'MODEL_TRIALS': 100,
    'PROB_THRESHOLD': 0.54,
    'MIN_CONFIDENCE_RATIO': 1.5,
    'TP_PCT': 0.02,
    'TSL_PCT': 0.014,
    'TRADE_SIZE': 1000,
    'TP_MECHANISM': 'partial-tp'
}


# ========================================
# Helper Functions
# ========================================

def clear_screen():
    """Clear terminal screen (cross-platform)"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(text: str):
    """Print colored header"""
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BLUE}  {text}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print()


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.NC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.YELLOW}{text}{Colors.NC}")


def input_with_default(prompt: str, default: str) -> str:
    """Get user input with default value"""
    user_input = input(f"{Colors.YELLOW}{prompt} (default: {default}): {Colors.NC}").strip()
    return user_input if user_input else default


def yes_no_prompt(prompt: str) -> bool:
    """Ask yes/no question"""
    while True:
        answer = input(f"{Colors.YELLOW}{prompt} (y/n): {Colors.NC}").strip().lower()
        if answer in ['y', 'yes']:
            return True
        elif answer in ['n', 'no']:
            return False
        print_error("Please answer 'y' or 'n'")


def run_python_command(args: List[str]) -> int:
    """Run Python command and return exit code"""
    try:
        # Use the same Python interpreter that's running this script
        cmd = [sys.executable] + args
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print_error(f"Failed to run command: {e}")
        return 1


def auto_detect_model_params(version: str) -> Optional[Tuple[str, str, List[str]]]:
    """
    Auto-detect model parameters from directory structure.

    Returns:
        (ticker, timeframe, helper_timeframes) or None if detection fails
    """
    models_dir = Path('models') / version

    if not models_dir.exists():
        print_error(f"Version directory does not exist: {models_dir}")
        return None

    # Find first model directory (_long or _short)
    model_dirs = list(models_dir.glob('*_long')) + list(models_dir.glob('*_short'))

    if not model_dirs:
        print_warning(f"No model directories found in {models_dir}")
        return None

    # Parse first model directory name
    dir_name = model_dirs[0].name

    # Remove _long or _short suffix
    base_name = dir_name.replace('_long', '').replace('_short', '')

    # Extract: TICKER_TIMEFRAME_plus_HELPERS
    # Pattern: TICKER_TF where TF is like 15m, 1h, 4h, 1D
    pattern = r'^([A-Z]+)_(\d+[mhDW])(.*)$'
    match = re.match(pattern, base_name)

    if match:
        ticker = match.group(1)
        timeframe = match.group(2)
        remainder = match.group(3)

        # Extract helper timeframes from "_plus_1h_4h_1D" part
        helper_timeframes = []
        if '_plus_' in remainder:
            helpers_str = remainder.split('_plus_')[1]
            # Replace underscores with spaces (1h_4h_1D -> [1h, 4h, 1D])
            helper_timeframes = helpers_str.split('_')

        print_success("Auto-detected from model directory:")
        print(f"  Ticker: {ticker}")
        print(f"  Timeframe: {timeframe}")
        print(f"  Helper Timeframes: {', '.join(helper_timeframes) if helper_timeframes else 'none'}")
        print()

        return ticker, timeframe, helper_timeframes
    else:
        print_warning(f"Could not parse model directory name: {dir_name}")
        return None


def prompt_version() -> Optional[str]:
    """Prompt user for model version"""
    version = input(f"{Colors.YELLOW}Enter model version (e.g., v1.0, v1.1, v2.0): {Colors.NC}").strip()

    if not version:
        print_error("Version is required!")
        return None

    print_success(f"Using version: {version}")

    # Check if version directory exists
    models_dir = Path('models') / version
    if models_dir.exists():
        print_warning(f"Warning: Version {version} already exists!")
        print("Existing files may be overwritten.")
        if not yes_no_prompt("Continue?"):
            print("Cancelled.")
            return None

    print()
    return version


# ========================================
# Training Functions
# ========================================

def train_model(side: str, skip_prompts: bool = False):
    """Train LONG or SHORT model"""

    # Prompt for version if not set
    if not CONFIG['VERSION']:
        version = prompt_version()
        if not version:
            return
        CONFIG['VERSION'] = version

    # Prompt for parameters
    if not skip_prompts:
        CONFIG['TICKER'] = input_with_default("Enter ticker symbol", CONFIG['TICKER']).upper()
        CONFIG['TIMEFRAME'] = input_with_default("Enter timeframe (e.g., 5m, 15m, 1h, 4h)", CONFIG['TIMEFRAME'])

        helpers_input = input_with_default(
            "Enter helper timeframes (space-separated, e.g., '1h 4h')",
            ' '.join(CONFIG['HELPER_TIMEFRAMES'])
        )
        CONFIG['HELPER_TIMEFRAMES'] = helpers_input.split()

        CONFIG['LABEL_TRIALS'] = int(input_with_default("Label trials", str(CONFIG['LABEL_TRIALS'])))
        CONFIG['MODEL_TRIALS'] = int(input_with_default("Model trials", str(CONFIG['MODEL_TRIALS'])))

    print_info("=" * 60)
    print_info(f"Training {side.upper()} model for {CONFIG['TICKER']} {CONFIG['TIMEFRAME']} (version: {CONFIG['VERSION']})")
    print_info(f"Label Trials: {CONFIG['LABEL_TRIALS']} | Model Trials: {CONFIG['MODEL_TRIALS']}")
    print_info("=" * 60)
    print("This will take approximately 2-4 hours...")
    print()

    # Build command
    args = [
        'main.py',
        '--train',
        '--side', side,
        '--ticker', CONFIG['TICKER'],
        '--timeframe', CONFIG['TIMEFRAME'],
        '--helper-timeframes', *CONFIG['HELPER_TIMEFRAMES'],
        '--limit', str(CONFIG['LIMIT_TRAIN']),
        '--label-trials', str(CONFIG['LABEL_TRIALS']),
        '--model-trials', str(CONFIG['MODEL_TRIALS']),
        '--version', CONFIG['VERSION']
    ]

    if CONFIG['DATE_FROM']:
        args.extend(['--date-from', CONFIG['DATE_FROM']])

    if CONFIG['FETCH_MAX_HISTORY']:
        args.append('--fetch-max-history')

    # Run training
    exit_code = run_python_command(args)

    if exit_code == 0:
        print_success(f"{side.upper()} model training completed successfully!")
        print_success(f"  Model saved to: models/{CONFIG['VERSION']}/{CONFIG['TICKER']}_{CONFIG['TIMEFRAME']}_plus_*_{side}/")
    else:
        print_error(f"{side.upper()} model training failed!")


def train_both_models():
    """Train both LONG and SHORT models"""
    print_info("Training BOTH models (LONG + SHORT)")
    print("This will take approximately 4-8 hours...")
    print()

    # Prompt for version once
    if not CONFIG['VERSION']:
        version = prompt_version()
        if not version:
            return
        CONFIG['VERSION'] = version

    # Prompt for parameters once
    CONFIG['TICKER'] = input_with_default("Enter ticker symbol", CONFIG['TICKER']).upper()
    CONFIG['TIMEFRAME'] = input_with_default("Enter timeframe (e.g., 5m, 15m, 1h, 4h)", CONFIG['TIMEFRAME'])

    helpers_input = input_with_default(
        "Enter helper timeframes (space-separated, e.g., '1h 4h')",
        ' '.join(CONFIG['HELPER_TIMEFRAMES'])
    )
    CONFIG['HELPER_TIMEFRAMES'] = helpers_input.split()

    CONFIG['LABEL_TRIALS'] = int(input_with_default("Label trials", str(CONFIG['LABEL_TRIALS'])))
    CONFIG['MODEL_TRIALS'] = int(input_with_default("Model trials", str(CONFIG['MODEL_TRIALS'])))

    # Train LONG first
    train_model('long', skip_prompts=True)
    print()

    # Train SHORT
    train_model('short', skip_prompts=True)


# ========================================
# Analysis Functions
# ========================================

def run_analysis(side: str):
    """Run analysis for LONG or SHORT model"""

    # Prompt for version if not set
    if not CONFIG['VERSION']:
        version = prompt_version()
        if not version:
            return
        CONFIG['VERSION'] = version

    # Auto-detect model parameters
    detected = auto_detect_model_params(CONFIG['VERSION'])

    if detected:
        ticker, timeframe, helpers = detected
        CONFIG['TICKER'] = ticker
        CONFIG['TIMEFRAME'] = timeframe
        CONFIG['HELPER_TIMEFRAMES'] = helpers
    else:
        print_warning("Auto-detection failed. Please enter parameters manually.")
        CONFIG['TICKER'] = input_with_default("Enter ticker symbol", CONFIG['TICKER']).upper()
        CONFIG['TIMEFRAME'] = input_with_default("Enter timeframe", CONFIG['TIMEFRAME'])

        helpers_input = input_with_default(
            "Enter helper timeframes (space-separated)",
            ' '.join(CONFIG['HELPER_TIMEFRAMES'])
        )
        CONFIG['HELPER_TIMEFRAMES'] = helpers_input.split()

    print_info("=" * 60)
    print_info(f"Running Analysis for {side} model (version: {CONFIG['VERSION']})")
    print_info("=" * 60)

    # Build command
    args = [
        'analysis.py',
        '--ticker', CONFIG['TICKER'],
        '--timeframe', CONFIG['TIMEFRAME'],
        '--side', side,
        '--helper-timeframes', *CONFIG['HELPER_TIMEFRAMES'],
        '--version', CONFIG['VERSION']
    ]

    # Run analysis
    exit_code = run_python_command(args)

    if exit_code == 0:
        print_success("Analysis completed successfully!")
    else:
        print_error("Analysis failed!")

    # Reset version for next operation
    CONFIG['VERSION'] = ''


def regenerate_all_analyses():
    """Regenerate ALL analyses for a version"""

    # Prompt for version if not set
    if not CONFIG['VERSION']:
        version = prompt_version()
        if not version:
            return
        CONFIG['VERSION'] = version

    models_dir = Path('models') / CONFIG['VERSION']

    if not models_dir.exists():
        print_error(f"Version directory does not exist: {models_dir}")
        return

    print_info("=" * 60)
    print_info(f"Regenerating ALL analyses for version: {CONFIG['VERSION']}")
    print_info("=" * 60)
    print()

    # Find all model directories (_long and _short)
    model_dirs = sorted(list(models_dir.glob('*_long')) + list(models_dir.glob('*_short')))

    if not model_dirs:
        print_error(f"No model directories found in {models_dir}")
        return

    print(f"{Colors.BLUE}Found models:{Colors.NC}")
    for model_dir in model_dirs:
        print(f"  - {model_dir.name}")
    print()

    count = 0
    success = 0
    failed = 0

    # Process each model directory
    for model_dir in model_dirs:
        dir_name = model_dir.name

        # Remove _long or _short suffix
        base_name = dir_name.replace('_long', '').replace('_short', '')

        # Extract side
        side = 'short' if dir_name.endswith('_short') else 'long'

        # Parse directory name
        pattern = r'^([A-Z]+)_(\d+[mhDW])(.*)$'
        match = re.match(pattern, base_name)

        if match:
            ticker = match.group(1)
            timeframe = match.group(2)
            remainder = match.group(3)

            # Extract helper timeframes
            helpers = []
            if '_plus_' in remainder:
                helpers_str = remainder.split('_plus_')[1]
                helpers = helpers_str.split('_')

            count += 1

            print_info(f"[{count}] Processing: {ticker} {timeframe} {side}")
            print(f"    Helpers: {', '.join(helpers) if helpers else 'none'}")

            # Check if required files exist
            if not (model_dir / 'model.joblib').exists() or not (model_dir / 'holdout_predictions.csv').exists():
                print_error("    ✗ Missing required files (model.joblib or holdout_predictions.csv)")
                failed += 1
                print()
                continue

            # Run analysis
            args = [
                'analysis.py',
                '--ticker', ticker,
                '--timeframe', timeframe,
                '--side', side,
                '--version', CONFIG['VERSION']
            ]

            if helpers:
                args.extend(['--helper-timeframes', *helpers])

            exit_code = run_python_command(args)

            if exit_code == 0:
                print_success("    ✓ Analysis completed")
                success += 1
            else:
                print_error("    ✗ Analysis failed")
                failed += 1

            print()
        else:
            print_warning(f"Could not parse model directory name: {dir_name}")
            failed += 1
            print()

    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BLUE}Summary:{Colors.NC}")
    print(f"  Total models: {count}")
    print(f"  {Colors.GREEN}Success: {success}{Colors.NC}")
    print(f"  {Colors.RED}Failed: {failed}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}")

    # Reset version for next operation
    CONFIG['VERSION'] = ''


# ========================================
# Backtest Functions
# ========================================

def run_backtest():
    """Run backtest"""

    # Prompt for version if not set
    if not CONFIG['VERSION']:
        version = prompt_version()
        if not version:
            return
        CONFIG['VERSION'] = version

    # Auto-detect model parameters
    detected = auto_detect_model_params(CONFIG['VERSION'])

    if detected:
        ticker, timeframe, helpers = detected
        CONFIG['TICKER'] = ticker
        CONFIG['TIMEFRAME'] = timeframe
        CONFIG['HELPER_TIMEFRAMES'] = helpers
    else:
        print_warning("Auto-detection failed. Please enter parameters manually.")
        CONFIG['TICKER'] = input_with_default("Enter ticker symbol", CONFIG['TICKER']).upper()
        CONFIG['TIMEFRAME'] = input_with_default("Enter timeframe", CONFIG['TIMEFRAME'])

        helpers_input = input_with_default(
            "Enter helper timeframes (space-separated)",
            ' '.join(CONFIG['HELPER_TIMEFRAMES'])
        )
        CONFIG['HELPER_TIMEFRAMES'] = helpers_input.split()

    print_info("=" * 60)
    print_info(f"Running Backtest for {CONFIG['TICKER']} {CONFIG['TIMEFRAME']} (version: {CONFIG['VERSION']})")
    print_info("=" * 60)
    print()
    print("Backtest Parameters:")
    print(f"  - Version: {CONFIG['VERSION']}")
    print(f"  - Probability Threshold: {CONFIG['PROB_THRESHOLD']}")
    print(f"  - Min Conf Ratio: {CONFIG['MIN_CONFIDENCE_RATIO']}")
    print(f"  - Take Profit: {CONFIG['TP_PCT']}%")
    print(f"  - Trailing Stop Loss: {CONFIG['TSL_PCT']}%")
    print(f"  - Trade Size: ${CONFIG['TRADE_SIZE']}")
    print(f"  - TP Mechanism: {CONFIG['TP_MECHANISM']}")
    print()

    # Build command
    args = [
        'main.py',
        '--backtest',
        '--ticker', CONFIG['TICKER'],
        '--timeframe', CONFIG['TIMEFRAME'],
        '--helper-timeframes', *CONFIG['HELPER_TIMEFRAMES'],
        '--limit', str(CONFIG['LIMIT_BACKTEST']),
        '--prob-threshold', str(CONFIG['PROB_THRESHOLD']),
        '--min-confidence-ratio', str(CONFIG['MIN_CONFIDENCE_RATIO']),
        '--tp-pct', str(CONFIG['TP_PCT']),
        '--tsl-pct', str(CONFIG['TSL_PCT']),
        '--trade-size', str(CONFIG['TRADE_SIZE']),
        '--version', CONFIG['VERSION'],
        '--partial-tp'
    ]

    # Run backtest
    exit_code = run_python_command(args)

    if exit_code == 0:
        print_success("Backtest completed successfully!")
        print_success(f"  Results saved to: models/{CONFIG['VERSION']}/backtests/")
    else:
        print_error("Backtest failed!")

    # Reset version for next operation
    CONFIG['VERSION'] = ''


# ========================================
# Menu Functions
# ========================================

def show_menu():
    """Display main menu"""
    print(f"{Colors.GREEN}Select an action:{Colors.NC}")
    print("1) Train LONG model")
    print("2) Train SHORT model")
    print("3) Train BOTH models (LONG + SHORT)")
    print("4) Run Analysis (LONG)")
    print("5) Run Analysis (SHORT)")
    print("6) Run Backtest")
    print("7) Regenerate ALL Analyses for Version")
    print("0) Exit")
    print()


def show_config():
    """Show current configuration"""
    print_header("ML Trading Workflow")
    print(f"{Colors.YELLOW}Current Parameters:{Colors.NC}")
    print(f"  VERSION: {CONFIG['VERSION'] or '(will be prompted)'}")
    print(f"  DATE_FROM: {CONFIG['DATE_FROM'] or '(fetch from current date backwards)'}")
    print(f"  FETCH_MAX_HISTORY: {CONFIG['FETCH_MAX_HISTORY']}")
    print(f"  LIMIT_TRAIN: {CONFIG['LIMIT_TRAIN']} (ignored if FETCH_MAX_HISTORY enabled)")
    print(f"  LABEL_TRIALS: {CONFIG['LABEL_TRIALS']}")
    print(f"  MODEL_TRIALS: {CONFIG['MODEL_TRIALS']}")
    print(f"  PROB_THRESHOLD: {CONFIG['PROB_THRESHOLD']}")
    print(f"  MIN_CONFIDENCE_RATIO: {CONFIG['MIN_CONFIDENCE_RATIO']}")
    print(f"  TP_PCT: {CONFIG['TP_PCT']}")
    print(f"  TSL_PCT: {CONFIG['TSL_PCT']}")
    print(f"  TP_MECHANISM: {CONFIG['TP_MECHANISM']}")
    print()


# ========================================
# Main Function
# ========================================

def main():
    """Main entry point"""

    # Show initial config
    show_config()

    # Main loop
    while True:
        show_menu()

        try:
            choice = input(f"Enter your choice [0-7]: ").strip()
            print()

            if choice == '1':
                train_model('long')
            elif choice == '2':
                train_model('short')
            elif choice == '3':
                train_both_models()
            elif choice == '4':
                run_analysis('long')
            elif choice == '5':
                run_analysis('short')
            elif choice == '6':
                run_backtest()
            elif choice == '7':
                regenerate_all_analyses()
            elif choice == '0':
                print(f"{Colors.BLUE}Goodbye!{Colors.NC}")
                sys.exit(0)
            else:
                print_error("Invalid option. Please try again.")

            print()
            input("Press Enter to continue...")
            clear_screen()
            show_config()

        except KeyboardInterrupt:
            print()
            print(f"{Colors.BLUE}Goodbye!{Colors.NC}")
            sys.exit(0)
        except Exception as e:
            print_error(f"An error occurred: {e}")
            print()
            input("Press Enter to continue...")
            clear_screen()
            show_config()


if __name__ == '__main__':
    main()
