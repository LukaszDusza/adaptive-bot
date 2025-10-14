#!/usr/bin/env python3
"""
Quick test to verify Optuna optimizer setup
This runs a mini-optimization with just 5 trials to verify everything works
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from optuna_optimizer import BacktesterOptimizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_quick_test():
    """Run a quick 5-trial test to verify setup"""
    
    print("\n" + "="*70)
    print("OPTUNA SETUP TEST")
    print("="*70)
    print("\nThis will run 5 quick optimization trials to verify everything works.")
    print("If successful, you can run the full optimization with 50-200 trials.\n")
    
    try:
        # Create optimizer with minimal settings
        optimizer = BacktesterOptimizer(
            ticker="SOLUSDT",
            timeframe="1h",
            helper_timeframes=["2h", "4h", "6h", "12h", "1D"],
            limit=500,  # Use less data for quick test
            initial_capital=10000.0,
            risk_pct=0.02,
            enable_partial_tp=True
        )
        
        print("\n✓ Optimizer created successfully")
        print("✓ Models loaded successfully")
        print("✓ Data prepared successfully")
        print("\nRunning 5 test trials...\n")
        
        # Run mini optimization
        results = optimizer.optimize(n_trials=5)
        
        print("\n" + "="*70)
        print("TEST SUCCESSFUL!")
        print("="*70)
        print("\nYour Optuna optimizer is set up correctly!")
        print("\nTo run full optimization:")
        print("1. ./run_solusdt_workflow.sh")
        print("2. Select option 10) Optimize Parameters")
        print("3. Enter desired number of trials (recommended: 100)")
        print("\n" + "="*70 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        print("\nPossible issues:")
        print("1. Models not trained yet - run option 3 in workflow menu first")
        print("2. Missing dependencies - run: pip install optuna")
        print("3. Data fetch error - check your internet connection")
        print("\n" + "="*70 + "\n")
        return False

if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
