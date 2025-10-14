#!/usr/bin/env python3
"""
Quick script to view the latest Optuna optimization results
"""

import os
import glob
from datetime import datetime

def view_latest_optimization():
    """Display the latest optimization results"""
    
    optuna_dir = "optuna"
    
    if not os.path.exists(optuna_dir):
        print("No optimization results found. Run optimization first!")
        return
    
    # Find all result files
    result_files = glob.glob(f"{optuna_dir}/optimization_results_*.txt")
    
    if not result_files:
        print("No optimization results found. Run optimization first!")
        return
    
    # Get the latest file
    latest_file = max(result_files, key=os.path.getmtime)
    
    print("\n" + "="*70)
    print(f"LATEST OPTIMIZATION RESULTS")
    print("="*70)
    print(f"File: {latest_file}")
    print(f"Modified: {datetime.fromtimestamp(os.path.getmtime(latest_file))}")
    print("="*70 + "\n")
    
    # Display the content
    with open(latest_file, 'r') as f:
        content = f.read()
        print(content)
    
    print("\n" + "="*70)
    print("To use these parameters, update the values in run_solusdt_workflow.sh:")
    print("="*70)
    
    # Extract parameters from file
    with open(latest_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'PROB_THRESHOLD=' in line:
                print(f"  {line.strip()}")
            elif 'TP_PCT=' in line:
                print(f"  {line.strip()}")
            elif 'TSL_PCT=' in line:
                print(f"  {line.strip()}")
    
    print("="*70 + "\n")

def list_all_optimizations():
    """List all available optimization results"""
    
    optuna_dir = "optuna"
    
    if not os.path.exists(optuna_dir):
        print("No optimization results found.")
        return
    
    result_files = glob.glob(f"{optuna_dir}/optimization_results_*.txt")
    
    if not result_files:
        print("No optimization results found.")
        return
    
    print("\n" + "="*70)
    print("ALL OPTIMIZATION RESULTS")
    print("="*70)
    
    for i, file in enumerate(sorted(result_files, key=os.path.getmtime, reverse=True), 1):
        mod_time = datetime.fromtimestamp(os.path.getmtime(file))
        file_name = os.path.basename(file)
        print(f"{i}. {file_name}")
        print(f"   Modified: {mod_time}")
        print()
    
    print("="*70 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        list_all_optimizations()
    else:
        view_latest_optimization()
