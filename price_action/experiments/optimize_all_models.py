#!/usr/bin/env python3
"""
Automated Optuna optimization for all trained models.

This script:
1. Scans models/ directory for all trained models
2. Runs Optuna optimization for each model
3. Saves best parameters to deployment_config.json
4. Generates updated docker-compose.yaml
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import time


class OptunaOptimizer:
    """Automated Optuna optimization for all models"""

    def __init__(self, trials: int = 100, timeout_hours: int = 2):
        """
        Initialize optimizer

        Args:
            trials: Number of Optuna trials per model
            timeout_hours: Maximum hours to wait per optimization
        """
        self.trials = trials
        self.timeout_seconds = timeout_hours * 3600
        self.models_dir = Path("models")
        self.results = []

    def find_all_models(self) -> List[Dict]:
        """
        Find all trained models

        Returns:
            List of dicts with model info
        """
        models = []

        for metadata_file in self.models_dir.glob("**/training_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)

                models.append({
                    "ticker": data.get("ticker"),
                    "timeframe": data.get("timeframe"),
                    "helper_timeframes": data.get("helper_timeframes", []),
                    "version": data.get("version"),
                    "side": data.get("side"),
                    "metadata_path": str(metadata_file)
                })
            except Exception as e:
                print(f"❌ Error reading {metadata_file}: {e}")

        return models

    def run_optuna_for_model(self, model: Dict) -> Optional[Dict]:
        """
        Run Optuna optimization for a single model

        Args:
            model: Model info dict

        Returns:
            Optimization results or None on failure
        """
        ticker = model['ticker']
        timeframe = model['timeframe']
        version = model['version']
        helpers = ' '.join(model['helper_timeframes'])

        print("=" * 80)
        print(f"🎯 Optimizing: {ticker} {timeframe} (version: {version})")
        print("=" * 80)

        # Build command
        cmd = [
            sys.executable,
            "optuna_optimizer.py",
            "--ticker", ticker,
            "--timeframe", timeframe,
            "--version", version,
            "--trials", str(self.trials),
            "--partial-tp"  # Use partial-tp by default
        ]

        if model['helper_timeframes']:
            cmd.extend(["--helper-timeframes"] + model['helper_timeframes'])

        print(f"📝 Command: {' '.join(cmd)}")
        print()

        # Run optimization
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                timeout=self.timeout_seconds,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ Optimization completed successfully in {duration/60:.1f} minutes")
                return {
                    "ticker": ticker,
                    "version": version,
                    "success": True,
                    "duration_minutes": duration / 60,
                    "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
                }
            else:
                print(f"❌ Optimization failed (exit code: {result.returncode})")
                print(f"Error: {result.stderr[-500:]}")
                return {
                    "ticker": ticker,
                    "version": version,
                    "success": False,
                    "error": result.stderr[-500:]
                }

        except subprocess.TimeoutExpired:
            print(f"⏰ Optimization timed out after {self.timeout_seconds/3600:.1f} hours")
            return {
                "ticker": ticker,
                "version": version,
                "success": False,
                "error": "Timeout"
            }
        except Exception as e:
            print(f"❌ Error running optimization: {e}")
            return {
                "ticker": ticker,
                "version": version,
                "success": False,
                "error": str(e)
            }

    def optimize_all(self, skip_existing: bool = True):
        """
        Run Optuna optimization for all models

        Args:
            skip_existing: Skip models that already have deployment_config.json
        """
        print("🔍 Scanning for trained models...")
        models = self.find_all_models()

        # Group by ticker (we optimize per ticker, not per side)
        tickers_seen = set()
        models_to_optimize = []

        for model in models:
            ticker = model['ticker']
            if ticker in tickers_seen:
                continue

            tickers_seen.add(ticker)

            # Check if optimization already exists
            version_dir = self.models_dir / model['version']
            optuna_dir = version_dir / "optuna"
            results_file = list(optuna_dir.glob("optimization_results_*.txt")) if optuna_dir.exists() else []

            if skip_existing and results_file:
                print(f"⏭️  Skipping {ticker} - optimization already exists")
                continue

            models_to_optimize.append(model)

        if not models_to_optimize:
            print("✓ All models already optimized!")
            return

        print(f"\n📊 Found {len(models_to_optimize)} models to optimize")
        print(f"⏱️  Estimated time: {len(models_to_optimize) * 2:.1f} hours (@ 2h per model)")
        print()

        # Ask for confirmation
        response = input("Continue with optimization? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Cancelled.")
            return

        # Run optimization for each model
        for i, model in enumerate(models_to_optimize, 1):
            print(f"\n{'='*80}")
            print(f"Progress: {i}/{len(models_to_optimize)}")
            print(f"{'='*80}\n")

            result = self.run_optuna_for_model(model)
            self.results.append(result)

            # Save intermediate results
            self._save_results()

        # Final summary
        self._print_summary()

    def _save_results(self):
        """Save optimization results to JSON"""
        results_file = Path("optuna_optimization_summary.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")

    def _print_summary(self):
        """Print optimization summary"""
        print("\n" + "=" * 80)
        print("OPTIMIZATION SUMMARY")
        print("=" * 80)

        successful = [r for r in self.results if r.get('success')]
        failed = [r for r in self.results if not r.get('success')]

        print(f"\n✅ Successful: {len(successful)}/{len(self.results)}")
        for r in successful:
            duration = r.get('duration_minutes', 0)
            print(f"   - {r['ticker']}: {duration:.1f} minutes")

        if failed:
            print(f"\n❌ Failed: {len(failed)}/{len(self.results)}")
            for r in failed:
                error = r.get('error', 'Unknown')[:50]
                print(f"   - {r['ticker']}: {error}")

        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("1. Check optimization results in models/*/optuna/")
        print("2. Run: python analyze_all_models.py")
        print("3. Update docker-compose.yaml with new parameters")
        print("4. Restart containers: docker compose down && docker compose up -d")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automated Optuna optimization for all models")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials per model")
    parser.add_argument("--timeout-hours", type=int, default=2, help="Max hours per optimization")
    parser.add_argument("--force", action="store_true", help="Force re-optimization even if results exist")

    args = parser.parse_args()

    optimizer = OptunaOptimizer(
        trials=args.trials,
        timeout_hours=args.timeout_hours
    )

    optimizer.optimize_all(skip_existing=not args.force)
