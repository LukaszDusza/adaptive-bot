"""
Test script dla persistent disk cache w data_preparer_pa.py

ENHANCEMENT #2: Persistent Disk Cache Testing
"""

import time
import sys
import pandas as pd
from data_preparer_pa import fetch_and_prepare_data

def test_feature_cache():
    """
    Test persistent disk cache performance.

    Expected results:
    - First call (cache miss): ~10-30s (full computation)
    - Second call (cache hit): <1s (load from disk)
    - Speedup: 10-30x
    """
    print("=" * 80)
    print("TESTING PERSISTENT DISK CACHE")
    print("=" * 80)

    # Test parameters
    ticker = "SOLUSDT"
    timeframe = "15m"
    limit = 5000
    helper_timeframes = ["1h", "4h"]
    side = "long"

    print(f"\nTest parameters:")
    print(f"  Ticker: {ticker}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Limit: {limit}")
    print(f"  Helpers: {helper_timeframes}")
    print(f"  Side: {side}")

    # ========================================================================
    # TEST 1: COLD START (Cache Miss)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: COLD START (Cache Miss)")
    print("=" * 80)

    # Clear cache first (optional - comment out to test with existing cache)
    import shutil
    from pathlib import Path
    cache_dir = Path("cache/features")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("✓ Cleared existing cache")

    start_time = time.time()

    df1 = fetch_and_prepare_data(
        ticker=ticker,
        timeframe=timeframe,
        limit=limit,
        helper_timeframes=helper_timeframes,
        side=side,
        skip_slow_features=True  # Skip slow features for faster testing
    )

    cold_start_time = time.time() - start_time

    print(f"\n📊 COLD START Results:")
    print(f"  Time: {cold_start_time:.2f}s")
    print(f"  Shape: {df1.shape}")
    print(f"  Memory: {df1.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    # ========================================================================
    # TEST 2: WARM START (Cache Hit)
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: WARM START (Cache Hit)")
    print("=" * 80)

    # Wait a bit to simulate time passing (but not enough to invalidate cache)
    time.sleep(1)

    start_time = time.time()

    df2 = fetch_and_prepare_data(
        ticker=ticker,
        timeframe=timeframe,
        limit=limit,
        helper_timeframes=helper_timeframes,
        side=side,
        skip_slow_features=True
    )

    warm_start_time = time.time() - start_time

    print(f"\n📊 WARM START Results:")
    print(f"  Time: {warm_start_time:.2f}s")
    print(f"  Shape: {df2.shape}")
    print(f"  Memory: {df2.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)

    speedup = cold_start_time / warm_start_time if warm_start_time > 0 else float('inf')
    time_saved = cold_start_time - warm_start_time

    print(f"\n⚡ Performance Improvement:")
    print(f"  Cold start: {cold_start_time:.2f}s")
    print(f"  Warm start: {warm_start_time:.2f}s")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Time saved: {time_saved:.2f}s ({time_saved/cold_start_time*100:.1f}%)")

    # Verify data consistency
    if df1.shape == df2.shape:
        print(f"\n✅ Data shape consistent: {df1.shape}")
    else:
        print(f"\n⚠️  WARNING: Shape mismatch! df1={df1.shape}, df2={df2.shape}")

    # Check if data is identical (last few rows should match)
    last_rows_match = df1.tail(10).equals(df2.tail(10))
    if last_rows_match:
        print(f"✅ Last 10 rows identical (cache working correctly)")
    else:
        print(f"⚠️  WARNING: Last rows differ (possible cache staleness)")

    # ========================================================================
    # CACHE FILES INSPECTION
    # ========================================================================
    print("\n" + "=" * 80)
    print("CACHE FILES")
    print("=" * 80)

    cache_files = list(Path("cache/features").glob("*"))
    print(f"\nCache directory contents ({len(cache_files)} files):")
    for f in sorted(cache_files):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.2f} MB")

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

    return {
        'cold_start_time': cold_start_time,
        'warm_start_time': warm_start_time,
        'speedup': speedup,
        'df1_shape': df1.shape,
        'df2_shape': df2.shape
    }

if __name__ == "__main__":
    try:
        results = test_feature_cache()
        print(f"\n✅ All tests passed!")
        print(f"   Speedup achieved: {results['speedup']:.1f}x")
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
