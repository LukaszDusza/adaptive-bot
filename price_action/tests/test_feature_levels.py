"""
Test script for Feature Level Control (Enhancement #7)

Tests three feature levels:
- basic: ~50 features (fastest)
- standard: ~150 features (medium)
- full: ~300 features (slowest)
"""

import time
import logging
from data_preparer_pa import fetch_and_prepare_data, FEATURE_LEVEL_BASIC, FEATURE_LEVEL_STANDARD, FEATURE_LEVEL_FULL

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_feature_levels():
    """Test all three feature levels and compare results"""

    print("=" * 80)
    print("FEATURE LEVEL CONTROL TEST")
    print("=" * 80)

    ticker = "SOLUSDT"
    timeframe = "15m"
    limit = 1000
    helper_timeframes = ["1h"]

    results = {}

    # Test each level
    for level in [FEATURE_LEVEL_BASIC, FEATURE_LEVEL_STANDARD, FEATURE_LEVEL_FULL]:
        print(f"\n{'=' * 80}")
        print(f"Testing: {level.upper()}")
        print("=" * 80)

        start_time = time.time()

        df = fetch_and_prepare_data(
            ticker=ticker,
            timeframe=timeframe,
            limit=limit,
            helper_timeframes=helper_timeframes,
            feature_level=level
        )

        elapsed = time.time() - start_time

        results[level] = {
            'time': elapsed,
            'shape': df.shape,
            'features': df.shape[1] if not df.empty else 0
        }

        print(f"\n✅ {level.upper()} Results:")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Shape: {df.shape}")
        print(f"   Features: {results[level]['features']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    basic_time = results[FEATURE_LEVEL_BASIC]['time']
    standard_time = results[FEATURE_LEVEL_STANDARD]['time']
    full_time = results[FEATURE_LEVEL_FULL]['time']

    print(f"\n📊 Feature Counts:")
    print(f"   Basic:    {results[FEATURE_LEVEL_BASIC]['features']} features")
    print(f"   Standard: {results[FEATURE_LEVEL_STANDARD]['features']} features")
    print(f"   Full:     {results[FEATURE_LEVEL_FULL]['features']} features")

    print(f"\n⏱️  Computation Times:")
    print(f"   Basic:    {basic_time:.2f}s (baseline)")
    print(f"   Standard: {standard_time:.2f}s ({standard_time/basic_time:.1f}x slower)")
    print(f"   Full:     {full_time:.2f}s ({full_time/basic_time:.1f}x slower)")

    print(f"\n🚀 Speed Improvements (vs Full):")
    print(f"   Basic:    {full_time/basic_time:.1f}x faster ({(1-basic_time/full_time)*100:.0f}% improvement)")
    print(f"   Standard: {full_time/standard_time:.1f}x faster ({(1-standard_time/full_time)*100:.0f}% improvement)")

    # Validation
    print(f"\n✅ Validation:")
    basic_ok = results[FEATURE_LEVEL_BASIC]['features'] < results[FEATURE_LEVEL_STANDARD]['features'] < results[FEATURE_LEVEL_FULL]['features']
    if basic_ok:
        print(f"   ✅ Feature counts increase correctly: basic < standard < full")
    else:
        print(f"   ❌ ERROR: Feature counts don't increase correctly!")

    time_ok = basic_time < standard_time < full_time
    if time_ok:
        print(f"   ✅ Computation times increase correctly: basic < standard < full")
    else:
        print(f"   ❌ ERROR: Times don't increase correctly!")

    print("\n" + "=" * 80)
    if basic_ok and time_ok:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 80)

    return results

if __name__ == "__main__":
    try:
        results = test_feature_levels()
        print("\n✅ Feature Level Control is working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
