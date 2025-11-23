# performance_utils.py
"""
Performance optimization utilities.

Provides:
- Memoization decorators for expensive computations
- Vectorized calculations
- Memory-efficient DataFrame operations
- Profiling helpers
"""

import functools
import time
import logging
from typing import Callable, Any, Dict, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Usage:
        @timer
        def slow_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if elapsed > 1.0:
            logger.info(f"⏱️  {func.__name__} took {elapsed:.2f}s")
        elif elapsed > 0.1:
            logger.debug(f"⏱️  {func.__name__} took {elapsed*1000:.0f}ms")

        return result
    return wrapper


def memoize_dataframe(maxsize: int = 128):
    """
    Memoization decorator for functions that return DataFrames.

    Uses DataFrame hash for cache key (content-based caching).

    Args:
        maxsize: Maximum cache size

    Usage:
        @memoize_dataframe(maxsize=64)
        def expensive_calculation(df: pd.DataFrame) -> pd.DataFrame:
            ...
    """
    def decorator(func: Callable) -> Callable:
        cache: Dict[int, Any] = {}
        cache_order = []

        @functools.wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            # Create cache key from DataFrame hash + args
            try:
                df_hash = pd.util.hash_pandas_object(df).sum()
                cache_key = hash((df_hash, args, tuple(sorted(kwargs.items()))))
            except TypeError:
                # Unhashable args - skip caching
                return func(df, *args, **kwargs)

            # Check cache
            if cache_key in cache:
                logger.debug(f"💾 Cache hit for {func.__name__}")
                return cache[cache_key].copy()  # Return copy to prevent mutations

            # Compute and cache
            result = func(df, *args, **kwargs)

            # LRU eviction
            if len(cache) >= maxsize:
                oldest_key = cache_order.pop(0)
                del cache[oldest_key]

            cache[cache_key] = result.copy() if isinstance(result, pd.DataFrame) else result
            cache_order.append(cache_key)

            return result

        # Add cache inspection methods
        wrapper.cache_info = lambda: {
            'hits': len([k for k in cache_order if k in cache]),
            'size': len(cache),
            'maxsize': maxsize
        }
        wrapper.cache_clear = lambda: (cache.clear(), cache_order.clear())

        return wrapper
    return decorator


def vectorized_apply(
    df: pd.DataFrame,
    func: Callable,
    columns: list,
    result_name: str,
    axis: int = 1
) -> pd.Series:
    """
    Optimized DataFrame apply using NumPy vectorization when possible.

    Args:
        df: DataFrame to operate on
        func: Function to apply (should work with NumPy arrays)
        columns: Columns to pass to function
        result_name: Name for result series
        axis: Apply axis (1=rows, 0=columns)

    Returns:
        Series with results

    Example:
        # Instead of:
        df['result'] = df.apply(lambda row: row['a'] + row['b'], axis=1)

        # Use:
        df['result'] = vectorized_apply(
            df, lambda a, b: a + b, ['a', 'b'], 'result'
        )
    """
    try:
        # Try vectorized NumPy operation
        arrays = [df[col].values for col in columns]
        result = func(*arrays)
        return pd.Series(result, index=df.index, name=result_name)
    except (TypeError, ValueError):
        # Fallback to pandas apply
        logger.debug(f"Vectorization failed for {result_name}, using pandas apply")
        return df[columns].apply(lambda row: func(*row), axis=axis)


def inplace_fillna(df: pd.DataFrame, value: Any = 0, columns: Optional[list] = None) -> None:
    """
    Memory-efficient in-place fillna for specific columns.

    Avoids creating DataFrame copy.

    Args:
        df: DataFrame to modify in-place
        value: Fill value
        columns: Columns to fill (None = all columns)
    """
    cols = columns if columns is not None else df.columns

    for col in cols:
        if col in df.columns:
            df[col].fillna(value, inplace=True)


def optimize_dtypes(df: pd.DataFrame, float_type: str = 'float32') -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting dtypes.

    Converts float64 -> float32, int64 -> int32 where safe.

    Args:
        df: DataFrame to optimize
        float_type: Target float type ('float32' or 'float16')

    Returns:
        Optimized DataFrame (in-place modification)
    """
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2

    # Downcast floats
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = df[col].astype(float_type)

    # Downcast ints (check for overflow)
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        col_min = df[col].min()
        col_max = df[col].max()

        if col_min >= -32768 and col_max <= 32767:
            df[col] = df[col].astype('int16')
        elif col_min >= -2147483648 and col_max <= 2147483647:
            df[col] = df[col].astype('int32')

    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    reduction = (1 - final_memory / initial_memory) * 100

    if reduction > 5:
        logger.info(
            f"📉 Memory optimized: {initial_memory:.1f}MB → {final_memory:.1f}MB "
            f"({reduction:.1f}% reduction)"
        )

    return df


def rolling_window_vectorized(
    series: pd.Series,
    window: int,
    func: Callable[[np.ndarray], float]
) -> pd.Series:
    """
    Optimized rolling window calculation using NumPy stride tricks.

    Faster than pandas rolling() for custom functions.

    Args:
        series: Input series
        window: Window size
        func: Function to apply to each window (takes numpy array)

    Returns:
        Series with rolling calculations

    Example:
        # Instead of:
        series.rolling(20).apply(lambda x: custom_func(x))

        # Use:
        rolling_window_vectorized(series, 20, custom_func)
    """
    arr = series.values
    n = len(arr)

    # Create sliding windows using stride tricks
    shape = (n - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    # Apply function to each window
    results = np.array([func(w) for w in windows])

    # Pad with NaN for first (window-1) values
    padded = np.concatenate([np.full(window - 1, np.nan), results])

    return pd.Series(padded, index=series.index, name=series.name)


class PerformanceMonitor:
    """
    Context manager for monitoring performance of code blocks.

    Usage:
        with PerformanceMonitor("Feature calculation"):
            df = calculate_features(df)
    """

    def __init__(self, name: str, log_level: int = logging.INFO):
        self.name = name
        self.log_level = log_level
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time

        if exc_type is None:
            logger.log(
                self.log_level,
                f"✅ {self.name} completed in {elapsed:.2f}s"
            )
        else:
            logger.error(
                f"❌ {self.name} failed after {elapsed:.2f}s: {exc_val}"
            )


def batch_process(
    items: list,
    func: Callable,
    batch_size: int = 100,
    desc: str = "Processing"
) -> list:
    """
    Process items in batches for better memory efficiency.

    Args:
        items: Items to process
        func: Function to apply to each batch
        batch_size: Number of items per batch
        desc: Description for logging

    Returns:
        List of results
    """
    results = []
    n_batches = (len(items) + batch_size - 1) // batch_size

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1

        logger.debug(f"{desc}: batch {batch_num}/{n_batches}")
        batch_result = func(batch)
        results.extend(batch_result if isinstance(batch_result, list) else [batch_result])

    return results


# Pre-compiled regex patterns for common operations
_TIMEFRAME_PATTERNS = {
    'm': 1,
    'h': 60,
    'D': 1440,
    'W': 10080
}


def parse_timeframe_minutes(timeframe: str) -> int:
    """
    Fast timeframe parsing to minutes.

    Optimized version with minimal string operations.

    Args:
        timeframe: Timeframe string (e.g., '15m', '1h', '4h', '1D')

    Returns:
        Minutes per candle
    """
    if not timeframe:
        return 15

    # Fast path for common timeframes
    if timeframe in ('15m', '1h', '4h', '1D'):
        return {'15m': 15, '1h': 60, '4h': 240, '1D': 1440}[timeframe]

    # Parse format: number + suffix
    try:
        suffix = timeframe[-1]
        number = int(timeframe[:-1])
        multiplier = _TIMEFRAME_PATTERNS.get(suffix, 1)
        return number * multiplier
    except (ValueError, KeyError):
        logger.warning(f"Invalid timeframe: {timeframe}, defaulting to 15m")
        return 15


# Vectorized helper functions
def fast_sma(series: pd.Series, window: int) -> pd.Series:
    """
    Fast Simple Moving Average using pandas rolling.

    More efficient than custom implementations.
    """
    return series.rolling(window=window, min_periods=1).mean()


def fast_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Fast Exponential Moving Average using pandas ewm.
    """
    return series.ewm(span=span, adjust=False).mean()


def fast_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Vectorized RSI calculation.

    Faster than loop-based implementations.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi
