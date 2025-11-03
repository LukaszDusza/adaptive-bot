"""
Cache manager for expensive computations (metrics, analytics)

Implements TTL-based caching to reduce redundant calculations.
"""
from functools import wraps
from typing import Callable, Any, Optional, Dict, Tuple
from datetime import datetime, timedelta
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Simple in-memory cache with TTL support.

    Features:
    - Time-to-live (TTL) expiration
    - Automatic cache invalidation
    - LRU-like behavior (size limit with FIFO eviction)
    """

    def __init__(self, default_ttl: int = 10, max_size: int = 100):
        """
        Initialize cache manager.

        Args:
            default_ttl: Default time-to-live in seconds
            max_size: Maximum number of cached items
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._access_order: list = []  # For LRU eviction

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if valid, None if expired or not found
        """
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]

        # Check if expired
        if datetime.now() > expiry:
            logger.debug(f"Cache expired for key: {key}")
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return None

        # Update access order (move to end - most recently used)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        logger.debug(f"Cache hit for key: {key}")
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        ttl = ttl or self.default_ttl
        expiry = datetime.now() + timedelta(seconds=ttl)

        # Evict oldest if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
            logger.debug(f"Cache full - evicted oldest key: {oldest_key}")

        self._cache[key] = (value, expiry)

        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        logger.debug(f"Cache set for key: {key} (TTL: {ttl}s)")

    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "keys": list(self._cache.keys())
        }


# Global cache instance
_cache_manager = CacheManager(default_ttl=10, max_size=100)


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    return _cache_manager


def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """
    Decorator to cache function results.

    Args:
        ttl: Time-to-live in seconds (uses default if None)
        key_prefix: Prefix for cache key

    Usage:
        @cached(ttl=30, key_prefix="metrics")
        def get_expensive_metrics():
            # ... expensive calculation
            return metrics
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key from function name and args
            cache_key = _generate_cache_key(func.__name__, args, kwargs, key_prefix)

            # Try to get from cache
            cache_manager = get_cache_manager()
            cached_value = cache_manager.get(cache_key)

            if cached_value is not None:
                return cached_value

            # Cache miss - call function
            result = await func(*args, **kwargs)

            # Store in cache
            cache_manager.set(cache_key, result, ttl)

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key from function name and args
            cache_key = _generate_cache_key(func.__name__, args, kwargs, key_prefix)

            # Try to get from cache
            cache_manager = get_cache_manager()
            cached_value = cache_manager.get(cache_key)

            if cached_value is not None:
                return cached_value

            # Cache miss - call function
            result = func(*args, **kwargs)

            # Store in cache
            cache_manager.set(cache_key, result, ttl)

            return result

        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict, prefix: str = "") -> str:
    """
    Generate cache key from function name and arguments.

    Args:
        func_name: Name of the function
        args: Positional arguments
        kwargs: Keyword arguments
        prefix: Optional prefix for the key

    Returns:
        Cache key string
    """
    # Serialize args and kwargs to JSON (sorted for consistency)
    try:
        args_str = json.dumps(args, sort_keys=True, default=str)
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Fallback to string representation if JSON serialization fails
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))

    # Create hash for compact key
    key_data = f"{prefix}:{func_name}:{args_str}:{kwargs_str}"
    key_hash = hashlib.md5(key_data.encode()).hexdigest()

    return f"{prefix}:{func_name}:{key_hash}"
