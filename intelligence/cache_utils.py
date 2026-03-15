"""
Shared caching utilities to prevent cache stampedes and redundancies.
"""
from __future__ import annotations

import time
import asyncio
from typing import Any
from threading import Lock


class _TieredCache:
    """Thread-safe {key: value} store with per-entry TTL."""
    def __init__(self) -> None:
        self._lock = Lock()
        self._data: dict[str, Any] = {}
        self._expires: dict[str, float] = {}

    def get(self, key: str) -> tuple[Any, bool]:
        """Returns (value, is_fresh)."""
        with self._lock:
            if key in self._data and time.time() < self._expires.get(key, 0):
                return self._data[key], True
            return self._data.get(key), False

    def get_stale(self, key: str) -> Any | None:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: Any, ttl: float) -> None:
        with self._lock:
            self._data[key] = value
            self._expires[key] = time.time() + ttl

    def invalidate(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._expires.clear()
            else:
                self._expires.pop(key, None)

    def is_valid(self, key: str) -> bool:
        with self._lock:
            return time.time() < self._expires.get(key, 0)


class AsyncCacheStampedeGuard:
    """
    Prevents cache stampedes by ensuring only one computation
    runs per cache key. Other concurrent requests for the same key
    will await the result of the first computation.
    """
    def __init__(self, cache: _TieredCache, max_size: int = 256):
        self.cache = cache
        self.max_size = max_size
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = Lock()
        
    def _evict_if_needed(self):
        with self.cache._lock:
            if len(self.cache._data) >= self.max_size:
                oldest = min(self.cache._expires, key=lambda k: self.cache._expires[k])
                del self.cache._data[oldest]
                del self.cache._expires[oldest]

    async def get_or_compute(self, key: str, ttl: float, compute_func):
        """
        Get from cache, or compute using `compute_func` if stale/missing.
        """
        val, is_fresh = self.cache.get(key)
        if is_fresh:
            return val

        with self._locks_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            key_lock = self._locks[key]

        async with key_lock:
            # Check again in case another task just computed it
            val, is_fresh = self.cache.get(key)
            if is_fresh:
                return val

            # Compute it
            result = await compute_func()
            
            self._evict_if_needed()
            self.cache.set(key, result, ttl)
            
            # Clean up lock (optional, keep simple for now)
            # with self._locks_lock: ...
            
            return result
