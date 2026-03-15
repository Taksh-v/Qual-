"""
intelligence/shared_embed_cache.py
------------------------------------
Single shared embedding vector cache used by BOTH retrieval pipelines:
  - rag/query.py           (async /ask endpoint)
  - intelligence/context_retriever.py (sync /intelligence endpoint)

Benefits over two separate in-process caches:
  - Avoids duplicate HTTP calls when both pipelines embed the same query string
  - Halves peak memory usage for cached vectors
  - Single eviction policy (LRU-via-insertion-order, configurable size)

Usage:
    from intelligence.shared_embed_cache import get_cached, put_cached, cache_size

    vec = get_cached(text)        # returns np.ndarray copy or None
    if vec is None:
        vec = ... (embed via HTTP) ...
        put_cached(text, vec)

Thread-safety: all public functions are protected by a single reentrant lock.
"""

from __future__ import annotations

import logging
import os
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)

_MAX_SIZE: int = int(os.getenv("EMBED_CACHE_SIZE", "1024"))

_cache: dict[str, np.ndarray] = {}   # ordered by insertion time (Python 3.7+)
_lock: Lock = Lock()


def get_cached(text: str) -> np.ndarray | None:
    """Return a copy of the cached embedding for *text*, or None if not cached."""
    with _lock:
        vec = _cache.get(text)
        return vec.copy() if vec is not None else None


def put_cached(text: str, vec: np.ndarray) -> None:
    """Store *vec* under *text*.  Evicts the oldest entry when the cache is full."""
    with _lock:
        if text in _cache:
            return  # already present — avoid duplicate writes
        if len(_cache) >= _MAX_SIZE:
            # Remove the insertion-order oldest key (first key in dict)
            oldest = next(iter(_cache))
            del _cache[oldest]
        _cache[text] = vec.copy()


def invalidate(text: str) -> bool:
    """Remove a specific key.  Returns True if key existed."""
    with _lock:
        return _cache.pop(text, None) is not None


def clear() -> None:
    """Clear all cached embeddings (e.g. after re-embedding with a new model)."""
    with _lock:
        _cache.clear()
    logger.info("[shared_embed_cache] cache cleared")


def cache_size() -> int:
    """Return current number of cached vectors."""
    with _lock:
        return len(_cache)


def cache_info() -> dict:
    """Diagnostic info for the /health or /metrics endpoint."""
    with _lock:
        return {
            "size":     len(_cache),
            "max_size": _MAX_SIZE,
            "fill_pct": round(len(_cache) / _MAX_SIZE * 100, 1),
        }
