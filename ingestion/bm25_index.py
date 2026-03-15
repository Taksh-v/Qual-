"""
ingestion/bm25_index.py
------------------------
BM25 sparse-retrieval index for the Finance RAG pipeline.

WHY THIS EXISTS
---------------
FAISS vector search is excellent at semantic similarity but misses exact keyword
matches (ticker symbols, company names, economic indicators like "PCE", "CPI",
regulatory terms like "8-K", "10-Q").  BM25 catches exactly these.

This module provides:
  1. BM25Index          — wraps rank-bm25.BM25Okapi with persistence
  2. build_bm25_index() — convenience builder from a chunk list
  3. bm25_search()      — returns scored (chunk, score) pairs
  4. reciprocal_rank_fusion() — merges BM25 + FAISS results (the gold-standard
                                 technique for hybrid retrieval)

PERSISTENCE
-----------
The index is pickled to data/vector_db/bm25.pkl so it survives process restarts
without re-tokenizing all chunks.  Rebuild by deleting the pickle file and
re-running run_embedding_index.py.

REQUIREMENTS
------------
    pip install rank-bm25

Falls back gracefully to a no-op stub if rank-bm25 is not installed.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import string
from typing import Any

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_BM25_PATH = os.path.join(_BASE_DIR, "data", "vector_db", "bm25.pkl")

# Common English stop-words (tiny set — financial terms like "rate", "debt" are kept)
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of", "for",
    "and", "or", "but", "by", "with", "this", "that", "as", "are", "was",
    "be", "been", "has", "have", "had", "do", "did", "not", "no", "from",
    "which", "who", "what", "when", "where", "will", "would", "could", "should",
    "its", "their", "our", "we", "they", "he", "she", "his", "her", "all",
    "each", "into", "than", "then", "so", "if", "about", "up", "out", "also",
    "after", "said", "there", "more", "other", "been", "over", "such",
})

_PUNCT_RE = re.compile(r"[\s" + re.escape(string.punctuation) + r"]+")


def _tokenize(text: str) -> list[str]:
    """Lower-case, split on whitespace+punctuation, drop stop-words and 1-char tokens."""
    tokens = _PUNCT_RE.split(text.lower().strip())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


# ── BM25 wrapper ──────────────────────────────────────────────────────────────

class _NullBM25:
    """No-op stub used when rank-bm25 is not installed."""

    def __init__(self) -> None:
        logger.warning(
            "[BM25] rank-bm25 not installed. BM25 retrieval disabled. "
            "Install with: pip install rank-bm25"
        )

    def search(self, query: str, chunks: list[dict], top_k: int = 10) -> list[tuple[dict, float]]:
        return []

    def save(self, path: str) -> None:
        pass


class BM25Index:
    """
    BM25Okapi index over a fixed list of chunks.

    After building, call search() to retrieve (chunk_dict, bm25_score) pairs.
    The index stores chunk references (not copies) so RAM usage is minimal.
    """

    def __init__(self, chunks: list[dict[str, Any]], tokenized: list[list[str]] | None = None) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 not installed. Run: pip install rank-bm25")

        self._chunks = chunks
        self._tokenized = tokenized or [_tokenize(c.get("text", "")) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)
        logger.info("[BM25] Index built over %d chunks", len(chunks))

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict[str, Any], float]]:
        """
        Return the top-k (chunk, score) pairs for *query*.
        Scores are raw BM25 values (higher = more relevant).
        """
        if not self._chunks:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # pair each score with its chunk index, sort descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def save(self, path: str = _DEFAULT_BM25_PATH) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"tokenized": self._tokenized}, f)
        logger.info("[BM25] Index saved to %s", path)


# ── Public helpers ─────────────────────────────────────────────────────────────

def build_bm25_index(chunks: list[dict[str, Any]], save_path: str = _DEFAULT_BM25_PATH) -> BM25Index | _NullBM25:
    """
    Build a BM25Index from a list of chunk dicts and persist it to disk.
    Returns a _NullBM25 stub if rank-bm25 is not installed.
    """
    try:
        idx = BM25Index(chunks)
        idx.save(save_path)
        return idx
    except ImportError:
        return _NullBM25()


def load_bm25_index(chunks: list[dict[str, Any]], pkl_path: str = _DEFAULT_BM25_PATH) -> BM25Index | _NullBM25:
    """
    Load a persisted BM25 index from pkl_path and attach the current chunk list.
    If the pickle doesn't exist or rank-bm25 is missing, returns a _NullBM25.
    """
    try:
        from rank_bm25 import BM25Okapi  # noqa: F401 — just check availability
    except ImportError:
        return _NullBM25()

    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            tokenized = data.get("tokenized")
            if tokenized and len(tokenized) == len(chunks):
                idx = BM25Index(chunks, tokenized=tokenized)
                logger.info("[BM25] Loaded persisted index from %s", pkl_path)
                return idx
            logger.warning("[BM25] Pickle length mismatch — rebuilding index.")
        except Exception as exc:
            logger.warning("[BM25] Could not load pickle: %s — rebuilding.", exc)

    return build_bm25_index(chunks, save_path=pkl_path)


def bm25_search(
    index: BM25Index | _NullBM25,
    query: str,
    top_k: int = 10,
) -> list[tuple[dict[str, Any], float]]:
    """Convenience wrapper handling both real index and null stub."""
    return index.search(query, top_k=top_k)


def reciprocal_rank_fusion(
    semantic_results: list[dict[str, Any]],
    bm25_results: list[tuple[dict[str, Any], float]],
    k: int = 60,
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """
    Merge semantic (FAISS) and sparse (BM25) results using Reciprocal Rank Fusion.

    RRF formula:  RRF(d) = Σ  weight / (k + rank(d))
    where rank is 1-indexed position in each result list.

    Args:
        semantic_results:  Ordered list of chunk dicts from FAISS.
        bm25_results:      (chunk, score) pairs from BM25, ordered by score.
        k:                 RRF smoothing constant (60 is standard).
        semantic_weight:   Weight multiplier for semantic ranking.
        bm25_weight:       Weight multiplier for BM25 ranking.

    Returns:
        Merged list of chunk dicts, highest RRF score first.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict[str, Any]] = {}

    def _chunk_key(chunk: dict[str, Any]) -> str:
        cid = chunk.get("chunk_id") or ""
        if cid:
            return cid
        return (chunk.get("text") or "")[:120]

    # Semantic results (FAISS order = rank 1, 2, 3, ...)
    for rank, chunk in enumerate(semantic_results, start=1):
        key = _chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + semantic_weight / (k + rank)
        chunk_map[key] = chunk

    # BM25 results
    for rank, (chunk, _score) in enumerate(bm25_results, start=1):
        key = _chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + bm25_weight / (k + rank)
        if key not in chunk_map:
            chunk_map[key] = chunk

    # Sort by combined RRF score descending
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    return [chunk_map[k] for k in sorted_keys]
