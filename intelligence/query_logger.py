"""
intelligence/query_logger.py
------------------------------
Persistent query + quality logger.

Every call to /ask or /intelligence is appended as a JSON-Line entry to
``data/query_log.jsonl``. This allows offline analysis of:
  - answer quality trends over time
  - model distribution (which model served what % of requests)
  - p95/avg latency per endpoint
  - hallucination / low-quality rate

Log schema per line:
{
  "ts":               "2026-03-09T12:34:56.789Z",
  "endpoint":         "/ask" | "/intelligence/analyze",
  "question":         "...",
  "model_used":       "mistral:latest",
  "latency_ms":       1234,
  "quality_score":    72,
  "quality_band":     "MEDIUM",
  "citation_count":   4,
  "chunk_count":      5,
  "cache_hit":        false,
  "hallucination_risk": 0.05,
  "sentiment_label":  "positive" | "negative" | "neutral" | null,
  "sentiment_score":  0.42,
  "error":            null   # or error message string
}
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from threading import Lock

logger = logging.getLogger(__name__)

_LOG_PATH = os.getenv(
    "QUERY_LOG_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "query_log.jsonl"),
)
_log_lock = Lock()


def _ensure_dir() -> None:
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)


def log_query(
    *,
    endpoint: str,
    question: str,
    model_used: str = "N/A",
    latency_ms: int = 0,
    quality_score: int | None = None,
    quality_band: str | None = None,
    citation_count: int = 0,
    chunk_count: int = 0,
    cache_hit: bool = False,
    hallucination_risk: float | None = None,
    sentiment_label: str | None = None,
    sentiment_score: float | None = None,
    error: str | None = None,
) -> None:
    """Append a single query event to the JSONL log. Thread-safe."""
    entry = {
        "ts":               datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "endpoint":         endpoint,
        "question":         question[:300],
        "model_used":       model_used,
        "latency_ms":       latency_ms,
        "quality_score":    quality_score,
        "quality_band":     quality_band,
        "citation_count":   citation_count,
        "chunk_count":      chunk_count,
        "cache_hit":        cache_hit,
        "hallucination_risk": round(hallucination_risk, 4) if hallucination_risk is not None else None,
        "sentiment_label":  sentiment_label,
        "sentiment_score":  round(sentiment_score, 4) if sentiment_score is not None else None,
        "error":            error,
    }
    try:
        _ensure_dir()
        with _log_lock:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("[query_logger] failed to write log entry: %s", exc)


def read_recent(n: int = 500) -> list[dict]:
    """Return the last *n* log entries (newest first). Used by /metrics."""
    entries: list[dict] = []
    try:
        if not os.path.exists(_LOG_PATH):
            return []
        with _log_lock:
            with open(_LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(entries) >= n:
                break
    except Exception as exc:
        logger.warning("[query_logger] failed to read log: %s", exc)
    return entries


def compute_metrics(entries: list[dict]) -> dict:
    """Compute aggregate metrics from a list of log entries."""
    if not entries:
        return {"message": "No log entries yet"}

    total       = len(entries)
    cache_hits  = sum(1 for e in entries if e.get("cache_hit"))
    errors      = sum(1 for e in entries if e.get("error"))
    latencies   = [e["latency_ms"] for e in entries if e.get("latency_ms")]
    scores      = [e["quality_score"] for e in entries if e.get("quality_score") is not None]

    latencies.sort()
    p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else None
    avg_latency = int(sum(latencies) / len(latencies)) if latencies else None

    # Hallucination rate
    hal_risks = [e["hallucination_risk"] for e in entries if e.get("hallucination_risk") is not None]
    avg_hallucination_risk = round(sum(hal_risks) / len(hal_risks), 4) if hal_risks else None
    high_hal_rate_pct = round(sum(1 for r in hal_risks if r > 0.2) / total * 100, 1) if hal_risks else None

    # Sentiment distribution over served queries
    sentiment_counts: dict[str, int] = {}
    for e in entries:
        label = e.get("sentiment_label") or "unknown"
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

    # Model distribution
    model_counts: dict[str, int] = {}
    for e in entries:
        m = e.get("model_used") or "N/A"
        model_counts[m] = model_counts.get(m, 0) + 1
    model_dist = {m: round(c / total * 100, 1) for m, c in sorted(model_counts.items(), key=lambda x: -x[1])}

    # Quality band distribution
    band_counts: dict[str, int] = {}
    for e in entries:
        b = e.get("quality_band") or "N/A"
        band_counts[b] = band_counts.get(b, 0) + 1

    return {
        "total_queries":          total,
        "cache_hit_rate_pct":     round(cache_hits / total * 100, 1),
        "error_rate_pct":         round(errors / total * 100, 1),
        "avg_quality_score":      round(sum(scores) / len(scores), 1) if scores else None,
        "quality_band_dist":      band_counts,
        "avg_latency_ms":         avg_latency,
        "p95_latency_ms":         p95_latency,
        "model_distribution":     model_dist,
        "avg_hallucination_risk": avg_hallucination_risk,
        "high_hallucination_rate_pct": high_hal_rate_pct,
        "sentiment_distribution": sentiment_counts,
        "log_path":               _LOG_PATH,
    }
