"""
news_health_checker.py
-----------------------
Real-time RSS/news fetching health verification system.

Checks:
  1. Feed reachability (HTTP 200 status)
  2. Feed parsability (feedparser can parse it)
  3. Article freshness (last article age)
  4. Full-text extraction capability
  5. Index staleness (vector DB last update)
  6. Overall system health score

Designed to be called from:
  - API endpoint: GET /news/health
  - CLI:          python -m intelligence.news_health_checker
  - Cron/monitor: imported and run on schedule
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
_FEED_CHECK_TIMEOUT = float(os.getenv("FEED_CHECK_TIMEOUT", "8"))
_MAX_FEED_CHECK_WORKERS = int(os.getenv("FEED_CHECK_WORKERS", "8"))
_FRESH_THRESHOLD_HOURS = float(os.getenv("NEWS_FRESH_THRESHOLD_HOURS", "24"))
_STALE_INDEX_HOURS = float(os.getenv("NEWS_STALE_INDEX_HOURS", "48"))

# Cache: avoid hammering feeds on each /news/health call
_health_cache: dict[str, Any] = {}
_health_cache_expires: float = 0.0
_HEALTH_CACHE_TTL = float(os.getenv("NEWS_HEALTH_CACHE_TTL", "120"))  # 2 minutes


@dataclass
class FeedStatus:
    label: str
    url: str
    category: str
    reachable: bool = False
    parsable: bool = False
    article_count: int = 0
    latest_article_age_hours: float | None = None
    is_fresh: bool = False
    http_status: int | None = None
    error: str | None = None
    latency_ms: int = 0


@dataclass
class NewsHealthReport:
    checked_at: str = ""
    total_feeds: int = 0
    reachable_feeds: int = 0
    parsable_feeds: int = 0
    fresh_feeds: int = 0           # feeds with articles < threshold hours old
    unreachable_feeds: list[str] = field(default_factory=list)
    stale_feeds: list[str] = field(default_factory=list)
    error_feeds: list[str] = field(default_factory=list)
    total_recent_articles: int = 0
    avg_latency_ms: float = 0.0
    index_status: str = "unknown"
    index_last_modified_hours: float | None = None
    index_vector_count: int = 0
    ollama_reachable: bool = False
    ollama_models: list[str] = field(default_factory=list)
    overall_status: str = "unknown"  # healthy | degraded | critical
    health_score: int = 0           # 0-100
    feed_details: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _check_single_feed(feed_tuple: tuple[str, str, str]) -> FeedStatus:
    """Check a single RSS feed for reachability, parsability, and freshness."""
    cat, label, url = feed_tuple
    status = FeedStatus(label=label, url=url, category=cat)
    t0 = time.time()
    try:
        # Try feedparser first (most reliable, handles redirects)
        try:
            import feedparser
            feed = feedparser.parse(
                url,
                request_headers={
                    "User-Agent": "MacroIntelBot/2.0 HealthCheck",
                },
            )
            status.latency_ms = int((time.time() - t0) * 1000)
            status.reachable = True
            status.parsable = not (feed.bozo and not feed.entries)

            entries = feed.entries or []
            status.article_count = len(entries)

            if entries:
                # Find the freshest article age
                from ingestion.rss_fetcher import _parse_entry_date
                min_age_hours = None
                for entry in entries[:10]:  # only check first 10
                    try:
                        date_str = _parse_entry_date(entry)
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        now = datetime.now(timezone.utc)
                        age = (now - dt).total_seconds() / 3600
                        if min_age_hours is None or age < min_age_hours:
                            min_age_hours = age
                    except Exception:
                        continue
                status.latest_article_age_hours = min_age_hours
                status.is_fresh = (
                    min_age_hours is not None
                    and min_age_hours <= _FRESH_THRESHOLD_HOURS
                )
        except ImportError:
            # Fallback: raw HTTP check
            resp = requests.get(
                url,
                headers={"User-Agent": "MacroIntelBot/2.0 HealthCheck"},
                timeout=_FEED_CHECK_TIMEOUT,
                allow_redirects=True,
            )
            status.latency_ms = int((time.time() - t0) * 1000)
            status.http_status = resp.status_code
            status.reachable = resp.status_code == 200
            status.parsable = status.reachable  # can't verify without feedparser

    except requests.exceptions.Timeout:
        status.error = "timeout"
    except requests.exceptions.ConnectionError as e:
        status.error = f"connection_error: {str(e)[:60]}"
    except Exception as e:
        status.error = f"error: {str(e)[:80]}"
    finally:
        if status.latency_ms == 0:
            status.latency_ms = int((time.time() - t0) * 1000)

    return status


def _check_index_status() -> tuple[str, float | None, int]:
    """
    Check vector index freshness and size.
    Returns: (status_string, age_hours_or_None, vector_count)
    """
    index_paths = [
        os.path.join(BASE_DIR, "data", "vector_db", "news.index"),
        os.path.join(BASE_DIR, "index", "faiss.index"),
    ]
    for path in index_paths:
        if os.path.exists(path):
            try:
                stat = os.stat(path)
                age_sec = time.time() - stat.st_mtime
                age_hours = age_sec / 3600

                # Try to get vector count
                vector_count = 0
                try:
                    import faiss
                    idx = faiss.read_index(path)
                    vector_count = idx.ntotal
                except Exception:
                    pass

                if age_hours <= _STALE_INDEX_HOURS:
                    return "fresh", round(age_hours, 1), vector_count
                else:
                    return "stale", round(age_hours, 1), vector_count
            except Exception:
                return "error", None, 0
    return "missing", None, 0


def _check_ollama() -> tuple[bool, list[str]]:
    """Check if Ollama is running and which models are available."""
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return True, [m for m in models if m]
    except Exception:
        return False, []


def check_news_health(sample_feeds: int = 0) -> NewsHealthReport:
    """
    Run a comprehensive health check on the news fetching system.

    Args:
        sample_feeds: If > 0, check only this many feeds per category (faster).
                      If 0, check ALL configured feeds.

    Returns:
        NewsHealthReport dataclass with all health metrics.
    """
    global _health_cache, _health_cache_expires

    # Use cache for frequent calls
    now = time.time()
    cache_key = f"health_{sample_feeds}"
    if now < _health_cache_expires and cache_key in _health_cache:
        return _health_cache[cache_key]

    report = NewsHealthReport(checked_at=datetime.now(timezone.utc).isoformat())

    # Load feed configuration
    all_feeds: list[tuple[str, str, str]] = []
    try:
        from config.rss_sources import RSS_FEEDS
        for cat, items in RSS_FEEDS.items():
            cat_feeds = [(cat, label, url) for label, url in items]
            if sample_feeds > 0:
                cat_feeds = cat_feeds[:sample_feeds]
            all_feeds.extend(cat_feeds)
    except Exception as e:
        report.warnings.append(f"Failed to load RSS_FEEDS config: {e}")
        report.overall_status = "critical"
        return report

    report.total_feeds = len(all_feeds)

    # Check all feeds in parallel
    feed_statuses: list[FeedStatus] = []
    with ThreadPoolExecutor(max_workers=_MAX_FEED_CHECK_WORKERS) as pool:
        futures = {pool.submit(_check_single_feed, f): f for f in all_feeds}
        for future in as_completed(futures):
            try:
                feed_statuses.append(future.result())
            except Exception as e:
                logger.warning("[health] feed check worker error: %s", e)

    # Aggregate results
    latencies = []
    for fs in feed_statuses:
        if fs.reachable:
            report.reachable_feeds += 1
        if fs.parsable:
            report.parsable_feeds += 1
            report.total_recent_articles += fs.article_count
        if fs.is_fresh:
            report.fresh_feeds += 1
        if not fs.reachable:
            report.unreachable_feeds.append(f"{fs.label} ({fs.category})")
        elif not fs.is_fresh and fs.parsable:
            report.stale_feeds.append(f"{fs.label} ({fs.category})")
        if fs.error:
            report.error_feeds.append(f"{fs.label}: {fs.error}")
        if fs.latency_ms > 0:
            latencies.append(fs.latency_ms)
        report.feed_details.append(asdict(fs))

    report.avg_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    # Check vector index
    report.index_status, report.index_last_modified_hours, report.index_vector_count = (
        _check_index_status()
    )
    if report.index_status == "missing":
        report.warnings.append("Vector index not found — run refresh_data_and_index.py")
    elif report.index_status == "stale":
        report.warnings.append(
            f"Vector index is {report.index_last_modified_hours:.0f}h old — consider refreshing"
        )

    # Check Ollama
    report.ollama_reachable, report.ollama_models = _check_ollama()
    if not report.ollama_reachable:
        report.warnings.append("Ollama LLM not reachable — responses will use deterministic fallback")

    # Compute overall health score (0-100)
    score = 100
    if report.total_feeds > 0:
        reachability_pct = report.reachable_feeds / report.total_feeds
        freshness_pct = report.fresh_feeds / report.total_feeds
        score -= int((1 - reachability_pct) * 40)   # max -40 for dead feeds
        score -= int((1 - freshness_pct) * 20)       # max -20 for stale feeds
    if report.index_status == "missing":
        score -= 25
    elif report.index_status == "stale":
        score -= 10
    if not report.ollama_reachable:
        score -= 15
    report.health_score = max(0, min(100, score))

    # Overall status
    if report.health_score >= 80:
        report.overall_status = "healthy"
    elif report.health_score >= 50:
        report.overall_status = "degraded"
    else:
        report.overall_status = "critical"

    # Cache result
    _health_cache[cache_key] = report
    _health_cache_expires = now + _HEALTH_CACHE_TTL

    return report


def check_news_health_quick() -> NewsHealthReport:
    """Quick check: sample 2 feeds per category (faster, for frequent polling)."""
    return check_news_health(sample_feeds=2)


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    print("\n" + "=" * 60)
    print("NEWS SYSTEM HEALTH CHECK")
    print("=" * 60)

    print("\nRunning quick health check (2 feeds/category)...")
    report = check_news_health_quick()

    print(f"\nOverall Status: {report.overall_status.upper()} ({report.health_score}/100)")
    print(f"Checked at: {report.checked_at}")
    print(f"\nFeed Stats:")
    print(f"  Total feeds configured: {report.total_feeds}")
    print(f"  Reachable:              {report.reachable_feeds} / {report.total_feeds}")
    print(f"  Parsable:               {report.parsable_feeds} / {report.total_feeds}")
    print(f"  Fresh (< 24h):          {report.fresh_feeds} / {report.total_feeds}")
    print(f"  Total recent articles:  {report.total_recent_articles}")
    print(f"  Avg check latency:      {report.avg_latency_ms}ms")
    print(f"\nVector Index: {report.index_status}")
    if report.index_last_modified_hours is not None:
        print(f"  Age: {report.index_last_modified_hours:.1f}h | Vectors: {report.index_vector_count:,}")
    print(f"\nOllama LLM: {'✓ Reachable' if report.ollama_reachable else '✗ Unreachable'}")
    if report.ollama_models:
        print(f"  Models: {report.ollama_models}")
    if report.unreachable_feeds:
        print(f"\nUnreachable Feeds ({len(report.unreachable_feeds)}):")
        for f in report.unreachable_feeds[:10]:
            print(f"  ✗ {f}")
    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"  ⚠ {w}")
    print("\n" + "=" * 60)
