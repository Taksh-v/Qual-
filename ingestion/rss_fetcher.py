"""
rss_fetcher.py
--------------
Fetches, parses and cleans articles from RSS / Atom feeds.

Features:
- Uses feedparser (no external API key needed)
- Optionally fetches full article body via requests + readability  
- Deduplication by URL hash across sessions (persisted to disk)
- Per-category parallelism with ThreadPoolExecutor
- Graceful fallback if full-text extraction fails (uses feed summary)
- Returns list of dicts compatible with the existing chunker / embedder
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import requests

try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

try:
    from readability import Document
    _HAS_READABILITY = True
except ImportError:
    _HAS_READABILITY = False

from ingestion.utils import clean_text

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEN_URLS_PATH = os.path.join(BASE_DIR, "data", "rss_seen_urls.json")
MAX_ARTICLE_AGE_DAYS = int(os.getenv("RSS_MAX_AGE_DAYS", "7"))
MAX_ARTICLES_PER_FEED = int(os.getenv("RSS_MAX_PER_FEED", "20"))
FETCH_FULL_TEXT = os.getenv("RSS_FETCH_FULL_TEXT", "1") == "1"
REQUEST_TIMEOUT = float(os.getenv("RSS_REQUEST_TIMEOUT", "10"))
MAX_WORKERS = int(os.getenv("RSS_MAX_WORKERS", "6"))
# Drop URL-hash entries older than this from the seen-URL cache (date-based expiry)
SEEN_URL_MAX_AGE_DAYS = int(os.getenv("RSS_SEEN_URL_MAX_AGE_DAYS", "60"))
MIN_TEXT_LEN = 200  # chars — discard very short articles


def _url_hash(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()


def _load_seen_urls() -> set[str]:
    """
    Load seen URL-hashes from disk.
    Storage format: dict {hash: ISO-timestamp} (new) or list (legacy).
    Entries older than SEEN_URL_MAX_AGE_DAYS are silently discarded on load.
    """
    try:
        if os.path.exists(SEEN_URLS_PATH):
            with open(SEEN_URLS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            cutoff = datetime.now(timezone.utc).timestamp() - SEEN_URL_MAX_AGE_DAYS * 86400

            if isinstance(data, list):
                # Legacy format: plain list of hashes — migrate, treat date as 'now'
                logger.info("[RSS] Migrating seen-URL cache to dated format (one-time).")
                return set(data)  # keep all on migration; next save will date-stamp them

            if isinstance(data, dict):
                valid = set()
                for h, ts_str in data.items():
                    try:
                        ts = datetime.fromisoformat(ts_str).timestamp()
                        if ts >= cutoff:
                            valid.add(h)
                    except Exception:
                        valid.add(h)  # keep entries with unparseable timestamps
                return valid
    except Exception:
        pass
    return set()


def _save_seen_urls(seen: set[str]) -> None:
    """
    Persist seen URL-hashes with a timestamp so old entries can be expired.
    Format: {hash: ISO-timestamp-string}.
    """
    try:
        os.makedirs(os.path.dirname(SEEN_URLS_PATH), exist_ok=True)
        # Reload existing dated entries so we don't lose timestamps for old hashes
        existing: dict[str, str] = {}
        if os.path.exists(SEEN_URLS_PATH):
            try:
                with open(SEEN_URLS_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    existing = raw
            except Exception:
                pass

        now_str = datetime.now(timezone.utc).isoformat()
        cutoff = datetime.now(timezone.utc).timestamp() - SEEN_URL_MAX_AGE_DAYS * 86400

        # Merge: add new hashes with current timestamp
        for h in seen:
            if h not in existing:
                existing[h] = now_str

        # Drop stale entries (date-based expiry)
        pruned = {
            h: ts for h, ts in existing.items()
            if _ts_to_unix(ts) >= cutoff
        }

        with open(SEEN_URLS_PATH, "w", encoding="utf-8") as f:
            json.dump(pruned, f)

        logger.debug("[RSS] Seen-URL cache: %d active entries", len(pruned))
    except Exception as exc:
        logger.warning("Could not save seen-URL cache: %s", exc)


def _ts_to_unix(ts_str: str) -> float:
    """Parse an ISO timestamp string to a Unix float; returns 0 on error."""
    try:
        return datetime.fromisoformat(ts_str).timestamp()
    except Exception:
        return 0.0


def _parse_entry_date(entry: Any) -> str:
    """Return ISO date string from a feedparser entry, or today."""
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        ts = getattr(entry, attr, None)
        if ts:
            try:
                return datetime(*ts[:6], tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
    return datetime.now(timezone.utc).isoformat()


def _entry_too_old(entry: Any) -> bool:
    for attr in ("published_parsed", "updated_parsed"):
        ts = getattr(entry, attr, None)
        if ts:
            try:
                pub = datetime(*ts[:6], tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - pub).days
                return age > MAX_ARTICLE_AGE_DAYS
            except Exception:
                pass
    return False  # if no date, keep it


def _fetch_full_text(url: str) -> str | None:
    """Try to extract full article body from the article URL."""
    if not (FETCH_FULL_TEXT and _HAS_BS4):
        return None
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        if _HAS_READABILITY:
            doc = Document(resp.text)
            html = doc.summary()
            soup = BeautifulSoup(html, "lxml")
        else:
            soup = BeautifulSoup(resp.text, "lxml")
        text = "\n".join(p.get_text(separator=" ") for p in soup.find_all("p"))
        text = clean_text(text) if text else ""
        return text if len(text) >= MIN_TEXT_LEN else None
    except Exception:
        return None


def _extract_summary(entry: Any) -> str:
    """Pull best available summary text from a feedparser entry."""
    for attr in ("summary", "summary_detail", "description", "content"):
        val = getattr(entry, attr, None)
        if val:
            if isinstance(val, list):
                val = val[0].get("value", "") if val else ""
            elif isinstance(val, dict):
                val = val.get("value", "")
            if _HAS_BS4:
                soup = BeautifulSoup(val, "lxml")
                val = soup.get_text(separator=" ")
            return clean_text(str(val))
    return ""


def fetch_feed(
    label: str,
    feed_url: str,
    category: str,
    seen_hashes: set[str],
) -> list[dict[str, Any]]:
    """
    Fetch a single RSS feed and return a list of article dicts.
    Each dict has keys: url, title, raw_text, source, category, date, extracted_at
    """
    if not _HAS_FEEDPARSER:
        logger.error("feedparser not installed — run: pip install feedparser")
        return []

    articles: list[dict[str, Any]] = []
    try:
        feed = feedparser.parse(feed_url, request_headers={"User-Agent": "MacroIntelBot/1.0"})
    except Exception as exc:
        logger.warning("[RSS] parse error %s: %s", feed_url, exc)
        return []

    if feed.bozo and not feed.entries:
        logger.debug("[RSS] bozo feed, no entries: %s", feed_url)
        return []

    for entry in feed.entries[:MAX_ARTICLES_PER_FEED]:
        url = getattr(entry, "link", "") or ""
        if not url:
            continue
        h = _url_hash(url)
        if h in seen_hashes:
            continue
        if _entry_too_old(entry):
            continue

        title = clean_text(getattr(entry, "title", "")) or "Untitled"
        date_str = _parse_entry_date(entry)

        # Try full text first; fall back to feed summary
        body = _fetch_full_text(url)
        if not body:
            body = _extract_summary(entry)

        if not body or len(body) < MIN_TEXT_LEN:
            # Skip title-only / ultra-short items; these hurt retrieval quality.
            continue

        if body.strip().lower() == title.strip().lower():
            # Avoid indexing entries where extracted body is just the headline.
            continue

        articles.append({
            "url": url,
            "title": title,
            "raw_text": body,
            "source": label,
            "category": category,
            "date": date_str,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        })
        seen_hashes.add(h)

    logger.info("[RSS] %s: %d new articles", label, len(articles))
    return articles


def fetch_all_feeds(
    feeds: list[tuple[str, str, str]],  # (category, label, url)
    max_workers: int = MAX_WORKERS,
    skip_seen: bool = True,
) -> list[dict[str, Any]]:
    """
    Fetch all feeds in parallel.

    Args:
        feeds:       list of (category, label, feed_url)
        max_workers: thread-pool size
        skip_seen:   skip URLs already fetched in prior runs

    Returns:
        List of article dicts, sorted newest-first.
    """
    seen_hashes = _load_seen_urls() if skip_seen else set()
    all_articles: list[dict[str, Any]] = []
    new_hashes: set[str] = set()

    def _worker(args: tuple[str, str, str]) -> list[dict[str, Any]]:
        cat, lbl, url = args
        return fetch_feed(lbl, url, cat, seen_hashes)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, f): f for f in feeds}
        for future in as_completed(futures):
            try:
                batch = future.result()
                all_articles.extend(batch)
                for art in batch:
                    new_hashes.add(_url_hash(art["url"]))
            except Exception as exc:
                logger.warning("[RSS] feed worker error: %s", exc)

    if skip_seen and new_hashes:
        seen_hashes.update(new_hashes)
        _save_seen_urls(seen_hashes)

    # Sort newest first
    def _sort_key(a: dict[str, Any]) -> str:
        return a.get("date") or ""

    all_articles.sort(key=_sort_key, reverse=True)
    logger.info("[RSS] Total new articles fetched: %d", len(all_articles))
    return all_articles


def fetch_priority_feeds() -> list[dict[str, Any]]:
    """Shortcut: fetch only the highest-priority market-moving feeds."""
    from config.rss_sources import PRIORITY_FEEDS
    return fetch_all_feeds(PRIORITY_FEEDS)


def fetch_category(category: str) -> list[dict[str, Any]]:
    """Fetch all feeds in a specific category."""
    from config.rss_sources import RSS_FEEDS
    feeds_in_cat = [
        (category, label, url)
        for label, url in RSS_FEEDS.get(category, [])
    ]
    return fetch_all_feeds(feeds_in_cat)
