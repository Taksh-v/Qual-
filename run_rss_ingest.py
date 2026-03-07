"""
run_rss_ingest.py
-----------------
End-to-end RSS ingestion pipeline:
  1. Fetch fresh articles from all configured RSS feeds
  2. Chunk each article into sentence windows
  3. Embed chunks with Ollama nomic-embed-text
  4. Upsert into the existing FAISS index + metadata JSON

Usage:
    python run_rss_ingest.py                      # full ingest (all categories)
    python run_rss_ingest.py --category us_macro  # one category only
    python run_rss_ingest.py --priority-only      # high-priority feeds only
    python run_rss_ingest.py --dry-run            # fetch & chunk, skip embedding
    python run_rss_ingest.py --no-skip-seen       # re-ingest already-seen URLs

Schedule this script (cron or systemd timer) to keep the index fresh, e.g.:
    */30 * * * * cd /home/kali/Downloads/Qual && python run_rss_ingest.py --priority-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

# Make sure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.rss_sources import ALL_FEEDS, PRIORITY_FEEDS
from ingestion.rss_fetcher import fetch_all_feeds
from ingestion.chunker import chunk_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "rss")
CHUNK_DIR = os.path.join(BASE_DIR, "data", "chunks", "rss")

INDEX_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "vector_db", "news.index"),
    os.path.join(BASE_DIR, "index", "faiss.index"),
]
METADATA_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "vector_db", "metadata_with_entities.json"),
    os.path.join(BASE_DIR, "data", "vector_db", "metadata.json"),
    os.path.join(BASE_DIR, "index", "metadata.json"),
]

OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT_SEC", "6"))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _load_metadata(path: str) -> list[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_metadata(path: str, data: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _embed(text: str) -> np.ndarray | None:
    try:
        import requests as req
        resp = req.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        vec = resp.json().get("embedding")
        if vec:
            return np.array(vec, dtype="float32")
    except Exception as exc:
        print(f"  [EMBED ERROR] {exc}")
    return None


def _load_faiss_index(path: str):
    try:
        import faiss
        return faiss.read_index(path)
    except Exception as exc:
        print(f"[FAISS] Could not load index from {path}: {exc}")
        return None


def _create_faiss_index(dim: int):
    try:
        import faiss
        idx = faiss.IndexFlatL2(dim)
        return idx
    except Exception as exc:
        print(f"[FAISS] Could not create index: {exc}")
        return None


def _save_faiss_index(index, path: str) -> None:
    import faiss
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_chunks(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chunk all articles and return flat list of chunk dicts."""
    chunks: list[dict[str, Any]] = []
    for art in articles:
        raw = art.get("raw_text") or art.get("title") or ""
        if not raw:
            continue
        texts = chunk_text(raw, chunk_size=500, overlap=75)
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            chunks.append({
                "text": text.strip(),
                "metadata": {
                    "title": art.get("title", ""),
                    "source": art.get("source", ""),
                    "category": art.get("category", ""),
                    "url": art.get("url", ""),
                    "date": art.get("date", ""),
                    "extracted_at": art.get("extracted_at", ""),
                    "chunk_index": i,
                    "chunk_total": len(texts),
                    "data_type": "rss",
                },
            })
    return chunks


def upsert_chunks_to_index(
    chunks: list[dict[str, Any]],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Embed each chunk and add to FAISS index + metadata list."""
    stats: dict[str, Any] = {
        "total_chunks": len(chunks),
        "embedded": 0,
        "skipped": 0,
        "index_path": None,
        "metadata_path": None,
    }

    if dry_run:
        print(f"[DRY RUN] Would embed {len(chunks)} chunks.")
        return stats

    # Resolve index & metadata paths
    index_path = _first_existing(INDEX_CANDIDATES) or INDEX_CANDIDATES[0]
    metadata_path = _first_existing(METADATA_CANDIDATES) or METADATA_CANDIDATES[0]
    stats["index_path"] = index_path
    stats["metadata_path"] = metadata_path

    # Load existing
    metadata = _load_metadata(metadata_path)
    index = _load_faiss_index(index_path) if os.path.exists(index_path) else None
    new_vecs: list[np.ndarray] = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        vec = _embed(text)
        if vec is None:
            stats["skipped"] += 1
            continue

        if index is None:
            index = _create_faiss_index(len(vec))
        if index is None:
            print("[FAISS] Cannot create index — faiss not installed?")
            break

        new_vecs.append(vec)
        metadata.append(chunk)
        stats["embedded"] += 1

        if (i + 1) % 50 == 0:
            elapsed_pct = round((i + 1) / len(chunks) * 100)
            print(f"  [{elapsed_pct}%] Embedded {i+1}/{len(chunks)} chunks...")

    # Batch-add to index
    if new_vecs and index is not None:
        mat = np.stack(new_vecs, axis=0)
        index.add(mat)
        _save_faiss_index(index, index_path)
        _save_metadata(metadata_path, metadata)
        print(f"[INDEX] Saved {len(new_vecs)} new vectors → {index_path}")
        print(f"[META]  Total metadata entries: {len(metadata)} → {metadata_path}")

    return stats


def save_raw_articles(articles: list[dict[str, Any]]) -> None:
    """Persist raw article JSON for audit / re-processing."""
    os.makedirs(RAW_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_path = os.path.join(RAW_DIR, f"batch_{ts}.json")
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"[RAW] Saved {len(articles)} articles → {batch_path}")


def save_chunks(chunks: list[dict[str, Any]]) -> None:
    """Persist chunked data for audit."""
    os.makedirs(CHUNK_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CHUNK_DIR, f"chunks_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[CHUNKS] Saved {len(chunks)} chunks → {path}")


def run_ingestion(
    feeds: list[tuple[str, str, str]],
    dry_run: bool = False,
    skip_seen: bool = True,
    save_raw: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    print(f"\n{'═'*60}")
    print(f"  RSS INGESTION  [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}]")
    print(f"  Feeds: {len(feeds)}  |  dry_run={dry_run}  |  skip_seen={skip_seen}")
    print(f"{'═'*60}\n")

    # Step 1: Fetch
    print("▶ Step 1 / 3: Fetching RSS feeds...")
    articles = fetch_all_feeds(feeds, skip_seen=skip_seen)
    print(f"  → {len(articles)} new articles fetched\n")

    if not articles:
        print("No new articles found. Index is up to date.")
        return {"articles": 0, "chunks": 0, "embedded": 0}

    if save_raw:
        save_raw_articles(articles)

    # Step 2: Chunk
    print("▶ Step 2 / 3: Chunking articles...")
    chunks = build_chunks(articles)
    print(f"  → {len(chunks)} chunks created\n")
    save_chunks(chunks)

    # Step 3: Embed + Index
    print("▶ Step 3 / 3: Embedding and indexing...")
    stats = upsert_chunks_to_index(chunks, dry_run=dry_run)

    elapsed = round(time.time() - t0, 1)
    print(f"\n{'─'*60}")
    print(f"  FINISHED in {elapsed}s")
    print(f"  Articles: {len(articles)}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Embedded: {stats['embedded']}  |  Skipped: {stats['skipped']}")
    if stats.get("index_path"):
        print(f"  Index: {stats['index_path']}")
    print(f"{'═'*60}\n")

    return {
        "articles": len(articles),
        "chunks": len(chunks),
        "embedded": stats["embedded"],
        "skipped": stats["skipped"],
        "elapsed_sec": elapsed,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest RSS feeds into the vector index."
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Ingest only this category (e.g. us_macro, india, commodities).",
    )
    parser.add_argument(
        "--priority-only",
        action="store_true",
        help="Only ingest high-priority market-moving feeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and chunk but do NOT embed or update the index.",
    )
    parser.add_argument(
        "--no-skip-seen",
        action="store_true",
        help="Re-ingest URLs already seen in prior runs.",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Print available categories and exit.",
    )
    args = parser.parse_args()

    from config.rss_sources import RSS_FEEDS

    if args.list_categories:
        print("Available categories:")
        for cat, feeds in RSS_FEEDS.items():
            print(f"  {cat:<20} ({len(feeds)} feeds)")
        return

    if args.category:
        if args.category not in RSS_FEEDS:
            print(f"[ERROR] Unknown category '{args.category}'. Use --list-categories.")
            sys.exit(1)
        feeds = [
            (args.category, lbl, url)
            for lbl, url in RSS_FEEDS[args.category]
        ]
        print(f"Category mode: {args.category} ({len(feeds)} feeds)")
    elif args.priority_only:
        feeds = PRIORITY_FEEDS
        print(f"Priority mode: {len(feeds)} feeds")
    else:
        feeds = ALL_FEEDS
        print(f"Full mode: {len(feeds)} feeds")

    run_ingestion(
        feeds=feeds,
        dry_run=args.dry_run,
        skip_seen=not args.no_skip_seen,
    )


if __name__ == "__main__":
    main()
