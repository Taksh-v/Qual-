import json
import os
from datetime import datetime
from typing import Any

import faiss
import numpy as np
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "vector_db", "news.index"),
    os.path.join(BASE_DIR, "index", "faiss.index"),
]

METADATA_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "vector_db", "metadata_with_entities.json"),
    os.path.join(BASE_DIR, "data", "vector_db", "metadata.json"),
    os.path.join(BASE_DIR, "index", "metadata.json"),
]

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
EMBED_TIMEOUT_SEC = float(os.getenv("OLLAMA_EMBED_TIMEOUT_SEC", "4"))


def _first_existing(paths: list[str]) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of the paths exist: {paths}")


def _load_json_len(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else -1
    except Exception:
        return -1


def _pick_best_metadata_path(index_ntotal: int) -> str:
    existing = [p for p in METADATA_CANDIDATES if os.path.exists(p)]
    if not existing:
        raise FileNotFoundError(f"None of the paths exist: {METADATA_CANDIDATES}")

    if index_ntotal <= 0:
        return existing[0]

    scored: list[tuple[int, str]] = []
    for path in existing:
        count = _load_json_len(path)
        if count <= 0:
            continue
        scored.append((abs(index_ntotal - count), path))
    if not scored:
        return existing[0]
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _safe_parse_dt(value: Any) -> datetime:
    if not value:
        return datetime.min

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            return datetime.min

    return datetime.min


def embed_query(text: str) -> np.ndarray:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=EMBED_TIMEOUT_SEC,
    )
    response.raise_for_status()

    embedding = response.json().get("embedding")
    if not embedding:
        raise ValueError("Embedding endpoint returned empty embedding")

    return np.array(embedding, dtype="float32").reshape(1, -1)


def _fallback_lexical_context(question: str, metadata: list[dict], top_k: int) -> list[dict]:
    tokens = {tok for tok in question.lower().split() if len(tok) > 2}
    if not tokens:
        return metadata[:top_k]
    scored: list[tuple[int, dict]] = []
    for item in metadata:
        text = (item.get("text") or "").lower()
        if not text:
            continue
        ent_text = " ".join((item.get("metadata", {}).get("entities") or []))
        ent_text = ent_text.lower()
        score = sum(1 for t in tokens if t in text)
        score += sum(2 for t in tokens if t in ent_text)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        return [i for _, i in scored[:top_k]]
    return metadata[:top_k]


def _relevance_score(question: str, item: dict) -> float:
    tokens = {tok for tok in question.lower().split() if len(tok) > 2}
    text = (item.get("text") or "").lower()
    if not text or not tokens:
        return 0.0
    entity_blob = " ".join((item.get("metadata", {}).get("entities") or [])).lower()
    text_hits = sum(1 for t in tokens if t in text)
    ent_hits = sum(1 for t in tokens if t in entity_blob)
    recency_bonus = 0.0
    dt = _safe_parse_dt(item.get("metadata", {}).get("extracted_at"))
    if dt != datetime.min:
        age_days = max((datetime.utcnow() - dt).days, 0)
        recency_bonus = max(0.0, 5.0 - min(age_days / 7.0, 5.0))
    return float(text_hits + (1.5 * ent_hits) + recency_bonus)


def retrieve_relevant_context(question: str, top_k: int = 8, keep_latest: int = 5) -> list[dict]:
    index_path = _first_existing(INDEX_CANDIDATES)
    index = faiss.read_index(index_path)
    metadata_path = _pick_best_metadata_path(index.ntotal)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if index.ntotal == 0 or not metadata:
        return []

    candidates: list[dict] = []
    try:
        query_vec = embed_query(question)
        _, indices = index.search(query_vec, top_k)
        for idx in indices[0]:
            if 0 <= idx < len(metadata):
                candidates.append(metadata[idx])
    except Exception:
        candidates = _fallback_lexical_context(question, metadata, top_k)

    candidates.sort(key=lambda item: _relevance_score(question, item), reverse=True)

    # Deduplicate near-identical chunks (common in repetitive scrape outputs).
    unique: list[dict] = []
    seen: set[str] = set()
    for item in candidates:
        md = item.get("metadata", {})
        title = (md.get("title") or "").strip().lower()
        text_head = (item.get("text") or "").strip().lower()[:220]
        key = f"{title}|{text_head}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique[:keep_latest]


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No indexed custom context available."

    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        md = chunk.get("metadata", {})
        title = md.get("title", "Unknown title")
        source = md.get("source", "Unknown source")
        date = md.get("date", "Unknown date")
        extracted_at = md.get("extracted_at", "Unknown extraction time")
        text = chunk.get("text", "")

        lines.append(
            f"[S{i}] title={title} | source={source} | date={date} | extracted_at={extracted_at}\n{text}"
        )

    return "\n\n".join(lines)
