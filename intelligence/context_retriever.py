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
    os.path.join(BASE_DIR, "data", "vector_db", "metadata.json"),
    os.path.join(BASE_DIR, "index", "metadata.json"),
]

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


def _first_existing(paths: list[str]) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of the paths exist: {paths}")


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
        timeout=60,
    )
    response.raise_for_status()

    embedding = response.json().get("embedding")
    if not embedding:
        raise ValueError("Embedding endpoint returned empty embedding")

    return np.array(embedding, dtype="float32").reshape(1, -1)


def retrieve_relevant_context(question: str, top_k: int = 8, keep_latest: int = 5) -> list[dict]:
    index_path = _first_existing(INDEX_CANDIDATES)
    metadata_path = _first_existing(METADATA_CANDIDATES)

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if index.ntotal == 0 or not metadata:
        return []

    query_vec = embed_query(question)
    _, indices = index.search(query_vec, top_k)

    candidates: list[dict] = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            candidates.append(metadata[idx])

    # Prefer more recent custom data among semantically relevant candidates.
    candidates.sort(
        key=lambda item: _safe_parse_dt(item.get("metadata", {}).get("extracted_at")),
        reverse=True,
    )

    return candidates[:keep_latest]


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
            f"[{i}] title={title} | source={source} | date={date} | extracted_at={extracted_at}\n{text}"
        )

    return "\n\n".join(lines)
