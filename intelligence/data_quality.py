from __future__ import annotations

import re
from typing import Any


def evaluate_vector_store_health(index_ntotal: int, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    count = len(metadata)

    if index_ntotal <= 0:
        issues.append("Vector index has zero entries.")
    if count == 0:
        issues.append("Metadata store is empty.")
    if index_ntotal > 0 and count > 0 and abs(index_ntotal - count) > max(50, int(0.05 * count)):
        issues.append(
            f"Index/metadata mismatch is high (index={index_ntotal}, metadata={count})."
        )

    missing_text = sum(1 for m in metadata if not (m.get("text") or "").strip()) if metadata else 0
    missing_ratio = (missing_text / count) if count else 1.0
    if missing_ratio > 0.1:
        issues.append(f"Too many empty chunks in metadata ({missing_ratio:.1%}).")

    lengths = []
    for m in metadata[: min(5000, count)]:
        text = (m.get("text") or "").strip()
        if text:
            lengths.append(len(text.split()))

    median_len = sorted(lengths)[len(lengths) // 2] if lengths else 0
    if lengths and median_len < 40:
        issues.append(
            f"Median chunk length appears too short ({median_len} words), likely hurting retrieval quality."
        )

    status = "GOOD" if not issues else ("WARN" if len(issues) <= 2 else "BAD")
    return {
        "status": status,
        "issues": issues,
        "index_ntotal": index_ntotal,
        "metadata_count": count,
        "missing_text_ratio": round(missing_ratio, 4),
        "median_chunk_words": median_len,
    }


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()) if len(t) > 2}


def evaluate_retrieval_quality(question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    q_tokens = _tokenize(question)
    if not chunks:
        return {
            "status": "BAD",
            "score": 0,
            "issues": ["No chunks retrieved."],
            "chunk_count": 0,
            "avg_token_overlap": 0.0,
        }

    overlaps: list[float] = []
    for c in chunks:
        c_tokens = _tokenize(c.get("text", ""))
        if not q_tokens or not c_tokens:
            overlaps.append(0.0)
            continue
        overlap = len(q_tokens.intersection(c_tokens)) / max(1, len(q_tokens))
        overlaps.append(overlap)

    avg_overlap = sum(overlaps) / len(overlaps)
    issues = []
    if len(chunks) < 3:
        issues.append("Too few retrieved chunks (<3).")
    if avg_overlap < 0.08:
        issues.append("Low lexical overlap between question and retrieved chunks.")

    score = 100
    score -= max(0, 3 - len(chunks)) * 15
    score -= int(max(0.0, 0.08 - avg_overlap) * 300)
    score = max(5, min(98, score))

    status = "GOOD" if score >= 70 else ("WARN" if score >= 45 else "BAD")
    return {
        "status": status,
        "score": score,
        "issues": issues,
        "chunk_count": len(chunks),
        "avg_token_overlap": round(avg_overlap, 4),
    }
