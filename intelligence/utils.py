"""
intelligence/utils.py
----------------------
Shared utility functions used across the intelligence and RAG layers.
Centralises code that was previously duplicated in macro_engine.py and rag/query.py.
"""
from __future__ import annotations

import re
from typing import Any


def tokenize(text: str) -> set[str]:
    """Return a set of lowercase word tokens (length > 2) from text."""
    return {t for t in re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()) if len(t) > 2}


def grounding_score(answer: str, chunks: list[dict[str, Any]]) -> float:
    """
    Compute what fraction of cited lines in *answer* are actually grounded
    in their referenced chunk text (token overlap >= 25%).

    Returns a float in [0.0, 1.0].
    """
    cited_lines = [
        line.strip()
        for line in (answer or "").splitlines()
        if "[S" in line
    ]
    if not cited_lines:
        return 0.0

    supported = 0
    for line in cited_lines:
        claim = re.sub(r"\[S\d+\]", "", line)
        claim_tokens = tokenize(claim)
        cites = [int(x) for x in re.findall(r"\[S(\d+)\]", line)]
        if not claim_tokens or not cites:
            continue
        best = 0.0
        for c in cites:
            if 1 <= c <= len(chunks):
                src_tokens = tokenize(chunks[c - 1].get("text", ""))
                if src_tokens:
                    overlap = len(claim_tokens.intersection(src_tokens)) / len(claim_tokens)
                    best = max(best, overlap)
        if best >= 0.25:
            supported += 1
    return supported / max(1, len(cited_lines))


def numeric_hallucination_risk(answer: str, chunks: list[dict[str, Any]]) -> float:
    """
    Estimate the fraction of numeric claims in cited lines that cannot be
    found verbatim in the referenced chunk text.

    Returns a float in [0.0, 1.0]; higher = more hallucination risk.
    """
    total_nums = 0
    missing_nums = 0
    for raw in (answer or "").splitlines():
        line = raw.strip()
        if "[S" not in line:
            continue
        line_wo_cites = re.sub(r"\[S\d+\]", "", line)
        nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", line_wo_cites)
        filtered: list[str] = []
        for n in nums:
            if n.endswith("%"):
                filtered.append(n)
                continue
            if n.replace(".", "").isdigit() and int(float(n)) >= 100:
                filtered.append(n)
        cites = [int(x) for x in re.findall(r"\[S(\d+)\]", line)]
        if not filtered or not cites:
            continue
        corpus = " ".join(
            chunks[c - 1].get("text", "")
            for c in cites
            if 1 <= c <= len(chunks)
        )
        for n in filtered:
            total_nums += 1
            if n not in corpus:
                missing_nums += 1
    if total_nums == 0:
        return 0.0
    return missing_nums / total_nums
