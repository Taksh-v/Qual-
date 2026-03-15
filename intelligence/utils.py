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

def _valid_answer(text: str) -> bool:
    required = [
        "Executive summary:",
        "Direct answer:",
        "Why this is likely:",
        "Main risks:",
        "What to watch next:",
        "Confidence:",
    ]
    return bool(text) and all(r in text for r in required)


def _sanitize_unsupported_numbers(answer: str, chunks: list[dict[str, Any]]) -> str:
    lines = []
    for raw in (answer or "").splitlines():
        line = raw
        cites = [int(x) for x in re.findall(r"\[S(\d+)\]", line)]
        nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", line)
        if cites and nums:
            cited_text = []
            for c in cites:
                if 1 <= c <= len(chunks):
                    cited_text.append(chunks[c - 1].get("text", ""))
            corpus = " ".join(cited_text)
            for num in nums:
                if num not in corpus:
                    line = re.sub(rf"\b{re.escape(num)}\b", "N/A", line)
        lines.append(line)
    return "\n".join(lines)


def build_fallback_answer(question: str, chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return (
            "Executive summary: Insufficient data from available news.\n"
            "Direct answer: Insufficient data from available news.\n"
            "Why this is likely:\n"
            "- Insufficient data from available news.\n"
            "- Insufficient data from available news.\n"
            "- Insufficient data from available news.\n"
            "Main risks:\n"
            "- Lack of high-quality retrieved evidence.\n"
            "- Potentially stale or incomplete source coverage.\n"
            "What to watch next:\n"
            "- Add more high-relevance sources.\n"
            "- Rebuild index and re-run query.\n"
            "- Validate date and topic coverage.\n"
            "Confidence: LOW - evidence is insufficient."
        )

    lines = [
        f"Executive summary: For '{question}', evidence is limited but indicates a cautious, evidence-first stance.",
        "Direct answer: Use available facts carefully and avoid strong conclusions until stronger evidence appears.",
        "Why this is likely:",
    ]
    for i, chunk in enumerate(chunks[:3], start=1):
        text = " ".join((chunk.get("text") or "").split())
        snippet = text[:180].rstrip()
        lines.append(f"- Source {i} highlights: {snippet} [S{i}]")
    lines += [
        "Main risks:",
        "- Retrieved context may not be fully aligned with the question.",
        "- Some required data points may be missing or stale.",
        "What to watch next:",
        "- Fresh, topic-specific sources.",
        "- New macro/market releases tied to the question.",
        "- Retrieval quality and citation coverage.",
        "Confidence: MEDIUM - usable but limited evidence quality.",
    ]
    return "\n".join(lines)

def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        md = c.get("metadata", {})
        title = md.get("title", "Unknown title")
        source = md.get("source", "Unknown source")
        date = md.get("date", "Unknown date")
        text = c.get("text", "")
        parts.append(f"[S{i}] title={title} | source={source} | date={date}\n{text}")
    return "\n\n".join(parts)


def build_prompt_from_scratch(chunks: list[dict[str, Any]], question: str) -> str:
    context = _format_context(chunks)
    return f"""
You are an expert investment research analyst.
Your reasoning should be expert-level, but your writing should be simple and clear.

Rules:
1. Use only facts from context blocks.
2. Never invent facts, numbers, dates, events, or sources.
3. If evidence is missing, write exactly: "Insufficient data from available news."
4. Add [Sx] citation tags for factual claims.
5. Keep language easy for non-expert readers.

Output format:
Executive summary: <2 short sentences with key takeaway>
Direct answer: <clear recommendation/assessment>
Why this is likely:
- <bullet with evidence [Sx]>
- <bullet with evidence [Sx]>
- <bullet with evidence [Sx] or insufficient data statement>
Main risks:
- <risk 1>
- <risk 2>
What to watch next:
- <item 1>
- <item 2>
- <item 3>
Confidence: <HIGH/MEDIUM/LOW> - <one reason>

Context blocks:
{context}

Question:
{question}

Answer:
""".strip()
