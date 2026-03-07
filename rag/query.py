import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

import faiss
import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from intelligence.utils import tokenize as _tokenize_shared, grounding_score, numeric_hallucination_risk
from intelligence.query_rewriter import rewrite_query
from intelligence.data_quality import evaluate_retrieval_quality, evaluate_vector_store_health
from intelligence.model_router import get_model_candidates

logger = logging.getLogger(__name__)

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

EMBED_MODEL = "nomic-embed-text"
TOP_K = 10
RETURN_K = 8

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_TIMEOUT_SEC = float(os.getenv("OLLAMA_EMBED_TIMEOUT_SEC", "6"))
OLLAMA_GENERATE_TIMEOUT_SEC = float(os.getenv("OLLAMA_GENERATE_TIMEOUT_SEC", "120"))


def _first_existing(paths: list[str]) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


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
        raise FileNotFoundError(f"None of these paths exist: {METADATA_CANDIDATES}")

    if index_ntotal <= 0:
        return existing[0]

    scored = []
    for path in existing:
        count = _load_json_len(path)
        if count <= 0:
            continue
        delta = abs(index_ntotal - count)
        scored.append((delta, path))
    if not scored:
        return existing[0]
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def embed_query(text: str) -> np.ndarray:
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=OLLAMA_EMBED_TIMEOUT_SEC,
    )
    response.raise_for_status()
    payload = response.json()
    emb = payload.get("embedding")
    if not emb:
        raise RuntimeError("Embedding generation failed: empty payload")
    return np.array(emb, dtype="float32").reshape(1, -1)


@lru_cache(maxsize=1)
def load_metadata() -> list[dict[str, Any]]:
    index = load_index()
    metadata_path = _pick_best_metadata_path(index.ntotal)
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_index():
    index_path = _first_existing(INDEX_CANDIDATES)
    return faiss.read_index(index_path)


def invalidate_index_cache() -> None:
    """Clear the in-process FAISS index and metadata caches.
    Call this after rebuilding the index so the API picks up new data
    without a full process restart.
    """
    load_index.cache_clear()
    load_metadata.cache_clear()
    logger.info("[rag.query] FAISS index and metadata caches cleared.")


def _safe_date(md: dict[str, Any]) -> str:
    return str(md.get("date") or md.get("extracted_at") or "")


def _tokenize(text: str) -> set[str]:
    """Delegate to shared intelligence.utils.tokenize."""
    return _tokenize_shared(text)



def dedupe_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = []
    seen = set()
    for chunk in chunks:
        md = chunk.get("metadata", {})
        key = (
            (md.get("title") or "").strip().lower(),
            (chunk.get("text") or "").strip().lower()[:220],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def retrieve_chunks(query_embedding, index, metadata, top_k: int = TOP_K) -> list[dict[str, Any]]:
    _, indices = index.search(query_embedding, top_k)
    chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            chunks.append(metadata[idx])
    chunks = dedupe_chunks(chunks)
    chunks.sort(key=lambda c: _safe_date(c.get("metadata", {})), reverse=True)
    return chunks[:RETURN_K]


def retrieve_chunks_lexical(question: str, metadata: list[dict[str, Any]], top_k: int = RETURN_K) -> list[dict[str, Any]]:
    tokens = {tok for tok in question.lower().split() if len(tok) > 2}
    if not tokens:
        return metadata[:top_k]

    scored = []
    for chunk in metadata:
        text = (chunk.get("text") or "").lower()
        if not text:
            continue
        score = sum(1 for tok in tokens if tok in text)
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return dedupe_chunks(metadata)[:top_k]
    chunks = [c for _, c in scored[: top_k * 3]]
    chunks = dedupe_chunks(chunks)
    chunks.sort(key=lambda c: _safe_date(c.get("metadata", {})), reverse=True)
    return chunks[:top_k]


def retrieve_chunks_hybrid(
    question: str,
    query_embedding: np.ndarray,
    index,
    metadata: list[dict[str, Any]],
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    semantic = retrieve_chunks(query_embedding, index, metadata, top_k=top_k)
    lexical = retrieve_chunks_lexical(question, metadata, top_k=top_k)
    combined = dedupe_chunks(semantic + lexical)
    q_tokens = _tokenize(question)

    def _score(c: dict[str, Any]) -> tuple[int, str]:
        text_tokens = _tokenize(c.get("text", ""))
        overlap = len(q_tokens.intersection(text_tokens))
        return (overlap, _safe_date(c.get("metadata", {})))

    combined.sort(key=_score, reverse=True)
    return combined[:RETURN_K]


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


def build_rewrite_prompt(answer: str) -> str:
    return f"""
Rewrite this answer for clarity and concision.
Do not add new facts or citations.
Keep the same section labels and structure.
Output only revised answer text.

Draft:
{answer}
""".strip()


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


def _grounding_score(answer: str, chunks: list[dict[str, Any]]) -> float:
    """Delegate to shared intelligence.utils.grounding_score."""
    return grounding_score(answer, chunks)


def _numeric_hallucination_risk(answer: str, chunks: list[dict[str, Any]]) -> float:
    """Delegate to shared intelligence.utils.numeric_hallucination_risk."""
    return numeric_hallucination_risk(answer, chunks)


def ask_llm(prompt: str) -> str:
    last_error: Exception | None = None
    for model in get_model_candidates():
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=OLLAMA_GENERATE_TIMEOUT_SEC,
            )
            response.raise_for_status()
            text = response.json().get("response", "")
            if text:
                return text.strip()
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"LLM generation failed across model candidates: {last_error}")


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


def run_query(question: str) -> tuple[str, list[dict[str, Any]]]:
    index = load_index()
    metadata = load_metadata()
    store_health = evaluate_vector_store_health(index.ntotal, metadata)

    optimized_question = rewrite_query(question)
    retrieval_error = None
    chunks = []
    try:
        qvec = embed_query(optimized_question)
        chunks = retrieve_chunks_hybrid(optimized_question, qvec, index, metadata, top_k=TOP_K)
    except Exception as exc:
        retrieval_error = exc
        chunks = retrieve_chunks_lexical(optimized_question, metadata)

    retrieval_health = evaluate_retrieval_quality(question, chunks)

    prompt = build_prompt_from_scratch(chunks, question)
    try:
        if store_health["status"] == "BAD" or retrieval_health["status"] == "BAD":
            raise RuntimeError("Data quality gate blocked generation due to low retrieval confidence.")
        draft = ask_llm(prompt)
        revised = ask_llm(build_rewrite_prompt(draft))
        answer = revised if _valid_answer(revised) else draft
        answer = _sanitize_unsupported_numbers(answer, chunks)
        if (
            not _valid_answer(answer)
            or _grounding_score(answer, chunks) < 0.75
            or _numeric_hallucination_risk(answer, chunks) > 0.2
        ):
            raise RuntimeError("Answer failed required structure")
    except Exception:
        answer = build_fallback_answer(question, chunks)
        if retrieval_error:
            answer += f"\n\nRetrieval fallback used due to: {retrieval_error}"

    quality_note = (
        f"\n\nSystem data quality: store={store_health['status']} "
        f"(index={store_health['index_ntotal']}, metadata={store_health['metadata_count']}), "
        f"retrieval={retrieval_health['status']} (score={retrieval_health['score']}, "
        f"chunks={retrieval_health['chunk_count']}, overlap={retrieval_health['avg_token_overlap']})."
    )
    if store_health["issues"] or retrieval_health["issues"]:
        issues = store_health["issues"] + retrieval_health["issues"]
        quality_note += "\nIssues: " + "; ".join(issues)
    answer += quality_note

    return answer, chunks


def main():
    print("\nFinance RAG ready. Type 'exit' to quit.\n")
    while True:
        q = input("Ask a question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        answer, _ = run_query(q)
        print("\nAnswer:\n")
        print(answer)
        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
