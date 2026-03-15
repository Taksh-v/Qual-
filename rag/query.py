import json
import logging
import os
import re
import time
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from threading import Lock
from typing import Any

import faiss
import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from intelligence.utils import (
    tokenize as _tokenize_shared,
    grounding_score as _shared_grounding_score,
    numeric_hallucination_risk as _shared_numeric_hallucination_risk,
)
from intelligence.query_rewriter import rewrite_query, _deterministic_expand
from intelligence.data_quality import evaluate_retrieval_quality, evaluate_vector_store_health
from intelligence.model_router import get_model_candidates
from intelligence.analyst_agents import run_fundamental_analyst, run_sentiment_analyst, run_portfolio_manager

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lazily import BM25 helpers (graceful no-op if rank-bm25 not installed)
try:
    from ingestion.bm25_index import load_bm25_index, reciprocal_rank_fusion as _rrf
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    def _rrf(sem, bm25, **kw):  # type: ignore[misc]
        return sem

# Lazily import SQLite MetadataStore for pre-filtered FAISS search
try:
    from ingestion.metadata_store import get_store as _get_metadata_store
    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False

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
RAG_GENERATE_TIMEOUT_SEC = float(os.getenv("RAG_GENERATE_TIMEOUT_SEC", "12"))
RAG_MAX_MODEL_CANDIDATES = max(1, int(os.getenv("RAG_MAX_MODEL_CANDIDATES", "1")))


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


RAG_ENABLE_QUERY_REWRITE = _env_flag("RAG_ENABLE_QUERY_REWRITE", False)
RAG_ENABLE_ANSWER_REWRITE = _env_flag("RAG_ENABLE_ANSWER_REWRITE", False)
RAG_ENABLE_DETERMINISTIC_REPAIR = _env_flag("RAG_ENABLE_DETERMINISTIC_REPAIR", True)
RAG_ENABLE_COMPACT_RETRY = _env_flag("RAG_ENABLE_COMPACT_RETRY", False)
RAG_CONTEXT_MAX_CHARS = max(320, int(os.getenv("RAG_CONTEXT_MAX_CHARS", "900")))
RAG_COMPACT_CONTEXT_MAX_CHARS = max(180, int(os.getenv("RAG_COMPACT_CONTEXT_MAX_CHARS", "420")))
RAG_COMPACT_MAX_CHUNKS = max(2, int(os.getenv("RAG_COMPACT_MAX_CHUNKS", "3")))
RAG_COMPACT_RETRY_TIMEOUT_SEC = float(os.getenv("RAG_COMPACT_RETRY_TIMEOUT_SEC", "28"))
RAG_COMPACT_NUM_PREDICT = max(80, int(os.getenv("RAG_COMPACT_NUM_PREDICT", "180")))
RAG_MAIN_NUM_PREDICT = max(180, int(os.getenv("RAG_MAIN_NUM_PREDICT", "260")))
RAG_MAIN_TEMPERATURE = float(os.getenv("RAG_MAIN_TEMPERATURE", "0.0"))
RAG_MIN_GROUNDING_SCORE = float(os.getenv("RAG_MIN_GROUNDING_SCORE", "0.75"))
RAG_MAX_NUMERIC_RISK = float(os.getenv("RAG_MAX_NUMERIC_RISK", "0.2"))
RAG_SQLITE_PREFILTER_ROWID_LIMIT = max(200, int(os.getenv("RAG_SQLITE_PREFILTER_ROWID_LIMIT", "2500")))
RAG_SQLITE_PREFILTER_TERMS = max(2, int(os.getenv("RAG_SQLITE_PREFILTER_TERMS", "8")))
RAG_ADAPTIVE_BUDGET = _env_flag("RAG_ADAPTIVE_BUDGET", True)
RAG_ADAPTIVE_TOP_K_MIN = max(RETURN_K, int(os.getenv("RAG_ADAPTIVE_TOP_K_MIN", str(RETURN_K))))
RAG_ADAPTIVE_TOP_K_MAX = max(RAG_ADAPTIVE_TOP_K_MIN, int(os.getenv("RAG_ADAPTIVE_TOP_K_MAX", str(TOP_K + 2))))
RAG_ADAPTIVE_CONTEXT_MIN_CHARS = max(320, int(os.getenv("RAG_ADAPTIVE_CONTEXT_MIN_CHARS", "560")))
RAG_ADAPTIVE_CONTEXT_MAX_CHARS = max(RAG_ADAPTIVE_CONTEXT_MIN_CHARS, int(os.getenv("RAG_ADAPTIVE_CONTEXT_MAX_CHARS", "1200")))
RAG_ANSWER_CACHE_ENABLED = _env_flag("RAG_ANSWER_CACHE_ENABLED", True)
RAG_ANSWER_CACHE_TTL_SEC = float(os.getenv("RAG_ANSWER_CACHE_TTL_SEC", "180"))
RAG_ANSWER_CACHE_MAX_KEYS = max(32, int(os.getenv("RAG_ANSWER_CACHE_MAX_KEYS", "256")))
RAG_RELAX_HARD_FILTER_FALLBACK = _env_flag("RAG_RELAX_HARD_FILTER_FALLBACK", True)

_ANSWER_CACHE: OrderedDict[str, tuple[float, tuple[str, list[dict[str, Any]]]]] = OrderedDict()
_ANSWER_CACHE_LOCK = Lock()


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


def _normalize_cache_question(question: str) -> str:
    terms = _extract_query_terms_in_order(question)
    if terms:
        return " ".join(terms[:24])
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def _relax_hard_filters(filters: dict[str, Any]) -> dict[str, Any]:
    return {
        "hard": {},
        "soft": dict(filters.get("soft") or {}),
        "soft_recent_hours": filters.get("soft_recent_hours"),
    }


def _metadata_version_token(metadata: list[dict[str, Any]]) -> str:
    latest = ""
    for item in metadata:
        md = item.get("metadata", {}) or {}
        ts = str(md.get("indexed_at") or md.get("extracted_at") or md.get("date") or "")
        if ts and ts > latest:
            latest = ts
    return f"{len(metadata)}:{latest}"


def _answer_cache_key(question: str, metadata_version: str) -> str:
    return "|".join(
        [
            _normalize_cache_question(question),
            f"ver={metadata_version}",
            f"rewrite={1 if RAG_ENABLE_QUERY_REWRITE else 0}",
            f"answer_rewrite={1 if RAG_ENABLE_ANSWER_REWRITE else 0}",
            f"compact={1 if RAG_ENABLE_COMPACT_RETRY else 0}",
            f"adaptive={1 if RAG_ADAPTIVE_BUDGET else 0}",
        ]
    )


def _answer_cache_get(key: str) -> tuple[str, list[dict[str, Any]]] | None:
    if not RAG_ANSWER_CACHE_ENABLED:
        return None
    now = time.time()
    with _ANSWER_CACHE_LOCK:
        entry = _ANSWER_CACHE.get(key)
        if not entry:
            return None
        expires_at, payload = entry
        if expires_at <= now:
            _ANSWER_CACHE.pop(key, None)
            return None
        _ANSWER_CACHE.move_to_end(key)
        answer, chunks = payload
        return answer, list(chunks)


def _answer_cache_set(key: str, answer: str, chunks: list[dict[str, Any]]) -> None:
    if not RAG_ANSWER_CACHE_ENABLED:
        return
    now = time.time()
    with _ANSWER_CACHE_LOCK:
        expired = [k for k, (expires_at, _) in _ANSWER_CACHE.items() if expires_at <= now]
        for cache_key in expired:
            _ANSWER_CACHE.pop(cache_key, None)

        _ANSWER_CACHE[key] = (now + max(1.0, RAG_ANSWER_CACHE_TTL_SEC), (answer, list(chunks)))
        _ANSWER_CACHE.move_to_end(key)

        while len(_ANSWER_CACHE) > RAG_ANSWER_CACHE_MAX_KEYS:
            _ANSWER_CACHE.popitem(last=False)


def _clear_answer_cache() -> None:
    with _ANSWER_CACHE_LOCK:
        _ANSWER_CACHE.clear()


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ingestion.embeddings import get_embedding

async def embed_query(text: str) -> np.ndarray:
    emb = get_embedding(text, normalize=True, role="query")
    return emb.reshape(1, -1)


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


@lru_cache(maxsize=1)
def _load_bm25():
    """Load (or build) BM25 index backed by the current metadata list."""
    if not _BM25_AVAILABLE:
        return None
    try:
        metadata = load_metadata()
        return load_bm25_index(metadata)
    except Exception as exc:
        logger.warning("[BM25] Could not load BM25 index: %s", exc)
        return None


def invalidate_index_cache() -> None:
    """Clear the in-process FAISS index, metadata, and BM25 caches.
    Call this after rebuilding the index so the API picks up new data
    without a full process restart.
    """
    load_index.cache_clear()
    load_metadata.cache_clear()
    _load_bm25.cache_clear()
    _clear_answer_cache()
    logger.info("[rag.query] FAISS index, metadata, and BM25 caches cleared.")


def _is_cache_running() -> bool:
    return True

def _safe_date(md: dict[str, Any]) -> str:
    return str(md.get("date") or md.get("extracted_at") or "")


_DATA_TYPE_HINTS_HARD: dict[str, str] = {
    r"\b8-k\b": "sec",
    r"\b10-k\b": "sec",
    r"\b10-q\b": "sec",
    r"\bsec\b": "sec",
    r"\bfiling\b": "sec",
    r"earnings call": "earnings_transcript",
    r"earnings transcript": "earnings_transcript",
    r"\btranscript\b": "earnings_transcript",
}

_DATA_TYPE_HINTS_SOFT: dict[str, str] = {
    r"\bearnings\b": "earnings_transcript",
    r"\bguidance\b": "earnings_transcript",
    r"\bmacro\b": "macro_commentary",
    r"\beconomic\b": "macro_commentary",
    r"\binflation\b": "macro_inflation",
    r"\bcpi\b": "macro_commentary",
    r"\bpce\b": "macro_inflation",
    r"\bgdp\b": "macro_commentary",
    r"\bgrowth\b": "macro_growth",
    r"\bpmi\b": "macro_activity",
    r"\bunemployment\b": "macro_employment",
    r"\bpayroll\b": "macro_employment",
    r"\bfed\b": "macro_commentary",
    r"\binterest rates\b": "macro_commentary",
    r"\brate\b": "macro_rates",
    r"\brates\b": "macro_rates",
    r"\byield\b": "macro_rates",
    r"\btreasury\b": "macro_rates",
    r"\bliquidity\b": "macro_liquidity",
    r"\bcredit\b": "macro_liquidity",
    r"\bspreads\b": "macro_liquidity",
    r"\bvolatility\b": "macro_market",
    r"\bvix\b": "macro_market",
    r"\bresearch\b": "research_report",
    r"analyst report": "research_report",
    r"equity research": "research_report",
    r"\bnews\b": "news",
    r"\bheadline\b": "news",
}


def _dtype_matches_hint(dtype: str, hint: str) -> bool:
    d = (dtype or "").strip().lower()
    h = (hint or "").strip().lower()
    if not d or not h:
        return False
    if d == h:
        return True
    # "macro_commentary" acts as a family hint for macro_* data types.
    if h == "macro_commentary" and d.startswith("macro_"):
        return True
    return False

_REGION_HINTS: dict[str, list[str]] = {
    "US": [r"\bunited states\b", r"\bu\.s\.\b", r"\bus\b", r"\bamerica\b", r"\bfed\b"],
    "Europe": [r"\beurope\b", r"\beurozone\b", r"\becb\b", r"\buk\b", r"\bboe\b"],
    "Asia": [r"\basia\b", r"\bchina\b", r"\bjapan\b", r"\bindia\b", r"\bboj\b", r"\bpboc\b"],
    "Emerging Markets": [r"\bemerging\b", r"\bbrics\b", r"\bbrazil\b", r"\bmexico\b", r"\bturkey\b"],
    "Global": [r"\bglobal\b", r"\bworldwide\b", r"\binternational\b", r"\bg20\b"],
}

_SECTOR_HINTS: dict[str, list[str]] = {
    "Technology": [r"\btech\b", r"\bai\b", r"\bsoftware\b", r"\bsemiconductor\b"],
    "Energy": [r"\boil\b", r"\bgas\b", r"\bbrent\b", r"\bwti\b", r"\bopec\b"],
    "Financials": [r"\bbank\b", r"\bcredit\b", r"\btreasury\b", r"\bbond\b", r"\byield\b"],
    "Healthcare": [r"\bpharma\b", r"\bbiotech\b", r"\bfda\b", r"\bhealthcare\b"],
    "Industrials": [r"\bmanufacturing\b", r"\blogistics\b", r"\baerospace\b"],
    "Consumer": [r"\bretail\b", r"\bconsumer\b", r"\bspending\b"],
    "Materials": [r"\bcopper\b", r"\bsteel\b", r"\bmining\b"],
    "Real Estate": [r"\breal estate\b", r"\breit\b", r"\bhousing\b"],
    "Utilities": [r"\butilities\b", r"\belectricity\b", r"\bgrid\b"],
}


def _tokenize(text: str) -> set[str]:
    """Delegate to shared intelligence.utils.tokenize."""
    return _tokenize_shared(text)


_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can",
    "could", "do", "does", "for", "from", "how", "if", "in", "into",
    "is", "it", "its", "latest", "likely", "may", "might", "more",
    "most", "of", "on", "or", "outlook", "performance", "recent",
    "risk", "should", "than", "that", "the", "their", "them", "these",
    "this", "those", "to", "us", "using", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "change", "changes",
    "driver", "drivers", "effect", "effects", "impact", "impacts",
    "market", "markets", "sudden", "current", "likely", "conditions",
}


def _normalize_term(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def _extract_query_terms_in_order(text: str) -> list[str]:
    ordered: list[str] = []
    for raw in re.findall(r"[a-zA-Z0-9_%&$-]+", (text or "").lower()):
        norm = _normalize_term(raw)
        if len(norm) <= 1 or norm in _QUERY_STOPWORDS:
            continue
        ordered.append(norm)
    return ordered


def _extract_query_tokens(text: str) -> set[str]:
    return set(_extract_query_terms_in_order(text))


def _query_phrases(text: str) -> list[str]:
    ordered = _extract_query_terms_in_order(text)
    phrases: list[str] = []
    for size in (3, 2):
        for i in range(len(ordered) - size + 1):
            phrase = " ".join(ordered[i : i + size])
            if len(phrase) > 4:
                phrases.append(phrase)
    return phrases


def _adaptive_retrieval_budget(question: str) -> tuple[int, int]:
    if not RAG_ADAPTIVE_BUDGET:
        return TOP_K, RAG_CONTEXT_MAX_CHARS

    terms = _extract_query_terms_in_order(question)
    term_count = len(terms)
    lower_q = (question or "").lower()

    complex_query = bool(
        re.search(
            r"\b(compare|versus|vs\.?|between|relative|trade[- ]off|scenario|sensitivity|probabilit|across|multi[- ]factor)\b",
            lower_q,
        )
    ) or term_count >= 8

    simple_query = term_count <= 3 and not complex_query

    top_k = TOP_K
    context_chars = RAG_CONTEXT_MAX_CHARS

    if simple_query:
        top_k = max(RETURN_K, min(TOP_K, RAG_ADAPTIVE_TOP_K_MIN))
        context_chars = max(RAG_ADAPTIVE_CONTEXT_MIN_CHARS, int(RAG_CONTEXT_MAX_CHARS * 0.75))
    elif complex_query:
        top_k = min(max(TOP_K + 2, RETURN_K), RAG_ADAPTIVE_TOP_K_MAX)
        context_chars = min(RAG_ADAPTIVE_CONTEXT_MAX_CHARS, int(RAG_CONTEXT_MAX_CHARS * 1.15))

    return int(top_k), int(context_chars)


def _chunk_search_tokens(chunk: dict[str, Any]) -> set[str]:
    md = _get_metadata(chunk)
    blob = " ".join(
        str(part)
        for part in (
            chunk.get("text") or "",
            md.get("title") or "",
            " ".join(md.get("entities") or []),
            md.get("company") or "",
            md.get("source") or "",
            md.get("sector") or "",
            md.get("region") or "",
            md.get("data_type") or "",
        )
        if part
    )
    return {_normalize_term(tok) for tok in _tokenize(blob)}


def _phrase_score(question: str, chunk: dict[str, Any]) -> float:
    phrases = _query_phrases(question)
    if not phrases:
        return 0.0
    md = _get_metadata(chunk)
    blob = " ".join(
        str(part).lower()
        for part in (
            md.get("title") or "",
            chunk.get("text") or "",
            " ".join(md.get("entities") or []),
        )
        if part
    )
    return float(sum(2.5 for phrase in phrases if phrase in blob))


def _article_key(chunk: dict[str, Any]) -> tuple[str, str, str]:
    md = _get_metadata(chunk)
    return (
        (md.get("title") or "").strip().lower(),
        (md.get("source") or "").strip().lower(),
        (md.get("date") or md.get("extracted_at") or "").strip().lower(),
    )


def _diversify_chunks(chunks: list[dict[str, Any]], limit: int, max_per_article: int = 2) -> list[dict[str, Any]]:
    diversified: list[dict[str, Any]] = []
    article_counts: dict[tuple[str, str, str], int] = {}
    for chunk in chunks:
        key = _article_key(chunk)
        if key[0]:
            count = article_counts.get(key, 0)
            if count >= max_per_article:
                continue
            article_counts[key] = count + 1
        diversified.append(chunk)
        if len(diversified) >= limit:
            break
    return diversified


def _get_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    md = chunk.get("metadata")
    return md if isinstance(md, dict) else {}


def _get_md_value(chunk: dict[str, Any], key: str, default: str = "") -> str:
    md = _get_metadata(chunk)
    val = md.get(key)
    if val:
        return str(val)
    if key in chunk and chunk.get(key):
        return str(chunk.get(key))
    return default


def _safe_parse_dt(value: Any) -> datetime:
    if not value:
        return datetime.min
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00").replace("/", "-")
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            return datetime.min
    return datetime.min


def _parse_time_filters(question: str) -> tuple[datetime | None, bool, int | None]:
    q = question.lower()
    now = datetime.now(timezone.utc)

    explicit = re.search(r"\b(?:last|past|previous)\s+(\d+)\s+(hour|hours|day|days|week|weeks|month|months|year|years)\b", q)
    if explicit:
        n = int(explicit.group(1))
        unit = explicit.group(2)
        hours = n
        if unit.startswith("day"):
            hours = n * 24
        elif unit.startswith("week"):
            hours = n * 24 * 7
        elif unit.startswith("month"):
            hours = n * 24 * 30
        elif unit.startswith("year"):
            hours = n * 24 * 365
        return now - timedelta(hours=hours), True, None

    if re.search(r"\b(today)\b", q):
        return now - timedelta(hours=24), True, None
    if re.search(r"\b(yesterday)\b", q):
        return now - timedelta(hours=48), True, None
    if re.search(r"\b(last week|past week|this week)\b", q):
        return now - timedelta(days=7), True, None
    if re.search(r"\b(last month|past month|this month)\b", q):
        return now - timedelta(days=30), True, None
    if re.search(r"\b(last year|past year|this year)\b", q):
        return now - timedelta(days=365), True, None

    iso = re.search(r"\b(?:since|from)\s+(\d{4}[-/]\d{2}[-/]\d{2})\b", q)
    if iso:
        dt = _safe_parse_dt(iso.group(1))
        if dt is not datetime.min:
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt, True, None

    if re.search(r"\b(recent|latest|most recent|breaking|current)\b", q):
        return None, False, 24 * 30

    return None, False, None


def _extract_filters(question: str) -> dict[str, Any]:
    q = question.lower()
    hard: dict[str, Any] = {}
    soft: dict[str, set[str]] = {}

    hard_types = {dt for pat, dt in _DATA_TYPE_HINTS_HARD.items() if re.search(pat, q)}
    if hard_types:
        hard["data_types"] = hard_types
    else:
        soft_types = {dt for pat, dt in _DATA_TYPE_HINTS_SOFT.items() if re.search(pat, q)}
        if soft_types:
            soft["data_types"] = soft_types

    regions: set[str] = set()
    for region, patterns in _REGION_HINTS.items():
        if any(re.search(pat, q) for pat in patterns):
            regions.add(region)
    if regions:
        soft["regions"] = regions

    sectors: set[str] = set()
    for sector, patterns in _SECTOR_HINTS.items():
        if any(re.search(pat, q) for pat in patterns):
            sectors.add(sector)
    if sectors:
        soft["sectors"] = sectors

    min_dt, hard_time, soft_recent_hours = _parse_time_filters(q)
    if hard_time and min_dt is not None:
        hard["min_dt"] = min_dt

    return {
        "hard": hard,
        "soft": soft,
        "soft_recent_hours": soft_recent_hours,
    }


def _passes_hard_filters(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
    hard = filters.get("hard") or {}
    md = _get_metadata(chunk)

    data_types = hard.get("data_types")
    if data_types:
        dtype = (md.get("data_type") or chunk.get("data_type") or "").lower()
        if not any(_dtype_matches_hint(dtype, hinted) for hinted in data_types):
            return False

    min_dt = hard.get("min_dt")
    if min_dt:
        raw_dt = md.get("date") or md.get("extracted_at") or chunk.get("date") or chunk.get("extracted_at")
        dt = _safe_parse_dt(raw_dt)
        if dt is datetime.min:
            return False
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt < min_dt:
            return False

    return True


def _recency_score(chunk: dict[str, Any]) -> float:
    md = _get_metadata(chunk)
    dt = _safe_parse_dt(md.get("extracted_at") or md.get("date"))
    if dt is datetime.min:
        return 0.0
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_hours = max((now - dt).total_seconds() / 3600, 0)
    if age_hours <= 24:
        return 2.5
    if age_hours <= 72:
        return 1.5
    if age_hours <= 168:
        return 0.9
    if age_hours <= 720:
        return 0.3
    return 0.0


def _title_score(tokens: set[str], chunk: dict[str, Any]) -> float:
    md = _get_metadata(chunk)
    title = md.get("title") or ""
    if not title or not tokens:
        return 0.0
    title_tokens = {_normalize_term(tok) for tok in _tokenize(title)}
    return float(sum(2.0 for t in tokens if t in title_tokens))


def _entity_score(tokens: set[str], chunk: dict[str, Any]) -> float:
    md = _get_metadata(chunk)
    entities = md.get("entities") or []
    if not entities or not tokens:
        return 0.0
    entity_tokens = {_normalize_term(tok) for tok in _tokenize(" ".join(entities))}
    return float(sum(1.6 for t in tokens if t in entity_tokens))


def _soft_filter_boost(chunk: dict[str, Any], filters: dict[str, Any]) -> float:
    soft = filters.get("soft") or {}
    if not soft:
        return 0.0

    md = _get_metadata(chunk)
    score = 0.0

    soft_types = soft.get("data_types")
    if soft_types:
        dtype = (md.get("data_type") or chunk.get("data_type") or "").lower()
        if any(_dtype_matches_hint(dtype, hinted) for hinted in soft_types):
            # Stronger signal for macro family because macro chunks are concise and
            # otherwise under-ranked against long-form news in overlap scoring.
            score += 3.2 if dtype.startswith("macro_") else 2.0

    soft_regions = soft.get("regions")
    if soft_regions:
        region = (md.get("region") or chunk.get("region") or "").strip()
        if region in soft_regions:
            score += 1.5

    soft_sectors = soft.get("sectors")
    if soft_sectors:
        sector = (md.get("sector") or chunk.get("sector") or "").strip()
        if sector in soft_sectors:
            score += 1.5

    return score


def _within_soft_window(chunk: dict[str, Any], hours: int | None) -> bool:
    if not hours:
        return False
    md = _get_metadata(chunk)
    dt = _safe_parse_dt(md.get("extracted_at") or md.get("date"))
    if dt is datetime.min:
        return False
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt >= now - timedelta(hours=hours)


def _rank_score(question: str, chunk: dict[str, Any], filters: dict[str, Any]) -> tuple[float, str]:
    q_tokens = _extract_query_tokens(question)
    text_tokens = {_normalize_term(tok) for tok in _tokenize(chunk.get("text", ""))}
    search_tokens = _chunk_search_tokens(chunk)
    overlap = len(q_tokens.intersection(text_tokens))
    coverage = len(q_tokens.intersection(search_tokens)) / max(1, len(q_tokens))

    score = float(overlap) * 1.8
    score += coverage * 6.0
    score += _recency_score(chunk)
    score += _title_score(q_tokens, chunk)
    score += _entity_score(q_tokens, chunk)
    score += _phrase_score(question, chunk)
    score += _soft_filter_boost(chunk, filters)

    soft_hours = filters.get("soft_recent_hours")
    if soft_hours and _within_soft_window(chunk, soft_hours):
        score += 1.0

    if len(q_tokens) >= 2 and coverage < 0.34:
        score -= 3.0
    if len(q_tokens) >= 3 and coverage < 0.2:
        score -= 4.0

    return (score, _safe_date(_get_metadata(chunk)))



def dedupe_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = []
    seen = set()
    for chunk in chunks:
        md = _get_metadata(chunk)
        key = (
            (md.get("title") or "").strip().lower(),
            (chunk.get("text") or "").strip().lower()[:220],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def _get_faiss_id_filter(
    filters: dict[str, Any] | None,
    question: str | None = None,
) -> tuple["faiss.IDSelector | None", np.ndarray | None]:
    """
    Build a FAISS IDSelectorBatch from hard metadata filters using the SQLite store.
    Returns None if SQLite is unavailable, no filters are set, or no rows match.
    """
    if not _SQLITE_AVAILABLE or not filters:
        return None, None
    hard = filters.get("hard") or {}
    soft = filters.get("soft") or {}

    min_dt = hard.get("min_dt")
    min_date = min_dt.strftime("%Y-%m-%d") if min_dt else None
    data_types = set(hard.get("data_types") or [])
    if not data_types:
        data_types = set(soft.get("data_types") or [])
    regions = set(soft.get("regions") or [])
    sectors = set(soft.get("sectors") or [])

    has_metadata_filter = bool(min_date or data_types or regions or sectors)
    if not has_metadata_filter:
        return None, None

    query_terms = _extract_query_terms_in_order(question or "")[:RAG_SQLITE_PREFILTER_TERMS]
    if len(query_terms) < 2:
        query_terms = []

    try:
        store = _get_metadata_store()
        rowids = store.get_rowids_for_filter(
            min_date=min_date,
            data_types=data_types if data_types else None,
            regions=regions if regions else None,
            sectors=sectors if sectors else None,
            text_terms=query_terms if query_terms else None,
            limit=RAG_SQLITE_PREFILTER_ROWID_LIMIT,
        )
        if not rowids:
            return None, None
        id_array = np.array(sorted(set(rowids)), dtype=np.int64)
        selector = faiss.IDSelectorBatch(len(id_array), faiss.swig_ptr(id_array))
        return selector, id_array
    except Exception as exc:
        logger.debug("[PreFilter] SQLite IDSelector failed: %s", exc)
        return None, None


def retrieve_chunks(
    query_embedding,
    index,
    metadata,
    top_k: int = TOP_K,
    filters: dict[str, Any] | None = None,
    question: str | None = None,
) -> list[dict[str, Any]]:
    n_fetch = min(max(top_k * 5, 20), index.ntotal)

    # Try pre-filtered FAISS ANN search using SQLite rowids
    id_selector, id_selector_keepalive = _get_faiss_id_filter(filters, question=question)
    if id_selector is not None:
        try:
            params = faiss.SearchParameters()
            params.sel = id_selector
            _, indices = index.search(query_embedding, n_fetch, params=params)
        except Exception:
            # Fallback if IDSelector not supported by this FAISS build
            _, indices = index.search(query_embedding, n_fetch)
    else:
        _, indices = index.search(query_embedding, n_fetch)

    _ = id_selector_keepalive  # keep numpy buffer alive until after search

    chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            chunk = metadata[idx]
            if filters and not _passes_hard_filters(chunk, filters):
                continue
            chunks.append(chunk)
    chunks = dedupe_chunks(chunks)
    limit = max(top_k * 3, RETURN_K * 2)
    return chunks[:limit]


def retrieve_chunks_lexical(question: str, metadata: list[dict[str, Any]], top_k: int = RETURN_K, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    tokens = _extract_query_tokens(question)
    if not tokens:
        if filters and filters.get("hard"):
            filtered = [m for m in metadata if _passes_hard_filters(m, filters)]
            return filtered[:top_k]
        return metadata[:top_k]

    scored = []
    for chunk in metadata:
        if filters and not _passes_hard_filters(chunk, filters):
            continue
        search_tokens = _chunk_search_tokens(chunk)
        if not search_tokens:
            continue
        score = sum(1.2 for tok in tokens if tok in search_tokens)
        score += _title_score(tokens, chunk)
        score += _entity_score(tokens, chunk)
        score += _phrase_score(question, chunk)
        score += _soft_filter_boost(chunk, filters or {})
        score += _recency_score(chunk)
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        if filters and filters.get("hard"):
            return []
        return dedupe_chunks(metadata)[:top_k]
    chunks = [c for _, c in scored[: top_k * 5]]
    chunks = dedupe_chunks(chunks)
    return _diversify_chunks(chunks, top_k)


def retrieve_chunks_hybrid(
    question: str,
    query_embedding: np.ndarray,
    index,
    metadata: list[dict[str, Any]],
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval: FAISS semantic + BM25 sparse, merged with
    Reciprocal Rank Fusion (RRF).  Falls back to pure semantic if
    BM25 index is unavailable.
    """
    filters = _extract_filters(question)

    # ── Semantic (FAISS) leg ──────────────────────────────────────────────────
    semantic = retrieve_chunks(query_embedding, index, metadata, top_k=top_k, filters=filters, question=question)

    # ── Sparse (BM25) leg ────────────────────────────────────────────────────
    bm25_idx = _load_bm25()
    bm25_pairs: list[tuple[dict, float]] = []
    if bm25_idx is not None:
        try:
            bm25_pairs = bm25_idx.search(question, top_k=min(max(top_k * 4, 20), len(metadata)))
            # Apply hard filters to BM25 results too
            if filters.get("hard"):
                bm25_pairs = [
                    (c, s) for c, s in bm25_pairs
                    if _passes_hard_filters(c, filters)
                ]
        except Exception as exc:
            logger.warning("[BM25] Search failed: %s", exc)

    # ── Merge with RRF ───────────────────────────────────────────────────────
    if bm25_pairs:
        combined = _rrf(semantic, bm25_pairs, semantic_weight=0.6, bm25_weight=0.4)
    else:
        combined = semantic

    combined = dedupe_chunks(combined)
    rank_filters = filters

    if not combined and filters.get("hard") and RAG_RELAX_HARD_FILTER_FALLBACK:
        rank_filters = _relax_hard_filters(filters)
        relaxed_semantic = retrieve_chunks(
            query_embedding,
            index,
            metadata,
            top_k=top_k,
            filters=rank_filters,
            question=question,
        )

        relaxed_bm25_pairs: list[tuple[dict, float]] = []
        if bm25_idx is not None:
            try:
                relaxed_bm25_pairs = bm25_idx.search(
                    question,
                    top_k=min(max(top_k * 4, 20), len(metadata)),
                )
            except Exception as exc:
                logger.warning("[BM25] Relaxed search failed: %s", exc)

        if relaxed_bm25_pairs:
            combined = _rrf(relaxed_semantic, relaxed_bm25_pairs, semantic_weight=0.6, bm25_weight=0.4)
        else:
            combined = relaxed_semantic
        combined = dedupe_chunks(combined)

    if not combined:
        lexical_fallback = retrieve_chunks_lexical(question, metadata, top_k=max(top_k, RETURN_K), filters=rank_filters)
        combined = dedupe_chunks(lexical_fallback)

    if not combined and filters.get("hard"):
        return []

    combined.sort(key=lambda c: _rank_score(question, c, rank_filters), reverse=True)
    return _diversify_chunks(combined, RETURN_K)


def _try_fetch_fundamentals_context(question: str) -> list[dict[str, Any]]:
    pass # Placeholder for the missing function body

def _format_context(chunks: list[dict[str, Any]], max_chars: int = RAG_CONTEXT_MAX_CHARS) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        md = c.get("metadata", {})
        title = md.get("title", "Unknown title")
        source = md.get("source", "Unknown source")
        date = md.get("date", "Unknown date")
        raw_text = str(c.get("text", "") or "")
        if len(raw_text) > max_chars:
            clipped = raw_text[:max_chars].rsplit(" ", 1)[0].strip()
            text = f"{clipped} ..." if clipped else raw_text[:max_chars]
        else:
            text = raw_text
        # Include sentiment so the LLM can reason about market tone
        sentiment_label = c.get("sentiment_label") or md.get("sentiment_label", "")
        sentiment_score  = c.get("sentiment") or md.get("sentiment", 0.0)
        sent_str = f" | sentiment={sentiment_label}({sentiment_score:+.2f})" if sentiment_label else ""
        parts.append(
            f"[S{i}] title={title} | source={source} | date={date}{sent_str}\n{text}"
        )
    return "\n\n".join(parts)


def build_prompt_from_scratch(
    chunks: list[dict[str, Any]],
    question: str,
    context_max_chars: int = RAG_CONTEXT_MAX_CHARS,
) -> str:
    context = _format_context(chunks, max_chars=context_max_chars)
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


def _format_compact_context(chunks: list[dict[str, Any]], max_chunks: int = RAG_COMPACT_MAX_CHUNKS) -> str:
    parts = []
    for i, c in enumerate(chunks[:max_chunks], start=1):
        md = c.get("metadata", {})
        title = md.get("title", "Unknown title")
        source = md.get("source", "Unknown source")
        date = md.get("date", "Unknown date")
        raw_text = str(c.get("text", "") or "")
        if len(raw_text) > RAG_COMPACT_CONTEXT_MAX_CHARS:
            clipped = raw_text[:RAG_COMPACT_CONTEXT_MAX_CHARS].rsplit(" ", 1)[0].strip()
            text = f"{clipped} ..." if clipped else raw_text[:RAG_COMPACT_CONTEXT_MAX_CHARS]
        else:
            text = raw_text
        parts.append(f"[S{i}] {title} | {source} | {date}\n{text}")
    return "\n\n".join(parts)


def build_compact_prompt(chunks: list[dict[str, Any]], question: str) -> str:
    context = _format_compact_context(chunks)
    return f"""
You are an expert investment research analyst.
Write a concise, well-structured answer using only the evidence snippets.

Rules:
1. Use only the provided evidence.
2. Do not invent facts, numbers, dates, or sources.
3. Keep each section short.
4. Add [Sx] citations on factual evidence lines.
5. If evidence is limited, state uncertainty plainly.

Output format:
Executive summary: <1-2 short sentences>
Direct answer: <1 short sentence>
Why this is likely:
- <bullet with evidence [Sx]>
- <bullet with evidence [Sx]>
- <bullet with evidence [Sx] or Insufficient data from available news.>
Main risks:
- <risk 1>
- <risk 2>
What to watch next:
- <item 1>
- <item 2>
- <item 3>
Confidence: <HIGH/MEDIUM/LOW> - <one reason>

Evidence snippets:
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
    return _shared_grounding_score(answer, chunks)


def _numeric_hallucination_risk(answer: str, chunks: list[dict[str, Any]]) -> float:
    return _shared_numeric_hallucination_risk(answer, chunks)


def ask_llm(
    prompt: str,
    *,
    timeout_sec: float | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    last_error: Exception | None = None
    timeout = timeout_sec if timeout_sec is not None else RAG_GENERATE_TIMEOUT_SEC
    for model in get_model_candidates()[:RAG_MAX_MODEL_CANDIDATES]:
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            text = (response.json().get("response") or "").strip()
            if text:
                return text
            last_error = RuntimeError(f"{model}:empty_response")
        except Exception as exc:
            last_error = RuntimeError(f"{model}:{exc.__class__.__name__}:{exc}")
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


def _extract_sentences(text: str, max_sentences: int = 3) -> list[str]:
    flat = re.sub(r"\s+", " ", (text or "").strip())
    if not flat:
        return []
    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+", flat) if p.strip()]
    return pieces[:max_sentences]


def _extract_cited_bullets(text: str, chunk_count: int, max_items: int = 3) -> list[str]:
    bullets: list[str] = []
    seen: set[str] = set()
    for raw in (text or "").splitlines():
        cleaned = raw.strip()
        if not cleaned:
            continue
        if cleaned.startswith(("-", "*")):
            cleaned = cleaned[1:].strip()
        cites = [int(x) for x in re.findall(r"\[S(\d+)\]", cleaned)]
        if not cites:
            continue
        if not any(1 <= c <= chunk_count for c in cites):
            continue
        normalized = re.sub(r"\s+", " ", cleaned)
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        bullets.append(f"- {normalized}")
        if len(bullets) >= max_items:
            break
    return bullets


def _evidence_bullets_from_chunks(chunks: list[dict[str, Any]], max_items: int = 3) -> list[str]:
    bullets: list[str] = []
    for i, chunk in enumerate(chunks[:max_items], start=1):
        text = " ".join((chunk.get("text") or "").split())
        if not text:
            continue
        snippet = text[:180].rstrip()
        bullets.append(f"- {snippet} [S{i}]")
    return bullets


def _extract_best_chunk_sentences(
    question: str,
    chunks: list[dict[str, Any]],
    max_candidates: int = 5,
    max_sentences_per_chunk: int = 6,
) -> tuple[str, str]:
    """Return (summary_sentence, direct_sentence) extracted from top chunks."""
    if not chunks:
        return "", ""
    q_lower = question.lower()
    q_terms = {t for t in re.sub(r"[^a-z0-9\s]", " ", q_lower).split() if len(t) > 3}
    candidates: list[tuple[int, str]] = []
    for chunk in chunks[:max_candidates]:
        text = str(chunk.get("text") or "")
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 35]
        for sent in sents[:max_sentences_per_chunk]:
            clean = re.sub(r"\s+", " ", sent).strip()
            if len(clean) > 200:
                clean = clean[:200].rsplit(" ", 1)[0].rstrip() + "..."
            low = clean.lower()
            overlap = sum(1 for t in q_terms if t in low)
            has_number = 1 if re.search(r"\b\d+(?:\.\d+)?%?\b", clean) else 0
            score = overlap * 2 + has_number
            if score > 0:
                candidates.append((score, clean))
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    unique: list[str] = []
    for _, c in candidates:
        key = c[:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
        if len(unique) >= 2:
            break
    summary_sent = unique[0] if unique else ""
    direct_sent = unique[1] if len(unique) > 1 else ""
    return summary_sent, direct_sent


def _repair_answer_structure(
    question: str,
    draft: str,
    chunks: list[dict[str, Any]],
    retrieval_health: dict[str, Any],
) -> str:
    sentences = _extract_sentences(draft, max_sentences=3)

    # When the LLM draft is empty or generic, extract real evidence sentences from chunks
    chunk_summary, chunk_direct = _extract_best_chunk_sentences(question, chunks)

    summary = (
        sentences[0]
        if sentences
        else (
            chunk_summary
            if chunk_summary
            else f"For '{question}', available evidence provides directional context but requires careful interpretation."
        )
    )
    direct = (
        sentences[1]
        if len(sentences) > 1
        else (
            chunk_direct
            if chunk_direct
            else "Based on retrieved evidence, monitor the latest releases for a definitive signal."
        )
    )

    why_lines = _extract_cited_bullets(draft, len(chunks), max_items=3)
    if len(why_lines) < 3:
        for bullet in _evidence_bullets_from_chunks(chunks, max_items=3):
            if bullet not in why_lines:
                why_lines.append(bullet)
            if len(why_lines) >= 3:
                break
    while len(why_lines) < 3:
        why_lines.append("- Insufficient data from available news.")

    risks: list[str] = []
    for raw in (draft or "").splitlines():
        line = re.sub(r"\s+", " ", raw.strip())
        low = line.lower()
        if not line:
            continue
        if any(k in low for k in ("risk", "uncertain", "volatil", "downside", "headwind")):
            risks.append(f"- {line.lstrip('- ').strip()}")
        if len(risks) >= 2:
            break
    if len(risks) < 2:
        risks += [
            "- Retrieved context may not fully capture all relevant macro drivers.",
            "- Some cited information may become stale as new releases arrive.",
        ]
        risks = risks[:2]

    watch: list[str] = []
    for chunk in chunks[:3]:
        md = chunk.get("metadata", {}) or {}
        source = str(md.get("source") or "source").strip()
        title = str(md.get("title") or "").strip()
        if title:
            watch.append(f"- Track updates from {source}: {title[:96]}")
        if len(watch) >= 3:
            break
    while len(watch) < 3:
        watch.append("- Monitor new releases tied to the question's core drivers.")

    confidence = "MEDIUM"
    if retrieval_health.get("status") == "GOOD" and len(chunks) >= 5:
        confidence = "HIGH"
    elif retrieval_health.get("status") == "BAD" or len(chunks) < 2:
        confidence = "LOW"

    return "\n".join(
        [
            f"Executive summary: {summary}",
            f"Direct answer: {direct}",
            "Why this is likely:",
            *why_lines[:3],
            "Main risks:",
            *risks[:2],
            "What to watch next:",
            *watch[:3],
            f"Confidence: {confidence} - response repaired from available evidence and validated against citations.",
        ]
    )


def _evaluate_answer_quality(answer: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    structure_ok = _valid_answer(answer)
    grounding = _grounding_score(answer, chunks)
    numeric_risk = _numeric_hallucination_risk(answer, chunks)
    reasons: list[str] = []
    if not structure_ok:
        reasons.append("invalid_structure")
    if grounding < RAG_MIN_GROUNDING_SCORE:
        reasons.append("low_grounding")
    if numeric_risk > RAG_MAX_NUMERIC_RISK:
        reasons.append("high_numeric_risk")
    return {
        "structure_ok": structure_ok,
        "grounding": round(float(grounding), 4),
        "numeric_risk": round(float(numeric_risk), 4),
        "reasons": reasons,
    }


def _classify_generation_reason(exc: Exception | str) -> str:
    msg = str(exc or "").strip().lower()
    if "data quality gate" in msg:
        return "data_quality_gate"
    if "compact_retry_failed:" in msg:
        return msg.split("compact_retry_failed:", 1)[1].strip() or "compact_retry_failed"
    if "llm generation failed" in msg:
        return "llm_generation_failed"
    if "empty_response" in msg:
        return "llm_empty_response"
    if "readtimeout" in msg or "timeout" in msg:
        return "llm_timeout"
    if "connectionerror" in msg or "connection refused" in msg:
        return "llm_connection_error"
    if "httperror" in msg:
        return "llm_http_error"
    if "validation_failed:" in msg:
        return msg.split("validation_failed:", 1)[1].strip() or "validation_failed"
    return "unknown_generation_error"


async def run_query(question: str) -> tuple[str, list[dict[str, Any]]]:
    index = load_index()
    metadata = load_metadata()
    metadata_version = _metadata_version_token(metadata)
    cache_key = _answer_cache_key(question, metadata_version)
    cached = _answer_cache_get(cache_key)
    if cached is not None:
        return cached

    store_health = evaluate_vector_store_health(index.ntotal, metadata)

    optimized_question = rewrite_query(question) if RAG_ENABLE_QUERY_REWRITE else _deterministic_expand(question)
    adaptive_top_k, adaptive_context_chars = _adaptive_retrieval_budget(question)
    retrieval_error = None
    chunks = []
    try:
        qvec = await embed_query(optimized_question)
        chunks = retrieve_chunks_hybrid(optimized_question, qvec, index, metadata, top_k=adaptive_top_k)
    except Exception as exc:
        retrieval_error = exc
        chunks = retrieve_chunks_lexical(optimized_question, metadata, top_k=adaptive_top_k)

    retrieval_health = evaluate_retrieval_quality(question, chunks)

    prompt = build_prompt_from_scratch(chunks, question, context_max_chars=adaptive_context_chars)
    generation_mode = "llm"
    generation_reason = "none"
    deterministic_repair_used = False
    compact_retry_used = False
    answer_quality = {
        "structure_ok": False,
        "grounding": 0.0,
        "numeric_risk": 0.0,
        "reasons": ["not_evaluated"],
    }
    try:
        if store_health["status"] == "BAD" or retrieval_health["status"] == "BAD":
            raise RuntimeError("Data quality gate blocked generation due to low retrieval confidence.")

        _main_options = {
            "num_predict": RAG_MAIN_NUM_PREDICT,
            "temperature": RAG_MAIN_TEMPERATURE,
            "top_p": 1.0,
        }
        try:
            draft = ask_llm(prompt, options=_main_options)
        except Exception as full_exc:
            if not (RAG_ENABLE_COMPACT_RETRY and chunks):
                raise
            compact_retry_used = True
            compact_prompt = build_compact_prompt(chunks, question)
            compact_options = {
                "num_predict": RAG_COMPACT_NUM_PREDICT,
                "temperature": 0.0,
                "top_p": 1.0,
            }
            try:
                draft = ask_llm(
                    compact_prompt,
                    timeout_sec=RAG_COMPACT_RETRY_TIMEOUT_SEC,
                    options=compact_options,
                )
                generation_mode = "llm_compact"
                generation_reason = f"compact_retry_after_{_classify_generation_reason(full_exc)}"
            except Exception as compact_exc:
                raise RuntimeError(
                    f"compact_retry_failed:{_classify_generation_reason(full_exc)}->{_classify_generation_reason(compact_exc)}"
                ) from compact_exc

        answer = draft

        if not _valid_answer(draft) and RAG_ENABLE_ANSWER_REWRITE:
            revised = ask_llm(build_rewrite_prompt(draft))
            answer = revised if _valid_answer(revised) else draft

        if not _valid_answer(answer) and RAG_ENABLE_DETERMINISTIC_REPAIR:
            answer = _repair_answer_structure(question, answer, chunks, retrieval_health)
            deterministic_repair_used = True

        answer = _sanitize_unsupported_numbers(answer, chunks)
        answer_quality = _evaluate_answer_quality(answer, chunks)
        if answer_quality["reasons"]:
            raise RuntimeError("validation_failed:" + ",".join(answer_quality["reasons"]))

    except Exception as exc:
        generation_reason = _classify_generation_reason(exc)
        repaired = ""
        repaired_quality = {
            "structure_ok": False,
            "grounding": 0.0,
            "numeric_risk": 0.0,
            "reasons": ["repair_not_attempted"],
        }

        can_repair = (
            RAG_ENABLE_DETERMINISTIC_REPAIR
            and bool(chunks)
            and retrieval_health.get("status") != "BAD"
        )
        if can_repair:
            repaired = _repair_answer_structure(question, "", chunks, retrieval_health)
            repaired = _sanitize_unsupported_numbers(repaired, chunks)
            repaired_quality = _evaluate_answer_quality(repaired, chunks)

        if can_repair and not repaired_quality["reasons"]:
            answer = repaired
            answer_quality = repaired_quality
            generation_mode = "repaired"
            deterministic_repair_used = True
        else:
            generation_mode = "fallback"
            answer = build_fallback_answer(question, chunks)
            answer_quality = _evaluate_answer_quality(answer, chunks)
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
    quality_note += (
        "\nGeneration diagnostics: "
        f"mode={generation_mode}; reason={generation_reason}; "
        f"adaptive_top_k={adaptive_top_k}; context_chars={adaptive_context_chars}; "
        f"compact_retry={'yes' if compact_retry_used else 'no'}; "
        f"deterministic_repair={'yes' if deterministic_repair_used else 'no'}; "
        f"grounding={answer_quality['grounding']}; "
        f"numeric_risk={answer_quality['numeric_risk']}; "
        f"structure_ok={answer_quality['structure_ok']}"
    )
    answer += quality_note

    _answer_cache_set(cache_key, answer, chunks)

    return answer, chunks


def main():
    print("\nFinance RAG ready. Type 'exit' to quit.\n")
    while True:
        q = input("Ask a question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        try:
            import asyncio
            answer, _ = asyncio.run(run_query(q))
        except Exception as e:
            print(f"Error querying: {e}")
            answer = "Error"
        print("\nAnswer:\n")
        print(answer)
        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
