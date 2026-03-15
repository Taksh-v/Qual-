import json
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

from intelligence.shared_embed_cache import get_cached as _cache_get, put_cached as _cache_put
from ingestion.embeddings import get_embedding

try:
    from ingestion.bm25_index import load_bm25_index, reciprocal_rank_fusion as _rrf
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

    def _rrf(sem, bm25, **kw):  # type: ignore[misc]
        return sem

try:
    from ingestion.metadata_store import get_store as _get_metadata_store
    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False

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

# Cross-encoder reranking flag: set INTEL_USE_RERANKER=0 to disable on low-RAM systems
_USE_RERANKER: bool = os.getenv("INTEL_USE_RERANKER", "1").strip() not in ("0", "false", "no")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


_DEFAULT_CONTEXT_REWRITE: bool = _env_flag("INTEL_CONTEXT_REWRITE_DEFAULT", False)
_INTEL_SQLITE_PREFILTER: bool = _env_flag("INTEL_SQLITE_PREFILTER", True)
_INTEL_SQLITE_PREFILTER_ROWID_LIMIT: int = max(200, int(os.getenv("INTEL_SQLITE_PREFILTER_ROWID_LIMIT", "2500")))
_INTEL_SQLITE_PREFILTER_TERMS: int = max(2, int(os.getenv("INTEL_SQLITE_PREFILTER_TERMS", "8")))
_INTEL_SQLITE_PREFILTER_MIN_NTOTAL: int = max(0, int(os.getenv("INTEL_SQLITE_PREFILTER_MIN_NTOTAL", "2500")))
_INTEL_RERANK_GATE: bool = _env_flag("INTEL_RERANK_GATE", True)
_INTEL_RERANK_MIN_CANDIDATES: int = max(2, int(os.getenv("INTEL_RERANK_MIN_CANDIDATES", "10")))
_INTEL_RERANK_MIN_MARGIN: float = float(os.getenv("INTEL_RERANK_MIN_MARGIN", "1.8"))
_INTEL_RERANK_MIN_TOP_SCORE: float = float(os.getenv("INTEL_RERANK_MIN_TOP_SCORE", "5.5"))
_INTEL_RERANK_MAX_ITEMS: int = max(4, int(os.getenv("INTEL_RERANK_MAX_ITEMS", "18")))
_INTEL_CONTEXT_CACHE: bool = _env_flag("INTEL_CONTEXT_CACHE", True)
_INTEL_CONTEXT_CACHE_TTL_SEC: float = float(os.getenv("INTEL_CONTEXT_CACHE_TTL_SEC", "180"))
_INTEL_CONTEXT_CACHE_MAX_KEYS: int = max(32, int(os.getenv("INTEL_CONTEXT_CACHE_MAX_KEYS", "512")))
_INTEL_RELAX_HARD_FILTER_FALLBACK: bool = _env_flag("INTEL_RELAX_HARD_FILTER_FALLBACK", True)

_CONTEXT_CACHE: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()
_CONTEXT_CACHE_LOCK = Lock()

_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can",
    "could", "do", "does", "for", "from", "how", "if", "in", "into",
    "is", "it", "its", "latest", "likely", "may", "might", "more",
    "most", "of", "on", "or", "outlook", "performance", "recent",
    "should", "than", "that", "the", "their", "them", "these", "this",
    "those", "to", "us", "using", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "change", "changes",
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


def _metadata_version_token(metadata: list[dict]) -> str:
    latest = ""
    for item in metadata:
        md = item.get("metadata", {}) or {}
        ts = str(md.get("indexed_at") or md.get("extracted_at") or md.get("date") or "")
        if ts and ts > latest:
            latest = ts
    return f"{len(metadata)}:{latest}"


def _context_cache_key(
    question: str,
    top_k: int,
    keep_latest: int,
    rewrite_enabled: bool,
    metadata_version: str,
) -> str:
    return "|".join(
        [
            _normalize_cache_question(question),
            f"top_k={top_k}",
            f"keep={keep_latest}",
            f"rewrite={1 if rewrite_enabled else 0}",
            f"reranker={1 if _USE_RERANKER else 0}",
            f"gate={1 if _INTEL_RERANK_GATE else 0}",
            f"ver={metadata_version}",
        ]
    )


def _context_cache_get(key: str) -> list[dict] | None:
    if not _INTEL_CONTEXT_CACHE:
        return None
    now = time.time()
    with _CONTEXT_CACHE_LOCK:
        entry = _CONTEXT_CACHE.get(key)
        if not entry:
            return None
        expires_at, payload = entry
        if expires_at <= now:
            _CONTEXT_CACHE.pop(key, None)
            return None
        _CONTEXT_CACHE.move_to_end(key)
        return list(payload)


def _context_cache_set(key: str, payload: list[dict]) -> None:
    if not _INTEL_CONTEXT_CACHE:
        return
    now = time.time()
    with _CONTEXT_CACHE_LOCK:
        expired = [k for k, (expires_at, _) in _CONTEXT_CACHE.items() if expires_at <= now]
        for cache_key in expired:
            _CONTEXT_CACHE.pop(cache_key, None)

        _CONTEXT_CACHE[key] = (now + max(1.0, _INTEL_CONTEXT_CACHE_TTL_SEC), list(payload))
        _CONTEXT_CACHE.move_to_end(key)

        while len(_CONTEXT_CACHE) > _INTEL_CONTEXT_CACHE_MAX_KEYS:
            _CONTEXT_CACHE.popitem(last=False)


def _item_search_tokens(item: dict) -> set[str]:
    md = item.get("metadata", {}) or {}
    blob = " ".join(
        str(part)
        for part in (
            item.get("text") or "",
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
    return {_normalize_term(tok) for tok in re.findall(r"[a-zA-Z0-9_]+", blob.lower()) if len(tok) > 1}


def _phrase_score(question: str, item: dict) -> float:
    phrases = _query_phrases(question)
    if not phrases:
        return 0.0
    md = item.get("metadata", {}) or {}
    blob = " ".join(
        str(part).lower()
        for part in (
            md.get("title") or "",
            item.get("text") or "",
            " ".join(md.get("entities") or []),
        )
        if part
    )
    return float(sum(2.5 for phrase in phrases if phrase in blob))


def _article_key(item: dict) -> tuple[str, str, str]:
    md = item.get("metadata", {}) or {}
    return (
        (md.get("title") or "").strip().lower(),
        (md.get("source") or "").strip().lower(),
        (md.get("date") or md.get("extracted_at") or "").strip().lower(),
    )


def _diversify_items(items: list[dict], limit: int, max_per_article: int = 2) -> list[dict]:
    diversified: list[dict] = []
    article_counts: dict[tuple[str, str, str], int] = {}
    for item in items:
        key = _article_key(item)
        if key[0]:
            count = article_counts.get(key, 0)
            if count >= max_per_article:
                continue
            article_counts[key] = count + 1
        diversified.append(item)
        if len(diversified) >= limit:
            break
    return diversified


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
    r"\bcpi\b": "macro_commentary",
    r"\bgdp\b": "macro_commentary",
    r"\bfed\b": "macro_commentary",
    r"\binterest rates\b": "macro_commentary",
    r"\bresearch\b": "research_report",
    r"analyst report": "research_report",
    r"equity research": "research_report",
    r"\bnews\b": "news",
    r"\bheadline\b": "news",
}

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


def _dtype_matches_hint(dtype: str, hint: str) -> bool:
    d = (dtype or "").strip().lower()
    h = (hint or "").strip().lower()
    if not d or not h:
        return False
    if d == h:
        return True
    if h == "macro_commentary" and d.startswith("macro_"):
        return True
    return False


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


def _passes_hard_filters(item: dict, filters: dict[str, Any]) -> bool:
    hard = filters.get("hard") or {}
    md = item.get("metadata", {}) or {}

    data_types = hard.get("data_types")
    if data_types:
        dtype = (md.get("data_type") or item.get("data_type") or "").lower()
        if not any(_dtype_matches_hint(dtype, hinted) for hinted in data_types):
            return False

    min_dt = hard.get("min_dt")
    if min_dt:
        raw_dt = md.get("extracted_at") or md.get("date")
        dt = _safe_parse_dt(raw_dt)
        if dt is datetime.min:
            return False
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt < min_dt:
            return False

    return True


def _soft_filter_boost(item: dict, filters: dict[str, Any]) -> float:
    soft = filters.get("soft") or {}
    if not soft:
        return 0.0

    md = item.get("metadata", {}) or {}
    score = 0.0

    soft_types = soft.get("data_types")
    if soft_types:
        dtype = (md.get("data_type") or item.get("data_type") or "").lower()
        if any(_dtype_matches_hint(dtype, hinted) for hinted in soft_types):
            score += 2.0

    soft_regions = soft.get("regions")
    if soft_regions:
        region = (md.get("region") or item.get("region") or "").strip()
        if region in soft_regions:
            score += 1.5

    soft_sectors = soft.get("sectors")
    if soft_sectors:
        sector = (md.get("sector") or item.get("sector") or "").strip()
        if sector in soft_sectors:
            score += 1.5

    return score


def _within_soft_window(item: dict, hours: int | None) -> bool:
    if not hours:
        return False
    md = item.get("metadata", {}) or {}
    dt = _safe_parse_dt(md.get("extracted_at") or md.get("date"))
    if dt is datetime.min:
        return False
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt >= now - timedelta(hours=hours)


def embed_query(text: str) -> np.ndarray:
    """Embed *text* using the configured embedding strategy."""
    cached = _cache_get(text)
    if cached is not None:
        return cached

    vec = get_embedding(text, normalize=True, role="query")
    vec = np.array(vec, dtype="float32").reshape(1, -1)
    _cache_put(text, vec)
    return vec


def _get_reranker():
    """Lazy-load the MS-MARCO cross-encoder.  Shared singleton (module-level)."""
    if not hasattr(_get_reranker, "_model"):
        try:
            from sentence_transformers import CrossEncoder
            _get_reranker._model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512
            )
        except Exception:
            _get_reranker._model = None
    return _get_reranker._model


def _fallback_lexical_context(question: str, metadata: list[dict], top_k: int, filters: dict[str, Any] | None = None) -> list[dict]:
    tokens = _extract_query_tokens(question)
    if not tokens:
        if filters and filters.get("hard"):
            filtered = [m for m in metadata if _passes_hard_filters(m, filters)]
            return filtered[:top_k]
        return metadata[:top_k]
    scored: list[tuple[int, dict]] = []
    for item in metadata:
        if filters and not _passes_hard_filters(item, filters):
            continue
        search_tokens = _item_search_tokens(item)
        if not search_tokens:
            continue
        score = sum(1.2 for t in tokens if t in search_tokens)
        score += _title_score(tokens, item)
        score += _phrase_score(question, item)
        score += _soft_filter_boost(item, filters or {})
        score += _recency_score(item)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        return _diversify_items([i for _, i in scored[: top_k * 4]], top_k)
    if filters and filters.get("hard"):
        return []
    return metadata[:top_k]


def _merge_unique_items(primary: list[dict], secondary: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for item in primary + secondary:
        md = item.get("metadata", {}) or {}
        title = (md.get("title") or "").strip().lower()
        text_head = (item.get("text") or "").strip().lower()[:220]
        key = f"{title}|{text_head}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _recency_score(item: dict) -> float:
    """
    Tiered recency bonus:
      - Last 24h:  +8.0 pts (breaking news most valuable)
      - Last 3d:   +5.0 pts
      - Last 7d:   +3.0 pts
      - Last 30d:  +1.0 pts
      - Older:     +0.0 pts
    Use 'extracted_at' preferentially over 'date' for most accurate recency.
    """
    md = item.get("metadata", {}) or {}
    # Prefer extraction timestamp for recency (reflects database freshness)
    dt = _safe_parse_dt(md.get("extracted_at") or md.get("date"))
    if dt is datetime.min:
        return 0.0
    # Ensure timezone-aware comparison
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_hours = max((now - dt).total_seconds() / 3600, 0)
    if age_hours <= 24:
        return 2.5
    if age_hours <= 72:   # 3 days
        return 1.5
    if age_hours <= 168:  # 7 days
        return 0.9
    if age_hours <= 720:  # 30 days
        return 0.3
    return 0.0


def _title_score(tokens: set[str], item: dict) -> float:
    """Extra weight for query terms appearing in the article title."""
    title = item.get("metadata", {}).get("title") or ""
    if not title or not tokens:
        return 0.0
    title_tokens = {_normalize_term(tok) for tok in re.findall(r"[a-zA-Z0-9_]+", title.lower()) if len(tok) > 1}
    return float(sum(2.5 for t in tokens if t in title_tokens))


def _relevance_score(question: str, item: dict, filters: dict[str, Any]) -> float:
    """Combined relevance score: text hits + entity hits + title hits + recency."""
    tokens = _extract_query_tokens(question)

    text = (item.get("text") or "").lower()
    if not text or not tokens:
        return _recency_score(item)
    search_tokens = _item_search_tokens(item)
    text_tokens = {_normalize_term(tok) for tok in re.findall(r"[a-zA-Z0-9_]+", text) if len(tok) > 1}

    text_hits  = sum(1.4 for t in tokens if t in text_tokens)
    ent_hits   = sum(1.8 for t in tokens if t in search_tokens)
    title_hits = _title_score(tokens, item)
    recency    = _recency_score(item)
    coverage   = len(tokens.intersection(search_tokens)) / max(1, len(tokens))

    score = text_hits + ent_hits + title_hits + recency + (coverage * 6.0)
    score += _phrase_score(question, item)
    score += _soft_filter_boost(item, filters)
    soft_hours = filters.get("soft_recent_hours")
    if soft_hours and _within_soft_window(item, soft_hours):
        score += 1.0
    if len(tokens) >= 2 and coverage < 0.34:
        score -= 3.0
    if len(tokens) >= 3 and coverage < 0.2:
        score -= 4.0
    return score


def _should_apply_reranker(
    score_trace: list[float],
    candidate_count: int,
    keep_latest: int,
) -> bool:
    if not _USE_RERANKER or candidate_count <= 1:
        return False
    if not _INTEL_RERANK_GATE:
        return True

    min_candidates = max(_INTEL_RERANK_MIN_CANDIDATES, keep_latest + 2)
    if candidate_count < min_candidates:
        return False
    if not score_trace:
        return False

    top = float(score_trace[0])
    second = float(score_trace[1]) if len(score_trace) > 1 else top
    margin = top - second

    if top < _INTEL_RERANK_MIN_TOP_SCORE:
        return True
    return margin < _INTEL_RERANK_MIN_MARGIN


@lru_cache(maxsize=1)
def _load_index_cached():
    index_path = _first_existing(INDEX_CANDIDATES)
    return faiss.read_index(index_path)


@lru_cache(maxsize=1)
def _load_metadata_cached() -> list:
    index = _load_index_cached()
    metadata_path = _pick_best_metadata_path(index.ntotal)
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_bm25_cached():
    if not _BM25_AVAILABLE:
        return None
    try:
        metadata = _load_metadata_cached()
        return load_bm25_index(metadata)
    except Exception:
        return None


def _get_faiss_id_filter(
    filters: dict[str, Any] | None,
    question: str | None = None,
) -> tuple["faiss.IDSelector | None", np.ndarray | None, int]:
    if not _INTEL_SQLITE_PREFILTER or not _SQLITE_AVAILABLE or not filters:
        return None, None, 0

    hard = filters.get("hard") or {}
    soft = filters.get("soft") or {}

    min_dt = hard.get("min_dt")
    min_date = min_dt.strftime("%Y-%m-%d") if min_dt else None

    data_types = set(hard.get("data_types") or [])
    if not data_types:
        soft_types = set(soft.get("data_types") or [])
        data_types = {dt for dt in soft_types if dt != "macro_commentary"}

    regions = set(soft.get("regions") or [])
    sectors = set(soft.get("sectors") or [])

    has_metadata_filter = bool(min_date or data_types or regions or sectors)
    if not has_metadata_filter:
        return None, None, 0

    query_terms = _extract_query_terms_in_order(question or "")[:_INTEL_SQLITE_PREFILTER_TERMS]
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
            limit=_INTEL_SQLITE_PREFILTER_ROWID_LIMIT,
        )
        if not rowids:
            return None, None, 0
        id_array = np.array(sorted(set(rowids)), dtype=np.int64)
        selector = faiss.IDSelectorBatch(len(id_array), faiss.swig_ptr(id_array))
        return selector, id_array, len(id_array)
    except Exception:
        return None, None, 0


def retrieve_relevant_context(
    question: str,
    top_k: int = 12,
    keep_latest: int = 8,
    rewrite: bool | None = None,
) -> list[dict]:
    """
    Retrieve relevant context chunks using two-stage retrieval:
      Stage 1: Dense vector search (FAISS) with optional query rewriting
      Stage 2: Rerank by combined relevance + recency score

    Args:
        question:    Raw user question
        top_k:       Number of candidates from FAISS (before reranking)
        keep_latest: Final number of chunks to return after reranking
        rewrite:     Override query rewriting; when None uses INTEL_CONTEXT_REWRITE_DEFAULT
    """
    index = _load_index_cached()
    metadata = _load_metadata_cached()
    filters = _extract_filters(question)

    if index.ntotal == 0 or not metadata:
        return []

    # Stage 1: Dense retrieval — optionally rewrite for better embedding
    retrieval_question = question
    rewrite_enabled = _DEFAULT_CONTEXT_REWRITE if rewrite is None else bool(rewrite)
    if rewrite_enabled:
        try:
            from intelligence.query_rewriter import rewrite_query
            retrieval_question = rewrite_query(question)
        except Exception:
            retrieval_question = question
    else:
        try:
            from intelligence.query_rewriter import _deterministic_expand
            retrieval_question = _deterministic_expand(question)
        except Exception:
            retrieval_question = question

    metadata_version = _metadata_version_token(metadata)
    cache_key = _context_cache_key(question, top_k, keep_latest, rewrite_enabled, metadata_version)
    cached = _context_cache_get(cache_key)
    if cached is not None:
        return cached

    candidates: list[dict] = []
    active_filters = filters
    try:
        query_vec = embed_query(retrieval_question)

        # Use SQLite-backed FAISS prefilter for metadata-constrained prompts.
        id_selector = None
        id_selector_keepalive = None
        eligible_count = 0
        if index.ntotal >= _INTEL_SQLITE_PREFILTER_MIN_NTOTAL:
            id_selector, id_selector_keepalive, eligible_count = _get_faiss_id_filter(
                filters,
                question=retrieval_question,
            )

        # Fetch more candidates than needed so reranking has room to work.
        n_fetch = min(max(top_k * 3, 20), index.ntotal)
        if id_selector is not None and eligible_count > 0:
            n_fetch = min(max(top_k * 4, 24), eligible_count, index.ntotal)
            n_fetch = max(top_k, n_fetch)

        if id_selector is not None:
            try:
                params = faiss.SearchParameters()
                params.sel = id_selector
                _, indices = index.search(query_vec, n_fetch, params=params)
            except Exception:
                _, indices = index.search(query_vec, n_fetch)
        else:
            _, indices = index.search(query_vec, n_fetch)

        _ = id_selector_keepalive  # keep selector buffer alive for FAISS call

        for idx in indices[0]:
            if 0 <= idx < len(metadata):
                item = metadata[idx]
                if not _passes_hard_filters(item, filters):
                    continue
                candidates.append(item)
    except Exception:
        # Fallback: lexical search on original question
        candidates = _fallback_lexical_context(question, metadata, top_k, filters=filters)

    bm25_pairs: list[tuple[dict, float]] = []
    bm25_idx = _load_bm25_cached()
    if bm25_idx is not None:
        try:
            bm25_pairs = bm25_idx.search(question, top_k=min(max(top_k * 4, 20), len(metadata)))
            if filters.get("hard"):
                bm25_pairs = [(item, score) for item, score in bm25_pairs if _passes_hard_filters(item, filters)]
        except Exception:
            bm25_pairs = []

    if bm25_pairs:
        candidates = _rrf(candidates, bm25_pairs, semantic_weight=0.55, bm25_weight=0.45)

    if not candidates and filters.get("hard") and _INTEL_RELAX_HARD_FILTER_FALLBACK:
        relaxed_filters = _relax_hard_filters(filters)
        relaxed_candidates = _fallback_lexical_context(
            question,
            metadata,
            top_k=max(top_k, keep_latest),
            filters=relaxed_filters,
        )
        if bm25_idx is not None:
            try:
                relaxed_bm25 = bm25_idx.search(question, top_k=min(max(top_k * 4, 20), len(metadata)))
                if relaxed_bm25:
                    relaxed_candidates = _merge_unique_items(
                        relaxed_candidates,
                        [item for item, _ in relaxed_bm25],
                    )
            except Exception:
                pass
        candidates = relaxed_candidates
        active_filters = relaxed_filters

    if not candidates and filters.get("hard"):
        _context_cache_set(cache_key, [])
        return []

    # Stage 2a: Relevance + recency pre-scoring (fast BM25-like pass)
    scored_candidates: list[tuple[float, dict]] = []
    for item in candidates:
        scored_candidates.append((_relevance_score(question, item, active_filters), item))
    scored_candidates.sort(key=lambda pair: pair[0], reverse=True)

    score_trace = [score for score, _ in scored_candidates[:8]]
    candidates = [item for _, item in scored_candidates]

    # Deduplicate near-identical chunks (common in repetitive scrape outputs).
    unique: list[dict] = []
    seen: set[str] = set()
    for item in candidates:
        md = item.get("metadata", {}) or {}
        title = (md.get("title") or "").strip().lower()
        text_head = (item.get("text") or "").strip().lower()[:220]
        key = f"{title}|{text_head}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    unique = _diversify_items(unique, max(keep_latest * 3, top_k * 2))

    # Stage 2b: Cross-encoder reranking on ambiguous/low-confidence cases only.
    should_rerank = _should_apply_reranker(score_trace, len(unique), keep_latest)
    if should_rerank and len(unique) > 1:
        reranker = _get_reranker()
        if reranker is not None:
            try:
                rerank_count = min(len(unique), _INTEL_RERANK_MAX_ITEMS)
                rerank_slice = unique[:rerank_count]
                pairs = []
                for item in rerank_slice:
                    md = item.get("metadata", {}) or {}
                    doc = f"{md.get('title', '')} {item.get('text', '')}"
                    pairs.append([question, doc])
                ce_scores = reranker.predict(pairs)
                reranked = [c for _, c in sorted(zip(ce_scores, rerank_slice), key=lambda x: x[0], reverse=True)]
                unique = reranked + unique[rerank_count:]
            except Exception:
                pass   # fall through to pre-scored order if reranker fails

    final_context = _diversify_items(unique, keep_latest)
    _context_cache_set(cache_key, final_context)
    return final_context


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
