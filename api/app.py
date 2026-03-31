import json
import logging
import logging.config
import os
import re
import asyncio
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.responses import PlainTextResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict

# ── Rate limiting ─────────────────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _SLOWAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SLOWAPI_AVAILABLE = False

from rag.rag_core import ask_rag
from rag.query import invalidate_index_cache
from intelligence.cross_asset_analyzer import analyze_cross_asset
from intelligence.indicator_parser import (
    extract_indicators_from_text,
    get_regime_inputs_from_indicators,
    sanitize_indicator_values,
)
from intelligence.context_retriever import retrieve_relevant_context
from intelligence.macro_engine import macro_intelligence_pipeline, get_last_model_used
from intelligence.live_market_data import (
    fetch_live_indicators,
    stream_live_indicators,
    invalidate_live_data_cache,
    any_price_market_open,
    market_status_summary,
)
from intelligence.question_classifier import classify_question
from intelligence.regime_detector import detect_regime
from intelligence.news_health_checker import check_news_health, check_news_health_quick
from intelligence.response_enhancer import score_response
from intelligence.response_middleware import normalize_api_payload
from intelligence.query_logger import log_query, read_recent, compute_metrics
from intelligence.shared_embed_cache import cache_info as embed_cache_info
from intelligence.sentiment_analyzer import score_sentiment, sentiment_summary
from ingestion.fundamentals import (
    get_fundamentals,
    get_batch_fundamentals,
    format_fundamentals_summary,
)
from ingestion.market_data import get_market_snapshot, format_snapshot_for_prompt
from config.indicators import INDICATOR_META

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party loggers
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

app = FastAPI(title="News Intelligence RAG API")

# ── Rate limiter ──────────────────────────────────────────────────────────────
# Default: 60 req/min globally; expensive LLM endpoints: 20 req/min.
# Set RATE_LIMIT_DEFAULT and RATE_LIMIT_LLM env vars to override.
_RL_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "60/minute")
_RL_LLM     = os.getenv("RATE_LIMIT_LLM",     "20/minute")
if _SLOWAPI_AVAILABLE:
    _limiter = Limiter(key_func=get_remote_address, default_limits=[_RL_DEFAULT])
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
else:
    _limiter = None

def _rl(limit_str: str):
    """Apply *limit_str* rate-limit when slowapi is available; no-op otherwise."""
    if _limiter is not None:
        return _limiter.limit(limit_str)
    return lambda f: f

# ── API-key authentication ────────────────────────────────────────────────────
# Set API_KEYS env var to a comma-separated list of valid keys.
# If unset → open / dev mode (all requests pass through).
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_VALID_API_KEYS: set[str] = set(k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip())

# Paths that are always public, regardless of key config.
_PUBLIC_PREFIXES = (
    "/static",
    "/.well-known",
    "/health",
    "/docs",
    "/openapi",
    "/redoc",
    "/favicon.ico",
)

@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    """Enforce X-API-Key on all non-public endpoints when API_KEYS is set."""
    if not _VALID_API_KEYS:
        return await call_next(request)  # Dev mode: no keys configured

    # Local development convenience:
    # If the request originates from localhost, allow access without an API key.
    # This prevents 403s during local testing when callers don't attach X-API-Key.
    #
    # (You can disable this behavior by setting BYPASS_API_KEY_FOR_LOCALHOST=0)
    bypass_local = os.getenv("BYPASS_API_KEY_FOR_LOCALHOST", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if bypass_local:
        client_host = (request.client.host if request.client else "") or ""
        xff = request.headers.get("X-Forwarded-For", "")
        xff_host = (xff.split(",")[0].strip() if xff else "")
        if client_host in {"127.0.0.1", "::1"} or xff_host in {"127.0.0.1", "::1"}:
            return await call_next(request)

    path = request.url.path
    if path == "/" or any(path.startswith(p) for p in _PUBLIC_PREFIXES):
        return await call_next(request)
    key = request.headers.get("X-API-Key", "")
    if key not in _VALID_API_KEYS:
        return JSONResponse(status_code=403, content={"detail": "Invalid or missing API key. Pass X-API-Key header."})
    return await call_next(request)

app.mount("/static", StaticFiles(directory="api/static"), name="static")

logger.info("News Intelligence RAG API starting up. Auth=%s RateLimit=%s",
    "enabled" if _VALID_API_KEYS else "dev-mode", _RL_DEFAULT)

from intelligence.cache_utils import _TieredCache, AsyncCacheStampedeGuard

# ── Response cache for /ask endpoint ─────────────────────────────────────────
# Avoids re-running the full RAG + LLM pipeline for identical questions
# within the TTL window (default 5 minutes).
_ASK_CACHE_TTL: float = float(os.getenv("ASK_CACHE_TTL", "300"))  # seconds
_ASK_CACHE_MAX: int = 256
_ask_guard = AsyncCacheStampedeGuard(_TieredCache(), max_size=_ASK_CACHE_MAX)

# ── Response cache for /intelligence/analyze endpoint ─────────────────────────
# Prevents re-running the full regime + LLM pipeline for repeated questions.
_INTEL_CACHE_TTL: float = float(os.getenv("INTEL_CACHE_TTL", "300"))
_INTEL_CACHE_MAX: int = 128
_intel_guard = AsyncCacheStampedeGuard(_TieredCache(), max_size=_INTEL_CACHE_MAX)

# ── Feedback store (in-memory + JSONL persistence) ────────────────────────────
import pathlib as _pathlib
_FEEDBACK_PATH = os.getenv(
    "FEEDBACK_LOG_PATH",
    str(_pathlib.Path("data") / "feedback_log.jsonl"),
)
_feedback_lock = Lock()


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
def chrome_devtools() -> dict:
    """Silences Chrome DevTools auto-probe 404s."""
    return {}


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    response_contract: dict[str, Any] | None = Field(default=None, alias="_response_contract")

    model_config = ConfigDict(populate_by_name=True)


class IntelligenceRequest(BaseModel):
    question: str
    geography: str = "US"
    horizon: str = "MEDIUM_TERM"
    response_mode: str = "brief"
    indicator_overrides: dict[str, float] = Field(default_factory=dict)


class HedgeRequest(BaseModel):
    """
    Request body for /intelligence/hedge.
    
    Fields:
        question:  Describe the risk or portfolio concern.
                   e.g. "I hold US equities and bonds. How do I hedge rising inflation?"
        tickers:   Optional list of ticker symbols in the portfolio.
                   Fundamentals are fetched and injected into the LLM prompt.
        geography: Regional focus (default: US)
        horizon:   NEAR_TERM | MEDIUM_TERM | LONG_TERM
    """
    question:   str
    tickers:    list[str] = Field(default_factory=list, max_length=10)
    geography:  str = "US"
    horizon:    str = "MEDIUM_TERM"



# INDICATOR_META is now maintained in config/indicators.py and imported above.


@app.post("/ask")
@_rl(_RL_LLM)
async def ask_question(req: QueryRequest, request: Request):
    question_key = req.question.strip().lower()

    async def _compute_ask() -> dict:
        t_start = time.time()
        error_msg: str | None = None
        try:
            result = await ask_rag(req.question)
        except Exception as exc:
            error_msg = str(exc)
            result = {
                "question": req.question,
                "answer": (
                    "RAG backend is currently degraded. "
                    f"Fallback message: {exc}. "
                    "Check Ollama/index availability."
                ),
                "sources": [],
            }

        result = normalize_api_payload(result, mode="brief")

        latency_ms = int((time.time() - t_start) * 1000)
        # Source count acts as a proxy for chunk_count on /ask
        answer_text = result.get("answer", "") if isinstance(result, dict) else str(result)
        sources = result.get("sources", []) if isinstance(result, dict) else []
        # Compute sentiment over the answer text
        sent = score_sentiment(answer_text)
        # Estimate hallucination risk (reuse grounding score proxy)
        from intelligence.utils import numeric_hallucination_risk as _hal_risk
        hal_risk = None
        try:
            chunks_for_hal = [{"text": s.get("text", s.get("content", ""))} for s in sources[:5]]
            hal_risk = _hal_risk(answer_text, chunks_for_hal)
        except Exception:
            pass
        log_query(
            endpoint="/ask",
            question=req.question,
            latency_ms=latency_ms,
            chunk_count=len(sources),
            cache_hit=False,
            hallucination_risk=hal_risk,
            sentiment_label=sent.get("label"),
            sentiment_score=sent.get("score"),
            error=error_msg,
        )
        return result

    val, is_fresh = _ask_guard.cache.get(question_key)
    if is_fresh:
        logger.info("[ask] cache hit for question: %s", req.question[:80])
        log_query(endpoint="/ask", question=req.question, cache_hit=True)
        return val

    # Cache miss — run full pipeline via guard to prevent stampede
    return await _ask_guard.get_or_compute(question_key, _ASK_CACHE_TTL, _compute_ask)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_overrides(overrides: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in (overrides or {}).items():
        parsed = _safe_float(value)
        if parsed is not None:
            normalized[key] = parsed
    return normalized


def _collect_indicator_inputs(
    req: IntelligenceRequest,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    overrides = _normalized_overrides(req.indicator_overrides)
    context_chunks: list[dict[str, Any]] = []
    context_text = ""
    live_meta: dict[str, Any] = {}
    try:
        # Reduced from top_k=25/keep_latest=15 → 12/8 to cut retrieval + prompt size.
        context_chunks = retrieve_relevant_context(req.question, top_k=12, keep_latest=8)
        context_text = " ".join(c.get("text", "") for c in context_chunks)
    except Exception:
        context_chunks = []
        context_text = ""

    # Fetch live FRED data first (authoritative baseline)
    try:
        live_data, live_meta = fetch_live_indicators()
    except Exception:
        live_data = {}

    from_context = extract_indicators_from_text(context_text)
    from_question = extract_indicators_from_text(req.question)
    # Priority: overrides > extracted from text/question > live FRED
    all_indicators = sanitize_indicator_values({**live_data, **from_context, **from_question, **overrides})
    return all_indicators, context_chunks, live_meta


def _build_snapshot(req: IntelligenceRequest) -> dict[str, Any]:
    classification = classify_question(req.question)
    all_indicators, context_chunks, live_meta = _collect_indicator_inputs(req)
    overrides = _normalized_overrides(req.indicator_overrides)

    regime_inputs = get_regime_inputs_from_indicators(all_indicators)
    regime = detect_regime(**regime_inputs)
    cross_asset = analyze_cross_asset(all_indicators)

    critical = []
    for key, (label, unit) in INDICATOR_META.items():
        value = all_indicators.get(key)
        direction = "flat"
        critical.append(
            {
                "key": key,
                "label": label,
                "value": value,
                "unit": unit,
                "direction": direction,
                "overridden": key in overrides,
            }
        )

    sources = []
    seen_titles = set()
    for c in context_chunks:
        md = c.get("metadata", {})
        title = md.get("title", "Unknown title")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        sources.append(
            {
                "title": title,
                "source": md.get("source", "Unknown source"),
                "date": md.get("date", "Unknown date"),
            }
        )
        if len(sources) >= 5:
            break

    return {
        "question": req.question,
        "geography": req.geography,
        "horizon": req.horizon,
        "classification": classification,
        "regime": regime,
        "cross_asset": cross_asset,
        "critical_indicators": critical,
        "detected_indicators": {k: all_indicators[k] for k in sorted(all_indicators.keys())},
        "live_data_meta": live_meta,
        "evidence_coverage": {
            "context_chunks": len(context_chunks),
            "has_overrides": bool(overrides),
            "sources": sources,
        },
    }


def _fallback_snapshot(req: IntelligenceRequest, reason: str = "") -> dict[str, Any]:
    missing_inputs = [reason] if reason else ["Snapshot temporarily unavailable"]
    return {
        "question": req.question,
        "geography": req.geography,
        "horizon": req.horizon,
        "classification": "Unknown",
        "regime": {
            "regime": "UNKNOWN",
            "confidence": "LOW",
            "missing_inputs": missing_inputs,
        },
        "cross_asset": {
            "overall_signal": "MIXED",
        },
        "critical_indicators": [],
        "detected_indicators": {},
        "evidence_coverage": {
            "context_chunks": 0,
            "has_overrides": bool(_normalized_overrides(req.indicator_overrides)),
            "sources": [],
        },
    }


def _parse_unified_response(response_text: str) -> dict[str, str]:
    fields = {
        "executive_summary": "",
        "direct_answer": "",
        "data_snapshot": "",
        "causal_chain": "",
        "what_is_happening": "",
        "market_impact": "",
        "scenarios": "",
        "consequences": "",
        "main_risks": "",
        "watch_next": "",
        "time_horizons": "",
        "confidence": "",
        # legacy / compat fields
        "why_likely": "",
        "market_map": "",
        "action_plan": "",
    }
    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("executive summary:"):
            fields["executive_summary"] = line.split(":", 1)[1].strip()
        elif lower.startswith("direct answer:") or lower.startswith("bottom line:"):
            fields["direct_answer"] = line.split(":", 1)[1].strip()
        elif lower.startswith("data snapshot:"):
            fields["data_snapshot"] = line.split(":", 1)[1].strip()
        elif lower.startswith("causal chain:"):
            fields["causal_chain"] = line.split(":", 1)[1].strip()
        elif lower.startswith("what is happening:"):
            fields["what_is_happening"] = line.split(":", 1)[1].strip()
            if not fields["why_likely"]:
                fields["why_likely"] = fields["what_is_happening"]
        elif lower.startswith("market impact:"):
            fields["market_impact"] = line.split(":", 1)[1].strip()
            if not fields["market_map"]:
                fields["market_map"] = fields["market_impact"]
        elif lower.startswith("scenarios"):
            fields["scenarios"] = line.split(":", 1)[1].strip() if ":" in line else ""
        elif lower.startswith("consequences & risks:") or lower.startswith("consequences:"):
            fields["consequences"] = line.split(":", 1)[1].strip()
            if not fields["main_risks"]:
                fields["main_risks"] = fields["consequences"]
        elif lower.startswith("time horizons:"):
            fields["time_horizons"] = line.split(":", 1)[1].strip()
        elif lower.startswith("what to watch:") or lower.startswith("what to watch next:") or lower.startswith("what to track next:"):
            fields["watch_next"] = line.split(":", 1)[1].strip()
        elif lower.startswith("key risk:") or lower.startswith("key risks:") or lower.startswith("main risks:"):
            if not fields["main_risks"]:
                fields["main_risks"] = line.split(":", 1)[1].strip()
        elif lower.startswith("why this is likely:") or lower.startswith("why it matters now:") or lower.startswith("key drivers:"):
            if not fields["why_likely"]:
                fields["why_likely"] = line.split(":", 1)[1].strip()
        elif lower.startswith("action plan:"):
            fields["action_plan"] = line.split(":", 1)[1].strip()
        elif lower.startswith("confidence:"):
            fields["confidence"] = line.split(":", 1)[1].strip()
    return fields


def _estimate_quality(snapshot: dict[str, Any], response_text: str) -> dict[str, Any]:
    full_text = response_text or ""
    citation_count = len(re.findall(r"\[S\d+\]", full_text))
    # Count both old and new evidence limitation phrases.
    missing_flags = (
        full_text.lower().count("insufficient custom evidence")
        + full_text.lower().count("limited evidence")
        + full_text.lower().count("limited live data")
    )
    context_chunks = int(snapshot.get("evidence_coverage", {}).get("context_chunks", 0))

    score = 55
    score += min(context_chunks, 8) * 4
    score += min(citation_count, 20)
    score -= min(missing_flags * 4, 24)
    if context_chunks >= 3 and citation_count == 0:
        score -= 20
    elif context_chunks >= 3 and citation_count < 3:
        score -= 10
    score = max(10, min(score, 98))

    if score >= 80:
        band = "HIGH"
    elif score >= 60:
        band = "MEDIUM"
    else:
        band = "LOW"

    return {
        "score": score,
        "band": band,
        "citation_count": citation_count,
        "missing_evidence_flags": missing_flags,
        "context_chunks": context_chunks,
    }


def _make_structured_payload(
    snapshot: dict[str, Any],
    response_text: str,
    model_used: str,
    response_mode: str = "brief",
) -> dict[str, Any]:
    quality = _estimate_quality(snapshot, response_text)
    response_struct = _parse_unified_response(response_text)
    contract_probe = normalize_api_payload({"answer": response_text}, mode=response_mode)

    def _add_point(points: list[dict[str, str]], title: str, text: str) -> None:
        t = (text or "").strip()
        if t:
            points.append({"title": title, "text": t})

    key_points: list[dict[str, str]] = []
    _add_point(key_points, "Executive summary", response_struct.get("executive_summary", ""))
    _add_point(key_points, "Direct answer", response_struct.get("direct_answer", ""))
    _add_point(key_points, "Market impact", response_struct.get("market_impact", ""))
    _add_point(key_points, "Main risks", response_struct.get("main_risks", ""))
    _add_point(key_points, "What to watch next", response_struct.get("watch_next", ""))
    _add_point(key_points, "Action plan", response_struct.get("action_plan", ""))
    return {
        "snapshot": snapshot,
        "response_text": response_text,
        "response_struct": response_struct,
        "key_points": key_points,
        "quality": quality,
        "model_used": model_used or "N/A",
        "_response_contract": contract_probe.get("_response_contract", {}),
    }


def _run_analysis(req: IntelligenceRequest) -> tuple[dict[str, Any], str, str]:
    snapshot = _build_snapshot(req)
    response_text = ""

    try:
        for chunk in macro_intelligence_pipeline(
            req.question,
            req.indicator_overrides,
            geography=req.geography,
            horizon=req.horizon,
            response_mode=req.response_mode,
        ):
            if "▸ RESPONSE" in chunk:
                continue
            if chunk.startswith("━━━") or chunk.strip().startswith("Regime:") or chunk.startswith("━"):
                continue
            response_text += chunk
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return snapshot, response_text.strip(), get_last_model_used()


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/intelligence/snapshot")
def intelligence_snapshot(req: IntelligenceRequest):
    return _build_snapshot(req)


@app.post("/intelligence/analyze")
@_rl(_RL_LLM)
async def intelligence_analyze(req: IntelligenceRequest, request: Request):
    # ── Cache check ──────────────────────────────────────────────────────────
    cache_key = f"{req.question.strip().lower()}|{req.geography}|{req.horizon}|{req.response_mode}"

    async def _compute_intel() -> dict:
        t_start = time.time()
        error_msg: str | None = None
        try:
            snapshot, response_text, model_used = _run_analysis(req)
            payload = _make_structured_payload(
                snapshot,
                response_text,
                model_used,
                response_mode=req.response_mode,
            )
        except HTTPException as exc:
            error_msg = str(exc.detail)
            raise
        except Exception as exc:
            error_msg = str(exc)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
        finally:
            latency_ms = int((time.time() - t_start) * 1000)
            if error_msg is None:
                q = payload.get("quality", {})
                # Compute sentiment on the response text
                _resp_text = payload.get("response", "") or ""
                _sent = score_sentiment(_resp_text)
                log_query(
                    endpoint="/intelligence/analyze",
                    question=req.question,
                    model_used=payload.get("model_used", "N/A"),
                    latency_ms=latency_ms,
                    quality_score=q.get("score"),
                    quality_band=q.get("band"),
                    citation_count=q.get("citation_count", 0),
                    chunk_count=q.get("context_chunks", 0),
                    cache_hit=False,
                    sentiment_label=_sent.get("label"),
                    sentiment_score=_sent.get("score"),
                )
            else:
                log_query(
                    endpoint="/intelligence/analyze",
                    question=req.question,
                    latency_ms=latency_ms,
                    error=error_msg,
                )
        return payload

    val, is_fresh = _intel_guard.cache.get(cache_key)
    if is_fresh:
        logger.info("[intelligence/analyze] cache hit")
        log_query(endpoint="/intelligence/analyze", question=req.question, cache_hit=True)
        return val

    # ── Wait for compute or cache directly ────────────────────────────────────
    return await _intel_guard.get_or_compute(cache_key, _INTEL_CACHE_TTL, _compute_intel)


@app.post("/intelligence/export", response_class=PlainTextResponse)
def intelligence_export(req: IntelligenceRequest):
    snapshot, response_text, model_used = _run_analysis(req)
    s = _make_structured_payload(
        snapshot,
        response_text,
        model_used,
        response_mode=req.response_mode,
    )

    regime = s["snapshot"]["regime"]
    quality = s.get("quality", {})
    lines = [
        "MORNING NOTE",
        "=",
        f"Question: {req.question}",
        f"Regime: {regime['regime']} ({regime['confidence']})",
        f"Cross-Asset Signal: {s['snapshot']['cross_asset']['overall_signal']}",
        f"Quality: {quality.get('band', 'N/A')} ({quality.get('score', 'N/A')}/100) | citations={quality.get('citation_count', 0)}",
        f"Model used: {model_used or 'N/A'}",
        "",
        "RESPONSE",
        response_text,
    ]
    return "\n".join(lines)


@app.post("/intelligence/stream")
@_rl(_RL_LLM)
async def intelligence_stream(req: IntelligenceRequest, request: Request):
    def event_stream():
        PROGRESS_PREFIX = "<<PROGRESS>>"
        try:
            snapshot = _build_snapshot(req)
        except Exception as exc:
            logger.exception("/intelligence/stream snapshot build failed")
            snapshot = _fallback_snapshot(req, f"Snapshot unavailable: {exc}")

        yield _sse("snapshot", snapshot)

        response_text = ""

        try:
            for chunk in macro_intelligence_pipeline(
                req.question,
                req.indicator_overrides,
                geography=req.geography,
                horizon=req.horizon,
                response_mode=req.response_mode,
            ):
                if isinstance(chunk, str) and chunk.startswith(PROGRESS_PREFIX):
                    # Pipeline progress is emitted as its own SSE event so the UI
                    # can show stages without polluting the response text.
                    try:
                        progress_payload = json.loads(chunk[len(PROGRESS_PREFIX) :])
                    except Exception:
                        progress_payload = {"stage": "unknown", "raw": chunk}
                    yield _sse("progress", progress_payload)
                    continue
                if "▸ RESPONSE" in chunk:
                    yield _sse("section_start", {"section": "response"})
                    continue
                if chunk.startswith("━━━") or chunk.startswith("━"):
                    continue

                response_text += chunk
                yield _sse("token", {"section": "response", "text": chunk})

            yield _sse(
                "final",
                _make_structured_payload(
                    snapshot,
                    response_text.strip(),
                    get_last_model_used(),
                    response_mode=req.response_mode,
                ),
            )
        except Exception as exc:
            logger.exception("/intelligence/stream generation failed")
            yield _sse("error", {"message": str(exc), "type": exc.__class__.__name__})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/market_data")
def market_data_live():
    """Return current live indicator values for the dashboard ticker."""
    try:
        live, meta = fetch_live_indicators()
    except Exception as exc:
        return {"error": str(exc), "indicators": {}}
    formatted = {}
    for key, (label, unit) in INDICATOR_META.items():
        val = live.get(key)
        formatted[key] = {
            "label": label,
            "unit": unit,
            "value": round(val, 4) if val is not None else None,
            "source": meta.get(key, {}).get("source", "FRED") if isinstance(meta.get(key), dict) else "FRED",
            "as_of": meta.get(key, {}).get("date", "") if isinstance(meta.get(key), dict) else "",
        }
    return {"indicators": formatted, "count": len(live)}


@app.get("/market_data/stream")
def market_data_live_stream(request: Request):
    """Server-Sent Events stream — progressive partial updates.
    Emits an ``event: update`` for EACH data source as it completes
    (yfinance first, then FRED, then WorldBank) so indicators appear
    on-screen as they arrive rather than all at once.
    Carries ``partial: true/false`` and ``source`` fields.
    After a full cycle, sleeps until the next scheduled refresh.
    """
    prev: dict[str, float] = {}

    sleep_open_s = int(os.getenv("MARKET_DATA_STREAM_SLEEP_OPEN_SEC", "60"))
    sleep_closed_s = int(os.getenv("MARKET_DATA_STREAM_SLEEP_CLOSED_SEC", "120"))

    def _format_live(live: dict, meta: dict) -> dict:
        """Build the indicator payload dict from a (possibly partial) live dict."""
        formatted: dict[str, Any] = {}
        for key, (label, unit) in INDICATOR_META.items():
            val = live.get(key)
            if val is None:
                continue   # omit keys not yet available in this partial
            prev_val = prev.get(key)
            direction = "flat"
            change    = None
            if prev_val is not None:
                direction = "up" if val > prev_val else ("down" if val < prev_val else "flat")
                change    = round(val - prev_val, 6)
            formatted[key] = {
                "label":     label,
                "unit":      unit,
                "value":     round(val, 4),
                "direction": direction,
                "change":    change,
            }
            prev[key] = val
        return formatted

    async def event_gen():
        nonlocal prev
        while True:
            try:
                for live, meta in stream_live_indicators():
                    formatted = _format_live(live, meta)
                    sleep_s   = sleep_open_s if any_price_market_open() else sleep_closed_s
                    payload   = {
                        "indicators":        formatted,
                        "timestamp":         datetime.utcnow().isoformat() + "Z",
                        "count":             len(formatted),
                        "partial":           meta.get("partial", False),
                        "source":            meta.get("source"),
                        "completed_sources": meta.get("completed_sources", []),
                        "pending_sources":   meta.get("pending_sources", []),
                        "fetch_ms":          meta.get("fetch_ms"),
                        "from_cache":        meta.get("from_cache", False),
                        "sleep_interval_s":  sleep_s,
                    }
                    yield f"event: update\ndata: {json.dumps(payload)}\n\n"
            except Exception as exc:
                yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"
            sleep_s = sleep_open_s if any_price_market_open() else sleep_closed_s
            yield f"event: tick\ndata: {json.dumps({'next_refresh_s': sleep_s, 'timestamp': datetime.utcnow().isoformat() + 'Z'})}\n\n"
            await asyncio.sleep(sleep_s)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/market_data/status")
def market_data_status():
    """Returns open/closed status for all exchange groups and active cache TTLs."""
    return market_status_summary()


@app.get("/health")
def health_check():
    """Liveness + readiness probe. Returns Ollama connectivity and index status."""
    from rag.query import load_index, load_metadata
    import requests as _req

    # Check Ollama
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_ok = False
    try:
        r = _req.get(f"{ollama_url}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    # Check FAISS index
    index_ok = False
    index_size = 0
    try:
        idx = load_index()
        index_size = idx.ntotal
        index_ok = index_size > 0
    except Exception:
        pass

    status = "ok" if (ollama_ok and index_ok) else "degraded"
    return {
        "status": status,
        "ollama": {"reachable": ollama_ok, "url": ollama_url},
        "faiss_index": {"loaded": index_ok, "vectors": index_size},
        "embed_cache": embed_cache_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/admin/reload-index")
def reload_index():
    """Force-reload the FAISS index and live data cache.
    Call this after running run_rss_ingest.py or any index rebuild script
    so the running API immediately picks up new vectors without a restart.
    """
    invalidate_index_cache()
    invalidate_live_data_cache()
    logger.info("Admin reload-index triggered.")
    return {"status": "ok", "message": "Index cache and live data cache cleared. Next request will reload from disk."}


# ── News System Health Endpoints ───────────────────────────────────────────────

@app.get("/news/health")
def news_health_full():
    """
    Full real-time health check of the news fetching system.
    Checks ALL configured RSS feeds (may take 20-40s).
    Returns:
        - Per-feed reachability, parsability, and freshness
        - Vector index age and size
        - Ollama LLM availability
        - Overall health score (0-100) and status (healthy/degraded/critical)
    """
    try:
        report = check_news_health(sample_feeds=0)
        return {
            "status": report.overall_status,
            "health_score": report.health_score,
            "checked_at": report.checked_at,
            "feeds": {
                "total": report.total_feeds,
                "reachable": report.reachable_feeds,
                "parsable": report.parsable_feeds,
                "fresh_lt_24h": report.fresh_feeds,
                "total_recent_articles": report.total_recent_articles,
                "avg_latency_ms": report.avg_latency_ms,
                "unreachable": report.unreachable_feeds[:10],
                "stale": report.stale_feeds[:10],
                "errors": report.error_feeds[:10],
            },
            "vector_index": {
                "status": report.index_status,
                "age_hours": report.index_last_modified_hours,
                "vector_count": report.index_vector_count,
            },
            "llm": {
                "ollama_reachable": report.ollama_reachable,
                "models": report.ollama_models,
            },
            "warnings": report.warnings,
        }
    except Exception as exc:
        logger.error("[/news/health] check failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Health check failed: {exc}")


@app.get("/news/health/quick")
def news_health_quick():
    """
    Quick health check (2 feeds/category, ~5-10s).
    Suitable for frequent monitoring / liveness probes.
    """
    try:
        report = check_news_health_quick()
        return {
            "status": report.overall_status,
            "health_score": report.health_score,
            "checked_at": report.checked_at,
            "feeds": {
                "total": report.total_feeds,
                "reachable": report.reachable_feeds,
                "fresh_lt_24h": report.fresh_feeds,
            },
            "vector_index": {
                "status": report.index_status,
                "age_hours": report.index_last_modified_hours,
                "vector_count": report.index_vector_count,
            },
            "llm": {
                "ollama_reachable": report.ollama_reachable,
            },
            "warnings": report.warnings,
        }
    except Exception as exc:
        logger.error("[/news/health/quick] check failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Health check failed: {exc}")


@app.get("/metrics")
def get_metrics():
    """
    Aggregate quality and performance metrics derived from the query log.

    Returns:
      - total_queries, cache_hit_rate_pct, error_rate_pct
      - avg_quality_score, quality_band_dist
      - avg_latency_ms, p95_latency_ms
      - model_distribution (which model served N% of requests)
    """
    entries = read_recent(n=2000)
    return compute_metrics(entries)


class FeedbackRequest(BaseModel):
    question: str
    answer_snippet: str = ""   # first 300 chars of answer for context
    rating: int                # 1 (bad) – 5 (excellent)
    comment: str = ""


@app.post("/feedback", status_code=201)
def submit_feedback(req: FeedbackRequest):
    """
    Collect user feedback on an answer.  Stored to data/feedback_log.jsonl.
    Use this to build a curated evaluation set and track quality regression.
    """
    if not 1 <= req.rating <= 5:
        raise HTTPException(status_code=422, detail="rating must be between 1 and 5")
    entry = {
        "ts":             datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "question":       req.question[:300],
        "answer_snippet": req.answer_snippet[:300],
        "rating":         req.rating,
        "comment":        req.comment[:1000],
    }
    try:
        os.makedirs(os.path.dirname(os.path.abspath(_FEEDBACK_PATH)), exist_ok=True)
        with _feedback_lock:
            with open(_FEEDBACK_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.error("[feedback] write failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to persist feedback") from exc
    return {"status": "ok", "message": "Feedback recorded. Thank you."}


@app.get("/feedback/summary")
def feedback_summary():
    """Return aggregate feedback statistics and the 20 most recent entries."""
    if not os.path.exists(_FEEDBACK_PATH):
        return {"message": "No feedback recorded yet", "entries": []}
    entries: list[dict] = []
    try:
        with _feedback_lock:
            with open(_FEEDBACK_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read feedback: {exc}") from exc

    if not entries:
        return {"message": "No feedback recorded yet", "entries": []}

    ratings = [e["rating"] for e in entries]
    avg_rating = round(sum(ratings) / len(ratings), 2)
    dist = {str(i): ratings.count(i) for i in range(1, 6)}
    return {
        "total_feedback": len(entries),
        "avg_rating":     avg_rating,
        "rating_distribution": dist,
        "recent": entries[-20:],
    }


# ── Company Fundamentals Endpoint ────────────────────────────────────────────────

@app.get("/fundamentals/{ticker}")
def fundamentals_single(ticker: str, refresh: bool = False):
    """
    Return company fundamental data for a single *ticker* (e.g. AAPL, MSFT).

    Data is disk-cached for 6 hours by default (override with
    FUNDAMENTALS_CACHE_TTL env var in seconds).

    Pass ?refresh=true to force a live yfinance pull.

    Response includes: valuation ratios, profitability, balance-sheet metrics,
    analyst estimates and last 4 quarterly earnings.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=422, detail="ticker must not be empty")
    try:
        data = get_fundamentals(ticker, force_refresh=refresh)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fundamentals fetch failed: {exc}") from exc
    if "error" in data:
        raise HTTPException(
            status_code=502,
            detail=f"Could not retrieve fundamentals for {ticker}: {data['error']}",
        )
    return data


@app.post("/fundamentals/batch")
def fundamentals_batch(tickers: list[str]):
    """
    Fetch fundamentals for multiple tickers in one call.
    Body: JSON array of ticker strings, e.g. ["AAPL","MSFT","NVDA"].
    Maximum 10 tickers per request.
    """
    if not tickers:
        raise HTTPException(status_code=422, detail="tickers list must not be empty")
    if len(tickers) > 10:
        raise HTTPException(status_code=422, detail="Maximum 10 tickers per request")
    try:
        return get_batch_fundamentals(tickers)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch fundamentals fetch failed: {exc}") from exc


# ── Hedge / Risk-Signal Endpoint ────────────────────────────────────────────────

def _build_hedge_prompt(
    question: str,
    regime: dict[str, Any],
    cross_asset: dict[str, Any],
    indicators: dict[str, float],
    fundamentals_text: str,
    geography: str,
    horizon: str,
) -> str:
    """
    Build a hedge-focused LLM prompt that combines macro regime, cross-asset
    signals, live indicators, and any supplied company fundamentals.
    """
    # Compact indicator summary
    _MAP = [
        ("sp500", "S&P500"), ("vix", "VIX"), ("dxy", "DXY"),
        ("yield_10y", "10Y%"), ("yield_2y", "2Y%"), ("yield_curve", "Curve_bps"),
        ("inflation_cpi", "CPI%"), ("fed_funds_rate", "FedFunds%"),
        ("oil_wti", "WTI$"), ("gold", "Gold$"), ("credit_hy", "HY_bps"),
    ]
    num_parts = [f"{lbl}={indicators[k]}" for k, lbl in _MAP if k in indicators]
    num_line = ", ".join(num_parts[:9]) or "No live data."

    return f"""You are a senior portfolio risk manager.
Your task: provide specific, actionable hedging strategies for the portfolio concern described below.

Current market regime: {regime.get('regime', 'UNKNOWN')} (confidence: {regime.get('confidence', 'N/A')})
Cross-asset signal: {cross_asset.get('overall_signal', 'N/A')}
Live indicators: {num_line}
Geography focus: {geography} | Investment horizon: {horizon}

Portfolio / concern:
{question}

{f'Portfolio holdings fundamentals:{chr(10)}{fundamentals_text}{chr(10)}' if fundamentals_text.strip() else ''}
Instructions:
1. Tailor hedges to the current regime ({regime.get('regime')}) and cross-asset signal.
2. Include instrument-specific suggestions (e.g. put options, inverse ETFs, gold, Treasuries, VIX calls).
3. Size guidance: suggest approximate allocation % per hedge.
4. Cover both tail-risk and everyday volatility management.
5. Never invent facts — if unsure, say so.

Respond in EXACTLY this format:
Executive summary: <2 sentences summarising the key risk and approach>
Primary risks identified:
- <risk 1 with supporting indicator or regime evidence>
- <risk 2>
- <risk 3>
Recommended hedges:
- Instrument: <name/ticker>  |  Rationale: <why>  |  Suggested allocation: <x%>
- Instrument: <name/ticker>  |  Rationale: <why>  |  Suggested allocation: <x%>
- Instrument: <name/ticker>  |  Rationale: <why>  |  Suggested allocation: <x%>
Portfolio adjustments:
- <positioning change 1>
- <positioning change 2>
Scenario stress-test:
- Bear case: <outcome if hedges fail>
- Bull case: <cost of hedges if market rises>
Confidence: <HIGH/MEDIUM/LOW> - <reason>
Answer:""".strip()


@app.post("/intelligence/hedge")
@_rl(_RL_LLM)
def intelligence_hedge(req: HedgeRequest, request: Request):
    """
    Generate hedging and risk-mitigation suggestions based on:
      - Current macro regime + cross-asset signal
      - Live indicator snapshot
      - Optional company-level fundamentals for supplied tickers

    Blueprint alignment: Section 2 – Risk & Portfolio Optimisation.
    """
    # 1. Fetch indicators + regime
    try:
        live_data, _meta = fetch_live_indicators()
    except Exception:
        live_data = {}
    from_q = extract_indicators_from_text(req.question)
    indicators = {**live_data, **from_q}
    regime = detect_regime(**get_regime_inputs_from_indicators(indicators))
    cross_asset = analyze_cross_asset(indicators)

    # 2. Fetch fundamentals for portfolio tickers (if provided)
    fundamentals_text = ""
    fundamentals_data: dict[str, Any] = {}
    if req.tickers:
        try:
            fundamentals_data = get_batch_fundamentals([t.upper() for t in req.tickers])
            lines = [format_fundamentals_summary(v) for v in fundamentals_data.values() if "error" not in v]
            fundamentals_text = "\n".join(lines)
        except Exception as exc:
            logger.warning("[/intelligence/hedge] fundamentals fetch error: %s", exc)

    # 3. Build hedge prompt + call LLM
    import requests as _req
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    num_ctx     = int(os.getenv("LLM_NUM_CTX",     "4096"))
    num_predict = int(os.getenv("LLM_NUM_PREDICT", "800"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.15"))

    from intelligence.model_router import get_model_candidates
    prompt = _build_hedge_prompt(
        req.question, regime, cross_asset, indicators,
        fundamentals_text, req.geography, req.horizon,
    )

    response_text = ""
    model_used = "N/A"
    hedge_error: str | None = None
    t_start = time.time()
    for model in get_model_candidates():
        try:
            resp = _req.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model, "prompt": prompt, "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_ctx": num_ctx, "num_predict": num_predict,
                        "top_p": 0.90, "repeat_penalty": 1.15,
                    },
                },
                timeout=float(os.getenv("OLLAMA_GENERATE_TIMEOUT_SEC", "90")),
            )
            resp.raise_for_status()
            txt = (resp.json().get("response") or "").strip()
            if txt:
                response_text = txt
                model_used = model
                break
        except Exception as exc:
            hedge_error = str(exc)
            continue

    latency_ms = int((time.time() - t_start) * 1000)
    if not response_text:
        response_text = (
            f"Executive summary: Unable to generate hedging suggestions at this time.\n"
            f"Primary risks identified:\n"
            f"- Regime: {regime.get('regime', 'UNKNOWN')} — monitor closely.\n"
            f"- Cross-asset signal: {cross_asset.get('overall_signal', 'N/A')}\n"
            f"Recommended hedges:\n"
            f"- Instrument: Gold (GLD)  |  Rationale: Safe-haven in risk-off regimes  |  Suggested allocation: 5%\n"
            f"- Instrument: US Treasuries (IEF/TLT)  |  Rationale: Rate risk buffer  |  Suggested allocation: 10%\n"
            f"- Instrument: VIX calls  |  Rationale: Volatility spike protection  |  Suggested allocation: 2%\n"
            f"Portfolio adjustments:\n"
            f"- Reduce equity beta exposure when VIX > 20\n"
            f"- Increase cash buffer in late-cycle regimes\n"
            f"Scenario stress-test:\n"
            f"- Bear case: Hedges cushion 30-40% of portfolio drawdown\n"
            f"- Bull case: Cost of carry ~2-3% p.a. for standard hedge basket\n"
            f"Confidence: LOW - LLM generation unavailable."
        )

    log_query(
        endpoint="/intelligence/hedge",
        question=req.question,
        model_used=model_used,
        latency_ms=latency_ms,
        error=hedge_error if not response_text else None,
    )

    return {
        "question":        req.question,
        "regime":          regime,
        "cross_asset":     cross_asset,
        "portfolio_tickers": req.tickers,
        "fundamentals":    fundamentals_data,
        "response_text":   response_text,
        "model_used":      model_used,
        "latency_ms":      latency_ms,
        "geography":       req.geography,
        "horizon":         req.horizon,
    }


# ── Sector Comparison ── (Blueprint §10: Sector comparison) ───────────────────
@app.get("/intelligence/sectors")
def intelligence_sectors(geography: str = "US", horizon: str = "MEDIUM_TERM"):
    """
    Return regime-aware sector rankings (OVERWEIGHT / UNDERWEIGHT / NEUTRAL).

    Automatically pulls live regime from the macro pipeline so the output is
    always tied to the current market environment.

    Blueprint reference: Section 10 — "Sector comparison"
    """
    from intelligence.sector_mapper import sector_impact
    try:
        live_data = fetch_live_indicators()
        indicators = live_data.get("indicators", {})
        regime_inputs = get_regime_inputs_from_indicators(indicators)
        regime = detect_regime(regime_inputs)
        regime_label = regime.get("regime", "UNKNOWN")
        cross_asset = analyze_cross_asset(indicators)
        sector_text = sector_impact(
            macro_analysis=cross_asset.get("overall_signal", ""),
            regime=regime_label,
            mcx_tickers=indicators,
        )
        return {
            "regime":          regime_label,
            "regime_confidence": regime.get("confidence", "LOW"),
            "geography":       geography,
            "horizon":         horizon,
            "sector_analysis": sector_text,
            "overall_signal":  cross_asset.get("overall_signal", "NEUTRAL"),
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.warning("[/intelligence/sectors] error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Sector analysis failed: {exc}") from exc


# ── Market Snapshot ── (Blueprint §3D: Market Data) ───────────────────────────
@app.get("/market/snapshot")
def market_snapshot(refresh: bool = False):
    """
    Return a compact real-time market snapshot for key global indices,
    commodities, currencies, and crypto.  Cached for 15 minutes by default
    (set MARKET_DATA_CACHE_TTL env var to change).

    Blueprint reference: Section 3D — "Market Data"
    """
    try:
        snapshot = get_market_snapshot()
        text = format_snapshot_for_prompt(snapshot)
        return {
            "snapshot":   snapshot,
            "prompt_text": text,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.warning("[/market/snapshot] error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Market snapshot failed: {exc}") from exc


# ── Sentiment Analysis ── (Blueprint §10: Sentiment analysis) ─────────────────
class SentimentRequest(BaseModel):
    text: str
    texts: list[str] = Field(default_factory=list)


@app.post("/intelligence/sentiment")
def intelligence_sentiment(req: SentimentRequest):
    """
    Analyse the financial sentiment of one or more text snippets.
    Returns bullish / bearish / neutral labels with confidence scores and
    the specific keyword signals that drove the classification.

    Blueprint reference: Section 10 — "Sentiment analysis"
    """
    if req.texts:
        from intelligence.sentiment_analyzer import batch_score_sentiment
        results = batch_score_sentiment(req.texts)
        labels = [r["label"] for r in results]
        pos = labels.count("positive")
        neg = labels.count("negative")
        neu = labels.count("neutral")
        avg_score = sum(r["score"] for r in results) / len(results)
        return {
            "results":       results,
            "aggregate": {
                "positive_count": pos,
                "negative_count": neg,
                "neutral_count":  neu,
                "avg_score":      round(avg_score, 4),
                "overall_label":  "positive" if avg_score > 0.10 else ("negative" if avg_score < -0.10 else "neutral"),
            },
        }
    result = score_sentiment(req.text)
    return result


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


# ── Agentic RAG Endpoints ─────────────────────────────────────────────────────


class AgenticRequest(BaseModel):
    """Request body for the Agentic RAG endpoints."""
    question: str
    geography: str = "US"
    horizon: str = "MEDIUM_TERM"
    response_mode: str = "detailed"
    max_iterations: int = Field(default=2, ge=1, le=4)
    bloomberg_format: str = "morning_note"  # morning_note | risk_matrix | trade_idea | brief


@app.post("/intelligence/agentic_analyze")
@_rl(_RL_LLM)
async def agentic_analyze(req: AgenticRequest, request: Request):
    """
    Bloomberg-grade Agentic RAG endpoint with full Plan→Act→Observe→Reflect loop.

    Streams Server-Sent Events at each pipeline stage:
      event: planning      — Sub-question decomposition
      event: retrieval     — Context chunks + live market data fetched
      event: agent_brief   — Each specialist agent's analysis
      event: reflection    — Gaps identified + follow-up queries
      event: final         — Complete structured output + audit trace

    Response mode 'detailed' produces a Bloomberg Morning Note.
    Response mode 'brief' produces a compact summary.
    """
    from intelligence.agentic_rag import AgenticOrchestrator

    orchestrator = AgenticOrchestrator(max_iterations=req.max_iterations)

    async def event_stream():
        t_start = time.time()
        try:
            async for event in orchestrator.run_async(
                question=req.question,
                geography=req.geography,
                horizon=req.horizon,
                response_mode=req.response_mode,
            ):
                yield event.to_sse()
        except Exception as exc:
            logger.exception("/intelligence/agentic_analyze failed")
            import json as _json
            yield f"event: error\ndata: {_json.dumps({'message': str(exc), 'type': exc.__class__.__name__})}\n\n"
        finally:
            latency_ms = int((time.time() - t_start) * 1000)
            log_query(
                endpoint="/intelligence/agentic_analyze",
                question=req.question,
                latency_ms=latency_ms,
                cache_hit=False,
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/intelligence/trace")
async def intelligence_trace(req: AgenticRequest):
    """
    Run a full agentic analysis and return the complete audit trace (non-streaming).

    Useful for debugging, observability, and compliance audit.
    Returns agent outputs, tool calls, gaps found, and timing per stage.
    """
    from intelligence.agentic_rag import AgenticOrchestrator, AgentState

    orchestrator = AgenticOrchestrator(max_iterations=req.max_iterations)
    trace_data: dict[str, Any] = {}
    final_answer: str = ""

    try:
        async for event in orchestrator.run_async(
            question=req.question,
            geography=req.geography,
            horizon=req.horizon,
            response_mode=req.response_mode,
        ):
            if event.stage == "final":
                trace_data = event.data.get("trace", {})
                final_answer = event.data.get("answer", "")
    except Exception as exc:
        logger.exception("/intelligence/trace failed")
        raise HTTPException(status_code=500, detail=f"Agentic trace failed: {exc}") from exc

    return {
        "question": req.question,
        "answer_preview": final_answer[:500],
        "trace": trace_data,
    }


@app.post("/intelligence/bloomberg")
@_rl(_RL_LLM)
async def intelligence_bloomberg(req: AgenticRequest, request: Request):
    """
    Run the existing analysis pipeline and return Bloomberg-formatted output.

    Wraps /intelligence/analyze with the BloombergFormatter.
    Formats: morning_note (default), risk_matrix, trade_idea, brief.
    """
    from intelligence.bloomberg_formatter import BloombergFormatter

    # Re-use existing analyze pipeline
    try:
        snapshot, response_text, model_used = _run_analysis(
            IntelligenceRequest(
                question=req.question,
                geography=req.geography,
                horizon=req.horizon,
                response_mode=req.response_mode,
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    formatter = BloombergFormatter()
    live_indicators: dict[str, Any] = {}
    try:
        from intelligence.live_market_data import fetch_live_indicators
        live_indicators_raw, _ = fetch_live_indicators()
        live_indicators = {k: v for k, v in live_indicators_raw.items() if v is not None}
    except Exception:
        pass

    formatted = formatter.format(
        mode=req.bloomberg_format or "morning_note",
        answer=response_text,
        indicators=live_indicators,
        regime=snapshot.get("regime", {}),
        cross_asset=snapshot.get("cross_asset", {}),
        question=req.question,
        geography=req.geography,
        horizon=req.horizon,
        model_used=model_used,
    )

    return {
        "question": req.question,
        "bloomberg_format": req.bloomberg_format,
        "formatted_output": formatted,
        "model_used": model_used,
        "quality": _estimate_quality(snapshot, response_text),
    }


@app.get("/", response_class=FileResponse)
def dashboard():
    return FileResponse("api/static/index.html")
