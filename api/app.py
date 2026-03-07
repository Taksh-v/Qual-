from __future__ import annotations

import json
import logging
import logging.config
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.rag_core import ask_rag
from rag.query import invalidate_index_cache
from intelligence.cross_asset_analyzer import analyze_cross_asset
from intelligence.indicator_parser import (
    extract_indicators_from_text,
    get_regime_inputs_from_indicators,
)
from intelligence.context_retriever import retrieve_relevant_context
from intelligence.macro_engine import macro_intelligence_pipeline, get_last_model_used
from intelligence.live_market_data import fetch_live_indicators, invalidate_live_data_cache
from intelligence.question_classifier import classify_question
from intelligence.regime_detector import detect_regime
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

app.mount("/static", StaticFiles(directory="api/static"), name="static")

logger.info("News Intelligence RAG API starting up.")


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


class IntelligenceRequest(BaseModel):
    question: str
    geography: str = "US"
    horizon: str = "MEDIUM_TERM"
    response_mode: str = "brief"
    indicator_overrides: dict[str, float] = Field(default_factory=dict)



# INDICATOR_META is now maintained in config/indicators.py and imported above.


@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    try:
        return ask_rag(req.question)
    except Exception as exc:
        return {
            "question": req.question,
            "answer": (
                "RAG backend is currently degraded. "
                f"Fallback message: {exc}. "
                "Check Ollama/index availability."
            ),
            "sources": [],
        }


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


def _collect_indicator_inputs(req: IntelligenceRequest) -> tuple[dict[str, float], list[dict[str, Any]]]:
    overrides = _normalized_overrides(req.indicator_overrides)
    context_chunks: list[dict[str, Any]] = []
    context_text = ""
    try:
        context_chunks = retrieve_relevant_context(req.question, top_k=25, keep_latest=15)
        context_text = " ".join(c.get("text", "") for c in context_chunks)
    except Exception:
        context_chunks = []
        context_text = ""

    # Fetch live FRED data first (authoritative baseline)
    try:
        live_data, _meta = fetch_live_indicators()
    except Exception:
        live_data = {}

    from_context = extract_indicators_from_text(context_text)
    from_question = extract_indicators_from_text(req.question)
    # Priority: overrides > extracted from text/question > live FRED
    all_indicators = {**live_data, **from_context, **from_question, **overrides}
    return all_indicators, context_chunks


def _build_snapshot(req: IntelligenceRequest) -> dict[str, Any]:
    classification = classify_question(req.question)
    all_indicators, context_chunks = _collect_indicator_inputs(req)
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
        "evidence_coverage": {
            "context_chunks": len(context_chunks),
            "has_overrides": bool(overrides),
            "sources": sources,
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


def _make_structured_payload(snapshot: dict[str, Any], response_text: str, model_used: str) -> dict[str, Any]:
    quality = _estimate_quality(snapshot, response_text)
    response_struct = _parse_unified_response(response_text)
    return {
        "snapshot": snapshot,
        "response_text": response_text,
        "response_struct": response_struct,
        "quality": quality,
        "model_used": model_used or "N/A",
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
def intelligence_analyze(req: IntelligenceRequest):
    snapshot, response_text, model_used = _run_analysis(req)
    return _make_structured_payload(snapshot, response_text, model_used)


@app.post("/intelligence/export", response_class=PlainTextResponse)
def intelligence_export(req: IntelligenceRequest):
    snapshot, response_text, model_used = _run_analysis(req)
    s = _make_structured_payload(snapshot, response_text, model_used)

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
def intelligence_stream(req: IntelligenceRequest):
    snapshot = _build_snapshot(req)

    def event_stream():
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
                if "▸ RESPONSE" in chunk:
                    yield _sse("section_start", {"section": "response"})
                    continue
                if chunk.startswith("━━━") or chunk.strip().startswith("Regime:") or chunk.startswith("━"):
                    continue

                response_text += chunk
                yield _sse("token", {"section": "response", "text": chunk})

            yield _sse("final", _make_structured_payload(snapshot, response_text.strip(), get_last_model_used()))
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

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
def market_data_live_stream():
    """Server-Sent Events stream — pushes live market data every 30 seconds.
    Emits ``event: update`` with direction/change vs previous tick.
    """
    prev: dict[str, float] = {}

    def event_gen():
        nonlocal prev
        while True:
            try:
                live, meta = fetch_live_indicators()
            except Exception as exc:
                yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"
                time.sleep(30)
                continue

            formatted: dict[str, Any] = {}
            for key, (label, unit) in INDICATOR_META.items():
                val = live.get(key)
                prev_val = prev.get(key)
                direction = "flat"
                change = None
                if val is not None and prev_val is not None:
                    if val > prev_val:
                        direction = "up"
                    elif val < prev_val:
                        direction = "down"
                    change = round(val - prev_val, 6)
                formatted[key] = {
                    "label": label,
                    "unit": unit,
                    "value": round(val, 4) if val is not None else None,
                    "direction": direction,
                    "change": change,
                    "source": (
                        meta.get(key, {}).get("source", "FRED")
                        if isinstance(meta.get(key), dict)
                        else "FRED"
                    ),
                    "as_of": (
                        meta.get(key, {}).get("date", "")
                        if isinstance(meta.get(key), dict)
                        else ""
                    ),
                }
                if val is not None:
                    prev[key] = val

            payload = {
                "indicators": formatted,
                "timestamp": datetime.utcnow().isoformat() + "Z",    # UTC — browser converts to local tz
                "count": len(live),
            }
            yield f"event: update\ndata: {json.dumps(payload)}\n\n"
            time.sleep(30)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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


@app.get("/", response_class=FileResponse)
def dashboard():
    return FileResponse("api/static/index.html")
