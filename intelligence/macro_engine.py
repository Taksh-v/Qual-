"""
macro_engine.py
---------------
Orchestrates the full macro intelligence pipeline, integrating all modules.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from threading import local as threading_local
from typing import Iterator, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Module Imports
from intelligence.question_classifier import classify_question
from intelligence.indicator_parser import extract_indicators_from_text, get_regime_inputs_from_indicators
from intelligence.regime_detector import detect_regime
from intelligence.cross_asset_analyzer import analyze_cross_asset
from intelligence.context_retriever import retrieve_relevant_context, format_context
from intelligence.data_quality import evaluate_retrieval_quality
from intelligence.live_market_data import fetch_live_indicators
from intelligence.model_router import get_model_candidates
from intelligence.reasoning_layer import build_reasoning_analysis
from intelligence.utils import tokenize, grounding_score, numeric_hallucination_risk
from intelligence.macro_reasoner import (
    build_citation_repair_prompt,
    build_quality_rewrite_prompt,
    build_unified_response_prompt,
    generate_contextual_fallback,
    generate_unified_fallback,
)
from intelligence.response_enhancer import enhance_response, score_response

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Configuration ---
LLM_CONNECT_TIMEOUT_SEC = os.getenv("OLLAMA_CONNECT_TIMEOUT_SEC", "5")
# Keep macro generation on a short leash by default; callers can override via env.
LLM_MAX_TIME_SEC = os.getenv("INTEL_GENERATE_TIMEOUT_SEC", "45")
INTEL_MAX_MODEL_CANDIDATES = max(1, int(os.getenv("INTEL_MAX_MODEL_CANDIDATES", "1")))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
APP_BRAND = os.getenv("APP_BRAND_NAME", "Macro AI")
INTEL_STRICT_REASONING_GUARD = _env_flag("INTEL_STRICT_REASONING_GUARD", True)
INTEL_MIN_GROUNDING_SCORE = float(os.getenv("INTEL_MIN_GROUNDING_SCORE", "0.22"))
INTEL_MAX_NUMERIC_RISK = float(os.getenv("INTEL_MAX_NUMERIC_RISK", "0.35"))
INTEL_REPAIR_MIN_LINE_OVERLAP = float(os.getenv("INTEL_REPAIR_MIN_LINE_OVERLAP", "0.18"))

# Thread-local storage so concurrent requests don't clobber each other's state.
_tl = threading_local()


def _tl_model() -> str:
    return getattr(_tl, "model_used", "N/A")


def _tl_status() -> str:
    return getattr(_tl, "generation_status", "not_started")


def _tl_error() -> str:
    return getattr(_tl, "llm_error", "")


def _collect_llm_text(prompt: str) -> str:
    last_error: str | None = None
    for model in get_model_candidates()[:INTEL_MAX_MODEL_CANDIDATES]:
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        # Lower temperature → more factual, less hallucination
                        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.15")),
                        # Increased tokens for complete structured responses
                        "num_predict": int(os.getenv("LLM_NUM_PREDICT", "800")),
                        # Context window — 4096 balances memory vs. context coverage
                        "num_ctx": int(os.getenv("LLM_NUM_CTX", "4096")),
                        # Nucleus sampling: prevents degenerate outputs
                        "top_p": float(os.getenv("LLM_TOP_P", "0.90")),
                        # Top-k: focus on high-probability tokens for factual tasks
                        "top_k": int(os.getenv("LLM_TOP_K", "40")),
                        # Penalise repetition — prevents looping sections
                        "repeat_penalty": float(os.getenv("LLM_REPEAT_PENALTY", "1.15")),
                        # Penalise tokens already present — increases response diversity
                        "presence_penalty": float(os.getenv("LLM_PRESENCE_PENALTY", "0.1")),
                    },
                },
                timeout=float(LLM_MAX_TIME_SEC),
            )
            response.raise_for_status()
            text = (response.json().get("response") or "").strip()
            if text:
                _tl.model_used = model
                _tl.generation_status = "llm_ok"
                return text
            last_error = f"empty_response:{model}"
        except Exception as exc:
            last_error = f"{model}:{exc}"
            _tl.llm_error = (last_error or "")[:220]
            logger.debug("[macro_engine] LLM call failed for model %s: %s", model, exc)
            continue
    _tl.model_used = "N/A"
    _tl.generation_status = "llm_failed"
    _tl.llm_error = (last_error or "")[:220]
    raise RuntimeError(last_error or "LLM generation failed for all model candidates.")



def get_last_model_used() -> str:
    """Returns a human-readable description of the model used in the last pipeline call.
    NOTE: This reads thread-local state, so it is accurate for the current request thread.
    """
    status = _tl_status()
    error = _tl_error()
    model = _tl_model()
    if status == "llm_ok":
        return model or "N/A"
    if status in ("llm_failed", "fallback_llm_failed"):
        if "Read timed out" in error or "timed out" in error:
            return "fallback (llm_timeout)"
        if "404" in error or "not found" in error.lower():
            return "fallback (model_not_found)"
        if "Connection" in error or "Failed to establish" in error:
            return "fallback (ollama_unreachable)"
        if error.startswith("empty_response"):
            return "fallback (empty_response)"
        return "fallback (llm_failed)"
    if status.startswith("fallback_"):
        return f"fallback ({status.replace('fallback_', '')})"
    return model or "N/A"


def _improve_quality(section_name: str, raw_text: str) -> str:
    if not raw_text:
        return raw_text
    polish_prompt = build_quality_rewrite_prompt(section_name, raw_text)
    # Snapshot thread-local status so a failed polish doesn't corrupt the
    # primary generation status that was already set by _collect_llm_text.
    saved_model = _tl_model()
    saved_status = _tl_status()
    saved_error = _tl_error()
    try:
        polished = _collect_llm_text(polish_prompt)
        return polished or raw_text
    except Exception:
        _tl.model_used = saved_model
        _tl.generation_status = saved_status
        _tl.llm_error = saved_error
        return raw_text


def _valid_response(text: str) -> bool:
    """Accept any response that contains a direct answer plus at least one
    substantive section. Strict label matching was causing valid LLM responses
    to be rejected and replaced by the deterministic fallback.
    """
    if not text or len(text.strip()) < 60:
        return False
    has_answer = "Direct answer:" in text or "Executive summary:" in text
    has_body = (
        "Market impact:" in text
        or "What is happening:" in text
        or "Scenarios" in text
        or "Consequences" in text
        or "Key risk" in text
        or "Why it matters" in text
        or "Causal chain:" in text
    )
    has_prediction = "Predicted events:" in text
    return has_answer and has_body and has_prediction


def _citation_count(text: str) -> int:
    return text.count("[S")


def _tokenize(text: str) -> set[str]:
    """Delegate to shared utility."""
    return tokenize(text)


def _response_grounding_score(answer: str, chunks: list[dict[str, Any]]) -> float:
    """Delegate to shared utility."""
    return grounding_score(answer, chunks)


def _numeric_hallucination_risk(answer: str, chunks: list[dict[str, Any]]) -> float:
    """Delegate to shared utility."""
    return numeric_hallucination_risk(answer, chunks)


def _best_claim_overlap(line: str, chunks: list[dict[str, Any]]) -> float:
    claim = re.sub(r"\[S\d+\]", "", line)
    claim_tokens = _tokenize(claim)
    cites = [int(x) for x in re.findall(r"\[S(\d+)\]", line)]
    if not claim_tokens or not cites:
        return 0.0
    best = 0.0
    for c in cites:
        if 1 <= c <= len(chunks):
            src_tokens = _tokenize(chunks[c - 1].get("text", ""))
            if src_tokens:
                overlap = len(claim_tokens.intersection(src_tokens)) / max(1, len(claim_tokens))
                best = max(best, overlap)
    return best


def _sanitize_unsupported_cited_numbers(answer: str, chunks: list[dict[str, Any]]) -> str:
    if not answer or not chunks:
        return answer
    out_lines: list[str] = []
    for raw in answer.splitlines():
        line = raw
        cites = [int(x) for x in re.findall(r"\[S(\d+)\]", raw)]
        if not cites:
            out_lines.append(line)
            continue
        corpus = " ".join(
            chunks[c - 1].get("text", "")
            for c in cites
            if 1 <= c <= len(chunks)
        )
        if not corpus:
            out_lines.append(line)
            continue

        nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", line)
        for n in nums:
            if not (n.endswith("%") or (n.replace(".", "", 1).isdigit() and float(n) >= 100)):
                continue
            if n not in corpus:
                line = re.sub(rf"\b{re.escape(n)}\b", "N/A", line)
        out_lines.append(line)
    return "\n".join(out_lines)


def _repair_unsupported_cited_lines(answer: str, chunks: list[dict[str, Any]], min_overlap: float) -> str:
    if not answer or not chunks:
        return answer
    repaired: list[str] = []
    for raw in answer.splitlines():
        line = raw
        stripped = raw.strip()
        cites = [int(x) for x in re.findall(r"\[S(\d+)\]", raw)]
        if cites and stripped:
            best = _best_claim_overlap(raw, chunks)
            if best < min_overlap:
                first_valid = next((c for c in cites if 1 <= c <= len(chunks)), None)
                cite_tag = f" [S{first_valid}]" if first_valid else ""
                if stripped.startswith("-"):
                    line = f"- Insufficient custom evidence.{cite_tag}".rstrip()
                elif ":" in stripped:
                    label = stripped.split(":", 1)[0]
                    line = f"{label}: Insufficient custom evidence.{cite_tag}".rstrip()
                else:
                    line = f"Insufficient custom evidence.{cite_tag}".rstrip()
        repaired.append(line)
    return "\n".join(repaired)


def _enforce_confidence_from_metrics(answer: str, grounding: float, numeric_risk: float) -> str:
    if grounding >= 0.55 and numeric_risk <= 0.10:
        label = "HIGH"
        reason = "strong citation grounding with low numeric inconsistency risk"
    elif grounding >= 0.30 and numeric_risk <= 0.25:
        label = "MEDIUM"
        reason = "partially grounded evidence with manageable numeric inconsistency risk"
    else:
        label = "LOW"
        reason = "limited grounding or elevated numeric inconsistency risk"

    confidence_line = f"Confidence: {label} - {reason}."
    if "Confidence:" in answer:
        return re.sub(r"^Confidence:.*$", confidence_line, answer, flags=re.MULTILINE)
    return answer.rstrip() + "\n" + confidence_line


def _apply_reasoning_guardrails(answer: str, chunks: list[dict[str, Any]]) -> tuple[str, float, float]:
    if not answer:
        return answer, 0.0, 1.0
    if not chunks:
        out = _enforce_confidence_from_metrics(answer, 1.0, 0.0)
        return out, 1.0, 0.0

    out = _sanitize_unsupported_cited_numbers(answer, chunks)
    out = _repair_unsupported_cited_lines(out, chunks, min_overlap=INTEL_REPAIR_MIN_LINE_OVERLAP)
    grounding = _response_grounding_score(out, chunks)
    numeric_risk = _numeric_hallucination_risk(out, chunks)
    out = _enforce_confidence_from_metrics(out, grounding, numeric_risk)
    return out, grounding, numeric_risk


def _finalize_response_text(
    response_text: str,
    *,
    question: str,
    regime: dict[str, Any],
    cross_asset: dict[str, Any],
    indicators: dict[str, Any],
    context_chunks: list[dict[str, Any]],
    reasoning_analysis: dict[str, Any],
    response_mode: str,
) -> tuple[str, Any, float, float]:
    response_text = _normalize_expert_structure(response_text)
    response_text, enh_report = enhance_response(response_text, mode=response_mode)
    response_text, grounding, numeric_risk = _apply_reasoning_guardrails(response_text, context_chunks)

    if (
        INTEL_STRICT_REASONING_GUARD
        and context_chunks
        and (grounding < INTEL_MIN_GROUNDING_SCORE or numeric_risk > INTEL_MAX_NUMERIC_RISK)
    ):
        _tl.generation_status = "fallback_guarded_accuracy"
        response_text = generate_contextual_fallback(
            question=question,
            regime=regime,
            cross_asset=cross_asset,
            indicators=indicators,
            context_chunks=context_chunks,
            reasoning_analysis=reasoning_analysis,
            response_mode=response_mode,
        )
        response_text = _normalize_expert_structure(response_text)
        response_text, enh_report = enhance_response(response_text, mode=response_mode)
        response_text, grounding, numeric_risk = _apply_reasoning_guardrails(response_text, context_chunks)

    return response_text, enh_report, grounding, numeric_risk


def _normalize_expert_structure(text: str) -> str:
    if not text:
        return text
    out = text
    # Normalize legacy section labels to current format.
    out = out.replace("Cause-Effect Map:", "What is happening:")
    out = out.replace("Key drivers:", "What is happening:")
    out = out.replace("Main risks:", "Consequences & risks:")
    out = out.replace("What to watch next:", "What to watch:")
    out = out.replace("Why this is likely:", "What is happening:")
    out = out.replace("Why it matters now:", "What is happening:")
    out = out.replace("Action plan:", "What to watch:")
    out = out.replace("Predictions:", "Predicted events:")
    out = out.replace("Predicted event:", "Predicted events:")
    out = out.replace("Forward view:", "Predicted events:")
    out = out.replace("Forward-looking view:", "Predicted events:")
    # Normalise scenario headers
    out = out.replace("Base case:", "- Base (~55%):").replace("Bull case:", "- Bull (~25%):")
    out = out.replace("Bear case:", "- Bear (~20%):")
    if "Bottom line:" in out and "Direct answer:" not in out:
        out = out.replace("Bottom line:", "Direct answer:")
    if "Confidence:" not in out:
        out = out.rstrip() + "\nConfidence: LOW - Limited evidence."
    return out


def _compact_context_for_prompt(chunks: list[dict[str, Any]], max_chunks: int = 6, max_chars: int = 900) -> str:
    if not chunks:
        return "No indexed custom context available."
    parts: list[str] = []
    for i, c in enumerate(chunks[:max_chunks], start=1):
        md = c.get("metadata", {})
        title = md.get("title", "Unknown title")
        source = md.get("source", "Unknown source")
        date = md.get("date", "Unknown date")
        text = " ".join((c.get("text") or "").split())
        text = text[:max_chars].rstrip()
        parts.append(f"[S{i}] {title} | {source} | {date}\n{text}")
    return "\n\n".join(parts)


def _build_fast_prompt(
    question: str,
    indicators: dict[str, Any],
    chunks: list[dict[str, Any]],
    reasoning_analysis: dict[str, Any] | None = None,
) -> str:
    # Compact prompt for CPU-constrained fallback generation.
    num_map = [
        ("sp500", "S&P500"), ("vix", "VIX"), ("dxy", "DXY"),
        ("yield_10y", "10Y%"), ("yield_2y", "2Y%"), ("yield_curve", "Curve_bps"),
        ("inflation_cpi", "CPI%"), ("fed_funds_rate", "FedFunds%"),
        ("oil_wti", "WTI$"), ("oil_brent", "Brent$"), ("gold", "Gold$"),
        ("credit_hy", "HY_bps"),
    ]
    nums = [f"{label}={indicators[key]}" for key, label in num_map if key in indicators]
    num_line = ", ".join(nums[:9]) if nums else "No live data."
    context_bits = []
    for i, c in enumerate(chunks[:3], start=1):
        md = c.get("metadata", {})
        title = md.get("title", "")
        text = " ".join((c.get("text") or "").split())[:240]
        context_bits.append(f"[S{i}] {title}: {text}" if title else f"[S{i}] {text}")
    context_line = "\n".join(context_bits) if context_bits else "No indexed context."
    reasoning = reasoning_analysis or {}
    signal = reasoning.get("signal_score", {})
    fast_regime = signal.get("market_regime", "mixed_transition")
    fast_conf = signal.get("confidence", "N/A")
    event = reasoning.get("market_analysis_object", {}).get("event", "Macro transition")
    return (
        f"You are a macro analyst. Answer: {question}\n\n"
        f"Key numbers: {num_line}\n"
        f"Reasoning prior: event={event}; regime={fast_regime}; confidence={fast_conf}\n"
        f"News context:\n{context_line}\n\n"
        "Write analysis in EXACTLY this format:\n"
        "Direct answer: <specific answer with at least one number>\n"
        "Data snapshot: <list key numbers>\n"
        "Causal chain: <trigger> → <effect> → <market impact>\n"
        "What is happening:\n"
        "- <specific event or development — use numbers not generic phrases>\n"
        "- <mechanism: how does it transmit to markets>\n"
        "Market impact:\n"
        "- Equities: <direction + sector + reason>\n"
        "- Rates/Bonds: <yield direction + bps if possible; safe-haven=yields fall>\n"
        "- FX: <currency direction + reason>\n"
        "Predicted events:\n"
        "- <event 1 (7-30d) + probability + trigger/invalidation>\n"
        "- <event 2 (7-30d) + probability + trigger/invalidation>\n"
        "Scenarios (must sum to 100%):\n"
        "- Base (~55%): <outcome with number>\n"
        "- Bull (~25%): <upside trigger>\n"
        "- Bear (~20%): <downside trigger>\n"
        "What to watch:\n"
        "- <specific data release or level>\n"
        "Confidence: <HIGH/MEDIUM/LOW> - <reason>\n"
    )

def macro_intelligence_pipeline(
    question: str,
    manual_indicators: dict | None = None,
    geography: str = "US",
    horizon: str = "MEDIUM_TERM",
    response_mode: str = "brief",
) -> Iterator[str]:
    """
    Main pipeline for processing a user query.
    Orchestrates all modules, with error recovery and performance tracking.
    Each call runs in isolation using thread-local state to avoid race conditions
    in concurrent FastAPI environments.
    """
    # Initialise thread-local state for this request
    _tl.model_used = "N/A"
    _tl.generation_status = "not_started"
    _tl.llm_error = ""
    # --- MONITORING: METRICS ---
    latencies: Dict[str, float] = {}
    error_count = 0
    start_time = time.time()

    # Pipeline state
    classification = {}
    context_chunks = []
    formatted_context = ""
    retrieval_health = {}
    live_indicators: dict[str, Any] = {}
    live_meta: dict[str, Any] = {}
    indicators = {}
    regime = {}
    cross_asset = {}
    reasoning_analysis: dict[str, Any] = {}
    response_text = ""
    enh_report = None
    grounding = 0.0
    numeric_risk = 0.0

    # --- PIPELINE SEQUENCE ---
    try:
        # 1. question_classifier
        s = time.time()
        classification = classify_question(question)
        latencies['question_classifier'] = time.time() - s
    except Exception as e:
        yield f"[ERROR in question_classifier: {e}]"
        error_count += 1
        classification = {'primary': 'general', 'time': 'unknown'}

    try:
        # 2. context retrieval + live data fetched IN PARALLEL
        s = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            # Reduced from 25/15 → 12/8: smaller context = shorter prompt = faster LLM.
            fut_ctx  = pool.submit(retrieve_relevant_context, question, 12, 8)
            fut_live = pool.submit(fetch_live_indicators)
            try:
                context_chunks   = fut_ctx.result(timeout=60)
                formatted_context = format_context(context_chunks)
            except Exception as e:
                yield f"[ERROR in context_retriever: {e}]"
                error_count += 1
                context_chunks    = []
                formatted_context = "No indexed custom context available."
            try:
                live_indicators, live_meta = fut_live.result(timeout=90)
            except Exception:
                live_indicators, live_meta = {}, {}
        latencies["context_retriever"] = time.time() - s
        latencies["live_data"]         = latencies["context_retriever"]  # parallel
    except Exception as e:
        yield f"[ERROR in parallel_data_fetch: {e}]"
        error_count += 1
        context_chunks    = []
        formatted_context = "No indexed custom context available."
        live_indicators, live_meta = {}, {}

    retrieval_health = evaluate_retrieval_quality(question, context_chunks)

    try:
        # 3. indicator_parser from question + custom context + manual overrides
        s = time.time()
        context_text = " ".join(c.get("text", "") for c in context_chunks)
        context_indicators = extract_indicators_from_text(context_text)
        question_indicators = extract_indicators_from_text(question)
        indicators = {}
        indicators.update(context_indicators)
        indicators.update(live_indicators)
        indicators.update(question_indicators)
        if manual_indicators:
            indicators.update(manual_indicators)
        latencies["indicator_parser"] = time.time() - s
    except Exception as e:
        yield f"[ERROR in indicator_parser: {e}]"
        error_count += 1
        indicators = dict(manual_indicators or {})

    try:
        # 4. regime_detector
        s = time.time()
        regime = detect_regime(**get_regime_inputs_from_indicators(indicators))
        latencies['regime_detector'] = time.time() - s
    except Exception as e:
        yield f"[ERROR in regime_detector: {e}]"
        error_count += 1
        regime = {'regime': 'UNKNOWN', 'confidence': 0}

    try:
        # 5. cross_asset_analyzer
        s = time.time()
        cross_asset = analyze_cross_asset(indicators)
        latencies['cross_asset_analyzer'] = time.time() - s
    except Exception as e:
        yield f"[ERROR in cross_asset_analyzer: {e}]"
        error_count += 1
        cross_asset = {'overall_signal': 'NEUTRAL / INSUFFICIENT_DATA', 'divergences': []}

    try:
        # 5b. deterministic reasoning layer (events → impact → confirmation → score → scenarios)
        s = time.time()
        reasoning_analysis = build_reasoning_analysis(
            question=question,
            context_chunks=context_chunks,
            indicators=indicators,
            regime=regime,
            cross_asset=cross_asset,
            retrieval_health=retrieval_health,
            classification=classification,
        )
        latencies['reasoning_layer'] = time.time() - s
    except Exception as e:
        yield f"[ERROR in reasoning_layer: {e}]"
        error_count += 1
        reasoning_analysis = {}

    # Display header after initial analysis
    yield (
        f"━━━ {APP_BRAND} [{datetime.now(timezone.utc).strftime('%Y-%m-%d')}] ━━━\n"
        f"Regime: {regime.get('regime', 'N/A')} [{regime.get('confidence', 'LOW')}] | "
        f"Signal: {cross_asset.get('overall_signal', 'N/A')}\n"
        f"Custom context chunks used: {len(context_chunks)} | Retrieval quality: {retrieval_health.get('status', 'WARN')} ({retrieval_health.get('score', 0)})\n"
        f"Reasoning confidence: {reasoning_analysis.get('signal_score', {}).get('confidence', 'N/A')} | "
        f"Reasoning regime: {reasoning_analysis.get('signal_score', {}).get('market_regime', 'N/A')}\n"
        f"Reasoning consistency: {reasoning_analysis.get('signal_score', {}).get('consistency', 'N/A')} | "
        f"Contradictions: {reasoning_analysis.get('cross_asset_confirmation', {}).get('contradiction_count', 'N/A')}\n"
        f"Live indicators: {len(live_indicators)} | mode={response_mode}\n\n"
    )

    try:
        # 6. unified response builder (LLM-first, deterministic fallback)
        s = time.time()
        response_prompt = build_unified_response_prompt(
            question=question,
            classification=classification,
            regime=regime,
            cross_asset=cross_asset,
            indicators=indicators,
            # Reduced from max_chunks=8/max_chars=800 → 4/600 → shorter prompt, faster prefill.
            formatted_context=_compact_context_for_prompt(context_chunks, max_chunks=4, max_chars=600),
            geography=geography,
            horizon=horizon,
            response_mode=response_mode,
            live_data_meta=live_meta,
            reasoning_analysis=reasoning_analysis,
        )
        latencies['response_builder'] = time.time() - s
    except Exception as e:
        yield f"[ERROR in response_builder: {e}]"
        error_count += 1
        response_prompt = ""

    yield "▸ RESPONSE\n"
    try:
        if not response_prompt:
            raise RuntimeError("Response prompt unavailable")
        raw_response = _collect_llm_text(response_prompt)
        if not _valid_response(_normalize_expert_structure(raw_response)):
            raise RuntimeError("Response output failed format validation")
        response_text, enh_report, grounding, numeric_risk = _finalize_response_text(
            raw_response,
            question=question,
            regime=regime,
            cross_asset=cross_asset,
            indicators=indicators,
            context_chunks=context_chunks,
            reasoning_analysis=reasoning_analysis,
            response_mode=response_mode,
        )
        yield response_text
    except Exception:
        # Second chance: use a compact prompt optimized for CPU-only inference.
        try:
            fast_prompt = _build_fast_prompt(question, indicators, context_chunks)
            fast_prompt = _build_fast_prompt(question, indicators, context_chunks, reasoning_analysis=reasoning_analysis)
            fast_response = _collect_llm_text(fast_prompt)
            if _valid_response(_normalize_expert_structure(fast_response)):
                response_text = fast_response
            else:
                raise RuntimeError("fast_prompt_invalid")
        except Exception:
            # LLM failed or guardrails rejected output; fallback path produced final answer.
            if _tl_status() == "llm_failed":
                _tl.generation_status = "fallback_llm_failed"
            else:
                _tl.generation_status = "fallback_guarded"
            response_text = generate_contextual_fallback(
                question=question,
                regime=regime,
                cross_asset=cross_asset,
                indicators=indicators,
                context_chunks=context_chunks,
                reasoning_analysis=reasoning_analysis,
                response_mode=response_mode,
            )
        response_text, enh_report, grounding, numeric_risk = _finalize_response_text(
            response_text,
            question=question,
            regime=regime,
            cross_asset=cross_asset,
            indicators=indicators,
            context_chunks=context_chunks,
            reasoning_analysis=reasoning_analysis,
            response_mode=response_mode,
        )
        yield response_text

    # Emit quality metadata footer
    try:
        q_score = enh_report.quality_score if enh_report is not None else 0
        q_warns = f" | Warnings: {len(enh_report.warnings)}" if enh_report and enh_report.warnings else ""
        yield (
            f"\n[Quality: {q_score}/100{q_warns} | Grounding: {grounding:.2f} | "
            f"NumericRisk: {numeric_risk:.2f} | ReasoningConsistency: {reasoning_analysis.get('signal_score', {}).get('consistency', 'N/A')} | "
            f"Model: {_tl_model()} | Mode: {response_mode}]\n"
        )
    except Exception:
        pass
    yield "\n"

    total_latency = time.time() - start_time

    # --- MONITORING: LOG (structured, uses logging not print) ---
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "regime": regime.get("regime", "N/A"),
        "signal": cross_asset.get("overall_signal", "N/A"),
        "reasoning_signal_confidence": reasoning_analysis.get("signal_score", {}).get("confidence") if reasoning_analysis else None,
        "reasoning_market_regime": reasoning_analysis.get("signal_score", {}).get("market_regime") if reasoning_analysis else None,
        "reasoning_consistency": reasoning_analysis.get("signal_score", {}).get("consistency") if reasoning_analysis else None,
        "reasoning_contradictions": reasoning_analysis.get("cross_asset_confirmation", {}).get("contradiction_count") if reasoning_analysis else None,
        "total_latency_ms": int(total_latency * 1000),
        "module_latencies_ms": {k: int(v * 1000) for k, v in latencies.items()},
        "error_count": error_count,
        "model_used": _tl_model(),
    }
    logger.info("[macro_engine] pipeline_complete | %s", json.dumps(log_entry))

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("\n▸ Kotak Macro AI Engine\n")
    while True:
        try:
            q = input("Query: ").strip()
            if q.lower() in ["exit", "quit"]:
                break
            if not q:
                continue
            print()
            for chunk in macro_intelligence_pipeline(q):
                print(chunk, end='', flush=True)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred in the main loop: {e}")
            break
