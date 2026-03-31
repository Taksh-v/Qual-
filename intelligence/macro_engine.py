"""
macro_engine.py
---------------
Orchestrates the full macro intelligence pipeline, integrating all modules.
"""

import subprocess
import json
import os
import time
import re
import requests
from datetime import datetime
from typing import Iterator, Dict, Any

# Module Imports
from intelligence.question_classifier import classify_question
from intelligence.indicator_parser import extract_indicators_from_text, get_regime_inputs_from_indicators
from intelligence.regime_detector import detect_regime
from intelligence.cross_asset_analyzer import analyze_cross_asset
from intelligence.context_retriever import retrieve_relevant_context, format_context
from intelligence.data_quality import evaluate_retrieval_quality
from intelligence.live_market_data import fetch_live_indicators
from intelligence.model_router import get_model_candidates
from intelligence.llm_provider import generate_text as _provider_generate_text
from intelligence.macro_reasoner import (
    build_citation_repair_prompt,
    build_quality_rewrite_prompt,
    build_unified_response_prompt,
    generate_contextual_fallback,
    generate_unified_fallback,
)

# --- Configuration ---
LLM_CONNECT_TIMEOUT_SEC = os.getenv("OLLAMA_CONNECT_TIMEOUT_SEC", "5")
LLM_MAX_TIME_SEC = os.getenv("OLLAMA_GENERATE_TIMEOUT_SEC", "60")
LAST_MODEL_USED = "N/A"
LAST_GENERATION_STATUS = "not_started"
LAST_LLM_ERROR = ""
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


MACRO_ENABLE_MCP_ENRICHMENT = _env_flag("MACRO_ENABLE_MCP_ENRICHMENT", False)
MACRO_MCP_SERVER = os.getenv("MACRO_MCP_SERVER", "").strip()
MACRO_MCP_TOOL = os.getenv("MACRO_MCP_TOOL", "").strip()
MACRO_MCP_TIMEOUT_SEC = max(1.0, float(os.getenv("MACRO_MCP_TIMEOUT_SEC", "6")))
MACRO_MCP_MAX_CHARS = max(120, int(os.getenv("MACRO_MCP_MAX_CHARS", "700")))

def call_llm(prompt: str, model: str) -> Iterator[str]:
    """
    Calls the Ollama API to get a streaming response.
    Includes timeouts and error handling.
    """
    payload = {"model": model, "prompt": prompt, "stream": True}
    try:
        process = subprocess.Popen(
            [
                "curl", "-s",
                "--connect-timeout", LLM_CONNECT_TIMEOUT_SEC,
                "--max-time", LLM_MAX_TIME_SEC,
                "http://localhost:11434/api/generate",
                "-d", json.dumps(payload),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if process.stdout:
            for line in process.stdout:
                try:
                    data = json.loads(line)
                    yield data.get("response", "")
                except json.JSONDecodeError:
                    continue # Ignore malformed lines
        process.wait()
        if process.returncode != 0:
            stderr = process.stderr.read() if process.stderr else "No stderr."
            yield f"[LLM_ERROR: Curl failed with code {process.returncode}. Stderr: {stderr}]"
    except Exception as e:
        yield f"[LLM_ERROR: Subprocess failed: {e}]"


# ── Quality gating threshold ───────────────────────────────────────────────
QUALITY_GATE_THRESHOLD = int(os.getenv("QUALITY_GATE_THRESHOLD", "55"))
QUALITY_GATE_ENABLED = os.getenv("QUALITY_GATE_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off",
}


def _collect_llm_text(prompt: str) -> str:
    """Generate text using the unified provider chain (cloud → Ollama).

    Tries the configured provider chain first (see LLM_PROVIDER_ORDER env).
    Falls back to iterating local Ollama model candidates if all providers fail.
    """
    global LAST_MODEL_USED, LAST_GENERATION_STATUS, LAST_LLM_ERROR
    last_error: str | None = None

    # ── Attempt 1: Unified provider chain (cloud + local) ──────────────────
    try:
        text, provider_name = _provider_generate_text(
            prompt,
            temperature=0.25,
            max_tokens=int(os.getenv("LLM_NUM_PREDICT", "800")),
            timeout_sec=float(LLM_MAX_TIME_SEC),
        )
        if text:
            LAST_MODEL_USED = provider_name
            LAST_GENERATION_STATUS = "llm_ok"
            return text
        last_error = f"empty_response:{provider_name}"
    except Exception as exc:
        last_error = f"provider_chain:{exc}"
        LAST_LLM_ERROR = (last_error or "")[:220]

    # ── Attempt 2: Legacy Ollama model candidates fallback ─────────────────
    for model in get_model_candidates():
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.25,
                        "num_predict": 480,
                    },
                },
                timeout=float(LLM_MAX_TIME_SEC),
            )
            response.raise_for_status()
            text = (response.json().get("response") or "").strip()
            if text:
                LAST_MODEL_USED = model
                LAST_GENERATION_STATUS = "llm_ok"
                return text
            last_error = f"empty_response:{model}"
        except Exception as exc:
            last_error = f"{model}:{exc}"
            LAST_LLM_ERROR = (last_error or "")[:220]
            continue
    LAST_MODEL_USED = "N/A"
    LAST_GENERATION_STATUS = "llm_failed"
    LAST_LLM_ERROR = (last_error or "")[:220]
    raise RuntimeError(last_error or "LLM generation failed for all model candidates.")


def get_last_model_used() -> str:
    if LAST_GENERATION_STATUS == "llm_ok":
        return LAST_MODEL_USED or "N/A"
    if LAST_GENERATION_STATUS in ("llm_failed", "fallback_llm_failed"):
        if "Read timed out" in LAST_LLM_ERROR or "timed out" in LAST_LLM_ERROR:
            return "fallback (llm_timeout)"
        if "404" in LAST_LLM_ERROR or "not found" in LAST_LLM_ERROR.lower():
            return "fallback (model_not_found)"
        if "Connection" in LAST_LLM_ERROR or "Failed to establish" in LAST_LLM_ERROR:
            return "fallback (ollama_unreachable)"
        if LAST_LLM_ERROR.startswith("empty_response"):
            return "fallback (empty_response)"
        return "fallback (llm_failed)"
    if LAST_GENERATION_STATUS.startswith("fallback_"):
        return f"fallback ({LAST_GENERATION_STATUS.replace('fallback_', '')})"
    return LAST_MODEL_USED or "N/A"


def _improve_quality(section_name: str, raw_text: str) -> str:
    global LAST_MODEL_USED, LAST_GENERATION_STATUS, LAST_LLM_ERROR
    if not raw_text:
        return raw_text
    polish_prompt = build_quality_rewrite_prompt(section_name, raw_text)
    # Snapshot status before the call so a failed polish doesn't corrupt the
    # primary generation status that was already set by _collect_llm_text.
    saved_model = LAST_MODEL_USED
    saved_status = LAST_GENERATION_STATUS
    saved_error = LAST_LLM_ERROR
    try:
        polished = _collect_llm_text(polish_prompt)
        return polished or raw_text
    except Exception:
        LAST_MODEL_USED = saved_model
        LAST_GENERATION_STATUS = saved_status
        LAST_LLM_ERROR = saved_error
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
    return has_answer and has_body


def _citation_count(text: str) -> int:
    return text.count("[S")


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()) if len(t) > 2}


def _response_grounding_score(answer: str, chunks: list[dict[str, Any]]) -> float:
    cited_lines = [line.strip() for line in (answer or "").splitlines() if "[S" in line]
    if not cited_lines:
        return 0.0
    supported = 0
    for line in cited_lines:
        claim = re.sub(r"\[S\d+\]", "", line)
        claim_tokens = _tokenize(claim)
        cites = [int(x) for x in re.findall(r"\[S(\d+)\]", line)]
        if not claim_tokens or not cites:
            continue
        best = 0.0
        for c in cites:
            if 1 <= c <= len(chunks):
                src_tokens = _tokenize(chunks[c - 1].get("text", ""))
                if src_tokens:
                    overlap = len(claim_tokens.intersection(src_tokens)) / max(1, len(claim_tokens))
                    best = max(best, overlap)
        if best >= 0.25:
            supported += 1
    return supported / max(1, len(cited_lines))


def _numeric_hallucination_risk(answer: str, chunks: list[dict[str, Any]]) -> float:
    total_nums = 0
    missing_nums = 0
    for raw in (answer or "").splitlines():
        line = raw.strip()
        if "[S" not in line:
            continue
        line_wo_cites = re.sub(r"\[S\d+\]", "", line)
        nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", line_wo_cites)
        filtered = []
        for n in nums:
            if n.endswith("%"):
                filtered.append(n)
                continue
            if n.replace(".", "").isdigit() and int(float(n)) >= 100:
                filtered.append(n)
        cites = [int(x) for x in re.findall(r"\[S(\d+)\]", line)]
        if not filtered or not cites:
            continue
        corpus = " ".join(chunks[c - 1].get("text", "") for c in cites if 1 <= c <= len(chunks))
        for n in filtered:
            total_nums += 1
            if n not in corpus:
                missing_nums += 1
    if total_nums == 0:
        return 0.0
    return missing_nums / total_nums


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


def _build_fast_prompt(question: str, indicators: dict[str, Any], chunks: list[dict[str, Any]]) -> str:
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
    return (
        f"You are a macro analyst. Answer: {question}\n\n"
        f"Key numbers: {num_line}\n"
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
        "Scenarios (must sum to 100%):\n"
        "- Base (~55%): <outcome with number>\n"
        "- Bull (~25%): <upside trigger>\n"
        "- Bear (~20%): <downside trigger>\n"
        "What to watch:\n"
        "- <specific data release or level>\n"
        "Confidence: <HIGH/MEDIUM/LOW> - <reason>\n"
    )


def _maybe_fetch_mcp_enrichment(question: str) -> tuple[str, str]:
    if not MACRO_ENABLE_MCP_ENRICHMENT:
        return "", "disabled"
    if not MACRO_MCP_SERVER or not MACRO_MCP_TOOL:
        return "", "misconfigured"
    try:
        from mcp_integration.runtime_enrichment import fetch_external_context_sync

        result = fetch_external_context_sync(
            question=question,
            server_name=MACRO_MCP_SERVER,
            tool_name=MACRO_MCP_TOOL,
            timeout_sec=MACRO_MCP_TIMEOUT_SEC,
            max_chars=MACRO_MCP_MAX_CHARS,
        )
        if result.get("ok"):
            return str(result.get("text") or ""), "used"
        if result.get("is_error"):
            return "", "tool_error"
        return "", "empty"
    except Exception:
        return "", "error"


def _append_mcp_context(response_text: str, mcp_text: str) -> str:
    clean_response = (response_text or "").strip()
    if not mcp_text:
        return clean_response
    if "External MCP context" in clean_response:
        return clean_response
    return f"{clean_response}\n\nExternal MCP context (supplementary, uncited):\n{mcp_text.strip()}"

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
    """
    global LAST_MODEL_USED, LAST_GENERATION_STATUS, LAST_LLM_ERROR
    LAST_MODEL_USED = "N/A"
    LAST_GENERATION_STATUS = "not_started"
    LAST_LLM_ERROR = ""
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
    response_text = ""
    mcp_context_text = ""
    mcp_enrichment_status = "disabled"

    # --- PIPELINE SEQUENCE ---
    try:
        # 1. question_classifier
        s = time.time()
        classification = classify_question(question)
        latencies['question_classifier'] = time.time() - s
        yield f"<<PROGRESS>>{json.dumps({'stage': 'question_classifier', 'time_ms': int(latencies['question_classifier'] * 1000)})}"
    except Exception as e:
        yield f"[ERROR in question_classifier: {e}]"
        error_count += 1
        classification = {'primary': 'general', 'time': 'unknown'}
        yield f"<<PROGRESS>>{json.dumps({'stage': 'question_classifier', 'error': str(e)})}"

    try:
        # 2. context retrieval from custom indexed data
        s = time.time()
        context_chunks = retrieve_relevant_context(question, top_k=10, keep_latest=8)
        formatted_context = format_context(context_chunks)
        latencies["context_retriever"] = time.time() - s
        yield f"<<PROGRESS>>{json.dumps({'stage': 'context_retriever', 'time_ms': int(latencies['context_retriever'] * 1000), 'chunks': len(context_chunks)})}"
    except Exception as e:
        yield f"[ERROR in context_retriever: {e}]"
        error_count += 1
        context_chunks = []
        formatted_context = "No indexed custom context available."
        yield f"<<PROGRESS>>{json.dumps({'stage': 'context_retriever', 'error': str(e)})}"

    retrieval_health = evaluate_retrieval_quality(question, context_chunks)

    try:
        s = time.time()
        live_indicators, live_meta = fetch_live_indicators()
        latencies["live_data"] = time.time() - s
        yield f"<<PROGRESS>>{json.dumps({'stage': 'live_data', 'time_ms': int(latencies['live_data'] * 1000), 'indicator_count': len(live_indicators), 'fetch_ms': live_meta.get('fetch_ms'), 'open_exchanges': live_meta.get('open_exchanges')})}"
    except Exception:
        live_indicators, live_meta = {}, {}
        yield f"<<PROGRESS>>{json.dumps({'stage': 'live_data', 'error': 'fetch_live_indicators failed'})}"

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
        yield f"<<PROGRESS>>{json.dumps({'stage': 'indicator_parser', 'time_ms': int(latencies['indicator_parser'] * 1000), 'indicator_count': len(indicators)})}"
    except Exception as e:
        yield f"[ERROR in indicator_parser: {e}]"
        error_count += 1
        indicators = dict(manual_indicators or {})
        yield f"<<PROGRESS>>{json.dumps({'stage': 'indicator_parser', 'error': str(e)})}"

    try:
        # 4. regime_detector
        s = time.time()
        regime = detect_regime(**get_regime_inputs_from_indicators(indicators))
        latencies['regime_detector'] = time.time() - s
        yield f"<<PROGRESS>>{json.dumps({'stage': 'regime_detector', 'time_ms': int(latencies['regime_detector'] * 1000), 'regime': regime.get('regime'), 'confidence': regime.get('confidence')})}"
    except Exception as e:
        yield f"[ERROR in regime_detector: {e}]"
        error_count += 1
        regime = {'regime': 'UNKNOWN', 'confidence': 0}
        yield f"<<PROGRESS>>{json.dumps({'stage': 'regime_detector', 'error': str(e)})}"

    try:
        # 5. cross_asset_analyzer
        s = time.time()
        cross_asset = analyze_cross_asset(indicators)
        latencies['cross_asset_analyzer'] = time.time() - s
        yield f"<<PROGRESS>>{json.dumps({'stage': 'cross_asset_analyzer', 'time_ms': int(latencies['cross_asset_analyzer'] * 1000), 'overall_signal': cross_asset.get('overall_signal')})}"
    except Exception as e:
        yield f"[ERROR in cross_asset_analyzer: {e}]"
        error_count += 1
        cross_asset = {'overall_signal': 'NEUTRAL / INSUFFICIENT_DATA', 'divergences': []}
        yield f"<<PROGRESS>>{json.dumps({'stage': 'cross_asset_analyzer', 'error': str(e)})}"

    # Display header after initial analysis
    yield (
        f"━━━ KOTAK MACRO AI [{datetime.now().strftime('%Y-%m-%d')}] ━━━\n"
        f"Regime: {regime.get('regime', 'N/A')} [{regime.get('confidence', 'LOW')}] | "
        f"Signal: {cross_asset.get('overall_signal', 'N/A')}\n"
        f"Custom context chunks used: {len(context_chunks)} | Retrieval quality: {retrieval_health.get('status', 'WARN')} ({retrieval_health.get('score', 0)})\n"
        f"Live indicators: {len(live_indicators)} | mode={response_mode}\n\n"
    )

    mcp_context_text, mcp_enrichment_status = _maybe_fetch_mcp_enrichment(question)

    try:
        # 6. unified response builder (LLM-first, deterministic fallback)
        s = time.time()
        response_prompt = build_unified_response_prompt(
            question=question,
            classification=classification,
            regime=regime,
            cross_asset=cross_asset,
            indicators=indicators,
            formatted_context=(
                _compact_context_for_prompt(context_chunks, max_chunks=6, max_chars=900)
                + (
                    "\n\nExternal MCP context (supplementary, uncited):\n" + mcp_context_text
                    if mcp_context_text
                    else ""
                )
            ),
            geography=geography,
            horizon=horizon,
            response_mode=response_mode,
            live_data_meta=live_meta,
        )
        latencies['response_builder'] = time.time() - s
        yield f"<<PROGRESS>>{json.dumps({'stage': 'response_builder', 'time_ms': int(latencies['response_builder'] * 1000)})}"
    except Exception as e:
        yield f"[ERROR in response_builder: {e}]"
        error_count += 1
        response_prompt = ""

    yield f"<<PROGRESS>>{json.dumps({'stage': 'llm_generation', 'ollama_url': OLLAMA_URL, 'timeout_sec': LLM_MAX_TIME_SEC})}"
    yield "▸ RESPONSE\n"
    try:
        if not response_prompt:
            raise RuntimeError("Response prompt unavailable")
        # The LLM call can be slow; we already sent an explicit progress event.
        raw_response = _collect_llm_text(response_prompt)
        response_text = _append_mcp_context(_normalize_expert_structure(raw_response), mcp_context_text)
        if not _valid_response(response_text):
            raise RuntimeError("Response output failed format validation")

        # ── Quality gating: auto-retry if quality score is too low ─────────
        if QUALITY_GATE_ENABLED:
            try:
                from intelligence.response_enhancer import enhance_response
                _, report = enhance_response(response_text, response_mode)
                if report.quality_score < QUALITY_GATE_THRESHOLD:
                    yield f'<<PROGRESS>>{json.dumps({"stage": "quality_gate", "score": report.quality_score, "threshold": QUALITY_GATE_THRESHOLD})}'
                    rewrite_prompt = build_quality_rewrite_prompt("full", response_text)
                    rewritten = _collect_llm_text(rewrite_prompt)
                    rewritten = _normalize_expert_structure(rewritten)
                    _, rewrite_report = enhance_response(rewritten, response_mode)
                    if rewrite_report.quality_score > report.quality_score:
                        response_text = _append_mcp_context(rewritten, mcp_context_text)
                        yield f'<<PROGRESS>>{json.dumps({"stage": "quality_gate_improved", "old_score": report.quality_score, "new_score": rewrite_report.quality_score})}'
            except Exception as qe:
                yield f'<<PROGRESS>>{json.dumps({"stage": "quality_gate_error", "error": str(qe)[:120]})}'

        yield response_text
    except Exception as exc:
        # Second chance: use a compact prompt optimized for CPU-only inference.
        try:
            yield f"<<PROGRESS>>{json.dumps({'stage': 'llm_generation_failed', 'error': str(exc)})}"
            fast_prompt = _build_fast_prompt(question, indicators, context_chunks)
            yield f"<<PROGRESS>>{json.dumps({'stage': 'llm_fast_prompt', 'ollama_url': OLLAMA_URL})}"
            fast_response = _collect_llm_text(fast_prompt)
            fast_response = _normalize_expert_structure(fast_response)
            if _valid_response(fast_response):
                response_text = fast_response
            else:
                raise RuntimeError("fast_prompt_invalid")
        except Exception:
            # LLM failed or guardrails rejected output; fallback path produced final answer.
            if LAST_GENERATION_STATUS == "llm_failed":
                LAST_GENERATION_STATUS = "fallback_llm_failed"
            else:
                LAST_GENERATION_STATUS = "fallback_guarded"
            response_text = generate_contextual_fallback(
                question=question,
                regime=regime,
                cross_asset=cross_asset,
                indicators=indicators,
                context_chunks=context_chunks,
                response_mode=response_mode,
            )
        response_text = _append_mcp_context(_normalize_expert_structure(response_text), mcp_context_text)
        yield response_text
    yield "\n\n"

    total_latency = time.time() - start_time
    
    # --- MONITORING: LOG ---
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "regime": regime.get('regime', 'N/A'),
        "signal": cross_asset.get('overall_signal', 'N/A'),
        "total_latency_ms": int(total_latency * 1000),
        "module_latencies_ms": {k: int(v * 1000) for k, v in latencies.items()},
        "error_count": error_count,
        "mcp_enrichment": mcp_enrichment_status,
    }
    # In a real system, this would write to a log file or service
    print(f"\n--- MONITORING LOG ---\n{json.dumps(log_entry, indent=2)}\n")

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
