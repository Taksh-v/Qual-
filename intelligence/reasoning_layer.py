"""
reasoning_layer.py
------------------
Deterministic multi-stage reasoning layer that converts raw context into a
structured market analysis object before narrative generation.

Pipeline stages:
  1) Event Detection
  2) Market Impact Mapping
  3) Cross-Asset Verification
  4) Signal Strength Scoring
  5) Historical Analog Reasoning
  6) Scenario Generation
  7) Structured Analysis Object
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import OrderedDict
from typing import Any


_SEVERITY_SCORE = {
    "LOW": 0.35,
    "MEDIUM": 0.65,
    "HIGH": 0.90,
}

_EVENT_IMPACT_RULES: dict[str, dict[str, int]] = {
    "Geopolitical Conflict": {
        "equities": -1,
        "oil": 1,
        "gold": 1,
        "usd": 1,
        "bonds": -1,
        "emerging_markets": -1,
    },
    "Central Bank Tightening": {
        "equities": -1,
        "oil": -1,
        "gold": -1,
        "usd": 1,
        "bonds": -1,
        "credit": -1,
    },
    "Central Bank Easing": {
        "equities": 1,
        "oil": 1,
        "gold": 1,
        "usd": -1,
        "bonds": 1,
        "credit": 1,
    },
    "Macro Data Upside": {
        "equities": 1,
        "oil": 1,
        "gold": -1,
        "usd": 0,
        "bonds": -1,
        "credit": 1,
    },
    "Macro Data Downside": {
        "equities": -1,
        "oil": -1,
        "gold": 1,
        "usd": 1,
        "bonds": 1,
        "credit": -1,
    },
    "Commodity Supply Shock": {
        "equities": -1,
        "oil": 1,
        "gold": 1,
        "usd": 1,
        "bonds": -1,
        "credit": -1,
    },
    "Corporate Earnings Shock": {
        "equities": -1,
        "oil": 0,
        "gold": 1,
        "usd": 0,
        "bonds": 1,
        "credit": -1,
    },
    "Credit Stress": {
        "equities": -1,
        "oil": -1,
        "gold": 1,
        "usd": 1,
        "bonds": 1,
        "credit": -1,
        "emerging_markets": -1,
    },
    "Macro Data Regime Shift": {
        "equities": 0,
        "oil": 0,
        "gold": 0,
        "usd": 0,
        "bonds": 0,
        "credit": 0,
    },
}

_HISTORICAL_ANALOGS: dict[str, dict[str, Any]] = {
    "Geopolitical Conflict": {
        "similar_events": ["Gulf War", "Russia-Ukraine War"],
        "oil_avg_move": "+6%",
        "sp500_avg_move": "-3%",
        "volatility_increase": "high",
        "similarity_score": 0.78,
    },
    "Central Bank Tightening": {
        "similar_events": ["Fed 2022 Hiking Cycle", "2018 Rate Tightening"],
        "oil_avg_move": "-2%",
        "sp500_avg_move": "-4%",
        "volatility_increase": "medium",
        "similarity_score": 0.74,
    },
    "Commodity Supply Shock": {
        "similar_events": ["1970s Oil Embargo", "2022 Energy Shock"],
        "oil_avg_move": "+10%",
        "sp500_avg_move": "-4%",
        "volatility_increase": "high",
        "similarity_score": 0.80,
    },
    "Corporate Earnings Shock": {
        "similar_events": ["Q4 2018 Earnings Reset", "COVID Earnings Collapse"],
        "oil_avg_move": "-3%",
        "sp500_avg_move": "-5%",
        "volatility_increase": "medium",
        "similarity_score": 0.68,
    },
    "Credit Stress": {
        "similar_events": ["GFC 2008", "Regional Banking Stress 2023"],
        "oil_avg_move": "-5%",
        "sp500_avg_move": "-8%",
        "volatility_increase": "high",
        "similarity_score": 0.82,
    },
    "Macro Data Regime Shift": {
        "similar_events": ["Soft-Landing Transition Episodes"],
        "oil_avg_move": "±3%",
        "sp500_avg_move": "±2%",
        "volatility_increase": "medium",
        "similarity_score": 0.55,
    },
}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


REASONING_CACHE_ENABLED = _env_flag("INTEL_REASONING_CACHE", True)
REASONING_CACHE_MAX_KEYS = max(16, int(os.getenv("INTEL_REASONING_CACHE_MAX_KEYS", "128")))
_REASONING_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    try:
        raw = str(value).strip().replace(",", "")
        if raw.endswith("%"):
            raw = raw[:-1]
        return float(raw) if raw else None
    except Exception:
        return None


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _indicator_completeness(indicators: dict[str, Any]) -> float:
    required_groups = [
        ("sp500",),
        ("vix",),
        ("dxy",),
        ("yield_10y",),
        ("yield_2y",),
        ("inflation_cpi",),
        ("credit_hy",),
        ("oil_wti", "oil_brent"),
    ]
    present = 0
    for group in required_groups:
        if any(indicators.get(k) is not None for k in group):
            present += 1
    return round(present / len(required_groups), 3)


def _reasoning_cache_key(
    *,
    question: str,
    indicators: dict[str, Any],
    context_chunks: list[dict[str, Any]],
    regime: dict[str, Any],
    cross_asset: dict[str, Any],
    classification: dict[str, Any] | None,
) -> str:
    indicator_keys = [
        "sp500", "vix", "dxy", "yield_10y", "yield_2y", "yield_curve",
        "inflation_cpi", "credit_hy", "oil_wti", "oil_brent", "gold",
    ]
    top_ctx = []
    for chunk in (context_chunks or [])[:6]:
        md = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
        top_ctx.append(
            {
                "title": (md.get("title") or "")[:120],
                "source": (md.get("source") or "")[:80],
                "date": md.get("date") or "",
                "fingerprint": md.get("fingerprint") or chunk.get("fingerprint") or "",
                "text_head": _normalize_text((chunk.get("text") or "")[:120]),
            }
        )

    payload = {
        "q": _normalize_text(question),
        "regime": regime.get("regime", "UNKNOWN"),
        "signal": cross_asset.get("overall_signal", "NEUTRAL / INSUFFICIENT_DATA"),
        "classification": classification or {},
        "indicators": {k: indicators.get(k) for k in indicator_keys},
        "ctx": top_ctx,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cache_get(cache_key: str) -> dict[str, Any] | None:
    if not REASONING_CACHE_ENABLED:
        return None
    hit = _REASONING_CACHE.get(cache_key)
    if hit is None:
        return None
    _REASONING_CACHE.move_to_end(cache_key)
    return copy.deepcopy(hit)


def _cache_set(cache_key: str, value: dict[str, Any]) -> None:
    if not REASONING_CACHE_ENABLED:
        return
    _REASONING_CACHE[cache_key] = copy.deepcopy(value)
    _REASONING_CACHE.move_to_end(cache_key)
    while len(_REASONING_CACHE) > REASONING_CACHE_MAX_KEYS:
        _REASONING_CACHE.popitem(last=False)


def _cross_asset_contradiction_penalty(cross_asset_confirmation: dict[str, Any]) -> float:
    ratio = float(cross_asset_confirmation.get("confirmation_ratio", 0.5))
    contradiction_count = int(cross_asset_confirmation.get("contradiction_count", 0))
    ratio_pen = max(0.0, 0.60 - ratio) * 0.20
    cnt_pen = min(0.12, 0.03 * contradiction_count)
    return round(_clamp(ratio_pen + cnt_pen, 0.0, 0.25), 3)


def _severity_from_text(text: str, indicators: dict[str, Any]) -> str:
    t = (text or "").lower()
    high_terms = [
        "war",
        "invasion",
        "sanction",
        "crisis",
        "default",
        "liquidity stress",
        "surge",
        "spike",
        "shock",
        "escalat",
    ]
    medium_terms = [
        "tension",
        "hawkish",
        "slowdown",
        "downgrade",
        "risk-off",
        "selloff",
        "widen",
    ]

    score = 0
    if any(term in t for term in high_terms):
        score += 2
    if any(term in t for term in medium_terms):
        score += 1

    vix = _to_float(indicators.get("vix"))
    hy = _to_float(indicators.get("credit_hy"))
    oil = _to_float(indicators.get("oil_wti") or indicators.get("oil_brent"))
    curve = _to_float(indicators.get("yield_curve"))
    if vix is not None and vix >= 25:
        score += 1
    if hy is not None and hy >= 500:
        score += 1
    if oil is not None and oil >= 90:
        score += 1
    if curve is not None and curve <= -80:
        score += 1

    if score >= 3:
        return "HIGH"
    if score >= 1:
        return "MEDIUM"
    return "LOW"


def _detect_location(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["middle east", "gulf", "iran", "israel", "red sea"]):
        return "Middle East"
    if any(k in t for k in ["ukraine", "russia", "europe", "eu", "ecb"]):
        return "Europe"
    if any(k in t for k in ["china", "taiwan", "asia", "boj", "japan"]):
        return "Asia"
    if any(k in t for k in ["us", "usa", "united states", "fed", "fomc"]):
        return "United States"
    return "Global"


def _extract_signal_text(question: str, context_chunks: list[dict[str, Any]]) -> str:
    parts = [question or ""]
    for chunk in (context_chunks or [])[:6]:
        md = chunk.get("metadata", {})
        title = md.get("title", "")
        txt = " ".join((chunk.get("text") or "").split())[:350]
        if title:
            parts.append(title)
        if txt:
            parts.append(txt)
    return "\n".join(parts)


def detect_events(question: str, context_chunks: list[dict[str, Any]], indicators: dict[str, Any]) -> list[dict[str, Any]]:
    text = _extract_signal_text(question, context_chunks)
    tl = text.lower()
    events: list[dict[str, Any]] = []
    ctx_blob = " ".join(
        (
            (chunk.get("metadata", {}) or {}).get("title", "")
            + " "
            + (chunk.get("text") or "")
        )
        for chunk in (context_chunks or [])[:8]
    ).lower()

    def add_event(
        event_type: str,
        economic_theme: str,
        affected_assets: list[str],
        probable_duration: str,
        sentiment: str,
        evidence_terms: list[str],
    ) -> None:
        severity = _severity_from_text(text, indicators)
        evidence = [term for term in evidence_terms if term in tl][:4]
        denom = max(1, min(4, len(evidence_terms)))
        evidence_cov = len(evidence) / denom
        ctx_hits = sum(1 for term in evidence_terms if term in ctx_blob)
        ctx_support = min(1.0, ctx_hits / denom)
        event_confidence = round(
            _clamp(
                0.45 * _SEVERITY_SCORE.get(severity, 0.35)
                + 0.35 * evidence_cov
                + 0.20 * ctx_support
            ),
            3,
        )
        events.append(
            {
                "event_type": event_type,
                "location": _detect_location(text),
                "severity": severity,
                "event_confidence": event_confidence,
                "economic_theme": economic_theme,
                "affected_assets": affected_assets,
                "probable_duration": probable_duration,
                "initial_sentiment": sentiment,
                "evidence_terms": evidence,
                "evidence_count": len(evidence),
            }
        )

    if any(k in tl for k in ["war", "conflict", "sanction", "geopolitical", "missile", "escalat"]):
        add_event(
            event_type="Geopolitical Conflict",
            economic_theme="Supply + risk sentiment shock",
            affected_assets=["Oil", "Equities", "Gold", "USD", "Rates"],
            probable_duration="Short-term",
            sentiment="Negative",
            evidence_terms=["war", "conflict", "sanction", "geopolitical", "escalat"],
        )

    if any(k in tl for k in ["fed", "fomc", "rate hike", "rate cut", "hawkish", "dovish", "policy"]):
        direction = "Tightening" if any(k in tl for k in ["hike", "higher for longer", "hawkish"]) else "Easing"
        add_event(
            event_type=f"Central Bank {direction}",
            economic_theme="Monetary policy transmission",
            affected_assets=["Rates", "Equities", "USD", "Credit"],
            probable_duration="Medium-term",
            sentiment="Negative" if direction == "Tightening" else "Positive",
            evidence_terms=["fed", "fomc", "rate", "hawkish", "dovish", "policy"],
        )

    if any(k in tl for k in ["cpi", "pce", "inflation", "pmi", "gdp", "payroll", "unemployment"]):
        downside = any(k in tl for k in ["hot inflation", "weak pmi", "slowdown", "recession", "miss"])
        add_event(
            event_type="Macro Data Downside" if downside else "Macro Data Upside",
            economic_theme="Macro data repricing",
            affected_assets=["Equities", "Rates", "USD", "Commodities"],
            probable_duration="Short-term",
            sentiment="Negative" if downside else "Positive",
            evidence_terms=["cpi", "pce", "inflation", "pmi", "gdp", "payroll"],
        )

    if any(k in tl for k in ["oil", "opec", "supply disruption", "pipeline", "shipping", "embargo"]):
        add_event(
            event_type="Commodity Supply Shock",
            economic_theme="Energy inflation impulse",
            affected_assets=["Oil", "Equities", "Rates", "USD", "EM"],
            probable_duration="Short-term",
            sentiment="Negative",
            evidence_terms=["oil", "opec", "supply disruption", "embargo", "shipping"],
        )

    if any(k in tl for k in ["earnings miss", "guidance cut", "profit warning", "downgrade", "margin"]):
        add_event(
            event_type="Corporate Earnings Shock",
            economic_theme="Earnings and valuation reset",
            affected_assets=["Equities", "Credit", "USD"],
            probable_duration="Short-term",
            sentiment="Negative",
            evidence_terms=["earnings", "guidance", "profit", "downgrade", "margin"],
        )

    hy = _to_float(indicators.get("credit_hy"))
    vix = _to_float(indicators.get("vix"))
    if (hy is not None and hy >= 420) or (vix is not None and vix >= 24) or any(
        k in tl for k in ["credit spread", "liquidity", "default risk", "funding stress", "bank stress"]
    ):
        add_event(
            event_type="Credit Stress",
            economic_theme="Financial conditions tightening",
            affected_assets=["Credit", "Equities", "USD", "EM", "Rates"],
            probable_duration="Medium-term",
            sentiment="Negative",
            evidence_terms=["credit spread", "liquidity", "default", "funding", "bank stress"],
        )

    if not events:
        events.append(
            {
                "event_type": "Macro Data Regime Shift",
                "location": _detect_location(text),
                "severity": _severity_from_text(text, indicators),
                "economic_theme": "Mixed macro transition",
                "affected_assets": ["Equities", "Rates", "FX", "Commodities"],
                "probable_duration": "Medium-term",
                "initial_sentiment": "Neutral",
                "evidence_terms": [],
            }
        )

    dedup: dict[str, dict[str, Any]] = {}
    for e in events:
        key = e["event_type"]
        old = dedup.get(key)
        if old is None or float(e.get("event_confidence", 0.0)) > float(old.get("event_confidence", 0.0)):
            dedup[key] = e

    out = sorted(
        dedup.values(),
        key=lambda x: (
            float(x.get("event_confidence", 0.0)),
            _SEVERITY_SCORE.get(x.get("severity", "LOW"), 0.35),
        ),
        reverse=True,
    )
    return out[:4]


def build_impact_map(events: list[dict[str, Any]]) -> dict[str, Any]:
    asset_scores: dict[str, float] = {}
    channels: list[dict[str, str]] = []
    event_weights: dict[str, float] = {}

    for ev in events:
        event_type = ev.get("event_type", "Macro Data Regime Shift")
        severity = ev.get("severity", "LOW")
        weight = float(ev.get("event_confidence", _SEVERITY_SCORE.get(severity, 0.35)))
        impacts = _EVENT_IMPACT_RULES.get(event_type, _EVENT_IMPACT_RULES["Macro Data Regime Shift"])
        event_weights[event_type] = round(weight, 3)

        if event_type == "Geopolitical Conflict":
            channels.append({"event": event_type, "economic_channel": "Energy supply risk", "market_impact": "Oil bullish"})
            channels.append({"event": event_type, "economic_channel": "Risk sentiment", "market_impact": "Equities bearish"})
            channels.append({"event": event_type, "economic_channel": "Safe-haven demand", "market_impact": "USD/Gold bullish"})
        elif "Central Bank" in event_type:
            channels.append({"event": event_type, "economic_channel": "Discount-rate repricing", "market_impact": "Long-duration assets sensitive"})
        elif event_type == "Commodity Supply Shock":
            channels.append({"event": event_type, "economic_channel": "Input-cost inflation", "market_impact": "Energy outperforms, cyclicals pressured"})
        elif event_type == "Credit Stress":
            channels.append({"event": event_type, "economic_channel": "Financing-cost shock", "market_impact": "Risk assets and HY pressured"})

        for asset, direction in impacts.items():
            asset_scores[asset] = asset_scores.get(asset, 0.0) + direction * weight

    def to_label(x: float) -> str:
        if x >= 0.30:
            return "bullish"
        if x <= -0.30:
            return "bearish"
        return "neutral"

    impact_map = {asset: to_label(score) for asset, score in asset_scores.items()}
    return {
        "impact_map": impact_map,
        "asset_scores": {k: round(v, 3) for k, v in asset_scores.items()},
        "event_weights": event_weights,
        "channels": channels,
    }


def verify_cross_asset(impact_map: dict[str, str], indicators: dict[str, Any]) -> dict[str, Any]:
    vix = _to_float(indicators.get("vix"))
    dxy = _to_float(indicators.get("dxy"))
    oil = _to_float(indicators.get("oil_wti") or indicators.get("oil_brent"))
    y10 = _to_float(indicators.get("yield_10y"))
    y2 = _to_float(indicators.get("yield_2y"))
    curve = _to_float(indicators.get("yield_curve"))
    hy = _to_float(indicators.get("credit_hy"))
    gold = _to_float(indicators.get("gold"))

    if curve is None and y10 is not None and y2 is not None:
        curve = (y10 - y2) * 100.0

    risk_off_obs = (
        (vix is not None and vix >= 20)
        or (hy is not None and hy >= 400)
        or (curve is not None and curve < 0)
    )
    energy_shock_obs = (oil is not None and oil >= 85)
    safe_haven_obs = (
        (dxy is not None and dxy >= 103)
        or (gold is not None and gold >= 2000)
        or (vix is not None and vix >= 20)
    )
    rate_pressure_obs = (
        (y10 is not None and y10 >= 4.4)
        or (y2 is not None and y2 >= 4.6)
        or (curve is not None and curve <= -40)
    )

    checks: list[dict[str, Any]] = []

    def add_check(name: str, expected: bool, observed: bool) -> None:
        checks.append(
            {
                "name": name,
                "expected": expected,
                "observed": observed,
                "matched": bool(expected and observed) or bool((not expected) and (not observed)),
            }
        )

    add_check("risk_sentiment", impact_map.get("equities") == "bearish", risk_off_obs)
    add_check("energy_shock", impact_map.get("oil") == "bullish", energy_shock_obs)
    add_check("safe_haven_flow", impact_map.get("usd") == "bullish" or impact_map.get("gold") == "bullish", safe_haven_obs)
    add_check("rate_pressure", impact_map.get("bonds") == "bearish", rate_pressure_obs)

    matched = sum(1 for c in checks if c["matched"])
    ratio = matched / max(1, len(checks))
    contradictions = [c["name"] for c in checks if bool(c["expected"]) != bool(c["observed"])]

    def status(expected: bool, observed: bool) -> str:
        if expected and observed:
            return "confirmed"
        if expected and not observed:
            return "not_confirmed"
        return "neutral"

    signal_conf = "HIGH" if ratio >= 0.75 else ("MEDIUM" if ratio >= 0.50 else "LOW")

    return {
        "risk_sentiment": status(impact_map.get("equities") == "bearish", risk_off_obs),
        "energy_shock": status(impact_map.get("oil") == "bullish", energy_shock_obs),
        "safe_haven_flow": status(impact_map.get("usd") == "bullish" or impact_map.get("gold") == "bullish", safe_haven_obs),
        "rate_pressure": status(impact_map.get("bonds") == "bearish", rate_pressure_obs),
        "signal_confidence": signal_conf,
        "confirmation_ratio": round(ratio, 3),
        "contradictions": contradictions,
        "contradiction_count": len(contradictions),
        "checks": checks,
    }


def _price_move_strength(indicators: dict[str, Any]) -> float:
    vix = _to_float(indicators.get("vix"))
    curve = _to_float(indicators.get("yield_curve"))
    oil = _to_float(indicators.get("oil_wti") or indicators.get("oil_brent"))
    hy = _to_float(indicators.get("credit_hy"))
    cpi = _to_float(indicators.get("inflation_cpi"))

    parts: list[float] = []
    if vix is not None:
        parts.append(_clamp((vix - 15.0) / 20.0))
    if curve is not None:
        parts.append(_clamp(abs(min(curve, 0.0)) / 150.0))
    if oil is not None:
        parts.append(_clamp((oil - 75.0) / 30.0))
    if hy is not None:
        parts.append(_clamp((hy - 300.0) / 450.0))
    if cpi is not None:
        parts.append(_clamp(abs(cpi - 2.0) / 3.0))

    if not parts:
        return 0.40
    return round(sum(parts) / len(parts), 3)


def _market_regime_tag(
    regime: dict[str, Any],
    impact_map: dict[str, str],
    indicators: dict[str, Any],
    signal_score: float,
) -> str:
    regime_name = (regime.get("regime") or "").upper()
    hy = _to_float(indicators.get("credit_hy"))

    if hy is not None and hy >= 550:
        return "liquidity_crisis"
    if "STAGFLATION" in regime_name:
        return "inflation_shock"
    if "RECESSION" in regime_name or "DEFLATION" in regime_name:
        return "risk_off"
    if signal_score >= 0.70 and impact_map.get("equities") == "bearish":
        return "risk_off"
    if signal_score >= 0.65 and impact_map.get("equities") == "bullish":
        return "risk_on"
    if "EARLY_RECOVERY" in regime_name or "GOLDILOCKS" in regime_name:
        return "growth_recovery"
    return "mixed_transition"


def build_historical_analog(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {
            "similar_events": ["No close analog"],
            "oil_avg_move": "N/A",
            "sp500_avg_move": "N/A",
            "volatility_increase": "unknown",
            "similarity_score": 0.40,
        }

    primary = events[0].get("event_type", "Macro Data Regime Shift")
    analog = _HISTORICAL_ANALOGS.get(primary, _HISTORICAL_ANALOGS["Macro Data Regime Shift"])
    return {
        "primary_event": primary,
        "similar_events": analog["similar_events"],
        "oil_avg_move": analog["oil_avg_move"],
        "sp500_avg_move": analog["sp500_avg_move"],
        "volatility_increase": analog["volatility_increase"],
        "similarity_score": analog["similarity_score"],
    }


def build_signal_score(
    events: list[dict[str, Any]],
    impact_map: dict[str, str],
    cross_asset_confirmation: dict[str, Any],
    historical_analog: dict[str, Any],
    indicators: dict[str, Any],
    regime: dict[str, Any],
    retrieval_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retrieval_health = retrieval_health or {}
    if events:
        sev = [
            float(e.get("event_confidence", _SEVERITY_SCORE.get(e.get("severity", "LOW"), 0.35)))
            for e in events
        ]
        news_severity = round(sum(sev) / len(sev), 3)
    else:
        news_severity = 0.40

    move_strength = _price_move_strength(indicators)
    confirmation = float(cross_asset_confirmation.get("confirmation_ratio", 0.5))
    historical = float(historical_analog.get("similarity_score", 0.5))
    data_completeness = _indicator_completeness(indicators)
    retrieval_quality = _clamp(float(retrieval_health.get("score", 50)) / 100.0)
    contradiction_penalty = _cross_asset_contradiction_penalty(cross_asset_confirmation)

    base_score = (
        0.4 * news_severity
        + 0.3 * move_strength
        + 0.2 * confirmation
        + 0.1 * historical
    )
    availability_adjust = 0.06 * (data_completeness - 0.5) + 0.04 * (retrieval_quality - 0.5)
    score = round(_clamp(base_score + availability_adjust - contradiction_penalty), 3)

    impact_level = "high" if score >= 0.74 else ("medium" if score >= 0.52 else "low")
    regime_tag = _market_regime_tag(regime, impact_map, indicators, score)
    consistency = "HIGH" if contradiction_penalty <= 0.05 and confirmation >= 0.70 else (
        "MEDIUM" if contradiction_penalty <= 0.12 and confirmation >= 0.50 else "LOW"
    )

    return {
        "confidence": score,
        "base_confidence": round(_clamp(base_score), 3),
        "market_regime": regime_tag,
        "impact_level": impact_level,
        "consistency": consistency,
        "retrieval_quality_status": retrieval_health.get("status", "WARN"),
        "components": {
            "news_severity": news_severity,
            "price_move_strength": move_strength,
            "cross_asset_confirmation": round(confirmation, 3),
            "historical_similarity": round(historical, 3),
            "data_completeness": round(data_completeness, 3),
            "retrieval_quality": round(retrieval_quality, 3),
            "contradiction_penalty": round(contradiction_penalty, 3),
            "availability_adjust": round(availability_adjust, 3),
        },
    }


def _pct_range_str(base: float, lo_pct: float, hi_pct: float, precision: int = 1) -> str:
    lo = base * (1.0 + lo_pct)
    hi = base * (1.0 + hi_pct)
    return f"{lo:.{precision}f}–{hi:.{precision}f}"


def _scenario_probabilities(market_regime: str, confidence: float) -> tuple[int, int, int]:
    regime = (market_regime or "").lower()
    conf = _clamp(confidence)

    if regime in {"risk_off", "inflation_shock", "liquidity_crisis"}:
        bull, bear = 15, 30
    elif regime in {"risk_on", "growth_recovery"}:
        bull, bear = 30, 15
    else:
        bull, bear = 20, 20

    tilt = int(round(max(0.0, conf - 0.5) * 10))  # 0..5
    if regime in {"risk_off", "inflation_shock", "liquidity_crisis"}:
        bear += tilt
        bull -= tilt
    elif regime in {"risk_on", "growth_recovery"}:
        bull += tilt
        bear -= tilt

    bull = max(10, min(45, bull))
    bear = max(10, min(45, bear))
    base = 100 - bull - bear

    if base < 40:
        if bull > bear:
            bull -= (40 - base)
        else:
            bear -= (40 - base)
        base = 100 - bull - bear
    elif base > 70:
        take = min(15, base - 70)
        if regime in {"risk_off", "inflation_shock", "liquidity_crisis"}:
            bear += take
        elif regime in {"risk_on", "growth_recovery"}:
            bull += take
        else:
            bull += take // 2
            bear += take - (take // 2)
        base = 100 - bull - bear

    return int(base), int(bull), int(bear)


def build_scenarios(
    market_regime: str,
    indicators: dict[str, Any],
    events: list[dict[str, Any]],
    signal_score: dict[str, Any] | None = None,
    historical_analog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spx = _to_float(indicators.get("sp500"))
    oil = _to_float(indicators.get("oil_wti") or indicators.get("oil_brent"))
    signal_score = signal_score or {}
    historical_analog = historical_analog or {}

    p_base, p_bull, p_bear = _scenario_probabilities(
        market_regime=market_regime,
        confidence=float(signal_score.get("confidence", 0.5)),
    )

    base_event = events[0]["event_type"] if events else "Macro transition"
    analog_names = historical_analog.get("similar_events", [])
    analog_hint = f" (analog: {analog_names[0]})" if analog_names else ""

    if spx is not None:
        base_spx = _pct_range_str(spx, -0.03, 0.01, precision=0)
        bull_spx = _pct_range_str(spx, 0.01, 0.04, precision=0)
        bear_spx = _pct_range_str(spx, -0.08, -0.04, precision=0)
    else:
        base_spx = "N/A"
        bull_spx = "N/A"
        bear_spx = "N/A"

    if oil is not None:
        base_oil = _pct_range_str(oil, -0.03, 0.05, precision=1)
        bull_oil = _pct_range_str(oil, -0.08, -0.02, precision=1)
        bear_oil = _pct_range_str(oil, 0.08, 0.20, precision=1)
    else:
        base_oil = "N/A"
        bull_oil = "N/A"
        bear_oil = "N/A"

    scenarios = {
        "base_case": {
            "probability_pct": p_base,
            "horizon": "7-30d",
            "narrative": f"{base_event} remains contained and markets stay range-bound{analog_hint}.",
            "sp500_range": base_spx,
            "oil_range": base_oil,
            "trigger": "Data broadly in-line with consensus",
            "invalidation": "Large inflation/policy surprise",
        },
        "bull_case": {
            "probability_pct": p_bull,
            "horizon": "7-30d",
            "narrative": "De-escalation / softer inflation enables risk-on repricing.",
            "sp500_range": bull_spx,
            "oil_range": bull_oil,
            "trigger": "Disinflation + easing policy tone",
            "invalidation": "Renewed hawkish or geopolitical shock",
        },
        "bear_case": {
            "probability_pct": p_bear,
            "horizon": "7-30d",
            "narrative": "Escalation or tighter financial conditions drive risk-off correction.",
            "sp500_range": bear_spx,
            "oil_range": bear_oil,
            "trigger": "Hot inflation / credit spread widening / conflict escalation",
            "invalidation": "Rapid policy relief + spread compression",
        },
    }

    return scenarios


def build_reasoning_graph(
    events: list[dict[str, Any]],
    impact: dict[str, Any],
    signal: dict[str, Any],
    scenarios: dict[str, Any],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    event_node_by_type: dict[str, str] = {}
    for i, ev in enumerate(events[:4], start=1):
        node_id = f"E{i}"
        event_node_by_type[ev.get("event_type", f"event_{i}")] = node_id
        nodes.append(
            {
                "id": node_id,
                "type": "event",
                "label": ev.get("event_type", "Unknown Event"),
                "severity": ev.get("severity", "LOW"),
                "confidence": ev.get("event_confidence", 0.35),
            }
        )

    channels = impact.get("channels", []) if isinstance(impact, dict) else []
    for i, ch in enumerate(channels[:10], start=1):
        channel_id = f"C{i}"
        nodes.append(
            {
                "id": channel_id,
                "type": "channel",
                "label": ch.get("economic_channel", "Unknown channel"),
                "market_impact": ch.get("market_impact", ""),
            }
        )
        src = event_node_by_type.get(ch.get("event", ""))
        if src:
            edges.append({"from": src, "to": channel_id, "type": "causes"})

    impact_map = impact.get("impact_map", {}) if isinstance(impact, dict) else {}
    for asset, view in list(impact_map.items())[:10]:
        asset_id = f"A_{asset}"
        nodes.append({"id": asset_id, "type": "asset", "label": asset, "view": view})
        for i, ch in enumerate(channels[:10], start=1):
            channel_id = f"C{i}"
            channel_text = f"{ch.get('market_impact', '')} {ch.get('economic_channel', '')}".lower()
            if asset.lower() in channel_text:
                edges.append({"from": channel_id, "to": asset_id, "type": "transmits"})

    regime_id = "R1"
    nodes.append(
        {
            "id": regime_id,
            "type": "regime",
            "label": signal.get("market_regime", "mixed_transition"),
            "confidence": signal.get("confidence", 0.5),
        }
    )
    for asset in list(impact_map.keys())[:10]:
        edges.append({"from": f"A_{asset}", "to": regime_id, "type": "confirms"})

    for key, label in (("base_case", "Base"), ("bull_case", "Bull"), ("bear_case", "Bear")):
        scenario = scenarios.get(key, {}) if isinstance(scenarios, dict) else {}
        node_id = f"S_{key}"
        nodes.append(
            {
                "id": node_id,
                "type": "scenario",
                "label": label,
                "probability_pct": scenario.get("probability_pct", "N/A"),
            }
        )
        edges.append({"from": regime_id, "to": node_id, "type": "branches_to"})

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def build_reasoning_analysis(
    *,
    question: str,
    context_chunks: list[dict[str, Any]],
    indicators: dict[str, Any],
    regime: dict[str, Any],
    cross_asset: dict[str, Any],
    retrieval_health: dict[str, Any] | None = None,
    classification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cache_key = _reasoning_cache_key(
        question=question,
        indicators=indicators,
        context_chunks=context_chunks,
        regime=regime,
        cross_asset=cross_asset,
        classification=classification,
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        cached.setdefault("diagnostics", {})["cache_hit"] = True
        return cached

    events = detect_events(question, context_chunks, indicators)
    impact = build_impact_map(events)
    confirmation = verify_cross_asset(impact.get("impact_map", {}), indicators)
    analog = build_historical_analog(events)
    signal = build_signal_score(
        events=events,
        impact_map=impact.get("impact_map", {}),
        cross_asset_confirmation=confirmation,
        historical_analog=analog,
        indicators=indicators,
        regime=regime,
        retrieval_health=retrieval_health,
    )
    scenarios = build_scenarios(
        signal.get("market_regime", "mixed_transition"),
        indicators,
        events,
        signal_score=signal,
        historical_analog=analog,
    )
    reasoning_graph = build_reasoning_graph(events, impact, signal, scenarios)

    primary_event = events[0] if events else {
        "event_type": "Macro Data Regime Shift",
        "severity": "LOW",
        "location": "Global",
    }

    structured = {
        "event": primary_event.get("event_type", "Macro Data Regime Shift"),
        "market_regime": signal.get("market_regime", "mixed_transition"),
        "signal_strength": signal.get("confidence", 0.5),
        "cross_asset_moves": impact.get("impact_map", {}),
        "confirmation": confirmation.get("signal_confidence", "LOW").lower(),
        "scenarios": {
            "base_case": f"{scenarios['base_case']['probability_pct']}%",
            "bull_case": f"{scenarios['bull_case']['probability_pct']}%",
            "bear_case": f"{scenarios['bear_case']['probability_pct']}%",
        },
    }

    result = {
        "event_detection": {
            "events": events,
            "primary_event": primary_event,
        },
        "market_impact_mapping": impact,
        "cross_asset_confirmation": confirmation,
        "signal_score": signal,
        "historical_analog": analog,
        "scenario_generator": scenarios,
        "reasoning_graph": reasoning_graph,
        "market_analysis_object": structured,
        "diagnostics": {
            "cache_hit": False,
            "indicator_completeness": _indicator_completeness(indicators),
            "contradiction_count": confirmation.get("contradiction_count", 0),
            "consistency": signal.get("consistency", "LOW"),
        },
        "inputs": {
            "classification": classification or {},
            "retrieval_health": retrieval_health or {},
            "cross_asset_signal": cross_asset.get("overall_signal", "NEUTRAL / INSUFFICIENT_DATA"),
            "regime": regime.get("regime", "UNKNOWN"),
        },
    }

    _cache_set(cache_key, result)
    return result
