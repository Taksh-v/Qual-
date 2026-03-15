"""
response_enhancer.py
---------------------
Post-processing layer that validates, cleans, and enhances LLM responses
to guarantee institutional-quality output even when the model produces
sub-optimal answers.

Enhancement stages:
  1. Section presence validation   — ensure all required sections exist
  2. Vagueness detector            — flag/replace empty buzzword bullets
  3. Number anchor enforcer        — detect bullets without numbers and flag them
  4. Scenario probability check    — scenarios must sum to ~100%
  5. Source citation normaliser    — standardise [Sx] refs
  6. Confidence section guardian   — add if missing
  7. Quality score reporter        — return a 0-100 quality score

Usage:
    from intelligence.response_enhancer import enhance_response, score_response
    enhanced, report = enhance_response(raw_text, mode="brief")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Required sections by mode ──────────────────────────────────────────────────
_REQUIRED_SECTIONS_BRIEF = [
    "Direct answer:",
    "Data snapshot:",
    "Causal chain:",
    "What is happening:",
    "Market impact:",
    "Predicted events:",
    "Scenarios",
    "What to watch:",
    "Confidence:",
]

_REQUIRED_SECTIONS_DETAILED = [
    "Executive summary:",
    "Direct answer:",
    "Data snapshot:",
    "Causal chain:",
    "What is happening:",
    "Market impact:",
    "Predicted events:",
    "Scenarios",
    "Key risks:",
    "Time horizons:",
    "What to watch:",
    "Confidence:",
]

# Vague buzzwords that add no signal — any bullet containing ONLY these is weak
_VAGUE_PHRASES = [
    r"\brightened uncertainty\b",
    r"\bdownward pressure\b",
    r"\bupward pressure\b",
    r"\brisk.?off sentiment\b",
    r"\bincreased volatility\b",
    r"\bmarket participants\b",
    r"\binvestors are concerned\b",
    r"\bsentiment turned\b",
    r"\bremains to be seen\b",
    r"\bvarious factors\b",
    r"\bmultiple headwinds\b",
    r"\buncertain environment\b",
    r"\bfurther monitoring required\b",
]

_VAGUE_RE = re.compile("|".join(_VAGUE_PHRASES), re.IGNORECASE)

# Pattern to detect numbers in text (percentages, bp, dollar amounts, etc.)
_NUMBER_RE = re.compile(r"\d+\.?\d*\s*(%|bps?|bp|\$|pts?|x\b)", re.IGNORECASE)

# Scenario probability pattern
_PROB_RE = re.compile(r"\(~?(\d+)%\)", re.IGNORECASE)

# Source citation pattern
_CITATION_RE = re.compile(r"\[S\d+\]", re.IGNORECASE)
_HORIZON_RE = re.compile(r"\b(\d+\s*-\s*\d+\s*(d|days|w|weeks|m|months)|24-72h|1-4\s*weeks|30-90\s*d)\b", re.IGNORECASE)


@dataclass
class EnhancementReport:
    """Structured report of all enhancements applied to the response."""
    missing_sections: list[str] = field(default_factory=list)
    vague_bullets_count: int = 0
    bullets_without_numbers: int = 0
    scenario_prob_sum: int = 0
    scenario_prob_ok: bool = True
    citations_count: int = 0
    predicted_event_format_issues: int = 0
    predicted_event_repairs: int = 0
    confidence_added: bool = False
    quality_score: int = 100  # 0-100
    warnings: list[str] = field(default_factory=list)


def _detect_missing_sections(text: str, mode: str) -> list[str]:
    required = _REQUIRED_SECTIONS_BRIEF if mode == "brief" else _REQUIRED_SECTIONS_DETAILED
    return [s for s in required if s not in text]


def _count_vague_bullets(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-") and _VAGUE_RE.search(stripped):
            # Vague only if no compensating number is present
            if not _NUMBER_RE.search(stripped):
                count += 1
    return count


def _count_bullets_without_numbers(text: str) -> int:
    """Count bullets in Market impact / Predicted events / Scenarios that have no numeric anchor."""
    in_market_block = False
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if "Market impact:" in stripped or "Predicted events:" in stripped or "Scenarios" in stripped:
            in_market_block = True
        elif stripped and stripped[0].isupper() and stripped.endswith(":") and stripped not in ("Market impact:",):
            # New top-level section
            in_market_block = False
        if in_market_block and stripped.startswith("-"):
            if not _NUMBER_RE.search(stripped) and len(stripped) > 10:
                count += 1
    return count


def _check_scenario_probabilities(text: str) -> tuple[int, bool]:
    """Extract scenario probabilities from the Scenarios section and check they sum near 100%."""
    probs: list[int] = []
    in_scenarios = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Scenarios"):
            in_scenarios = True
            continue
        if in_scenarios and stripped and stripped[0].isupper() and stripped.endswith(":"):
            in_scenarios = False
            continue
        if in_scenarios and stripped.startswith("-"):
            m = _PROB_RE.search(stripped)
            if m:
                probs.append(int(m.group(1)))
    if not probs:
        return 0, True  # No scenarios found — not an error
    total = sum(probs)
    # Allow ±5% tolerance
    ok = 95 <= total <= 105
    return total, ok


def _fix_scenario_probabilities(text: str, prob_sum: int) -> str:
    """If probabilities don't sum to ~100%, inject a note."""
    if 95 <= prob_sum <= 105 or prob_sum == 0:
        return text
    note = f"\n[NOTE: Scenario probabilities sum to {prob_sum}% — treat as directional estimates only]\n"
    # Insert after the last scenario bullet
    return text + note


def _normalise_citations(text: str) -> str:
    """Normalise citation variants like [Src1], [source1], [1] to [S1] format."""
    text = re.sub(r"\[Src(\d+)\]", r"[S\1]", text, flags=re.IGNORECASE)
    text = re.sub(r"\[source(\d+)\]", r"[S\1]", text, flags=re.IGNORECASE)
    text = re.sub(r"\[ref(\d+)\]", r"[S\1]", text, flags=re.IGNORECASE)
    # Bare [1] [2] etc — only convert if surrounded by context that looks like a citation
    text = re.sub(r"\[(\d+)\](?=\s*$|\s*[,\.])", r"[S\1]", text, flags=re.MULTILINE)
    return text


def _ensure_confidence_line(text: str) -> tuple[str, bool]:
    """Add a Confidence line if missing."""
    if "Confidence:" in text:
        return text, False
    text = text.rstrip() + "\nConfidence: MEDIUM - Partial data available; interpret with caution.\n"
    return text, True


def _predicted_event_format_issues(text: str) -> tuple[int, list[str]]:
    issues = 0
    issue_notes: list[str] = []
    in_predicted = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Predicted events:"):
            in_predicted = True
            continue
        if in_predicted and stripped and stripped[0].isupper() and stripped.endswith(":"):
            in_predicted = False
            continue
        if not in_predicted or not stripped.startswith("-"):
            continue

        low = stripped.lower()
        has_prob = "%" in stripped
        has_horizon = bool(_HORIZON_RE.search(stripped))
        has_trigger = "trigger" in low
        has_invalidation = "invalid" in low

        missing = []
        if not has_horizon:
            missing.append("horizon")
        if not has_prob:
            missing.append("probability")
        if not has_trigger:
            missing.append("trigger")
        if not has_invalidation:
            missing.append("invalidation")
        if missing:
            issues += 1
            issue_notes.append(f"predicted-event bullet missing {', '.join(missing)}")

    return issues, issue_notes


def _repair_predicted_event_bullets(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    out: list[str] = []
    in_predicted = False
    repairs = 0

    for raw in lines:
        stripped = raw.strip()
        line = raw

        if stripped.startswith("Predicted events:"):
            in_predicted = True
            out.append(line)
            continue

        if in_predicted and stripped and stripped[0].isupper() and stripped.endswith(":"):
            in_predicted = False

        if in_predicted and stripped.startswith("-"):
            low = stripped.lower()
            changed = False
            if not _HORIZON_RE.search(stripped):
                line = line.rstrip() + " | horizon: 7-30d"
                changed = True
            if "%" not in stripped:
                line = line.rstrip() + " | probability: ~50%"
                changed = True
            if "trigger" not in low:
                line = line.rstrip() + " | trigger: confirmation from next macro release"
                changed = True
            if "invalid" not in low:
                line = line.rstrip() + " | invalidation: opposite move in rates/credit/oil"
                changed = True
            if changed:
                repairs += 1

        out.append(line)

    return "\n".join(out), repairs


def _add_missing_section_stubs(text: str, missing: list[str]) -> str:
    """Append minimal stubs for completely missing sections."""
    stubs: dict[str, str] = {
        "Data snapshot:": "Data snapshot: Refer to live indicator section above.",
        "Predicted events:": "Predicted events:\n- Event 1 (7-30d, ~55%): Baseline continuation of current regime dynamics.\n- Event 2 (7-30d, ~45%): Opposite move if next major macro release surprises.",
        "What to watch:": "What to watch:\n- Next scheduled central bank meeting or CPI release.",
        "Confidence:": "Confidence: LOW - Limited context available.",
    }
    for section in missing:
        if section in stubs and section not in text:
            text = text.rstrip() + "\n" + stubs[section] + "\n"
    return text


def _compute_quality_score(report: EnhancementReport) -> int:
    """
    Score 0-100 based on:
      - Missing sections:          -10 per missing critical section
      - Vague bullets:             -5 per vague bullet  (max -20)
      - Bullets without numbers:   -3 per bullet         (max -15)
      - Bad scenario probs:        -10
      - No citations:              -5
    """
    score = 100
    critical = {"Direct answer:", "Market impact:", "Predicted events:", "Causal chain:", "Scenarios"}
    critical_missing = len([s for s in report.missing_sections if s in critical])
    score -= critical_missing * 10
    score -= min(report.vague_bullets_count * 5, 20)
    score -= min(report.bullets_without_numbers * 3, 15)
    score -= min(report.predicted_event_repairs * 2, 8)
    score -= min(report.predicted_event_format_issues * 4, 12)
    if not report.scenario_prob_ok and report.scenario_prob_sum > 0:
        score -= 10
    if report.citations_count == 0:
        score -= 5
    return max(0, min(100, score))


def enhance_response(text: str, mode: str = "brief") -> tuple[str, EnhancementReport]:
    """
    Apply all enhancement stages to a raw LLM response.

    Args:
        text: Raw LLM response text
        mode: 'brief' or 'detailed' (controls required section list)

    Returns:
        (enhanced_text, enhancement_report)
    """
    if not text:
        report = EnhancementReport(quality_score=0, warnings=["Empty response received"])
        return text, report

    report = EnhancementReport()

    # Stage 1: Detect missing sections
    report.missing_sections = _detect_missing_sections(text, mode)

    # Stage 2: Count vague bullets
    report.vague_bullets_count = _count_vague_bullets(text)

    # Stage 3: Count bullets without numeric anchors
    report.bullets_without_numbers = _count_bullets_without_numbers(text)

    # Stage 4: Check scenario probability sum
    report.scenario_prob_sum, report.scenario_prob_ok = _check_scenario_probabilities(text)

    # Stage 5: Normalise source citations
    text = _normalise_citations(text)
    report.citations_count = len(_CITATION_RE.findall(text))

    # Stage 6: Ensure confidence line
    text, report.confidence_added = _ensure_confidence_line(text)

    # Stage 7: Fix scenario probabilities if wrong
    text = _fix_scenario_probabilities(text, report.scenario_prob_sum)

    # Stage 8: Add stubs for missing critical sections
    text = _add_missing_section_stubs(text, report.missing_sections)

    # Stage 9: Ensure predicted-event bullets include horizon/probability/trigger/invalidation
    pre_pred_issues, pre_pred_notes = _predicted_event_format_issues(text)
    text, report.predicted_event_repairs = _repair_predicted_event_bullets(text)
    report.predicted_event_format_issues, post_pred_notes = _predicted_event_format_issues(text)

    # Stage 10: Build warnings
    if report.vague_bullets_count > 0:
        report.warnings.append(
            f"{report.vague_bullets_count} vague bullet(s) detected — no numeric anchor"
        )
    if report.bullets_without_numbers > 2:
        report.warnings.append(
            f"{report.bullets_without_numbers} market-impact bullets lack numeric grounding"
        )
    if report.missing_sections:
        report.warnings.append(f"Missing sections: {report.missing_sections}")
    if not report.scenario_prob_ok and report.scenario_prob_sum > 0:
        report.warnings.append(
            f"Scenario probabilities sum to {report.scenario_prob_sum}% (not 100%)"
        )
    if pre_pred_issues > 0:
        report.warnings.append(
            f"{pre_pred_issues} predicted-event bullet(s) needed structural repair"
        )
    if report.predicted_event_format_issues > 0:
        report.warnings.append(
            f"{report.predicted_event_format_issues} predicted-event bullet(s) still miss required fields"
        )
        report.warnings.extend(post_pred_notes[:2])

    # Stage 11: Quality score
    report.quality_score = _compute_quality_score(report)

    return text, report


def score_response(text: str, mode: str = "brief") -> dict[str, Any]:
    """
    Convenience wrapper returning a plain dict with quality metrics.
    Useful for API response metadata.
    """
    _, report = enhance_response(text, mode)
    return {
        "quality_score": report.quality_score,
        "missing_sections": report.missing_sections,
        "vague_bullets": report.vague_bullets_count,
        "bullets_without_numbers": report.bullets_without_numbers,
        "predicted_event_format_issues": report.predicted_event_format_issues,
        "predicted_event_repairs": report.predicted_event_repairs,
        "scenario_prob_sum": report.scenario_prob_sum,
        "scenario_prob_ok": report.scenario_prob_ok,
        "citations_count": report.citations_count,
        "warnings": report.warnings,
    }
