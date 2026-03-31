from __future__ import annotations

import re
from typing import Any

from intelligence.response_schema import (
    CausalArchitecture,
    CausalChainLink,
    CrossAssetImpact,
    KeyLevel,
    PositioningIdea,
    PredictedEvent,
    Scenario,
    StructuredResponse,
)


_HEADER_MAP = {
    "regime:": "regime",
    "dominant theme:": "dominant_theme",
    "direct answer:": "direct_answer",
    "data snapshot:": "data_snapshot",
    "causal chain:": "causal_chain",
    "causal architecture:": "causal_architecture",
    "what is happening:": "what_is_happening",
    "cross-asset impact:": "cross_asset_impacts",
    "market impact:": "market_impact",
    "positioning:": "positioning",
    "predicted events:": "predicted_events",
    "scenarios": "scenarios",
    "key levels:": "key_levels",
    "historical analog:": "historical_analog",
    "what to watch:": "what_to_watch",
    "data gaps:": "data_gaps",
    "confidence:": "confidence",
    "executive summary:": "executive_summary",
    "key risks:": "key_risks",
    "time horizons:": "time_horizons",
}


def _section_key(line: str) -> str | None:
    low = line.strip().lower()
    for header, key in _HEADER_MAP.items():
        if low.startswith(header):
            return key
    return None


def normalize_text_response(text: str, mode: str = "brief") -> StructuredResponse:
    text = (text or "").strip()
    sections: dict[str, Any] = {
        "regime": None,
        "dominant_theme": None,
        "direct_answer": "",
        "data_snapshot": "",
        "causal_chain": "",
        "causal_architecture": {"primary": [], "secondary": [], "feedback": ""},
        "what_is_happening": [],
        "cross_asset_impacts": [],
        "market_impact": [],
        "positioning": [],
        "predicted_events": [],
        "scenarios": [],
        "key_levels": [],
        "historical_analog": None,
        "what_to_watch": [],
        "data_gaps": [],
        "confidence": "MEDIUM - Partial data available; interpret with caution.",
        "executive_summary": None,
        "key_risks": [],
        "time_horizons": [],
    }

    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        section = _section_key(line)
        if section is not None:
            current = section
            if ":" in line and section in {
                "regime", "dominant_theme", "direct_answer", "data_snapshot",
                "causal_chain", "historical_analog", "confidence", "executive_summary"
            }:
                sections[section] = line.split(":", 1)[1].strip()
            continue

        if current is None:
            continue

        if current == "causal_architecture":
            low_line = line.lower()
            if low_line.startswith("primary:"):
                chain = line.split(":", 1)[1].strip()
                chain = chain.replace("->", "→")
                parts = [p.strip() for p in chain.split("→")]
                if len(parts) >= 2:
                    sections[current]["primary"].append(CausalChainLink(parts[0], " → ".join(parts[1:])))
            elif low_line.startswith("secondary:"):
                chain = line.split(":", 1)[1].strip()
                chain = chain.replace("->", "→")
                parts = [p.strip() for p in chain.split("→")]
                if len(parts) >= 2:
                    sections[current]["secondary"].append(CausalChainLink(parts[0], " → ".join(parts[1:])))
            elif low_line.startswith("feedback:"):
                sections[current]["feedback"] = line.split(":", 1)[1].strip()
            continue

        if line.startswith("-"):
            value = line[1:].strip()
            if current == "predicted_events":
                sections[current].append(_parse_predicted_event(value))
            elif current == "scenarios":
                sections[current].append(_parse_scenario(value))
            elif current == "cross_asset_impacts":
                sections[current].append(_parse_cross_asset(value))
            elif current == "positioning":
                sections[current].append(_parse_positioning(value))
            elif current == "key_levels":
                sections[current].append(_parse_key_level(value))
            elif current in {"what_is_happening", "market_impact", "what_to_watch", "key_risks", "time_horizons", "data_gaps"}:
                sections[current].append(value)
        elif current in {
            "regime", "dominant_theme", "direct_answer", "data_snapshot", 
            "causal_chain", "historical_analog", "confidence", "executive_summary"
        } and not sections[current]:
            sections[current] = line

    ca_data = sections["causal_architecture"]
    ca_obj = None
    if ca_data["primary"] or ca_data["secondary"] or ca_data["feedback"]:
        ca_obj = CausalArchitecture(
            primary=ca_data["primary"],
            secondary=ca_data["secondary"],
            feedback_loop=ca_data["feedback"]
        )

    return StructuredResponse(
        regime=sections["regime"],
        dominant_theme=sections["dominant_theme"],
        direct_answer=sections["direct_answer"] or "No direct answer provided.",
        data_snapshot=sections["data_snapshot"] or "No data snapshot provided.",
        causal_chain=sections["causal_chain"] or "No causal chain provided.",
        causal_architecture=ca_obj,
        what_is_happening=sections["what_is_happening"],
        cross_asset_impacts=sections["cross_asset_impacts"],
        market_impact=sections["market_impact"],
        positioning=sections["positioning"],
        predicted_events=sections["predicted_events"],
        scenarios=sections["scenarios"],
        key_levels=sections["key_levels"],
        historical_analog=sections["historical_analog"],
        what_to_watch=sections["what_to_watch"],
        data_gaps=sections["data_gaps"],
        confidence=sections["confidence"],
        executive_summary=sections["executive_summary"],
        key_risks=sections["key_risks"],
        time_horizons=sections["time_horizons"],
    )


def _parse_predicted_event(line: str) -> PredictedEvent:
    horizon = "7-30d"
    prob: int | None = None
    trigger = "N/A"
    invalidation = "N/A"
    label = "Event"
    narrative = line

    m_head = re.match(r"([^\(]+)\(([^\)]*)\):\s*(.*)", line)
    if m_head:
        label = m_head.group(1).strip()
        details = m_head.group(2)
        narrative = m_head.group(3).strip()
        h = re.search(r"(24-72h|\d+\s*-\s*\d+\s*(?:d|days|w|weeks|m|months))", details, flags=re.I)
        p = re.search(r"(\d+)%", details)
        if h:
            horizon = h.group(1).replace(" ", "")
        if p:
            prob = int(p.group(1))

    t = re.search(r"trigger:\s*([^;]+)", line, flags=re.I)
    inv = re.search(r"invalid(?:ation)?:\s*([^;]+)", line, flags=re.I)
    if t:
        trigger = t.group(1).strip().rstrip(".")
    if inv:
        invalidation = inv.group(1).strip().rstrip(".")

    return PredictedEvent(
        label=label,
        horizon=horizon,
        probability_pct=prob,
        narrative=narrative,
        trigger=trigger,
        invalidation=invalidation,
    )


def _parse_scenario(line: str) -> Scenario:
    # Handle scenario names with probabilities
    # Examples:
    # Base (~55%): narrative; trigger: foo; invalidation: bar.
    # Base (55%): narrative.
    
    name = "Scenario"
    prob = None
    narrative = line
    trigger = ""
    inv = ""
    
    m_head = re.match(r"([^\(]+)\((?:~)?(\d+)%\):\s*(.*)", line)
    if m_head:
        name = m_head.group(1).strip()
        prob = int(m_head.group(2))
        narrative = m_head.group(3).strip()
        
    t = re.search(r"trigger:\s*([^;]+)", narrative, flags=re.I)
    i = re.search(r"invalid(?:ation)?:\s*([^;]+)", narrative, flags=re.I)
    
    if t:
        trigger = t.group(1).strip().rstrip(".")
        narrative = narrative[:t.start()].strip().rstrip(";")
    if i:
        inv = i.group(1).strip().rstrip(".")
        # If invalidation came before trigger for some reason, re-strip narrative
        if "invalid" in narrative.lower():
            narrative = narrative.split("invalid")[0].strip().rstrip(";")
            
    return Scenario(name=name, probability_pct=prob, narrative=narrative, trigger=trigger, invalidation=inv)


def _parse_cross_asset(line: str) -> CrossAssetImpact:
    # Example: Equities: ▼ S&P -1-3% — Valuation multiple compression
    parts = line.split(":", 1)
    if len(parts) != 2:
        return CrossAssetImpact("Asset", "Unknown", line)
    
    asset_class = parts[0].strip()
    rest = parts[1].strip()
    
    # Split by em-dash or en-dash or hyphen surrounded by spaces
    sub = re.split(r"\s+[—–-]\s+", rest, maxsplit=1)
    if len(sub) == 2:
        return CrossAssetImpact(asset_class, sub[0].strip(), sub[1].strip())
    
    return CrossAssetImpact(asset_class, rest, "N/A")


def _parse_positioning(line: str) -> PositioningIdea:
    # Example: Overweight: Energy (XLE) — Supply shock benefits producers
    parts = line.split(":", 1)
    if len(parts) != 2:
        return PositioningIdea("Idea", line, "N/A")
        
    action = parts[0].strip()
    rest = parts[1].strip()
    
    sub = re.split(r"\s+[—–-]\s+", rest, maxsplit=1)
    if len(sub) == 2:
        return PositioningIdea(action, sub[0].strip(), sub[1].strip())
        
    return PositioningIdea(action, rest, "N/A")


def _parse_key_level(line: str) -> KeyLevel:
    # Example: WTI: $95 — Breakout level
    parts = line.split(":", 1)
    if len(parts) != 2:
        return KeyLevel("Instrument", "Level", line)
        
    instrument = parts[0].strip()
    rest = parts[1].strip()
    
    sub = re.split(r"\s+[—–-]\s+", rest, maxsplit=1)
    if len(sub) == 2:
        return KeyLevel(instrument, sub[0].strip(), sub[1].strip())
        
    return KeyLevel(instrument, rest, "N/A")
