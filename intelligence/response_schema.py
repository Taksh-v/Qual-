from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class Citation:
    source_id: str
    confidence: float | None = None


@dataclass
class ResponseSection:
    title: str
    bullets: list[str] = field(default_factory=list)
    paragraph: str = ""
    citations: list[Citation] = field(default_factory=list)

    def as_text(self) -> str:
        lines: list[str] = [f"{self.title}:"]
        if self.paragraph:
            lines.append(self.paragraph)
        lines.extend(f"- {b}" for b in self.bullets)
        return "\n".join(lines)


@dataclass
class PredictedEvent:
    label: str
    horizon: str
    probability_pct: int | None
    narrative: str
    trigger: str
    invalidation: str

    def as_bullet(self) -> str:
        prob = f"{self.probability_pct}%" if self.probability_pct is not None else "N/A"
        return (
            f"{self.label} ({self.horizon}, {prob}): {self.narrative}; "
            f"trigger: {self.trigger}; invalidation: {self.invalidation}."
        )


@dataclass
class Scenario:
    name: str
    probability_pct: int | None
    narrative: str
    trigger: str = ""
    invalidation: str = ""

    def as_bullet(self) -> str:
        prob = f"~{self.probability_pct}%" if self.probability_pct is not None else "N/A"
        return f"{self.name} ({prob}): {self.narrative}."


# ── New dataclasses for Macro Intelligence Briefing ───────────────────────────


@dataclass
class CausalChainLink:
    """Single step in a causal chain: trigger → effect."""
    trigger: str
    effect: str

    def as_text(self) -> str:
        return f"{self.trigger} → {self.effect}"


@dataclass
class CausalArchitecture:
    """Multi-dimensional causal reasoning with primary, secondary, and feedback chains."""
    primary: list[CausalChainLink] = field(default_factory=list)
    secondary: list[CausalChainLink] = field(default_factory=list)
    feedback_loop: str = ""

    def as_text(self) -> str:
        lines: list[str] = []
        if self.primary:
            chain = " → ".join(link.effect for link in self.primary)
            if self.primary:
                chain = self.primary[0].trigger + " → " + chain
            lines.append(f"Primary: {chain}")
        if self.secondary:
            chain = " → ".join(link.effect for link in self.secondary)
            if self.secondary:
                chain = self.secondary[0].trigger + " → " + chain
            lines.append(f"Secondary: {chain}")
        if self.feedback_loop:
            lines.append(f"Feedback: {self.feedback_loop}")
        return "\n".join(lines)


@dataclass
class CrossAssetImpact:
    """Impact on a specific asset class with direction and mechanism."""
    asset_class: str     # e.g. "Equities", "Rates", "FX", "Commodities", "Credit"
    direction: str       # e.g. "▼ S&P -1-3% over 2-4wk"
    mechanism: str       # e.g. "Multiple compression from higher inflation expectations"

    def as_row(self) -> str:
        return f"| {self.asset_class} | {self.direction} | {self.mechanism} |"


@dataclass
class PositioningIdea:
    """Actionable positioning recommendation."""
    action: str          # "Overweight", "Underweight", "Hedge", "Long", "Short"
    instruments: str     # e.g. "Energy (XLE), Gold miners"
    rationale: str       # e.g. "Supply shock benefits energy producers"

    def as_bullet(self) -> str:
        return f"{self.action}: {self.instruments} — {self.rationale}"


@dataclass
class KeyLevel:
    """Critical price/yield level that would change the market view."""
    instrument: str      # e.g. "WTI", "VIX", "S&P 500"
    level: str           # e.g. "$95 (breakout)", "25 (normalization)"
    significance: str    # e.g. "Above triggers bear scenario"

    def as_bullet(self) -> str:
        return f"{self.instrument}: {self.level} — {self.significance}"


@dataclass
class ResponseMetadata:
    mode: str = "brief"
    model_used: str = "N/A"
    quality_score: int | None = None
    citation_count: int | None = None
    warnings: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class StructuredResponse:
    # ── Core fields (backward-compatible) ─────────────────────────────────
    direct_answer: str
    data_snapshot: str
    causal_chain: str
    what_is_happening: list[str] = field(default_factory=list)
    market_impact: list[str] = field(default_factory=list)
    predicted_events: list[PredictedEvent] = field(default_factory=list)
    scenarios: list[Scenario] = field(default_factory=list)
    what_to_watch: list[str] = field(default_factory=list)
    confidence: str = "MEDIUM - Partial data available; interpret with caution."
    executive_summary: str | None = None
    key_risks: list[str] = field(default_factory=list)
    time_horizons: list[str] = field(default_factory=list)
    metadata: ResponseMetadata = field(default_factory=ResponseMetadata)

    # ── New Macro Intelligence Briefing fields ────────────────────────────
    regime: str | None = None                                        # "Risk-On" / "Risk-Off" / "Transitional"
    dominant_theme: str | None = None                                # e.g. "Supply-Shock Inflation"
    causal_architecture: CausalArchitecture | None = None            # Multi-chain causal reasoning
    cross_asset_impacts: list[CrossAssetImpact] = field(default_factory=list)
    positioning: list[PositioningIdea] = field(default_factory=list)
    key_levels: list[KeyLevel] = field(default_factory=list)
    historical_analog: str | None = None                             # Comparable episode
    data_gaps: list[str] = field(default_factory=list)               # Explicitly missing data

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_text(self) -> str:
        """Render to flat text — backward-compatible with legacy format,
        extended with new sections when populated."""
        lines: list[str] = []

        # ── Regime dashboard (new) ────────────────────────────────────
        if self.regime:
            lines.append(f"Regime: {self.regime}")
        if self.dominant_theme:
            lines.append(f"Dominant theme: {self.dominant_theme}")
        if self.regime or self.dominant_theme:
            lines.append("")

        # ── Legacy-compatible core ────────────────────────────────────
        if self.executive_summary:
            lines.append(f"Executive summary: {self.executive_summary}")
        lines.extend(
            [
                f"Direct answer: {self.direct_answer}",
                f"Data snapshot: {self.data_snapshot}",
                f"Causal chain: {self.causal_chain}",
            ]
        )

        # ── Causal architecture (new) ─────────────────────────────────
        if self.causal_architecture:
            lines.append("Causal architecture:")
            lines.append(self.causal_architecture.as_text())

        lines.append("What is happening:")
        lines.extend(f"- {item}" for item in self.what_is_happening)

        # ── Cross-asset impacts (new, also emits legacy market_impact) ─
        if self.cross_asset_impacts:
            lines.append("Cross-asset impact:")
            for impact in self.cross_asset_impacts:
                lines.append(f"- {impact.asset_class}: {impact.direction} — {impact.mechanism}")
        if self.market_impact:
            lines.append("Market impact:")
            lines.extend(f"- {item}" for item in self.market_impact)

        lines.append("Predicted events:")
        lines.extend(f"- {item.as_bullet()}" for item in self.predicted_events)
        lines.append("Scenarios (probabilities must add to 100%):")
        lines.extend(f"- {item.as_bullet()}" for item in self.scenarios)

        if self.key_risks:
            lines.append("Key risks:")
            lines.extend(f"- {item}" for item in self.key_risks)
        if self.time_horizons:
            lines.append("Time horizons:")
            lines.extend(f"- {item}" for item in self.time_horizons)

        # ── Positioning (new) ─────────────────────────────────────────
        if self.positioning:
            lines.append("Positioning:")
            lines.extend(f"- {p.as_bullet()}" for p in self.positioning)

        # ── Key levels (new) ──────────────────────────────────────────
        if self.key_levels:
            lines.append("Key levels:")
            lines.extend(f"- {kl.as_bullet()}" for kl in self.key_levels)

        # ── Historical analog (new) ───────────────────────────────────
        if self.historical_analog:
            lines.append(f"Historical analog: {self.historical_analog}")

        lines.append("What to watch:")
        lines.extend(f"- {item}" for item in self.what_to_watch)

        # ── Data gaps (new) ───────────────────────────────────────────
        if self.data_gaps:
            lines.append("Data gaps:")
            lines.extend(f"- {g}" for g in self.data_gaps)

        lines.append(f"Confidence: {self.confidence}")
        return "\n".join(lines)
