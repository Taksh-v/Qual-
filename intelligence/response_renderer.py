"""
response_renderer.py
--------------------
Multi-format renderer for StructuredResponse objects.

Separates data (StructuredResponse) from presentation (rendering).
Replaces the fragile to_text() → regex → re-parse round-trip with
clean object-level validation followed by format-specific rendering.

Supported formats:
  - render_markdown()  — Rich markdown with bold headers and tables
  - render_plain()     — Flat text (backward-compatible with legacy to_text())
  - render_json()      — API response body (dict)
  - render_api()       — Full structured API payload with metadata

Usage:
    from intelligence.response_renderer import ResponseRenderer
    md = ResponseRenderer.render_markdown(response)
"""

from __future__ import annotations

from typing import Any

from intelligence.response_schema import StructuredResponse


class ResponseRenderer:
    """Stateless renderer — all methods are static."""

    # ── Plain text (legacy-compatible) ─────────────────────────────────────

    @staticmethod
    def render_plain(resp: StructuredResponse) -> str:
        """Render to flat text — matches the existing to_text() output exactly."""
        return resp.to_text()

    # ── Rich Markdown ──────────────────────────────────────────────────────

    @staticmethod
    def render_markdown(resp: StructuredResponse) -> str:
        """Render to rich markdown with bold headers, emoji indicators, and tables."""
        lines: list[str] = []

        # ── Regime Dashboard ───────────────────────────────────────────────────
        if resp.regime or resp.dominant_theme:
            regime_str = resp.regime or "Transitional"
            theme_str = resp.dominant_theme or "Mixed Signals"
            lines.append("## 🧭 Regime & Signal Dashboard")
            lines.append("```text")
            lines.append(f"Regime:         {regime_str}")
            lines.append(f"Dominant Theme: {theme_str}")
            lines.append(f"Conviction:     {resp.confidence.split()[0] if resp.confidence else 'N/A'}")
            lines.append("```\n")

        if resp.executive_summary:
            lines.append(f"## 📋 Executive Summary\n{resp.executive_summary}\n")

        lines.append(f"## 🎯 Direct Answer\n{resp.direct_answer}\n")
        lines.append(f"## 📊 Data Snapshot\n`{resp.data_snapshot}`\n")
        
        # ── Causal Architecture ────────────────────────────────────────────────
        if resp.causal_architecture:
            lines.append("## 🔗 Causal Architecture")
            if resp.causal_architecture.primary:
                chain = " → ".join(link.effect for link in resp.causal_architecture.primary)
                chain = f"{resp.causal_architecture.primary[0].trigger} → {chain}"
                lines.append(f"**Primary:** {chain}")
            if resp.causal_architecture.secondary:
                chain = " → ".join(link.effect for link in resp.causal_architecture.secondary)
                chain = f"{resp.causal_architecture.secondary[0].trigger} → {chain}"
                lines.append(f"**Secondary:** {chain}")
            if resp.causal_architecture.feedback_loop:
                lines.append(f"**Feedback:** {resp.causal_architecture.feedback_loop}")
            lines.append("")
        elif resp.causal_chain:
            lines.append(f"## 🔗 Causal Chain\n> {resp.causal_chain}\n")

        if resp.what_is_happening:
            lines.append("## 📰 What Is Happening")
            for item in resp.what_is_happening:
                lines.append(f"- {item}")
            lines.append("")

        # ── Cross-Asset Impact / Market Impact ─────────────────────────────────
        if resp.cross_asset_impacts:
            lines.append("## 📈 Cross-Asset Impact")
            lines.append("| Asset Class | Direction | Mechanism |")
            lines.append("|-------------|-----------|-----------|")
            for ca in resp.cross_asset_impacts:
                lines.append(ca.as_row())
            lines.append("")
        elif resp.market_impact:
            lines.append("## 📈 Market Impact")
            for item in resp.market_impact:
                lines.append(f"- {item}")
            lines.append("")

        # ── Positioning ────────────────────────────────────────────────────────
        if resp.positioning:
            lines.append("## 🛡️ Positioning Implications")
            for p in resp.positioning:
                lines.append(f"- **{p.action}:** {p.instruments} — {p.rationale}")
            lines.append("")

        if resp.predicted_events:
            lines.append("## 🔮 Predicted Catalysts")
            lines.append("| Event | Horizon | Probability | Trigger | Invalidation |")
            lines.append("|-------|---------|-------------|---------|--------------|")
            for ev in resp.predicted_events:
                prob = f"{ev.probability_pct}%" if ev.probability_pct is not None else "N/A"
                lines.append(
                    f"| {ev.label} | {ev.horizon} | {prob} | {ev.trigger} | {ev.invalidation} |"
                )
            lines.append("")

        if resp.scenarios:
            lines.append("## 📐 Scenario Analysis")
            lines.append("| Scenario | Prob | Narrative | Trigger | Invalidation |")
            lines.append("|----------|------|-----------|---------|--------------|")
            for sc in resp.scenarios:
                prob = f"~{sc.probability_pct}%" if sc.probability_pct is not None else "N/A"
                lines.append(f"| **{sc.name}** | {prob} | {sc.narrative} | {sc.trigger} | {sc.invalidation} |")
            lines.append("")

        # ── Key Levels ─────────────────────────────────────────────────────────
        if resp.key_levels:
            lines.append("## 📏 Key Levels to Watch")
            for kl in resp.key_levels:
                lines.append(f"- **{kl.instrument}**: {kl.level} — *{kl.significance}*")
            lines.append("")

        if resp.key_risks:
            lines.append("## ⚠️ Key Risks")
            for item in resp.key_risks:
                lines.append(f"- {item}")
            lines.append("")

        # ── Historical Analog ──────────────────────────────────────────────────
        if resp.historical_analog:
            lines.append("## 🕰️ Historical Analog")
            lines.append(f"> {resp.historical_analog}\n")

        if resp.what_to_watch:
            lines.append("## 👁️ What to Watch")
            for item in resp.what_to_watch:
                lines.append(f"- {item}")
            lines.append("")

        if resp.data_gaps:
            lines.append("## 🕳️ Data Gaps")
            for gap in resp.data_gaps:
                lines.append(f"- {gap}")
            lines.append("")

        lines.append(f"**Confidence:** {resp.confidence}")

        return "\n".join(lines)

    # ── JSON / dict ────────────────────────────────────────────────────────

    @staticmethod
    def render_json(resp: StructuredResponse) -> dict[str, Any]:
        """Render to a structured dict suitable for JSON API responses."""
        result: dict[str, Any] = {}

        if resp.regime:
            result["regime"] = resp.regime
        if resp.dominant_theme:
            result["dominant_theme"] = resp.dominant_theme

        if resp.executive_summary:
            result["executive_summary"] = resp.executive_summary

        result["direct_answer"] = resp.direct_answer
        result["data_snapshot"] = resp.data_snapshot
        
        if resp.causal_architecture:
            result["causal_architecture"] = {
                "primary": [{"trigger": l.trigger, "effect": l.effect} for l in resp.causal_architecture.primary],
                "secondary": [{"trigger": l.trigger, "effect": l.effect} for l in resp.causal_architecture.secondary],
                "feedback_loop": resp.causal_architecture.feedback_loop
            }
        else:
            result["causal_chain"] = resp.causal_chain

        result["what_is_happening"] = resp.what_is_happening
        
        if resp.cross_asset_impacts:
            result["cross_asset_impacts"] = [
                {"asset_class": c.asset_class, "direction": c.direction, "mechanism": c.mechanism}
                for c in resp.cross_asset_impacts
            ]
        else:
            result["market_impact"] = resp.market_impact
            
        if resp.positioning:
            result["positioning"] = [
                {"action": p.action, "instruments": p.instruments, "rationale": p.rationale}
                for p in resp.positioning
            ]

        result["predicted_events"] = [
            {
                "label": ev.label,
                "horizon": ev.horizon,
                "probability_pct": ev.probability_pct,
                "narrative": ev.narrative,
                "trigger": ev.trigger,
                "invalidation": ev.invalidation,
            }
            for ev in resp.predicted_events
        ]

        result["scenarios"] = [
            {
                "name": sc.name,
                "probability_pct": sc.probability_pct,
                "narrative": sc.narrative,
                "trigger": sc.trigger,
                "invalidation": sc.invalidation,
            }
            for sc in resp.scenarios
        ]
        
        if resp.key_levels:
            result["key_levels"] = [
                {"instrument": kl.instrument, "level": kl.level, "significance": kl.significance}
                for kl in resp.key_levels
            ]

        if resp.historical_analog:
            result["historical_analog"] = resp.historical_analog

        if resp.key_risks:
            result["key_risks"] = resp.key_risks
        if resp.time_horizons:
            result["time_horizons"] = resp.time_horizons

        result["what_to_watch"] = resp.what_to_watch
        
        if resp.data_gaps:
            result["data_gaps"] = resp.data_gaps
            
        result["confidence"] = resp.confidence

        return result

    # ── Full API payload ───────────────────────────────────────────────────

    @staticmethod
    def render_api(
        resp: StructuredResponse,
        *,
        model_used: str = "N/A",
        quality_score: int | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        """Render to full API payload with metadata."""
        payload = ResponseRenderer.render_json(resp)
        payload["metadata"] = {
            "mode": resp.metadata.mode,
            "model_used": model_used or resp.metadata.model_used,
            "quality_score": quality_score if quality_score is not None else resp.metadata.quality_score,
            "citation_count": resp.metadata.citation_count,
            "warnings": warnings if warnings is not None else resp.metadata.warnings,
            "generated_at": resp.metadata.generated_at,
        }
        return payload
