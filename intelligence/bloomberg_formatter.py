"""
intelligence/bloomberg_formatter.py
-------------------------------------
Bloomberg Terminal-style output formatter for the Finance AI System.

Transforms structured AgentState or raw response text into Bloomberg-grade
analytical output in multiple format modes:

  1. morning_note   — Full Bloomberg Morning Note format (default for detailed mode)
  2. risk_matrix    — Probability × Impact risk/opportunity grid
  3. trade_idea     — Actionable setup with entry/target/stop/R:R
  4. brief          — Compact single-section summary (backward compatible)

Usage:
    from intelligence.bloomberg_formatter import BloombergFormatter
    formatter = BloombergFormatter()
    output = formatter.morning_note(state)
    output = formatter.format(state, mode="morning_note")
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intelligence.agentic_rag.agent_state import AgentState


class BloombergFormatter:
    """
    Formats analytical output in Bloomberg Terminal-grade styles.

    All methods accept either an AgentState (full agentic run) or a plain dict
    (for backward-compatible use with the existing pipeline output).
    """

    # ── Visual markers (Bloomberg terminal style) ─────────────────────────────

    DIVIDER = "━" * 68
    THIN_DIV = "─" * 68
    UP = "▲"
    DOWN = "▼"
    FLAT = "●"
    BULLET = "•"

    # Regime → display badge
    _REGIME_BADGES: dict[str, str] = {
        "RISK_ON": "🟢 RISK ON",
        "RISK_OFF": "🔴 RISK OFF",
        "STAGFLATION": "🟠 STAGFLATION",
        "RECESSION": "🔴 RECESSION",
        "RECOVERY": "🟡 RECOVERY",
        "REFLATION": "🟢 REFLATION",
        "UNKNOWN": "⚪ REGIME UNKNOWN",
    }

    # ── Direction helper ───────────────────────────────────────────────────────

    def _direction_arrow(self, value: float | None, prev: float | None = None) -> str:
        if value is None:
            return self.FLAT
        if prev is not None:
            if value > prev:
                return self.UP
            if value < prev:
                return self.DOWN
            return self.FLAT
        return self.FLAT

    # ── Indicator table builder ────────────────────────────────────────────────

    def _build_indicator_table(self, indicators: dict[str, float]) -> str:
        """Build a Bloomberg-style indicator table."""
        INDICATOR_LABELS: dict[str, tuple[str, str]] = {
            "sp500": ("S&P 500", "pts"),
            "nasdaq": ("Nasdaq", "pts"),
            "vix": ("VIX", "vol"),
            "yield_10y": ("US 10Y Yield", "%"),
            "yield_2y": ("US 2Y Yield", "%"),
            "yield_curve": ("Yield Curve (2s10s)", "bps"),
            "dxy": ("DXY (USD Index)", ""),
            "oil_wti": ("WTI Crude", "$/bbl"),
            "oil_brent": ("Brent Crude", "$/bbl"),
            "gold": ("Gold", "$/oz"),
            "inflation_cpi": ("CPI Inflation", "% YoY"),
            "fed_funds_rate": ("Fed Funds Rate", "%"),
            "credit_hy": ("HY Credit Spreads", "bps"),
            "unemployment": ("Unemployment", "%"),
            "gdp_growth": ("GDP Growth", "% QoQ"),
        }

        rows: list[str] = []
        for key, (label, unit) in INDICATOR_LABELS.items():
            val = indicators.get(key)
            if val is None:
                continue
            arrow = self.UP if val > 0 else (self.DOWN if val < 0 else self.FLAT)
            # Special cases: VIX up = bad, yield curve negative = inverted
            if key == "vix":
                arrow = self.DOWN if val < 20 else self.UP
            val_str = f"{val:,.2f}" if abs(val) >= 100 else f"{val:.2f}"
            rows.append(f"  {label:<28} {val_str:>10} {unit:<8} {arrow}")

        if not rows:
            return "  [No live market data available]"
        return "\n".join(rows[:12])  # Cap at 12 rows

    # ── Public formatting methods ──────────────────────────────────────────────

    def morning_note(
        self,
        state: "AgentState | None" = None,
        *,
        answer: str = "",
        indicators: dict[str, float] | None = None,
        regime: dict[str, Any] | None = None,
        cross_asset: dict[str, Any] | None = None,
        question: str = "",
        geography: str = "US",
        horizon: str = "MEDIUM_TERM",
        model_used: str = "",
    ) -> str:
        """
        Render a full Bloomberg Morning Note.

        Can be called with an AgentState (from agentic pipeline) or with
        individual keyword arguments (from existing pipeline — backward compat).
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d %H:%M UTC")

        # Extract from AgentState if provided
        if state is not None:
            answer = answer or state.final_answer or state.draft_answer
            indicators = indicators or state.live_indicators
            regime = regime or state.live_meta.get("regime", {})
            cross_asset = cross_asset or state.live_meta.get("cross_asset", {})
            question = question or state.question
            geography = state.geography
            horizon = state.horizon
            agreeing, total = state.agent_agreement_score()
            agent_agreement_str = f"{agreeing}/{total} agents agree"
        else:
            agent_agreement_str = "N/A"

        indicators = indicators or {}
        regime = regime or {}
        cross_asset = cross_asset or {}

        regime_label = self._REGIME_BADGES.get(
            (regime.get("regime") or "UNKNOWN").upper(), "⚪ UNKNOWN"
        )
        signal = (cross_asset.get("overall_signal") or "MIXED").upper()

        # Parse structured fields from the answer text
        fields = self._parse_answer_fields(answer)

        sections: list[str] = []

        # Header
        sections.append(
            f"{self.DIVIDER}\n"
            f"  MACRO AI INTELLIGENCE [{date_str}]\n"
            f"  {geography} | {horizon.replace('_', ' ')} | {regime_label} | Signal: {signal}\n"
            f"  Agent Agreement: {agent_agreement_str}\n"
            f"{self.DIVIDER}"
        )

        # Question
        sections.append(f"\n  QUERY: {question}\n")

        # Executive Summary
        exec_summary = fields.get("executive_summary") or fields.get("direct_answer") or ""
        if exec_summary:
            sections.append(f"{self.THIN_DIV}\n  EXECUTIVE SUMMARY\n{self.THIN_DIV}")
            sections.append(f"  {exec_summary}\n")

        # Live Data Snapshot
        if indicators:
            sections.append(f"{self.THIN_DIV}\n  LIVE MARKET SNAPSHOT\n{self.THIN_DIV}")
            sections.append(self._build_indicator_table(indicators))
            sections.append("")

        # Causal Chain
        causal = fields.get("causal_chain", "")
        if causal:
            sections.append(f"{self.THIN_DIV}\n  CAUSAL CHAIN\n{self.THIN_DIV}")
            sections.append(f"  {causal}\n")

        # What Is Happening + Market Impact
        what_happening = fields.get("what_is_happening", fields.get("why_likely", ""))
        market_impact = fields.get("market_impact", fields.get("market_map", ""))

        if what_happening or market_impact:
            sections.append(f"{self.THIN_DIV}\n  ANALYSIS\n{self.THIN_DIV}")
            if what_happening:
                sections.append(f"  What is happening:\n  {what_happening}\n")
            if market_impact:
                sections.append(f"  Market impact:\n  {market_impact}\n")

        # Scenarios
        scenarios = fields.get("scenarios", "")
        if scenarios:
            sections.append(f"{self.THIN_DIV}\n  SCENARIO MATRIX\n{self.THIN_DIV}")
            sections.append(f"  {scenarios}\n")

        # Consequences + What to Watch
        consequences = fields.get("consequences", fields.get("main_risks", ""))
        watch = fields.get("watch_next", "")
        if consequences or watch:
            sections.append(f"{self.THIN_DIV}\n  RISKS & CATALYSTS TO WATCH\n{self.THIN_DIV}")
            if consequences:
                sections.append(f"  {self.BULLET} Risks: {consequences}")
            if watch:
                sections.append(f"  {self.BULLET} Watch: {watch}")
            sections.append("")

        # Confidence + Footer
        confidence = fields.get("confidence", "")
        footer_parts = [f"Confidence: {confidence}" if confidence else ""]
        if model_used:
            footer_parts.append(f"Model: {model_used}")
        footer_str = " | ".join(p for p in footer_parts if p)
        sections.append(f"{self.DIVIDER}")
        if footer_str:
            sections.append(f"  {footer_str}")
        sections.append(f"{self.DIVIDER}\n")

        return "\n".join(sections)

    def risk_matrix(
        self,
        state: "AgentState | None" = None,
        *,
        answer: str = "",
        indicators: dict[str, float] | None = None,
        question: str = "",
    ) -> str:
        """
        Render a Risk/Opportunity Matrix with Probability × Impact assessment.
        """
        if state is not None:
            answer = answer or state.final_answer
            indicators = indicators or state.live_indicators
            question = question or state.question

        indicators = indicators or {}
        fields = self._parse_answer_fields(answer)

        vix = indicators.get("vix")
        hy_spreads = indicators.get("credit_hy")
        yield_curve = indicators.get("yield_curve")

        # Infer risk level from market data
        risk_level = "MODERATE"
        if vix is not None:
            if vix > 30:
                risk_level = "HIGH"
            elif vix < 15:
                risk_level = "LOW"

        sections: list[str] = [
            f"{self.DIVIDER}",
            f"  RISK MATRIX | {question[:60]}",
            f"{self.DIVIDER}",
            f"",
            f"  Market Stress Indicators:",
            f"  {'VIX':<20} {f'{vix:.1f}' if vix else 'N/A':>8}  {'(ELEVATED)' if vix and vix > 20 else '(NORMAL)'}",
            f"  {'HY Spread (bps)':<20} {f'{hy_spreads:.0f}' if hy_spreads else 'N/A':>8}  {'(WIDE)' if hy_spreads and hy_spreads > 450 else '(TIGHT)'}",
            f"  {'Yield Curve (bps)':<20} {f'{yield_curve:.0f}' if yield_curve else 'N/A':>8}  {'(INVERTED)' if yield_curve and yield_curve < 0 else '(NORMAL)'}",
            f"",
            f"  Overall Risk Level: {risk_level}",
            f"",
            f"  Scenario Probability Breakdown:",
        ]

        # Parse scenarios from answer
        answer_lower = answer.lower()
        base_pct = self._extract_probability(answer_lower, "base")
        bull_pct = self._extract_probability(answer_lower, "bull")
        bear_pct = self._extract_probability(answer_lower, "bear")

        sections.append(f"  {'Base Case':<20} {base_pct or '~55%':>8}  Most likely path")
        sections.append(f"  {'Bull Case':<20} {bull_pct or '~25%':>8}  Upside scenario")
        sections.append(f"  {'Bear Case':<20} {bear_pct or '~20%':>8}  Tail risk")
        sections.append(f"")

        risks = fields.get("consequences") or fields.get("main_risks") or ""
        if risks:
            sections.append(f"  Key Risks:")
            sections.append(f"  {risks}")

        sections.append(f"{self.DIVIDER}\n")
        return "\n".join(sections)

    def trade_idea(
        self,
        state: "AgentState | None" = None,
        *,
        answer: str = "",
        indicators: dict[str, float] | None = None,
        question: str = "",
    ) -> str:
        """
        Render an actionable Trade Idea format.
        Extracts direction, rationale, and key levels from the answer.
        """
        if state is not None:
            answer = answer or state.final_answer
            indicators = indicators or state.live_indicators
            question = question or state.question

        fields = self._parse_answer_fields(answer)
        summary = fields.get("executive_summary") or fields.get("direct_answer") or answer[:200]

        sections = [
            f"{self.DIVIDER}",
            f"  TRADE IDEA / ACTIONABLE VIEW",
            f"{self.DIVIDER}",
            f"",
            f"  Thesis: {summary[:200]}",
            f"",
            f"  Setup: Based on {question[:80]}",
            f"  Rationale: {fields.get('causal_chain', fields.get('what_is_happening', 'See full analysis'))[:200]}",
            f"",
            f"  Key risks: {fields.get('consequences', fields.get('main_risks', 'See risk matrix'))[:150]}",
            f"  What to watch: {fields.get('watch_next', 'Key data releases and policy decisions')[:150]}",
            f"",
            f"  Confidence: {fields.get('confidence', 'See analysis')}",
            f"{self.DIVIDER}\n",
        ]
        return "\n".join(sections)

    def brief(
        self,
        answer: str,
        *,
        question: str = "",
        model_used: str = "",
    ) -> str:
        """
        Compact brief format — backward compatible with existing pipeline output.
        Just adds a Bloomberg-style header/footer around the existing answer.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        header = f"{self.DIVIDER}\n  MACRO AI [{now}]\n{self.DIVIDER}\n"
        footer = f"\n{self.DIVIDER}" + (f"\n  Model: {model_used}" if model_used else "") + f"\n{self.DIVIDER}\n"
        return header + answer + footer

    def format(
        self,
        state: "AgentState | None" = None,
        mode: str = "morning_note",
        **kwargs: Any,
    ) -> str:
        """
        Unified entry point — dispatch to the appropriate format method.

        Args:
            state: AgentState from the agentic pipeline, or None
            mode: "morning_note" | "risk_matrix" | "trade_idea" | "brief"
            **kwargs: For non-agentic usage (answer, indicators, question, etc.)
        """
        if mode == "morning_note":
            return self.morning_note(state, **kwargs)
        if mode == "risk_matrix":
            return self.risk_matrix(state, **kwargs)
        if mode == "trade_idea":
            return self.trade_idea(state, **kwargs)
        return self.brief(kwargs.get("answer", ""), **{k: v for k, v in kwargs.items() if k != "answer"})

    # ── Internal parsers ───────────────────────────────────────────────────────

    def _parse_answer_fields(self, text: str) -> dict[str, str]:
        """Extract structured fields from a free-text LLM answer."""
        fields: dict[str, str] = {
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
            "why_likely": "",
            "market_map": "",
            "confidence": "",
        }
        current_field: str | None = None
        buffer: list[str] = []

        field_map: dict[str, str] = {
            "executive summary:": "executive_summary",
            "direct answer:": "direct_answer",
            "bottom line:": "direct_answer",
            "data snapshot:": "data_snapshot",
            "causal chain:": "causal_chain",
            "what is happening:": "what_is_happening",
            "market impact:": "market_impact",
            "scenarios": "scenarios",
            "consequences & risks:": "consequences",
            "consequences:": "consequences",
            "main risks:": "main_risks",
            "key risks:": "main_risks",
            "what to watch:": "watch_next",
            "what to watch next:": "watch_next",
            "confidence:": "confidence",
            "key drivers:": "why_likely",
            "why it matters:": "why_likely",
        }

        def _flush():
            if current_field and buffer:
                fields[current_field] = " ".join(buffer).strip()
            buffer.clear()

        for raw_line in text.splitlines():
            line = raw_line.strip()
            lower = line.lower()
            matched = False
            for prefix, fname in field_map.items():
                if lower.startswith(prefix):
                    _flush()
                    current_field = fname
                    val = line[len(prefix):].strip()
                    if val:
                        buffer.append(val)
                    matched = True
                    break
            if not matched and current_field and line:
                buffer.append(line)

        _flush()

        # Aliases
        if not fields["market_map"]:
            fields["market_map"] = fields["market_impact"]
        if not fields["why_likely"] and fields["what_is_happening"]:
            fields["why_likely"] = fields["what_is_happening"]

        return fields

    def _extract_probability(self, text: str, scenario: str) -> str:
        """Extract probability string for a named scenario case."""
        pattern = rf"{re.escape(scenario)}[^%]{{0,30}}?(\d{{1,3}}%)"
        m = re.search(pattern, text)
        return m.group(1) if m else ""
