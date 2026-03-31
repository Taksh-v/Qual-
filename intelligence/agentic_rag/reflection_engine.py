"""
intelligence/agentic_rag/reflection_engine.py
----------------------------------------------
Self-critique and gap detection for the agentic RAG loop.

The ReflectionEngine acts as an internal critic that:
  1. Assesses whether the current draft answer covers the question adequately
  2. Identifies specific gaps (missing data, unsupported claims, vague scenarios)
  3. Generates targeted follow-up queries to fill those gaps
  4. Decides whether another iteration is warranted

This implements the "Observe → Reflect" part of the Plan→Act→Observe→Reflect loop.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .agent_state import AgentState

logger = logging.getLogger(__name__)


# ── Gap detection heuristics ─────────────────────────────────────────────────


def _citation_density(text: str) -> float:
    """Fraction of lines that contain at least one citation [Sx]."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    cited = sum(1 for l in lines if "[S" in l)
    return cited / len(lines)


def _has_quantitative_data(text: str) -> bool:
    """Check if text contains at least 3 numbers/percentages."""
    nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
    return len(nums) >= 3


def _has_scenario_coverage(text: str) -> bool:
    """Check if text covers at least 2 of the 3 scenarios."""
    t = text.lower()
    count = sum(1 for s in ["base", "bull", "bear", "upside", "downside", "risk"] if s in t)
    return count >= 2


def _has_direct_answer(text: str) -> bool:
    """Check if text contains a direct answer statement."""
    t = text.lower()
    return any(
        phrase in t
        for phrase in [
            "direct answer:",
            "executive summary:",
            "bottom line:",
            "in summary",
            "the answer is",
        ]
    )


def _has_market_impact(text: str) -> bool:
    """Check if text covers market impact."""
    t = text.lower()
    return any(
        kw in t
        for kw in ["equity", "equities", "bond", "yield", "rate", "fx", "currency", "commodity"]
    )


# ── Gap descriptions ───────────────────────────────────────────────────────────


_GAP_CHECKS: list[tuple[str, Any, str]] = [
    ("No direct answer found", lambda text, _state: not _has_direct_answer(text), "Provide a direct, specific answer to the question."),
    ("Missing quantitative data", lambda text, _state: not _has_quantitative_data(text), "Include specific numbers, percentages, or basis points to support the analysis."),
    ("No citation support", lambda text, _state: _citation_density(text) < 0.15, "Improve citation coverage: each key claim needs a source reference."),
    ("Missing scenario analysis", lambda text, _state: not _has_scenario_coverage(text), "Add scenario analysis with base/bull/bear cases and probability estimates."),
    ("No market impact assessment", lambda text, _state: not _has_market_impact(text), "Assess impact across asset classes: equities, rates, FX, commodities."),
    ("No retrieved context", lambda text, state: len(state.retrieved_chunks) < 2, "Retrieve more relevant news and research context for this question."),
]


class ReflectionEngine:
    """
    Critiques the current draft answer and identifies gaps that warrant
    additional retrieval or agent analysis.

    Usage:
        engine = ReflectionEngine()
        gaps = engine.assess_gaps(state)
        queries = engine.generate_follow_up_queries(gaps, state.question)
        should_go_again = engine.should_iterate(state)
    """

    def __init__(self, llm_generate_fn: Any | None = None) -> None:
        """
        Args:
            llm_generate_fn: Optional LLM callable for deeper gap analysis.
                             Falls back to heuristic checks if None or on failure.
        """
        self._llm_fn = llm_generate_fn

    def _get_llm(self):
        if self._llm_fn is not None:
            return self._llm_fn
        from intelligence.llm_provider import generate_text

        def _call(prompt: str) -> str:
            text, _ = generate_text(
                prompt,
                temperature=0.0,
                max_tokens=300,
                timeout_sec=20.0,
            )
            return text

        return _call

    def _heuristic_gaps(self, draft: str, state: AgentState) -> list[str]:
        """Fast, LLM-free gap detection using pattern matching."""
        gaps: list[str] = []
        for _name, check_fn, gap_description in _GAP_CHECKS:
            try:
                if check_fn(draft, state):
                    gaps.append(gap_description)
            except Exception:
                pass
        return gaps

    def _llm_critique(self, draft: str, question: str) -> list[str]:
        """
        Use LLM as a critic to identify analytical gaps in the draft answer.
        Returns list of gap descriptions, or empty list on failure.
        """
        prompt = (
            "You are a senior financial editor reviewing a draft analysis. "
            "Identify the top 1-3 specific gaps, missing data points, or unsupported claims in this draft. "
            "Be specific: name the exact missing piece (e.g. 'Missing: Fed Funds rate impact on mortgage spreads').\n\n"
            "Be concise. Output ONLY the gaps, one per line, no explanation, no numbering.\n"
            "If the draft is complete and high quality, output: COMPLETE\n\n"
            f"Question: {question}\n\n"
            f"Draft (first 1200 chars):\n{draft[:1200]}\n\n"
            "Gaps:"
        )
        try:
            llm = self._get_llm()
            raw = llm(prompt).strip()

            if "COMPLETE" in raw.upper() and len(raw) < 30:
                return []

            gaps: list[str] = []
            for line in raw.splitlines():
                line = line.strip().lstrip("-•*1234567890.)").strip()
                if len(line) >= 10 and len(line) <= 200:
                    if not any(line.lower() == g.lower() for g in gaps):
                        gaps.append(line)
            return gaps[:3]
        except Exception as exc:
            logger.debug("[ReflectionEngine] LLM critique failed: %s", exc)
            return []

    def assess_gaps(self, state: AgentState) -> list[str]:
        """
        Assess the current draft answer and return a list of gap descriptions.

        Combines fast heuristic checks with optional LLM critique.
        """
        draft = state.draft_answer or state.final_answer or ""

        # Always run heuristics (fast, no LLM)
        heuristic_gaps = self._heuristic_gaps(draft, state)

        # On first iteration with a non-trivial draft, also try LLM critique
        llm_gaps: list[str] = []
        if state.iteration == 0 and len(draft) >= 100:
            try:
                llm_gaps = self._llm_critique(draft, state.question)
            except Exception:
                pass

        # Merge and deduplicate
        all_gaps: list[str] = list(heuristic_gaps)
        for g in llm_gaps:
            if not any(g.lower()[:40] in existing.lower() for existing in all_gaps):
                all_gaps.append(g)

        logger.info("[ReflectionEngine] Found %d gaps after iteration %d.", len(all_gaps), state.iteration)
        return all_gaps

    def should_iterate(self, state: AgentState) -> bool:
        """
        Decide whether to run another retrieval + agent iteration.

        Returns True if:
        - Iteration limit not yet reached
        - Meaningful gaps were identified
        - We have gap-targeted queries to run
        """
        if state.is_complete():
            return False
        if not state.gaps:
            return False
        if not state.gap_queries:
            return False
        return True

    def generate_follow_up_queries(
        self, gaps: list[str], original_question: str
    ) -> list[str]:
        """
        Generate targeted retrieval queries to address the identified gaps.

        Each gap gets one focused search query designed to retrieve
        relevant documents from the FAISS index.
        """
        if not gaps:
            return []

        queries: list[str] = []

        # Fast deterministic mapping for common gaps
        gap_query_rules: list[tuple[str, str]] = [
            ("quantitative", f"{original_question} data statistics numbers"),
            ("citation", f"{original_question} evidence sources"),
            ("scenario", f"{original_question} scenarios outlook risks"),
            ("market impact", f"{original_question} market impact assets"),
            ("context", original_question + " background analysis"),
            ("direct answer", f"{original_question} conclusion result"),
        ]

        for gap in gaps[:3]:
            matched = False
            for keyword, query_template in gap_query_rules:
                if keyword.lower() in gap.lower():
                    queries.append(query_template)
                    matched = True
                    break
            if not matched:
                # Generic fallback: use the gap as a search query
                queries.append(f"{original_question} {gap[:60]}")

        # Deduplicate
        seen: set[str] = set()
        unique_queries: list[str] = []
        for q in queries:
            q_norm = q.lower()[:80]
            if q_norm not in seen:
                seen.add(q_norm)
                unique_queries.append(q)

        logger.info("[ReflectionEngine] Generated %d follow-up queries.", len(unique_queries))
        return unique_queries[:3]
