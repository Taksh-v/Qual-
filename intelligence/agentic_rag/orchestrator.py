"""
intelligence/agentic_rag/orchestrator.py
-----------------------------------------
Main orchestration loop for the Bloomberg-grade Agentic RAG system.

Implements the Plan → Act → Observe → Reflect → Synthesize cycle:

  PLAN:     QueryPlanner decomposes the question into sub-questions
  ACT:      Tools run in parallel (semantic search + live market data)
  OBSERVE:  Specialist agents run in parallel, each producing a brief
  REFLECT:  ReflectionEngine assesses gaps, generates follow-up queries
  ITERATE:  If gaps exist and budget allows, go back to ACT with new queries
  SYNTHESIZE: Final synthesis agent produces Bloomberg-grade output

Each stage emits an AgentEvent (SSE-ready) so the UI can show progress.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .agent_state import AgentState, AgentOutput, ToolCall
from .query_planner import QueryPlanner
from .tool_registry import ToolRegistry, build_default_registry
from .reflection_engine import ReflectionEngine

logger = logging.getLogger(__name__)


# ── Event types for SSE streaming ────────────────────────────────────────────


@dataclass
class AgentEvent:
    """
    Typed SSE event emitted by the orchestrator at each pipeline stage.
    Can be serialized to JSON for Server-Sent Events.
    """

    stage: str  # planning | retrieval | agent_brief | reflection | synthesis | final | error
    data: dict[str, Any] = field(default_factory=dict)
    agent_name: str = ""
    iteration: int = 0
    elapsed_ms: int = 0

    def to_sse(self) -> str:
        """Serialize to SSE wire format."""
        payload = {
            "stage": self.stage,
            "agent_name": self.agent_name,
            "iteration": self.iteration,
            "elapsed_ms": self.elapsed_ms,
            **self.data,
        }
        return f"event: {self.stage}\ndata: {json.dumps(payload)}\n\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "agent_name": self.agent_name,
            "iteration": self.iteration,
            "elapsed_ms": self.elapsed_ms,
            **self.data,
        }


# ── Specialist Agent Runners ──────────────────────────────────────────────────


async def _run_macro_strategist(state: AgentState) -> AgentOutput:
    """
    MacroStrategist: Focuses on regime, yield curve, central bank signals.
    Uses live indicators + retrieved macro context.
    """
    t0 = time.time()
    agent_name = "MacroStrategist"

    macro_context = "\n".join(
        c.get("text", "")
        for c in state.retrieved_chunks
        if "macro" in (c.get("metadata", {}).get("data_type") or "").lower()
    )[:800] or "\n".join(c.get("text", "")[:200] for c in state.retrieved_chunks[:3])

    indicators_str = ", ".join(
        f"{k}={v:.2f}" for k, v in list(state.live_indicators.items())[:12]
    ) or "No live data available."

    prompt = (
        "You are the head of global macro strategy at a top-tier investment bank.\n"
        "Your job: assess the macro regime and its implications for the user's question.\n\n"
        f"QUESTION: {state.question}\n\n"
        f"LIVE INDICATORS:\n{indicators_str}\n\n"
        f"MACRO CONTEXT (from indexed research):\n{macro_context[:700]}\n\n"
        "Write a concise macro brief (150-200 words) covering:\n"
        "1. Current regime assessment (e.g. late-cycle, stagflation risk, reflation)\n"
        "2. Key macro driver for this question (with 1-2 data points)\n"
        "3. Rate/yield curve signal and its transmission mechanism\n"
        "4. Confidence level: HIGH/MEDIUM/LOW and why\n"
        "Be specific with numbers. No fluff."
    )

    try:
        from intelligence.llm_provider import generate_text
        brief, _ = generate_text(prompt, temperature=0.1, max_tokens=350, timeout_sec=45.0)
        confidence = 0.75 if any(kw in brief.lower() for kw in ["rate", "yield", "inflation", "gdp"]) else 0.50
        citations = [f"[S{i+1}]" for i in range(min(len(state.retrieved_chunks), 3))]
    except Exception as exc:
        logger.warning("[MacroStrategist] LLM failed: %s", exc)
        brief = f"Macro analysis unavailable: {exc}"
        confidence = 0.0
        citations = []

    return AgentOutput(
        agent_name=agent_name,
        brief=brief,
        confidence=confidence,
        evidence_citations=citations,
        elapsed_ms=int((time.time() - t0) * 1000),
    )


async def _run_sentiment_analyst(state: AgentState) -> AgentOutput:
    """
    SentimentAnalyst: Reads news/SEC sentiment, insider signals, earnings tone.
    """
    t0 = time.time()
    agent_name = "SentimentAnalyst"

    news_chunks = [
        c for c in state.retrieved_chunks
        if (c.get("metadata", {}).get("data_type") or "").lower() in ("news", "sec", "earnings_transcript")
    ][:5]
    if not news_chunks:
        news_chunks = state.retrieved_chunks[:4]

    news_text = "\n\n".join(
        f"[S{i+1}] {c.get('metadata', {}).get('title', 'Untitled')} | "
        f"{c.get('metadata', {}).get('source', '')} | "
        f"{c.get('metadata', {}).get('date', '')}\n{c.get('text', '')[:250]}"
        for i, c in enumerate(news_chunks)
    ) or "No news context retrieved."

    prompt = (
        "You are a behavioral finance and sentiment analyst.\n"
        "Assess the qualitative sentiment signals relevant to the user's question.\n\n"
        f"QUESTION: {state.question}\n\n"
        f"NEWS & FILINGS CONTEXT:\n{news_text}\n\n"
        "Write a focused sentiment brief (100-150 words) covering:\n"
        "1. Overall sentiment direction (bullish/bearish/neutral/mixed) with evidence\n"
        "2. Key narrative driving markets/media coverage\n"
        "3. Any contradictions (e.g. positive headlines but negative flow data)\n"
        "4. Sentiment momentum: is it accelerating or fading?\n"
        "Cite sources as [S1], [S2] etc. Be specific."
    )

    try:
        from intelligence.llm_provider import generate_text
        brief, _ = generate_text(prompt, temperature=0.1, max_tokens=300, timeout_sec=40.0)
        citations = [f"[S{i+1}]" for i in range(len(news_chunks))]
        confidence = 0.70 if news_chunks else 0.30
    except Exception as exc:
        logger.warning("[SentimentAnalyst] LLM failed: %s", exc)
        brief = f"Sentiment analysis unavailable: {exc}"
        confidence = 0.0
        citations = []

    return AgentOutput(
        agent_name=agent_name,
        brief=brief,
        confidence=confidence,
        evidence_citations=citations,
        elapsed_ms=int((time.time() - t0) * 1000),
    )


async def _run_risk_analyst(state: AgentState) -> AgentOutput:
    """
    RiskAnalyst: Focuses on tail risks, volatility, and probability-weighted scenarios.
    """
    t0 = time.time()
    agent_name = "RiskAnalyst"

    indicators_str = ", ".join(
        f"{k}={v:.2f}" for k, v in list(state.live_indicators.items())[:10]
    ) or "No live indicators."

    prompt = (
        "You are the chief risk officer at a global macro hedge fund.\n"
        "Provide a risk-focused analysis for the following question.\n\n"
        f"QUESTION: {state.question}\n\n"
        f"MARKET INDICATORS:\n{indicators_str}\n\n"
        "Write a risk brief (120-180 words) covering:\n"
        "- Base case (~55%): Most likely outcome with key catalyst\n"
        "- Bull case (~25%): Upside trigger and magnitude\n"
        "- Bear case (~20%): Tail risk trigger and severity\n"
        "- Top 2 risk factors to monitor (with specific levels/triggers)\n"
        "Use actual probabilities. Be direct. No hedging."
    )

    try:
        from intelligence.llm_provider import generate_text
        brief, _ = generate_text(prompt, temperature=0.15, max_tokens=320, timeout_sec=40.0)
        confidence = 0.65
        # Boost confidence if scenarios present
        if "base" in brief.lower() and ("bull" in brief.lower() or "bear" in brief.lower()):
            confidence = 0.80
    except Exception as exc:
        logger.warning("[RiskAnalyst] LLM failed: %s", exc)
        brief = f"Risk analysis unavailable: {exc}"
        confidence = 0.0

    return AgentOutput(
        agent_name=agent_name,
        brief=brief,
        confidence=confidence,
        elapsed_ms=int((time.time() - t0) * 1000),
    )


async def _run_technical_analyst(state: AgentState) -> AgentOutput:
    """
    TechnicalAnalyst: Reads momentum signals from VIX, DXY, yield curve, spreads.
    """
    t0 = time.time()
    agent_name = "TechnicalAnalyst"

    technical_indicators = {
        k: v for k, v in state.live_indicators.items()
        if k in ("vix", "dxy", "yield_10y", "yield_2y", "yield_curve", "sp500", "credit_hy", "oil_wti", "gold")
    }
    ind_str = ", ".join(f"{k}={v:.2f}" for k, v in technical_indicators.items()) or "No technical data."

    prompt = (
        "You are a quantitative technical analyst at a systematic trading desk.\n"
        "Assess momentum, positioning, and technical signals for the user's question.\n\n"
        f"QUESTION: {state.question}\n\n"
        f"TECHNICAL INDICATORS:\n{ind_str}\n\n"
        "Write a technical brief (100-150 words) covering:\n"
        "1. Key momentum signal from the most relevant indicator\n"
        "2. VIX/vol regime: Is fear elevated or suppressed?\n"
        "3. Dollar trend and its cross-asset implications\n"
        "4. One specific level to watch (with price/rate level)\n"
        "Be quantitative. State direction clearly: bullish/bearish/neutral."
    )

    try:
        from intelligence.llm_provider import generate_text
        brief, _ = generate_text(prompt, temperature=0.05, max_tokens=280, timeout_sec=35.0)
        confidence = 0.65 if technical_indicators else 0.30
    except Exception as exc:
        logger.warning("[TechnicalAnalyst] LLM failed: %s", exc)
        brief = f"Technical analysis unavailable: {exc}"
        confidence = 0.0

    return AgentOutput(
        agent_name=agent_name,
        brief=brief,
        confidence=confidence,
        elapsed_ms=int((time.time() - t0) * 1000),
    )


def _run_synthesis_agent_stream(state: AgentState):
    """
    SynthesisAgent (Head Macro Strategist): Synthesizes all agent briefs into a
    conversational Markdown final analysis. Yields tokens.
    """
    agent_briefs = "\n\n".join(
        f"--- {ao.agent_name} (confidence: {ao.confidence:.0%}) ---\n{ao.brief}"
        for ao in state.agent_outputs
        if ao.brief and "unavailable" not in ao.brief.lower()
    ) or "No agent briefs available."

    indicators_str = ", ".join(
        f"{k}={v:.2f}" for k, v in list(state.live_indicators.items())[:12]
    ) or "No live data."

    sources_str = "\n".join(
        f"[S{i+1}] {c.get('metadata', {}).get('title', 'Untitled')} | "
        f"{c.get('metadata', {}).get('source', '')} | "
        f"{c.get('metadata', {}).get('date', '')}"
        for i, c in enumerate(state.retrieved_chunks[:8])
    ) or "No sources available."

    from intelligence.prompt_templates import CONVERSATIONAL_SYNTHESIS_TEMPLATE
    prompt = CONVERSATIONAL_SYNTHESIS_TEMPLATE.format(
        question=state.question,
        geography=state.geography,
        horizon=state.horizon,
        indicators_str=indicators_str,
        agent_briefs=agent_briefs,
        sources_str=sources_str
    )

    try:
        from intelligence.llm_provider import generate_text_stream
        return generate_text_stream(prompt, temperature=0.20, max_tokens=1500, timeout_sec=120.0)
    except Exception as exc:
        logger.error("[SynthesisAgent] LLM failed: %s", exc)
        def fallback():
            yield f"Synthesis unavailable: {exc}"
        return fallback()


# ── Main Orchestrator ─────────────────────────────────────────────────────────


class AgenticOrchestrator:
    """
    Bloomberg-grade Agentic RAG Orchestrator.

    Implements the full Plan → Act → Observe → Reflect → Synthesize loop.
    Each stage emits AgentEvent objects for real-time SSE streaming.

    Usage:
        orchestrator = AgenticOrchestrator()
        async for event in orchestrator.run_async(question, ...):
            yield event.to_sse()
    """

    def __init__(
        self,
        max_iterations: int = 2,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.registry = registry or build_default_registry()
        self.planner = QueryPlanner()
        self.reflection = ReflectionEngine()

    async def run_async(
        self,
        question: str,
        geography: str = "US",
        horizon: str = "MEDIUM_TERM",
        response_mode: str = "detailed",
    ) -> AsyncIterator[AgentEvent]:
        """
        Run the full agentic pipeline asynchronously, yielding events as they occur.

        Yields AgentEvent objects at each pipeline stage.
        Final event (stage='final') contains the complete structured output.
        """
        state = AgentState(
            question=question,
            geography=geography,
            horizon=horizon,
            response_mode=response_mode,
            max_iterations=self.max_iterations,
        )
        state.mark_stage("start")
        overall_start = time.time()

        # ── STAGE 1: PLAN ────────────────────────────────────────────────────
        yield AgentEvent(
            stage="planning",
            data={"message": "Decomposing query into sub-questions..."},
            elapsed_ms=state.elapsed_ms(),
        )
        try:
            sub_qs = await asyncio.get_event_loop().run_in_executor(
                None, self.planner.decompose, question
            )
            state.sub_questions = sub_qs
        except Exception as exc:
            state.sub_questions = [question]
            logger.warning("[Orchestrator] Planning failed: %s", exc)

        state.mark_stage("planning")
        yield AgentEvent(
            stage="planning",
            data={"sub_questions": state.sub_questions, "count": len(state.sub_questions)},
            elapsed_ms=state.elapsed_ms(),
        )

        # ── STAGE 2: ACT — Parallel tool calls ──────────────────────────────
        await self._act_phase(state)
        state.mark_stage("retrieval")
        yield AgentEvent(
            stage="retrieval",
            data={
                "chunks_retrieved": len(state.retrieved_chunks),
                "indicators_fetched": len(state.live_indicators),
                "message": f"Retrieved {len(state.retrieved_chunks)} context chunks + {len(state.live_indicators)} indicators",
            },
            elapsed_ms=state.elapsed_ms(),
        )

        # ── STAGE 3: OBSERVE — Run specialist agents in parallel ─────────────
        yield AgentEvent(
            stage="agent_start",
            data={"message": "Running specialist agents in parallel..."},
            elapsed_ms=state.elapsed_ms(),
        )
        await self._observe_phase(state)
        state.mark_stage("agents")

        for ao in state.agent_outputs:
            yield AgentEvent(
                stage="agent_brief",
                agent_name=ao.agent_name,
                data={
                    "brief": ao.brief[:400],  # Truncate for SSE
                    "confidence": ao.confidence,
                    "elapsed_ms": ao.elapsed_ms,
                },
                iteration=state.iteration,
                elapsed_ms=state.elapsed_ms(),
            )

        # ── STAGE 4: SYNTHESIZE (first pass) ────────────────────────────────
        yield AgentEvent(
            stage="synthesis",
            data={"message": "Synthesizing agent briefs..."},
            elapsed_ms=state.elapsed_ms(),
        )
        
        stream_gen = _run_synthesis_agent_stream(state)
        loop = asyncio.get_event_loop()
        def get_next(gen):
            try:
                return next(gen)
            except StopIteration:
                return None
                
        draft_text = ""
        while True:
            token = await loop.run_in_executor(None, get_next, stream_gen)
            if token is None:
                break
            draft_text += token
            yield AgentEvent(stage="token", data={"text": token}, elapsed_ms=state.elapsed_ms())
            
        state.draft_answer = draft_text
        state.mark_stage("first_synthesis")

        # ── STAGE 5: REFLECT → ITERATE if needed ────────────────────────────
        while not state.is_complete():
            state.iteration += 1
            gaps = self.reflection.assess_gaps(state)
            state.gaps = gaps

            yield AgentEvent(
                stage="reflection",
                iteration=state.iteration,
                data={
                    "gaps": gaps,
                    "should_iterate": bool(gaps),
                    "iteration": state.iteration,
                },
                elapsed_ms=state.elapsed_ms(),
            )

            if not gaps:
                logger.info("[Orchestrator] No gaps found. Terminating loop.")
                break

            # Generate gap-targeted queries
            gap_queries = self.reflection.generate_follow_up_queries(gaps, question)
            state.gap_queries = gap_queries

            if not self.reflection.should_iterate(state):
                break

            # Additional retrieval for gaps
            yield AgentEvent(
                stage="gap_retrieval",
                iteration=state.iteration,
                data={"gap_queries": gap_queries, "message": f"Filling {len(gaps)} gaps..."},
                elapsed_ms=state.elapsed_ms(),
            )
            await self._act_phase(state, queries=gap_queries)

            # Re-synthesize with enriched context
            stream_gen = _run_synthesis_agent_stream(state)
            draft_text = ""
            while True:
                token = await loop.run_in_executor(None, get_next, stream_gen)
                if token is None:
                    break
                draft_text += token
                yield AgentEvent(stage="token", data={"text": token}, elapsed_ms=state.elapsed_ms())
            
            state.draft_answer = draft_text
            state.mark_stage(f"synthesis_iter_{state.iteration}")

        state.final_answer = state.draft_answer
        state.complete = True

        # ── FINAL EVENT ──────────────────────────────────────────────────────
        agreeing, total = state.agent_agreement_score()
        from intelligence.bloomberg_formatter import BloombergFormatter
        formatter = BloombergFormatter()

        state.mark_stage("complete")
        yield AgentEvent(
            stage="final",
            data={
                "question": question,
                "answer": state.final_answer,
                "sources": [
                    {
                        "title": c.get("metadata", {}).get("title", "Unknown"),
                        "source": c.get("metadata", {}).get("source", ""),
                        "date": c.get("metadata", {}).get("date", ""),
                    }
                    for c in state.retrieved_chunks[:8]
                ],
                "agent_agreement": f"{agreeing}/{total}",
                "iterations": state.iteration,
                "live_indicators_count": len(state.live_indicators),
                "chunks_used": len(state.retrieved_chunks),
                "sub_questions": state.sub_questions,
                "gaps_found": state.gaps,
                "trace": state.to_trace(),
                "elapsed_ms": state.elapsed_ms(),
            },
            elapsed_ms=state.elapsed_ms(),
        )

    async def _act_phase(
        self,
        state: AgentState,
        queries: list[str] | None = None,
    ) -> None:
        """
        Parallel tool execution phase (ACT).
        Runs semantic search (for each query) and live market fetch concurrently.
        """
        search_tool = self.registry.get("semantic_search")
        live_tool = self.registry.get("live_market")
        cross_tool = self.registry.get("cross_asset")

        search_queries = queries or state.sub_questions or [state.question]

        async def _search(q: str) -> None:
            if search_tool:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, search_tool.run, q
                )
                state.tool_calls.append(ToolCall("semantic_search", q, state.iteration, None))
                if result.success and result.data:
                    state.add_observation(result.data)

        async def _live_market() -> None:
            if live_tool and not state.live_indicators:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, live_tool.run, ""
                )
                if result.success and result.data:
                    state.live_indicators = result.data.get("indicators", {})
                    state.live_meta = result.data.get("meta", {})

        async def _cross_asset() -> None:
            if cross_tool and state.live_indicators:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: cross_tool.run("", indicators=state.live_indicators)
                )
                if result.success:
                    state.live_meta["cross_asset"] = result.data

        # Run all searches + live market fetch in parallel
        tasks = [_search(q) for q in search_queries] + [_live_market(), _cross_asset()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _observe_phase(self, state: AgentState) -> None:
        """
        Parallel specialist agent execution phase (OBSERVE).
        Runs all four agents concurrently via asyncio.gather.
        """
        results = await asyncio.gather(
            _run_macro_strategist(state),
            _run_sentiment_analyst(state),
            _run_risk_analyst(state),
            _run_technical_analyst(state),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, AgentOutput):
                state.add_agent_output(result)
            elif isinstance(result, Exception):
                logger.warning("[Orchestrator] Agent failed: %s", result)
