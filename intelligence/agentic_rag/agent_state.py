"""
intelligence/agentic_rag/agent_state.py
----------------------------------------
Working memory for an agentic RAG session.

Holds all intermediate state across Plan→Act→Observe→Reflect iterations,
allowing agents to share context, track tool usage, and identify gaps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Result of a single tool invocation by an agent."""

    tool_name: str
    query: str
    result: Any
    elapsed_ms: int
    success: bool
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "query": self.query,
            "success": self.success,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


@dataclass
class ToolCall:
    """Record of a tool call made during the agentic loop."""

    tool_name: str
    query: str
    iteration: int
    result: ToolResult | None = None


@dataclass
class AgentOutput:
    """Structured output produced by a single specialist agent."""

    agent_name: str
    brief: str  # The agent's analysis
    confidence: float  # 0.0 – 1.0
    evidence_citations: list[str] = field(default_factory=list)  # [S1], [S2] etc.
    gaps_identified: list[str] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "brief": self.brief,
            "confidence": self.confidence,
            "evidence_citations": self.evidence_citations,
            "gaps_identified": self.gaps_identified,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class AgentState:
    """
    Full working memory for one agentic RAG session.

    This is the shared state object passed through the entire
    Plan → Act → Observe → Reflect loop. All agents and tools
    read from and write to this state.
    """

    # Core query
    question: str
    geography: str = "US"
    horizon: str = "MEDIUM_TERM"
    response_mode: str = "detailed"

    # Decomposition
    sub_questions: list[str] = field(default_factory=list)

    # Retrieved context (cumulative across iterations)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    live_indicators: dict[str, float] = field(default_factory=dict)
    live_meta: dict[str, Any] = field(default_factory=dict)

    # Tool calls record
    tool_calls: list[ToolCall] = field(default_factory=list)

    # Agent outputs
    agent_outputs: list[AgentOutput] = field(default_factory=list)

    # Synthesis / final state
    draft_answer: str = ""
    final_answer: str = ""

    # Gaps found by the reflection engine
    gaps: list[str] = field(default_factory=list)
    gap_queries: list[str] = field(default_factory=list)  # targeted retrieval queries

    # Loop control
    iteration: int = 0
    max_iterations: int = 2
    complete: bool = False

    # Timing
    start_time: float = field(default_factory=time.time)
    stage_times: dict[str, float] = field(default_factory=dict)

    # Quality
    quality_score: float = 0.0

    def add_observation(self, chunks: list[dict[str, Any]]) -> None:
        """Add newly retrieved chunks, deduplicating by title+text prefix."""
        existing_keys: set[tuple[str, str]] = set()
        for c in self.retrieved_chunks:
            md = c.get("metadata", {})
            key = (
                (md.get("title") or "").strip().lower()[:80],
                (c.get("text") or "").strip().lower()[:80],
            )
            existing_keys.add(key)

        for c in chunks:
            md = c.get("metadata", {})
            key = (
                (md.get("title") or "").strip().lower()[:80],
                (c.get("text") or "").strip().lower()[:80],
            )
            if key not in existing_keys:
                self.retrieved_chunks.append(c)
                existing_keys.add(key)

    def mark_gap(self, gap_description: str) -> None:
        """Record an identified gap in the current analysis."""
        if gap_description not in self.gaps:
            self.gaps.append(gap_description)

    def add_agent_output(self, output: AgentOutput) -> None:
        """Record output from a specialist agent."""
        self.agent_outputs.append(output)

    def is_complete(self) -> bool:
        """Check if the loop should terminate."""
        return self.complete or self.iteration >= self.max_iterations

    def agent_agreement_score(self) -> tuple[int, int]:
        """
        Return (agreeing_agents, total_agents) based on confidence scores.
        Agents with confidence >= 0.6 are considered "agreeing".
        """
        total = len(self.agent_outputs)
        agreeing = sum(1 for a in self.agent_outputs if a.confidence >= 0.6)
        return agreeing, total

    def elapsed_ms(self) -> int:
        return int((time.time() - self.start_time) * 1000)

    def mark_stage(self, stage: str) -> None:
        self.stage_times[stage] = time.time() - self.start_time

    def to_trace(self) -> dict[str, Any]:
        """Produce a full audit trace of this agent session."""
        agreeing, total = self.agent_agreement_score()
        return {
            "question": self.question,
            "geography": self.geography,
            "horizon": self.horizon,
            "response_mode": self.response_mode,
            "iterations": self.iteration,
            "sub_questions": self.sub_questions,
            "chunks_retrieved": len(self.retrieved_chunks),
            "live_indicators": len(self.live_indicators),
            "agent_outputs": [a.to_dict() for a in self.agent_outputs],
            "gaps_found": self.gaps,
            "gap_queries": self.gap_queries,
            "tool_calls": [
                {"tool": t.tool_name, "query": t.query, "iteration": t.iteration}
                for t in self.tool_calls
            ],
            "agent_agreement": f"{agreeing}/{total}",
            "quality_score": self.quality_score,
            "elapsed_ms": self.elapsed_ms(),
            "stage_times_ms": {k: int(v * 1000) for k, v in self.stage_times.items()},
        }
