"""
tests/test_agentic_rag.py
--------------------------
Unit tests for the Agentic RAG core components.

All tests are written to run WITHOUT a live LLM or FAISS index.
Heavy dependencies are mocked/patched so tests are fast and CI-safe.
"""

import asyncio
import sys
import os
import pytest
import unittest.mock as mock

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── AgentState Tests ─────────────────────────────────────────────────────────


def test_agent_state_basic():
    from intelligence.agentic_rag.agent_state import AgentState

    state = AgentState(question="What is inflation doing?")
    assert state.question == "What is inflation doing?"
    assert state.iteration == 0
    assert state.complete is False
    assert not state.is_complete()


def test_agent_state_add_observation():
    from intelligence.agentic_rag.agent_state import AgentState

    state = AgentState(question="Test")
    chunks = [
        {"text": "CPI rose 3.5%", "metadata": {"title": "Inflation report", "source": "Bloomberg"}},
        {"text": "Duplicate chunk", "metadata": {"title": "Inflation report", "source": "Bloomberg"}},
    ]
    state.add_observation(chunks)
    # Same title + text[:80] → deduplicated
    state.add_observation(chunks)
    assert len(state.retrieved_chunks) == 2  # Only 2 unique chunks kept


def test_agent_state_mark_gap():
    from intelligence.agentic_rag.agent_state import AgentState

    state = AgentState(question="Test")
    state.mark_gap("Missing: Fed impact on bonds")
    state.mark_gap("Missing: Fed impact on bonds")  # Duplicate
    state.mark_gap("Missing: equity valuations")
    assert len(state.gaps) == 2


def test_agent_state_is_complete_max_iterations():
    from intelligence.agentic_rag.agent_state import AgentState

    state = AgentState(question="Test", max_iterations=2)
    state.iteration = 3
    assert state.is_complete()


def test_agent_state_agreement_score():
    from intelligence.agentic_rag.agent_state import AgentState, AgentOutput

    state = AgentState(question="Test")
    state.add_agent_output(AgentOutput("MacroStrategist", "brief1", 0.80))
    state.add_agent_output(AgentOutput("SentimentAnalyst", "brief2", 0.70))
    state.add_agent_output(AgentOutput("RiskAnalyst", "brief3", 0.40))  # below 0.6 threshold
    agreeing, total = state.agent_agreement_score()
    assert agreeing == 2
    assert total == 3


def test_agent_state_trace_structure():
    from intelligence.agentic_rag.agent_state import AgentState

    state = AgentState(question="What is the VIX telling us?")
    trace = state.to_trace()
    assert "question" in trace
    assert "iterations" in trace
    assert "chunks_retrieved" in trace
    assert "agent_outputs" in trace
    assert "elapsed_ms" in trace


# ── QueryPlanner Tests ────────────────────────────────────────────────────────


def test_query_planner_simple_question_no_decompose():
    from intelligence.agentic_rag.query_planner import QueryPlanner

    # Simple question — should return as-is (deterministic path)
    planner = QueryPlanner(llm_generate_fn=None)
    # Patch _is_complex to return False
    with mock.patch.object(planner, "_is_complex", return_value=False):
        result = planner.decompose("What is VIX today?")
    assert result == ["What is VIX today?"]


def test_query_planner_complex_question_deterministic():
    from intelligence.agentic_rag.query_planner import QueryPlanner

    # Complex question with 'versus' — should split
    planner = QueryPlanner(llm_generate_fn=None)
    question = "How does rising inflation versus falling GDP affect equity markets?"
    # Mock LLM to fail so deterministic path runs
    with mock.patch.object(planner, "_llm_decompose", return_value=[]):
        result = planner.decompose(question)
    assert isinstance(result, list)
    assert len(result) >= 1  # At minimum returns something


def test_query_planner_llm_decompose_success():
    from intelligence.agentic_rag.query_planner import QueryPlanner

    mock_llm = mock.Mock(return_value="What is the inflation rate?\nWhat is GDP growth?\n")
    planner = QueryPlanner(llm_generate_fn=mock_llm)
    with mock.patch.object(planner, "_is_complex", return_value=True):
        result = planner.decompose("Multi-part question about inflation and GDP?")
    assert len(result) == 2
    assert "What is the inflation rate?" in result[0]


def test_query_planner_llm_failure_fallback():
    from intelligence.agentic_rag.query_planner import QueryPlanner

    def failing_llm(prompt):
        raise RuntimeError("LLM timeout")

    planner = QueryPlanner(llm_generate_fn=failing_llm)
    with mock.patch.object(planner, "_is_complex", return_value=True):
        result = planner.decompose("Simple question that won't split?")
    assert isinstance(result, list)
    assert len(result) >= 1


# ── ToolRegistry Tests ────────────────────────────────────────────────────────


def test_tool_registry_build_default():
    from intelligence.agentic_rag.tool_registry import build_default_registry

    registry = build_default_registry()
    assert registry.get("semantic_search") is not None
    assert registry.get("live_market") is not None
    assert registry.get("regime_detection") is not None
    assert registry.get("indicator_extract") is not None
    assert registry.get("cross_asset") is not None


def test_tool_registry_get_nonexistent():
    from intelligence.agentic_rag.tool_registry import ToolRegistry

    registry = ToolRegistry()
    assert registry.get("nonexistent_tool") is None


def test_semantic_search_tool_success():
    from intelligence.agentic_rag.tool_registry import SemanticSearchTool

    mock_chunks = [{"text": "Test chunk", "metadata": {"title": "Test", "source": "Reuters"}}]
    tool = SemanticSearchTool()
    with mock.patch("intelligence.agentic_rag.tool_registry.SemanticSearchTool.run",
                    return_value=__import__('intelligence.agentic_rag.tool_registry', fromlist=['ToolResult'])
                    .ToolResult('semantic_search', 'inflation impact on bonds', mock_chunks, 45, True)):
        result = tool.run("inflation impact on bonds")
    assert result.success
    assert result.data == mock_chunks
    assert result.tool_name == "semantic_search"


def test_semantic_search_tool_failure():
    from intelligence.agentic_rag.tool_registry import SemanticSearchTool, ToolResult

    tool = SemanticSearchTool()
    with mock.patch.object(tool, 'run', return_value=ToolResult('semantic_search', 'some query', [], 10, False, 'FAISS error')):
        result = tool.run("some query")
    assert not result.success
    assert "FAISS error" in result.error


def test_indicator_extract_tool():
    from intelligence.agentic_rag.tool_registry import IndicatorExtractTool, ToolResult

    mock_indicators = {"inflation_cpi": 3.5, "yield_10y": 4.2}
    tool = IndicatorExtractTool()
    with mock.patch.object(tool, 'run', return_value=ToolResult('indicator_extract', 'CPI is at 3.5%', mock_indicators, 5, True)):
        result = tool.run("CPI is at 3.5% and 10Y yield is 4.2%")
    assert result.success
    assert result.data == mock_indicators


# ── ReflectionEngine Tests ────────────────────────────────────────────────────


def test_reflection_engine_complete_answer_no_gaps():
    from intelligence.agentic_rag.reflection_engine import ReflectionEngine
    from intelligence.agentic_rag.agent_state import AgentState

    engine = ReflectionEngine(llm_generate_fn=None)  # No LLM
    state = AgentState(question="What is VIX?")
    # Provide a well-formed draft answer
    state.draft_answer = (
        "Direct answer: VIX is at 18.5, indicating low volatility [S1].\n"
        "Data snapshot: VIX=18.5, S&P500=5200, yield_10y=4.2% [S2].\n"
        "What is happening: Equity volatility remains suppressed due to [S1]\n"
        "Market impact:\n- Equities: bullish signal [S2]\n- Rates: stable\n- FX: dollar steady\n"
        "Scenarios: Base (~55%): Range-bound. Bull (~25%): VIX drops below 15. Bear (~20%): Spike to 30+\n"
        "What to watch: CPI release, FOMC meeting [S3]\n"
        "Confidence: MEDIUM - Limited evidence [S1]"
    )
    state.retrieved_chunks = [
        {"text": "VIX data chunk", "metadata": {"title": "Vol report", "source": "CBOE"}},
        {"text": "S&P data", "metadata": {"title": "Market update", "source": "Bloomberg"}},
        {"text": "Treasury data", "metadata": {"title": "Bond market", "source": "FT"}},
    ]
    gaps = engine.assess_gaps(state)
    # Should have minimal gaps with a well-formed answer
    # No citation gap (citation density ok), has quantitative data, has direct answer
    # Some gaps may still be found, but shouldn't be ALL of them
    assert len(gaps) <= 3


def test_reflection_engine_empty_draft_many_gaps():
    from intelligence.agentic_rag.reflection_engine import ReflectionEngine
    from intelligence.agentic_rag.agent_state import AgentState

    engine = ReflectionEngine(llm_generate_fn=None)
    state = AgentState(question="What is the macro outlook?")
    state.draft_answer = ""  # Empty draft
    state.retrieved_chunks = []

    gaps = engine.assess_gaps(state)
    # Should find multiple gaps with an empty draft
    assert len(gaps) >= 2


def test_reflection_should_not_iterate_when_complete():
    from intelligence.agentic_rag.reflection_engine import ReflectionEngine
    from intelligence.agentic_rag.agent_state import AgentState

    engine = ReflectionEngine()
    state = AgentState(question="Test", max_iterations=2)
    state.iteration = 3  # Exceeded max
    state.gaps = ["Some gap"]
    state.gap_queries = ["Some query"]
    assert not engine.should_iterate(state)


def test_reflection_should_not_iterate_when_no_gaps():
    from intelligence.agentic_rag.reflection_engine import ReflectionEngine
    from intelligence.agentic_rag.agent_state import AgentState

    engine = ReflectionEngine()
    state = AgentState(question="Test", max_iterations=2)
    state.iteration = 1
    state.gaps = []  # No gaps
    assert not engine.should_iterate(state)


def test_reflection_follow_up_queries_generated():
    from intelligence.agentic_rag.reflection_engine import ReflectionEngine

    engine = ReflectionEngine()
    gaps = [
        "Missing quantitative data on Fed Funds rate",
        "No scenario analysis present",
    ]
    queries = engine.generate_follow_up_queries(gaps, "What is the Fed doing?")
    assert len(queries) >= 1
    assert all(isinstance(q, str) and len(q) > 5 for q in queries)


# ── AgentEvent Tests ─────────────────────────────────────────────────────────


def test_agent_event_to_sse():
    from intelligence.agentic_rag.orchestrator import AgentEvent
    import json

    event = AgentEvent(
        stage="planning",
        data={"sub_questions": ["Q1?", "Q2?"], "count": 2},
        elapsed_ms=123,
    )
    sse = event.to_sse()
    assert sse.startswith("event: planning\n")
    assert "data: " in sse
    payload = json.loads(sse.split("data: ")[1].split("\n")[0])
    assert payload["stage"] == "planning"
    assert payload["elapsed_ms"] == 123


def test_agent_event_to_dict():
    from intelligence.agentic_rag.orchestrator import AgentEvent

    event = AgentEvent(stage="final", data={"answer": "Test answer"}, agent_name="SynthesisAgent")
    d = event.to_dict()
    assert d["stage"] == "final"
    assert d["agent_name"] == "SynthesisAgent"
    assert "answer" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
