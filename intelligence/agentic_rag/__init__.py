"""
intelligence/agentic_rag/__init__.py
------------------------------------
Agentic RAG package — Bloomberg Terminal-grade multi-agent financial intelligence.

Core loop: Plan → Act → Observe → Reflect → Synthesize

Components:
    - agent_state: Working memory for an agentic session
    - query_planner: Decomposes complex queries into focused sub-questions
    - tool_registry: Typed tool abstractions (retrieval, live data, indicators)
    - reflection_engine: Self-critique, gap detection, re-query planning
    - orchestrator: Main orchestration loop, coordinates all agents
"""

from .agent_state import AgentState, AgentOutput, ToolCall, ToolResult
from .query_planner import QueryPlanner
from .tool_registry import ToolRegistry, AgentTool
from .reflection_engine import ReflectionEngine
from .orchestrator import AgenticOrchestrator, AgentEvent

__all__ = [
    "AgentState",
    "AgentOutput",
    "ToolCall",
    "ToolResult",
    "QueryPlanner",
    "ToolRegistry",
    "AgentTool",
    "ReflectionEngine",
    "AgenticOrchestrator",
    "AgentEvent",
]
