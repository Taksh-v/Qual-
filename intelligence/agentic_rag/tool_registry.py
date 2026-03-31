"""
intelligence/agentic_rag/tool_registry.py
------------------------------------------
Tool abstraction layer for agentic RAG.

Each tool is a typed, callable unit that agents can invoke to gather
specific information. The ToolRegistry manages tool discovery and dispatch.

Available tools:
    - SemanticSearchTool: FAISS vector search for news/research chunks
    - BM25SearchTool: Keyword-based retrieval (sparse complement to FAISS)
    - LiveMarketTool: Fetches current market indicators (yfinance + FRED)
    - RegimeTool: Runs regime detection on current indicators
    - IndicatorExtractTool: Extracts indicator signals from text
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class AgentTool(ABC):
    """Abstract base class for all agent-callable tools."""

    name: str = "base_tool"
    description: str = ""

    @abstractmethod
    def run(self, query: str, **kwargs: Any) -> "ToolResult":
        """Execute the tool and return a structured result."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"


# ── Import here to avoid circular issues ─────────────────────────────────────

class ToolResult:
    """Structured output from a tool invocation."""

    def __init__(
        self,
        tool_name: str,
        query: str,
        data: Any,
        elapsed_ms: int,
        success: bool = True,
        error: str = "",
    ) -> None:
        self.tool_name = tool_name
        self.query = query
        self.data = data
        self.elapsed_ms = elapsed_ms
        self.success = success
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "query": self.query,
            "success": self.success,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "data_summary": str(self.data)[:200] if self.data else None,
        }


# ── Concrete Tools ─────────────────────────────────────────────────────────────


class SemanticSearchTool(AgentTool):
    """
    Retrieves semantically relevant chunks from the FAISS vector index.
    Uses the existing context_retriever which handles embedding + ranking.
    """

    name = "semantic_search"
    description = (
        "Search the news/research vector database for chunks relevant to a query. "
        "Returns ranked text chunks with source metadata and citations."
    )

    def __init__(self, top_k: int = 10, keep_latest: int = 8) -> None:
        self.top_k = top_k
        self.keep_latest = keep_latest

    def run(self, query: str, **kwargs: Any) -> ToolResult:
        t0 = time.time()
        try:
            from intelligence.context_retriever import retrieve_relevant_context

            chunks = retrieve_relevant_context(
                query,
                top_k=kwargs.get("top_k", self.top_k),
                keep_latest=kwargs.get("keep_latest", self.keep_latest),
            )
            elapsed = int((time.time() - t0) * 1000)
            logger.debug("[SemanticSearchTool] Retrieved %d chunks in %dms", len(chunks), elapsed)
            return ToolResult(
                tool_name=self.name,
                query=query,
                data=chunks,
                elapsed_ms=elapsed,
                success=True,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.warning("[SemanticSearchTool] Failed: %s", exc)
            return ToolResult(
                tool_name=self.name,
                query=query,
                data=[],
                elapsed_ms=elapsed,
                success=False,
                error=str(exc)[:200],
            )


class LiveMarketTool(AgentTool):
    """
    Fetches live market indicators from yfinance (prices) and FRED (macro).
    Returns a dict of {indicator_key: float_value}.
    """

    name = "live_market"
    description = (
        "Fetch current live market data: equity indices (S&P500, Nasdaq, VIX), "
        "FX rates (DXY), Treasury yields (2Y, 10Y), commodities (WTI, Gold), "
        "and FRED macro indicators (CPI, PCE, unemployment, Fed Funds rate)."
    )

    def run(self, query: str = "", **kwargs: Any) -> ToolResult:
        t0 = time.time()
        try:
            from intelligence.live_market_data import fetch_live_indicators

            live, meta = fetch_live_indicators()
            elapsed = int((time.time() - t0) * 1000)
            logger.debug("[LiveMarketTool] Fetched %d indicators in %dms", len(live), elapsed)
            return ToolResult(
                tool_name=self.name,
                query=query or "live_market_snapshot",
                data={"indicators": live, "meta": meta},
                elapsed_ms=elapsed,
                success=True,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.warning("[LiveMarketTool] Failed: %s", exc)
            return ToolResult(
                tool_name=self.name,
                query=query or "live_market_snapshot",
                data={"indicators": {}, "meta": {}},
                elapsed_ms=elapsed,
                success=False,
                error=str(exc)[:200],
            )


class RegimeTool(AgentTool):
    """
    Runs macro regime detection on current indicators.
    Returns regime label (e.g. RISK_ON, STAGFLATION, RECESSION) with confidence.
    """

    name = "regime_detection"
    description = (
        "Detect the current macro regime (e.g. RISK_ON, RISK_OFF, RECESSION, "
        "STAGFLATION, RECOVERY) based on live indicator values. "
        "Returns regime label and confidence level."
    )

    def run(self, query: str = "", **kwargs: Any) -> ToolResult:
        t0 = time.time()
        indicators: dict[str, float] = kwargs.get("indicators", {})
        try:
            from intelligence.regime_detector import detect_regime
            from intelligence.indicator_parser import get_regime_inputs_from_indicators

            regime_inputs = get_regime_inputs_from_indicators(indicators)
            regime = detect_regime(**regime_inputs)
            elapsed = int((time.time() - t0) * 1000)
            return ToolResult(
                tool_name=self.name,
                query=query or "regime_detection",
                data=regime,
                elapsed_ms=elapsed,
                success=True,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            logger.warning("[RegimeTool] Failed: %s", exc)
            return ToolResult(
                tool_name=self.name,
                query=query or "regime_detection",
                data={"regime": "UNKNOWN", "confidence": "LOW"},
                elapsed_ms=elapsed,
                success=False,
                error=str(exc)[:200],
            )


class IndicatorExtractTool(AgentTool):
    """
    Extracts numeric indicator values from free-form text.
    Useful for parsing LLM outputs or retrieved chunks to get structured values.
    """

    name = "indicator_extract"
    description = (
        "Extract named macroeconomic indicator values from text. "
        "E.g., 'CPI at 3.5%' → {inflation_cpi: 3.5}. "
        "Useful for parsing retrieved chunks or question context."
    )

    def run(self, query: str, **kwargs: Any) -> ToolResult:
        t0 = time.time()
        try:
            from intelligence.indicator_parser import extract_indicators_from_text

            indicators = extract_indicators_from_text(query)
            elapsed = int((time.time() - t0) * 1000)
            return ToolResult(
                tool_name=self.name,
                query=query[:100],
                data=indicators,
                elapsed_ms=elapsed,
                success=True,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            return ToolResult(
                tool_name=self.name,
                query=query[:100],
                data={},
                elapsed_ms=elapsed,
                success=False,
                error=str(exc)[:200],
            )


class CrossAssetTool(AgentTool):
    """
    Runs multi-asset correlation and signal analysis.
    Returns equity/rates/FX/commodity signals and divergences.
    """

    name = "cross_asset"
    description = (
        "Analyze cross-asset relationships and market signals: equities vs bonds, "
        "dollar vs commodities, credit spreads vs equity vol. Returns directional "
        "signals and notable divergences between asset classes."
    )

    def run(self, query: str = "", **kwargs: Any) -> ToolResult:
        t0 = time.time()
        indicators: dict[str, float] = kwargs.get("indicators", {})
        try:
            from intelligence.cross_asset_analyzer import analyze_cross_asset

            cross_asset = analyze_cross_asset(indicators)
            elapsed = int((time.time() - t0) * 1000)
            return ToolResult(
                tool_name=self.name,
                query=query or "cross_asset_analysis",
                data=cross_asset,
                elapsed_ms=elapsed,
                success=True,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            return ToolResult(
                tool_name=self.name,
                query=query or "cross_asset_analysis",
                data={"overall_signal": "MIXED", "divergences": []},
                elapsed_ms=elapsed,
                success=False,
                error=str(exc)[:200],
            )


# ── Registry ───────────────────────────────────────────────────────────────────


class ToolRegistry:
    """
    Registry that maps tool names to AgentTool instances.
    Agents use this to discover and invoke tools by name.
    """

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[AgentTool]:
        return list(self._tools.values())

    def tool_descriptions(self) -> str:
        """Human-readable summary of all registered tools."""
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}: {t.description[:120]}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={list(self._tools.keys())}>"


def build_default_registry() -> ToolRegistry:
    """Build and return the default tool registry with all standard tools."""
    registry = ToolRegistry()
    registry.register(SemanticSearchTool())
    registry.register(LiveMarketTool())
    registry.register(RegimeTool())
    registry.register(IndicatorExtractTool())
    registry.register(CrossAssetTool())
    return registry
