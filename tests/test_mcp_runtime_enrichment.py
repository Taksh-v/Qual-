import asyncio
import json
import sys
from pathlib import Path

from mcp_integration.client import ExternalMCPClient
from mcp_integration.runtime_enrichment import (
    fetch_external_context_async,
    fetch_external_context_sync,
    summarize_external_result,
)


def _mock_stdio_server_config() -> str:
    script = Path(__file__).resolve().parent / "fixtures" / "mock_external_mcp_server.py"
    return json.dumps(
        {
            "mock-stdio": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(script)],
            }
        }
    )


def test_summarize_external_result_merges_structured_and_content() -> None:
    payload = {
        "is_error": False,
        "structured": {"summary": "Macro signal: disinflation bias"},
        "content": [{"type": "text", "text": "Fed pause odds rising"}],
    }

    text = summarize_external_result(payload, max_chars=200)
    assert "Macro signal: disinflation bias" in text
    assert "Fed pause odds rising" in text


def test_summarize_external_result_ignores_error_payload() -> None:
    payload = {
        "is_error": True,
        "structured": {"summary": "should not be used"},
    }

    assert summarize_external_result(payload) == ""


def test_call_tool_sync_without_running_loop(monkeypatch) -> None:
    client = ExternalMCPClient()

    async def fake_call_tool(self, *, server_name, tool_name, arguments=None, timeout_sec=None):
        return {"is_error": False, "structured": {"text": "ok"}, "content": []}

    monkeypatch.setattr(ExternalMCPClient, "call_tool", fake_call_tool)

    result = client.call_tool_sync(server_name="srv", tool_name="tool")
    assert result["structured"]["text"] == "ok"


def test_call_tool_sync_with_running_loop(monkeypatch) -> None:
    client = ExternalMCPClient()

    async def fake_call_tool(self, *, server_name, tool_name, arguments=None, timeout_sec=None):
        await asyncio.sleep(0)
        return {"is_error": False, "structured": {"text": "ok-loop"}, "content": []}

    monkeypatch.setattr(ExternalMCPClient, "call_tool", fake_call_tool)

    async def _run() -> dict:
        return client.call_tool_sync(server_name="srv", tool_name="tool", timeout_sec=2)

    result = asyncio.run(_run())
    assert result["structured"]["text"] == "ok-loop"


def test_fetch_external_context_sync_uses_summarizer(monkeypatch) -> None:
    def fake_call_tool_sync(self, *, server_name, tool_name, arguments=None, timeout_sec=None):
        return {
            "is_error": False,
            "structured": {"summary": "US rates pressure easing"},
            "content": [],
        }

    monkeypatch.setattr(ExternalMCPClient, "call_tool_sync", fake_call_tool_sync)

    payload = fetch_external_context_sync(
        question="What changed in rates?",
        server_name="srv",
        tool_name="tool",
        timeout_sec=2,
        max_chars=200,
    )

    assert payload["ok"] is True
    assert "US rates pressure easing" in payload["text"]


def test_fetch_external_context_sync_stdio_roundtrip(monkeypatch) -> None:
    monkeypatch.setenv("MCP_CLIENT_SERVERS_JSON", _mock_stdio_server_config())

    payload = fetch_external_context_sync(
        question="How is inflation trending?",
        server_name="mock-stdio",
        tool_name="external_macro_context",
        timeout_sec=8,
        max_chars=240,
    )

    assert payload["ok"] is True
    assert "External macro context" in payload["text"]
    assert "inflation trending" in payload["text"]


def test_fetch_external_context_async_stdio_roundtrip(monkeypatch) -> None:
    monkeypatch.setenv("MCP_CLIENT_SERVERS_JSON", _mock_stdio_server_config())

    async def _run() -> dict:
        return await fetch_external_context_async(
            question="What is the near-term rates setup?",
            server_name="mock-stdio",
            tool_name="external_macro_context",
            timeout_sec=8,
            max_chars=240,
        )

    payload = asyncio.run(_run())
    assert payload["ok"] is True
    assert "near-term rates setup" in payload["text"]
