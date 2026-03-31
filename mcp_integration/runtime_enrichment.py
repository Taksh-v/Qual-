from __future__ import annotations

import json
import logging
from typing import Any

from mcp_integration.client import ExternalMCPClient

logger = logging.getLogger(__name__)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value[:8]:
            text = _normalize_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        preferred_keys = (
            "text",
            "summary",
            "message",
            "answer",
            "analysis",
            "result",
            "content",
            "value",
        )
        for key in preferred_keys:
            if key in value:
                text = _normalize_text(value.get(key))
                if text:
                    return text
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def summarize_external_result(result: dict[str, Any], max_chars: int = 900) -> str:
    if not isinstance(result, dict):
        return ""
    if result.get("is_error"):
        return ""

    structured_text = _normalize_text(result.get("structured"))
    content_text = _normalize_text(result.get("content"))

    merged: list[str] = []
    if structured_text:
        merged.append(structured_text)
    if content_text and content_text != structured_text:
        merged.append(content_text)

    text = "\n".join(part.strip() for part in merged if part and part.strip())
    text = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    if max_chars <= 0:
        return text
    return text[:max_chars].strip()


async def fetch_external_context_async(
    *,
    question: str,
    server_name: str,
    tool_name: str,
    timeout_sec: float,
    max_chars: int = 900,
) -> dict[str, Any]:
    client = ExternalMCPClient()
    result = await client.call_tool(
        server_name=server_name,
        tool_name=tool_name,
        arguments={"question": question},
        timeout_sec=timeout_sec,
    )
    text = summarize_external_result(result, max_chars=max_chars)
    return {
        "ok": bool(text),
        "text": text,
        "is_error": bool(result.get("is_error", False)),
    }


def fetch_external_context_sync(
    *,
    question: str,
    server_name: str,
    tool_name: str,
    timeout_sec: float,
    max_chars: int = 900,
) -> dict[str, Any]:
    client = ExternalMCPClient()
    result = client.call_tool_sync(
        server_name=server_name,
        tool_name=tool_name,
        arguments={"question": question},
        timeout_sec=timeout_sec,
    )
    text = summarize_external_result(result, max_chars=max_chars)
    return {
        "ok": bool(text),
        "text": text,
        "is_error": bool(result.get("is_error", False)),
    }
