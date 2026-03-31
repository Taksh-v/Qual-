from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from threading import Thread
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from mcp_integration.config import load_settings


@dataclass(frozen=True)
class ExternalServerConfig:
    name: str
    transport: str
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None


def load_external_server_configs() -> dict[str, ExternalServerConfig]:
    raw = os.getenv("MCP_CLIENT_SERVERS_JSON", "{}")
    parsed = json.loads(raw) if raw.strip() else {}
    if not isinstance(parsed, dict):
        return {}

    configs: dict[str, ExternalServerConfig] = {}
    for name, value in parsed.items():
        if not isinstance(value, dict):
            continue
        transport = str(value.get("transport", "")).strip().lower()
        if transport not in {"stdio", "streamable-http"}:
            continue
        cfg = ExternalServerConfig(
            name=name,
            transport=transport,
            url=(str(value.get("url")).strip() if value.get("url") else None),
            command=(str(value.get("command")).strip() if value.get("command") else None),
            args=tuple(str(arg) for arg in (value.get("args") or [])),
            env=(value.get("env") if isinstance(value.get("env"), dict) else None),
        )
        configs[name] = cfg
    return configs


class ExternalMCPClient:
    def __init__(self) -> None:
        self._settings = load_settings()
        self._configs = load_external_server_configs()

    def configured_servers(self) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        for cfg in self._configs.values():
            output.append({
                "name": cfg.name,
                "transport": cfg.transport,
                "url": cfg.url or "",
                "command": cfg.command or "",
            })
        return output

    async def call_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        cfg = self._configs.get(server_name)
        if cfg is None:
            raise ValueError(f"unknown external server '{server_name}'")

        timeout = timeout_sec if timeout_sec is not None else self._settings.client_timeout_sec
        args = arguments or {}

        if cfg.transport == "stdio":
            return await asyncio.wait_for(self._call_stdio(cfg, tool_name, args), timeout=timeout)
        return await asyncio.wait_for(self._call_streamable_http(cfg, tool_name, args), timeout=timeout)

    def call_tool_sync(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        async def _runner() -> dict[str, Any]:
            return await self.call_tool(
                server_name=server_name,
                tool_name=tool_name,
                arguments=arguments,
                timeout_sec=timeout_sec,
            )

        try:
            asyncio.get_running_loop()
            in_running_loop = True
        except RuntimeError:
            in_running_loop = False

        if not in_running_loop:
            return asyncio.run(_runner())

        result: dict[str, Any] = {}
        error: Exception | None = None

        def _in_thread() -> None:
            nonlocal result, error
            try:
                result = asyncio.run(_runner())
            except Exception as exc:
                error = exc

        max_wait = max(1.0, float(timeout_sec if timeout_sec is not None else self._settings.client_timeout_sec) + 1.0)
        thread = Thread(target=_in_thread, daemon=True)
        thread.start()
        thread.join(timeout=max_wait)

        if thread.is_alive():
            raise TimeoutError("external MCP sync call timed out")
        if error is not None:
            raise error
        return result

    async def _call_stdio(
        self,
        cfg: ExternalServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if not cfg.command:
            raise ValueError(f"stdio server '{cfg.name}' missing command")
        server_params = StdioServerParameters(
            command=cfg.command,
            args=list(cfg.args),
            env=cfg.env,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return {
                    "is_error": bool(getattr(result, "isError", False)),
                    "structured": getattr(result, "structuredContent", None),
                    "content": [content.model_dump() for content in (result.content or [])],
                }

    async def _call_streamable_http(
        self,
        cfg: ExternalServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if not cfg.url:
            raise ValueError(f"streamable-http server '{cfg.name}' missing url")
        async with streamable_http_client(cfg.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return {
                    "is_error": bool(getattr(result, "isError", False)),
                    "structured": getattr(result, "structuredContent", None),
                    "content": [content.model_dump() for content in (result.content or [])],
                }
