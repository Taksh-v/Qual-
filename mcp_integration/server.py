from __future__ import annotations

import logging
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_integration.audit import log_mcp_event
from mcp_integration.client import ExternalMCPClient
from mcp_integration.config import load_settings
from mcp_integration.guardrails import (
    AuthManager,
    ConfirmationStore,
    IdempotencyStore,
    MCPAuthError,
    MCPValidationError,
    OperationLocks,
)
from mcp_integration.tools import (
    execute_admin_action,
    list_admin_actions,
    market_status,
    metadata_query,
    metadata_sentiment_summary,
    rag_query,
    system_health,
)

logger = logging.getLogger(__name__)

_settings = load_settings()
_auth = AuthManager(set(_settings.read_keys), set(_settings.admin_keys))
_confirmations = ConfirmationStore(ttl_sec=_settings.confirmation_ttl_sec)
_idempotency = IdempotencyStore(ttl_sec=_settings.idempotency_ttl_sec)
_op_locks = OperationLocks()
_external_client = ExternalMCPClient()

mcp = FastMCP(
    "Qual MCP",
    instructions=(
        "MCP server for Qual financial intelligence system. "
        "Use read tools for query and diagnostics. "
        "Use prepare/execute flow for admin actions."
    ),
    stateless_http=True,
    json_response=True,
)


def _actor(api_key: str | None, scope: str) -> str:
    principal = _auth.authenticate(api_key, scope)
    return principal.actor_id


def _validation_error(action: str, actor: str, exc: Exception) -> dict[str, Any]:
    payload = {
        "ok": False,
        "error": str(exc),
        "error_type": exc.__class__.__name__,
    }
    log_mcp_event(event_type="tool", action=action, actor=actor, status="error", detail=str(exc), result=payload)
    return payload


@mcp.tool()
async def rag_query_tool(question: str, api_key: str | None = None) -> dict[str, Any]:
    action = "rag_query"
    actor = "unknown"
    started = time.time()
    try:
        actor = _actor(api_key, "read")
        result = await rag_query(question)
        elapsed_ms = int((time.time() - started) * 1000)
        payload = {
            "ok": True,
            "elapsed_ms": elapsed_ms,
            "result": result,
        }
        log_mcp_event(event_type="tool", action=action, actor=actor, status="ok", result={"elapsed_ms": elapsed_ms})
        return payload
    except Exception as exc:
        return _validation_error(action, actor, exc)


@mcp.tool()
def system_health_tool(api_key: str | None = None) -> dict[str, Any]:
    action = "system_health"
    actor = "unknown"
    try:
        actor = _actor(api_key, "read")
        result = system_health()
        log_mcp_event(event_type="tool", action=action, actor=actor, status="ok")
        return {"ok": True, "result": result}
    except Exception as exc:
        return _validation_error(action, actor, exc)


@mcp.tool()
def market_status_tool(api_key: str | None = None) -> dict[str, Any]:
    action = "market_status"
    actor = "unknown"
    try:
        actor = _actor(api_key, "read")
        result = market_status()
        log_mcp_event(event_type="tool", action=action, actor=actor, status="ok")
        return {"ok": True, "result": result}
    except Exception as exc:
        return _validation_error(action, actor, exc)


@mcp.tool()
def metadata_query_tool(
    api_key: str | None = None,
    sector: str | None = None,
    region: str | None = None,
    data_type: str | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
    source: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    action = "metadata_query"
    actor = "unknown"
    try:
        actor = _actor(api_key, "read")
        result = metadata_query(
            sector=sector,
            region=region,
            data_type=data_type,
            min_date=min_date,
            max_date=max_date,
            source=source,
            limit=limit,
        )
        log_mcp_event(event_type="tool", action=action, actor=actor, status="ok", result={"count": result.get("count", 0)})
        return {"ok": True, "result": result}
    except Exception as exc:
        return _validation_error(action, actor, exc)


@mcp.tool()
def sentiment_summary_tool(
    api_key: str | None = None,
    sector: str | None = None,
    min_date: str | None = None,
) -> dict[str, Any]:
    action = "sentiment_summary"
    actor = "unknown"
    try:
        actor = _actor(api_key, "read")
        result = metadata_sentiment_summary(sector=sector, min_date=min_date)
        log_mcp_event(event_type="tool", action=action, actor=actor, status="ok")
        return {"ok": True, "result": result}
    except Exception as exc:
        return _validation_error(action, actor, exc)


@mcp.tool()
def list_admin_actions_tool(api_key: str | None = None) -> dict[str, Any]:
    action = "list_admin_actions"
    actor = "unknown"
    try:
        actor = _actor(api_key, "read")
        result = list_admin_actions()
        log_mcp_event(event_type="tool", action=action, actor=actor, status="ok")
        return {"ok": True, "result": result}
    except Exception as exc:
        return _validation_error(action, actor, exc)


@mcp.tool()
def list_external_mcp_servers_tool(api_key: str | None = None) -> dict[str, Any]:
    action = "list_external_mcp_servers"
    actor = "unknown"
    try:
        actor = _actor(api_key, "read")
        result = _external_client.configured_servers()
        log_mcp_event(event_type="tool", action=action, actor=actor, status="ok", result={"count": len(result)})
        return {"ok": True, "result": result}
    except Exception as exc:
        return _validation_error(action, actor, exc)


@mcp.tool()
async def call_external_mcp_tool(
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    timeout_sec: float | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    action = "call_external_mcp_tool"
    actor = "unknown"
    started = time.time()
    try:
        actor = _actor(api_key, "read")
        result = await _external_client.call_tool(
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments,
            timeout_sec=timeout_sec,
        )
        elapsed_ms = int((time.time() - started) * 1000)
        log_mcp_event(
            event_type="tool",
            action=action,
            actor=actor,
            status="ok",
            payload={"server_name": server_name, "tool_name": tool_name},
            result={"elapsed_ms": elapsed_ms, "is_error": result.get("is_error", False)},
        )
        return {
            "ok": True,
            "elapsed_ms": elapsed_ms,
            "result": result,
        }
    except Exception as exc:
        return _validation_error(action, actor, exc)


@mcp.tool()
def prepare_admin_action_tool(
    action: str,
    params: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    event_action = "prepare_admin_action"
    actor = "unknown"
    try:
        actor = _actor(api_key, "admin")
        action_name = (action or "").strip().lower()
        if not action_name:
            raise MCPValidationError("missing action")
        prepared = _confirmations.create(
            action=action_name,
            params=params or {},
            actor_id=actor,
        )
        payload = {
            "ok": True,
            "confirmation_token": prepared.token,
            "action": prepared.action,
            "expires_at_unix": int(prepared.expires_at),
        }
        log_mcp_event(
            event_type="admin_prepare",
            action=action_name,
            actor=actor,
            status="ok",
            payload={"params": params or {}},
            result={"expires_at_unix": int(prepared.expires_at)},
        )
        return payload
    except Exception as exc:
        return _validation_error(event_action, actor, exc)


@mcp.tool()
def execute_admin_action_tool(
    action: str,
    confirmation_token: str,
    idempotency_key: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    event_action = "execute_admin_action"
    actor = "unknown"
    action_name = (action or "").strip().lower()
    try:
        actor = _actor(api_key, "admin")
        if not action_name:
            raise MCPValidationError("missing action")
        if not idempotency_key or len(idempotency_key.strip()) < 8:
            raise MCPValidationError("idempotency_key must be at least 8 chars")

        replay = _idempotency.get(
            idempotency_key=idempotency_key.strip(),
            action=action_name,
            actor_id=actor,
        )
        if replay is not None:
            replay_payload = {"ok": True, "idempotent_replay": True, "result": replay}
            log_mcp_event(
                event_type="admin_execute",
                action=action_name,
                actor=actor,
                status="replay",
                result={"idempotent_replay": True},
            )
            return replay_payload

        pending = _confirmations.consume(
            token=confirmation_token,
            action=action_name,
            actor_id=actor,
        )

        with _op_locks.held(action_name):
            result = execute_admin_action(action_name, dict(pending.params))

        _idempotency.put(
            idempotency_key=idempotency_key.strip(),
            action=action_name,
            actor_id=actor,
            result=result,
        )

        payload = {"ok": True, "idempotent_replay": False, "result": result}
        log_mcp_event(
            event_type="admin_execute",
            action=action_name,
            actor=actor,
            status="ok",
            payload={"params": pending.params},
            result={"ok": bool(result.get("ok", True)) if isinstance(result, dict) else True},
        )
        return payload
    except (MCPAuthError, MCPValidationError) as exc:
        return _validation_error(event_action, actor, exc)
    except Exception as exc:
        logger.exception("[mcp] admin execution failed")
        return _validation_error(event_action, actor, exc)
