from __future__ import annotations

import os
from dataclasses import dataclass


def _split_csv(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class MCPSettings:
    read_keys: frozenset[str]
    admin_keys: frozenset[str]
    confirmation_ttl_sec: int
    idempotency_ttl_sec: int
    require_origin_check: bool
    allowed_origins: frozenset[str]
    audit_log_path: str
    host: str
    port: int
    client_timeout_sec: float


def load_settings() -> MCPSettings:
    fallback_keys = _split_csv(os.getenv("API_KEYS", ""))
    read_keys = _split_csv(os.getenv("MCP_READ_KEYS", "")) or set(fallback_keys)
    admin_keys = _split_csv(os.getenv("MCP_ADMIN_KEYS", "")) or set(fallback_keys)

    return MCPSettings(
        read_keys=frozenset(read_keys),
        admin_keys=frozenset(admin_keys),
        confirmation_ttl_sec=max(15, int(os.getenv("MCP_CONFIRMATION_TTL_SEC", "180"))),
        idempotency_ttl_sec=max(30, int(os.getenv("MCP_IDEMPOTENCY_TTL_SEC", "3600"))),
        require_origin_check=_env_bool("MCP_REQUIRE_ORIGIN_CHECK", False),
        allowed_origins=frozenset(_split_csv(os.getenv("MCP_ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1"))),
        audit_log_path=os.getenv("MCP_AUDIT_LOG_PATH", "data/mcp_audit_log.jsonl"),
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "8010")),
        client_timeout_sec=float(os.getenv("MCP_CLIENT_TIMEOUT_SEC", "20")),
    )
