from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from mcp_integration.config import load_settings

logger = logging.getLogger(__name__)
_settings = load_settings()
_lock = Lock()


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def log_mcp_event(
    *,
    event_type: str,
    action: str,
    actor: str,
    status: str,
    detail: str = "",
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    entry = {
        "ts": _ts(),
        "event_type": event_type,
        "action": action,
        "actor": actor,
        "status": status,
        "detail": detail,
        "payload": payload or {},
        "result": result or {},
    }
    path = _settings.audit_log_path
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("[mcp.audit] failed to write event: %s", exc)
