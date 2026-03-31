from __future__ import annotations

import hashlib
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any


class MCPAuthError(PermissionError):
    pass


class MCPValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Principal:
    actor_id: str
    scopes: frozenset[str]
    auth_mode: str


@dataclass(frozen=True)
class PendingAction:
    token: str
    action: str
    params: dict[str, Any]
    actor_id: str
    expires_at: float


class AuthManager:
    def __init__(self, read_keys: set[str], admin_keys: set[str]) -> None:
        self._read_keys = read_keys
        self._admin_keys = admin_keys
        self._auth_enabled = bool(read_keys or admin_keys)

    @staticmethod
    def _fingerprint(key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"key_{digest[:12]}"

    def authenticate(self, api_key: str | None, required_scope: str) -> Principal:
        if required_scope not in {"read", "admin"}:
            raise MCPValidationError(f"unknown scope '{required_scope}'")

        if not self._auth_enabled:
            return Principal(actor_id="dev_mode", scopes=frozenset({"read", "admin"}), auth_mode="dev")

        key = (api_key or "").strip()
        if not key:
            raise MCPAuthError("missing api_key")

        scopes: set[str] = set()
        if key in self._read_keys:
            scopes.add("read")
        if key in self._admin_keys:
            scopes.update({"read", "admin"})

        if required_scope not in scopes:
            raise MCPAuthError("insufficient scope")

        return Principal(actor_id=self._fingerprint(key), scopes=frozenset(scopes), auth_mode="keys")


class ConfirmationStore:
    def __init__(self, ttl_sec: int) -> None:
        self._ttl_sec = ttl_sec
        self._items: dict[str, PendingAction] = {}
        self._lock = Lock()

    def _cleanup(self) -> None:
        now = time.time()
        expired = [token for token, item in self._items.items() if item.expires_at <= now]
        for token in expired:
            self._items.pop(token, None)

    def create(self, *, action: str, params: dict[str, Any], actor_id: str) -> PendingAction:
        token = secrets.token_urlsafe(24)
        now = time.time()
        item = PendingAction(
            token=token,
            action=action,
            params=dict(params),
            actor_id=actor_id,
            expires_at=now + self._ttl_sec,
        )
        with self._lock:
            self._cleanup()
            self._items[token] = item
        return item

    def consume(self, *, token: str, action: str, actor_id: str) -> PendingAction:
        with self._lock:
            self._cleanup()
            item = self._items.pop(token, None)
        if item is None:
            raise MCPValidationError("invalid_or_expired_confirmation_token")
        if item.action != action:
            raise MCPValidationError("confirmation_action_mismatch")
        if item.actor_id != actor_id:
            raise MCPAuthError("confirmation_actor_mismatch")
        return item


@dataclass(frozen=True)
class IdempotencyEntry:
    action: str
    actor_id: str
    expires_at: float
    result: dict[str, Any]


class IdempotencyStore:
    def __init__(self, ttl_sec: int) -> None:
        self._ttl_sec = ttl_sec
        self._items: dict[str, IdempotencyEntry] = {}
        self._lock = Lock()

    def _cleanup(self) -> None:
        now = time.time()
        expired = [key for key, entry in self._items.items() if entry.expires_at <= now]
        for key in expired:
            self._items.pop(key, None)

    def get(self, *, idempotency_key: str, action: str, actor_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._cleanup()
            entry = self._items.get(idempotency_key)
        if entry is None:
            return None
        if entry.action != action or entry.actor_id != actor_id:
            raise MCPValidationError("idempotency_key_reused_for_different_action_or_actor")
        return dict(entry.result)

    def put(self, *, idempotency_key: str, action: str, actor_id: str, result: dict[str, Any]) -> None:
        entry = IdempotencyEntry(
            action=action,
            actor_id=actor_id,
            expires_at=time.time() + self._ttl_sec,
            result=dict(result),
        )
        with self._lock:
            self._cleanup()
            self._items[idempotency_key] = entry


class OperationLocks:
    def __init__(self) -> None:
        self._locks: dict[str, Lock] = {}
        self._lock = Lock()

    def _get_lock(self, action: str) -> Lock:
        with self._lock:
            if action not in self._locks:
                self._locks[action] = Lock()
            return self._locks[action]

    @contextmanager
    def held(self, action: str):
        lock = self._get_lock(action)
        acquired = lock.acquire(blocking=False)
        if not acquired:
            raise MCPValidationError("operation_already_running")
        try:
            yield
        finally:
            lock.release()
