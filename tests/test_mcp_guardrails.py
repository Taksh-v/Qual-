import pytest

from mcp_integration.guardrails import (
    AuthManager,
    ConfirmationStore,
    IdempotencyStore,
    MCPAuthError,
    MCPValidationError,
    OperationLocks,
)


def test_auth_manager_scope_enforcement() -> None:
    auth = AuthManager(read_keys={"read-key"}, admin_keys={"admin-key"})

    read_principal = auth.authenticate("read-key", "read")
    assert read_principal.auth_mode == "keys"
    assert "read" in read_principal.scopes
    assert "admin" not in read_principal.scopes

    admin_principal = auth.authenticate("admin-key", "admin")
    assert "admin" in admin_principal.scopes
    assert "read" in admin_principal.scopes

    with pytest.raises(MCPAuthError):
        auth.authenticate("read-key", "admin")


def test_confirmation_store_single_use() -> None:
    store = ConfirmationStore(ttl_sec=60)
    pending = store.create(action="reload_index", params={"force": True}, actor_id="actor-1")

    consumed = store.consume(token=pending.token, action="reload_index", actor_id="actor-1")
    assert consumed.action == "reload_index"
    assert consumed.params == {"force": True}

    with pytest.raises(MCPValidationError):
        store.consume(token=pending.token, action="reload_index", actor_id="actor-1")


def test_idempotency_store_rejects_key_reuse_across_actor() -> None:
    store = IdempotencyStore(ttl_sec=120)
    key = "idem-123"
    store.put(idempotency_key=key, action="quality_gate", actor_id="actor-a", result={"ok": True})

    cached = store.get(idempotency_key=key, action="quality_gate", actor_id="actor-a")
    assert cached == {"ok": True}

    with pytest.raises(MCPValidationError):
        store.get(idempotency_key=key, action="quality_gate", actor_id="actor-b")


def test_operation_locks_prevent_reentrant_action() -> None:
    locks = OperationLocks()

    with locks.held("vector_audit"):
        with pytest.raises(MCPValidationError):
            with locks.held("vector_audit"):
                pass
