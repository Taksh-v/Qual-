import importlib
import json
from typing import Iterator

import pytest
from fastapi.testclient import TestClient


def _fake_snapshot(req) -> dict:
    return {
        "question": req.question,
        "geography": req.geography,
        "horizon": req.horizon,
        "classification": {"primary": "macro"},
        "regime": {"regime": "SOFT_LANDING", "confidence": "MEDIUM"},
        "cross_asset": {"overall_signal": "RISK_ON"},
        "critical_indicators": [],
        "detected_indicators": {},
        "evidence_coverage": {
            "context_chunks": 2,
            "has_overrides": False,
            "sources": [],
        },
    }


def _fake_pipeline(*args, **kwargs) -> Iterator[str]:
    yield "━━━ KOTAK MACRO AI [2026-03-20] ━━━\n"
    yield "Regime: SOFT_LANDING [MEDIUM] | Signal: RISK_ON\n"
    yield "▸ RESPONSE\n"
    yield "Direct answer: Growth is moderating while inflation cools.\n"
    yield "Market impact: Equities supported, yields drift lower.\n"
    yield "Confidence: MEDIUM - Stable macro backdrop.\n"
    yield "\nExternal MCP context (supplementary, uncited):\n"
    yield "External macro context: disinflation trend remains intact.\n"


def _fake_pipeline_no_enrichment(*args, **kwargs) -> Iterator[str]:
    yield "━━━ KOTAK MACRO AI [2026-03-20] ━━━\n"
    yield "Regime: SOFT_LANDING [MEDIUM] | Signal: RISK_ON\n"
    yield "▸ RESPONSE\n"
    yield "Direct answer: Growth is moderating while inflation cools.\n"
    yield "Market impact: Equities supported, yields drift lower.\n"
    yield "Confidence: MEDIUM - External enrichment unavailable, base analysis used.\n"


@pytest.fixture()
def client(monkeypatch):
    api_app = importlib.import_module("api.app")

    monkeypatch.setattr(api_app, "_VALID_API_KEYS", set(), raising=False)
    monkeypatch.setattr(api_app, "_build_snapshot", _fake_snapshot)
    monkeypatch.setattr(api_app, "macro_intelligence_pipeline", _fake_pipeline)
    monkeypatch.setattr(api_app, "get_last_model_used", lambda: "unit-test-model")
    monkeypatch.setattr(api_app, "log_query", lambda **kwargs: None)

    api_app._intel_guard.cache.invalidate()
    with TestClient(api_app.app) as tc:
        yield tc
    api_app._intel_guard.cache.invalidate()


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        event_name = ""
        data_raw = "{}"
        for line in lines:
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data_raw = line[len("data: "):]
        if event_name:
            events.append((event_name, json.loads(data_raw)))
    return events


def test_intelligence_analyze_includes_external_mcp_context(client: TestClient) -> None:
    response = client.post(
        "/intelligence/analyze",
        json={
            "question": "How is inflation affecting rates?",
            "geography": "US",
            "horizon": "MEDIUM_TERM",
            "response_mode": "brief",
            "indicator_overrides": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "response_text" in payload
    assert "External MCP context (supplementary, uncited):" in payload["response_text"]
    assert "disinflation trend remains intact" in payload["response_text"]
    assert payload.get("_response_contract", {}).get("schema_version") == "v2"
    assert isinstance(payload.get("_response_contract", {}).get("validation_ok"), bool)


def test_intelligence_stream_emits_enriched_final_payload(client: TestClient) -> None:
    response = client.post(
        "/intelligence/stream",
        json={
            "question": "What is the macro outlook for risk assets?",
            "geography": "US",
            "horizon": "MEDIUM_TERM",
            "response_mode": "brief",
            "indicator_overrides": {},
        },
    )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/event-stream")

    events = _parse_sse(response.text)
    event_names = [name for name, _ in events]

    assert "snapshot" in event_names
    assert "token" in event_names
    assert "final" in event_names

    final_payload = next(data for name, data in events if name == "final")
    assert "response_text" in final_payload
    assert "External MCP context (supplementary, uncited):" in final_payload["response_text"]
    assert "disinflation trend remains intact" in final_payload["response_text"]
    assert final_payload.get("_response_contract", {}).get("schema_version") == "v2"
    assert isinstance(final_payload.get("_response_contract", {}).get("validation_ok"), bool)


def test_intelligence_analyze_succeeds_without_external_enrichment(
    client: TestClient,
    monkeypatch,
) -> None:
    api_app = importlib.import_module("api.app")
    monkeypatch.setattr(api_app, "macro_intelligence_pipeline", _fake_pipeline_no_enrichment)

    response = client.post(
        "/intelligence/analyze",
        json={
            "question": "Does policy uncertainty change the base case?",
            "geography": "US",
            "horizon": "MEDIUM_TERM",
            "response_mode": "brief",
            "indicator_overrides": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "response_text" in payload
    assert "Direct answer:" in payload["response_text"]
    assert "External MCP context (supplementary, uncited):" not in payload["response_text"]
    assert payload.get("_response_contract", {}).get("schema_version") == "v2"


def test_intelligence_stream_succeeds_without_external_enrichment(
    client: TestClient,
    monkeypatch,
) -> None:
    api_app = importlib.import_module("api.app")
    monkeypatch.setattr(api_app, "macro_intelligence_pipeline", _fake_pipeline_no_enrichment)

    response = client.post(
        "/intelligence/stream",
        json={
            "question": "What if external context service is down?",
            "geography": "US",
            "horizon": "MEDIUM_TERM",
            "response_mode": "brief",
            "indicator_overrides": {},
        },
    )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/event-stream")

    events = _parse_sse(response.text)
    event_names = [name for name, _ in events]
    assert "snapshot" in event_names
    assert "token" in event_names
    assert "final" in event_names

    final_payload = next(data for name, data in events if name == "final")
    assert "response_text" in final_payload
    assert "Direct answer:" in final_payload["response_text"]
    assert "External MCP context (supplementary, uncited):" not in final_payload["response_text"]
    assert final_payload.get("_response_contract", {}).get("schema_version") == "v2"
