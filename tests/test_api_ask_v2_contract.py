import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
	api_app = importlib.import_module("api.app")
	monkeypatch.setattr(api_app, "_VALID_API_KEYS", set(), raising=False)
	monkeypatch.setattr(api_app, "log_query", lambda **kwargs: None)
	api_app._ask_guard.cache.invalidate()
	with TestClient(api_app.app) as tc:
		yield tc
	api_app._ask_guard.cache.invalidate()


def test_ask_includes_response_contract_metadata(client: TestClient, monkeypatch) -> None:
	api_app = importlib.import_module("api.app")

	async def _fake_ask_rag(question: str) -> dict:
		return {
			"question": question,
			"answer": (
				"Direct answer: A\n"
				"Data snapshot: B\n"
				"Causal chain: C\n"
				"What is happening:\n"
				"- x\n"
				"Market impact:\n"
				"- y\n"
				"Predicted events:\n"
				"- Event 1 (7-30d, 50%): n; trigger: t; invalidation: i.\n"
				"Scenarios (probabilities must add to 100%):\n"
				"- Base (~50%): a\n"
				"- Bull (~30%): b\n"
				"- Bear (~20%): c\n"
				"What to watch:\n"
				"- z\n"
				"Confidence: MEDIUM - m"
			),
			"sources": [],
		}

	monkeypatch.setattr(api_app, "ask_rag", _fake_ask_rag)

	response = client.post("/ask", json={"question": "test question"})
	assert response.status_code == 200
	payload = response.json()
	assert "_response_contract" in payload
	assert payload["_response_contract"]["schema_version"] == "v2"
	assert isinstance(payload["_response_contract"]["validation_ok"], bool)


def test_ask_fallback_still_has_response_contract(client: TestClient, monkeypatch) -> None:
	api_app = importlib.import_module("api.app")

	async def _failing_ask_rag(question: str) -> dict:
		raise RuntimeError("forced failure")

	monkeypatch.setattr(api_app, "ask_rag", _failing_ask_rag)

	response = client.post("/ask", json={"question": "test failure"})
	assert response.status_code == 200
	payload = response.json()
	assert "_response_contract" in payload
	assert payload["_response_contract"]["schema_version"] == "v2"

