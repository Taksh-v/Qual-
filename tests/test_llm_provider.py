"""
test_llm_provider.py
--------------------
Unit tests for intelligence/llm_provider.py
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from intelligence.llm_provider import (
    GeminiProvider,
    OllamaProvider,
    OpenAICompatProvider,
    generate_text,
    get_provider_chain,
)


# ── OllamaProvider ────────────────────────────────────────────────────────────


class TestOllamaProvider:
    def test_generate_success(self):
        provider = OllamaProvider(base_url="http://fake:11434", model="test-model")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Analysis complete."}
        mock_resp.raise_for_status = MagicMock()

        with patch("intelligence.llm_provider.requests.post", return_value=mock_resp):
            result = provider.generate("Test prompt")
        assert result == "Analysis complete."

    def test_generate_empty_response_raises(self):
        provider = OllamaProvider(base_url="http://fake:11434", model="test-model")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": ""}
        mock_resp.raise_for_status = MagicMock()

        with patch("intelligence.llm_provider.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="empty response"):
                provider.generate("Test prompt")

    def test_is_available_true(self):
        provider = OllamaProvider(base_url="http://fake:11434")
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("intelligence.llm_provider.requests.get", return_value=mock_resp):
            assert provider.is_available() is True

    def test_is_available_false_on_error(self):
        provider = OllamaProvider(base_url="http://fake:11434")

        with patch("intelligence.llm_provider.requests.get", side_effect=Exception("conn")):
            assert provider.is_available() is False


# ── OpenAICompatProvider ──────────────────────────────────────────────────────


class TestOpenAICompatProvider:
    def test_generate_success(self):
        provider = OpenAICompatProvider(
            provider_name="groq",
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-70b-versatile",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Macro analysis result."}}],
            "usage": {"total_tokens": 42},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("intelligence.llm_provider.requests.post", return_value=mock_resp):
            result = provider.generate("Test prompt")
        assert result == "Macro analysis result."

    def test_generate_rate_limited_raises(self):
        provider = OpenAICompatProvider(
            provider_name="groq",
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            model="test",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.raise_for_status = MagicMock()

        with patch("intelligence.llm_provider.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="rate limited"):
                provider.generate("Test prompt")

    def test_generate_no_choices_raises(self):
        provider = OpenAICompatProvider(
            provider_name="together",
            api_key="test-key",
            base_url="https://api.together.xyz/v1",
            model="test",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("intelligence.llm_provider.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="no choices"):
                provider.generate("Test prompt")

    def test_is_available_true(self):
        provider = OpenAICompatProvider(
            provider_name="groq",
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
        )
        assert provider.is_available() is True

    def test_is_available_false_no_key(self):
        provider = OpenAICompatProvider(
            provider_name="groq",
            api_key="",
            base_url="https://api.groq.com/openai/v1",
        )
        assert provider.is_available() is False


# ── GeminiProvider ────────────────────────────────────────────────────────────


class TestGeminiProvider:
    def test_generate_success(self):
        provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Gemini result."}]}}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("intelligence.llm_provider.requests.post", return_value=mock_resp):
            result = provider.generate("Test prompt")
        assert result == "Gemini result."

    def test_generate_no_candidates_raises(self):
        provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"candidates": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("intelligence.llm_provider.requests.post", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="no candidates"):
                provider.generate("Test prompt")


# ── Provider chain ────────────────────────────────────────────────────────────


class TestProviderChain:
    def test_chain_defaults_to_ollama(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_PROVIDER_ORDER", "cloud,ollama")
        monkeypatch.delenv("LLM_CLOUD_API_KEY", raising=False)
        chain = get_provider_chain()
        assert len(chain) >= 1
        assert chain[-1].name == "ollama"

    def test_chain_with_groq(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_CLOUD_API_KEY", "test-key")
        monkeypatch.setenv("LLM_PROVIDER_ORDER", "cloud,ollama")
        chain = get_provider_chain()
        names = [p.name for p in chain]
        assert "groq" in names

    def test_generate_text_fallback(self, monkeypatch):
        """First provider fails → falls back to second."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_PROVIDER_ORDER", "ollama")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Fallback answer."}
        mock_resp.raise_for_status = MagicMock()

        with patch("intelligence.llm_provider.requests.post", return_value=mock_resp):
            text, provider_name = generate_text("Test question")
        assert text == "Fallback answer."
        assert provider_name == "ollama"

    def test_generate_text_all_fail_raises(self, monkeypatch):
        """All providers fail → RuntimeError."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_PROVIDER_ORDER", "ollama")

        with patch("intelligence.llm_provider.requests.post", side_effect=Exception("down")):
            with patch("intelligence.llm_provider.requests.get", side_effect=Exception("down")):
                with pytest.raises(RuntimeError, match="All LLM providers failed"):
                    generate_text("Test question")
