"""
llm_provider.py
---------------
Unified LLM provider abstraction layer.

Supports:
  - Local Ollama (default)
  - Groq  (OpenAI-compatible, free tier)
  - Google Gemini (REST API, free tier)
  - OpenRouter, Together AI, Mistral, Cerebras, NVIDIA NIM (all OpenAI-compatible)

Usage:
    from intelligence.llm_provider import get_provider_chain, generate_text
    text = generate_text(prompt)              # auto chain
    text = generate_text(prompt, provider="groq")  # force provider
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Iterator

import requests

logger = logging.getLogger(__name__)

# ── Provider registry & env config ────────────────────────────────────────────

# Base URLs for known OpenAI-compatible providers.
_KNOWN_PROVIDERS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "mistral": "https://api.mistral.ai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
}

# Default model per provider (free tier best pick).
_DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "together": "meta-llama/Llama-3.1-70B-Instruct-Turbo",
    "openrouter": "meta-llama/llama-3.1-70b-instruct:free",
    "mistral": "mistral-small-latest",
    "cerebras": "llama3.1-70b",
    "nvidia": "meta/llama-3.1-70b-instruct",
    "gemini": "gemini-2.5-flash",
    "ollama": "mistral:latest",
}


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ── Abstract base ─────────────────────────────────────────────────────────────


class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.15,
        max_tokens: int = 800,
        timeout_sec: float = 60.0,
    ) -> str:
        """Generate a text completion. Returns the response text."""
        ...

    def generate_stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.15,
        max_tokens: int = 800,
        timeout_sec: float = 60.0,
    ) -> Iterator[str]:
        """Streaming generation. Default: non-streaming fallback."""
        yield self.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and reachable."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"


# ── Ollama (local) ────────────────────────────────────────────────────────────


class OllamaProvider(LLMProvider):
    """Local Ollama provider — wraps the existing inference path."""

    name = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (base_url or _env("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or _env("LLM_MODEL", _DEFAULT_MODELS["ollama"])

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.15,
        max_tokens: int = 800,
        timeout_sec: float = 60.0,
    ) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        text = (resp.json().get("response") or "").strip()
        if not text:
            raise RuntimeError(f"Ollama returned empty response for model={self.model}")
        return text

    def generate_stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.15,
        max_tokens: int = 800,
        timeout_sec: float = 60.0,
    ) -> Iterator[str]:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=timeout_sec,
            stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                except json.JSONDecodeError:
                    continue

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False


# ── OpenAI-compatible (Groq, OpenRouter, Together, Mistral, Cerebras, NVIDIA) ─


class OpenAICompatProvider(LLMProvider):
    """
    Works with any provider that implements the OpenAI chat/completions API.
    Covers: Groq, OpenRouter, Together AI, Mistral, Cerebras, NVIDIA NIM.
    """

    def __init__(
        self,
        provider_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.name = provider_name
        self.api_key = api_key or _env("LLM_CLOUD_API_KEY")
        self.base_url = (
            base_url
            or _env("LLM_CLOUD_BASE_URL")
            or _KNOWN_PROVIDERS.get(provider_name, "")
        ).rstrip("/")
        self.model = model or _env("LLM_CLOUD_MODEL", _DEFAULT_MODELS.get(provider_name, ""))

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        # OpenRouter requires extra headers
        if self.name == "openrouter":
            headers["HTTP-Referer"] = _env("APP_SITE_URL", "http://localhost:8000")
            headers["X-Title"] = _env("APP_BRAND_NAME", "Macro AI")
        return headers

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.15,
        max_tokens: int = 800,
        timeout_sec: float = 60.0,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        t0 = time.time()
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=timeout_sec,
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        if resp.status_code == 429:
            raise RuntimeError(f"{self.name}: rate limited (429). Retry later.")
        resp.raise_for_status()

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"{self.name}: no choices in response")

        text = (choices[0].get("message", {}).get("content") or "").strip()
        if not text:
            raise RuntimeError(f"{self.name}: empty response content")

        logger.info(
            "[llm_provider] %s model=%s elapsed=%dms tokens=%s",
            self.name,
            self.model,
            elapsed_ms,
            data.get("usage", {}).get("total_tokens", "?"),
        )
        return text

    def generate_stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.15,
        max_tokens: int = 800,
        timeout_sec: float = 60.0,
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=timeout_sec,
            stream=True,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
            if not decoded.startswith("data:"):
                continue
            data_str = decoded[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    yield token
            except (json.JSONDecodeError, IndexError):
                continue

    def is_available(self) -> bool:
        return bool(self.api_key and self.base_url)


# ── Google Gemini ─────────────────────────────────────────────────────────────


class GeminiProvider(LLMProvider):
    """Google Gemini via REST API (generativelanguage.googleapis.com)."""

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or _env("GEMINI_API_KEY") or _env("LLM_CLOUD_API_KEY")
        self.model = model or _env("GEMINI_MODEL", _DEFAULT_MODELS["gemini"])
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.15,
        max_tokens: int = 800,
        timeout_sec: float = 60.0,
    ) -> str:
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=timeout_sec)
        elapsed_ms = int((time.time() - t0) * 1000)

        if resp.status_code == 429:
            raise RuntimeError("Gemini: rate limited (429). Retry later.")
        resp.raise_for_status()

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            error_info = data.get("error", {}).get("message", "unknown error")
            raise RuntimeError(f"Gemini: no candidates — {error_info}")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise RuntimeError("Gemini: empty response")

        logger.info("[llm_provider] gemini model=%s elapsed=%dms", self.model, elapsed_ms)
        return text

    def is_available(self) -> bool:
        return bool(self.api_key)


# ── Provider chain builder ────────────────────────────────────────────────────


def _build_provider(name: str) -> LLMProvider | None:
    """Instantiate a provider by name. Returns None if not configured."""
    name = name.strip().lower()
    if name == "ollama":
        return OllamaProvider()
    if name == "gemini":
        p = GeminiProvider()
        return p if p.is_available() else None
    if name in _KNOWN_PROVIDERS:
        p = OpenAICompatProvider(provider_name=name)
        return p if p.is_available() else None
    logger.warning("[llm_provider] Unknown provider: %s", name)
    return None


def get_provider_chain() -> list[LLMProvider]:
    """
    Build the ordered list of LLM providers to try.

    Reads LLM_PROVIDER_ORDER env var (comma-separated).
    Default: "cloud,ollama" — tries cloud first, then local.

    If LLM_PROVIDER is set (single provider name), it is used as the
    sole cloud provider in the chain.
    """
    order_str = _env("LLM_PROVIDER_ORDER", "cloud,ollama")
    primary = _env("LLM_PROVIDER", "ollama")

    chain: list[LLMProvider] = []
    seen: set[str] = set()

    for slot in order_str.split(","):
        slot = slot.strip().lower()
        if slot in seen:
            continue
        seen.add(slot)

        if slot == "cloud":
            # "cloud" expands to LLM_PROVIDER if it's a cloud provider
            if primary and primary != "ollama":
                p = _build_provider(primary)
                if p:
                    chain.append(p)
        elif slot == "ollama":
            p = _build_provider("ollama")
            if p:
                chain.append(p)
        else:
            p = _build_provider(slot)
            if p:
                chain.append(p)

    # Failsafe: always have at least Ollama
    if not chain:
        chain.append(OllamaProvider())

    return chain


# ── Convenience API ───────────────────────────────────────────────────────────


def generate_text(
    prompt: str,
    *,
    provider: str | None = None,
    temperature: float = 0.15,
    max_tokens: int = 800,
    timeout_sec: float | None = None,
) -> tuple[str, str]:
    """
    Generate text using the configured provider chain.

    Args:
        prompt: The prompt to send.
        provider: Force a specific provider name (bypass chain).
        temperature: LLM temperature.
        max_tokens: Max output tokens.
        timeout_sec: Per-call timeout (default from env).

    Returns:
        (response_text, provider_name) tuple.

    Raises:
        RuntimeError: All providers in the chain failed.
    """
    if timeout_sec is None:
        timeout_sec = _env_float("LLM_CLOUD_TIMEOUT_SEC", 60.0)

    if provider:
        p = _build_provider(provider)
        if p is None:
            raise RuntimeError(f"Provider '{provider}' is not configured or unavailable.")
        text = p.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
        return text, p.name

    chain = get_provider_chain()
    last_error: str = ""
    for p in chain:
        try:
            text = p.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
            )
            return text, p.name
        except Exception as exc:
            last_error = f"{p.name}: {exc}"
            logger.warning("[llm_provider] %s failed: %s", p.name, exc)
            continue

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def generate_text_stream(
    prompt: str,
    *,
    provider: str | None = None,
    temperature: float = 0.15,
    max_tokens: int = 800,
    timeout_sec: float | None = None,
) -> Iterator[str]:
    """
    Streaming text generation. Yields tokens as they arrive.
    Falls through the provider chain on failure.
    """
    if timeout_sec is None:
        timeout_sec = _env_float("LLM_CLOUD_TIMEOUT_SEC", 60.0)

    if provider:
        p = _build_provider(provider)
        if p is None:
            raise RuntimeError(f"Provider '{provider}' is not configured.")
        yield from p.generate_stream(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
        return

    chain = get_provider_chain()
    for p in chain:
        try:
            yield from p.generate_stream(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
            )
            return
        except Exception as exc:
            logger.warning("[llm_provider] stream %s failed: %s", p.name, exc)
            continue

    raise RuntimeError("All LLM providers failed for streaming generation.")
