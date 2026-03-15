from __future__ import annotations

import os
import time
from threading import Lock
from typing import Set

import requests

# ── TTL cache for installed-model list ────────────────────────────────────────
_models_cache: Set[str] = set()
_models_cache_expires: float = 0.0
_models_cache_lock = Lock()
_MODELS_TTL: float = float(os.getenv("MODEL_LIST_CACHE_TTL", "60"))  # seconds
_AUTO_MIN_MEM_GB_FOR_BEST: float = float(os.getenv("MODEL_AUTO_MIN_MEM_GB_FOR_BEST", "24"))
_AUTO_MIN_CPUS_FOR_BEST: int = int(os.getenv("MODEL_AUTO_MIN_CPUS_FOR_BEST", "8"))


def _cpu_count() -> int:
    return os.cpu_count() or 2


def _mem_available_gb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    return kb / (1024 * 1024)
    except Exception:
        pass
    return 8.0


def _prefer_best_model() -> bool:
    """
    In auto mode, prefer the best model only when the local machine has enough
    headroom to do so without hurting interactivity.

    Override via MODEL_TIER=best|current|auto.
    """
    tier = os.getenv("MODEL_TIER", "auto").strip().lower()
    if tier == "best":
        return True
    if tier == "current":
        return False

    # Auto mode: local CPU-only environments are often better served by the
    # faster current model unless there is clear memory/CPU headroom.
    return (
        _mem_available_gb() >= _AUTO_MIN_MEM_GB_FOR_BEST
        and _cpu_count() >= _AUTO_MIN_CPUS_FOR_BEST
    )


def _installed_models() -> Set[str]:
    global _models_cache, _models_cache_expires
    now = time.time()
    with _models_cache_lock:
        if now < _models_cache_expires and _models_cache:
            return _models_cache
    # Cache miss — hit the API once and store the result.
    try:
        resp = requests.get(
            f"{os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')}/api/tags",
            timeout=1.5,
        )
        resp.raise_for_status()
        models = resp.json().get("models", [])
        names: Set[str] = set()
        for m in models:
            name = (m.get("name") or "").strip()
            if name:
                names.add(name)
        with _models_cache_lock:
            _models_cache = names
            _models_cache_expires = now + _MODELS_TTL
        return names
    except Exception:
        return set()


def get_model_candidates() -> list[str]:
    """
    Multi-tier model routing — tries best available model first.

    Priority tiers (best → fastest fallback):
      Tier 1 (best quality):  mistral:latest, llama3.1:8b, llama3:latest
      Tier 2 (balanced):      qwen2.5:7b-instruct, llama3.2:3b, gemma2:9b
      Tier 3 (fast fallback): phi3:mini, phi3:medium, tinyllama

    Env knobs:
    - MODEL_TIER: auto|best|current
    - LLM_MODEL_CURRENT  (default: phi3:mini)
    - LLM_MODEL_BEST     (default: mistral:latest)
    - LLM_MODEL_BALANCED (default: llama3.2:3b)
    """
    current_model  = os.getenv("LLM_MODEL_CURRENT",  "phi3:mini").strip()
    best_model     = os.getenv("LLM_MODEL_BEST",     "mistral:latest").strip()
    balanced_model = os.getenv("LLM_MODEL_BALANCED", "llama3.2:3b").strip()

    # Full ranked preference list — highest quality first
    _TIER1 = [best_model, "llama3.1:8b", "llama3:latest", "mixtral:8x7b"]
    _TIER2 = [balanced_model, "qwen2.5:7b-instruct", "qwen2.5:3b", "gemma2:9b"]
    _TIER3 = [current_model, "phi3:medium", "phi3:mini", "tinyllama"]

    if _prefer_best_model():
        preference = _TIER1 + _TIER2 + _TIER3
    else:
        preference = _TIER3 + _TIER2 + _TIER1

    installed = _installed_models()
    unique: list[str] = []
    for model in preference:
        if not model or model in unique:
            continue
        # Only include installed models when we can detect them
        if installed and model not in installed:
            continue
        unique.append(model)

    # Failsafe: if filtering removed everything, always keep the current model
    if not unique:
        unique.append(current_model)
    return unique
