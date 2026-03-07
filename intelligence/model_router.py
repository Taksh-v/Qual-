from __future__ import annotations

import os
from typing import Set

import requests


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
    Use best model by default on 16GB+ memory and >=6 CPU cores.
    Override via MODEL_TIER=best|current.
    """
    tier = os.getenv("MODEL_TIER", "auto").strip().lower()
    if tier == "best":
        return True
    if tier == "current":
        return False
    # Conservative auto policy for CPU-only systems.
    # Prefer best model automatically only on clearly strong hardware.
    return _mem_available_gb() >= 20 and _cpu_count() >= 8


def _installed_models() -> Set[str]:
    try:
        resp = requests.get(
            f"{os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')}/api/tags",
            timeout=1.5,
        )
        resp.raise_for_status()
        models = resp.json().get("models", [])
        names = set()
        for m in models:
            name = (m.get("name") or "").strip()
            if name:
                names.add(name)
        return names
    except Exception:
        return set()


def get_model_candidates() -> list[str]:
    """
    Exactly two-model routing:
    - current model (existing stable default): phi3:mini
    - better model for 16GB CPU systems: qwen2.5:7b-instruct

    Env knobs:
    - MODEL_TIER: auto|best|current
    - LLM_MODEL_CURRENT
    - LLM_MODEL_BEST
    """
    current_model = os.getenv("LLM_MODEL_CURRENT", "phi3:mini").strip()
    best_model = os.getenv("LLM_MODEL_BEST", "mistral:latest").strip()

    ordered = [best_model, current_model] if _prefer_best_model() else [current_model, best_model]

    installed = _installed_models()
    unique: list[str] = []
    for model in ordered:
        if not model or model in unique:
            continue
        # If we can detect installed models, only return installed candidates.
        if installed and model not in installed:
            continue
        if model not in unique:
            unique.append(model)
    # Failsafe: if filtering removed everything, keep current as a last try.
    if not unique and current_model:
        unique.append(current_model)
    return unique
