import os
import warnings
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

# Suppress some noisy HuggingFace warnings
warnings.filterwarnings("ignore", category=FutureWarning)

_FINANCE_TYPES = {
    "news",
    "rss",
    "earnings_transcript",
    "research_report",
    "macro_commentary",
    "sec",
}

_models: dict[str, SentenceTransformer] = {}
_expected_dims: dict[str, int] = {}
_model_lock: Lock = Lock()


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _embed_provider() -> str:
    return _env_str("EMBED_PROVIDER", "ollama").lower()


def _embed_model() -> str:
    return _env_str("EMBED_MODEL", "nomic-embed-text")


def _finance_embed_model() -> str:
    return _env_str("EMBED_MODEL_FINANCE", "")


def _ollama_url() -> str:
    return _env_str("OLLAMA_URL", "http://localhost:11434").rstrip("/")


def _ollama_embed_url() -> str:
    return _env_str("OLLAMA_EMBED_URL", f"{_ollama_url()}/api/embeddings")


def _ollama_embed_batch_url() -> str:
    return _env_str("OLLAMA_EMBED_BATCH_URL", f"{_ollama_url()}/api/embed")


def _embed_timeout_sec() -> float:
    return float(_env_str("EMBED_TIMEOUT_SEC", "120"))


def _embed_use_prefix() -> bool:
    return _env_str("EMBED_USE_PREFIX", "0").lower() not in ("0", "false", "no", "")


def _embed_query_prefix() -> str:
    return _env_str("EMBED_QUERY_PREFIX", "")


def _embed_passage_prefix() -> str:
    return _env_str("EMBED_PASSAGE_PREFIX", "")


def _default_prefix(model_name: str, role: str) -> str | None:
    name = model_name.lower()
    if "bge-" in name or "e5-" in name:
        return "query: " if role == "query" else "passage: "
    return None


def _apply_prefix(text: str, model_name: str, role: str | None) -> str:
    if not _embed_use_prefix() or role not in ("query", "passage"):
        return text
    query_prefix = _embed_query_prefix()
    passage_prefix = _embed_passage_prefix()
    if role == "query" and query_prefix:
        return query_prefix + text
    if role == "passage" and passage_prefix:
        return passage_prefix + text
    prefix = _default_prefix(model_name, role)
    return f"{prefix}{text}" if prefix else text


def _select_model(data_type: str | None) -> str:
    finance_model = _finance_embed_model()
    if finance_model and (data_type is None or data_type in _FINANCE_TYPES):
        return finance_model
    return _embed_model()


def _as_float32(vec: Any) -> np.ndarray:
    if isinstance(vec, np.ndarray):
        return vec.astype("float32")
    return np.array(vec, dtype="float32")


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec


def _normalize_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _ensure_dim(model_name: str, emb: np.ndarray) -> None:
    if model_name not in _expected_dims:
        _expected_dims[model_name] = len(emb)
        return
    if len(emb) != _expected_dims[model_name]:
        raise ValueError(
            f"Embedding dimension mismatch: {len(emb)} vs {_expected_dims[model_name]}"
        )


def _get_model(model_name: str) -> SentenceTransformer:
    with _model_lock:
        model = _models.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name)
            _models[model_name] = model
    return model


def _ollama_embed(text: str, model_name: str) -> np.ndarray:
    import requests

    response = requests.post(
        _ollama_embed_url(),
        json={"model": model_name, "prompt": text},
        timeout=_embed_timeout_sec(),
    )
    response.raise_for_status()
    embedding = response.json().get("embedding")
    if not embedding:
        raise ValueError("Embedding endpoint returned empty embedding")
    return _as_float32(embedding)


def _ollama_embed_batch(texts: list[str], model_name: str) -> list[np.ndarray | None]:
    import requests

    try:
        response = requests.post(
            _ollama_embed_batch_url(),
            json={"model": model_name, "input": texts},
            timeout=_embed_timeout_sec() * max(1, len(texts) // 2),
        )
        response.raise_for_status()
        vecs = response.json().get("embeddings")
        if vecs and len(vecs) == len(texts):
            return [_as_float32(v) if v else None for v in vecs]
    except Exception:
        pass

    results: list[np.ndarray | None] = []
    for text in texts:
        try:
            results.append(_ollama_embed(text, model_name))
        except Exception:
            results.append(None)
    return results


def get_embedding(
    text: str,
    data_type: str | None = None,
    normalize: bool = True,
    role: str | None = None,
) -> np.ndarray:
    model_name = _select_model(data_type)
    text = _apply_prefix(text, model_name, role)

    if _embed_provider() == "ollama":
        emb = _ollama_embed(text, model_name)
    else:
        model = _get_model(model_name)
        emb = model.encode(text, normalize_embeddings=False)
        emb = _as_float32(emb)

    _ensure_dim(model_name, emb)
    return _normalize(emb) if normalize else emb


def get_embeddings(
    texts: list[str],
    data_type: str | None = None,
    normalize: bool = True,
    role: str | None = None,
) -> list[np.ndarray | None]:
    if not texts:
        return []

    model_name = _select_model(data_type)
    texts = [_apply_prefix(t, model_name, role) for t in texts]

    if _embed_provider() == "ollama":
        vecs = _ollama_embed_batch(texts, model_name)
        out: list[np.ndarray | None] = []
        for vec in vecs:
            if vec is None:
                out.append(None)
                continue
            _ensure_dim(model_name, vec)
            out.append(_normalize(vec) if normalize else vec)
        return out

    model = _get_model(model_name)
    mat = model.encode(texts, normalize_embeddings=False)
    mat = np.asarray(mat, dtype="float32")
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.shape[0] != len(texts):
        raise ValueError("Batch embedding size mismatch")
    _ensure_dim(model_name, mat[0])
    if normalize:
        mat = _normalize_matrix(mat)
    return [mat[i] for i in range(mat.shape[0])]
