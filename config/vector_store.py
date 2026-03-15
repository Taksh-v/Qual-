from __future__ import annotations

import os

_SUPPORTED_BACKENDS = {"faiss", "chroma", "pinecone", "weaviate"}
_DEFAULT_BACKEND = "faiss"


def get_vector_backend() -> str:
    raw = (os.getenv("VECTOR_BACKEND", _DEFAULT_BACKEND) or _DEFAULT_BACKEND).strip().lower()
    if raw in _SUPPORTED_BACKENDS:
        return raw
    return _DEFAULT_BACKEND


def backend_recommendation() -> dict[str, str]:
    return {
        "local_default": "faiss",
        "prototype": "chroma",
        "production_managed": "pinecone_or_weaviate",
    }


def backend_status() -> dict[str, str]:
    backend = get_vector_backend()
    status = "active" if backend == "faiss" else "configured_but_not_integrated_in_current_pipeline"
    return {
        "backend": backend,
        "status": status,
    }
