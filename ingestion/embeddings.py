import requests
import numpy as np

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

EXPECTED_DIM = None  # will be set dynamically

def get_embedding(text: str) -> np.ndarray:
    global EXPECTED_DIM

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": text
        },
        timeout=60
    )
    response.raise_for_status()

    embedding = response.json().get("embedding")

    if not embedding or not isinstance(embedding, list):
        raise ValueError("Invalid embedding returned")

    emb = np.array(embedding, dtype="float32")

    # lock dimension
    if EXPECTED_DIM is None:
        EXPECTED_DIM = len(emb)
    elif len(emb) != EXPECTED_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: {len(emb)} vs {EXPECTED_DIM}"
        )

    return emb
