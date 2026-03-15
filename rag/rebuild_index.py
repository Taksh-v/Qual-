import json
import os
import faiss
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ingestion.embeddings import get_embedding

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METADATA_PATH = os.path.join(BASE_DIR, "data", "vector_db", "metadata.json")
INDEX_DIR = os.path.join(BASE_DIR, "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

os.makedirs(INDEX_DIR, exist_ok=True)


def embed_text(text: str) -> np.ndarray:
    return np.array(get_embedding(text, normalize=True, role="passage"), dtype="float32")


def main():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = []
    for item in data:
        emb = embed_text(item["text"])
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    print(f"✅ FAISS index rebuilt with {index.ntotal} vectors")


if __name__ == "__main__":
    main()
