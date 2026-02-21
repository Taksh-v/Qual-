import json
import os
import faiss
import numpy as np
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METADATA_PATH = os.path.join(BASE_DIR, "data", "vector_db", "metadata.json")
INDEX_DIR = os.path.join(BASE_DIR, "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

EMBED_MODEL = "nomic-embed-text"

os.makedirs(INDEX_DIR, exist_ok=True)


def embed_text(text: str) -> np.ndarray:
    result = subprocess.run(
        ["ollama", "run", EMBED_MODEL],
        input=text,
        text=True,
        capture_output=True
    )
    return np.array(json.loads(result.stdout), dtype="float32")


def main():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = []
    for item in data:
        emb = embed_text(item["text"])
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    print(f"✅ FAISS index rebuilt with {index.ntotal} vectors")


if __name__ == "__main__":
    main()
