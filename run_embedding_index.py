import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from ingestion.embeddings import get_embedding

CHUNKS_DIR = "data/chunks/news"
INDEX_DIR = "data/vector_db"
os.makedirs(INDEX_DIR, exist_ok=True)

index = None
metadata_store = []
skipped = 0

def run():
    global index, skipped

    for file in os.listdir(CHUNKS_DIR):
        if not file.endswith("_chunks.json"):
            continue

        with open(os.path.join(CHUNKS_DIR, file), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in tqdm(chunks, desc=f"Embedding {file}"):
            try:
                emb = get_embedding(chunk["text"])

                if index is None:
                    index = faiss.IndexFlatL2(len(emb))

                index.add(np.expand_dims(emb, axis=0))

                metadata_store.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
                })

            except Exception as e:
                skipped += 1
                print(f"⚠️ Skipped chunk {chunk['chunk_id']}: {e}")

    faiss.write_index(index, os.path.join(INDEX_DIR, "news.index"))

    with open(os.path.join(INDEX_DIR, "metadata.json"), "w") as f:
        json.dump(metadata_store, f, indent=2)

    print(f"\n✅ Index built successfully")
    print(f"📦 Total vectors: {index.ntotal}")
    print(f"⚠️ Skipped chunks: {skipped}")

if __name__ == "__main__":
    run()
