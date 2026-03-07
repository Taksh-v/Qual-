import os
import json
import argparse
import faiss
import numpy as np
from tqdm import tqdm
from ingestion.embeddings import get_embedding

CHUNKS_DIR = "data/chunks/news"
INDEX_DIR = "data/vector_db"
INDEX_PATH = os.path.join(INDEX_DIR, "news.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
os.makedirs(INDEX_DIR, exist_ok=True)

index = None
metadata_store = []
skipped = 0
resume_skipped = 0

def _load_existing_if_resume(resume: bool):
    global index, metadata_store
    if not resume:
        index = None
        metadata_store = []
        return
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata_store = json.load(f)
        print(f"🔁 Resume mode: loaded existing index vectors={index.ntotal} metadata={len(metadata_store)}")
    else:
        index = None
        metadata_store = []
        print("🔁 Resume mode: no existing index found, starting fresh.")


def run(resume: bool = False):
    global index, skipped, resume_skipped
    skipped = 0
    resume_skipped = 0
    _load_existing_if_resume(resume)
    seen_chunk_ids = {m.get("chunk_id") for m in metadata_store if m.get("chunk_id")}
    added = 0

    for file in sorted(os.listdir(CHUNKS_DIR)):
        if not file.endswith("_chunks.json"):
            continue

        with open(os.path.join(CHUNKS_DIR, file), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in tqdm(chunks, desc=f"Embedding {file}"):
            try:
                if resume and chunk["chunk_id"] in seen_chunk_ids:
                    resume_skipped += 1
                    continue
                emb = get_embedding(chunk["text"])

                if index is None:
                    index = faiss.IndexFlatL2(len(emb))

                index.add(np.expand_dims(emb, axis=0))

                metadata_store.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
                })
                seen_chunk_ids.add(chunk["chunk_id"])
                added += 1

            except Exception as e:
                skipped += 1
                print(f"⚠️ Skipped chunk {chunk['chunk_id']}: {e}")

    if index is None:
        print("⚠️ No chunks available to index. Nothing written.")
        return

    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=2)

    print(f"\n✅ Index built successfully")
    print(f"📦 Total vectors: {index.ntotal}")
    print(f"➕ Added vectors this run: {added}")
    print(f"⏭️ Resume-skipped chunks: {resume_skipped}")
    print(f"⚠️ Skipped chunks: {skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or resume embedding index.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing data/vector_db/news.index + metadata.json and skip known chunk_ids.",
    )
    args = parser.parse_args()
    run(resume=args.resume)
