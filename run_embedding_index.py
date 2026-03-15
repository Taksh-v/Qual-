import os
import json
import argparse
from datetime import datetime, timezone
import faiss
import numpy as np
import concurrent.futures
from threading import Lock
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ingestion.embeddings import get_embedding
from ingestion.metadata_store import MetadataStore, get_store
from ingestion.bm25_index import build_bm25_index

# ---------------------------------------------------------------------------
# IndexFlatIP + L2-normalised vectors gives exact cosine-similarity ranking.
# Previously IndexFlatL2 was used, which gives squared-Euclidean distance and
# produces suboptimal ranking for sentence-transformer embeddings.
# ---------------------------------------------------------------------------

CHUNKS_DIRS = [
    "data/chunks/news",
    "data/chunks/sec",
    "data/chunks/earnings",
    "data/chunks/research",
    "data/chunks/macro",
]
INDEX_DIR = "data/vector_db"
INDEX_PATH = os.path.join(INDEX_DIR, "news.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
os.makedirs(INDEX_DIR, exist_ok=True)

index = None
metadata_store = []
skipped = 0
resume_skipped = 0
added = 0
lock = Lock()
seen_chunk_ids = set()

def _load_existing_if_resume(resume: bool):
    global index, metadata_store
    if not resume:
        index = None
        metadata_store = []
        return

    # Try loading from SQLite store (more reliable than JSON for rowid alignment)
    db_store = get_store()
    db_count = db_store.count()

    if os.path.exists(INDEX_PATH) and db_count > 0:
        index = faiss.read_index(INDEX_PATH)
        # Load from SQLite to guarantee rowids match the FAISS index positions
        metadata_store = db_store.get_all_as_list()
        print(f"🔁 Resume mode: loaded index vectors={index.ntotal}, SQLite rows={db_count}")
        return

    # Fall back to JSON if SQLite is empty (first run after upgrade)
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata_store = json.load(f)
        print(f"🔁 Resume mode (JSON fallback): loaded index vectors={index.ntotal} metadata={len(metadata_store)}")
    else:
        index = None
        metadata_store = []
        print("🔁 Resume mode: no existing index found, starting fresh.")


def _infer_data_type(md: dict) -> str:
    data_type = (md.get("data_type") or "").strip().lower()
    if data_type:
        return data_type
    doc_type = (md.get("doc_type") or "").strip().lower()
    if "8-k" in doc_type or "sec" in doc_type or "filing" in doc_type:
        return "sec"
    if "earnings" in doc_type or "transcript" in doc_type:
        return "earnings_transcript"
    if "macro" in doc_type or "economic" in doc_type:
        return "macro_commentary"
    if "research" in doc_type or "report" in doc_type:
        return "research_report"
    if "news" in doc_type or "article" in doc_type:
        return "news"
    return ""


def _normalize_metadata(chunk: dict) -> tuple[dict, str, str]:
    raw_md = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    md = dict(raw_md)

    sector = chunk.get("sector") or md.get("sector") or "Unknown"
    region = chunk.get("region") or md.get("region") or "Global"

    if not md.get("source"):
        md["source"] = "Unknown"
    if not md.get("date"):
        md["date"] = md.get("published_at") or md.get("extracted_at") or ""
    if not md.get("company"):
        md["company"] = "Unknown"
    if not md.get("sector"):
        md["sector"] = sector
    if not md.get("region"):
        md["region"] = region

    if not md.get("indexed_at"):
        md["indexed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if not md.get("data_type"):
        inferred = _infer_data_type(md)
        if inferred:
            md["data_type"] = inferred

    if not md.get("company_sector"):
        company = md.get("company") or ""
        if company and company != "Unknown":
            md["company_sector"] = f"{company} - {sector}"

    return md, sector, region


def process_chunk(chunk, resume):
    global added, skipped, resume_skipped, index, metadata_store, seen_chunk_ids
    
    if resume and chunk["chunk_id"] in seen_chunk_ids:
        with lock:
            resume_skipped += 1
        return

    try:
        md = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        data_type = _infer_data_type(md) or None
        emb = get_embedding(chunk["text"], data_type=data_type, normalize=True, role="passage")

        with lock:
            global index
            if index is None:
                # IndexFlatIP: inner-product search on normalised vectors = cosine similarity
                index = faiss.IndexFlatIP(len(emb))
            
            index.add(np.expand_dims(emb, axis=0))

            norm_md, sector, region = _normalize_metadata(chunk)

            meta_out = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "sector": sector,
                "region": region,
                "metadata": norm_md,
            }
            metadata_store.append(meta_out)
            seen_chunk_ids.add(chunk["chunk_id"])
            added += 1

    except Exception as e:
        with lock:
            skipped += 1
        print(f"⚠️ Skipped chunk {chunk['chunk_id']}: {e}")

def run(resume: bool = False, workers: int = 8):
    global index, skipped, resume_skipped, seen_chunk_ids, added
    skipped = 0
    resume_skipped = 0
    added = 0
    _load_existing_if_resume(resume)
    seen_chunk_ids = {m.get("chunk_id") for m in metadata_store if m.get("chunk_id")}

    for active_dir in CHUNKS_DIRS:
        if not os.path.exists(active_dir):
            continue
            
        files = [f for f in sorted(os.listdir(active_dir)) if f.endswith("_chunks.json")]

        for file in files:
            with open(os.path.join(active_dir, file), "r", encoding="utf-8") as f:
                chunks = json.load(f)

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                # We wrap with tqdm to show progress, though order is non-deterministic
                list(tqdm(executor.map(lambda c: process_chunk(c, resume), chunks), total=len(chunks), desc=f"Embedding {active_dir}/{file}"))

    if index is None:
        print("⚠️ No chunks available to index. Nothing written.")
        return

    # 1. Write FAISS index
    faiss.write_index(index, INDEX_PATH)

    # 2. Write JSON (backward-compat for any code still reading metadata.json)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=2)

    # 3. Write / update SQLite metadata store (fast filtered queries)
    print("\n💾 Writing to SQLite metadata store...")
    db_store = get_store()
    # In fresh builds, rowid = position in metadata_store list.
    # In resume builds, newly-added items start at the existing rowid offset.
    start_rowid = index.ntotal - added
    new_chunks = metadata_store[start_rowid:] if added < len(metadata_store) else metadata_store
    written = db_store.upsert_chunks(new_chunks, start_rowid=start_rowid)
    print(f"   SQLite rows written: {written}  |  total rows: {db_store.count()}")

    # 4. Build BM25 index over all current chunks
    print("\n📑 Building BM25 sparse index...")
    all_chunks = metadata_store  # full list for fresh builds
    if resume and len(metadata_store) > added:
        all_chunks = metadata_store  # always re-build over all chunks for accuracy
    build_bm25_index(all_chunks)
    print(f"   BM25 index built over {len(all_chunks)} chunks")

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
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of threads for Ollama embedding generation.",
    )
    args = parser.parse_args()
    run(resume=args.resume, workers=args.workers)
