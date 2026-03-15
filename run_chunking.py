import json
import os
import argparse
import concurrent.futures
from threading import Lock
from ingestion.chunker import chunk_text

INPUT_DIRS = {
    "news": "data/processed/news",
    "sec": "data/processed/sec"
}
OUTPUT_DIRS = {
    "news": "data/chunks/news",
    "sec": "data/chunks/sec"
}

for out_dir in OUTPUT_DIRS.values():
    os.makedirs(out_dir, exist_ok=True)

lock = Lock()
total_chunks_created = 0
skipped_files = 0

def chunk_file(doc_type, file, resume):
    global total_chunks_created, skipped_files

    input_dir = INPUT_DIRS[doc_type]
    output_dir = OUTPUT_DIRS[doc_type]
    
    in_path = os.path.join(input_dir, file)
    out_file = file.replace("_clean.json", "_chunks.json")
    out_path = os.path.join(output_dir, out_file)

    if resume and os.path.exists(out_path):
        with lock:
            skipped_files += 1
        print(f"⏭️  Resume skip (already chunked): {doc_type}/{out_file}")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        article = json.load(f)

    # Convert the structured article text into chunks with context
    raw_meta = article.get("metadata", {}) if isinstance(article.get("metadata"), dict) else {}
    body_text = (
        article.get("body")
        or article.get("structured_text")
        or article.get("content")
        or raw_meta.get("content")
        or ""
    )

    default_doc_type = "News Article" if doc_type == "news" else "SEC Filing"
    published_at = (
        article.get("published_at")
        or raw_meta.get("published_at")
        or raw_meta.get("date")
        or ""
    )
    date_val = article.get("date") or raw_meta.get("date") or published_at

    metadata = {
        "title": article.get("title") or raw_meta.get("title") or "",
        "source": article.get("source") or raw_meta.get("source") or "",
        "company": article.get("company") or raw_meta.get("company") or "Unknown",
        "doc_type": article.get("doc_type") or raw_meta.get("doc_type") or default_doc_type,
        "published_at": published_at,
        "date": date_val or "",
        "url": article.get("url") or raw_meta.get("url") or "",
        "extracted_at": article.get("extracted_at") or raw_meta.get("extracted_at") or "",
        "data_type": "news" if doc_type == "news" else "sec",
    }
    
    chunks = chunk_text(body_text, chunk_size=2000, overlap=250, with_metadata=True, extra_metadata={"metadata": metadata})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    with lock:
        total_chunks_created += len(chunks)
    print(f"Chunked [{doc_type}]: {out_file} -> {len(chunks)} chunks")

def run_type(doc_type, resume, workers):
    if not os.path.exists(INPUT_DIRS[doc_type]):
        return 0, 0
        
    files = [f for f in sorted(os.listdir(INPUT_DIRS[doc_type])) if f.endswith("_clean.json")]
    
    if not files:
        print(f"⚠️ No files to chunk in {INPUT_DIRS[doc_type]}. Skipping.")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(chunk_file, doc_type, f, resume) for f in files]
        concurrent.futures.wait(futures)

def run(resume: bool = False, workers: int = 4):
    global total_chunks_created, skipped_files
    total_chunks_created = 0
    skipped_files = 0

    run_type("news", resume, workers)
    run_type("sec", resume, workers)

    print(f"✅ Chunking complete | total_new_chunks={total_chunks_created} | files_skipped={skipped_files}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk documents for embedding.")
    parser.add_argument("--resume", action="store_true", help="Skip already chunked files.")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent workers.")
    args = parser.parse_args()
    run(resume=args.resume, workers=args.workers)
