import os
import json
import argparse
import concurrent.futures
from threading import Lock
from datetime import datetime

INPUT_DIR = "data/raw/sec"
OUTPUT_DIR = "data/processed/sec"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cleaned_count = 0
skipped_existing = 0
lock = Lock()

def clean_sec_filing(filing):
    """Simple cleaner for SEC filings."""
    # Already relatively clean from extraction, just standardize schema
    return {
        "id": filing.get("url", ""),
        "title": filing.get("title", "8-K Filing"),
        "source": filing.get("source", "SEC EDGAR"),
        "company": filing.get("company", "Unknown"),
        "doc_type": filing.get("doc_type", "8-K"),
        "published_at": filing.get("published_at", datetime.utcnow().isoformat()),
        "summary": filing.get("summary", ""),
        "body": filing.get("raw_text", ""), # the actual text to chunk
        "word_count": len(filing.get("raw_text", "").split())
    }

def process_file(file, resume):
    global cleaned_count, skipped_existing
    
    output_file = file.replace(".json", "_clean.json")
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    if resume and os.path.exists(output_path):
        with lock:
            skipped_existing += 1
        print(f"⏭️  Resume skip (already cleaned): {output_file}")
        return

    with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
        article = json.load(f)

    structured = clean_sec_filing(article)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2)

    with lock:
        cleaned_count += 1
    print(f"Cleaned SEC: {output_file}")

def run(resume: bool = False, workers: int = 4):
    global cleaned_count, skipped_existing
    cleaned_count = 0
    skipped_existing = 0
    
    if not os.path.exists(INPUT_DIR):
        print(f"No raw SEC data found in {INPUT_DIR}")
        return
        
    files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.endswith(".json")]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_file, f, resume) for f in files]
        concurrent.futures.wait(futures)

    print(f"✅ SEC Cleaning complete | cleaned={cleaned_count} | resume_skipped={skipped_existing}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and structure raw SEC filings.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files already cleaned in data/processed/sec.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent workers for cleaning.",
    )
    args = parser.parse_args()
    run(resume=args.resume, workers=args.workers)
