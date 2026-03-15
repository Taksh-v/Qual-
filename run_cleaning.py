import os
import json
import argparse
import concurrent.futures
from threading import Lock
from ingestion.cleaner import structure_article

INPUT_DIR = "data/raw/news"
OUTPUT_DIR = "data/processed/news"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cleaned_count = 0
skipped_existing = 0
quality_skipped = 0
lock = Lock()

def process_file(file, resume):
    global cleaned_count, skipped_existing, quality_skipped
    
    output_file = file.replace(".json", "_clean.json")
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    if resume and os.path.exists(output_path):
        with lock:
            skipped_existing += 1
        print(f"⏭️  Resume skip (already cleaned): {output_file}")
        return

    with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
        article = json.load(f)

    structured = structure_article(article)

    # Quality gate: structure_article returns None for articles that are too short
    if structured is None:
        with lock:
            quality_skipped += 1
        print(f"🚫 Quality skip (too short/empty): {file}")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2)

    with lock:
        cleaned_count += 1
    print(f"Cleaned: {output_file}")

def run(resume: bool = False, workers: int = 4):
    global cleaned_count, skipped_existing, quality_skipped
    cleaned_count = 0
    skipped_existing = 0
    quality_skipped = 0
    
    files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.endswith(".json")]
    if not files:
        print("⚠️  No raw news files found in", INPUT_DIR)
        return
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_file, f, resume) for f in files]
        concurrent.futures.wait(futures)

    print(f"✅ Cleaning complete | cleaned={cleaned_count} | resume_skipped={skipped_existing} | quality_skipped={quality_skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and structure raw news articles.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files already cleaned in data/processed/news.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent workers for cleaning.",
    )
    args = parser.parse_args()
    run(resume=args.resume, workers=args.workers)
