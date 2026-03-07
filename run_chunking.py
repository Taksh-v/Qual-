import os
import json
import argparse
from ingestion.chunker import chunk_text

INPUT_DIR = "data/processed/news"
OUTPUT_DIR = "data/chunks/news"
MIN_CHUNK_WORDS = int(os.getenv("MIN_CHUNK_WORDS", "45"))
MAX_CHUNK_WORDS = int(os.getenv("MAX_CHUNK_WORDS", "700"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run(resume: bool = False):
    chunked_count = 0
    skipped_existing = 0
    for file in sorted(os.listdir(INPUT_DIR)):
        if not file.endswith("_clean.json"):
            continue
        out_file = file.replace("_clean.json", "_chunks.json")
        out_path = os.path.join(OUTPUT_DIR, out_file)
        if resume and os.path.exists(out_path):
            skipped_existing += 1
            print(f"⏭️  Resume skip (already chunked): {out_file}")
            continue

        with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
            article = json.load(f)

        text = article["structured_text"]
        chunks = chunk_text(text)
        cleaned_chunks = []
        seen = set()
        for c in chunks:
            c_norm = " ".join(c.split())
            wc = len(c_norm.split())
            if wc < MIN_CHUNK_WORDS or wc > MAX_CHUNK_WORDS:
                continue
            key = c_norm[:220].lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned_chunks.append(c_norm)

        chunked_data = []
        for i, chunk in enumerate(cleaned_chunks):
            chunked_data.append({
                "chunk_id": f"{file}_{i}",
                "text": chunk,
                "metadata": article["metadata"]
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunked_data, f, indent=2)

        print(f"✅ Chunked: {out_file}")
        chunked_count += 1

    print(f"✅ Chunking complete | chunked={chunked_count} | resume_skipped={skipped_existing}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk structured news into retrieval chunks.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files already chunked in data/chunks/news.",
    )
    args = parser.parse_args()
    run(resume=args.resume)
