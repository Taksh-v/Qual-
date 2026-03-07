import os
import json
import argparse
from ingestion.cleaner import structure_article

INPUT_DIR = "data/raw/news"
OUTPUT_DIR = "data/processed/news"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run(resume: bool = False):
    cleaned_count = 0
    skipped_existing = 0
    for file in sorted(os.listdir(INPUT_DIR)):
        if not file.endswith(".json"):
            continue
        output_file = file.replace(".json", "_clean.json")
        output_path = os.path.join(OUTPUT_DIR, output_file)
        if resume and os.path.exists(output_path):
            skipped_existing += 1
            print(f"⏭️  Resume skip (already cleaned): {output_file}")
            continue

        with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
            article = json.load(f)

        structured = structure_article(article)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2)

        print(f"Cleaned: {output_file}")
        cleaned_count += 1

    print(f"✅ Cleaning complete | cleaned={cleaned_count} | resume_skipped={skipped_existing}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and structure raw news articles.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files already cleaned in data/processed/news.",
    )
    args = parser.parse_args()
    run(resume=args.resume)
