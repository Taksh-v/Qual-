import os
import json
from ingestion.cleaner import structure_article

INPUT_DIR = "data/raw/news"
OUTPUT_DIR = "data/processed/news"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run():
    for file in os.listdir(INPUT_DIR):
        with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
            article = json.load(f)

        structured = structure_article(article)

        output_file = file.replace(".json", "_clean.json")
        with open(os.path.join(OUTPUT_DIR, output_file), "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2)

        print(f"Cleaned: {output_file}")

if __name__ == "__main__":
    run()
