import os
import json
from ingestion.chunker import chunk_text

INPUT_DIR = "data/processed/news"
OUTPUT_DIR = "data/chunks/news"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run():
    for file in os.listdir(INPUT_DIR):
        if not file.endswith("_clean.json"):
            continue

        with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
            article = json.load(f)

        text = article["structured_text"]
        chunks = chunk_text(text)

        chunked_data = []
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "chunk_id": f"{file}_{i}",
                "text": chunk,
                "metadata": article["metadata"]
            })

        out_file = file.replace("_clean.json", "_chunks.json")
        with open(os.path.join(OUTPUT_DIR, out_file), "w", encoding="utf-8") as f:
            json.dump(chunked_data, f, indent=2)

        print(f"✅ Chunked: {out_file}")

if __name__ == "__main__":
    run()
