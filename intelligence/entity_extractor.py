import json
import subprocess
import os
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data/vector_db/metadata.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/vector_db/metadata_with_entities.json")

MODEL = "phi3:mini"

PROMPT_TEMPLATE = """
You are a financial information extraction system.

Extract ONLY financial entities from the text below.

Rules:
- Output ONLY valid JSON
- No explanations
- No markdown
- No extra text
- Use empty arrays if none found

Schema:
{{
  "companies": [],
  "indices": [],
  "sectors": [],
  "macros": []
}}

TEXT:
{text}
"""


def extract_entities(text: str) -> dict:
    text = text[:600]

    prompt = PROMPT_TEMPLATE.format(text=text)

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60
        )

        output = result.stdout.strip()

        return json.loads(output)

    except Exception as e:
        print(f"⚠ Extraction failed: {e}")
        return {
            "companies": [],
            "indices": [],
            "sectors": [],
            "macros": []
        }


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    enriched = []

    for i, chunk in enumerate(tqdm(chunks, desc="Extracting entities")):
        entities = extract_entities(chunk["text"])
        chunk["entities"] = entities
        enriched.append(chunk)

        if i % 25 == 0:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(enriched, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    print("✅ Entity extraction complete")


if __name__ == "__main__":
    main()
