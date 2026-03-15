import json
import os
import requests
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data/vector_db/metadata.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/vector_db/metadata_with_entities.json")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("ENTITY_EXTRACT_MODEL", "phi3:mini")
TIMEOUT = int(os.getenv("ENTITY_EXTRACT_TIMEOUT_SEC", "60"))

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
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 200}},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        output = (resp.json().get("response") or "").strip()
        # Extract first JSON object from potentially noisy output
        start = output.find("{")
        end   = output.rfind("}")
        if start != -1 and end != -1:
            return json.loads(output[start:end + 1])
    except Exception as e:
        print(f"⚠ Extraction failed: {e}")
    return {"companies": [], "indices": [], "sectors": [], "macros": []}


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
