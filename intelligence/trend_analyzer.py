import os
import json
import subprocess
from collections import Counter
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH = os.path.join(BASE_DIR, "data", "vector_db", "metadata.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "intelligence_trends.json")

LLM_MODEL = "phi3:mini"


PROMPT_TEMPLATE = """
You are a financial market analyst.

From the news text below, extract:
1. Companies mentioned
2. Sector (if any)
3. Overall sentiment: bullish, bearish, or neutral
4. Key market themes (comma separated)

Return STRICT JSON only in this format:
{{
  "companies": [],
  "sector": "",
  "sentiment": "",
  "themes": []
}}

News:
{text}
"""



def call_llm(prompt: str) -> dict | None:
    try:
        result = subprocess.run(
            ["ollama", "run", LLM_MODEL],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=90
        )

        if result.returncode != 0:
            return None

        # Extract first JSON block safely
        output = result.stdout.strip()
        start = output.find("{")
        end = output.rfind("}")
        if start == -1 or end == -1:
            return None

        return json.loads(output[start:end + 1])

    except Exception:
        return None


def main():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("metadata.json not found")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    company_counter = Counter()
    sector_counter = Counter()
    sentiment_counter = Counter()
    theme_counter = Counter()

    print("\n📊 Extracting market signals...\n")

    for chunk in tqdm(metadata):
        text = chunk.get("text", "")
        if not text:
            continue

        prompt = PROMPT_TEMPLATE.format(text=text[:1500])
        data = call_llm(prompt)

        if not data:
            continue

        for c in data.get("companies", []):
            company_counter[c] += 1

        if data.get("sector"):
            sector_counter[data["sector"]] += 1

        if data.get("sentiment"):
            sentiment_counter[data["sentiment"]] += 1

        for t in data.get("themes", []):
            theme_counter[t] += 1

    trends = {
        "top_companies": company_counter.most_common(10),
        "top_sectors": sector_counter.most_common(5),
        "sentiment_distribution": dict(sentiment_counter),
        "top_themes": theme_counter.most_common(10)
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(trends, f, indent=2)

    print("\n✅ Market intelligence generated:")
    print(json.dumps(trends, indent=2))


if __name__ == "__main__":
    main()
