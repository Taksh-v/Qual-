import json
import os
import argparse
import re
from typing import List, Dict

from config.sources import NEWS_SOURCES
from ingestion.news_extractor import extract_news

OUTPUT_DIR = "data/raw/news"
URL_QUALITY_PATTERNS = [
    r"\b(market|markets|economy|economic|inflation|cpi|gdp|fed|federal-reserve|interest-rate|yield|bond|currency|dollar|fx|forex|gold|silver|oil|commodity|stocks|equity|earnings|trade|tariff|exports|imports|fmcg|banking|finance)\b",
]


def _is_relevant_url(url: str) -> bool:
    u = (url or "").lower()
    return any(re.search(p, u) for p in URL_QUALITY_PATTERNS)


def ingest_news(resume: bool = False, strict_source_filter: bool = True) -> List[Dict]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    collected: List[Dict] = []
    skipped_existing = 0
    skipped_irrelevant = 0

    for idx, url in enumerate(NEWS_SOURCES):
        if strict_source_filter and not _is_relevant_url(url):
            skipped_irrelevant += 1
            print(f"⏭️  Source filter skip (low relevance): {url}")
            continue
        output_path = os.path.join(OUTPUT_DIR, f"article_{idx}.json")
        if resume and os.path.exists(output_path):
            skipped_existing += 1
            print(f"⏭️  Resume skip (already ingested): {output_path}")
            continue
        try:
            article = extract_news(url)
            collected.append(article)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(article, f, indent=2, ensure_ascii=False)
            print(f"✅ Ingested: {url}")
        except Exception as exc:
            print(f"⚠️ Skipped: {url} | {exc}")

    print(
        f"\n📦 Ingestion completed. Saved {len(collected)} new articles to {OUTPUT_DIR}"
        f" | resume_skipped={skipped_existing} | relevance_skipped={skipped_irrelevant}"
    )
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and store raw news articles.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip URLs whose output file already exists.",
    )
    parser.add_argument(
        "--no-strict-source-filter",
        action="store_true",
        help="Disable URL relevance filtering before ingestion.",
    )
    args = parser.parse_args()
    ingest_news(
        resume=args.resume,
        strict_source_filter=not args.no_strict_source_filter,
    )


if __name__ == "__main__":
    main()
