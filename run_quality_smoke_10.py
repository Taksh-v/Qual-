import json
import os
from pathlib import Path
from statistics import median

import faiss
import numpy as np

from config.sources import NEWS_SOURCES
from ingestion.chunker import chunk_text
from ingestion.cleaner import structure_article
from ingestion.embeddings import get_embedding
from ingestion.news_extractor import extract_news
from intelligence.data_quality import evaluate_vector_store_health


def run(limit: int = 10, from_existing_raw: bool = False) -> int:
    root = Path("/tmp/qual_smoke_10")
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    chunks_dir = root / "chunks"
    vector_dir = root / "vector_db"
    for d in (raw_dir, processed_dir, chunks_dir, vector_dir):
        d.mkdir(parents=True, exist_ok=True)

    ingested = 0
    requested = limit
    if from_existing_raw:
        src_raw_dir = Path("data/raw/news")
        candidates = sorted(src_raw_dir.glob("*.json"))[:limit]
        requested = len(candidates)
        for idx, src in enumerate(candidates):
            dst = raw_dir / f"article_{idx}.json"
            with open(src, "r", encoding="utf-8") as f:
                payload = json.load(f)
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            ingested += 1
            print(f"✅ Copied local raw file {idx + 1}/{len(candidates)}")
    else:
        selected = NEWS_SOURCES[:limit]
        requested = len(selected)
        for idx, url in enumerate(selected):
            try:
                article = extract_news(url)
                with open(raw_dir / f"article_{idx}.json", "w", encoding="utf-8") as f:
                    json.dump(article, f, indent=2, ensure_ascii=False)
                ingested += 1
                print(f"✅ Ingested {idx + 1}/{len(selected)}")
            except Exception as exc:
                print(f"⚠️ Ingestion failed for {url}: {exc}")

    cleaned = []
    for file in sorted(raw_dir.glob("*.json")):
        with open(file, "r", encoding="utf-8") as f:
            article = json.load(f)
        structured = structure_article(article)
        out = processed_dir / file.name.replace(".json", "_clean.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2)
        cleaned.append(out)
    print(f"✅ Cleaned files: {len(cleaned)}")

    chunk_records = []
    for file in cleaned:
        with open(file, "r", encoding="utf-8") as f:
            article = json.load(f)
        chunks = chunk_text(article["structured_text"])
        filtered = []
        seen = set()
        for c in chunks:
            c_norm = " ".join(c.split())
            wc = len(c_norm.split())
            if wc < 45 or wc > 700:
                continue
            key = c_norm[:220].lower()
            if key in seen:
                continue
            seen.add(key)
            filtered.append(c_norm)
        payload = []
        for i, c in enumerate(filtered):
            rec = {
                "chunk_id": f"{file.name}_{i}",
                "text": c,
                "metadata": article["metadata"],
            }
            payload.append(rec)
            chunk_records.append(rec)
        out = chunks_dir / file.name.replace("_clean.json", "_chunks.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    print(f"✅ Total chunks: {len(chunk_records)}")

    index = None
    metadata_store = []
    skipped = 0
    for rec in chunk_records:
        try:
            emb = get_embedding(rec["text"], role="passage")
            if index is None:
                index = faiss.IndexFlatIP(len(emb))
            index.add(np.expand_dims(emb, axis=0))
            metadata_store.append(rec)
        except Exception as exc:
            skipped += 1
            print(f"⚠️ Embedding skipped {rec['chunk_id']}: {exc}")

    if index is None:
        print("❌ No vectors created. Is Ollama embedding model available?")
        return 1

    index_path = vector_dir / "news.index"
    meta_path = vector_dir / "metadata.json"
    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=2)

    health = evaluate_vector_store_health(index.ntotal, metadata_store)
    lengths = [len((m.get("text") or "").split()) for m in metadata_store if (m.get("text") or "").strip()]
    report = {
        "ingested_requested": requested,
        "ingested_success": ingested,
        "cleaned_count": len(cleaned),
        "chunk_count": len(chunk_records),
        "embedded_count": index.ntotal,
        "embedding_skipped": skipped,
        "vector_store_health": health,
        "chunk_word_stats": {
            "count": len(lengths),
            "min": min(lengths) if lengths else 0,
            "median": int(median(lengths)) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "paths": {
            "root": str(root),
            "index": str(index_path),
            "metadata": str(meta_path),
        },
    }
    report_path = vector_dir / "quality_report_10.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n✅ 10-link smoke test complete")
    print(json.dumps(report, indent=2))
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run isolated quality test on 10 items.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--from-existing-raw",
        action="store_true",
        help="Use first N files from data/raw/news instead of downloading URLs.",
    )
    args = parser.parse_args()
    raise SystemExit(run(args.limit, from_existing_raw=args.from_existing_raw))
