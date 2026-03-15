"""
run_migrate_metadata.py
------------------------
One-time migration: import the existing metadata.json into the new SQLite
metadata store and build the BM25 index from it.

Run once after upgrading to the new pipeline:
    python run_migrate_metadata.py

After this, run_embedding_index.py will maintain the SQLite DB + BM25 index
automatically on every subsequent build / resume run.
"""

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

METADATA_JSON = os.path.join(BASE_DIR, "data", "vector_db", "metadata.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate metadata.json → SQLite + BM25")
    parser.add_argument(
        "--metadata-path",
        default=METADATA_JSON,
        help="Path to the existing metadata.json file (default: data/vector_db/metadata.json)",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip BM25 index build (useful on memory-constrained systems).",
    )
    args = parser.parse_args()

    # ── 1. Import into SQLite ──────────────────────────────────────────────────
    print(f"\n📂 Importing metadata from: {args.metadata_path}")
    if not os.path.exists(args.metadata_path):
        print(f"❌ File not found: {args.metadata_path}")
        print("   If you haven't built the FAISS index yet, run run_embedding_index.py first.")
        sys.exit(1)

    from ingestion.metadata_store import get_store
    store = get_store()
    n = store.import_json(args.metadata_path)
    print(f"✅ SQLite import complete  — {n} rows written")
    print(f"   DB path: {store.db_path}")

    # Sanity check: count and print a sentiment summary
    total = store.count()
    print(f"   Total rows in DB: {total}")

    if total > 0:
        summary = store.sentiment_summary()
        print(f"\n📊 Overall sentiment (all chunks):")
        print(f"   Label  : {summary['overall_label']}")
        print(f"   Score  : {summary['avg_score']:+.4f}")
        print(f"   pos={summary['positive']}  neg={summary['negative']}  neu={summary['neutral']}")

    # ── 2. Build BM25 index ────────────────────────────────────────────────────
    if args.skip_bm25:
        print("\n⏭️  BM25 build skipped (--skip-bm25 flag set).")
        return

    print("\n📑 Building BM25 sparse index...")
    try:
        from ingestion.bm25_index import build_bm25_index

        # Load chunks from JSON (already in memory via import above if small; reload to be safe)
        with open(args.metadata_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        bm25_idx = build_bm25_index(chunks)
        print(f"✅ BM25 index built over {len(chunks)} chunks")

        # Quick smoke test
        results = bm25_idx.search("Fed interest rate hike inflation", top_k=3)
        if results:
            print(f"\n🔍 BM25 smoke test — top 3 results for 'Fed interest rate hike inflation':")
            for rank, (chunk, score) in enumerate(results, start=1):
                md = chunk.get("metadata", {})
                title = md.get("title", chunk.get("title", ""))[:60]
                print(f"   [{rank}] score={score:.3f} | {title}")
        else:
            print("   (No results — index may be empty)")

    except ImportError:
        print("⚠️  rank-bm25 not installed. Run: pip install rank-bm25")
        print("   BM25 index not built.")

    print("\n🎉 Migration complete. The system will now use SQLite + BM25 on next query.")


if __name__ == "__main__":
    main()
