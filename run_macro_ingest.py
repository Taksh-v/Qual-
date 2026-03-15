"""
run_macro_ingest.py
--------------------
Fetch FRED economic data, store to SQLite, and export RAG chunks.

Run periodically (e.g., daily) to keep macro context fresh.
The generated chunks are picked up by run_embedding_index.py automatically
since data/chunks/macro/macro_context_chunks.json is in CHUNKS_DIRS.

Usage:
    .venv/bin/python run_macro_ingest.py              # uses cache (6h TTL)
    .venv/bin/python run_macro_ingest.py --force      # force refresh all series
    .venv/bin/python run_macro_ingest.py --print      # print current indicators to stdout
"""

import argparse
import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FRED macro data and build RAG chunks")
    parser.add_argument("--force", action="store_true", help="Force re-fetch even if data is fresh")
    parser.add_argument("--print", dest="print_summary", action="store_true",
                        help="Print current macro indicators to stdout")
    parser.add_argument("--chunks-only", action="store_true",
                        help="Skip fetching; just rebuild RAG chunks from existing DB")
    args = parser.parse_args()

    from ingestion.macro_data import (
        fetch_and_store_macro_data,
        build_macro_rag_chunks,
        get_macro_context_text,
        CHUNKS_PATH,
    )

    if not args.chunks_only:
        print("📡 Fetching FRED macroeconomic data…")
        summary = fetch_and_store_macro_data(force=args.force)
        if not summary:
            print("⚠️  No series updated. Data may be fresh (use --force to override).")
        else:
            print(f"✅ Updated {len(summary)} series")
            for sid, info in summary.items():
                delta_str = f"  Δ{info['delta']:+.3f}" if info.get("delta") is not None else ""
                print(f"   {info['name']:<35} {info['latest_value']:.3f} {info['unit']}"
                      f"  ({info['latest_date']}){delta_str}")

    print("\n📑 Building macro RAG chunks…")
    chunks = build_macro_rag_chunks()
    print(f"✅ {len(chunks)} RAG chunks → {CHUNKS_PATH}")

    if args.print_summary:
        print("\n" + get_macro_context_text())

    print("\n🎯 Done. Re-run run_embedding_index.py to embed macro chunks into the vector store.")


if __name__ == "__main__":
    main()
