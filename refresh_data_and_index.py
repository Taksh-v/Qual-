import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(name: str, command: list[str]) -> None:
    start = time.time()
    print(f"\n▶ {name}")
    print(f"$ {' '.join(command)}")

    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name} (exit code {result.returncode})")

    elapsed = time.time() - start
    print(f"✅ Completed: {name} ({elapsed:.1f}s)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh custom data artifacts for retrieval/macro engine by running "
            "cleaning, chunking, and embedding-index build."
        )
    )
    parser.add_argument(
        "--with-ingestion",
        action="store_true",
        help="Run run_news_ingestion.py before cleaning/chunking/indexing.",
    )
    args = parser.parse_args()

    pipeline: list[tuple[str, list[str]]] = []

    if args.with_ingestion:
        pipeline.append(("News ingestion", [sys.executable, "run_news_ingestion.py"]))

    pipeline.extend(
        [
            ("Cleaning raw articles", [sys.executable, "run_cleaning.py"]),
            ("Chunking cleaned articles", [sys.executable, "run_chunking.py"]),
            ("Building embedding index", [sys.executable, "run_embedding_index.py"]),
        ]
    )

    print("🚀 Starting refresh pipeline")
    for step_name, step_cmd in pipeline:
        run_step(step_name, step_cmd)

    print(
        "\n📦 Refresh complete. Your latest custom data should now be reflected in:\n"
        "- data/chunks/news/*\n"
        "- data/vector_db/metadata.json\n"
        "- data/vector_db/news.index\n"
        "\nYou can now run intelligence/macro_engine.py against updated context."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n❌ Pipeline failed: {exc}")
        raise SystemExit(1)
