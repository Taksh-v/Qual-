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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume mode: skip already-processed files and continue index build from existing artifacts.",
    )
    parser.add_argument(
        "--with-audit",
        action="store_true",
        help="Run data quality audit after index build.",
    )
    parser.add_argument(
        "--with-rag-eval",
        action="store_true",
        help="Run retrieval/grounding evaluation and industry pass/fail gate after index build.",
    )
    parser.add_argument(
        "--no-strict-source-filter",
        action="store_true",
        help="Disable relevance filtering during ingestion.",
    )
    args = parser.parse_args()

    pipeline: list[tuple[str, list[str]]] = []

    def cmd(script: str, is_ingestion: bool = False) -> list[str]:
        c = [sys.executable, script]
        if args.resume:
            c.append("--resume")
        if is_ingestion and args.no_strict_source_filter:
            c.append("--no-strict-source-filter")
        return c

    if args.with_ingestion:
        pipeline.append(("News ingestion", cmd("run_news_ingestion.py", is_ingestion=True)))

    pipeline.extend(
        [
            ("Cleaning raw articles", cmd("run_cleaning.py")),
            ("Chunking cleaned articles", cmd("run_chunking.py")),
            ("Building embedding index", cmd("run_embedding_index.py")),
        ]
    )
    if args.with_audit:
        pipeline.append(("Data quality audit", [sys.executable, "run_data_quality_audit.py"]))
    if args.with_rag_eval:
        pipeline.append(("RAG evaluation", [sys.executable, "run_rag_eval.py"]))

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
