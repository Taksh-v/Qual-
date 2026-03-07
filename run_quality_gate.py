from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent


def _run_step(name: str, cmd: list[str], *, allow_fail: bool = False) -> bool:
    print(f"\n▶ {name}")
    print(f"$ {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"✅ {name} ({elapsed:.1f}s)")
        return True
    print(f"❌ {name} failed with exit code {result.returncode} ({elapsed:.1f}s)")
    return allow_fail


def _ollama_reachable(url: str, timeout_sec: float = 3.0) -> bool:
    try:
        response = requests.get(f"{url.rstrip('/')}/api/tags", timeout=timeout_sec)
        return response.status_code == 200
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run production-style quality gate for data and RAG response accuracy."
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip run_data_quality_audit.py.",
    )
    parser.add_argument(
        "--skip-rag-eval",
        action="store_true",
        help="Skip run_rag_eval.py.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if Ollama is unavailable when RAG eval is enabled.",
    )
    parser.add_argument(
        "--run-pytest",
        action="store_true",
        help="Run pytest in addition to quality gates.",
    )
    args = parser.parse_args()

    steps_ok = True

    steps_ok = _run_step(
        "Python compile check",
        [sys.executable, "-m", "py_compile", "rag/query.py", "run_data_quality_audit.py", "run_rag_eval.py"],
    ) and steps_ok

    if args.run_pytest:
        steps_ok = _run_step("Pytest", [sys.executable, "-m", "pytest", "-q"]) and steps_ok

    if not args.skip_audit:
        steps_ok = _run_step("Data quality audit", [sys.executable, "run_data_quality_audit.py"]) and steps_ok

    if not args.skip_rag_eval:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        reachable = _ollama_reachable(ollama_url)
        if not reachable:
            msg = (
                f"Ollama not reachable at {ollama_url}. "
                "RAG evaluation requires embedding/generation access."
            )
            if args.strict:
                print(f"❌ {msg}")
                return 1
            print(f"⚠️  {msg} Skipping RAG eval (non-strict mode).")
        else:
            steps_ok = _run_step("RAG evaluation", [sys.executable, "run_rag_eval.py"]) and steps_ok

    if steps_ok:
        print("\n✅ Quality gate passed.")
        return 0
    print("\n❌ Quality gate failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
