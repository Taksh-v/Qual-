from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import requests

from ingestion.metadata_store import get_store
from intelligence.live_market_data import invalidate_live_data_cache, market_status_summary
from rag.rag_core import ask_rag
from rag.query import invalidate_index_cache, load_index, load_metadata
from run_scheduler import (
    run_fundamental_ingestion,
    run_index_rebuild,
    run_insider_ingestion,
    run_macro_ingestion,
    run_rss_ingestion,
    run_sec_ingestion,
    run_sector_ingestion,
    run_transcript_ingestion,
)


ADMIN_ACTIONS: dict[str, dict[str, Any]] = {
    "reload_index": {"description": "Invalidate in-process FAISS and live-data caches", "kind": "write"},
    "scheduler_job": {
        "description": "Run one ingestion/index scheduler job",
        "kind": "write",
        "params": {"job": "rss|sec|fundamentals|macro|sector|insider|transcripts|index|ingest_all"},
    },
    "quality_gate": {
        "description": "Run production quality gate script",
        "kind": "write",
        "params": {"strict": "bool", "run_pytest": "bool", "skip_audit": "bool", "skip_rag_eval": "bool"},
    },
    "rag_eval": {
        "description": "Run retrieval/grounding evaluation suite",
        "kind": "write",
        "params": {"config": "path", "report": "path", "max_queries": "int"},
    },
    "vector_audit": {"description": "Run vector-store consistency audit", "kind": "write"},
    "data_quality_audit": {"description": "Run vector-store data-quality audit", "kind": "write"},
}


def list_admin_actions() -> dict[str, Any]:
    return {"actions": ADMIN_ACTIONS}


async def rag_query(question: str) -> dict[str, Any]:
    return await ask_rag(question)


def system_health() -> dict[str, Any]:
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_ok = False
    try:
        response = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=3)
        ollama_ok = response.status_code == 200
    except Exception:
        ollama_ok = False

    index_ok = False
    index_vectors = 0
    metadata_rows = 0
    try:
        index = load_index()
        metadata = load_metadata()
        index_vectors = int(index.ntotal)
        metadata_rows = len(metadata)
        index_ok = index_vectors > 0 and metadata_rows > 0
    except Exception:
        index_ok = False

    return {
        "status": "ok" if ollama_ok and index_ok else "degraded",
        "ollama": {"reachable": ollama_ok, "url": ollama_url},
        "vector_store": {
            "index_loaded": index_ok,
            "vectors": index_vectors,
            "metadata_rows": metadata_rows,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def market_status() -> dict[str, Any]:
    return market_status_summary()


def metadata_query(
    *,
    sector: str | None = None,
    region: str | None = None,
    data_type: str | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
    source: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    safe_limit = max(1, min(int(limit), 100))
    rows = get_store().query(
        sector=sector,
        region=region,
        data_type=data_type,
        min_date=min_date,
        max_date=max_date,
        source=source,
        limit=safe_limit,
    )
    return {
        "count": len(rows),
        "rows": rows,
    }


def metadata_sentiment_summary(
    *,
    sector: str | None = None,
    min_date: str | None = None,
) -> dict[str, Any]:
    return get_store().sentiment_summary(sector=sector, min_date=min_date)


def _run_script(args: list[str], timeout_sec: int) -> dict[str, Any]:
    started = time.time()
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout_sec)
    elapsed_ms = int((time.time() - started) * 1000)
    return {
        "command": " ".join(args),
        "returncode": result.returncode,
        "elapsed_ms": elapsed_ms,
        "stdout_tail": (result.stdout or "")[-4000:],
        "stderr_tail": (result.stderr or "")[-4000:],
        "ok": result.returncode == 0,
    }


def _scheduler_job(job: str) -> dict[str, Any]:
    mapping = {
        "rss": run_rss_ingestion,
        "sec": run_sec_ingestion,
        "fundamentals": run_fundamental_ingestion,
        "macro": run_macro_ingestion,
        "sector": run_sector_ingestion,
        "insider": run_insider_ingestion,
        "transcripts": run_transcript_ingestion,
        "index": run_index_rebuild,
    }
    if job == "ingest_all":
        order = ["rss", "sec", "fundamentals", "macro", "sector", "insider", "transcripts", "index"]
        outcomes: list[dict[str, Any]] = []
        overall = True
        for name in order:
            ok = bool(mapping[name]())
            outcomes.append({"job": name, "ok": ok})
            overall = overall and ok
        return {"ok": overall, "results": outcomes}

    if job not in mapping:
        raise ValueError("unknown scheduler job")

    ok = bool(mapping[job]())
    return {"ok": ok, "job": job}


def _quality_gate(params: dict[str, Any]) -> dict[str, Any]:
    cmd = [sys.executable, "run_quality_gate.py"]
    if params.get("strict"):
        cmd.append("--strict")
    if params.get("run_pytest"):
        cmd.append("--run-pytest")
    if params.get("skip_audit"):
        cmd.append("--skip-audit")
    if params.get("skip_rag_eval"):
        cmd.append("--skip-rag-eval")
    return _run_script(cmd, timeout_sec=5400)


def _rag_eval(params: dict[str, Any]) -> dict[str, Any]:
    cmd = [sys.executable, "run_rag_eval.py"]
    if params.get("config"):
        cmd.extend(["--config", str(params["config"])])
    if params.get("report"):
        cmd.extend(["--report", str(params["report"])])
    if params.get("max_queries") is not None:
        cmd.extend(["--max-queries", str(int(params["max_queries"]))])
    return _run_script(cmd, timeout_sec=5400)


def execute_admin_action(action: str, params: dict[str, Any]) -> dict[str, Any]:
    if action == "reload_index":
        invalidate_index_cache()
        invalidate_live_data_cache()
        return {
            "ok": True,
            "message": "index and live-data caches invalidated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    if action == "scheduler_job":
        job = str(params.get("job", "")).strip().lower()
        if not job:
            raise ValueError("scheduler_job requires 'job'")
        return _scheduler_job(job)

    if action == "quality_gate":
        return _quality_gate(params)

    if action == "rag_eval":
        return _rag_eval(params)

    if action == "vector_audit":
        return _run_script([sys.executable, "run_vector_store_audit.py"], timeout_sec=1800)

    if action == "data_quality_audit":
        return _run_script([sys.executable, "run_data_quality_audit.py"], timeout_sec=1800)

    raise ValueError(f"unknown action '{action}'")


def sync_rag_query(question: str) -> dict[str, Any]:
    return asyncio.run(rag_query(question))
