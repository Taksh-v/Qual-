from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import faiss

from config.vector_store import backend_recommendation, backend_status
from ingestion.metadata_store import get_store

ROOT = Path(__file__).resolve().parent
VECTOR_DIR = ROOT / "data" / "vector_db"
INDEX_PATH = VECTOR_DIR / "news.index"
METADATA_JSON_PATH = VECTOR_DIR / "metadata.json"
METADATA_DB_PATH = VECTOR_DIR / "metadata.db"
REPORT_PATH = VECTOR_DIR / "vector_store_audit.json"


def _timestamp_present(item: dict[str, Any]) -> bool:
    md = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return bool(
        (md.get("indexed_at") or "").strip()
        or (md.get("extracted_at") or "").strip()
        or (md.get("date") or "").strip()
        or (item.get("indexed_at") or "").strip()
        or (item.get("extracted_at") or "").strip()
        or (item.get("date") or "").strip()
    )


def _json_field_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "text_chunk_present_ratio": 0.0,
            "metadata_present_ratio": 0.0,
            "timestamp_present_ratio": 0.0,
        }

    text_ok = 0
    metadata_ok = 0
    timestamp_ok = 0

    for row in rows:
        text = (row.get("text") or "").strip()
        md = row.get("metadata") if isinstance(row.get("metadata"), dict) else None

        if text:
            text_ok += 1
        if md is not None:
            metadata_ok += 1
        if _timestamp_present(row):
            timestamp_ok += 1

    return {
        "total": total,
        "text_chunk_present_ratio": round(text_ok / total, 4),
        "metadata_present_ratio": round(metadata_ok / total, 4),
        "timestamp_present_ratio": round(timestamp_ok / total, 4),
    }


def _sqlite_summary(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"exists": False}

    conn = sqlite3.connect(db_path)
    try:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone() is not None

        if not table_exists:
            return {"exists": True, "chunks_table": False}

        cols = [r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()]
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        timestamp_expr = "COALESCE(indexed_at, extracted_at, date, '')"
        ts_count = conn.execute(
            f"SELECT COUNT(*) FROM chunks WHERE TRIM({timestamp_expr}) <> ''"
        ).fetchone()[0]

        return {
            "exists": True,
            "chunks_table": True,
            "row_count": int(count),
            "columns": cols,
            "has_indexed_at": "indexed_at" in cols,
            "timestamp_present_ratio": round((ts_count / count), 4) if count else 0.0,
        }
    finally:
        conn.close()


def main() -> int:
    backend = backend_status()
    report: dict[str, Any] = {
        "backend": backend,
        "recommendation": backend_recommendation(),
        "vector_store": {},
    }

    if backend["backend"] != "faiss":
        report["vector_store"] = {
            "active_backend_supported_by_current_pipeline": False,
            "note": "Current code path is optimized for FAISS + SQLite metadata. Non-FAISS backends require adapter integration.",
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        print(f"\nReport saved to: {REPORT_PATH}")
        return 0

    if not INDEX_PATH.exists() or not METADATA_JSON_PATH.exists():
        print("❌ Missing FAISS index or metadata.json. Run run_embedding_index.py first.")
        return 1

    # Ensure metadata schema migrations are applied (e.g., indexed_at column).
    get_store()

    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        metadata_rows = json.load(f)

    json_cov = _json_field_coverage(metadata_rows if isinstance(metadata_rows, list) else [])
    sqlite_info = _sqlite_summary(METADATA_DB_PATH)

    report["vector_store"] = {
        "active_backend_supported_by_current_pipeline": True,
        "faiss": {
            "index_path": str(INDEX_PATH),
            "vector_count": int(index.ntotal),
            "dimension": int(index.d),
        },
        "metadata_json": {
            "path": str(METADATA_JSON_PATH),
            "row_count": len(metadata_rows) if isinstance(metadata_rows, list) else 0,
            "field_coverage": json_cov,
        },
        "metadata_sqlite": sqlite_info,
        "consistency": {
            "faiss_vs_metadata_json_match": int(index.ntotal) == (len(metadata_rows) if isinstance(metadata_rows, list) else 0),
            "faiss_vs_metadata_sqlite_match": int(index.ntotal) == int(sqlite_info.get("row_count", -1)) if sqlite_info.get("chunks_table") else False,
        },
        "required_fields_expected": [
            "embedding_vector (FAISS index)",
            "text_chunk (metadata.json / metadata.db chunks.text)",
            "metadata (metadata.* payload)",
            "timestamp (indexed_at or extracted_at or date)",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("✅ Vector store audit complete")
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
