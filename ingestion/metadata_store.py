"""
ingestion/metadata_store.py
-----------------------------
SQLite-backed metadata store replacing the flat metadata.json list.

WHY THIS EXISTS
---------------
The flat JSON approach loads ~2.5 MB (and growing) into RAM on every query,
supports no indexed filtering, and has O(n) scan time for sector/date filters.
SQLite gives us:
  - Indexed lookups by chunk_id, sector, region, source, date, data_type
  - Fast pre-filtering BEFORE the FAISS ANN search (cuts irrelevant results)
  - Persistent, incremental: re-ingestion adds rows without a full rebuild
  - Backward-compatible JSON export so existing code keeps working

SCHEMA
------
  chunks(
    rowid        INTEGER PRIMARY KEY,   -- FAISS vector index position
    chunk_id     TEXT UNIQUE,
    text         TEXT,
    source       TEXT,
    date         TEXT,
    title        TEXT,
    sector       TEXT,
    region       TEXT,
    data_type    TEXT,
    company      TEXT,
    url          TEXT,
    sentiment    REAL,                  -- FinLexicon score [-1, +1]
    sentiment_label TEXT,              -- positive / negative / neutral
    fingerprint  TEXT,
        extracted_at TEXT,
        indexed_at   TEXT                  -- vector insertion timestamp (UTC)
  )

USAGE
-----
    from ingestion.metadata_store import MetadataStore
    store = MetadataStore()                    # opens / creates DB
    store.upsert_chunks(chunk_list)            # bulk insert/update
    rows = store.query(sector="Technology", min_date="2026-03-01", limit=20)
    store.export_json("data/vector_db/metadata.json")  # backward-compat
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Generator

from intelligence.sentiment_analyzer import score_sentiment

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DB_PATH = os.path.join(_BASE_DIR, "data", "vector_db", "metadata.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    rowid        INTEGER PRIMARY KEY,
    chunk_id     TEXT UNIQUE NOT NULL,
    text         TEXT NOT NULL DEFAULT '',
    source       TEXT NOT NULL DEFAULT '',
    date         TEXT NOT NULL DEFAULT '',
    title        TEXT NOT NULL DEFAULT '',
    sector       TEXT NOT NULL DEFAULT 'Unknown',
    region       TEXT NOT NULL DEFAULT 'Global',
    data_type    TEXT NOT NULL DEFAULT '',
    company      TEXT NOT NULL DEFAULT 'Unknown',
    url          TEXT NOT NULL DEFAULT '',
    sentiment    REAL NOT NULL DEFAULT 0.0,
    sentiment_label TEXT NOT NULL DEFAULT 'neutral',
    fingerprint  TEXT NOT NULL DEFAULT '',
    extracted_at TEXT NOT NULL DEFAULT '',
    indexed_at   TEXT NOT NULL DEFAULT ''
);
"""

_REQUIRED_COLUMNS: dict[str, str] = {
    "indexed_at": "TEXT NOT NULL DEFAULT ''",
}

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sector      ON chunks(sector);",
    "CREATE INDEX IF NOT EXISTS idx_region      ON chunks(region);",
    "CREATE INDEX IF NOT EXISTS idx_data_type   ON chunks(data_type);",
    "CREATE INDEX IF NOT EXISTS idx_date        ON chunks(date);",
    "CREATE INDEX IF NOT EXISTS idx_source      ON chunks(source);",
        "CREATE INDEX IF NOT EXISTS idx_company     ON chunks(company);",
        "CREATE INDEX IF NOT EXISTS idx_fingerprint ON chunks(fingerprint);",
    "CREATE INDEX IF NOT EXISTS idx_sentiment   ON chunks(sentiment);",
    "CREATE INDEX IF NOT EXISTS idx_indexed_at  ON chunks(indexed_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_chunk_id    ON chunks(chunk_id);",
        "CREATE INDEX IF NOT EXISTS idx_type_date   ON chunks(data_type, date DESC);",
        "CREATE INDEX IF NOT EXISTS idx_region_date ON chunks(region, date DESC);",
        "CREATE INDEX IF NOT EXISTS idx_sector_date ON chunks(sector, date DESC);",
        "CREATE INDEX IF NOT EXISTS idx_source_date ON chunks(source, date DESC);",
        "CREATE INDEX IF NOT EXISTS idx_company_date ON chunks(company, date DESC);",
]

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        text,
        title,
        company,
        source,
        data_type,
        sector,
        region,
        content='chunks',
        content_rowid='rowid',
        tokenize='unicode61'
);
"""

_CREATE_FTS_TRIGGERS = [
        """
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text, title, company, source, data_type, sector, region)
            VALUES (new.rowid, new.text, new.title, new.company, new.source, new.data_type, new.sector, new.region);
        END;
        """,
        """
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, title, company, source, data_type, sector, region)
            VALUES ('delete', old.rowid, old.text, old.title, old.company, old.source, old.data_type, old.sector, old.region);
        END;
        """,
        """
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, title, company, source, data_type, sector, region)
            VALUES ('delete', old.rowid, old.text, old.title, old.company, old.source, old.data_type, old.sector, old.region);
            INSERT INTO chunks_fts(rowid, text, title, company, source, data_type, sector, region)
            VALUES (new.rowid, new.text, new.title, new.company, new.source, new.data_type, new.sector, new.region);
        END;
        """,
]


class MetadataStore:
    """
    Thread-safe SQLite-backed metadata store for RAG chunks.

    Each row corresponds to exactly one FAISS vector (same rowid / position).
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._fts_available = False
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        try:
            self._init_db()
        except sqlite3.DatabaseError as exc:
            msg = str(exc).lower()
            if "malformed" not in msg and "disk image" not in msg:
                raise
            logger.error("[MetadataStore] DB appears corrupted (%s). Attempting automatic recovery.", exc)
            self._recover_from_corruption()

    # ── Private helpers ────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")     # enable concurrent reads
        conn.execute("PRAGMA synchronous=NORMAL;")   # safe but fast
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-64000;")    # ~64MB page cache
        conn.execute("PRAGMA mmap_size=268435456;")  # 256MB mmap when supported
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE)
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            for col_name, col_def in _REQUIRED_COLUMNS.items():
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE chunks ADD COLUMN {col_name} {col_def}")

            # Backfill indexed_at for legacy rows that predate this column.
            conn.execute(
                """
                UPDATE chunks
                SET indexed_at = COALESCE(
                    NULLIF(TRIM(extracted_at), ''),
                    NULLIF(TRIM(date), ''),
                    strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                )
                WHERE TRIM(COALESCE(indexed_at, '')) = ''
                """
            )

            for idx_sql in _CREATE_INDEXES:
                conn.execute(idx_sql)
            try:
                conn.execute(_CREATE_FTS)
                for trig_sql in _CREATE_FTS_TRIGGERS:
                    conn.execute(trig_sql)
                self._fts_available = True

                row_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                if row_count:
                    fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
                    if fts_count < row_count:
                        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
            except sqlite3.OperationalError as exc:
                self._fts_available = False
                logger.warning("[MetadataStore] FTS5 unavailable, continuing without text index: %s", exc)
        logger.info("[MetadataStore] Initialized DB at %s", self.db_path)

    def _recover_from_corruption(self) -> None:
        base = self.db_path
        backup = f"{base}.corrupt.{int(time.time())}"

        if os.path.exists(base):
            try:
                os.replace(base, backup)
                logger.warning("[MetadataStore] Backed up corrupted DB to %s", backup)
            except OSError as exc:
                logger.error("[MetadataStore] Failed to back up corrupted DB: %s", exc)

        for sidecar in (f"{base}-wal", f"{base}-shm"):
            if os.path.exists(sidecar):
                try:
                    os.remove(sidecar)
                except OSError:
                    pass

        self._fts_available = False
        self._init_db()

        metadata_json_path = os.path.join(os.path.dirname(base), "metadata.json")
        if os.path.exists(metadata_json_path):
            try:
                restored = self.import_json(metadata_json_path)
                logger.info("[MetadataStore] Recovered DB from metadata.json with %d rows", restored)
            except Exception as exc:
                logger.error("[MetadataStore] Recovery import failed: %s", exc)

    @staticmethod
    def _extract_fields(chunk: dict[str, Any]) -> dict[str, Any]:
        """Flatten a chunk dict (which may have a nested 'metadata' sub-dict)."""
        md: dict[str, Any] = chunk.get("metadata") or {}
        if not isinstance(md, dict):
            md = {}

        text = chunk.get("text") or ""

        # Run sentiment scoring using the existing finance lexicon
        sent = score_sentiment(text)

        return {
            "chunk_id":       chunk.get("chunk_id") or "",
            "text":           text,
            "source":         md.get("source") or chunk.get("source") or "",
            "date":           md.get("date") or chunk.get("date") or "",
            "title":          md.get("title") or chunk.get("title") or "",
            "sector":         chunk.get("sector") or md.get("sector") or "Unknown",
            "region":         chunk.get("region") or md.get("region") or "Global",
            "data_type":      md.get("data_type") or chunk.get("data_type") or "",
            "company":        md.get("company") or chunk.get("company") or "Unknown",
            "url":            md.get("url") or chunk.get("url") or "",
            "sentiment":      sent["score"],
            "sentiment_label": sent["label"],
            "fingerprint":    chunk.get("fingerprint") or "",
            "extracted_at":   md.get("extracted_at") or chunk.get("extracted_at") or "",
            "indexed_at":     md.get("indexed_at") or chunk.get("indexed_at") or md.get("extracted_at") or chunk.get("extracted_at") or md.get("date") or chunk.get("date") or "",
        }

    # ── Public write API ───────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[dict[str, Any]], start_rowid: int = 0) -> int:
        """
        Insert or replace chunks in bulk.

        Args:
            chunks:      List of chunk dicts (from chunker / run_embedding_index).
            start_rowid: The FAISS index position of the first chunk in this batch.
                         Subsequent chunks get start_rowid+1, start_rowid+2, etc.

        Returns:
            Number of rows inserted/updated.
        """
        if not chunks:
            return 0

        rows = []
        for i, chunk in enumerate(chunks):
            fields = self._extract_fields(chunk)
            fields["rowid"] = start_rowid + i
            rows.append(fields)

        sql = """
        INSERT OR REPLACE INTO chunks
            (rowid, chunk_id, text, source, date, title, sector, region,
               data_type, company, url, sentiment, sentiment_label, fingerprint, extracted_at, indexed_at)
        VALUES
            (:rowid, :chunk_id, :text, :source, :date, :title, :sector, :region,
               :data_type, :company, :url, :sentiment, :sentiment_label, :fingerprint, :extracted_at, :indexed_at)
        """
        with self._conn() as conn:
            conn.executemany(sql, rows)

        logger.info("[MetadataStore] Upserted %d chunks (rowid %d–%d)", len(rows), start_rowid, start_rowid + len(rows) - 1)
        return len(rows)

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        placeholders = ",".join("?" * len(chunk_ids))
        with self._conn() as conn:
            cur = conn.execute(f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
            return cur.rowcount

    # ── Public read API ────────────────────────────────────────────────────────

    def count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return row[0] if row else 0

    def get_by_rowid(self, rowid: int) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE rowid=?", (rowid,)).fetchone()
            return dict(row) if row else None

    def get_all_as_list(self) -> list[dict[str, Any]]:
        """Return all rows as a list of dicts (backward-compat with load_metadata())."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def query(
        self,
        sector: str | None = None,
        region: str | None = None,
        data_type: str | None = None,
        min_date: str | None = None,
        max_date: str | None = None,
        source: str | None = None,
        min_sentiment: float | None = None,
        max_sentiment: float | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """
        Fast indexed query with optional filters.
        Returns chunk dicts sorted by date DESC.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if sector:
            conditions.append("sector = ?")
            params.append(sector)
        if region:
            conditions.append("region = ?")
            params.append(region)
        if data_type:
            conditions.append("data_type = ?")
            params.append(data_type)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if min_date:
            conditions.append("date >= ?")
            params.append(min_date)
        if max_date:
            conditions.append("date <= ?")
            params.append(max_date)
        if min_sentiment is not None:
            conditions.append("sentiment >= ?")
            params.append(min_sentiment)
        if max_sentiment is not None:
            conditions.append("sentiment <= ?")
            params.append(max_sentiment)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM chunks {where} ORDER BY date DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_rowids_for_filter(
        self,
        sector: str | None = None,
        region: str | None = None,
        data_type: str | None = None,
        min_date: str | None = None,
        source: str | None = None,
        company: str | None = None,
        sectors: set[str] | list[str] | tuple[str, ...] | None = None,
        regions: set[str] | list[str] | tuple[str, ...] | None = None,
        data_types: set[str] | list[str] | tuple[str, ...] | None = None,
        sources: set[str] | list[str] | tuple[str, ...] | None = None,
        companies: set[str] | list[str] | tuple[str, ...] | None = None,
        text_terms: list[str] | tuple[str, ...] | None = None,
        limit: int = 5000,
    ) -> list[int]:
        """
        Return FAISS rowids that pass the given metadata filters.
        Used to build a FAISS IDSelector for pre-filtered ANN search.
        """
        def _as_values(single: str | None, many: set[str] | list[str] | tuple[str, ...] | None) -> list[str]:
            values: list[str] = []
            if single:
                values.append(str(single).strip())
            if many:
                for item in many:
                    val = str(item).strip()
                    if val:
                        values.append(val)
            dedup: list[str] = []
            seen: set[str] = set()
            for value in values:
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(value)
            return dedup

        def _build_fts_query(terms: list[str] | tuple[str, ...] | None) -> str:
            if not terms:
                return ""
            cleaned: list[str] = []
            for term in terms:
                norm = "".join(ch if (ch.isalnum() or ch in {"_", "-", " ", "."}) else " " for ch in str(term).lower())
                norm = " ".join(norm.split()).strip()
                if len(norm) < 2:
                    continue
                if " " in norm:
                    cleaned.append(f'"{norm}"')
                else:
                    cleaned.append(f"{norm}*")
                if len(cleaned) >= 8:
                    break
            if not cleaned:
                return ""
            return " AND ".join(cleaned[:3])

        conditions: list[str] = []
        params: list[Any] = []

        sector_values = _as_values(sector, sectors)
        region_values = _as_values(region, regions)
        dtype_values = _as_values(data_type, data_types)
        source_values = _as_values(source, sources)
        company_values = _as_values(company, companies)

        if sector_values:
            placeholders = ",".join("?" for _ in sector_values)
            conditions.append(f"c.sector IN ({placeholders})")
            params.extend(sector_values)
        if region_values:
            placeholders = ",".join("?" for _ in region_values)
            conditions.append(f"c.region IN ({placeholders})")
            params.extend(region_values)
        if dtype_values:
            placeholders = ",".join("?" for _ in dtype_values)
            conditions.append(f"c.data_type IN ({placeholders})")
            params.extend(dtype_values)
        if source_values:
            placeholders = ",".join("?" for _ in source_values)
            conditions.append(f"c.source IN ({placeholders})")
            params.extend(source_values)
        if company_values:
            placeholders = ",".join("?" for _ in company_values)
            conditions.append(f"c.company IN ({placeholders})")
            params.extend(company_values)
        if min_date:
            conditions.append("c.date >= ?")
            params.append(min_date)

        hard_limit = max(100, min(int(limit), 50000))
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        fts_query = _build_fts_query(text_terms)

        with self._conn() as conn:
            if self._fts_available and fts_query:
                sql = (
                    "SELECT c.rowid "
                    "FROM chunks_fts "
                    "JOIN chunks c ON c.rowid = chunks_fts.rowid "
                    f"WHERE chunks_fts MATCH ? {'AND ' + ' AND '.join(conditions) if conditions else ''} "
                    "ORDER BY bm25(chunks_fts), c.date DESC "
                    "LIMIT ?"
                )
                try:
                    rows = conn.execute(sql, [fts_query, *params, hard_limit]).fetchall()
                    if rows:
                        return [r[0] for r in rows]
                except sqlite3.Error as exc:
                    logger.debug("[MetadataStore] FTS rowid query failed, falling back to indexed scan: %s", exc)

            sql = f"SELECT c.rowid FROM chunks c {where} ORDER BY c.date DESC LIMIT ?"
            rows = conn.execute(sql, [*params, hard_limit]).fetchall()
            return [r[0] for r in rows]

    def sentiment_summary(
        self,
        sector: str | None = None,
        min_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate sentiment stats (overall label, avg score, distribution).
        Useful for "what is the current sentiment on tech stocks?" queries.
        """
        conditions: list[str] = []
        params: list[Any] = []
        if sector:
            conditions.append("sector = ?")
            params.append(sector)
        if min_date:
            conditions.append("date >= ?")
            params.append(min_date)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
        SELECT
            AVG(sentiment)                               AS avg_score,
            SUM(CASE WHEN sentiment_label='positive' THEN 1 ELSE 0 END) AS pos,
            SUM(CASE WHEN sentiment_label='negative' THEN 1 ELSE 0 END) AS neg,
            SUM(CASE WHEN sentiment_label='neutral'  THEN 1 ELSE 0 END) AS neu,
            COUNT(*) AS total
        FROM chunks {where}
        """
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        if not row or not row["total"]:
            return {"overall_label": "neutral", "avg_score": 0.0, "positive": 0, "negative": 0, "neutral": 0, "total": 0}

        avg = row["avg_score"] or 0.0
        return {
            "overall_label": "positive" if avg > 0.10 else ("negative" if avg < -0.10 else "neutral"),
            "avg_score":     round(avg, 4),
            "positive":      row["pos"],
            "negative":      row["neg"],
            "neutral":       row["neu"],
            "total":         row["total"],
        }

    # ── Export / migration ─────────────────────────────────────────────────────

    def export_json(self, path: str) -> None:
        """
        Write all rows to a JSON file (same format as the old metadata.json).
        Keeps backward compatibility with code that still reads the JSON file.
        """
        rows = self.get_all_as_list()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        logger.info("[MetadataStore] Exported %d rows to %s", len(rows), path)

    def import_json(self, path: str) -> int:
        """
        One-time migration: load an existing metadata.json into the DB.
        Assigns rowid = list-position (matching original FAISS index order).
        """
        if not os.path.exists(path):
            logger.warning("[MetadataStore] import_json: file not found: %s", path)
            return 0
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("[MetadataStore] import_json: expected list, got %s", type(data).__name__)
            return 0
        return self.upsert_chunks(data, start_rowid=0)

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a DB row back to the chunk dict format used by query.py."""
        d = dict(row)
        return {
            "chunk_id":  d.get("chunk_id", ""),
            "text":      d.get("text", ""),
            "sector":    d.get("sector", "Unknown"),
            "region":    d.get("region", "Global"),
            "sentiment": d.get("sentiment", 0.0),
            "sentiment_label": d.get("sentiment_label", "neutral"),
            "_rowid":    d.get("rowid"),
            "metadata": {
                "title":       d.get("title", ""),
                "source":      d.get("source", ""),
                "date":        d.get("date", ""),
                "data_type":   d.get("data_type", ""),
                "company":     d.get("company", "Unknown"),
                "url":         d.get("url", ""),
                "sector":      d.get("sector", "Unknown"),
                "region":      d.get("region", "Global"),
                "extracted_at": d.get("extracted_at", ""),
                "indexed_at":  d.get("indexed_at", ""),
                "fingerprint": d.get("fingerprint", ""),
            },
        }


# ── Module-level singleton ─────────────────────────────────────────────────────

_store: MetadataStore | None = None


def get_store(db_path: str = _DEFAULT_DB_PATH) -> MetadataStore:
    """Return a singleton MetadataStore instance (safe for multi-threaded use)."""
    global _store
    if _store is None or _store.db_path != db_path:
        _store = MetadataStore(db_path)
    return _store
