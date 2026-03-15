"""
ingestion/macro_data.py
------------------------
Fetch FRED macroeconomic data and store it for RAG context injection.

KEY IMPROVEMENTS over the original:
  1. No API key required  — uses FRED's free CSV endpoint for all public series
     (FRED_API_KEY env var is optional; enables the JSON endpoint for more series)
  2. Trend context       — fetches last N observations so the LLM sees direction,
                           not just the latest single data point
  3. RAG-injectable chunks — exports self-contained text chunks for the vector DB,
                             so macro data appears alongside news in retrieval
  4. TTL caching         — only refreshes if data is older than CACHE_HOURS (default 6h)
  5. Change detection    — computes MoM / QoQ delta and attaches to each row
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from typing import Any

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
FRED_API_KEY  = os.getenv("FRED_API_KEY", "")     # optional — unlocks extra series
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH       = os.path.join(BASE_DIR, "data", "sql_db", "macro.db")
CHUNKS_PATH   = os.path.join(BASE_DIR, "data", "chunks", "macro", "macro_context_chunks.json")
LEGACY_CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks", "macro", "macro_context.json")
CACHE_HOURS   = int(os.getenv("MACRO_CACHE_HOURS", "6"))     # re-fetch every 6h
N_OBS         = int(os.getenv("MACRO_N_OBS", "4"))           # observations to fetch per series
REQUEST_TIMEOUT = float(os.getenv("MACRO_REQUEST_TIMEOUT", "15"))

# ── FRED series ────────────────────────────────────────────────────────────────
# All of these are public — no API key required for CSV download
FRED_SERIES: dict[str, dict[str, str]] = {
    "GDP Growth Rate":         {"id": "A191RL1Q225SBEA", "freq": "quarterly",   "unit": "%",   "type": "macro_growth"},
    "CPI Inflation (YoY)":    {"id": "CPIAUCSL",         "freq": "monthly",    "unit": "index","type": "macro_inflation"},
    "Core PCE Inflation":      {"id": "PCEPILFE",         "freq": "monthly",    "unit": "index","type": "macro_inflation"},
    "Fed Funds Rate":          {"id": "FEDFUNDS",         "freq": "monthly",    "unit": "%",   "type": "macro_rates"},
    "10-Year Treasury Yield":  {"id": "DGS10",            "freq": "daily",      "unit": "%",   "type": "macro_rates"},
    "2-Year Treasury Yield":   {"id": "DGS2",             "freq": "daily",      "unit": "%",   "type": "macro_rates"},
    "Unemployment Rate":       {"id": "UNRATE",           "freq": "monthly",    "unit": "%",   "type": "macro_employment"},
    "Nonfarm Payrolls":        {"id": "PAYEMS",           "freq": "monthly",    "unit": "thousands", "type": "macro_employment"},
    "M2 Money Supply":         {"id": "M2SL",             "freq": "monthly",    "unit": "billions",  "type": "macro_liquidity"},
    "Consumer Sentiment":      {"id": "UMCSENT",          "freq": "monthly",    "unit": "index","type": "macro_sentiment"},
    "Retail Sales":            {"id": "RSAFS",            "freq": "monthly",    "unit": "millions",  "type": "macro_activity"},
    "ISM Manufacturing PMI":   {"id": "MANEMP",           "freq": "monthly",    "unit": "index","type": "macro_activity"},
    "30-Year Mortgage Rate":   {"id": "MORTGAGE30US",     "freq": "weekly",     "unit": "%",   "type": "macro_housing"},
    "Housing Starts":          {"id": "HOUST",            "freq": "monthly",    "unit": "thousands", "type": "macro_housing"},
    "VIX Volatility":          {"id": "VIXCLS",           "freq": "daily",      "unit": "index","type": "macro_market"},
    "Business Inventories":    {"id": "ISRATIO",          "freq": "monthly",    "unit": "ratio","type": "macro_activity"},
}


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _init_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS macro_indicators (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        indicator_name   TEXT    NOT NULL,
        series_id        TEXT    NOT NULL,
        observation_date TEXT    NOT NULL,
        value            REAL    NOT NULL,
        delta            REAL,               -- change from previous observation
        unit             TEXT    NOT NULL DEFAULT '',
        data_type        TEXT    NOT NULL DEFAULT 'macro_commentary',
        fetched_at       TEXT    NOT NULL,
        UNIQUE(series_id, observation_date)
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_series ON macro_indicators(series_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_date   ON macro_indicators(observation_date DESC);")

    # Backward-compatible migration for legacy DBs that predate newer columns.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(macro_indicators)").fetchall()}
    if "delta" not in cols:
        conn.execute("ALTER TABLE macro_indicators ADD COLUMN delta REAL")
    if "unit" not in cols:
        conn.execute("ALTER TABLE macro_indicators ADD COLUMN unit TEXT NOT NULL DEFAULT ''")
    if "data_type" not in cols:
        conn.execute("ALTER TABLE macro_indicators ADD COLUMN data_type TEXT NOT NULL DEFAULT 'macro_commentary'")
    if "fetched_at" not in cols:
        conn.execute("ALTER TABLE macro_indicators ADD COLUMN fetched_at TEXT NOT NULL DEFAULT ''")

    now_str = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE macro_indicators SET fetched_at=? WHERE fetched_at IS NULL OR fetched_at=''",
        (now_str,),
    )
    conn.execute(
        "UPDATE macro_indicators SET unit='' WHERE unit IS NULL",
    )
    conn.execute(
        "UPDATE macro_indicators SET data_type='macro_commentary' WHERE data_type IS NULL OR data_type=''",
    )

    conn.commit()
    return conn


def _last_fetch_time(conn: sqlite3.Connection, series_id: str) -> datetime | None:
    row = conn.execute(
        "SELECT MAX(fetched_at) FROM macro_indicators WHERE series_id=?", (series_id,)
    ).fetchone()
    if row and row[0]:
        try:
            return datetime.fromisoformat(row[0])
        except Exception:
            pass
    return None


def _is_stale(conn: sqlite3.Connection, series_id: str) -> bool:
    last = _last_fetch_time(conn, series_id)
    if last is None:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last) > timedelta(hours=CACHE_HOURS)


# ── FRED fetch helpers ─────────────────────────────────────────────────────────

def _fetch_via_csv(series_id: str, n: int = N_OBS) -> list[tuple[str, float]]:
    """
    Fetch recent observations using FRED's free CSV export endpoint.
    No API key required. Returns list of (date_str, value) tuples, newest first.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT,
                            headers={"User-Agent": "FinanceRAG/1.0 (research use)"})
        resp.raise_for_status()
        reader = csv.reader(io.StringIO(resp.text))
        next(reader)  # skip header
        rows: list[tuple[str, float]] = []
        for row in reader:
            if len(row) < 2 or row[1].strip() == ".":
                continue
            try:
                rows.append((row[0].strip(), float(row[1].strip())))
            except ValueError:
                continue
        return list(reversed(rows))[:n]   # newest first
    except Exception as exc:
        logger.warning("[FRED-CSV] Failed %s: %s", series_id, exc)
        return []


def _fetch_via_api(series_id: str, n: int = N_OBS) -> list[tuple[str, float]]:
    """Fetch via FRED JSON API (requires FRED_API_KEY). Better for rate-limited use."""
    if not FRED_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": FRED_API_KEY,
                    "file_type": "json", "sort_order": "desc", "limit": n},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        return [(o["date"], float(o["value"])) for o in obs if o.get("value") not in (".", None, "")]
    except Exception as exc:
        logger.warning("[FRED-API] Failed %s: %s", series_id, exc)
        return []


def _fetch_series(series_id: str) -> list[tuple[str, float]]:
    """Try API first (if key available), fall back to CSV."""
    if FRED_API_KEY:
        result = _fetch_via_api(series_id)
        if result:
            return result
    return _fetch_via_csv(series_id)


# ── Main ingestion ─────────────────────────────────────────────────────────────

def fetch_and_store_macro_data(force: bool = False) -> dict[str, Any]:
    """
    Fetch all FRED series and store to SQLite.
    Skips series that were fetched within CACHE_HOURS unless force=True.

    Returns a summary dict: {series_id: {"name", "latest_value", "delta", "unit"}}
    """
    conn = _init_db()
    now_str = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {}
    total_stored = 0

    for name, meta in FRED_SERIES.items():
        sid = meta["id"]
        if not force and not _is_stale(conn, sid):
            logger.debug("[FRED] %s is fresh, skipping.", name)
            continue

        observations = _fetch_series(sid)
        if not observations:
            logger.warning("[FRED] No data for %s (%s)", name, sid)
            continue

        delta = None
        for i, (date_str, value) in enumerate(observations):
            if i + 1 < len(observations):
                delta = value - observations[i + 1][1]
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO macro_indicators
                       (indicator_name, series_id, observation_date, value, delta, unit, data_type, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (name, sid, date_str, value, delta if i == 0 else None,
                     meta["unit"], meta["type"], now_str),
                )
                total_stored += 1
            except Exception as exc:
                logger.warning("[FRED] DB insert failed for %s on %s: %s", name, date_str, exc)

        if observations:
            latest_val = observations[0][1]
            latest_dt  = observations[0][0]
            summary[sid] = {
                "name": name, "latest_date": latest_dt,
                "latest_value": latest_val, "delta": delta,
                "unit": meta["unit"], "type": meta["type"],
            }
            logger.info("[FRED] %-30s  %s %s  (Δ%s)", name, latest_val, meta["unit"],
                        f"{delta:+.3f}" if delta is not None else "n/a")

        time.sleep(0.2)  # polite to FRED servers

    conn.commit()
    conn.close()
    logger.info("[FRED] Stored/refreshed %d observations across %d series", total_stored, len(summary))
    return summary


# ── RAG chunk export ───────────────────────────────────────────────────────────

def build_macro_rag_chunks() -> list[dict[str, Any]]:
    """
    Build self-contained RAG chunks from stored macro data.
    Each series becomes one chunk with a narrative text suitable for LLM context.
    Returns list of chunk dicts compatible with run_embedding_index.py format.
    """
    conn = _init_db()
    chunks: list[dict[str, Any]] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for name, meta in FRED_SERIES.items():
        sid = meta["id"]
        rows = conn.execute(
            """SELECT observation_date, value, delta FROM macro_indicators
               WHERE series_id=? ORDER BY observation_date DESC LIMIT ?""",
            (sid, N_OBS),
        ).fetchall()

        if not rows:
            continue

        latest_date, latest_val, delta = rows[0]
        unit = meta["unit"]

        # Build trend description
        if delta is not None:
            direction = "up" if delta > 0 else ("down" if delta < 0 else "unchanged")
            trend_str = f"{direction} {abs(delta):.3f} {unit} from previous observation"
        else:
            trend_str = "no prior comparison available"

        # Historical context
        history = ", ".join(f"{r[0]}: {r[1]:.3f}" for r in rows)

        text = (
            f"{name} ({sid}): Latest reading {latest_val:.3f} {unit} as of {latest_date}. "
            f"Trend: {trend_str}. "
            f"Recent history — {history}."
        )

        import hashlib
        chunk_id = f"macro_{sid}_{latest_date}"
        fingerprint = hashlib.md5(text.encode()).hexdigest()[:16]

        chunks.append({
            "chunk_id":    chunk_id,
            "text":        text,
            "sector":      "Macro",
            "region":      "Global",
            "fingerprint": fingerprint,
            "metadata": {
                "title":       f"FRED: {name}",
                "source":      "Federal Reserve Economic Data (FRED)",
                "date":        latest_date,
                "data_type":   meta["type"],
                "extracted_at": today,
                "series_id":   sid,
                "unit":        unit,
            },
        })

    conn.close()

    # Persist to chunks directory so run_embedding_index.py can pick them up
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Keep legacy path for compatibility with older tooling/docs.
    with open(LEGACY_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info("[FRED] Exported %d macro RAG chunks to %s", len(chunks), CHUNKS_PATH)
    return chunks


def get_macro_context_text() -> str:
    """
    Return a compact text summary of all current macro indicators
    suitable for direct injection into an LLM prompt-context block.
    """
    conn = _init_db()
    lines = ["=== MACRO INDICATORS (FRED) ==="]
    for name, meta in FRED_SERIES.items():
        sid = meta["id"]
        row = conn.execute(
            "SELECT observation_date, value, delta FROM macro_indicators "
            "WHERE series_id=? ORDER BY observation_date DESC LIMIT 1", (sid,)
        ).fetchone()
        if row:
            date_str, val, delta = row
            delta_str = f"  Δ{delta:+.3f}" if delta is not None else ""
            lines.append(f"  {name}: {val:.3f} {meta['unit']} ({date_str}){delta_str}")
    conn.close()
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Fetching FRED macro data…")
    summary = fetch_and_store_macro_data(force=True)
    print(f"\nFetched {len(summary)} series\n")
    chunks = build_macro_rag_chunks()
    print(f"Built {len(chunks)} RAG chunks → {CHUNKS_PATH}")
    print("\n" + get_macro_context_text())

