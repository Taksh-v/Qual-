"""
run_scheduler.py — Continuous ingestion daemon for the RAG pipeline.
─────────────────────────────────────────────────────────────────────
Runs two recurring jobs:

  1. RSS refresh      — every SCHEDULER_RSS_INTERVAL_MIN minutes (default 30)
                        Fetches new articles from all RSS feeds, cleans and
                        chunks them, and appends to the vector index.

  2. SEC 8-K fetch  — every SCHEDULER_SEC_INTERVAL_MIN minutes (default 60)
                        Fetches new 8-K regulatory filings from SEC EDGAR.

  3. Index rebuild    — every SCHEDULER_INDEX_INTERVAL_HOURS hours (default 6)
                        Full re-embed + rebuild of the FAISS index to
                        incorporate all accumulated chunks cleanly.

Usage:
    python run_scheduler.py                  # run forever
    python run_scheduler.py --once-rss       # single RSS run then exit
    python run_scheduler.py --once-sec       # single SEC run then exit
    python run_scheduler.py --once-index     # single index rebuild then exit

Environment variables (see .env.example):
    SCHEDULER_RSS_INTERVAL_MIN      default 30
    SCHEDULER_SEC_INTERVAL_MIN      default 60
    SCHEDULER_FUNDAMENTALS_INTERVAL_HOURS default 12
    SCHEDULER_MACRO_INTERVAL_HOURS  default 12
    SCHEDULER_SECTOR_INTERVAL_HOURS default 6
    SCHEDULER_INSIDER_INTERVAL_HOURS default 12
    SCHEDULER_TRANSCRIPT_INTERVAL_HOURS default 168
    SCHEDULER_INDEX_INTERVAL_HOURS  default 6
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [scheduler] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("scheduler")

RSS_INTERVAL_MIN     = int(os.getenv("SCHEDULER_RSS_INTERVAL_MIN",     "30"))
SEC_INTERVAL_MIN     = int(os.getenv("SCHEDULER_SEC_INTERVAL_MIN",     "60"))
FUNDAMENTALS_INTERVAL_HOURS = int(os.getenv("SCHEDULER_FUNDAMENTALS_INTERVAL_HOURS", "12"))
MACRO_INTERVAL_HOURS = int(os.getenv("SCHEDULER_MACRO_INTERVAL_HOURS", "12"))
SECTOR_INTERVAL_HOURS= int(os.getenv("SCHEDULER_SECTOR_INTERVAL_HOURS", "6"))
INSIDER_INTERVAL_HOURS=int(os.getenv("SCHEDULER_INSIDER_INTERVAL_HOURS", "12"))
TRANSCRIPT_INTERVAL_HOURS=int(os.getenv("SCHEDULER_TRANSCRIPT_INTERVAL_HOURS", "168"))
INDEX_INTERVAL_HOURS = int(os.getenv("SCHEDULER_INDEX_INTERVAL_HOURS", "6"))

_PYTHON = sys.executable


# ── Job implementations ───────────────────────────────────────────────────────

def run_rss_ingestion() -> bool:
    """Fetch RSS feeds, clean, chunk, and append to the index."""
    logger.info("Starting RSS ingestion run …")
    t = time.time()
    try:
        result = subprocess.run(
            [_PYTHON, "run_rss_ingest.py"],
            capture_output=True, text=True, timeout=1800,
        )
        elapsed = time.time() - t
        if result.returncode == 0:
            logger.info("RSS ingestion OK (%.1fs)", elapsed)
            return True
        else:
            logger.error("RSS ingestion FAILED (%.1fs):\n%s", elapsed, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        logger.error("RSS ingestion timed out after 1800s")
        return False
    except Exception as exc:
        logger.exception("RSS ingestion raised: %s", exc)
        return False

def run_sec_ingestion() -> bool:
    """Fetch SEC EDGAR 8-K filings."""
    logger.info("Starting SEC EDGAR ingestion run …")
    t = time.time()
    try:
        result = subprocess.run(
            [_PYTHON, "run_sec_ingestion.py"],
            capture_output=True, text=True, timeout=1800,
        )
        elapsed = time.time() - t
        if result.returncode == 0:
            logger.info("SEC ingestion OK (%.1fs)", elapsed)
            return True
        else:
            logger.error("SEC ingestion FAILED (%.1fs):\n%s", elapsed, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        logger.error("SEC ingestion timed out after 1800s")
        return False
    except Exception as exc:
        logger.exception("SEC ingestion raised: %s", exc)
        return False


def run_fundamental_ingestion() -> bool:
    """Fetch structured company fundamentals into SQLite."""
    logger.info("Starting Fundamentals database sync …")
    t = time.time()
    try:
        result = subprocess.run(
            [_PYTHON, "ingestion/fundamental_ingest.py"],
            capture_output=True, text=True, timeout=1800,
        )
        elapsed = time.time() - t
        if result.returncode == 0:
            logger.info("Fundamentals sync OK (%.1fs)", elapsed)
            return True
        else:
            logger.error("Fundamentals sync FAILED (%.1fs):\n%s", elapsed, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        logger.error("Fundamentals sync timed out after 1800s")
        return False
    except Exception as exc:
        logger.exception("Fundamentals sync raised: %s", exc)
        return False


def run_macro_ingestion() -> bool:
    """Fetch broad macroeconomic indicators into SQLite."""
    logger.info("Starting Macroeconomic database sync …")
    t = time.time()
    try:
        result = subprocess.run(
            [_PYTHON, "ingestion/macro_data.py"],
            capture_output=True, text=True, timeout=1800,
        )
        elapsed = time.time() - t
        if result.returncode == 0:
            logger.info("Macro sync OK (%.1fs)", elapsed)
            return True
        else:
            logger.error("Macro sync FAILED (%.1fs):\n%s", elapsed, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        logger.error("Macro sync timed out after 1800s")
        return False
    except Exception as exc:
        logger.exception("Macro sync raised: %s", exc)
        return False


def run_sector_ingestion() -> bool:
    """Fetch structured Sector ETFs into SQLite."""
    logger.info("Starting Sector database sync …")
    t = time.time()
    try:
        result = subprocess.run(
            [_PYTHON, "ingestion/sector_data.py"],
            capture_output=True, text=True, timeout=900,
        )
        elapsed = time.time() - t
        if result.returncode == 0:
            logger.info("Sector sync OK (%.1fs)", elapsed)
            return True
        else:
            logger.error("Sector sync FAILED (%.1fs):\n%s", elapsed, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        logger.error("Sector sync timed out after 900s")
        return False
    except Exception as exc:
        logger.exception("Sector sync raised: %s", exc)
        return False
        
def run_insider_ingestion() -> bool:
    """Fetch SEC EDGAR Form 4 Insider Trades into SQLite."""
    logger.info("Starting Insider Form 4 database sync …")
    t = time.time()
    try:
        result = subprocess.run(
            [_PYTHON, "ingestion/insider_trading.py"],
            capture_output=True, text=True, timeout=900,
        )
        elapsed = time.time() - t
        if result.returncode == 0:
            logger.info("Insider sync OK (%.1fs)", elapsed)
            return True
        else:
            logger.error("Insider sync FAILED (%.1fs):\n%s", elapsed, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        logger.error("Insider sync timed out after 900s")
        return False
    except Exception as exc:
        logger.exception("Insider sync raised: %s", exc)
        return False

def run_transcript_ingestion() -> bool:
    """Fetch Executive Q&A Earnings Call Transcripts into SQLite."""
    logger.info("Starting Earnings Transcripts database sync (FMP) …")
    t = time.time()
    try:
        result = subprocess.run(
            [_PYTHON, "ingestion/earnings_transcripts.py"],
            capture_output=True, text=True, timeout=900,
        )
        elapsed = time.time() - t
        if result.returncode == 0:
            logger.info("Transcripts sync OK (%.1fs)", elapsed)
            return True
        else:
            logger.error("Transcripts sync FAILED (%.1fs):\n%s", elapsed, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        logger.error("Transcripts sync timed out after 900s")
        return False
    except Exception as exc:
        logger.exception("Transcripts sync raised: %s", exc)
        return False

def run_index_rebuild() -> bool:
    """Re-embed all chunks and rebuild the FAISS index from scratch."""
    logger.info("Starting full index rebuild …")
    t = time.time()
    try:
        result = subprocess.run(
            [_PYTHON, "run_embedding_index.py", "--resume"],
            capture_output=True, text=True, timeout=3600,
        )
        elapsed = time.time() - t
        if result.returncode == 0:
            logger.info("Index rebuild OK (%.1fs)", elapsed)
            return True
        else:
            logger.error("Index rebuild FAILED (%.1fs):\n%s", elapsed, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        logger.error("Index rebuild timed out after 3600s")
        return False
    except Exception as exc:
        logger.exception("Index rebuild raised: %s", exc)
        return False


# ── Scheduler loop ────────────────────────────────────────────────────────────

def run_forever() -> None:
    """Main loop: run jobs on their respective intervals."""
    logger.info(
        "Scheduler starting. RSS every %dm, SEC every %dm, Index rebuild every %dh.",
        RSS_INTERVAL_MIN, SEC_INTERVAL_MIN, INDEX_INTERVAL_HOURS,
    )

    rss_interval_s   = RSS_INTERVAL_MIN * 60
    sec_interval_s   = SEC_INTERVAL_MIN * 60
    fund_interval_s  = FUNDAMENTALS_INTERVAL_HOURS * 3600
    macro_interval_s = MACRO_INTERVAL_HOURS * 3600
    sector_interval_s= SECTOR_INTERVAL_HOURS * 3600
    insider_interval_s= INSIDER_INTERVAL_HOURS * 3600
    transcript_interval_s= TRANSCRIPT_INTERVAL_HOURS * 3600
    index_interval_s = INDEX_INTERVAL_HOURS * 3600

    # Stagger the first runs slightly to avoid startup collision with the API.
    last_rss   = time.time() - rss_interval_s + 30    # first RSS run in 30s
    last_sec   = time.time() - sec_interval_s + 60    # first SEC run in 60s
    last_fund  = time.time() - fund_interval_s + 90   # first Fund run in 90s
    last_macro = time.time() - macro_interval_s + 105 # first Macro run in 105s
    last_sector= time.time() - sector_interval_s + 115# first Sector run in 115s
    last_insider=time.time() - insider_interval_s + 125# first Insider run in 125s
    last_transcript=time.time() - transcript_interval_s + 135# first Transcripts run in 135s
    last_index = time.time() - index_interval_s + 140 # first rebuild in 140s

    while True:
        now = time.time()

        if now - last_rss >= rss_interval_s:
            run_rss_ingestion()
            last_rss = time.time()

        if now - last_sec >= sec_interval_s:
            run_sec_ingestion()
            last_sec = time.time()
            
        if now - last_fund >= fund_interval_s:
            run_fundamental_ingestion()
            last_fund = time.time()
            
        if now - last_macro >= macro_interval_s:
            run_macro_ingestion()
            last_macro = time.time()

        if now - last_sector >= sector_interval_s:
            run_sector_ingestion()
            last_sector = time.time()
            
        if now - last_insider >= insider_interval_s:
            run_insider_ingestion()
            last_insider = time.time()
            
        if now - last_transcript >= transcript_interval_s:
            run_transcript_ingestion()
            last_transcript = time.time()

        if now - last_index >= index_interval_s:
            run_index_rebuild()
            last_index = time.time()

        # Sleep 60s between checks — low CPU cost
        time.sleep(60)


# ── APScheduler support (optional, richer logging & job history) ──────────────

def run_with_apscheduler() -> None:
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        logger.warning("APScheduler not installed; falling back to simple loop.")
        run_forever()
        return

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_rss_ingestion, IntervalTrigger(minutes=RSS_INTERVAL_MIN),
        id="rss", name="RSS ingestion", max_instances=1,
        next_run_time=datetime.now(tz=timezone.utc),
    )
    scheduler.add_job(
        run_sec_ingestion, IntervalTrigger(minutes=SEC_INTERVAL_MIN),
        id="sec", name="SEC EDGAR ingestion", max_instances=1,
    )
    scheduler.add_job(
        run_fundamental_ingestion, IntervalTrigger(hours=FUNDAMENTALS_INTERVAL_HOURS),
        id="fundamentals", name="SQLite Fundamentals Sync", max_instances=1,
    )
    scheduler.add_job(
        run_macro_ingestion, IntervalTrigger(hours=MACRO_INTERVAL_HOURS),
        id="macro", name="SQLite Macro Sync", max_instances=1,
    )
    scheduler.add_job(
        run_sector_ingestion, IntervalTrigger(hours=SECTOR_INTERVAL_HOURS),
        id="sector", name="SQLite Sector Sync", max_instances=1,
    )
    scheduler.add_job(
        run_insider_ingestion, IntervalTrigger(hours=INSIDER_INTERVAL_HOURS),
        id="insider", name="SQLite Insider Form 4 Sync", max_instances=1,
    )
    scheduler.add_job(
        run_transcript_ingestion, IntervalTrigger(hours=TRANSCRIPT_INTERVAL_HOURS),
        id="transcripts", name="SQLite Earnings Transcripts Sync", max_instances=1,
    )
    scheduler.add_job(
        run_index_rebuild, IntervalTrigger(hours=INDEX_INTERVAL_HOURS),
        id="index", name="Index rebuild", max_instances=1,
    )
    logger.info(
        "APScheduler started. RSS every %dm | SEC every %dm | Index rebuild every %dh.",
        RSS_INTERVAL_MIN, SEC_INTERVAL_MIN, INDEX_INTERVAL_HOURS,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG ingestion scheduler")
    parser.add_argument("--once-rss",   action="store_true", help="Run one RSS pass and exit")
    parser.add_argument("--once-sec",   action="store_true", help="Run one SEC pass and exit")
    parser.add_argument("--once-fund",  action="store_true", help="Run one Fundamentals sync and exit")
    parser.add_argument("--once-macro", action="store_true", help="Run one Macro sync and exit")
    parser.add_argument("--once-sector",action="store_true", help="Run one Sector sync and exit")
    parser.add_argument("--once-insider",action="store_true", help="Run one Insider Form 4 sync and exit")
    parser.add_argument("--once-transcript",action="store_true", help="Run one Earnings Transcript sync and exit")
    parser.add_argument("--once-index", action="store_true", help="Run one index rebuild and exit")
    parser.add_argument("--ingest-all", action="store_true", help="Run all ingestions sequentially and exit")
    args = parser.parse_args()

    if args.once_rss:
        ok = run_rss_ingestion()
        sys.exit(0 if ok else 1)
    elif args.once_sec:
        ok = run_sec_ingestion()
        sys.exit(0 if ok else 1)
    elif args.once_fund:
        ok = run_fundamental_ingestion()
        sys.exit(0 if ok else 1)
    elif args.once_macro:
        ok = run_macro_ingestion()
        sys.exit(0 if ok else 1)
    elif args.once_sector:
        ok = run_sector_ingestion()
        sys.exit(0 if ok else 1)
    elif args.once_insider:
        ok = run_insider_ingestion()
        sys.exit(0 if ok else 1)
    elif args.once_transcript:
        ok = run_transcript_ingestion()
        sys.exit(0 if ok else 1)
    elif args.once_index:
        ok = run_index_rebuild()
        sys.exit(0 if ok else 1)
    elif args.ingest_all:
        ok = run_rss_ingestion()
        ok = ok and run_sec_ingestion()
        ok = ok and run_fundamental_ingestion()
        ok = ok and run_macro_ingestion()
        ok = ok and run_sector_ingestion()
        ok = ok and run_insider_ingestion()
        ok = ok and run_transcript_ingestion()
        ok = ok and run_index_rebuild()
        sys.exit(0 if ok else 1)
    else:
        run_with_apscheduler()
