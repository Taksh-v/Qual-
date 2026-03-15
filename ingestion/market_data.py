"""
market_data.py
--------------
Fetches real-time and historical price data for global market instruments
via yfinance.  All fetches are disk-cached (JSON) with configurable TTL to
avoid hammering Yahoo Finance rate limits.

Key exports:
  get_market_snapshot()      -> compact dict ready for prompt injection
  get_realtime_data(symbols) -> per-symbol dict (legacy compat)
  get_price_history(symbol)  -> DataFrame (daily OHLCV, 30-day)
  GLOBAL_MARKETS             -> list of representative tickers

Cache location: data/market_data_cache.json
TTL env var   : MARKET_DATA_CACHE_TTL  (default 900 s = 15 min)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_FILE = os.path.join(_BASE_DIR, "data", "market_data_cache.json")
_CACHE_TTL = int(os.getenv("MARKET_DATA_CACHE_TTL", "900"))  # 15 min default

# Representative global market symbols used for snapshot
GLOBAL_MARKETS = [
    "^GSPC",   # S&P 500
    "^DJI",    # Dow Jones Industrial Average
    "^IXIC",   # NASDAQ Composite
    "^NSEI",   # NIFTY 50 (India)
    "^BSESN",  # BSE SENSEX (India)
    "^FTSE",   # FTSE 100 (UK)
    "^N225",   # Nikkei 225 (Japan)
    "^HSI",    # Hang Seng (HK)
    "^STOXX50E",  # Euro STOXX 50
    "GC=F",    # Gold futures
    "CL=F",    # WTI Crude Oil futures
    "BZ=F",    # Brent Crude futures
    "SI=F",    # Silver futures
    "HG=F",    # Copper futures
    "BTC-USD", # Bitcoin
    "ETH-USD", # Ethereum
    "DX-Y.NYB",# US Dollar Index
    "^TNX",    # US 10-year Treasury yield
    "^TYX",    # US 30-year Treasury yield
    "^IRX",    # US 13-week Treasury yield
]

# Human-readable labels for snapshot display
_SNAPSHOT_LABELS: dict[str, str] = {
    "^GSPC":    "S&P500",
    "^DJI":     "Dow",
    "^IXIC":    "Nasdaq",
    "^NSEI":    "Nifty50",
    "^BSESN":   "Sensex",
    "^FTSE":    "FTSE100",
    "^N225":    "Nikkei",
    "^HSI":     "HangSeng",
    "^STOXX50E":"EuroStoxx50",
    "GC=F":     "Gold",
    "CL=F":     "WTI_Oil",
    "BZ=F":     "Brent_Oil",
    "SI=F":     "Silver",
    "HG=F":     "Copper",
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
    "DX-Y.NYB": "USD_Index",
    "^TNX":     "US10Y_Yield",
    "^TYX":     "US30Y_Yield",
    "^IRX":     "US13W_Yield",
}


# ─── Cache helpers ──────────────────────────────────────────────────────────────
def _load_cache() -> dict[str, Any]:
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError as exc:
        logger.warning("[market_data] Cache write failed: %s", exc)


def _is_stale(entry: dict[str, Any]) -> bool:
    fetched_at = entry.get("_fetched_at", 0)
    return (time.time() - fetched_at) > _CACHE_TTL


# ─── Core fetch functions ───────────────────────────────────────────────────────
def _fetch_quote(symbol: str) -> dict[str, Any]:
    """Fetch latest quote for a single symbol. Returns {} on any error."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        hist = ticker.history(period="2d")

        prev_close = None
        current_price = None
        pct_change = None

        if hasattr(info, "last_price"):
            current_price = info.last_price
        if not hist.empty:
            if len(hist) >= 2:
                prev_close = float(hist["Close"].iloc[-2])
            if current_price is None:
                current_price = float(hist["Close"].iloc[-1])

        if current_price is not None and prev_close and prev_close != 0:
            pct_change = round((current_price - prev_close) / prev_close * 100, 3)

        return {
            "price":      round(current_price, 4) if current_price is not None else None,
            "prev_close": round(prev_close, 4) if prev_close else None,
            "pct_change": pct_change,
            "_fetched_at": time.time(),
        }
    except Exception as exc:
        logger.debug("[market_data] fetch_quote(%s) failed: %s", symbol, exc)
        return {"price": None, "prev_close": None, "pct_change": None, "_fetched_at": time.time()}


def get_realtime_data(symbols: list[str]) -> dict[str, Any]:
    """
    Fetch current price data for a list of symbols, using disk cache to
    respect Yahoo Finance rate limits.  Returns a dict keyed by symbol.
    """
    cache = _load_cache()
    result: dict[str, Any] = {}
    updated = False

    for symbol in symbols:
        cached = cache.get(symbol, {})
        if cached and not _is_stale(cached):
            result[symbol] = cached
        else:
            data = _fetch_quote(symbol)
            cache[symbol] = data
            result[symbol] = data
            updated = True

    if updated:
        _save_cache(cache)

    return result


def get_price_history(symbol: str, period: str = "1mo") -> Any:
    """
    Return a pandas DataFrame of daily OHLCV data for *symbol*.
    Falls back to an empty DataFrame on errors.
    """
    try:
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period)
    except Exception as exc:
        logger.warning("[market_data] get_price_history(%s): %s", symbol, exc)
        try:
            import pandas as pd
            return pd.DataFrame()
        except ImportError:
            return None


def get_market_snapshot(symbols: list[str] | None = None) -> dict[str, Any]:
    """
    Return a lean, JSON-serialisable snapshot dict suitable for LLM prompt injection.

    Format:
        {
            "S&P500":     {"price": 5200.4, "pct_change": +0.43},
            "Gold":       {"price": 2345.0, "pct_change": -0.12},
            ...
            "_fetched_at": 1712345678.0
        }
    """
    use_symbols = symbols if symbols is not None else GLOBAL_MARKETS
    raw = get_realtime_data(use_symbols)

    snapshot: dict[str, Any] = {}
    for sym in use_symbols:
        label = _SNAPSHOT_LABELS.get(sym, sym)
        entry = raw.get(sym, {})
        snapshot[label] = {
            "price":      entry.get("price"),
            "pct_change": entry.get("pct_change"),
        }
    snapshot["_fetched_at"] = time.time()
    return snapshot


def format_snapshot_for_prompt(snapshot: dict[str, Any] | None = None) -> str:
    """
    Produce a compact text block for injection into an LLM prompt.
    Example output:
        S&P500=5200.40 (+0.43%) | Gold=2345.00 (-0.12%) | WTI_Oil=82.10 (+1.05%) | ...
    """
    if snapshot is None:
        snapshot = get_market_snapshot()

    parts: list[str] = []
    for label, data in snapshot.items():
        if label.startswith("_"):
            continue
        price = data.get("price")
        change = data.get("pct_change")
        if price is None:
            continue
        if change is not None:
            sign = "+" if change >= 0 else ""
            parts.append(f"{label}={price:,.2f} ({sign}{change:.2f}%)")
        else:
            parts.append(f"{label}={price:,.2f}")

    return " | ".join(parts) if parts else "No live market data available."
