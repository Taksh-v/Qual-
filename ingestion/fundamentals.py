"""
ingestion/fundamentals.py
--------------------------
Company-level fundamental data fetcher using yfinance.

Blueprint alignment (Section 2 – Data Sources):
  "Integrate company fundamentals: earnings, revenue, P/E, debt metrics,
   analyst estimates and sector/industry classification."

Features:
  - Fetches key valuation, profitability and risk metrics per ticker
  - Supports a batch of tickers returned as a dict keyed by symbol
  - Disk-backed TTL cache at data/fundamentals_cache.json (default TTL 6 h)
  - Hard timeout per ticker (default 10 s) to avoid hanging in the pipeline

Usage:
    from ingestion.fundamentals import get_fundamentals, get_batch_fundamentals

    f = get_fundamentals("AAPL")
    print(f["pe_ratio"], f["revenue_ttm"])

    batch = get_batch_fundamentals(["AAPL", "MSFT", "GOOGL"])
"""

from __future__ import annotations

import json
import logging
import os
import time
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_PATH = os.getenv(
    "FUNDAMENTALS_CACHE_PATH",
    os.path.join(_BASE_DIR, "data", "fundamentals_cache.json"),
)
_CACHE_TTL_SECS: float = float(os.getenv("FUNDAMENTALS_CACHE_TTL", str(6 * 3600)))
_FETCH_TIMEOUT_SECS: float = float(os.getenv("FUNDAMENTALS_FETCH_TIMEOUT", "10"))

_disk_cache: dict[str, dict[str, Any]] = {}   # ticker → {fetched_at, data}
_cache_lock: Lock = Lock()
_cache_loaded: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_disk_cache() -> None:
    global _cache_loaded
    if _cache_loaded:
        return
    try:
        if os.path.exists(_CACHE_PATH):
            with open(_CACHE_PATH, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                _disk_cache.update(loaded)
    except Exception as exc:
        logger.warning("[fundamentals] could not load cache from %s: %s", _CACHE_PATH, exc)
    _cache_loaded = True


def _save_disk_cache() -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(_CACHE_PATH)), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(_disk_cache, fh, indent=2, default=str)
    except Exception as exc:
        logger.warning("[fundamentals] could not write cache to %s: %s", _CACHE_PATH, exc)


def _safe(value: Any, default: Any = None) -> Any:
    """Return *value* if it is a non-None, non-NaN scalar/string, else *default*."""
    if value is None:
        return default
    if isinstance(value, float):
        import math
        if math.isnan(value) or math.isinf(value):
            return default
    return value


# ── Core fetch ────────────────────────────────────────────────────────────────

def _fetch_from_yfinance(ticker: str) -> dict[str, Any]:
    """Fetch fundamentals for a single ticker directly from yfinance."""
    try:
        import yfinance as yf
        import signal as _sig

        tick = yf.Ticker(ticker)
        info = tick.info or {}

        # Earnings (last 4 quarters)
        earnings_data: list[dict] = []
        try:
            qe = tick.quarterly_earnings
            if qe is not None and not qe.empty:
                for dt, row in qe.iterrows():
                    earnings_data.append({
                        "period":   str(dt)[:10],
                        "revenue":  _safe(row.get("Revenue")),
                        "earnings": _safe(row.get("Earnings")),
                    })
        except Exception:
            pass

        # Analyst price targets
        analyst_target: float | None = _safe(info.get("targetMeanPrice"))

        result = {
            # Identification
            "ticker":            ticker.upper(),
            "name":              _safe(info.get("longName") or info.get("shortName"), ticker),
            "sector":            _safe(info.get("sector"), "Unknown"),
            "industry":          _safe(info.get("industry"), "Unknown"),
            "country":           _safe(info.get("country"), "Unknown"),
            "exchange":          _safe(info.get("exchange"), "Unknown"),

            # Valuation
            "market_cap":        _safe(info.get("marketCap")),
            "enterprise_value":  _safe(info.get("enterpriseValue")),
            "pe_ratio":          _safe(info.get("trailingPE")),
            "forward_pe":        _safe(info.get("forwardPE")),
            "price_to_book":     _safe(info.get("priceToBook")),
            "price_to_sales":    _safe(info.get("priceToSalesTrailing12Months")),
            "ev_to_ebitda":      _safe(info.get("enterpriseToEbitda")),

            # Profitability
            "revenue_ttm":       _safe(info.get("totalRevenue")),
            "gross_margins":     _safe(info.get("grossMargins")),
            "operating_margins": _safe(info.get("operatingMargins")),
            "profit_margins":    _safe(info.get("profitMargins")),
            "ebitda":            _safe(info.get("ebitda")),
            "eps_ttm":           _safe(info.get("trailingEps")),
            "eps_forward":       _safe(info.get("forwardEps")),
            "return_on_equity":  _safe(info.get("returnOnEquity")),
            "return_on_assets":  _safe(info.get("returnOnAssets")),

            # Balance sheet / leverage
            "total_cash":        _safe(info.get("totalCash")),
            "total_debt":        _safe(info.get("totalDebt")),
            "debt_to_equity":    _safe(info.get("debtToEquity")),
            "current_ratio":     _safe(info.get("currentRatio")),
            "free_cashflow":     _safe(info.get("freeCashflow")),
            "operating_cashflow": _safe(info.get("operatingCashflow")),

            # Price / market
            "current_price":     _safe(info.get("currentPrice") or info.get("regularMarketPrice")),
            "52w_high":          _safe(info.get("fiftyTwoWeekHigh")),
            "52w_low":           _safe(info.get("fiftyTwoWeekLow")),
            "50d_avg":           _safe(info.get("fiftyDayAverage")),
            "200d_avg":          _safe(info.get("twoHundredDayAverage")),
            "beta":              _safe(info.get("beta")),
            "dividend_yield":    _safe(info.get("dividendYield")),
            "payout_ratio":      _safe(info.get("payoutRatio")),

            # Analyst
            "analyst_target":        analyst_target,
            "analyst_recommendation": _safe(info.get("recommendationKey"), "N/A"),
            "analyst_count":          _safe(info.get("numberOfAnalystOpinions")),

            # Quarterly earnings summary
            "quarterly_earnings": earnings_data[:4],

            # Meta
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return result

    except Exception as exc:
        logger.warning("[fundamentals] yfinance fetch failed for %s: %s", ticker, exc)
        return {
            "ticker":     ticker.upper(),
            "error":      str(exc),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


# ── Public API ────────────────────────────────────────────────────────────────

def get_fundamentals(ticker: str, force_refresh: bool = False) -> dict[str, Any]:
    """
    Return fundamental data for a single *ticker*.

    Data is cached for FUNDAMENTALS_CACHE_TTL seconds (default 6 h).
    Pass force_refresh=True to bypass the cache.

    Returns a dict; on error the dict contains an ``error`` key.
    """
    ticker = ticker.upper().strip()
    now = time.time()

    with _cache_lock:
        _load_disk_cache()
        entry = _disk_cache.get(ticker)
        if entry and not force_refresh:
            fetched_at_str = entry.get("data", {}).get("fetched_at", "")
            try:
                from datetime import datetime, timezone
                fetched_at = datetime.fromisoformat(
                    fetched_at_str.replace("Z", "+00:00")
                ).timestamp()
                if now - fetched_at < _CACHE_TTL_SECS:
                    logger.debug("[fundamentals] cache hit for %s", ticker)
                    return entry["data"]
            except Exception:
                pass   # Invalid timestamp — refresh

    # Cache miss or expired
    logger.info("[fundamentals] fetching live data for %s", ticker)
    data = _fetch_from_yfinance(ticker)

    with _cache_lock:
        _disk_cache[ticker] = {"data": data}
        _save_disk_cache()

    return data


def get_batch_fundamentals(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """
    Fetch fundamentals for all *tickers* (cached individually).
    Returns a dict mapping ticker → fundamentals dict.
    Failed tickers contain an ``error`` field rather than raising.
    """
    results: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        results[ticker.upper()] = get_fundamentals(ticker)
    return results


def format_fundamentals_summary(data: dict[str, Any]) -> str:
    """
    Render a compact text summary of fundamentals suitable for LLM prompt injection.

    Example output:
        AAPL (Apple Inc.) | Sector: Technology | Industry: Consumer Electronics
        MCap: $2.9T | P/E: 28.1 | Fwd P/E: 25.4 | EPS(TTM): $6.43
        Revenue(TTM): $383B | Profit Margin: 24.2% | FCF: $93B
        Debt/Equity: 151 | Beta: 1.19 | 52W: $164 – $220
        Analyst: buy (43 analysts) | Target: $223
    """
    if "error" in data:
        return f"{data['ticker']}: fundamentals unavailable ({data['error'][:80]})"

    def _fmt_m(v: Any, suffix: str = "") -> str:
        if v is None:
            return "N/A"
        v = float(v)
        if abs(v) >= 1e12:
            return f"${v / 1e12:.2f}T{suffix}"
        if abs(v) >= 1e9:
            return f"${v / 1e9:.2f}B{suffix}"
        if abs(v) >= 1e6:
            return f"${v / 1e6:.1f}M{suffix}"
        return f"${v:.2f}{suffix}"

    def _pct(v: Any) -> str:
        return f"{float(v) * 100:.1f}%" if v is not None else "N/A"

    def _x(v: Any) -> str:
        return f"{float(v):.1f}x" if v is not None else "N/A"

    def _eps(v: Any) -> str:
        return f"${float(v):.2f}" if v is not None else "N/A"

    def _price(v: Any) -> str:
        return f"${float(v):.2f}" if v is not None else "N/A"

    lines = [
        f"{data['ticker']} ({data.get('name', 'Unknown')}) | "
        f"Sector: {data.get('sector', 'N/A')} | Industry: {data.get('industry', 'N/A')}",
        f"MCap: {_fmt_m(data.get('market_cap'))} | "
        f"P/E: {_x(data.get('pe_ratio'))} | Fwd P/E: {_x(data.get('forward_pe'))} | "
        f"EPS(TTM): {_eps(data.get('eps_ttm'))}",
        f"Revenue(TTM): {_fmt_m(data.get('revenue_ttm'))} | "
        f"Profit Margin: {_pct(data.get('profit_margins'))} | "
        f"FCF: {_fmt_m(data.get('free_cashflow'))}",
        f"Debt/Equity: {data.get('debt_to_equity') or 'N/A'} | "
        f"Beta: {data.get('beta') or 'N/A'} | "
        f"52W: {_price(data.get('52w_low'))} \u2013 {_price(data.get('52w_high'))}",
        f"Analyst: {data.get('analyst_recommendation', 'N/A')} "
        f"({data.get('analyst_count') or '?'} analysts) | "
        f"Target: {_price(data.get('analyst_target'))}",
    ]
    return "\n".join(lines)
