from __future__ import annotations

import logging
import os
import time
import json
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# API keys loaded from environment — never hardcode secrets in source.
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_TIMEOUT_SEC = float(os.getenv("LIVE_DATA_TIMEOUT_SEC", "6"))

# Optional Alpha Vantage key (free tier: 25 calls/day)
# Set ALPHA_VANTAGE_API_KEY in .env to enable.
ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# ── TTL cache for live indicators ─────────────────────────────────────────────
# Avoids making 30+ HTTP calls on every /intelligence/analyze request.
_CACHE_TTL: float = float(os.getenv("LIVE_DATA_CACHE_TTL", "300"))  # seconds
_cache_lock = Lock()
_cached_indicators: dict[str, float] = {}
_cached_details: dict[str, Any] = {}
_cache_expires_at: float = 0.0

# yfinance ticker → indicator key mapping (no API key needed)
_YF_MAP = {
    # ── US Equities ─────────────────────────────────────────────────────────
    "sp500":          "^GSPC",
    "nasdaq":         "^IXIC",
    "dow":            "^DJI",
    "russell2000":    "^RUT",    # Small-cap benchmark
    "vix":            "^VIX",
    # ── US Sectors (ETFs) ───────────────────────────────────────────────────
    "sector_tech":    "XLK",
    "sector_energy":  "XLE",
    "sector_finance": "XLF",
    "sector_health":  "XLV",
    "sector_consumer":"XLY",
    # ── India Indices ────────────────────────────────────────────────────────
    "nifty50":        "^NSEI",
    "sensex":         "^BSESN",
    "nifty_bank":     "^NSEBANK",
    "nifty_it":       "^CNXIT",
    "nifty_mid150":   "^NSEMDCP50",
    # ── Treasury Yields ─────────────────────────────────────────────────────
    "yield_10y":      "^TNX",
    "yield_30y":      "^TYX",
    "yield_2y":       "^IRX",
    # ── Commodities ─────────────────────────────────────────────────────────
    "gold":           "GC=F",
    "silver":         "SI=F",
    "oil_wti":        "CL=F",
    "oil_brent":      "BZ=F",
    "natural_gas":    "NG=F",
    "copper":         "HG=F",    # Industrial demand proxy
    # ── Forex ───────────────────────────────────────────────────────────────
    "dxy":            "DX-Y.NYB",
    "eur_usd":        "EURUSD=X",
    "gbp_usd":        "GBPUSD=X",
    "usd_inr":        "INR=X",
    "usd_jpy":        "JPY=X",
    "usd_cny":        "CNY=X",
    # ── Crypto ──────────────────────────────────────────────────────────────
    "btc_usd":        "BTC-USD",
    "eth_usd":        "ETH-USD",
    # ── Global Indices ───────────────────────────────────────────────────────
    "ftse100":        "^FTSE",
    "nikkei225":      "^N225",
    "hangseng":       "^HSI",
    "dax":            "^GDAXI",
}


def check_fred_key() -> dict[str, Any]:
    """
    Validates the FRED API key by fetching one known series.
    Returns a dict with 'valid', 'latency_ms', and 'sample' fields.
    """
    if not FRED_API_KEY:
        return {"valid": False, "error": "No FRED_API_KEY set", "latency_ms": None}
    t0 = time.time()
    try:
        resp = requests.get(
            FRED_URL,
            params={
                "series_id": "FEDFUNDS",
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            },
            timeout=REQUEST_TIMEOUT_SEC,
        )
        latency_ms = round((time.time() - t0) * 1000, 1)
        if resp.status_code == 200:
            obs = resp.json().get("observations", [{}])
            sample = obs[0] if obs else {}
            return {
                "valid": True,
                "latency_ms": latency_ms,
                "sample_series": "FEDFUNDS",
                "sample_value": sample.get("value"),
                "sample_date": sample.get("date"),
            }
        return {"valid": False, "http_status": resp.status_code, "latency_ms": latency_ms}
    except Exception as exc:
        return {"valid": False, "error": str(exc), "latency_ms": None}


def _latest_fred_value(series_id: str) -> float | None:
    if not FRED_API_KEY:
        return None
    try:
        resp = requests.get(
            FRED_URL,
            params={
                "series_id": series_id,
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 12,
            },
            timeout=REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        for obs in observations:
            value = obs.get("value")
            if value is None or value == ".":
                continue
            return float(value)
    except Exception:
        return None
    return None


def _fetch_worldbank_india() -> tuple[dict[str, float], list[str]]:
    """
    Fetches India macro data from World Bank Open Data API (free, no API key).
    Indicators: GDP growth, inflation, current account, FDI, external debt.
    Returns (results_dict, sources_list).
    """
    # World Bank indicator codes for India (country code: IND)
    wb_map = {
        "india_gdp_growth":       "NY.GDP.MKTP.KD.ZG",  # GDP growth %
        "india_inflation_cpi":    "FP.CPI.TOTL.ZG",     # CPI inflation %
        "india_current_account":  "BN.CAB.XOKA.GD.ZS",  # Current account % GDP
        "india_fdi_inflow":       "BX.KLT.DINV.WD.GD.ZS", # FDI net inflows % GDP
        "india_unemployment":     "SL.UEM.TOTL.ZS",     # Unemployment % labour force
        "india_ext_debt_gdp":     "GD.DOD.DECT.GN.ZS",  # External debt % GNI
    }
    WB_URL = "https://api.worldbank.org/v2/country/IND/indicator/{indicator}"
    results: dict[str, float] = {}
    sources: list[str] = []

    for key, indicator in wb_map.items():
        try:
            resp = requests.get(
                WB_URL.format(indicator=indicator),
                params={"format": "json", "mrv": 2, "per_page": 2},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            if resp.status_code != 200:
                continue
            payload = resp.json()
            # World Bank returns [metadata, [data_entries]]
            if len(payload) < 2 or not payload[1]:
                continue
            for entry in payload[1]:
                val = entry.get("value")
                if val is not None:
                    results[key] = round(float(val), 4)
                    sources.append(f"WorldBank:{indicator}")
                    break
        except Exception:
            continue

    return results, sources


def _fetch_alpha_vantage_forex() -> tuple[dict[str, float], list[str]]:
    """
    Fetches live forex rates via Alpha Vantage (free tier: 25 calls/day).
    Set ALPHA_VANTAGE_API_KEY env var to enable.
    """
    if not ALPHA_VANTAGE_API_KEY:
        return {}, []

    forex_pairs = [
        ("USD", "INR", "usd_inr_av"),
        ("USD", "CNY", "usd_cny_av"),
        ("EUR", "USD", "eur_usd_av"),
    ]
    results: dict[str, float] = {}
    sources: list[str] = []

    for from_cur, to_cur, key in forex_pairs:
        try:
            resp = requests.get(
                ALPHA_VANTAGE_URL,
                params={
                    "function": "CURRENCY_EXCHANGE_RATE",
                    "from_currency": from_cur,
                    "to_currency": to_cur,
                    "apikey": ALPHA_VANTAGE_API_KEY,
                },
                timeout=REQUEST_TIMEOUT_SEC,
            )
            data = resp.json()
            rate_info = data.get("Realtime Currency Exchange Rate", {})
            rate = rate_info.get("5. Exchange Rate")
            if rate:
                results[key] = round(float(rate), 4)
                sources.append(f"AlphaVantage:{from_cur}/{to_cur}")
        except Exception:
            continue

    return results, sources


def _fetch_yfinance_snapshot() -> tuple[dict[str, float], list[str]]:
    """Fetch live/near-real-time market data via yfinance (no API key required).

    Strategy:
      1. Try intraday batch download (period='1d', interval='2m') — gives the
         most recent 2-minute bar, i.e. effectively live prices.
      2. Fall back to (period='5d', interval='1d') if intraday returns no rows
         (e.g. all markets closed on a weekend/holiday).
      3. Final per-ticker fallback for any symbol still missing.
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        return {}, []

    results: dict[str, float] = {}
    sources: list[str] = []
    tickers = list(_YF_MAP.values())

    def _extract_closes(data: "pd.DataFrame") -> "pd.Series":
        """Return the last non-NaN row of the Close column regardless of
        whether yfinance returned single- or multi-level columns."""
        if data is None or data.empty:
            return pd.Series(dtype=float)
        close_col = data.get("Close") if hasattr(data, "get") else None
        if close_col is None and "Close" in data.columns:
            close_col = data["Close"]
        if close_col is None:
            return pd.Series(dtype=float)
        clean = close_col.dropna(how="all")
        if clean.empty:
            return pd.Series(dtype=float)
        last_row = clean.iloc[-1]
        # yfinance may return a scalar (single ticker) or a Series (multi)
        if isinstance(last_row, pd.Series):
            return last_row
        # Single-ticker scalar — the column header IS the ticker
        return pd.Series({data.columns.get_level_values(-1)[0]: last_row}
                         if hasattr(data.columns, "get_level_values")
                         else {tickers[0]: last_row})

    # ── Pass 1: intraday (live) ──────────────────────────────────────────────
    intraday_ok = False
    try:
        data_intraday = yf.download(
            tickers,
            period="1d",
            interval="2m",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        closes = _extract_closes(data_intraday)
        for key, ticker in _YF_MAP.items():
            val = closes.get(ticker) if hasattr(closes, "get") else None
            if val is not None and not pd.isna(float(val)):
                results[key] = round(float(val), 4)
                sources.append(f"yfinance_live:{ticker}")
                intraday_ok = True
    except Exception:
        pass

    # ── Pass 2: daily fallback for symbols still missing ────────────────────
    missing_tickers = {k: v for k, v in _YF_MAP.items() if k not in results}
    if missing_tickers:
        try:
            data_daily = yf.download(
                list(missing_tickers.values()),
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            closes_d = _extract_closes(data_daily)
            for key, ticker in missing_tickers.items():
                val = closes_d.get(ticker) if hasattr(closes_d, "get") else None
                if val is not None and not pd.isna(float(val)):
                    results[key] = round(float(val), 4)
                    sources.append(f"yfinance:{ticker}")
        except Exception:
            pass

    # ── Pass 3: per-ticker fallback for still-missing symbols ───────────────
    still_missing = {k: v for k, v in _YF_MAP.items() if k not in results}
    for key, ticker in still_missing.items():
        try:
            # fast_info gives real-time last price with minimal overhead
            fi = yf.Ticker(ticker).fast_info
            val = getattr(fi, "last_price", None)
            if val is None:
                hist = yf.Ticker(ticker).history(period="5d")
                if not hist.empty:
                    val = float(hist["Close"].dropna().iloc[-1])
            if val is not None:
                results[key] = round(float(val), 4)
                sources.append(f"yfinance_fallback:{ticker}")
        except Exception:
            pass

    return results, sources


def fetch_live_indicators() -> tuple[dict[str, float], dict[str, Any]]:
    """
    Fetches live macro indicators from all available sources.

    Sources (in priority order):
      1. yfinance       – equities, commodities, forex, yields (no key required)
      2. FRED           – US macro: CPI, GDP, unemployment, credit spreads, yields
      3. World Bank     – India macro: GDP, inflation, FDI, current account (no key)
      4. Alpha Vantage  – forex real-time cross-check (requires ALPHA_VANTAGE_API_KEY)

    Results are cached for LIVE_DATA_CACHE_TTL seconds (default 300s) to avoid
    hammering external APIs on every request.

    Returns:
        indicators: flat dict of indicator_name → float value
        details:    metadata dict (sources used, missing keys, fetch time)
    """
    global _cached_indicators, _cached_details, _cache_expires_at

    now = time.time()
    with _cache_lock:
        if _cached_indicators and now < _cache_expires_at:
            logger.debug("[live_market_data] Returning cached indicators (TTL %.0fs remaining)",
                         _cache_expires_at - now)
            return dict(_cached_indicators), dict(_cached_details)

    indicators: dict[str, float] = {}
    details: dict[str, Any] = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [],
        "missing": [],
        "source_coverage": {},
    }

    # ── 1. yfinance (always attempted) ──────────────────────────────────────
    yf_data, yf_sources = _fetch_yfinance_snapshot()
    indicators.update(yf_data)
    details["sources"].extend(yf_sources)
    details["source_coverage"]["yfinance"] = len(yf_data)

    # ── 2. FRED (US macro — authoritative, overrides yfinance where both exist)
    fred_map = {
        # ── Policy rates ───────────────────────────────────────────────────
        "fed_funds_rate":       "FEDFUNDS",
        # ── Treasury yields ────────────────────────────────────────────────
        "yield_10y":            "DGS10",
        "yield_2y":             "DGS2",
        "yield_30y":            "DGS30",
        "yield_1y":             "DGS1",
        "yield_3m":             "DTB3",
        # ── Inflation ──────────────────────────────────────────────────────
        "inflation_cpi":        "CPIAUCSL",
        "inflation_core_cpi":   "CPILFESL",   # Core CPI (ex food & energy)
        "pce_deflator":         "PCEPI",       # PCE — Fed's preferred measure
        "pce_core":             "PCEPILFE",    # Core PCE
        "breakeven_5y":         "T5YIE",       # 5Y breakeven inflation
        "breakeven_10y":        "T10YIE",      # 10Y breakeven inflation
        # ── Growth & labour ────────────────────────────────────────────────
        "gdp_growth":           "A191RL1Q225SBEA",
        "unemployment":         "UNRATE",
        "initial_claims":       "ICSA",        # Initial jobless claims (weekly)
        "continued_claims":     "CCSA",        # Continued claims
        "jolts_openings":       "JTSJOL",      # Job openings (JOLTS)
        "nonfarm_payrolls":     "PAYEMS",      # Nonfarm payrolls
        "participation_rate":   "CIVPART",     # Labour force participation
        # ── Activity ───────────────────────────────────────────────────────
        "pmi_mfg":              "NAPM",
        "us_retail_sales":      "RSAFS",
        "us_industrial_prod":   "INDPRO",
        "capacity_utilization": "TCU",         # Capacity utilization %
        "us_housing_starts":    "HOUST",
        "us_building_permits":  "PERMIT",
        "us_trade_balance":     "BOPGSTB",
        "consumer_sentiment":   "UMCSENT",
        "conf_board_lei":       "USSLIND",     # Conference Board LEI
        # ── Credit & spreads ───────────────────────────────────────────────
        "credit_hy":            "BAMLH0A0HYM2",
        "credit_ig":            "BAMLC0A0CM",
        "credit_bb":            "BAMLH0A1HYBBm",   # BB-rated spread
        "mort_rate_30y":        "MORTGAGE30US",    # 30Y fixed mortgage rate
        "ted_spread":           "TEDRATE",          # TED spread (bank risk)
        # ── Money & Fed balance sheet ──────────────────────────────────────
        "m2_money_supply":      "M2SL",
        "fed_balance_sheet":    "WALCL",           # Fed total assets
        "m2_velocity":          "M2V",
        # ── Markets ────────────────────────────────────────────────────────
        "dxy":                  "DTWEXBGS",
        "gold":                 "GOLDAMGBD228NLBM",
        "oil_wti":              "DCOILWTICO",
        "oil_brent":            "DCOILBRENTEU",
        "sp500":                "SP500",
        "vix_fred":             "VIXCLS",
        # ── Real rates ─────────────────────────────────────────────────────
        "real_rate_10y":        "DFII10",           # 10Y TIPS yield (real)
        "real_rate_5y":         "DFII5",
    }
    fred_count = 0
    if FRED_API_KEY:
        for key, series in fred_map.items():
            value = _latest_fred_value(series)
            if value is None:
                details["missing"].append(f"FRED:{key}")
                continue
            indicators[key] = value
            details["sources"].append(f"FRED:{series}")
            fred_count += 1
        details["source_coverage"]["FRED"] = fred_count
    else:
        details["sources"].append("FRED:SKIPPED(no API key)")
        details["source_coverage"]["FRED"] = 0

    # ── 3. World Bank — India macro (free, no API key, annual data) ──────────
    wb_data, wb_sources = _fetch_worldbank_india()
    indicators.update(wb_data)
    details["sources"].extend(wb_sources)
    details["source_coverage"]["WorldBank"] = len(wb_data)

    # ── 4. Alpha Vantage — real-time forex (optional, 25 calls/day free) ────
    av_data, av_sources = _fetch_alpha_vantage_forex()
    indicators.update(av_data)
    details["sources"].extend(av_sources)
    details["source_coverage"]["AlphaVantage"] = len(av_data)

    # ── 5. Derived / computed indicators ─────────────────────────────────────
    if "yield_10y" in indicators and "yield_2y" in indicators:
        indicators["yield_curve"] = round(
            (indicators["yield_10y"] - indicators["yield_2y"]) * 100, 1
        )
    if "yield_10y" in indicators and "yield_3m" in indicators:
        indicators["yield_curve_10y3m"] = round(
            (indicators["yield_10y"] - indicators["yield_3m"]) * 100, 1
        )
    if "yield_30y" in indicators and "yield_2y" in indicators:
        indicators["yield_curve_30y2y"] = round(
            (indicators["yield_30y"] - indicators["yield_2y"]) * 100, 1
        )
    # Real rate proxy: 10Y nominal minus CPI (if TIPS yield not available)
    if "real_rate_10y" not in indicators:
        if "yield_10y" in indicators and "inflation_cpi" in indicators:
            indicators["real_rate_proxy"] = round(
                indicators["yield_10y"] - indicators["inflation_cpi"], 2
            )
    else:
        indicators["real_rate_proxy"] = indicators["real_rate_10y"]
    # Fed real rate (funds rate minus core PCE)
    if "fed_funds_rate" in indicators and "pce_core" in indicators:
        indicators["fed_real_rate"] = round(
            indicators["fed_funds_rate"] - indicators["pce_core"], 2
        )
    # Breakeven spread (inflation expectation measure)
    if "breakeven_10y" in indicators and "breakeven_5y" in indicators:
        indicators["breakeven_5y10y_slope"] = round(
            indicators["breakeven_10y"] - indicators["breakeven_5y"], 3
        )
    # HY - IG gap (credit risk appetite)
    if "credit_hy" in indicators and "credit_ig" in indicators:
        indicators["credit_spread_gap"] = round(
            indicators["credit_hy"] - indicators["credit_ig"], 2
        )
    # Equity risk premium proxy: earnings yield (1/PE) minus 10Y
    # Approximated if sp500 data is available (no PE data from FRED)
    # Term premium: 30y - 2y spread
    if "yield_30y" in indicators and "yield_3m" in indicators:
        indicators["term_premium_proxy"] = round(
            (indicators["yield_30y"] - indicators["yield_3m"]) * 100, 1
        )

    details["values"] = {k: indicators[k] for k in sorted(indicators.keys())}
    details["total_indicators"] = len(indicators)

    # Store in TTL cache
    with _cache_lock:
        _cached_indicators = dict(indicators)
        _cached_details = dict(details)
        _cache_expires_at = time.time() + _CACHE_TTL
    logger.debug("[live_market_data] Fetched %d indicators; cached for %.0fs",
                 len(indicators), _CACHE_TTL)

    return indicators, details


def invalidate_live_data_cache() -> None:
    """Force the next fetch_live_indicators() call to bypass the TTL cache."""
    global _cache_expires_at
    with _cache_lock:
        _cache_expires_at = 0.0
    logger.info("[live_market_data] TTL cache invalidated.")


# ═══════════════════════════════════════════════════════════════════════════════
# Efficiency Testing & Benchmarking
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_data_sources(verbose: bool = True) -> dict[str, Any]:
    """
    Benchmarks every data source individually.

    Measures for each source:
      - Latency (ms)
      - Coverage (# indicators returned)
      - Data freshness (days since most recent value)
      - Success/failure

    Then runs the full fetch_live_indicators() and scores overall efficiency.

    Returns a report dict. Pass verbose=True to print a formatted summary.
    """
    report: dict[str, Any] = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {},
        "overall": {},
    }

    # ── 1. FRED key check ────────────────────────────────────────────────────
    t0 = time.time()
    fred_check = check_fred_key()
    fred_latency = round((time.time() - t0) * 1000, 1)
    report["sources"]["FRED"] = {
        "valid": fred_check["valid"],
        "latency_ms": fred_check.get("latency_ms", fred_latency),
        "sample_value": fred_check.get("sample_value"),
        "sample_date": fred_check.get("sample_date"),
        "note": "US macro authority (CPI, GDP, unemployment, credit spreads)",
    }
    if not fred_check["valid"]:
        report["sources"]["FRED"]["error"] = fred_check.get("error", "key invalid")

    # ── 2. yfinance snapshot ─────────────────────────────────────────────────
    t0 = time.time()
    yf_data, yf_sources = _fetch_yfinance_snapshot()
    yf_latency = round((time.time() - t0) * 1000, 1)
    report["sources"]["yfinance"] = {
        "valid": len(yf_data) > 0,
        "latency_ms": yf_latency,
        "coverage": len(yf_data),
        "total_symbols": len(_YF_MAP),
        "coverage_pct": round(len(yf_data) / len(_YF_MAP) * 100, 1),
        "note": "Equities, indices, forex, commodities, crypto (no key)",
    }

    # ── 3. World Bank India ──────────────────────────────────────────────────
    t0 = time.time()
    wb_data, _ = _fetch_worldbank_india()
    wb_latency = round((time.time() - t0) * 1000, 1)
    report["sources"]["WorldBank"] = {
        "valid": len(wb_data) > 0,
        "latency_ms": wb_latency,
        "coverage": len(wb_data),
        "note": "India macro: GDP, inflation, FDI, unemployment (annual, no key)",
    }

    # ── 4. Alpha Vantage ─────────────────────────────────────────────────────
    t0 = time.time()
    av_data, _ = _fetch_alpha_vantage_forex()
    av_latency = round((time.time() - t0) * 1000, 1)
    report["sources"]["AlphaVantage"] = {
        "valid": len(av_data) > 0 if ALPHA_VANTAGE_API_KEY else False,
        "latency_ms": av_latency,
        "coverage": len(av_data),
        "enabled": bool(ALPHA_VANTAGE_API_KEY),
        "note": "Real-time forex cross-check (set ALPHA_VANTAGE_API_KEY env var)",
    }

    # ── 5. Full pipeline efficiency run ──────────────────────────────────────
    t0 = time.time()
    indicators, details = fetch_live_indicators()
    total_latency = round((time.time() - t0) * 1000, 1)

    total_possible = len(_YF_MAP) + len({  # max theoretical indicators
        "fed_funds_rate", "inflation_cpi", "gdp_growth", "unemployment",
        "pmi_mfg", "credit_hy", "credit_ig", "consumer_sentiment",
        "yield_10y", "yield_2y", "yield_30y", "dxy", "gold", "oil_wti",
        "oil_brent", "sp500", "us_housing_starts", "us_retail_sales",
        "us_industrial_prod", "m2_money_supply", "fed_balance_sheet",
        "india_gdp_growth", "india_inflation_cpi", "india_current_account",
        "india_fdi_inflow", "india_unemployment", "india_ext_debt_gdp",
    })
    coverage_pct = round(len(indicators) / total_possible * 100, 1)

    # Efficiency score (0-100):
    #   40pts coverage, 30pts latency (<10s=30, <20s=20, <30s=10, else 0), 30pts for all sources up
    latency_score = 30 if total_latency < 10000 else (20 if total_latency < 20000 else (10 if total_latency < 30000 else 0))
    coverage_score = round(min(coverage_pct / 100 * 40, 40), 1)
    sources_up = sum(1 for s in report["sources"].values() if s.get("valid", False))
    source_score = round(sources_up / len(report["sources"]) * 30, 1)
    efficiency_score = round(coverage_score + latency_score + source_score, 1)

    report["overall"] = {
        "total_indicators_fetched": len(indicators),
        "total_possible_indicators": total_possible,
        "coverage_pct": coverage_pct,
        "total_latency_ms": total_latency,
        "sources_up": sources_up,
        "sources_total": len(report["sources"]),
        "missing_count": len(details["missing"]),
        "missing_keys": details["missing"],
        "efficiency_score": efficiency_score,
        "efficiency_grade": (
            "A" if efficiency_score >= 80 else
            "B" if efficiency_score >= 65 else
            "C" if efficiency_score >= 50 else "D"
        ),
        "source_breakdown": details["source_coverage"],
    }

    if verbose:
        _print_benchmark_report(report)

    return report


def _print_benchmark_report(report: dict[str, Any]) -> None:
    """Prints a human-readable benchmark report to stdout."""
    SEP = "─" * 60
    print(f"\n{'═'*60}")
    print(f"  LIVE DATA BENCHMARK REPORT")
    print(f"  {report['run_at_utc']} UTC")
    print(f"{'═'*60}")

    for name, info in report["sources"].items():
        status = "✓ OK" if info.get("valid") else ("✗ DISABLED" if info.get("enabled") is False else "✗ FAIL")
        latency = f"{info.get('latency_ms','?')} ms"
        coverage = f"{info.get('coverage', '?')} indicators"
        print(f"\n  [{name}]  {status}  |  latency: {latency}  |  {coverage}")
        print(f"    {info.get('note','')}")
        if not info.get("valid") and info.get("error"):
            print(f"    ERROR: {info['error']}")
        if name == "yfinance":
            print(f"    Coverage: {info.get('coverage_pct','?')}% ({info.get('coverage')}/{info.get('total_symbols')} symbols)")
        if name == "FRED" and info.get("sample_value"):
            print(f"    Sample FEDFUNDS={info['sample_value']}% on {info['sample_date']}")

    o = report["overall"]
    print(f"\n{SEP}")
    print(f"  OVERALL EFFICIENCY")
    print(SEP)
    print(f"  Score         : {o['efficiency_score']}/100  →  Grade {o['efficiency_grade']}")
    print(f"  Coverage      : {o['total_indicators_fetched']}/{o['total_possible_indicators']} indicators  ({o['coverage_pct']}%)")
    print(f"  Total latency : {o['total_latency_ms']} ms")
    print(f"  Sources up    : {o['sources_up']}/{o['sources_total']}")
    if o["missing_keys"]:
        print(f"  Missing       : {', '.join(o['missing_keys'][:8])}{'...' if len(o['missing_keys'])>8 else ''}")
    print(f"  Breakdown     : {json.dumps(o['source_breakdown'])}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    """
    Run directly to benchmark all data sources:
        python -m intelligence.live_market_data
    or:
        python intelligence/live_market_data.py
    """
    import argparse
    parser = argparse.ArgumentParser(description="Live data source benchmark")
    parser.add_argument("--fred-only", action="store_true", help="Only test FRED API key")
    parser.add_argument("--fetch", action="store_true", help="Fetch and print all indicators")
    args = parser.parse_args()

    if args.fred_only:
        result = check_fred_key()
        if result["valid"]:
            print(f"✓  FRED API key valid  |  FEDFUNDS={result['sample_value']}%  |  date={result['sample_date']}  |  latency={result['latency_ms']}ms")
        else:
            print(f"✗  FRED API key FAILED: {result.get('error', result.get('http_status', 'unknown error'))}")
    elif args.fetch:
        indicators, details = fetch_live_indicators()
        print(json.dumps(details, indent=2, default=str))
    else:
        # Default: full benchmark
        benchmark_data_sources(verbose=True)
