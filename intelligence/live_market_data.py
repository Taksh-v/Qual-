"""
live_market_data.py  (performance-optimised)
─────────────────────────────────────────────
Design
  1. THREE-TIER TTL CACHE
     • price_cache   yfinance prices
                     TTL = 90 s  (market OPEN)   — live price refresh
                     TTL = 6 h   (market CLOSED)  — no point polling closed exchange
     • fred_cache    FRED macro series (CPI, GDP, unemployment …)
                     TTL = 6 h always — they publish monthly / weekly
     • wb_cache      World Bank India  (annual data)
                     TTL = 24 h

  2. MARKET-HOURS-AWARE FETCHING
     Tickers are tagged to an exchange group with UTC open/close windows.
     Only tickers on OPEN exchanges are fetched live; closed-exchange tickers
     are served from the stale price cache — zero HTTP cost.

  3. FULLY PARALLEL FRED FETCHING
     All 36 FRED series are dispatched simultaneously in a ThreadPoolExecutor.
     Latency drops from O(N × timeout) → O(1 × timeout + overhead).

  4. PARALLEL SOURCE DISPATCH
     yfinance  ║  FRED  ║  World Bank (only if 24 h cache is stale)
     All three submitted concurrently; results merged after futures settle.

  5. BACKWARD-COMPATIBLE PUBLIC API
     fetch_live_indicators() → (indicators: dict, details: dict)
     invalidate_live_data_cache() clears price + FRED caches.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import requests
from intelligence.indicator_parser import sanitize_indicator_values

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── API keys ──────────────────────────────────────────────────────────────────
FRED_API_KEY:          str = os.getenv("FRED_API_KEY", "")
ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FRED_URL          = "https://api.stlouisfed.org/fred/series/observations"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# ── Per-source network timeouts ───────────────────────────────────────────────
_FRED_TIMEOUT = float(os.getenv("FRED_TIMEOUT_SEC",  "15"))  # per series
_YF_TIMEOUT   = float(os.getenv("YF_TIMEOUT_SEC",    "30"))  # total yfinance budget per call
_YF_BATCH_TIMEOUT = max(3.0, float(os.getenv("YF_BATCH_TIMEOUT_SEC", "9")))
_YF_PASS3_TIMEOUT = max(2.0, float(os.getenv("YF_PASS3_TIMEOUT_SEC", "6")))
_YF_ENABLE_PASS3  = os.getenv("YF_ENABLE_PASS3", "0").strip().lower() not in ("0", "false", "no")
_WB_TIMEOUT   = float(os.getenv("WB_TIMEOUT_SEC",    "15"))
_AV_TIMEOUT   = float(os.getenv("AV_TIMEOUT_SEC",    "10"))

# ── Thread-pool sizes ─────────────────────────────────────────────────────────
_FRED_WORKERS = int(os.getenv("FRED_PARALLEL_WORKERS", "16"))
_YF_WORKERS   = int(os.getenv("YF_PARALLEL_WORKERS",   "8"))

# ── TTL constants (seconds) ───────────────────────────────────────────────────
_TTL_PRICE_OPEN   = int(os.getenv("PRICE_CACHE_TTL_OPEN",   "90"))     # 1.5 min
_TTL_PRICE_CLOSED = int(os.getenv("PRICE_CACHE_TTL_CLOSED", "21600"))  # 6 h
_TTL_FRED         = int(os.getenv("FRED_CACHE_TTL",         "21600"))  # 6 h
_TTL_WORLDBANK    = int(os.getenv("WB_CACHE_TTL",           "86400"))  # 24 h
_TTL_AV           = int(os.getenv("AV_CACHE_TTL",           "300"))    # 5 min

# ═══════════════════════════════════════════════════════════════════════════════
# Exchange-hours registry  (all times UTC, weekday int Mon=0 … Sun=6)
# ═══════════════════════════════════════════════════════════════════════════════

_EXCHANGE_HOURS: dict[str, dict] = {
    "US_EQUITY": {
        # NYSE/NASDAQ  09:30–16:00 ET = 14:30–21:00 UTC
        "open": (14, 30), "close": (21, 0),
        "weekdays": {0, 1, 2, 3, 4},
    },
    "INDIA_EQUITY": {
        # NSE/BSE  09:15–15:30 IST = 03:45–10:00 UTC
        "open": (3, 45), "close": (10, 0),
        "weekdays": {0, 1, 2, 3, 4},
    },
    "EUROPE_EQUITY": {
        # XETRA/LSE  08:00–16:30 CET ≈ 07:00–15:30 UTC
        "open": (7, 0), "close": (15, 30),
        "weekdays": {0, 1, 2, 3, 4},
    },
    "JAPAN_EQUITY": {
        # TSE  09:00–15:30 JST = 00:00–06:30 UTC
        "open": (0, 0), "close": (6, 30),
        "weekdays": {0, 1, 2, 3, 4},
    },
    "HONGKONG_EQUITY": {
        # HKEX  09:30–16:00 HKT = 01:30–08:00 UTC
        "open": (1, 30), "close": (8, 0),
        "weekdays": {0, 1, 2, 3, 4},
    },
    "COMMODITIES": {
        # CME/NYMEX — nearly 24 h on weekdays; simplified: 23:00–22:00 UTC  (crosses midnight)
        "open": (23, 0), "close": (22, 0),
        "weekdays": {0, 1, 2, 3, 4},
        "crosses_midnight": True,
    },
    "FOREX": {
        # Spot FX 24/5  (Sun 22:00 – Fri 22:00 UTC)
        "always_open_weekdays": True,
        "weekdays": {0, 1, 2, 3, 4, 6},   # 6 = Sunday
    },
    "BONDS": {
        # Treasury cash market follows US session
        "open": (14, 30), "close": (21, 0),
        "weekdays": {0, 1, 2, 3, 4},
    },
    "CRYPTO": {
        "always_open": True,
    },
}

# ── Ticker → exchange group ───────────────────────────────────────────────────
_TICKER_EXCHANGE: dict[str, str] = {
    "sp500": "US_EQUITY", "nasdaq": "US_EQUITY", "dow": "US_EQUITY",
    "russell2000": "US_EQUITY", "vix": "US_EQUITY",
    "sector_tech": "US_EQUITY", "sector_energy": "US_EQUITY",
    "sector_finance": "US_EQUITY", "sector_health": "US_EQUITY",
    "sector_consumer": "US_EQUITY",
    "nifty50": "INDIA_EQUITY", "sensex": "INDIA_EQUITY",
    "nifty_bank": "INDIA_EQUITY", "nifty_it": "INDIA_EQUITY",
    "nifty_mid150": "INDIA_EQUITY",
    "ftse100": "EUROPE_EQUITY", "dax": "EUROPE_EQUITY",
    "nikkei225": "JAPAN_EQUITY", "hangseng": "HONGKONG_EQUITY",
    "yield_10y": "BONDS", "yield_30y": "BONDS", "yield_2y": "BONDS",
    "gold": "COMMODITIES", "silver": "COMMODITIES",
    "oil_wti": "COMMODITIES", "oil_brent": "COMMODITIES",
    "natural_gas": "COMMODITIES", "copper": "COMMODITIES",
    "dxy": "FOREX", "eur_usd": "FOREX", "gbp_usd": "FOREX",
    "usd_inr": "FOREX", "usd_jpy": "FOREX", "usd_cny": "FOREX",
    "btc_usd": "CRYPTO", "eth_usd": "CRYPTO",
}

_YF_MAP: dict[str, str] = {
    "sp500": "^GSPC", "nasdaq": "^IXIC", "dow": "^DJI",
    "russell2000": "^RUT", "vix": "^VIX",
    "sector_tech": "XLK", "sector_energy": "XLE", "sector_finance": "XLF",
    "sector_health": "XLV", "sector_consumer": "XLY",
    "nifty50": "^NSEI", "sensex": "^BSESN", "nifty_bank": "^NSEBANK",
    "nifty_it": "^CNXIT", "nifty_mid150": "^NSEMDCP50",
    "yield_10y": "^TNX", "yield_30y": "^TYX", "yield_2y": "^IRX",
    "gold": "GC=F", "silver": "SI=F", "oil_wti": "CL=F", "oil_brent": "BZ=F",
    "natural_gas": "NG=F", "copper": "HG=F",
    "dxy": "DX-Y.NYB", "eur_usd": "EURUSD=X", "gbp_usd": "GBPUSD=X",
    "usd_inr": "INR=X", "usd_jpy": "JPY=X", "usd_cny": "CNY=X",
    "btc_usd": "BTC-USD", "eth_usd": "ETH-USD",
    "ftse100": "^FTSE", "nikkei225": "^N225", "hangseng": "^HSI", "dax": "^GDAXI",
}

_FRED_MAP: dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "yield_10y": "DGS10", "yield_2y": "DGS2", "yield_30y": "DGS30",
    "yield_1y": "DGS1", "yield_3m": "DTB3",
    "inflation_cpi": "CPIAUCSL", "inflation_core_cpi": "CPILFESL",
    "pce_deflator": "PCEPI", "pce_core": "PCEPILFE",
    "breakeven_5y": "T5YIE", "breakeven_10y": "T10YIE",
    "gdp_growth": "A191RL1Q225SBEA",
    "unemployment": "UNRATE",
    "initial_claims": "ICSA", "continued_claims": "CCSA",
    "jolts_openings": "JTSJOL", "nonfarm_payrolls": "PAYEMS",
    "participation_rate": "CIVPART",
    "pmi_mfg": "NAPM",
    "us_retail_sales": "RSAFS", "us_industrial_prod": "INDPRO",
    "capacity_utilization": "TCU",
    "us_housing_starts": "HOUST", "us_building_permits": "PERMIT",
    "us_trade_balance": "BOPGSTB",
    "consumer_sentiment": "UMCSENT", "conf_board_lei": "USSLIND",
    "credit_hy": "BAMLH0A0HYM2", "credit_ig": "BAMLC0A0CM",
    "credit_bb": "BAMLH0A1HYBBm",
    "mort_rate_30y": "MORTGAGE30US", "ted_spread": "TEDRATE",
    "m2_money_supply": "M2SL", "fed_balance_sheet": "WALCL", "m2_velocity": "M2V",
    "gold": "GOLDAMGBD228NLBM",
    "oil_wti": "DCOILWTICO", "oil_brent": "DCOILBRENTEU",
    "sp500": "SP500", "vix_fred": "VIXCLS",
    "real_rate_10y": "DFII10", "real_rate_5y": "DFII5",
}

_FRED_UNITS: dict[str, str] = {
    "inflation_cpi": "pc1",
    "inflation_core_cpi": "pc1",
    "pce_deflator": "pc1",
    "pce_core": "pc1",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Thread-safe three-tier cache
# ═══════════════════════════════════════════════════════════════════════════════

from intelligence.cache_utils import _TieredCache

_price_cache = _TieredCache()
_fred_cache  = _TieredCache()
_wb_cache    = _TieredCache()
_av_cache    = _TieredCache()

_final_lock: Lock = Lock()
_final_cache:         dict[str, float] = {}
_final_details:       dict[str, Any]   = {}
_final_cache_expires: float            = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Market-hours helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _is_exchange_open(group: str, now_utc: datetime | None = None) -> bool:
    """True if the exchange group is currently trading."""
    cfg = _EXCHANGE_HOURS.get(group)
    if not cfg:
        return True   # unknown group — play safe
    if cfg.get("always_open"):
        return True
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    wd = now_utc.weekday()
    if wd not in cfg.get("weekdays", set()):
        return False
    if cfg.get("always_open_weekdays"):
        return True
    oh, om = cfg["open"]
    ch, cm = cfg["close"]
    t   = now_utc.hour * 60 + now_utc.minute
    ot  = oh * 60 + om
    ct  = ch * 60 + cm
    if cfg.get("crosses_midnight"):
        return t >= ot or t < ct
    return ot <= t < ct


def any_price_market_open() -> bool:
    """True if at least one price-producing exchange is open right now."""
    return any(
        _is_exchange_open(g)
        for g in ["US_EQUITY", "INDIA_EQUITY", "EUROPE_EQUITY",
                  "JAPAN_EQUITY", "COMMODITIES", "FOREX", "CRYPTO"]
    )


def get_open_exchanges() -> list[str]:
    return [g for g in _EXCHANGE_HOURS if _is_exchange_open(g)]


def market_status_summary() -> dict[str, Any]:
    """Human-readable market-status dict.  Served by GET /market_data/status."""
    now = datetime.now(timezone.utc)
    return {
        "checked_at_utc": now.isoformat(),
        "any_price_market_open": any_price_market_open(),
        "open_exchanges": get_open_exchanges(),
        "exchange_detail": {
            g: {"open": _is_exchange_open(g, now)}
            for g in _EXCHANGE_HOURS
        },
        "price_cache_ttl_active_s": (
            _TTL_PRICE_OPEN if any_price_market_open() else _TTL_PRICE_CLOSED
        ),
        "fred_cache_ttl_s": _TTL_FRED,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FRED — fully parallel fetcher
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_single_fred(key: str, series_id: str, units: str | None = None) -> tuple[str, float | None]:
    if not FRED_API_KEY:
        return key, None
    try:
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 24,
        }
        if units:
            params["units"] = units
        r = requests.get(
            FRED_URL,
            params=params,
            timeout=_FRED_TIMEOUT,
        )
        r.raise_for_status()
        for obs in r.json().get("observations", []):
            v = obs.get("value")
            if v and v != ".":
                return key, float(v)
    except Exception:
        pass
    return key, None


def _fetch_fred_parallel() -> tuple[dict[str, float], list[str]]:
    """
    Dispatch all FRED series concurrently.
    Series already in the 6h cache are served instantly; only stale ones make HTTP calls.
    Worst-case wall time ≈ one round-trip timeout instead of N × timeout.
    """
    results: dict[str, float] = {}
    sources: list[str]        = []
    to_fetch: list[tuple[str, str, str | None]] = []

    for key, series in _FRED_MAP.items():
        units = _FRED_UNITS.get(key)
        cache_key = f"f:{series}:{units or 'lin'}"
        val, fresh = _fred_cache.get(cache_key)
        if fresh and val is not None:
            results[key] = val
        else:
            stale = _fred_cache.get_stale(cache_key)
            if stale is not None:
                results[key] = stale   # use stale while we refresh
            to_fetch.append((key, series, units))

    if not to_fetch or not FRED_API_KEY:
        if not FRED_API_KEY:
            sources.append("FRED:skipped(no_api_key)")
        return results, sources

    logger.debug("[FRED] Fetching %d/%d series in parallel", len(to_fetch), len(_FRED_MAP))
    with ThreadPoolExecutor(max_workers=min(_FRED_WORKERS, len(to_fetch))) as pool:
        futures = {
            pool.submit(_fetch_single_fred, k, s, u): (k, s, u)
            for k, s, u in to_fetch
        }
        for fut in as_completed(futures, timeout=_FRED_TIMEOUT + 5):
            try:
                key, val = fut.result(timeout=1)
                if val is not None:
                    _, series_id, units = futures[fut]
                    results[key] = val
                    sources.append(f"FRED:{series_id}{':' + units if units else ''}")
                    _fred_cache.set(f"f:{series_id}:{units or 'lin'}", val, _TTL_FRED)
            except Exception:
                pass

    return results, sources


# ═══════════════════════════════════════════════════════════════════════════════
# yfinance — market-hours-aware smart fetcher
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_yfinance_smart() -> tuple[dict[str, float], list[str]]:
    """
    Fetch yfinance prices ONLY for tickers whose exchange is currently open.
    Tickers on closed exchanges are served from the stale price cache — no network cost.

    First-run (empty cache): fetches ALL tickers regardless of exchange status
    so the cache is populated.
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        return {}, []

    results: dict[str, float] = {}
    sources: list[str]        = []
    now_utc   = datetime.now(timezone.utc)
    price_ttl = _TTL_PRICE_OPEN if any_price_market_open() else _TTL_PRICE_CLOSED
    deadline  = time.time() + max(6.0, _YF_TIMEOUT)

    def _time_left() -> float:
        return deadline - time.time()

    def _phase_timeout(cap: float) -> float | None:
        left = _time_left()
        if left <= 1.0:
            return None
        return max(2.0, min(cap, left))

    # Step 1: Serve everything we have in cache already
    for key in _YF_MAP:
        cached, fresh = _price_cache.get(key)
        if cached is not None:
            results[key] = cached
            if fresh:
                sources.append(f"price_cache:{key}")

    # Step 2: Decide which tickers need a live fetch
    #   • Open exchange  + stale/missing cache  → fetch
    #   • Closed exchange                        → skip (serve stale cache)
    #   • First run (nothing cached)             → fetch everything
    cache_is_cold = not results   # nothing at all in cache yet
    refresh: dict[str, str] = {}  # key → ticker

    for key, ticker in _YF_MAP.items():
        exchange  = _TICKER_EXCHANGE.get(key, "US_EQUITY")
        is_open   = _is_exchange_open(exchange, now_utc)
        _, is_fresh = _price_cache.get(key)

        if is_fresh:
            continue   # cache is valid — already in results
        if cache_is_cold or is_open:
            refresh[key] = ticker
        # else: closed exchange, stale cache is fine — no refresh

    if not refresh:
        logger.debug("[yfinance] All %d tickers served from cache", len(results))
        return results, sources

    logger.debug("[yfinance] Refreshing %d tickers (open exchanges or cold cache)", len(refresh))
    tickers = list(refresh.values())

    def _extract_closes(df: "pd.DataFrame") -> "pd.Series":
        if df is None or df.empty:
            return pd.Series(dtype=float)
        col = df.get("Close") if hasattr(df, "get") else (df["Close"] if "Close" in df.columns else None)
        if col is None:
            return pd.Series(dtype=float)
        clean = col.dropna(how="all")
        if clean.empty:
            return pd.Series(dtype=float)
        row = clean.iloc[-1]
        return row if isinstance(row, pd.Series) else pd.Series({tickers[0]: row})

    fetched: dict[str, float] = {}

    # ── Bucket tickers: equities/crypto support intraday; futures/forex/bonds need daily ──
    equity_keys  = {k: v for k, v in refresh.items()
                    if _TICKER_EXCHANGE.get(k) in {"US_EQUITY", "INDIA_EQUITY",
                                                    "EUROPE_EQUITY", "JAPAN_EQUITY",
                                                    "HONGKONG_EQUITY", "CRYPTO"}}
    non_eq_keys  = {k: v for k, v in refresh.items() if k not in equity_keys}

    # Pass 1a — intraday 5-min bars for equities/crypto (most reliable)
    if equity_keys:
        try:
            t_budget = _phase_timeout(_YF_BATCH_TIMEOUT)
            if t_budget is not None:
                d = yf.download(list(equity_keys.values()), period="1d", interval="5m",
                                progress=False, auto_adjust=True, threads=False,
                                timeout=t_budget)
                closes = _extract_closes(d)
                for key, ticker in equity_keys.items():
                    v = closes.get(ticker)
                    if v is not None and not pd.isna(float(v)):
                        fetched[key] = round(float(v), 4)
                        sources.append(f"yf_live:{ticker}")
            else:
                logger.debug("[yfinance] skip intraday pass (budget exhausted)")
        except Exception as exc:
            logger.debug("[yfinance] equity intraday batch: %s", exc)

    # Pass 1b — daily bars for futures/forex/bonds/commodities (intraday unavailable for these)
    if non_eq_keys:
        try:
            t_budget = _phase_timeout(_YF_BATCH_TIMEOUT)
            if t_budget is not None:
                d2 = yf.download(list(non_eq_keys.values()), period="5d", interval="1d",
                                 progress=False, auto_adjust=True, threads=False,
                                 timeout=t_budget)
                closes2 = _extract_closes(d2)
                for key, ticker in non_eq_keys.items():
                    v = closes2.get(ticker)
                    if v is not None and not pd.isna(float(v)):
                        fetched[key] = round(float(v), 4)
                        sources.append(f"yf_daily:{ticker}")
            else:
                logger.debug("[yfinance] skip non-equity daily pass (budget exhausted)")
        except Exception as exc:
            logger.debug("[yfinance] futures/fx daily batch: %s", exc)

    # Pass 2 — daily fallback for equity tickers still missing after intraday
    miss2 = {k: v for k, v in equity_keys.items() if k not in fetched}
    if miss2:
        try:
            t_budget = _phase_timeout(_YF_BATCH_TIMEOUT)
            if t_budget is not None:
                d3 = yf.download(list(miss2.values()), period="5d", interval="1d",
                                 progress=False, auto_adjust=True, threads=False,
                                 timeout=t_budget)
                closes3 = _extract_closes(d3)
                for key, ticker in miss2.items():
                    v = closes3.get(ticker)
                    if v is not None and not pd.isna(float(v)):
                        fetched[key] = round(float(v), 4)
                        sources.append(f"yf_daily:{ticker}")
            else:
                logger.debug("[yfinance] skip equity daily fallback pass (budget exhausted)")
        except Exception as exc:
            logger.debug("[yfinance] equity daily fallback: %s", exc)

    # Pass 3 — per-ticker fallback only for genuinely missing entries
    miss3 = {k: v for k, v in refresh.items() if k not in fetched and k not in results}
    if miss3 and _YF_ENABLE_PASS3:
        def _single_yf(kv: tuple[str, str]) -> tuple[str, float | None]:
            k, t = kv
            try:
                fi = yf.Ticker(t).fast_info
                v  = getattr(fi, "last_price", None)
                if v is None:
                    if _time_left() <= 1.0:
                        return k, None
                    h = yf.Ticker(t).history(period="5d")
                    v = float(h["Close"].dropna().iloc[-1]) if not h.empty else None
                return k, round(float(v), 4) if v is not None else None
            except Exception:
                return k, None

        with ThreadPoolExecutor(max_workers=min(_YF_WORKERS, len(miss3))) as pool:
            future_map = {pool.submit(_single_yf, kv): kv[0] for kv in miss3.items()}
            try:
                wait_budget = _phase_timeout(_YF_PASS3_TIMEOUT)
                if wait_budget is None:
                    raise FuturesTimeoutError()
                for fut in as_completed(future_map, timeout=wait_budget):
                    try:
                        k, v = fut.result(timeout=3)
                        if v is not None:
                            fetched[k] = v
                            sources.append(f"yf_fallback:{_YF_MAP[k]}")
                    except Exception:
                        pass
            except FuturesTimeoutError:
                unfinished = sum(1 for fut in future_map if not fut.done())
                logger.debug("[yfinance] per-ticker fallback timed out; unfinished=%d", unfinished)
    elif miss3:
        logger.debug("[yfinance] skipped per-ticker fallback for %d symbols (YF_ENABLE_PASS3=0)", len(miss3))

    # Update price cache + merge
    for key, val in fetched.items():
        _price_cache.set(key, val, price_ttl)
        results[key] = val

    return results, sources


# ═══════════════════════════════════════════════════════════════════════════════
# World Bank India — 24 h cache, parallel fetch
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_worldbank_india() -> tuple[dict[str, float], list[str]]:
    CKEY = "wb_india"
    bundle, fresh = _wb_cache.get(CKEY)
    if fresh and bundle:
        logger.debug("[WorldBank] cache hit (24h TTL)")
        return bundle["d"], bundle["s"]

    wb_map = {
        "india_gdp_growth":      "NY.GDP.MKTP.KD.ZG",
        "india_inflation_cpi":   "FP.CPI.TOTL.ZG",
        "india_current_account": "BN.CAB.XOKA.GD.ZS",
        "india_fdi_inflow":      "BX.KLT.DINV.WD.GD.ZS",
        "india_unemployment":    "SL.UEM.TOTL.ZS",
        "india_ext_debt_gdp":    "GD.DOD.DECT.GN.ZS",
    }
    TMPL = "https://api.worldbank.org/v2/country/IND/indicator/{ind}"
    res: dict[str, float] = {}
    src: list[str]        = []

    def _one_wb(kv: tuple[str, str]) -> tuple[str, float | None]:
        key, ind = kv
        try:
            r = requests.get(TMPL.format(ind=ind),
                             params={"format": "json", "mrv": 2, "per_page": 2},
                             timeout=_WB_TIMEOUT)
            pay = r.json()
            if len(pay) >= 2 and pay[1]:
                for entry in pay[1]:
                    v = entry.get("value")
                    if v is not None:
                        return key, round(float(v), 4)
        except Exception:
            pass
        return key, None

    with ThreadPoolExecutor(max_workers=6) as pool:
        for fut in as_completed(
            {pool.submit(_one_wb, kv): kv[0] for kv in wb_map.items()},
            timeout=_WB_TIMEOUT + 4
        ):
            try:
                k, v = fut.result(timeout=2)
                if v is not None:
                    res[k] = v
                    src.append(f"WorldBank:{k}")
            except Exception:
                pass

    _wb_cache.set(CKEY, {"d": res, "s": src}, _TTL_WORLDBANK)
    return res, src


# ═══════════════════════════════════════════════════════════════════════════════
# Alpha Vantage forex — 5 min cache
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_alpha_vantage_forex() -> tuple[dict[str, float], list[str]]:
    if not ALPHA_VANTAGE_API_KEY:
        return {}, []
    CKEY = "av_forex"
    bundle, fresh = _av_cache.get(CKEY)
    if fresh and bundle:
        return bundle["d"], bundle["s"]

    pairs = [("USD", "INR", "usd_inr_av"), ("USD", "CNY", "usd_cny_av"), ("EUR", "USD", "eur_usd_av")]
    res: dict[str, float] = {}
    src: list[str]        = []

    def _one_av(p: tuple[str, str, str]) -> tuple[str, float | None]:
        fc, tc, key = p
        try:
            r = requests.get(ALPHA_VANTAGE_URL, params={
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": fc, "to_currency": tc,
                "apikey": ALPHA_VANTAGE_API_KEY,
            }, timeout=_AV_TIMEOUT)
            rate = r.json().get("Realtime Currency Exchange Rate", {}).get("5. Exchange Rate")
            return key, round(float(rate), 4) if rate else None
        except Exception:
            return key, None

    with ThreadPoolExecutor(max_workers=3) as pool:
        for k, v in [f.result() for f in [pool.submit(_one_av, p) for p in pairs]]:
            if v is not None:
                res[k] = v
                src.append(f"AV:{k}")

    _av_cache.set(CKEY, {"d": res, "s": src}, _TTL_AV)
    return res, src


# ═══════════════════════════════════════════════════════════════════════════════
# Derived indicators — pure computation, zero I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_derived(ind: dict[str, float]) -> dict[str, float]:
    d: dict[str, float] = {}
    g = ind.get
    if g("yield_10y") and g("yield_2y"):
        d["yield_curve"]          = round((ind["yield_10y"] - ind["yield_2y"]) * 100, 1)
    if g("yield_10y") and g("yield_3m"):
        d["yield_curve_10y3m"]    = round((ind["yield_10y"] - ind["yield_3m"]) * 100, 1)
    if g("yield_30y") and g("yield_2y"):
        d["yield_curve_30y2y"]    = round((ind["yield_30y"] - ind["yield_2y"]) * 100, 1)
    if g("yield_30y") and g("yield_3m"):
        d["term_premium_proxy"]   = round((ind["yield_30y"] - ind["yield_3m"]) * 100, 1)
    if "real_rate_10y" not in ind and g("yield_10y") and g("inflation_cpi"):
        d["real_rate_proxy"]      = round(ind["yield_10y"] - ind["inflation_cpi"], 2)
    elif "real_rate_10y" in ind:
        d["real_rate_proxy"]      = ind["real_rate_10y"]
    if g("fed_funds_rate") and g("pce_core"):
        d["fed_real_rate"]        = round(ind["fed_funds_rate"] - ind["pce_core"], 2)
    if g("breakeven_10y") and g("breakeven_5y"):
        d["breakeven_5y10y_slope"]= round(ind["breakeven_10y"] - ind["breakeven_5y"], 3)
    if g("credit_hy") and g("credit_ig"):
        d["credit_spread_gap"]    = round(ind["credit_hy"] - ind["credit_ig"], 2)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_live_indicators() -> tuple[dict[str, float], dict[str, Any]]:
    """
    Returns (indicators, details) — same contract as the original function.

    Fast path: if the merged final cache is still valid, returns in <1 ms.
    Slow path: dispatches yfinance + FRED + WorldBank concurrently.
    """
    global _final_cache, _final_details, _final_cache_expires

    now = time.time()
    price_ttl  = _TTL_PRICE_OPEN if any_price_market_open() else _TTL_PRICE_CLOSED
    merged_ttl = min(price_ttl, _TTL_FRED)

    with _final_lock:
        if _final_cache and now < _final_cache_expires:
            logger.debug("[live] Final cache hit (%.0fs remaining)", _final_cache_expires - now)
            return dict(_final_cache), dict(_final_details)

    t0 = time.time()
    ind: dict[str, float] = {}
    details: dict[str, Any] = {
        "fetched_at_utc":  datetime.now(timezone.utc).isoformat(),
        "sources":         [],
        "source_coverage": {},
        "open_exchanges":  get_open_exchanges(),
        "price_ttl_s":     price_ttl,
        "fred_ttl_s":      _TTL_FRED,
    }

    def _cached_yf_snapshot() -> tuple[dict[str, float], list[str]]:
        cached: dict[str, float] = {}
        cached_sources: list[str] = []
        for key in _YF_MAP:
            val, fresh = _price_cache.get(key)
            if val is None:
                val = _price_cache.get_stale(key)
                if val is None:
                    continue
            cached[key] = float(val)
            cached_sources.append(f"price_cache:{key}{':fresh' if fresh else ':stale'}")
        return cached, cached_sources

    # Dispatch three sources concurrently
    pool = ThreadPoolExecutor(max_workers=3)
    fut_yf = fut_fred = fut_wb = None
    try:
        fut_yf   = pool.submit(_fetch_yfinance_smart)
        fut_fred = pool.submit(_fetch_fred_parallel)
        fut_wb   = pool.submit(_fetch_worldbank_india)

        try:
            yf_d, yf_s = fut_yf.result(timeout=_YF_TIMEOUT + 5)
        except Exception as e:
            yf_d, yf_s = _cached_yf_snapshot()
            if not yf_d:
                logger.debug("[live] yfinance unavailable (%s): %r", e.__class__.__name__, e)
            else:
                logger.debug(
                    "[live] yfinance timeout/unavailable (%s); using cached prices=%d",
                    e.__class__.__name__,
                    len(yf_d),
                )
            if fut_yf is not None and not fut_yf.done():
                fut_yf.cancel()

        try:
            fr_d, fr_s = fut_fred.result(timeout=_FRED_TIMEOUT + 8)
        except Exception as e:
            fr_d, fr_s = {}, []
            logger.warning("[live] FRED error: %s", e)
            if fut_fred is not None and not fut_fred.done():
                fut_fred.cancel()

        try:
            wb_d, wb_s = fut_wb.result(timeout=_WB_TIMEOUT + 5)
        except Exception as e:
            wb_d, wb_s = {}, []
            logger.warning("[live] WorldBank error: %s", e)
            if fut_wb is not None and not fut_wb.done():
                fut_wb.cancel()
    finally:
        # Avoid blocking on slow yfinance workers after timeout; continue with partial data.
        pool.shutdown(wait=False, cancel_futures=True)

    # FRED overrides yfinance where both have the same key (more authoritative)
    ind.update(yf_d)
    ind.update(fr_d)
    ind.update(wb_d)
    ind = sanitize_indicator_values(ind)

    # Optional Alpha Vantage (rate-limited; keep outside the pool)
    if ALPHA_VANTAGE_API_KEY:
        try:
            av_d, av_s = _fetch_alpha_vantage_forex()
            ind.update(av_d)
            details["sources"].extend(av_s)
            details["source_coverage"]["AlphaVantage"] = len(av_d)
        except Exception:
            pass

    # Derived indicators — zero I/O
    ind.update(_compute_derived(ind))
    ind = sanitize_indicator_values(ind)

    details["sources"].extend(yf_s + fr_s + wb_s)
    details["source_coverage"].update(
        {"yfinance": len(yf_d), "FRED": len(fr_d), "WorldBank": len(wb_d)}
    )
    details["total_indicators"] = len(ind)
    details["fetch_ms"]         = round((time.time() - t0) * 1000)

    logger.info(
        "[live] %d indicators in %dms | yf=%d fred=%d wb=%d | open=%s | price_ttl=%ds",
        len(ind), details["fetch_ms"],
        len(yf_d), len(fr_d), len(wb_d),
        get_open_exchanges(), price_ttl,
    )

    with _final_lock:
        _final_cache         = dict(ind)
        _final_details       = dict(details)
        _final_cache_expires = now + merged_ttl

    return ind, details


def stream_live_indicators():
    """
    Progressive generator — yields (indicators_so_far: dict, meta: dict) immediately
    after EACH source (yfinance / FRED / WorldBank) completes.

    • Fastest source lands on-screen first; slower sources augment incrementally.
    • Derived indicators (_compute_derived) are recalculated on every yield so
      yield_curve / real_rate etc. appear as soon as their inputs arrive.
    • Cache fast-path: if the merged final cache is still valid, yields once and
      returns — no network I/O at all.
    • On a cold fetch a typical timeline is:
        t ≈ 0 ms   — generator entered
        t ≈ 3-6 s  — yfinance batch done  → first yield  (prices, equities, FX …)
        t ≈ 4-7 s  — FRED done            → second yield (macro, yields, credit …)
        t ≈ 5-9 s  — WorldBank done       → third yield  (India macro …)
    """
    global _final_cache, _final_details, _final_cache_expires

    now = time.time()
    price_ttl  = _TTL_PRICE_OPEN if any_price_market_open() else _TTL_PRICE_CLOSED
    merged_ttl = min(price_ttl, _TTL_FRED)

    # ── Fast path: complete cache is valid ─────────────────────────────────
    with _final_lock:
        if _final_cache and now < _final_cache_expires:
            cached_meta = dict(_final_details)
            cached_meta["partial"] = False
            cached_meta["completed_sources"] = ["yfinance", "FRED", "WorldBank"]
            cached_meta["pending_sources"]   = []
            cached_meta["from_cache"]        = True
            yield dict(_final_cache), cached_meta
            return

    # ── Slow path: dispatch all three sources concurrently ─────────────────
    t0          = time.time()
    ind_so_far: dict[str, float] = {}
    all_sources: list[str]       = []
    all_sources_map: dict[str, int] = {}
    completed:  list[str]        = []
    _ALL_SOURCES = ["yfinance", "FRED", "WorldBank"]

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures: dict = {
            pool.submit(_fetch_yfinance_smart): "yfinance",
            pool.submit(_fetch_fred_parallel):  "FRED",
            pool.submit(_fetch_worldbank_india):"WorldBank",
        }

        for fut in as_completed(futures):
            source_name = futures[fut]
            completed.append(source_name)

            try:
                data, srcs = fut.result(timeout=3)
                ind_so_far.update(data)
                all_sources.extend(srcs)
                all_sources_map[source_name] = len(data)
            except Exception as exc:
                logger.warning("[stream] %s: %s", source_name, exc)
                all_sources_map[source_name] = 0

            merged = sanitize_indicator_values(dict(ind_so_far))
            merged.update(_compute_derived(merged))
            merged = sanitize_indicator_values(merged)

            meta = {
                "fetched_at_utc":    datetime.now(timezone.utc).isoformat(),
                "source":            source_name,
                "completed_sources": list(completed),
                "pending_sources":   [s for s in _ALL_SOURCES if s not in completed],
                "fetch_ms":          round((time.time() - t0) * 1000),
                "total_indicators":  len(merged),
                "partial":           len(completed) < len(_ALL_SOURCES),
                "open_exchanges":    get_open_exchanges(),
                "price_ttl_s":       price_ttl,
                "fred_ttl_s":        _TTL_FRED,
                "from_cache":        False,
            }
            yield merged, meta

    # ── Update the merged final cache ──────────────────────────────────────
    final = sanitize_indicator_values(dict(ind_so_far))
    final.update(_compute_derived(final))
    final = sanitize_indicator_values(final)

    if ALPHA_VANTAGE_API_KEY:
        try:
            av_d, _ = _fetch_alpha_vantage_forex()
            final.update(av_d)
        except Exception:
            pass

    final_details = {
        "fetched_at_utc":  datetime.now(timezone.utc).isoformat(),
        "sources":         all_sources,
        "source_coverage": all_sources_map,
        "total_indicators": len(final),
        "fetch_ms":        round((time.time() - t0) * 1000),
        "open_exchanges":  get_open_exchanges(),
        "price_ttl_s":     price_ttl,
        "fred_ttl_s":      _TTL_FRED,
        "partial":         False,
        "from_cache":      False,
    }

    with _final_lock:
        _final_cache         = dict(final)
        _final_details       = dict(final_details)
        _final_cache_expires = time.time() + merged_ttl


def invalidate_live_data_cache() -> None:
    """Force next call to bypass all caches (WorldBank preserved — 24h data)."""
    global _final_cache_expires
    with _final_lock:
        _final_cache_expires = 0.0
    _price_cache.invalidate()
    _fred_cache.invalidate()
    logger.info("[live] Price + FRED caches invalidated.  WorldBank (24h) preserved.")


def check_fred_key() -> dict[str, Any]:
    if not FRED_API_KEY:
        return {"valid": False, "error": "No FRED_API_KEY set", "latency_ms": None}
    t0 = time.time()
    _, val = _fetch_single_fred("fed_funds_rate", "FEDFUNDS")
    ms = round((time.time() - t0) * 1000, 1)
    if val is not None:
        return {"valid": True, "latency_ms": ms, "sample_series": "FEDFUNDS", "sample_value": val}
    return {"valid": False, "latency_ms": ms, "error": "no value returned"}


# ── benchmark  (kept for backward-compat) ─────────────────────────────────────
def benchmark_data_sources(verbose: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {"run_at_utc": datetime.now(timezone.utc).isoformat(), "sources": {}}
    for name, fn in [("yfinance", _fetch_yfinance_smart),
                     ("FRED",     _fetch_fred_parallel),
                     ("WorldBank",_fetch_worldbank_india)]:
        t0 = time.time()
        data, _ = fn()
        ms = round((time.time() - t0) * 1000)
        report["sources"][name] = {"latency_ms": ms, "coverage": len(data)}
        if verbose:
            print(f"  {name:15s}: {ms:>6} ms  |  {len(data)} indicators")
    return report


if __name__ == "__main__":
    import json, logging
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(market_status_summary(), indent=2))
    print("\nBenchmarking (first call — cold cache)...")
    benchmark_data_sources()
    print("\nFetching live indicators...")
    t0 = time.time()
    ind, det = fetch_live_indicators()
    print(f"Done in {(time.time()-t0)*1000:.0f} ms — {len(ind)} indicators")
    for k in ["sp500", "vix", "dxy", "yield_10y", "inflation_cpi", "fed_funds_rate", "oil_wti", "gold"]:
        if k in ind:
            print(f"  {k}: {ind[k]}")
    print(f"\nopen_exchanges={det.get('open_exchanges')}")
    print(f"price_ttl={det.get('price_ttl_s')}s  fred_ttl={det.get('fred_ttl_s')}s")
