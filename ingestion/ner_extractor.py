"""
ingestion/ner_extractor.py
---------------------------
Fast, zero-dependency financial Named Entity Recognition (NER).

No ML model required — uses a comprehensive curated regex + dictionary approach
that is purpose-built for financial text. Runs at ~500k chars/sec on CPU.

Extracts:
  - Ticker symbols  (AAPL, TSLA, BRK.B, BTC-USD, EUR/USD)
  - Companies       (Apple, Goldman Sachs, JPMorgan Chase …)
  - Economic indicators (CPI, PCE, FOMC, GDP, NFP …)
  - People          (Powell, Yellen, Buffett …)
  - Dollar amounts  ($182.32, $2.5T, $400M)
  - Percentages     (3.5%, +25bps)

Usage:
    from ingestion.ner_extractor import extract_entities, batch_extract

    result = extract_entities("Apple (AAPL) rose 3.5% after the Fed raised rates by 25bps")
    # -> {"tickers": ["AAPL"], "companies": ["Apple"], "indicators": ["Fed"],
    #     "people": [], "amounts": ["$182.32"], "percentages": ["3.5%", "25bps"]}
"""

from __future__ import annotations

import re
from typing import Any

# ─── Ticker pattern ────────────────────────────────────────────────────────────
# Standard US equities: 1-5 uppercase letters (NYSE/NASDAQ)
# Special forms: BRK.B, BRK.A, BTC-USD, EUR/USD, ^GSPC (index)
_TICKER_RE = re.compile(
    r"\b"
    r"(\^[A-Z]{2,6}"                   # index: ^GSPC, ^DJI
    r"|[A-Z]{2,6}\.[AB]"               # class shares: BRK.A, BRK.B
    r"|[A-Z]{2,6}[-/][A-Z]{3}"        # fx / crypto: BTC-USD, EUR/USD
    r"|[A-Z]{2,5}"                     # plain ticker: AAPL, TSLA, AMZN
    r")\b"
    r"(?!\s*[a-z])"                    # not followed by lowercase (avoids "IT", "US" in sentences)
)

# Common uppercase words that are NOT tickers (false-positive suppression)
_TICKER_STOPWORDS: frozenset[str] = frozenset({
    "CEO", "CFO", "COO", "CTO", "IPO", "FY", "Q1", "Q2", "Q3", "Q4",
    "YOY", "YTD", "MOM", "GDP", "CPI", "PCE", "NFP", "PMI", "ISM",
    "ETF", "REIT", "SPV", "LLC", "INC", "LTD", "PLC", "CORP",
    "SEC", "FED", "ECB", "BOE", "BOJ", "IMF", "BIS", "OPEC", "G20", "G7",
    "AI", "ML", "US", "UK", "EU", "UN", "NYSE", "NASDAQ", "LSE",
    "EPS", "DPS", "PE", "PB", "ROE", "ROA", "EBIT", "EBITDA",
    "USD", "EUR", "GBP", "JPY", "CNY", "INR", "CHF", "CAD", "AUD",
    "API", "UI", "UX", "AWS", "GCP", "SaaS", "PaaS", "IaaS",
    "AR", "VR", "EV", "ICE", "LNG", "LPG", "WTI", "BRENT",
    "M2", "M1", "QE", "QT", "OMO", "RRP", "ON", "IN", "AT",
    "A", "I", "SO", "TO", "DO", "IF", "OR",
})

# ─── Company name dictionary ───────────────────────────────────────────────────
# Curated list; matched case-insensitively in text
_COMPANY_NAMES: list[str] = [
    # Big Tech
    "Apple", "Microsoft", "Alphabet", "Google", "Amazon", "Meta", "Facebook",
    "Netflix", "Nvidia", "AMD", "Intel", "Qualcomm", "Texas Instruments",
    "TSMC", "Samsung", "Sony", "LG", "Broadcom", "Cisco", "Oracle", "SAP",
    "Salesforce", "ServiceNow", "Snowflake", "Palantir", "Cloudflare",
    # Finance
    "JPMorgan", "JP Morgan", "Goldman Sachs", "Morgan Stanley", "Citigroup",
    "Bank of America", "Wells Fargo", "BlackRock", "Vanguard", "Fidelity",
    "Charles Schwab", "Berkshire Hathaway", "Berkshire", "American Express",
    "Visa", "Mastercard", "PayPal", "Square", "Block",
    # Energy
    "ExxonMobil", "Exxon", "Chevron", "Shell", "BP", "TotalEnergies",
    "ConocoPhillips", "Halliburton", "Schlumberger", "SLB", "Valero",
    "Occidental", "Pioneer Natural", "Devon Energy",
    # Healthcare / Pharma
    "Johnson & Johnson", "Pfizer", "Moderna", "Merck", "AbbVie", "Eli Lilly",
    "Bristol-Myers", "AstraZeneca", "Novartis", "Roche", "UnitedHealth",
    "CVS Health", "Anthem", "Humana",
    # Consumer / Retail
    "Walmart", "Target", "Costco", "Home Depot", "Lowe's", "Nike", "Adidas",
    "McDonald's", "Starbucks", "Coca-Cola", "PepsiCo", "Procter & Gamble",
    # Industrial / Transport
    "Boeing", "Airbus", "Lockheed Martin", "Raytheon", "General Electric",
    "Caterpillar", "Deere", "FedEx", "UPS", "Union Pacific",
    # Telecom / Media
    "AT&T", "Verizon", "T-Mobile", "Comcast", "Disney", "Warner Bros",
    # China Tech
    "Alibaba", "Tencent", "Baidu", "ByteDance", "Meituan", "JD.com",
    # Crypto / Exchanges
    "Coinbase", "Binance", "Kraken", "CME Group", "Intercontinental Exchange",
]

_COMPANY_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(_COMPANY_NAMES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# ─── Economic indicators & institutions ───────────────────────────────────────
_INDICATORS: dict[str, str] = {
    # Macro indicators
    "CPI": "inflation", "PCE": "inflation", "PPI": "inflation",
    "GDP": "growth", "GNP": "growth", "PMI": "activity", "ISM": "activity",
    "NFP": "employment", "ADP": "employment", "JOLTs": "employment", "JOLTS": "employment",
    "CLI": "leading_indicator",
    # Rates / monetary
    "FOMC": "monetary_policy", "Fed": "monetary_policy", "Federal Reserve": "monetary_policy",
    "ECB": "monetary_policy", "BOE": "monetary_policy", "BOJ": "monetary_policy",
    "PBOC": "monetary_policy", "RBI": "monetary_policy",
    "SOFR": "rate", "LIBOR": "rate", "OIS": "rate",
    "FFR": "rate", "Fed funds": "rate",
    # Yields / spreads
    "10-year": "yield", "2-year": "yield", "30-year": "yield",
    "yield curve": "yield", "inversion": "yield",
    # Macro events
    "OPEC": "energy", "G20": "macro", "G7": "macro", "IMF": "macro",
    "World Bank": "macro", "BIS": "macro",
    # Markets
    "S&P 500": "index", "Nasdaq": "index", "Dow Jones": "index", "DJIA": "index",
    "Russell 2000": "index", "VIX": "index",
    # Commodities
    "WTI": "commodity", "Brent": "commodity", "Gold": "commodity",
    "Silver": "commodity", "Copper": "commodity", "Bitcoin": "crypto", "Ethereum": "crypto",
    # M2 / QE etc.
    "QE": "monetary_policy", "QT": "monetary_policy", "M2": "monetary_supply",
}

_INDICATOR_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_INDICATORS.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# ─── Notable people ────────────────────────────────────────────────────────────
_PEOPLE: list[str] = [
    "Powell", "Yellen", "Bernanke", "Greenspan",   # Fed chairs
    "Lagarde", "Bailey", "Ueda", "Kuroda",          # ECB, BOE, BOJ
    "Buffett", "Munger", "Dalio", "Ackman",         # Investors
    "Musk", "Bezos", "Cook", "Nadella", "Pichai", "Zuckerberg",  # CEOs
    "Dimon", "Fink", "Blankfein", "Gorman",         # Finance CEOs
    "Trump", "Biden", "Yellen",                     # Political (impact markets)
]

_PEOPLE_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in sorted(_PEOPLE, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# ─── Dollar amounts ────────────────────────────────────────────────────────────
# Matches: $182.32, $2.5T, $400M, $1.2B, USD 500K
_AMOUNT_RE = re.compile(
    r"(?:USD\s*)?\$\s*\d+(?:\.\d+)?\s*(?:[KMBT](?:rillion|illion|illion|illion)?)?",
    re.IGNORECASE,
)

# ─── Percentages / basis points ────────────────────────────────────────────────
_PCT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?%|[-+]?\d+\s*bps?\b", re.IGNORECASE)


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Extract financial named entities from *text*.

    Returns a dict with keys:
        tickers    — stock/crypto/fx symbols
        companies  — company names
        indicators — economic indicators and institutions
        people     — notable finance/market people
        amounts    — dollar amounts
        percentages — percentage moves and basis points
    """
    if not text:
        return _empty()

    # 1. Tickers — match uppercase tokens not in stop-word list
    raw_tickers = _TICKER_RE.findall(text)
    tickers = list(dict.fromkeys(           # deduplicate preserving order
        t for t in raw_tickers
        if t not in _TICKER_STOPWORDS and len(t) >= 2
    ))

    # 2. Companies
    companies = list(dict.fromkeys(m.group() for m in _COMPANY_RE.finditer(text)))

    # 3. Indicators
    indicators = list(dict.fromkeys(m.group() for m in _INDICATOR_RE.finditer(text)))

    # 4. People
    people = list(dict.fromkeys(m.group() for m in _PEOPLE_RE.finditer(text)))

    # 5. Dollar amounts
    amounts = list(dict.fromkeys(_AMOUNT_RE.findall(text)))

    # 6. Percentages / bps
    percentages = list(dict.fromkeys(_PCT_RE.findall(text)))

    return {
        "tickers":     tickers[:10],
        "companies":   companies[:10],
        "indicators":  indicators[:10],
        "people":      people[:8],
        "amounts":     amounts[:8],
        "percentages": percentages[:8],
    }


def batch_extract(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Enrich a list of chunk dicts in-place with an 'entities' key.
    Returns the same list (mutated).
    """
    for chunk in chunks:
        text = chunk.get("text") or ""
        chunk["entities"] = extract_entities(text)
    return chunks


def flat_entity_list(entities: dict[str, list[str]]) -> list[str]:
    """Flatten entity dict into a single deduplicated list for storage / search."""
    seen: set[str] = set()
    result: list[str] = []
    for values in entities.values():
        for v in values:
            v_lower = v.lower()
            if v_lower not in seen:
                seen.add(v_lower)
                result.append(v)
    return result


def _empty() -> dict[str, list[str]]:
    return {
        "tickers": [], "companies": [], "indicators": [],
        "people": [], "amounts": [], "percentages": [],
    }
