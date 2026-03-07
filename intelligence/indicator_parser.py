"""
indicator_parser.py
-------------------
Extracts key macro indicator values from:
  1. Retrieved context chunks (your indexed data)
  2. The user's question (if they paste in data)
  3. A manual override dict (for live API feeds)

This is the "data normalization" layer. Every indicator gets
a standard key, a value, a unit, and a staleness flag.

Standard indicator keys used across the system:
  gdp_growth       - YoY real GDP growth %
  inflation_cpi    - CPI YoY %
  inflation_core   - Core CPI YoY %
  pce              - PCE YoY %
  fed_funds_rate   - Current Fed Funds Rate %
  yield_10y        - 10Y Treasury yield %
  yield_2y         - 2Y Treasury yield %
  yield_curve      - 10Y - 2Y spread (computed)
  credit_hy        - HY spread bps
  credit_ig        - IG spread bps
  vix              - VIX level
  dxy              - Dollar Index level
  oil_wti          - WTI Crude price USD
  unemployment     - Unemployment rate %
  pmi_mfg          - Manufacturing PMI
  pmi_services     - Services PMI
  retail_sales_mom - Retail sales MoM %
  housing_starts   - Housing starts (000s annualized)
"""

import re
from typing import Optional


# Regex patterns: each maps a keyword pattern to a standard indicator key and capture group for value
EXTRACTION_PATTERNS = [
    ("gdp_growth",      r"(?:gdp|real gdp)[^\d]*?([\-]?\d+\.?\d*)\s*%"),
    ("inflation_cpi",   r"(?:cpi|consumer price)[^\d]*?([\-]?\d+\.?\d*)\s*%"),
    ("inflation_core",  r"(?:core cpi|core inflation)[^\d]*?([\-]?\d+\.?\d*)\s*%"),
    ("pce",             r"(?:pce)[^\d]*?([\-]?\d+\.?\d*)\s*%"),
    ("fed_funds_rate",  r"(?:fed funds|federal funds|policy rate)[^\d]*?([\d\.]+)\s*%"),
    ("yield_30y",       r"(?:30.?year|30y)[^\d]*?([\d\.]+)\s*%"),
    ("yield_10y",       r"(?:10.?year|10y)[^\d]*?([\d\.]+)\s*%"),
    ("yield_2y",        r"(?:2.?year|2y)[^\d]*?([\d\.]+)\s*%"),
    ("credit_hy",       r"(?:hy spread|high yield spread)[^\d]*?([\d]+)\s*bps?"),
    ("credit_ig",       r"(?:ig spread|investment grade spread)[^\d]*?([\d]+)\s*bps?"),
    ("vix",             r"(?:vix)[^\d]*?([\d\.]+)"),
    ("dxy",             r"(?:dxy|dollar index)[^\d]*?([\d\.]+)"),
    ("oil_brent",       r"(?:brent)[^\d]*?\$?([\d\.]+)"),
    ("oil_wti",         r"(?:wti|crude oil|oil price)[^\d]*?\$?([\d\.]+)"),
    ("gold",            r"(?:gold)[^\d]*?\$?([\d]+(?:\.\d+)?)"),
    ("sp500",           r"(?:s&p\s*500|s&p500|spx)[^\d]*?([\d,]+(?:\.\d+)?)"),
    ("nasdaq",          r"(?:nasdaq|ndx|qqq)[^\d]*?([\d,]+(?:\.\d+)?)"),
    ("unemployment",    r"(?:unemployment|jobless rate)[^\d]*?([\d\.]+)\s*%"),
    ("pmi_mfg",         r"(?:manufacturing pmi|mfg pmi|ism mfg)[^\d]*?([\d\.]+)"),
    ("pmi_services",    r"(?:services pmi|service pmi|ism services)[^\d]*?([\d\.]+)"),
]


def extract_indicators_from_text(text: str) -> dict:
    """
    Parse indicator values from free-form text (context chunks or user message).
    Returns dict of {indicator_key: float}.
    """
    text_lower = text.lower()
    indicators = {}

    for key, pattern in EXTRACTION_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            try:
                # Strip commas (e.g. "5,432" → "5432") before parsing
                raw = match.group(1).replace(",", "")
                indicators[key] = float(raw)
            except ValueError:
                pass

    # Derived indicators
    if "yield_10y" in indicators and "yield_2y" in indicators:
        indicators["yield_curve"] = round(
            (indicators["yield_10y"] - indicators["yield_2y"]) * 100, 1
        )  # in bps

    # Best available credit spread
    if "credit_hy" in indicators and "credit_spread" not in indicators:
        indicators["credit_spread"] = indicators["credit_hy"]
    elif "credit_ig" in indicators and "credit_spread" not in indicators:
        indicators["credit_spread"] = indicators["credit_ig"]

    return indicators


def merge_indicators(*sources: dict) -> dict:
    """
    Merge multiple indicator dicts. Later sources override earlier ones.
    Use order: [defaults, extracted_from_context, manual_overrides]
    """
    merged = {}
    for source in sources:
        if source:
            merged.update(source)
    return merged


def format_indicators_for_prompt(indicators: dict) -> str:
    """Format indicator dict for injection into LLM prompts."""
    if not indicators:
        return "[No indicator data available]"

    LABELS = {
        "gdp_growth":      ("GDP Growth (YoY)", "%"),
        "inflation_cpi":   ("CPI Inflation (YoY)", "%"),
        "inflation_core":  ("Core CPI (YoY)", "%"),
        "pce":             ("PCE (YoY)", "%"),
        "fed_funds_rate":  ("Fed Funds Rate", "%"),
        "yield_30y":       ("30Y Treasury", "%"),
        "yield_10y":       ("10Y Treasury", "%"),
        "yield_2y":        ("2Y Treasury", "%"),
        "yield_curve":     ("Yield Curve (10Y-2Y)", "bps"),
        "real_rate_proxy": ("Real Rate Proxy", "%"),
        "credit_hy":       ("HY Spread", "bps"),
        "credit_ig":       ("IG Spread", "bps"),
        "vix":             ("VIX", ""),
        "dxy":             ("DXY", ""),
        "sp500":           ("S&P 500", ""),
        "nasdaq":          ("Nasdaq", ""),
        "gold":            ("Gold", "USD/oz"),
        "oil_wti":         ("WTI Oil", "USD"),
        "oil_brent":       ("Brent Oil", "USD"),
        "unemployment":    ("Unemployment", "%"),
        "pmi_mfg":         ("Mfg PMI", ""),
        "pmi_services":    ("Services PMI", ""),
        "consumer_sentiment": ("Consumer Sentiment", ""),
    }

    lines = []
    for key, value in indicators.items():
        label, unit = LABELS.get(key, (key, ""))
        lines.append(f"  {label}: {value}{unit}")

    return "\n".join(lines)


def get_regime_inputs_from_indicators(indicators: dict) -> dict:
    """Extract the 4 regime detection inputs from normalized indicators."""
    return {
        "gdp_growth":    indicators.get("gdp_growth"),
        "inflation":     indicators.get("inflation_cpi") or indicators.get("pce"),
        "credit_spread": indicators.get("credit_spread") or indicators.get("credit_hy") or indicators.get("credit_ig"),
        "yield_curve":   indicators.get("yield_curve"),
    }


if __name__ == "__main__":
    test_text = """
    The Fed funds rate is at 5.25%. CPI came in at 3.7% YoY. Core CPI at 4.1%.
    GDP growth was 2.1% last quarter. The 10-year yield is 4.8% and 2-year at 5.1%.
    HY spreads are at 420 bps. VIX is at 22. Manufacturing PMI at 47.6.
    """
    result = extract_indicators_from_text(test_text)
    print(format_indicators_for_prompt(result))
    print("\nRegime inputs:", get_regime_inputs_from_indicators(result))