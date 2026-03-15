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
    ("gdp_growth",      r"(?:\bgdp\b|\breal gdp\b)[^\d\n]{0,24}([\-]?\d+\.?\d*)\s*%"),
    ("inflation_cpi",   r"(?:\bcpi\b|\bconsumer price(?: index)?\b)[^\d\n]{0,24}([\-]?\d+\.?\d*)\s*%"),
    ("inflation_core",  r"(?:\bcore cpi\b|\bcore inflation\b)[^\d\n]{0,24}([\-]?\d+\.?\d*)\s*%"),
    ("pce",             r"\bpce\b[^\d\n]{0,24}([\-]?\d+\.?\d*)\s*%"),
    ("fed_funds_rate",  r"(?:\bfed funds\b|\bfederal funds\b|\bpolicy rate\b)[^\d\n]{0,24}([\d\.]+)\s*%"),
    ("yield_30y",       r"(?:\b30.?year\b|\b30y\b)[^\d\n]{0,24}([\d\.]+)\s*%"),
    ("yield_10y",       r"(?:\b10.?year\b|\b10y\b)[^\d\n]{0,24}([\d\.]+)\s*%"),
    ("yield_2y",        r"(?:\b2.?year\b|\b2y\b)[^\d\n]{0,24}([\d\.]+)\s*%"),
    ("credit_hy",       r"(?:\bhy spread\b|\bhigh yield spread\b)[^\d\n]{0,24}([\d]+)\s*bps?"),
    ("credit_ig",       r"(?:\big spread\b|\binvestment grade spread\b)[^\d\n]{0,24}([\d]+)\s*bps?"),
    ("vix",             r"\bvix\b[^\d\n]{0,24}([\d\.]+)"),
    ("dxy",             r"(?:\bdxy\b|\bdollar index\b)[^\d\n]{0,24}([\d\.]+)"),
    ("oil_brent",       r"\bbrent\b[^\d\n]{0,20}\$?([\d\.]+)"),
    ("oil_wti",         r"(?:\bwti\b|\bcrude oil\b|\boil price\b)[^\d\n]{0,20}\$?([\d\.]+)"),
    ("gold",            r"(?:\bgold\b|\bxau\b)[^\d\n]{0,20}\$?([\d]+(?:\.\d+)?)"),
    ("sp500",           r"(?:\bs&p\s*500\b|\bs&p500\b|\bspx\b)[^\d\n]{0,24}([\d,]+(?:\.\d+)?)"),
    ("nasdaq",          r"(?:\bnasdaq\b|\bndx\b|\bqqq\b)[^\d\n]{0,24}([\d,]+(?:\.\d+)?)"),
    ("unemployment",    r"(?:\bunemployment\b|\bjobless rate\b)[^\d\n]{0,24}([\d\.]+)\s*%"),
    ("pmi_mfg",         r"(?:\bmanufacturing pmi\b|\bmfg pmi\b|\bism mfg\b)[^\d\n]{0,24}([\d\.]+)"),
    ("pmi_services",    r"(?:\bservices pmi\b|\bservice pmi\b|\bism services\b)[^\d\n]{0,24}([\d\.]+)"),
]


_BPS_KEYS = {
    "credit_hy",
    "credit_ig",
    "credit_bb",
    "credit_spread",
    "credit_spread_gap",
}


_RANGE_BOUNDS = {
    "gdp_growth": (-15.0, 20.0),
    "inflation_cpi": (-5.0, 20.0),
    "inflation_core": (-5.0, 20.0),
    "inflation_core_cpi": (-5.0, 20.0),
    "pce": (-5.0, 20.0),
    "pce_deflator": (-5.0, 20.0),
    "pce_core": (-5.0, 20.0),
    "fed_funds_rate": (-1.0, 20.0),
    "yield_30y": (-1.0, 20.0),
    "yield_10y": (-1.0, 20.0),
    "yield_2y": (-1.0, 20.0),
    "yield_1y": (-1.0, 20.0),
    "yield_3m": (-1.0, 20.0),
    "yield_curve": (-500.0, 500.0),
    "yield_curve_10y3m": (-500.0, 500.0),
    "yield_curve_30y2y": (-500.0, 500.0),
    "term_premium_proxy": (-500.0, 500.0),
    "credit_hy": (50.0, 2500.0),
    "credit_ig": (5.0, 1000.0),
    "credit_bb": (10.0, 2000.0),
    "credit_spread": (5.0, 2500.0),
    "credit_spread_gap": (-300.0, 2000.0),
    "vix": (5.0, 120.0),
    "dxy": (70.0, 140.0),
    "oil_brent": (5.0, 250.0),
    "oil_wti": (5.0, 250.0),
    "gold": (200.0, 5000.0),
    "sp500": (500.0, 15000.0),
    "nasdaq": (1000.0, 30000.0),
    "unemployment": (1.0, 30.0),
    "pmi_mfg": (20.0, 80.0),
    "pmi_services": (20.0, 80.0),
}


def _to_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def sanitize_indicator_values(indicators: dict, *, drop_out_of_range: bool = True) -> dict:
    cleaned: dict = {}
    for key, value in (indicators or {}).items():
        numeric = _to_float(value)
        if numeric is None:
            if value is not None:
                cleaned[key] = value
            continue

        adjusted = numeric
        if key in _BPS_KEYS and 0 < abs(adjusted) < 25:
            adjusted *= 100.0

        bounds = _RANGE_BOUNDS.get(key)
        if bounds and drop_out_of_range:
            low, high = bounds
            if adjusted < low or adjusted > high:
                continue

        cleaned[key] = round(adjusted, 4)

    if "yield_10y" in cleaned and "yield_2y" in cleaned and "yield_curve" not in cleaned:
        cleaned["yield_curve"] = round((cleaned["yield_10y"] - cleaned["yield_2y"]) * 100, 1)

    if "credit_hy" in cleaned and "credit_spread" not in cleaned:
        cleaned["credit_spread"] = cleaned["credit_hy"]
    elif "credit_ig" in cleaned and "credit_spread" not in cleaned:
        cleaned["credit_spread"] = cleaned["credit_ig"]

    return cleaned


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

    return sanitize_indicator_values(indicators)


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