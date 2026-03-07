"""
regime_detector.py
------------------
Classifies the current macro regime from a set of key indicators.
Regime detection is the most important upstream input — it conditions
every downstream analysis (sector calls, positioning, scenario weights).

Regimes (based on growth + inflation quadrant framework):
  Goldilocks   = Growth ↑, Inflation ↓  → Risk-On
  Reflation    = Growth ↑, Inflation ↑  → Cyclicals, Commodities
  Stagflation  = Growth ↓, Inflation ↑  → Defensive, Short Duration
  Deflation    = Growth ↓, Inflation ↓  → Long Duration, Quality

Sub-regimes layered on top:
  Credit Stress  = Spreads widening fast
  Liquidity Trap = Rate cuts not transmitting
  Late Cycle     = Curve inverted + credit deteriorating
  Early Recovery = PMI bottoming + credit loosening
"""

from typing import Optional


REGIME_RULES = [
    # (name, condition_fn, description)
    (
        "STAGFLATION",
        lambda g, i, c, y: g < 0 and i > 3.5 and c > 300,
        "Growth falling, inflation sticky, credit stress — worst quadrant",
    ),
    (
        "LATE_CYCLE",
        lambda g, i, c, y: g > 0 and g < 1.5 and y < 0 and c > 200,
        "Slowing growth, inverted curve, credit spreads widening — late expansion",
    ),
    (
        "RECESSION",
        lambda g, i, c, y: g < -1.0 and y < 0 and c > 400,
        "Contraction confirmed, curve inverted, spreads blown out",
    ),
    (
        "REFLATION",
        lambda g, i, c, y: g > 2.0 and i > 2.5 and c < 200,
        "Strong growth + rising inflation + tight spreads — commodity-friendly",
    ),
    (
        "GOLDILOCKS",
        lambda g, i, c, y: g > 2.0 and i < 2.5 and c < 150,
        "Strong growth, low inflation, calm credit — risk-on ideal",
    ),
    (
        "DEFLATION_RISK",
        lambda g, i, c, y: g < 1.0 and i < 1.5,
        "Weak growth + falling inflation — long duration, quality assets",
    ),
    (
        "EARLY_RECOVERY",
        lambda g, i, c, y: g > 0 and g < 2.0 and i < 2.5 and c < 200 and y > -0.5,
        "Growth recovering from trough, inflation benign — buy risk",
    ),
]


def detect_regime(
    gdp_growth: Optional[float] = None,       # YoY GDP growth %
    inflation: Optional[float] = None,         # CPI YoY %
    credit_spread: Optional[float] = None,     # IG or HY spread in bps
    yield_curve: Optional[float] = None,       # 10Y - 2Y spread in bps (negative = inverted)
) -> dict:
    """
    Classify macro regime from key indicators.

    Returns:
        {
            "regime": str,
            "description": str,
            "confidence": str,   # HIGH / MEDIUM / LOW
            "missing_inputs": list[str],
            "quadrant": str,     # Growth/Inflation label
        }
    """
    missing = []
    if gdp_growth is None:
        missing.append("gdp_growth")
        gdp_growth = 1.5  # neutral default
    if inflation is None:
        missing.append("inflation")
        inflation = 2.5
    if credit_spread is None:
        missing.append("credit_spread")
        credit_spread = 200
    if yield_curve is None:
        missing.append("yield_curve")
        yield_curve = 0

    confidence = "HIGH" if len(missing) == 0 else ("MEDIUM" if len(missing) <= 2 else "LOW")

    # Quadrant label
    g_label = "Growth↑" if gdp_growth >= 2.0 else ("Growth~" if gdp_growth >= 0 else "Growth↓")
    i_label = "Inflation↑" if inflation >= 2.5 else "Inflation↓"
    quadrant = f"{g_label} / {i_label}"

    # Match regime
    for name, condition, description in REGIME_RULES:
        try:
            if condition(gdp_growth, inflation, credit_spread, yield_curve):
                return {
                    "regime": name,
                    "description": description,
                    "confidence": confidence,
                    "missing_inputs": missing,
                    "quadrant": quadrant,
                    "inputs": {
                        "gdp_growth": gdp_growth,
                        "inflation": inflation,
                        "credit_spread_bps": credit_spread,
                        "yield_curve_bps": yield_curve,
                    }
                }
        except Exception:
            continue

    return {
        "regime": "TRANSITIONAL",
        "description": "No single regime matches cleanly — transition or mixed signals",
        "confidence": "LOW",
        "missing_inputs": missing,
        "quadrant": quadrant,
        "inputs": {
            "gdp_growth": gdp_growth,
            "inflation": inflation,
            "credit_spread_bps": credit_spread,
            "yield_curve_bps": yield_curve,
        }
    }


def format_regime_block(regime_data: dict) -> str:
    """Format regime dict into a readable block for prompt injection."""
    r = regime_data
    lines = [
        f"Regime: {r['regime']} [{r['confidence']} confidence]",
        f"Quadrant: {r['quadrant']}",
        f"Description: {r['description']}",
    ]
    if r["missing_inputs"]:
        lines.append(f"[DATA MISSING: {', '.join(r['missing_inputs'])}]")
    return "\n".join(lines)


if __name__ == "__main__":
    # Example: Stagflation test
    result = detect_regime(
        gdp_growth=0.5,
        inflation=4.2,
        credit_spread=350,
        yield_curve=-80,
    )
    print(format_regime_block(result))