"""
cross_asset_analyzer.py
-----------------------
Analyzes cross-asset relationships to detect:
  - Confirmations (multiple assets agree → high conviction)
  - Divergences (assets disagree → caution flag or opportunity)
  - Classic macro relationships under stress

Key cross-asset signals used by macro traders:
  1. Equities vs. Credit (lead/lag relationship)
  2. Dollar vs. Commodities (inverse, especially oil + gold)
  3. Yield curve vs. Equities (curve steepening = growth rebound)
  4. VIX vs. Equity trend (vol crushes = complacency or bull)
  5. Oil vs. Inflation breakevens (commodity-driven inflation)
  6. EM FX vs. DXY (EM vulnerability)
  7. Gold vs. Real Rates (inverse — gold = real rate hedge)
"""

from typing import Optional


def analyze_cross_asset(indicators: dict) -> dict:
    """
    Run cross-asset signal checks.
    Returns a dict with confirmations, divergences, and a summary signal.
    """
    confirmations = []
    divergences = []
    alerts = []

    vix = indicators.get("vix")
    dxy = indicators.get("dxy")
    oil = indicators.get("oil_wti")
    yield_curve = indicators.get("yield_curve")  # bps, negative = inverted
    credit_hy = indicators.get("credit_hy")
    inflation_cpi = indicators.get("inflation_cpi")
    pmi_mfg = indicators.get("pmi_mfg")
    unemployment = indicators.get("unemployment")
    fed_funds = indicators.get("fed_funds_rate")
    yield_10y = indicators.get("yield_10y")

    # --- 1. Yield curve signal ---
    if yield_curve is not None:
        if yield_curve < -50:
            alerts.append(f"CURVE INVERTED ({yield_curve:.0f}bps) — recession signal, historically leads by 12-18m")
        elif yield_curve < 0:
            alerts.append(f"CURVE FLAT/SLIGHTLY INVERTED ({yield_curve:.0f}bps) — late cycle warning")
        elif yield_curve > 100:
            confirmations.append(f"CURVE STEEP ({yield_curve:.0f}bps) — early recovery / reflation signal")

    # --- 2. Credit vs. equities divergence check ---
    if credit_hy is not None:
        if credit_hy > 600:
            alerts.append(f"HY SPREADS ELEVATED ({credit_hy}bps) — credit stress, equity rally may be fake")
        elif credit_hy < 300:
            confirmations.append(f"TIGHT HY SPREADS ({credit_hy}bps) — credit confirming risk-on")

    # --- 3. VIX regime ---
    if vix is not None:
        if vix > 30:
            alerts.append(f"VIX ELEVATED ({vix}) — fear regime, vol selling may be opportunity")
        elif vix < 15:
            alerts.append(f"VIX COMPRESSED ({vix}) — complacency risk, potential vol spike")
        else:
            confirmations.append(f"VIX NORMAL ({vix}) — orderly market conditions")

    # --- 4. Dollar vs. Commodities ---
    if dxy is not None and oil is not None:
        if dxy > 105 and oil < 75:
            confirmations.append("STRONG USD + WEAK OIL — disinflationary, bad for EM and commodities")
        elif dxy < 100 and oil > 85:
            confirmations.append("WEAK USD + STRONG OIL — inflationary combo, supports reflation trade")
        elif dxy > 105 and oil > 85:
            divergences.append("STRONG USD + STRONG OIL — divergence: supply shock or geopolitical premium on oil")

    # --- 5. Real rate proxy check ---
    if yield_10y is not None and inflation_cpi is not None:
        real_rate = yield_10y - inflation_cpi
        if real_rate > 2.0:
            alerts.append(f"REAL RATES VERY RESTRICTIVE ({real_rate:.1f}%) — significant drag on growth + valuations")
        elif real_rate < 0:
            alerts.append(f"NEGATIVE REAL RATES ({real_rate:.1f}%) — financial repression, gold + commodities favored")
        else:
            confirmations.append(f"REAL RATES POSITIVE BUT MODERATE ({real_rate:.1f}%) — balanced conditions")

    # --- 6. PMI vs. Credit signal ---
    if pmi_mfg is not None and credit_hy is not None:
        if pmi_mfg < 50 and credit_hy > 400:
            divergences.append("PMI CONTRACTING + WIDE SPREADS — double confirmation of growth slowdown")
        elif pmi_mfg > 52 and credit_hy < 300:
            confirmations.append("PMI EXPANDING + TIGHT SPREADS — growth and credit aligned bullish")

    # --- 7. Fed vs. Yield curve positioning ---
    if fed_funds is not None and yield_10y is not None:
        if fed_funds > yield_10y:
            alerts.append(f"FED FUNDS ({fed_funds}%) ABOVE 10Y ({yield_10y}%) — curve inverted at policy level, highly restrictive")

    # Compute overall cross-asset signal
    n_confirm = len(confirmations)
    n_alert = len(alerts)
    n_div = len(divergences)

    if n_confirm >= 3 and n_alert <= 1:
        overall = "ALIGNED_BULLISH"
    elif n_alert >= 3 and n_confirm <= 1:
        overall = "ALIGNED_BEARISH"
    elif n_div >= 2:
        overall = "DIVERGENT — mixed signals, lower conviction"
    elif n_alert >= 2 and n_confirm >= 2:
        overall = "CONFLICTED — risk events may resolve direction"
    else:
        overall = "NEUTRAL / INSUFFICIENT_DATA"

    return {
        "overall_signal": overall,
        "confirmations": confirmations,
        "divergences": divergences,
        "alerts": alerts,
    }


def format_cross_asset_block(analysis: dict) -> str:
    """Format cross-asset analysis for prompt injection."""
    lines = [f"CROSS-ASSET SIGNAL: {analysis['overall_signal']}"]

    if analysis["confirmations"]:
        lines.append("  Confirming signals:")
        for c in analysis["confirmations"]:
            lines.append(f"    ✓ {c}")

    if analysis["divergences"]:
        lines.append("  Divergences (caution):")
        for d in analysis["divergences"]:
            lines.append(f"    ⚡ {d}")

    if analysis["alerts"]:
        lines.append("  Alerts:")
        for a in analysis["alerts"]:
            lines.append(f"    ⚠ {a}")

    return "\n".join(lines)


if __name__ == "__main__":
    test_indicators = {
        "vix": 28,
        "dxy": 107,
        "oil_wti": 82,
        "yield_curve": -80,
        "credit_hy": 520,
        "inflation_cpi": 3.7,
        "yield_10y": 4.8,
        "pmi_mfg": 46.5,
        "fed_funds_rate": 5.25,
    }
    result = analyze_cross_asset(test_indicators)
    print(format_cross_asset_block(result))