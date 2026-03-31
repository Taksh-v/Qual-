"""
market_context.py
-----------------
Builds structured thematic market context sections from live indicator data.

These sections are injected into the LLM prompt alongside news context chunks,
giving the model a broad, calibrated picture of the macro environment before
it reads any news — preventing it from hallucinating "unknown" conditions.

Sections produced:
  1. Yield curve & rates regime
  2. Inflation & real rate landscape
  3. Credit cycle positioning
  4. Equity & risk sentiment
  5. FX & dollar regime
  6. Commodities & energy
  7. Global macro divergence (India, Europe, Asia)
  8. Labour & activity dashboard
  9. Money & liquidity conditions
  10. Upcoming catalyst checklist (rules-based)
"""

from __future__ import annotations

from typing import Any


# ── Helper utilities ──────────────────────────────────────────────────────────

def _v(ind: dict[str, Any], key: str, fmt: str = ".2f") -> str:
    """Return formatted value or 'n/a'."""
    val = ind.get(key)
    if val is None:
        return "n/a"
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return str(val)


def _sign(ind: dict[str, Any], key: str) -> str:
    val = ind.get(key)
    if val is None:
        return ""
    try:
        return "+" if float(val) >= 0 else ""
    except (TypeError, ValueError):
        return ""


def _flag(condition: bool, true_tag: str = "▲", false_tag: str = "▼") -> str:
    """Standardized upward/downward indicators."""
    return true_tag if condition else false_tag


def _safe(ind: dict[str, Any], key: str) -> float | None:
    try:
        val = ind.get(key)
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# ── Section builders ──────────────────────────────────────────────────────────

def build_yield_curve_section(ind: dict[str, Any]) -> str:
    """Yield curve shape, inversion status, and rate regime."""
    lines = ["=== YIELD CURVE & RATES REGIME ==="]

    ffr = _safe(ind, "fed_funds_rate")
    y2  = _safe(ind, "yield_2y")
    y10 = _safe(ind, "yield_10y")
    y30 = _safe(ind, "yield_30y")
    y3m = _safe(ind, "yield_3m")
    curve_2_10  = _safe(ind, "yield_curve")          # bps
    curve_10_3m = _safe(ind, "yield_curve_10y3m")    # bps
    term_prem   = _safe(ind, "term_premium_proxy")

    lines.append(
        f"Fed Funds Rate: {_v(ind,'fed_funds_rate')}%  |  "
        f"3M: {_v(ind,'yield_3m')}%  |  2Y: {_v(ind,'yield_2y')}%  |  "
        f"10Y: {_v(ind,'yield_10y')}%  |  30Y: {_v(ind,'yield_30y')}%"
    )

    if curve_2_10 is not None:
        inverted = curve_2_10 < 0
        status = "INVERTED ⚠ (recession signal)" if inverted else ("FLAT" if abs(curve_2_10) < 20 else "NORMAL")
        lines.append(f"10Y-2Y Spread: {curve_2_10:+.0f} bps → Curve is {status}")

    if curve_10_3m is not None:
        inv2 = curve_10_3m < 0
        lines.append(
            f"10Y-3M Spread: {curve_10_3m:+.0f} bps → "
            + ("INVERTED ⚠ (historically most reliable recession predictor)" if inv2 else "positive")
        )

    if term_prem is not None:
        lines.append(f"Term Premium (30Y-3M): {term_prem:+.0f} bps")

    # Implied Fed stance
    if ffr is not None and y2 is not None:
        gap = y2 - ffr
        if gap > 0.5:
            stance = "Market pricing RATE CUTS (2Y below FFR — dovish pivot expected)"
        elif gap < -0.5:
            stance = "Market pricing MORE HIKES (2Y above FFR — hawkish)"
        else:
            stance = "Rate expectations broadly stable (2Y near FFR)"
        lines.append(f"Rate path signal: {stance}")

    return "\n".join(lines)


def build_inflation_section(ind: dict[str, Any]) -> str:
    """CPI, PCE, breakevens, and real rate landscape."""
    lines = ["=== INFLATION & REAL RATE LANDSCAPE ==="]

    cpi      = _safe(ind, "inflation_cpi")
    core_cpi = _safe(ind, "inflation_core_cpi")
    pce_core = _safe(ind, "pce_core")
    be5      = _safe(ind, "breakeven_5y")
    be10     = _safe(ind, "breakeven_10y")
    real10   = _safe(ind, "real_rate_proxy")
    fed_real = _safe(ind, "fed_real_rate")

    parts = []
    if cpi is not None:      parts.append(f"CPI: {cpi:.2f}%")
    if core_cpi is not None: parts.append(f"Core CPI: {core_cpi:.2f}%")
    if pce_core is not None: parts.append(f"Core PCE (Fed target): {pce_core:.2f}%")
    lines.append("  ".join(parts) if parts else "  Inflation data unavailable")

    if be5 is not None or be10 is not None:
        lines.append(
            f"Breakeven inflation:  5Y={_v(ind,'breakeven_5y')}%  |  10Y={_v(ind,'breakeven_10y')}%"
        )
        if be5 is not None and be10 is not None:
            slope = be10 - be5
            direction = "steepening (long-term inflation risk rising)" if slope > 0.05 else (
                "flattening (inflation expected to moderate)" if slope < -0.05 else "flat"
            )
            lines.append(f"5Y-10Y breakeven slope: {slope:+.3f}% → {direction}")

    if real10 is not None:
        regime = (
            "restrictive (above neutral ~0.5%)" if real10 > 0.5
            else "accommodative (below neutral)" if real10 < -0.5
            else "near neutral"
        )
        lines.append(f"Real rate (10Y-CPI proxy): {real10:.2f}% → Policy is {regime}")

    if fed_real is not None:
        lines.append(
            f"Fed real rate (FFR - Core PCE): {fed_real:+.2f}%  "
            + ("→ Monetary policy RESTRICTIVE" if fed_real > 1.5 else
               "→ Monetary policy accommodative" if fed_real < 0 else
               "→ Monetary policy near neutral")
        )

    return "\n".join(lines)


def build_credit_section(ind: dict[str, Any]) -> str:
    """Credit spreads, TED spread, mortgage rates — risk appetite proxy."""
    lines = ["=== CREDIT CYCLE & RISK APPETITE ==="]

    hy    = _safe(ind, "credit_hy")
    ig    = _safe(ind, "credit_ig")
    bb    = _safe(ind, "credit_bb")
    gap   = _safe(ind, "credit_spread_gap")
    ted   = _safe(ind, "ted_spread")
    mort  = _safe(ind, "mort_rate_30y")

    if hy is not None:
        hy_regime = (
            "WIDE (stress/risk-off)" if hy > 600
            else "elevated (caution)" if hy > 400
            else "tight (risk appetite healthy)" if hy < 300
            else "normal"
        )
        lines.append(f"HY Spread: {hy:.0f} bps → {hy_regime}")

    if ig is not None:
        ig_regime = (
            "WIDE (credit stress)" if ig > 180
            else "tight (benign)" if ig < 100
            else "normal"
        )
        lines.append(f"IG Spread: {ig:.0f} bps → {ig_regime}")

    if bb is not None:
        lines.append(f"BB Spread: {bb:.0f} bps")

    if gap is not None:
        lines.append(f"HY-IG Compression Gap: {gap:.0f} bps {'(spreads compressing — risk ON)' if gap < 300 else '(spreads wide — risk OFF)'}")

    if ted is not None:
        lines.append(
            f"TED Spread (bank funding stress): {ted:.2f}%  "
            + ("⚠ ELEVATED (interbank stress)" if ted > 0.5 else "normal")
        )

    if mort is not None:
        lines.append(
            f"30Y Mortgage Rate: {mort:.2f}%  "
            + ("→ housing UNDER PRESSURE" if mort > 7.0 else "→ housing moderately stressed" if mort > 6.0 else "→ housing accessible")
        )

    return "\n".join(lines)


def build_equity_section(ind: dict[str, Any]) -> str:
    """Equities, VIX, sector rotation signals."""
    lines = ["=== EQUITY & RISK SENTIMENT ==="]

    sp500 = _safe(ind, "sp500")
    nas   = _safe(ind, "nasdaq")
    vix   = _safe(ind, "vix")
    russ  = _safe(ind, "russell2000")

    if sp500 is not None: lines.append(f"S&P 500: {sp500:,.0f}")
    if nas   is not None: lines.append(f"Nasdaq: {nas:,.0f}")
    if russ  is not None: lines.append(f"Russell 2000 (small cap): {russ:,.0f}")

    if vix is not None:
        vix_regime = (
            "EXTREME FEAR (>30)" if vix > 30
            else "ELEVATED FEAR (20-30)" if vix > 20
            else "COMPLACENCY (<13)" if vix < 13
            else "normal range"
        )
        lines.append(f"VIX: {vix:.1f} → {vix_regime}")

    # Sector snapshot
    sectors = {
        "sector_tech":    "Tech (XLK)",
        "sector_energy":  "Energy (XLE)",
        "sector_finance": "Finance (XLF)",
        "sector_health":  "Health (XLV)",
        "sector_consumer":"Consumer (XLY)",
    }
    sec_parts = [f"{name}={_v(ind,key,',.0f')}" for key, name in sectors.items() if ind.get(key)]
    if sec_parts:
        lines.append("Sectors: " + "  |  ".join(sec_parts))

    # Global
    global_idx = {
        "ftse100": "FTSE100", "nikkei225": "Nikkei225",
        "hangseng": "HangSeng", "dax": "DAX",
        "nifty50": "Nifty50", "sensex": "Sensex",
    }
    g_parts = [f"{name}={_v(ind,key,',.0f')}" for key, name in global_idx.items() if ind.get(key)]
    if g_parts:
        lines.append("Global indices: " + "  |  ".join(g_parts))

    return "\n".join(lines)


def build_fx_section(ind: dict[str, Any]) -> str:
    """Dollar index, major pairs, EM currencies."""
    lines = ["=== FX & DOLLAR REGIME ==="]

    dxy = _safe(ind, "dxy")
    if dxy is not None:
        dxy_regime = (
            "STRONG DOLLAR (>105) → EM outflows, commodity headwinds, US exporter pressure"
            if dxy > 105 else
            "WEAK DOLLAR (<95) → EM relief rally, commodity tailwind"
            if dxy < 95 else
            "neutral dollar range"
        )
        lines.append(f"DXY (dollar index): {dxy:.2f} → {dxy_regime}")

    pairs = [
        ("eur_usd", "EUR/USD"), ("gbp_usd", "GBP/USD"),
        ("usd_jpy", "USD/JPY"), ("usd_cny", "USD/CNY"),
        ("usd_inr",  "USD/INR"),
    ]
    pair_parts = [f"{name}={_v(ind,key)}" for key, name in pairs if ind.get(key)]
    if pair_parts:
        lines.append("FX crosses: " + "  |  ".join(pair_parts))

    return "\n".join(lines)


def build_commodities_section(ind: dict[str, Any]) -> str:
    """Oil, gold, metals, energy."""
    lines = ["=== COMMODITIES & ENERGY ==="]

    wti    = _safe(ind, "oil_wti")
    brent  = _safe(ind, "oil_brent")
    gold   = _safe(ind, "gold")
    silver = _safe(ind, "silver")
    copper = _safe(ind, "copper")
    natgas = _safe(ind, "natural_gas")

    if wti is not None:
        oil_regime = (
            "HIGH OIL (>90) → inflation pressure, consumer squeeze, stagflation risk"
            if wti > 90 else
            "LOW OIL (<60) → deflationary signal, EM demand concern"
            if wti < 60 else
            "oil in moderate range"
        )
        lines.append(f"WTI: ${wti:.2f}  |  Brent: ${_v(ind,'oil_brent')}  →  {oil_regime}")

    if gold is not None:
        lines.append(
            f"Gold: ${gold:,.0f}  "
            + ("→ safe-haven bid elevated" if gold > 2200 else "→ normal range")
        )

    others = []
    if silver  is not None: others.append(f"Silver=${_v(ind,'silver')}")
    if copper  is not None: others.append(f"Copper=${_v(ind,'copper')} (industrial demand proxy)")
    if natgas  is not None: others.append(f"Nat Gas=${_v(ind,'natural_gas')}")
    if others:
        lines.append("  |  ".join(others))

    return "\n".join(lines)


def build_labour_activity_section(ind: dict[str, Any]) -> str:
    """Labour market & activity summary."""
    lines = ["=== LABOUR & ACTIVITY DASHBOARD ==="]

    unemp  = _safe(ind, "unemployment")
    claims = _safe(ind, "initial_claims")
    jolts  = _safe(ind, "jolts_openings")
    nfp    = _safe(ind, "nonfarm_payrolls")
    part   = _safe(ind, "participation_rate")
    pmi    = _safe(ind, "pmi_mfg")
    ret    = _safe(ind, "us_retail_sales")
    ind_p  = _safe(ind, "us_industrial_prod")
    cap    = _safe(ind, "capacity_utilization")
    hse    = _safe(ind, "us_housing_starts")
    lei    = _safe(ind, "conf_board_lei")
    sent   = _safe(ind, "consumer_sentiment")

    labour_parts = []
    if unemp  is not None: labour_parts.append(f"Unemployment={unemp:.1f}%")
    if claims is not None: labour_parts.append(f"Initial Claims={claims:,.0f}k")
    if jolts  is not None: labour_parts.append(f"JOLTS Openings={jolts:.2f}mn")
    if nfp    is not None: labour_parts.append(f"NFP={nfp:+,.0f}k")
    if part   is not None: labour_parts.append(f"Participation={part:.1f}%")
    if labour_parts:
        lines.append("Labour: " + "  |  ".join(labour_parts))

    act_parts = []
    if pmi   is not None: act_parts.append(f"PMI Mfg={pmi:.1f}{'(contraction<50)' if pmi<50 else ''}")
    if ret   is not None: act_parts.append(f"Retail Sales=${ret:.0f}bn")
    if ind_p is not None: act_parts.append(f"Industrial Prod={ind_p:.1f}")
    if cap   is not None: act_parts.append(f"Capacity Util={cap:.1f}%")
    if hse   is not None: act_parts.append(f"Housing Starts={hse:,.0f}k")
    if lei   is not None: act_parts.append(f"LEI={lei:.2f}")
    if sent  is not None: act_parts.append(f"Consumer Sentiment={sent:.1f}")
    if act_parts:
        lines.append("Activity: " + "  |  ".join(act_parts))

    return "\n".join(lines)


def build_money_liquidity_section(ind: dict[str, Any]) -> str:
    """M2, Fed balance sheet, money velocity."""
    lines = ["=== MONEY & LIQUIDITY CONDITIONS ==="]

    m2    = _safe(ind, "m2_money_supply")
    fedbs = _safe(ind, "fed_balance_sheet")
    m2v   = _safe(ind, "m2_velocity")

    if m2    is not None: lines.append(f"M2 Money Supply: ${m2:,.0f}bn")
    if fedbs is not None:
        lines.append(
            f"Fed Balance Sheet: ${fedbs:,.0f}bn  "
            + ("→ QT in progress (declining)" if fedbs < 8_000_000 else "→ still elevated post-QE")
        )
    if m2v   is not None:
        lines.append(
            f"M2 Velocity: {m2v:.3f}  "
            + ("→ low (deflationary pressure)" if m2v < 1.5 else "→ rising (potential inflation signal)")
        )

    return "\n".join(lines)


def build_india_section(ind: dict[str, Any]) -> str:
    """India-specific macro context."""
    lines = ["=== INDIA MACRO SNAPSHOT ==="]

    used = []
    for key, label in [
        ("nifty50",              "Nifty 50"),
        ("sensex",               "Sensex"),
        ("nifty_bank",           "Nifty Bank"),
        ("nifty_it",             "Nifty IT"),
        ("usd_inr",              "USD/INR"),
        ("india_gdp_growth",     "India GDP Growth %"),
        ("india_inflation_cpi",  "India CPI %"),
        ("india_current_account","India Current Account % GDP"),
        ("india_fdi_inflow",     "India FDI % GDP"),
    ]:
        val = ind.get(key)
        if val is not None:
            used.append(f"{label}: {val}")

    if used:
        lines.extend(used)
    else:
        lines.append("India data unavailable")

    return "\n".join(lines)


def build_catalyst_checklist(ind: dict[str, Any]) -> str:
    """
    Rules-based checklist of key market conditions and flags.
    Helps the model quickly identify what regime/stress to address.
    """
    lines = ["=== MARKET CONDITION FLAGS ==="]
    flags: list[str] = []

    vix   = _safe(ind, "vix")
    curve = _safe(ind, "yield_curve")
    hy    = _safe(ind, "credit_hy")
    ig    = _safe(ind, "credit_ig")
    real  = _safe(ind, "real_rate_proxy")
    dxy   = _safe(ind, "dxy")
    wti   = _safe(ind, "oil_wti")
    ffr   = _safe(ind, "fed_funds_rate")
    cpi   = _safe(ind, "inflation_cpi")

    if vix   is not None and vix > 25:    flags.append("⚠ HIGH VOLATILITY: VIX elevated (risk-off environment)")
    if curve is not None and curve < 0:   flags.append("⚠ YIELD CURVE INVERTED: recession signal (10Y-2Y < 0)")
    if hy    is not None and hy > 500:    flags.append("⚠ CREDIT STRESS: HY spreads very wide")
    if real  is not None and real > 2.0:  flags.append("⚠ VERY RESTRICTIVE REAL RATES: growth headwind")
    if real  is not None and real < -1.5: flags.append("⚠ DEEPLY NEGATIVE REAL RATES: inflation-suppressive policy failure")
    if dxy   is not None and dxy > 107:   flags.append("⚠ DOLLAR VERY STRONG: EM FX pressure, commodity headwind")
    if wti   is not None and wti > 95:    flags.append("⚠ HIGH OIL: stagflation risk elevated")
    if wti   is not None and wti < 55:    flags.append("⚠ VERY LOW OIL: demand/deflation signal")
    if cpi   is not None and cpi > 5.0:   flags.append("⚠ ABOVE-TARGET INFLATION: Fed under pressure")
    if ffr   is not None and cpi is not None and ffr < cpi - 1:
        flags.append("⚠ NEGATIVE REAL FUNDS RATE: Fed behind the curve")

    if not flags:
        flags.append("✓ No extreme stress signals detected in current indicator set")

    lines.extend(flags)
    return "\n".join(lines)


# ── Master builder ──────────────────────────────────────────────────────────

def build_full_market_context(ind: dict[str, Any]) -> str:
    """
    Assemble all thematic sections into a single block that can be
    injected into the LLM prompt before the news context chunks.
    """
    sections = [
        build_catalyst_checklist(ind),
        build_yield_curve_section(ind),
        build_inflation_section(ind),
        build_credit_section(ind),
        build_equity_section(ind),
        build_fx_section(ind),
        build_commodities_section(ind),
        build_labour_activity_section(ind),
        build_money_liquidity_section(ind),
        build_india_section(ind),
    ]
    return "\n\n".join(s for s in sections if s.strip())


def build_compact_market_context(ind: dict[str, Any]) -> str:
    """
    Compact version (for brief-mode prompts) — just the flags + 3 key sections.
    """
    sections = [
        build_catalyst_checklist(ind),
        build_yield_curve_section(ind),
        build_inflation_section(ind),
        build_credit_section(ind),
    ]
    return "\n\n".join(s for s in sections if s.strip())
