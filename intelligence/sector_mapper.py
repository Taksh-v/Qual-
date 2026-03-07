from __future__ import annotations

from typing import Any


SECTOR_UNIVERSE = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]


def _bucket_for_regime(regime: str) -> tuple[list[str], list[str], list[str]]:
    r = (regime or "").upper()
    if "GOLDILOCKS" in r or "REFLATION" in r or "EARLY_RECOVERY" in r:
        return (
            ["Information Technology", "Industrials", "Financials", "Consumer Discretionary"],
            ["Utilities", "Consumer Staples", "Real Estate"],
            ["Energy", "Materials", "Health Care", "Communication Services"],
        )
    if "RECESSION" in r or "STAGFLATION" in r or "DEFLATION" in r:
        return (
            ["Health Care", "Consumer Staples", "Utilities"],
            ["Consumer Discretionary", "Industrials", "Financials", "Real Estate"],
            ["Energy", "Materials", "Information Technology", "Communication Services"],
        )
    return (
        ["Health Care", "Information Technology", "Communication Services"],
        ["Consumer Discretionary", "Real Estate"],
        ["Energy", "Materials", "Industrials", "Consumer Staples", "Financials", "Utilities"],
    )


def _conviction(sector: str, overweight: list[str], underweight: list[str]) -> str:
    if sector in overweight:
        return "H"
    if sector in underweight:
        return "M"
    return "L"


def sector_impact(macro_analysis: str, regime: str, mcx_tickers: dict[str, Any]) -> str:
    """
    Emit parser-friendly sector calls used by the dashboard:
      OVERWEIGHT / UNDERWEIGHT / NEUTRAL
      • Sector | Conviction H/M/L | rationale
    """
    _ = mcx_tickers
    first_line = (macro_analysis or "").splitlines()[0] if macro_analysis else "Macro signal mixed."
    overweight, underweight, neutral = _bucket_for_regime(regime)

    def make_line(sector: str, stance: str) -> str:
        conv = _conviction(sector, overweight, underweight)
        rationale = (
            f"{stance} on relative earnings/revision sensitivity under {regime.lower()} regime. "
            f"Anchor: {first_line[:110]}"
        )
        return f"• {sector} | Conviction {conv} | {rationale}"

    lines = ["OVERWEIGHT"]
    lines.extend(make_line(s, "Overweight") for s in overweight)
    lines.append("")
    lines.append("UNDERWEIGHT")
    lines.extend(make_line(s, "Underweight") for s in underweight)
    lines.append("")
    lines.append("NEUTRAL")
    for s in SECTOR_UNIVERSE:
        if s in overweight or s in underweight:
            continue
        lines.append(make_line(s, "Neutral"))

    return "\n".join(lines).strip()
