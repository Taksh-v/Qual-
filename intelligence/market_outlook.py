from __future__ import annotations

from datetime import datetime, timedelta


def _regime_bias(regime: str) -> tuple[str, str]:
    r = (regime or "").upper()
    if "GOLDILOCKS" in r or "REFLATION" in r or "EARLY_RECOVERY" in r:
        return (
            "Soft-landing with improving risk appetite and broadening earnings breadth.",
            "Stay pro-risk but hedge left-tail volatility around data/event clusters.",
        )
    if "RECESSION" in r or "STAGFLATION" in r or "DEFLATION" in r:
        return (
            "Growth shock risk remains elevated; earnings revisions likely skew lower.",
            "Prioritize quality, duration balance, and tactical downside protection.",
        )
    return (
        "Mixed macro with rotating leadership and elevated factor dispersion.",
        "Use barbell positioning and tighten entry levels around catalysts.",
    )


def market_outlook(macro: str, sectors: str, regime: str, portfolio_size: float = 500_000_000) -> str:
    _ = macro, sectors, portfolio_size
    market_view, trade_frame = _regime_bias(regime)
    today = datetime.now()

    timeline = [
        (today + timedelta(days=2), "US ISM / PMI release", "BULL if breadth improves, BEAR if contraction deepens"),
        (today + timedelta(days=8), "US CPI print", "BULL on downside surprise, BEAR on sticky core inflation"),
        (today + timedelta(days=14), "Central bank communication", "BULL if easing bias grows, BEAR if policy stays restrictive"),
        (today + timedelta(days=21), "Labor market update", "BULL for soft-landing data, BEAR if jobless trend accelerates"),
    ]

    lines = [
        f"Regime bias: {regime}",
        f"Base tactical view: {market_view}",
        f"Trade framework: {trade_frame}",
        "",
        "CATALYST TIMELINE",
    ]
    for date, event, impact in timeline:
        lines.append(f"• {date.strftime('%Y-%m-%d')} {event}: {impact}")

    lines.extend(
        [
            "",
            "Market believes: Disinflation will continue without meaningful growth damage.",
            "Reality check: Services inflation and credit pass-through can delay policy easing.",
            "Trade implication: Prefer quality cyclicals over high-beta duration proxies; add hedges into event risk.",
        ]
    )

    return "\n".join(lines)
