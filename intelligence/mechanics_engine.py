"""
mechanics_engine.py
Evaluates live market data to determine dominant forces and return programmatic
financial mechanics instructions for the LLM.
"""

from typing import Optional
from intelligence.market_data import MarketDataSnapshot

def parse_float(val: str) -> Optional[float]:
    """Parse numeric strings like '4500', '4.50%', '$82.50' into floats."""
    if not val or val == "N/A":
        return None
    try:
        clean = val.replace("$", "").replace("%", "").replace(",", "").strip()
        return float(clean)
    except ValueError:
        return None

def get_dominant_mechanics_block(snap: MarketDataSnapshot) -> str:
    """
    Returns the specific financial mechanics instruction block based on 
    current market realities to force strict Causal Architecture adherence.
    """
    vix = parse_float(snap.vix)
    wti = parse_float(snap.wti)
    ten_year = parse_float(snap.ten_year)
    dxy = parse_float(snap.dxy)

    # Base rules that always apply
    base_rules = (
        "FINANCIAL MECHANICS — follow these exactly, never contradict:\n"
        "• Yield curve INVERTS (2Y > 10Y) → historical recession predictor within 12-18 months.\n"
    )

    dominant_driver = ""

    # Rule 1: Extreme Risk-Off (VIX > 25)
    if vix is not None and vix >= 25.0:
        dominant_driver += (
            "• [DOMINANT THEME: FEAR/RISK-OFF] VIX is highly elevated (>25). Institutional hedging demand is surging. "
            "Safe-haven demand dominates: investors BUY Treasuries/Gold/JPY → bond prices RISE → Treasury yields FALL. "
            "Equities face severe multiple compression and short-term drawdowns.\n"
        )
    # Rule 2: Complacency / Greed (VIX < 13)
    elif vix is not None and vix <= 13.0:
        dominant_driver += (
            "• [DOMINANT THEME: COMPLACENCY] VIX is very low (<13). Risk-taking is elevated. "
            "Equities grind higher, credit spreads tightly compress, and short-volatility trades dominate.\n"
        )
        
    # Rule 3: Inflation Shock / Supply Shock (WTI > 85 OR 10Y > 4.5%)
    if wti is not None and wti > 85.0:
        dominant_driver += (
            "• [DOMINANT THEME: CHRONIC INFLATION/SUPPLY SHOCK] Oil > $85. Input cost inflation erodes consumer spending power and transport margins. "
            "Central banks face growth-inflation tradeoff, keeping rates higher for longer.\n"
        )
    if ten_year is not None and ten_year > 4.5:
        dominant_driver += (
            "• [DOMINANT THEME: HIGHER FOR LONGER RATES] 10Y Yield > 4.5%. High cost of capital compresses "
            "PE multiples, particularly for unprofitable/growth tech. Refinancing risks rise for highly-levered companies.\n"
        )

    # Rule 4: Strong Dollar (DXY > 105)
    if dxy is not None and dxy > 105.0:
        dominant_driver += (
            "• [DOMINANT THEME: STRONG DOLLAR] DXY > 105. Commodities priced in USD face headwinds. "
            "EM currencies weaken, raising EM debt/equity capital outflow risks. US multinational exporter earnings face FX headwinds.\n"
        )

    # Fallback to general mechanics if no extremes
    if not dominant_driver:
        dominant_driver = (
            "• Safe-haven demand → Treasuries up → yields DOWN.\n"
            "• Rate hike fear → yields UP → PE multiples compress.\n"
            "• Dollar STRENGTHENS → EM currencies weaken, commodities pressured.\n"
        )

    resolution_instruction = (
        "\nRULE-CONFLICT RESOLUTION (CRITICAL):\n"
        "If you encounter conflicting rules based on the news, favor the DOMINANT THEMES established above based on the live data snapshot. "
        "Do not hedge. State explicitly: 'Event X causes Y, dominating over Z due to [Dominant Theme].'\n"
    )

    return base_rules + dominant_driver + resolution_instruction
