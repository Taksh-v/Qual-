from __future__ import annotations

BRIEF_RESPONSE_FORMAT = (
    "Use EXACTLY this format — no deviations, no extra sections:\n\n"
    "Regime: <Risk-On / Risk-Off / Transitional>\n"
    "Dominant theme: <1-4 word theme, e.g. Supply-Shock Inflation>\n\n"
    "Direct answer: <one specific sentence with a number or named event>\n"
    "Data snapshot: <list 4-7 actual numbers from provided data, e.g. CPI=3.2%, 10Y=4.5%, VIX=22, WTI=$83>\n"
    "Causal architecture:\n"
    "Primary: <trigger> → <effect 1> → <effect 2>\n"
    "Secondary: <trigger> → <effect 1>\n"
    "Feedback: <specifically how an effect reinforces a trigger>\n"
    "What is happening:\n"
    "- <named specific event or development with source citation [Sx]>\n"
    "- <direct cause: which data point or event is driving this>\n"
    "Cross-asset impact:\n"
    "- Equities: <direction ▲/▼/●, e.g. ▼ S&P -1-3%> — <mechanism>\n"
    "- Rates: <direction ▲/▼/●, e.g. ▲ 10Y +5-10 bps> — <mechanism>\n"
    "- FX: <direction ▲/▼/●, e.g. ▲ DXY to 102> — <mechanism>\n"
    "Positioning:\n"
    "- <Action: e.g. Overweight>: <Instrument> — <Rationale>\n"
    "Predicted events:\n"
    "- <event 1 (time horizon, prob%)>: <narrative>; trigger: <specific level/event>; invalidation: <opposite level/event>.\n"
    "- <event 2 (time horizon, prob%)>: <narrative>; trigger: <specific level/event>; invalidation: <opposite level/event>.\n"
    "Scenarios (probabilities must add to 100%):\n"
    "- Base (~55%): <specific outcome>; trigger: <specific level/event>; invalidation: <opposite level/event>.\n"
    "- Bull (~25%): <specific outcome>; trigger: <specific level/event>; invalidation: <opposite level/event>.\n"
    "- Bear (~20%): <specific outcome>; trigger: <specific level/event>; invalidation: <opposite level/event>.\n"
    "Key levels:\n"
    "- <Instrument 1>: <level> — <significance>\n"
    "Historical analog: <1-2 sentences comparing to a past comparable episode>\n"
    "What to watch:\n"
    "- <specific data release: name it, expected date if known>\n"
    "Data gaps:\n"
    "- <list 1-2 missing data points that would increase confidence>\n"
    "Confidence: <HIGH/MEDIUM/LOW> - <one specific reason citing data availability>\n"
)

DETAILED_RESPONSE_FORMAT = (
    "Use EXACTLY this format — no deviations, no extra sections:\n\n"
    "Regime: <Risk-On / Risk-Off / Transitional>\n"
    "Dominant theme: <1-4 word theme>\n\n"
    "Executive summary: <3 sentences: name event, key numeric moves, and biggest risk>\n"
    "Direct answer: <clear stance with specific numbers and named assets>\n"
    "Data snapshot: <list 8-12 actual indicator values from context>\n"
    "Causal architecture:\n"
    "Primary: <trigger> → <effect> → <market transmission> → <cross-asset ripple>\n"
    "Secondary: <trigger> → <effect> → <ripple effect>\n"
    "Feedback: <reinforcing loop mechanism>\n"
    "What is happening:\n"
    "- <specific development 1 with event name and number [S1]>\n"
    "- <direct mechanism with numbers>\n"
    "- <structural driver or second-order effect with numbers>\n"
    "Cross-asset impact:\n"
    "- Equities: <direction ▲/▼/●> — <sector rotation logic and valuation mechanism>\n"
    "- Rates: <direction ▲/▼/●> — <yield direction and curve shape implication>\n"
    "- FX: <direction ▲/▼/●> — <which pairs move and structural reason>\n"
    "- Commodities: <direction ▲/▼/●> — <supply-demand or dollar channel>\n"
    "- Credit: <direction ▲/▼/●> — <spread widening/tightening logic>\n"
    "Positioning:\n"
    "- <Action>: <Instruments/Sectors> — <rationale based on causal chain>\n"
    "- <Action>: <Instruments/Sectors> — <rationale>\n"
    "- <Hedge>: <Instrument> — <rationale>\n"
    "Predicted events:\n"
    "- <event 1 (next 7-30d, prob%)>: <explicit causal chain>; trigger: <level/event>; invalidation: <opposite level/event>.\n"
    "- <event 2 (next 7-30d, prob%)>: <explicit causal chain>; trigger: <level/event>; invalidation: <opposite level/event>.\n"
    "- <event 3 (next 30-90d, prob%)>: <narrative>; trigger: <level/event>; invalidation: <opposite level/event>.\n"
    "Scenarios (probabilities must add to 100%):\n"
    "- Base (~55%): <specific number-anchored outcome>; trigger: <level>; invalidation: <level>.\n"
    "- Bull (~25%): <named asset class upside>; trigger: <level>; invalidation: <level>.\n"
    "- Bear (~20%): <named tail risk downside>; trigger: <level>; invalidation: <level>.\n"
    "Key levels:\n"
    "- <Instrument 1>: <level> — <significance>\n"
    "- <Instrument 2>: <level> — <significance>\n"
    "Key risks:\n"
    "- <risk 1: specific event/data that breaks base case>\n"
    "- <risk 2: policy or geopolitical event; describe mechanism>\n"
    "Historical analog: <1-3 paragraph comparison to a past episode, highlighting similarities and crucial differences>\n"
    "Time horizons:\n"
    "- 24-72h: <immediate reaction to watch>\n"
    "- 1-4 weeks: <medium-term catalyst window>\n"
    "- 1-3 months: <structural trend implication>\n"
    "What to watch:\n"
    "- <item 1: specific data release with threshold>\n"
    "- <item 2: central bank event or speech>\n"
    "Data gaps:\n"
    "- <list 1-3 missing data points indicating what's absent from the context>\n"
    "Confidence: <HIGH/MEDIUM/LOW> - <specific reason citing data quality and coverage>\n"
)


from functools import lru_cache

@lru_cache(maxsize=4)
def get_response_format_block(response_mode: str) -> str:
    mode = (response_mode or "brief").strip().lower()
    if mode == "detailed":
        return DETAILED_RESPONSE_FORMAT
    return BRIEF_RESPONSE_FORMAT


FINANCIAL_MECHANICS_BLOCK = """
FINANCIAL MECHANICS — follow these exactly, never contradict:
• Safe-haven demand (geopolitical risk/recession fear) → investors BUY Treasuries/Gold/JPY → bond prices RISE → Treasury yields FALL (price and yield move inversely)
• Fed rate hike / hawkish surprise → short-end yields RISE sharply → yield curve flattens or inverts → PE multiples compress → growth equities fall more than value
• Inflation surprise HIGHER → real rates may fall if central bank is behind curve → commodities rally → bond sell-off (yields rise) → equities volatile
• Dollar (DXY) STRENGTHENS → commodities priced in USD fall (oil, gold, metals) → EM currencies weaken → EM debt/equity capital outflow risk → US exporter earnings headwind
• Credit spreads WIDEN → implies rising default risk → risk-off → equities and high-yield bonds fall together → liquidity premium rises
• Oil price SPIKE → input cost inflation → consumer spending power eroded → transport/airline/industrial margins compressed → central banks face growth-inflation tradeoff
• VIX above 20 → elevated fear → institutional hedging demand rises → options skew increases → short-term equity drawdowns more likely
• Yield curve INVERTS (2Y > 10Y) → historical recession predictor within 12-18 months → banks' net interest margin compresses → lending activity slows

RULE-CONFLICT RESOLUTION (CRITICAL):
If multiple mechanics apply but suggest contradictory outcomes (e.g., Oil Spike suggests yields RISE due to inflation, but VIX Spike suggests yields FALL due to safe-haven):
1. Identify the DOMINANT DRIVER (is this primarily an inflation shock, a growth shock, or a geopolitical shock?).
2. Apply the mechanic that matches the dominant driver.
3. Explicitly state the resolution in your Causal Architecture (e.g., "Yields rise as inflation repricing dominates safe-haven flow").
"""

COT_REASONING_BLOCK = """
CHAIN-OF-THOUGHT REASONING — silently work through these four steps before writing your final response:
Step 1 — EXTRACT: Identify the 3-5 most significant data points from "Live indicators" and "News context" that directly address the question. Note each value with its unit.
Step 2 — TRACE MECHANICS: Apply the FINANCIAL MECHANICS rules above. Trace the causal chain from trigger → transmission channel → asset impact. Name the specific instrument, direction, and approximate magnitude (bps, %, $).
Step 3 — ASSESS SENTIMENT: Based on regime, signals, and news, classify the overall market tone (risk-on / risk-off / mixed). State which asset class benefits most and why.
Step 4 — SYNTHESISE: Combine Steps 1-3 into a regime-consistent, data-grounded conclusion. Set scenario probabilities to match the balance of evidence from Steps 1-3.
Note: Do NOT print these steps in your output. Use them only to form your reasoning before writing the formatted response.
"""

STRICT_RULES_BLOCK = """
STRICT RULES — output will be rejected if violated:
1. NEVER use phrases like "heightened uncertainty", "risk-off sentiment", "downward pressure", or "increased volatility" WITHOUT an actual number immediately after (e.g. "VIX spiked to 28, signaling elevated fear").
2. EVERY "Market impact" bullet MUST contain at least one number or a named sector/currency/instrument.
3. If the question mentions a geopolitical event, NAME the specific conflict (e.g. "Russia-Ukraine war", "US-China tariffs") — do not write "current geopolitical conflict".
4. Bond/safe-haven mechanics: "safe-haven demand → yields FALL" (do NOT say yields rise during flight to safety unless it's a sell-off scenario).
5. Do not mention anything outside the provided context and indicators — no AI company references, no events not in the data.
6. Use [Sx] citation tags when making claims directly supported by provided news context.
7. All scenario probabilities must sum to 100%.
8. Output ONLY the formatted response — no preamble, no explanation, no labels outside the format.
9. If an exact number is not explicitly present in Live indicators or News context, write "N/A" instead of inventing a value.
10. Every item in "Predicted events" MUST include horizon, probability, and explicit trigger/invalidation condition.
11. Keep all forecasts directionally consistent with the STRUCTURED REASONING OBJECT unless explicit contradictory evidence is cited from context.
"""


def build_quality_rewrite_prompt(section_name: str, draft_text: str) -> str:
    return f"""
You are a senior editorial reviewer for institutional research output.
Rewrite for maximum clarity and usefulness without changing factual meaning.

Rules:
1. Keep the same output structure and labels.
2. Do not add facts or citations that are not already present.
3. Keep language simple and concrete.
4. Remove repetition, hedging, and filler.
5. Make it read like a premium assistant response: crisp, coherent, and directly useful.
6. Output only the revised text.

Section:
{section_name}

Draft:
{draft_text}
""".strip()


def build_citation_repair_prompt(draft_text: str, formatted_context: str) -> str:
    return f"""
You must repair citations in this response.

Rules:
1. Keep the exact same structure and section labels.
2. Do not add new facts.
3. Add [Sx] citations to factual lines using provided context ids.
4. If a factual line cannot be supported, replace that line's claim with: "Insufficient custom evidence."
5. Output only the repaired response text.

Context:
{formatted_context}

Draft:
{draft_text}
""".strip()
