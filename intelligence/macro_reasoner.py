from __future__ import annotations

import json
import re
from typing import Any

from intelligence.market_context import (
    build_full_market_context,
    build_compact_market_context,
)


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    try:
        cleaned = str(value).strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        return float(cleaned) if cleaned else None
    except Exception:
        return None


def _contains_any(text: str, terms: list[str]) -> bool:
    t = (text or "").lower()
    return any(term in t for term in terms)


def _build_event_linking_block(
    question: str,
    indicators: dict[str, Any],
    formatted_context: str,
    regime: dict[str, Any],
    cross_asset: dict[str, Any],
) -> str:
    q = (question or "").lower()
    ctx = (formatted_context or "").lower()
    regime_name = regime.get("regime", "TRANSITIONAL")
    signal = cross_asset.get("overall_signal", "NEUTRAL")

    y10 = _to_float(indicators.get("yield_10y"))
    y2 = _to_float(indicators.get("yield_2y"))
    curve = _to_float(indicators.get("yield_curve"))
    if curve is None and y10 is not None and y2 is not None:
        curve = round((y10 - y2) * 100, 1)
    oil = _to_float(indicators.get("oil_wti") or indicators.get("oil_brent"))
    dxy = _to_float(indicators.get("dxy"))
    vix = _to_float(indicators.get("vix"))
    credit_hy = _to_float(indicators.get("credit_hy"))

    links: list[dict[str, str]] = []

    rate_tight = (
        (y10 is not None and y10 >= 4.4)
        or (curve is not None and curve < 0)
        or _contains_any(ctx + " " + q, ["hawkish", "rate hike", "higher for longer", "policy tightening", "treasury yields rose"])
    )
    if rate_tight:
        trigger = []
        if y10 is not None:
            trigger.append(f"10Y={y10}%")
        if curve is not None:
            trigger.append(f"curve={curve}bps")
        trigger_text = ", ".join(trigger) if trigger else "Hawkish rates narrative in recent context"
        links.append(
            {
                "event": "Long-duration growth and tech valuations face downside pressure",
                "chain": "Policy/yield pressure persists → discount rates rise → long-duration equity multiples compress",
                "prob": "40%",
                "horizon": "7-30d",
                "trigger": trigger_text,
                "invalid": "10Y breaks below 4.2% with softer inflation signals",
            }
        )

    oil_disruption = (
        (oil is not None and oil >= 85)
        or _contains_any(
            ctx + " " + q,
            [
                "supply disruption",
                "pipeline outage",
                "shipping disruption",
                "opec cut",
                "middle east",
                "sanctions",
                "oil shock",
            ],
        )
    )
    if oil_disruption:
        oil_trigger = f"Oil={oil}" if oil is not None else "Energy supply shock keywords in news context"
        links.append(
            {
                "event": "Energy complex outperforms while transport and rate-sensitive sectors underperform",
                "chain": "Supply shock / elevated oil → input costs rise → inflation risk reprices → sector dispersion widens",
                "prob": "30%",
                "horizon": "7-30d",
                "trigger": oil_trigger,
                "invalid": "Oil falls below recent support with easing geopolitical risk",
            }
        )

    earnings_stress = _contains_any(
        ctx + " " + q,
        ["weak earnings", "earnings miss", "guidance cut", "profit warning", "downgrade", "margin compression"],
    )
    if earnings_stress:
        links.append(
            {
                "event": "Earnings downgrades propagate into broader equity downside",
                "chain": "Earnings disappointments broaden → analyst revisions turn negative → index-level drawdown risk rises",
                "prob": "30%",
                "horizon": "7-30d",
                "trigger": "Recent earnings stress language in retrieved context",
                "invalid": "Positive guidance revisions outnumber cuts",
            }
        )

    credit_stress = (
        (credit_hy is not None and credit_hy >= 420)
        or _contains_any(ctx + " " + q, ["credit spreads widened", "liquidity stress", "default risk", "funding stress"])
    )
    if credit_stress:
        hy_trigger = f"HY spread={credit_hy}bps" if credit_hy is not None else "Credit stress flags in context"
        links.append(
            {
                "event": "Risk assets weaken as financing conditions tighten",
                "chain": "Credit stress widens risk premia → financing costs rise → equities and HY credit reprice lower",
                "prob": "35%",
                "horizon": "7-30d",
                "trigger": hy_trigger,
                "invalid": "HY spreads compress sustainably and liquidity indicators improve",
            }
        )

    if not links:
        trigger = []
        if vix is not None:
            trigger.append(f"VIX={vix}")
        if dxy is not None:
            trigger.append(f"DXY={dxy}")
        if y10 is not None:
            trigger.append(f"10Y={y10}%")
        trigger_text = ", ".join(trigger) if trigger else "Mixed live signals without dominant catalyst"
        links.append(
            {
                "event": f"{regime_name} regime persistence with range-bound cross-asset behavior",
                "chain": "No single dominant shock → mixed data keeps conviction moderate → markets trade by data surprises",
                "prob": "45%",
                "horizon": "7-30d",
                "trigger": trigger_text,
                "invalid": "A major policy/geopolitical shock creates one-way positioning",
            }
        )

    lines = [
        "EVENT LINK MAP (deterministic priors from live data + recent news):",
        f"Current regime prior: {regime_name} | Cross-asset prior: {signal}",
    ]
    for i, item in enumerate(links[:4], start=1):
        lines.extend(
            [
                f"[E{i}] Trigger evidence: {item['trigger']}",
                f"     Causal chain: {item['chain']}",
                f"     Predicted event ({item['horizon']}): {item['event']}",
                f"     Probability: {item['prob']}",
                f"     Invalidation: {item['invalid']}",
            ]
        )
    return "\n".join(lines)


def summarize_indicators(indicators: dict[str, Any]) -> str:
    if not indicators:
        return "No parsed indicators."
    priority = [
        # Policy & rates
        "fed_funds_rate", "yield_3m", "yield_2y", "yield_10y", "yield_30y",
        "yield_curve", "yield_curve_10y3m", "term_premium_proxy",
        # Real rates & inflation
        "real_rate_proxy", "real_rate_10y", "fed_real_rate",
        "inflation_cpi", "inflation_core_cpi", "pce_core",
        "breakeven_5y", "breakeven_10y",
        # Credit
        "credit_hy", "credit_ig", "credit_spread_gap", "ted_spread", "mort_rate_30y",
        # Markets
        "sp500", "nasdaq", "vix", "dxy", "gold", "oil_wti", "oil_brent",
        # Activity
        "gdp_growth", "unemployment", "pmi_mfg", "initial_claims", "jolts_openings",
        "us_retail_sales", "us_industrial_prod", "capacity_utilization",
        "consumer_sentiment", "conf_board_lei",
        # Money
        "m2_money_supply", "fed_balance_sheet", "m2_velocity",
        # FX
        "eur_usd", "usd_jpy", "usd_inr",
        # India
        "nifty50", "sensex", "india_gdp_growth", "india_inflation_cpi",
    ]
    lines = []
    for key in priority:
        if key in indicators:
            lines.append(f"- {key}: {indicators[key]}")
    for key in sorted(indicators.keys()):
        if key not in priority:
            lines.append(f"- {key}: {indicators[key]}")
    return "\n".join(lines[:45]) if lines else "No parsed indicators."


def summarize_key_numbers(indicators: dict[str, Any]) -> str:
    keys = [
        "sp500", "nasdaq", "vix", "dxy",
        "gold", "oil_wti", "oil_brent", "copper",
        "fed_funds_rate", "yield_2y", "yield_10y", "yield_30y",
        "yield_curve", "yield_curve_10y3m", "term_premium_proxy",
        "real_rate_proxy", "real_rate_10y", "fed_real_rate",
        "inflation_cpi", "inflation_core_cpi", "pce_core",
        "breakeven_5y", "breakeven_10y",
        "credit_hy", "credit_ig", "credit_spread_gap", "ted_spread",
        "mort_rate_30y", "unemployment", "initial_claims", "jolts_openings",
        "gdp_growth", "pmi_mfg", "us_retail_sales", "us_industrial_prod",
        "m2_money_supply", "fed_balance_sheet",
        "eur_usd", "usd_jpy", "usd_inr",
        "nifty50", "sensex",
        "btc_usd",
    ]
    parts = [f"{k}={indicators[k]}" for k in keys if k in indicators]
    return ", ".join(parts) if parts else "No reliable live numbers available."


def summarize_cross_asset(cross_asset: dict[str, Any]) -> str:
    if not cross_asset:
        return "No cross-asset summary."
    lines = [f"overall_signal: {cross_asset.get('overall_signal', 'N/A')}"]
    for k in ("confirmations", "divergences", "alerts"):
        vals = cross_asset.get(k, []) or []
        if vals:
            lines.append(f"{k}:")
            lines.extend(f"- {v}" for v in vals[:8])
    return "\n".join(lines)


def _format_reasoning_object(reasoning_analysis: dict[str, Any] | None) -> str:
    if not reasoning_analysis:
        return "No structured reasoning object available."
    try:
        mao = reasoning_analysis.get("market_analysis_object", {})
        signal = reasoning_analysis.get("signal_score", {})
        confirm = reasoning_analysis.get("cross_asset_confirmation", {})
        analog = reasoning_analysis.get("historical_analog", {})
        scenarios = reasoning_analysis.get("scenario_generator", {})
        diagnostics = reasoning_analysis.get("diagnostics", {})
        graph = reasoning_analysis.get("reasoning_graph", {})

        compact = {
            "event": mao.get("event") or reasoning_analysis.get("event_detection", {}).get("primary_event", {}).get("event_type"),
            "market_regime": signal.get("market_regime", mao.get("market_regime", "mixed_transition")),
            "signal_strength": signal.get("confidence", mao.get("signal_strength", 0.5)),
            "impact_level": signal.get("impact_level", "medium"),
            "cross_asset_moves": mao.get("cross_asset_moves", {}),
            "cross_asset_confirmation": {
                "risk_sentiment": confirm.get("risk_sentiment", "neutral"),
                "energy_shock": confirm.get("energy_shock", "neutral"),
                "safe_haven_flow": confirm.get("safe_haven_flow", "neutral"),
                "signal_confidence": confirm.get("signal_confidence", "LOW"),
                "confirmation_ratio": confirm.get("confirmation_ratio", 0.0),
                "contradiction_count": confirm.get("contradiction_count", 0),
            },
            "historical_analog": {
                "similar_events": analog.get("similar_events", []),
                "oil_avg_move": analog.get("oil_avg_move", "N/A"),
                "sp500_avg_move": analog.get("sp500_avg_move", "N/A"),
                "volatility_increase": analog.get("volatility_increase", "unknown"),
            },
            "scenarios": {
                "base_case": scenarios.get("base_case", {}),
                "bull_case": scenarios.get("bull_case", {}),
                "bear_case": scenarios.get("bear_case", {}),
            },
            "diagnostics": {
                "consistency": signal.get("consistency", diagnostics.get("consistency", "LOW")),
                "indicator_completeness": diagnostics.get("indicator_completeness", signal.get("components", {}).get("data_completeness", 0.0)),
                "cache_hit": diagnostics.get("cache_hit", False),
            },
            "reasoning_graph_summary": {
                "node_count": graph.get("node_count", 0),
                "edge_count": graph.get("edge_count", 0),
            },
        }
        return json.dumps(compact, ensure_ascii=False, indent=2)
    except Exception:
        return "No structured reasoning object available."


def _reasoning_predicted_event_lines(
    reasoning_analysis: dict[str, Any] | None,
    *,
    max_items: int = 3,
) -> list[str]:
    if not reasoning_analysis:
        return []
    scenarios = reasoning_analysis.get("scenario_generator", {})
    if not isinstance(scenarios, dict):
        return []

    lines: list[str] = []
    order = ("base_case", "bull_case", "bear_case")
    for idx, key in enumerate(order, start=1):
        if len(lines) >= max_items:
            break
        item = scenarios.get(key, {})
        if not isinstance(item, dict):
            continue
        prob = item.get("probability_pct")
        horizon = item.get("horizon", "7-30d")
        narrative = item.get("narrative")
        trigger = item.get("trigger", "N/A")
        invalid = item.get("invalidation", "N/A")
        if not narrative:
            continue
        prob_text = f"{prob}%" if prob is not None else "N/A"
        lines.append(
            f"- Event {idx} ({horizon}, {prob_text}): {narrative}; trigger: {trigger}; invalidation: {invalid}."
        )
    return lines


def _reasoning_scenario_lines(reasoning_analysis: dict[str, Any] | None) -> list[str]:
    if not reasoning_analysis:
        return []
    scenarios = reasoning_analysis.get("scenario_generator", {})
    if not isinstance(scenarios, dict):
        return []

    labels = {
        "base_case": "Base",
        "bull_case": "Bull",
        "bear_case": "Bear",
    }
    out: list[str] = []
    for key in ("base_case", "bull_case", "bear_case"):
        item = scenarios.get(key, {})
        if not isinstance(item, dict):
            continue
        label = labels[key]
        prob = item.get("probability_pct")
        narrative = item.get("narrative")
        spx = item.get("sp500_range")
        oil = item.get("oil_range")
        if not narrative:
            continue
        prob_text = f"~{prob}%" if prob is not None else "N/A"
        extras = []
        if spx and spx != "N/A":
            extras.append(f"S&P500={spx}")
        if oil and oil != "N/A":
            extras.append(f"Oil={oil}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        out.append(f"- {label} ({prob_text}): {narrative}{suffix}.")
    return out


def build_unified_response_prompt(
    question: str,
    classification: dict[str, Any],
    regime: dict[str, Any],
    cross_asset: dict[str, Any],
    indicators: dict[str, Any],
    formatted_context: str,
    geography: str = "US",
    horizon: str = "MEDIUM_TERM",
    response_mode: str = "brief",
    live_data_meta: dict[str, Any] | None = None,
    reasoning_analysis: dict[str, Any] | None = None,
) -> str:
    mode = (response_mode or "brief").strip().lower()
    if mode not in {"brief", "detailed"}:
        mode = "brief"
    live_data_meta = live_data_meta or {}

    key_nums = summarize_key_numbers(indicators)
    regime_str = f"{regime.get('regime','UNKNOWN')} | confidence={regime.get('confidence','LOW')}"
    signal_str = cross_asset.get("overall_signal", "NEUTRAL")

    # Build a rich numeric snapshot line for the prompt
    snapshot_parts: list[str] = []
    num_map = [
        ("sp500", "S&P500"), ("nasdaq", "Nasdaq"), ("gold", "Gold"),
        ("oil_wti", "WTI"), ("oil_brent", "Brent"), ("vix", "VIX"),
        ("dxy", "DXY"), ("yield_10y", "10Y"), ("yield_2y", "2Y"),
        ("yield_30y", "30Y"), ("yield_curve", "Curve(bps)"),
        ("inflation_cpi", "CPI%"), ("fed_funds_rate", "FedFunds%"),
        ("credit_hy", "HY_bps"), ("credit_ig", "IG_bps"),
        ("unemployment", "Unemp%"), ("gdp_growth", "GDP%"),
        ("pmi_mfg", "PMI_Mfg"), ("real_rate_proxy", "RealRate"),
    ]
    for key, label in num_map:
        val = indicators.get(key)
        if val is not None:
            snapshot_parts.append(f"{label}={val}")
    live_snapshot = ", ".join(snapshot_parts) if snapshot_parts else "No live data available."

    mechanics_block = """
FINANCIAL MECHANICS — follow these exactly, never contradict:
• Safe-haven demand (geopolitical risk/recession fear) → investors BUY Treasuries/Gold/JPY → bond prices RISE → Treasury yields FALL (price and yield move inversely)
• Fed rate hike / hawkish surprise → short-end yields RISE sharply → yield curve flattens or inverts → PE multiples compress → growth equities fall more than value
• Inflation surprise HIGHER → real rates may fall if central bank is behind curve → commodities rally → bond sell-off (yields rise) → equities volatile
• Dollar (DXY) STRENGTHENS → commodities priced in USD fall (oil, gold, metals) → EM currencies weaken → EM debt/equity capital outflow risk → US exporter earnings headwind
• Credit spreads WIDEN → implies rising default risk → risk-off → equities and high-yield bonds fall together → liquidity premium rises
• Oil price SPIKE → input cost inflation → consumer spending power eroded → transport/airline/industrial margins compressed → central banks face growth-inflation tradeoff
• VIX above 20 → elevated fear → institutional hedging demand rises → options skew increases → short-term equity drawdowns more likely
• Yield curve INVERTS (2Y > 10Y) → historical recession predictor within 12-18 months → banks' net interest margin compresses → lending activity slows
"""

    if mode == "brief":
        format_block = (
            "Use EXACTLY this format — no deviations, no extra sections:\n\n"
            "Direct answer: <one specific sentence with a number or named event>\n"
            "Data snapshot: <list 4-7 actual numbers from provided data, e.g. CPI=3.2%, 10Y=4.5%, VIX=22, WTI=$83>\n"
            "Causal chain: <specific trigger> → <immediate market effect> → <transmission channel> → <asset impact>\n"
            "What is happening:\n"
            "- <named specific event or development with source citation [Sx] if available>\n"
            "- <direct cause: which data point or event is driving this>"
            " — use actual numbers e.g. 'yield_10y rose 18bps' not 'yields rose'\n"
            "- <second-order feedback: how does this feed back into other markets>\n"
            "Market impact:\n"
            "- Equities: <specific index direction + which sectors gain/lose and WHY>\n"
            "- Rates/Bonds: <yield direction with bps magnitude if possible; safe-haven = yields DOWN>\n"
            "- FX: <which currency strengthens/weakens and mechanism>\n"
            "Predicted events:\n"
            "- <event 1 over next 7-30d + probability + cause/effect chain + trigger level>\n"
            "- <event 2 over next 7-30d + probability + cause/effect chain + trigger level>\n"
            "Scenarios (probabilities must add to 100%):\n"
            "- Base (~55%): <specific outcome with a number, e.g. 'S&P holds above X'>\n"
            "- Bull (~25%): <upside trigger + market reaction>\n"
            "- Bear (~20%): <downside trigger + market reaction>\n"
            "What to watch:\n"
            "- <specific data release: name it, expected date if known>\n"
            "- <specific indicator level that would change the view>\n"
            "Confidence: <HIGH/MEDIUM/LOW> - <one specific reason citing data availability>\n"
        )
    else:
        format_block = (
            "Use EXACTLY this format — no deviations, no extra sections:\n\n"
            "Executive summary: <3 sentences: name the specific event, key numeric moves, and biggest risk>\n"
            "Direct answer: <clear stance with specific numbers and named assets>\n"
            "Data snapshot: <list 8-12 actual indicator values>\n"
            "Causal chain: <specific trigger> → <direct effect> → <market transmission> → <cross-asset ripple>\n"
            "What is happening:\n"
            "- <specific development 1 with event name and number [S1]>\n"
            "- <direct mechanism: e.g. 'Fed raised 25bps → 2Y yield spiked to X%'>\n"
            "- <structural driver or second-order effect with numbers>\n"
            "Market impact (include specific sector or currency names):\n"
            "- Equities: <index + sector rotation logic with earnings/valuation mechanism>\n"
            "- Rates/Bonds: <yield direction + bps move + curve shape implication>\n"
            "- FX: <which pairs move + direction + reason>\n"
            "- Commodities: <oil/gold/metals direction + supply-demand or dollar channel>\n"
            "Predicted events:\n"
            "- <event 1 (next 7-30d) + probability + explicit causal chain from data/news>\n"
            "- <event 2 (next 7-30d) + probability + explicit causal chain from data/news>\n"
            "- <event 3 (next 30-90d) + probability + invalidation trigger>\n"
            "Scenarios (probabilities must add to 100%):\n"
            "- Base (~55%): <specific number-anchored outcome over 4-8 weeks>\n"
            "- Bull (~25%): <named trigger + asset class upside>\n"
            "- Bear (~20%): <named tail risk + asset class downside>\n"
            "Key risks:\n"
            "- <risk 1: name the specific event/data that breaks the base case>\n"
            "- <risk 2: policy or geopolitical event; describe mechanism>\n"
            "Time horizons:\n"
            "- 24-72h: <immediate market reaction to watch>\n"
            "- 1-4 weeks: <medium-term catalyst window>\n"
            "- 1-3 months: <structural trend implication>\n"
            "What to watch:\n"
            "- <item 1: specific data release with name and threshold>\n"
            "- <item 2: central bank event or speech>\n"
            "- <item 3: cross-asset level that confirms or denies the scenario>\n"
            "Confidence: <HIGH/MEDIUM/LOW> - <specific reason citing data quality and coverage>\n"
        )

    # Build structured market context (thematic sections from indicators)
    if mode == "brief":
        market_context_block = build_compact_market_context(indicators)
    else:
        market_context_block = build_full_market_context(indicators)

    event_linking_block = _build_event_linking_block(
        question=question,
        indicators=indicators,
        formatted_context=formatted_context,
        regime=regime,
        cross_asset=cross_asset,
    )
    reasoning_object_block = _format_reasoning_object(reasoning_analysis)

    cot_block = """
CHAIN-OF-THOUGHT REASONING — silently work through these four steps before writing your final response:
Step 1 — EXTRACT: Identify the 3-5 most significant data points from "Live indicators" and "News context" that directly address the question. Note each value with its unit.
Step 2 — TRACE MECHANICS: Apply the FINANCIAL MECHANICS rules above. Trace the causal chain from trigger → transmission channel → asset impact. Name the specific instrument, direction, and approximate magnitude (bps, %, $).
Step 3 — ASSESS SENTIMENT: Based on regime, signals, and news, classify the overall market tone (risk-on / risk-off / mixed). State which asset class benefits most and why.
Step 4 — SYNTHESISE: Combine Steps 1-3 into a regime-consistent, data-grounded conclusion. Set scenario probabilities to match the balance of evidence from Steps 1-3.
Note: Do NOT print these steps in your output. Use them only to form your reasoning before writing the formatted response.
"""

    return f"""You are a senior macro strategist at a tier-1 investment bank. Your output is read by institutional portfolio managers who need real, numbered, actionable intelligence — not commentary.
{mechanics_block}
{cot_block}
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

{format_block}
---
Question: {question}
Geography: {geography} | Horizon: {horizon}
Regime: {regime_str}
Cross-asset signal: {signal_str}
Live indicators: {live_snapshot}
STRUCTURED MARKET CONTEXT (derived from live data — use this for quantitative grounding):
{market_context_block}
DETERMINISTIC EVENT-LINK PRIORS (use these as forecasting anchors and keep directional consistency):
{event_linking_block}
STRUCTURED REASONING OBJECT (NEWS → EVENTS → IMPACT MAP → SIGNAL → SCENARIOS):
{reasoning_object_block}
News context (cite as [S1], [S2], ... when supporting specific claims):
{formatted_context}
---
Write the response now:""".strip()


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
5. Output only the revised text.

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


def generate_unified_fallback(
    question: str,
    regime: dict[str, Any],
    cross_asset: dict[str, Any],
    reasoning_analysis: dict[str, Any] | None = None,
    response_mode: str = "brief",
) -> str:
    mode = (response_mode or "brief").strip().lower()
    regime_name = regime.get("regime", "TRANSITIONAL")
    signal = cross_asset.get("overall_signal", "NEUTRAL")
    pred_lines = _reasoning_predicted_event_lines(reasoning_analysis, max_items=3)
    scenario_lines = _reasoning_scenario_lines(reasoning_analysis)

    if mode == "detailed":
        if not pred_lines:
            pred_lines = [
                "- Event 1 (7-30d, 40%): Rate-sensitive growth equities face valuation pressure if 10Y holds above 4.5%; invalidation: 10Y breaks below 4.2%.",
                "- Event 2 (7-30d, 35%): Credit-risk repricing broadens if HY spreads widen further; invalidation: sustained spread compression.",
                "- Event 3 (30-90d, 25%): Growth scare lifts safe-haven demand, pulling long yields lower; invalidation: upside growth surprise.",
            ]
        if not scenario_lines:
            scenario_lines = [
                "- Base (~55%): Markets range-trade; next data print is within expectations. No major repricing.",
                "- Bull (~25%): Inflation prints below forecast → rate cut expectations front-run → equities and bonds rally.",
                "- Bear (~20%): Growth or inflation data surprise forces hawkish pivot → equities sell off >5%, credit spreads widen.",
            ]
        return "\n".join([
            "Executive summary: Live market data feed is unavailable. Current regime signals a transitional environment. The primary risk drivers are policy rate trajectory, growth-inflation balance, and central bank communication. Positioning should remain defensive until data clarity improves.",
            "Direct answer: Hold balanced positioning — no high-conviction directional call is warranted without live data confirmation.",
            "Data snapshot: Live data unavailable — indicators below are from last known regime state.",
            f"Causal chain: Uncertain macro regime → mixed growth/inflation signals → elevated cross-asset correlation → reduced diversification benefit",
            "What is happening:",
            f"- Macro regime is {regime_name} with cross-asset signal: {signal}.",
            "- Central bank policy path remains data-dependent; rate expectations are volatile.",
            "- Growth and inflation trends are not yet confirmed in either direction.",
            "Market impact:",
            "- Equities: Risk of multiple compression if rates stay elevated above 4.5%; defensives (utilities, healthcare) outperform cyclicals.",
            "- Rates/Bonds: Safe-haven demand would push 10Y yields lower; a hawkish surprise pushes 2Y higher, flattening the curve.",
            "- FX: Dollar (DXY) typically strengthens in risk-off; EM currencies face outflow pressure.",
            "- Commodities: Oil sensitive to growth outlook; gold rises on safe-haven flows when real rates fall.",
            "Predicted events:",
            *pred_lines,
            "Scenarios (probabilities must add to 100%):",
            *scenario_lines,
            "Key risks:",
            "- A data surprise (CPI, payrolls) outside ±0.2% of consensus can shift rate expectations rapidly.",
            "- Central bank forward guidance divergence between Fed, ECB, BoJ can trigger FX volatility.",
            "Time horizons:",
            "- 24-72h: Watch any scheduled Fed speeches or PMI releases.",
            "- 1-4 weeks: Next CPI/PCE release will determine if rate expectations reprice.",
            "- 1-3 months: If inflation plateau confirmed, soft-landing narrative strengthens equities.",
            "What to watch:",
            "- Next CPI / PCE print versus consensus (±0.2% matters).",
            "- 10Y Treasury yield: above 4.7% signals renewed rate pressure on equities.",
            "- VIX above 20 indicates elevated fear; below 15 confirms risk-on stability.",
            "Confidence: LOW - Live data unavailable; framework-only reasoning.",
        ])
    if not pred_lines:
        pred_lines = [
            "- Event 1 (7-30d, 40%): Elevated rates keep pressure on long-duration equities; invalidation: clear disinflation surprise.",
            "- Event 2 (7-30d, 35%): Risk-off episodes favor USD and defensives if credit stress rises; invalidation: spread tightening.",
        ]
    if not scenario_lines:
        scenario_lines = [
            "- Base (~55%): Data in-line → range-bound markets, no trend breakout.",
            "- Bull (~25%): Soft inflation data → rate cut pricing → equity/bond rally.",
            "- Bear (~20%): Hot inflation or weak growth → stagflation fear → risk-off selloff.",
        ]
    return "\n".join([
        f"Direct answer: Regime is {regime_name} (signal: {signal}) — no high-conviction trade without live data confirmation.",
        "Data snapshot: Live data unavailable — limited to regime framework.",
        f"Causal chain: {regime_name} regime → mixed signals → no dominant directional catalyst identified",
        "What is happening:",
        "- Macro signals are mixed across growth, rates, and risk appetite — no single driver dominates.",
        "- Rate sensitivity is elevated; one data surprise can shift market direction quickly.",
        "- Cross-asset correlations are elevated (bonds and equities moving together), limiting hedges.",
        "Market impact:",
        "- Equities: Vulnerable to multiple compression (P/E falls) if rates stay above 4.5%.",
        "- Rates/Bonds: Flight to safety = 10Y yields fall; hawkish shock = 2Y yields spike.",
        "- FX: Dollar strength in risk-off; EM currencies weaken on capital outflows.",
        "Predicted events:",
        *pred_lines,
        "Scenarios (probabilities must add to 100%):",
        *scenario_lines,
        "What to watch:",
        "- Next CPI and payrolls print versus consensus.",
        "- 10Y yield direction (above 4.7% = equity headwind; below 4.2% = support).",
        "Confidence: LOW - Limited live evidence; reasoning from regime framework only.",
    ])


def generate_contextual_fallback(
    question: str,
    regime: dict[str, Any],
    cross_asset: dict[str, Any],
    indicators: dict[str, Any],
    context_chunks: list[dict[str, Any]] | None = None,
    reasoning_analysis: dict[str, Any] | None = None,
    response_mode: str = "brief",
) -> str:
    q = (question or "").lower()
    context_chunks = context_chunks or []
    has_evidence = len(context_chunks) > 0
    signal = cross_asset.get("overall_signal", "NEUTRAL / INSUFFICIENT_DATA")
    regime_name = regime.get("regime", "TRANSITIONAL")
    regime_conf = regime.get("confidence", "LOW")
    pred_lines = _reasoning_predicted_event_lines(reasoning_analysis, max_items=3)
    scenario_lines = _reasoning_scenario_lines(reasoning_analysis)

    c1 = "[S1]" if has_evidence else ""
    c2 = "[S2]" if has_evidence and len(context_chunks) > 1 else c1
    c3 = "[S3]" if has_evidence and len(context_chunks) > 2 else c2

    if "silver" in q:
        dxy = indicators.get("dxy")
        y10 = indicators.get("yield_10y")
        infl = indicators.get("inflation_cpi")
        real_rate = None
        if isinstance(y10, (int, float)) and isinstance(infl, (int, float)):
            real_rate = round(float(y10) - float(infl), 2)

        why_1 = "Silver often moves with changes in the US dollar and real yields."
        if isinstance(dxy, (int, float)):
            why_1 = f"Current dollar context (DXY {dxy}) can materially drive short-term silver moves."

        why_2 = "Silver has both precious-metal and industrial-demand behavior, so growth headlines matter."
        why_3 = "Fast price jumps are usually flow-driven (positioning/liquidity) rather than single-factor."
        if real_rate is not None:
            why_3 = f"Estimated real-rate proxy ({real_rate}%) can shift opportunity cost for holding silver."

        data_points = []
        if isinstance(dxy, (int, float)):
            data_points.append(f"dxy={dxy}")
        if isinstance(y10, (int, float)):
            data_points.append(f"yield_10y={y10}%")
        if isinstance(infl, (int, float)):
            data_points.append(f"inflation_cpi={infl}%")
        if real_rate is not None:
            data_points.append(f"real_rate_proxy={real_rate}%")
        snapshot = ", ".join(data_points) if data_points else "Insufficient custom evidence."

        risks = [
            "Headline-driven volatility can reverse quickly when liquidity is thin.",
            "If macro data surprises, silver can decouple from recent trend intraday.",
        ]
        watch = [
            "US dollar move (DXY) and real-yield direction over the next 24-72 hours.",
            "Gold-silver relative move and any industrial-demand headlines (especially PMIs).",
            "Central-bank communication and inflation surprises that reprice rate expectations.",
        ]
        confidence_reason = (
            "topic-specific drivers are clear, but direct high-quality evidence in retrieved context is limited"
            if not has_evidence
            else "core drivers are consistent, but attribution confidence remains medium without cleaner source alignment"
        )
        if (response_mode or "brief").lower() == "detailed":
            lines = [
                f"Executive summary: Sudden silver moves are typically explained by a joint shift in dollar, real-rate, and positioning channels under a {regime_name} ({regime_conf}) regime.",
                "Direct answer: Use dollar direction plus real-rate change as the primary diagnostic, then confirm with cross-asset behavior before adding risk.",
                f"Data snapshot: {snapshot}",
                "Key drivers:",
                f"- Dollar move -> silver USD pricing and EM demand sensitivity -> {why_1} {c1}".strip(),
                f"- Growth-demand narrative -> industrial silver demand expectations -> {why_2} {c2}".strip(),
                f"- Real-rate repricing and flows -> opportunity-cost and momentum effects -> {why_3} {c3}".strip(),
                "Market impact:",
                f"- Equities: risk-sensitive sectors may weaken when silver spike is risk-off driven (signal={signal}).",
                "- Rates: real-yield shifts can dominate metal pricing in the short run.",
                "- FX/Commodities: dollar trend and broader metals complex provide confirmation.",
                "Predicted events:",
                "- Event 1 (7-30d, 45%): If real yields stay firm, silver upside fades into consolidation; invalidation: real rates fall sharply.",
                "- Event 2 (7-30d, 30%): If dollar weakens while PMIs stabilize, silver resumes upside; invalidation: DXY re-acceleration.",
                "Action plan:",
                "- Now: size positions smaller until two-factor confirmation appears (dollar + real yields).",
                "- 1-4 weeks: reassess after next inflation and policy communication cycle.",
                "Key risks:",
                f"- {risks[0]}",
                f"- {risks[1]}",
                "What to watch:",
                f"- {watch[0]}",
                f"- {watch[1]}",
                "Confidence: MEDIUM - driver map is clear, but attribution remains moderate without cleaner corroboration.",
            ]
            return "\n".join(lines)

        lines = [
            "Direct answer: Silver move is most likely from a combination of dollar direction, real-rate repricing, and flow/positioning shock.",
            f"Data snapshot: {snapshot}",
            "Why it matters now:",
            f"- Dollar move -> silver USD pricing, EM demand impulse -> {why_1} {c1}".strip(),
            f"- Growth-demand narrative -> industrial silver demand expectations -> {why_2} {c2}".strip(),
            f"- Real-rate repricing and flows -> holding-cost and momentum effects -> {why_3} {c3}".strip(),
            "Predicted events:",
            "- Event 1 (7-30d, 45%): Silver mean-reverts if real-rate pressure persists; invalidation: sharp disinflation signal.",
            "- Event 2 (7-30d, 30%): Silver extends rally if dollar softens and industrial-demand headlines improve; invalidation: growth downgrade cycle.",
            "What to do:",
            "- Now: check if dollar and real yields confirm the move before adding risk.",
            "- Next: reassess after next inflation print and policy guidance.",
            "Key risk:",
            f"- {risks[0]}",
            f"Confidence: MEDIUM - {confidence_reason}.",
        ]
        return "\n".join(lines)

    base = generate_unified_fallback(
        question,
        regime,
        cross_asset,
        reasoning_analysis=reasoning_analysis,
        response_mode=response_mode,
    )
    if not context_chunks:
        return base
    # When news context is available, inject actual headlines and key text into the fallback.
    headlines: list[str] = []
    snippets: list[str] = []
    for i, chunk in enumerate(context_chunks[:4], start=1):
        md = chunk.get("metadata", {})
        title = md.get("title", "")
        text = " ".join((chunk.get("text") or "").split())[:200]
        if title:
            headlines.append(f"[S{i}] {title}")
        if text:
            snippets.append(f"[S{i}] {text}")
    news_block = "\n".join(headlines[:3]) if headlines else ""
    snippet_block = "\n".join(snippets[:2]) if snippets else ""
    signal = cross_asset.get("overall_signal", "NEUTRAL")
    regime_name = regime.get("regime", "TRANSITIONAL")
    regime_conf = regime.get("confidence", "LOW")

    dxy = indicators.get("dxy")
    y10 = indicators.get("yield_10y")
    y2 = indicators.get("yield_2y")
    cpi = indicators.get("inflation_cpi")
    vix = indicators.get("vix")
    sp500 = indicators.get("sp500")
    gold = indicators.get("gold")
    oil = indicators.get("oil_wti") or indicators.get("oil_brent")
    credit_hy = indicators.get("credit_hy")

    nums = []
    if sp500 is not None:
        nums.append(f"S&P500={sp500}")
    if vix is not None:
        nums.append(f"VIX={vix}")
    if y10 is not None:
        nums.append(f"10Y={y10}%")
    if y2 is not None:
        nums.append(f"2Y={y2}%")
    if cpi is not None:
        nums.append(f"CPI={cpi}%")
    if dxy is not None:
        nums.append(f"DXY={dxy}")
    if gold is not None:
        nums.append(f"Gold=${gold}")
    if oil is not None:
        nums.append(f"Oil=${oil}")
    if credit_hy is not None:
        nums.append(f"HY_spread={credit_hy}bps")
    snapshot = ", ".join(nums) if nums else "Limited live data."

    # Determine equity/volatility sentiment from available data
    vix_note = ""
    if isinstance(vix, (int, float)):
        if vix >= 25:
            vix_note = f"VIX at {vix} indicates elevated fear — institutional hedging activity elevated."
        elif vix >= 18:
            vix_note = f"VIX at {vix} signals moderate caution — not panic, but not complacent."
        else:
            vix_note = f"VIX at {vix} signals low fear — risk-appetite is firm."

    # Bond mechanics note
    bond_note = ""
    if isinstance(y10, (int, float)) and isinstance(y2, (int, float)):
        curve = round((y10 - y2) * 100, 1)
        if curve < 0:
            bond_note = f"Yield curve inverted ({curve}bps): 2Y={y2}%, 10Y={y10}% — historical recession signal within 12-18 months."
        elif curve < 50:
            bond_note = f"Yield curve flat ({curve}bps): reflects rate uncertainty and growth concerns."
        else:
            bond_note = f"Yield curve positive ({curve}bps): {y2}% 2Y vs {y10}% 10Y."

    if (response_mode or "brief").lower() == "detailed":
        if not pred_lines:
            pred_lines = [
                f"- Event 1 (7-30d, 40%): {regime_name} regime persistence keeps markets range-bound unless CPI surprises; invalidation: large policy shock.",
                "- Event 2 (7-30d, 35%): If yields stay elevated, duration-heavy equities underperform; invalidation: sharp long-yield decline.",
                "- Event 3 (30-90d, 25%): Credit stress spillover broadens risk-off move if spreads widen; invalidation: sustained spread compression.",
            ]
        if not scenario_lines:
            scenario_lines = [
                f"- Base (~55%): {regime_name} regime persists; data in-line with consensus → range-bound markets.",
                "- Bull (~25%): Inflation print below forecast → rate cut narrative strengthens → equities and bonds rally together.",
                "- Bear (~20%): Stagflation signal (hot CPI + weak PMI) → aggressive rate pricing → equities -5-10%, HY spreads widen.",
            ]
        lines = [
            f"Executive summary: Evidence from {len(context_chunks)} retrieved news sources. Regime: {regime_name} ({regime_conf}), cross-asset: {signal}. Key market-moving developments are cited below with available numeric context.",
            f"Direct answer: Navigate a {regime_name} regime with {signal} cross-asset signal — {vix_note or 'monitor VIX and yield curve for regime confirmation.'}",
            f"Data snapshot: {snapshot}",
            f"Causal chain: {regime_name} macro regime → {signal} asset signal → see market impact below for transmission",
            "What is happening:",
        ]
        for i, chunk in enumerate(context_chunks[:3], start=1):
            md = chunk.get("metadata", {})
            title = md.get("title", "")
            text = " ".join((chunk.get("text") or "").split())[:200]
            entry = title if title else text
            if entry:
                lines.append(f"- {entry} [S{i}]")
        if bond_note:
            lines.append(f"- {bond_note}")
        lines += [
            "Market impact:",
            f"- Equities: {signal} signal with VIX={vix or 'N/A'}. In a {regime_name} regime, defensive sectors (utilities, consumer staples) tend to outperform cyclicals.",
            f"- Rates/Bonds: {bond_note or 'Yield direction depends on next inflation/growth data.'}  Safe-haven flows push yields DOWN; rate-hike fears push short-end yields UP.",
            f"- FX: DXY={dxy or 'N/A'}. Dollar strength → EM currency weakness → EM outflows. Watch USD/JPY, EUR/USD for safe-haven signals.",
            f"- Commodities: Oil=${oil or 'N/A'}, Gold=${gold or 'N/A'}. Dollar strength typically pressures commodity prices; geopolitical risk supports oil and gold.",
            "Predicted events:",
            *pred_lines,
            "Scenarios (probabilities must add to 100%):",
            *scenario_lines,
            "Key risks:",
            "- CPI or payrolls surprise outside ±0.2% of consensus triggers rapid repricing.",
            "- Central bank policy divergence (Fed vs ECB vs BoJ) triggers FX dislocations.",
            "Time horizons:",
            "- 24-72h: Monitor scheduled Fed speeches and any PMI/jobless claims data.",
            "- 1-4 weeks: Next CPI/PCE print is the primary catalyst window.",
            "- 1-3 months: If inflation plateaus confirmed, soft-landing narrative can re-rate equities.",
            "What to watch:",
            "- CPI/PCE print vs. consensus (a miss of ±0.2% matters significantly).",
            f"- 10Y yield crossing 4.7% (equity headwind) or 4.2% (support for equities).",
            f"- VIX: above 25 = elevated fear; below 15 = risk-on confirmed.",
            f"Confidence: {regime_conf} - Based on {len(context_chunks)} indexed news chunks; numeric attribution is approximate.",
        ]
        return "\n".join(lines)

    if not pred_lines:
        pred_lines = [
            f"- Event 1 (7-30d, 40%): {regime_name} regime continues with range-bound risk assets; invalidation: major macro surprise.",
            "- Event 2 (7-30d, 35%): Elevated yields sustain pressure on duration assets; invalidation: disinflation-led yield drop.",
        ]
    if not scenario_lines:
        scenario_lines = [
            "- Base (~55%): Data in-line → no major trend break; market consolidates.",
            "- Bull (~25%): Soft inflation print → rate cut expectations drive risk rally.",
            "- Bear (~20%): Hot CPI or weak growth → stagflation fear → equities/credit sell-off.",
        ]

    lines = [
        f"Direct answer: {regime_name} regime, {signal} signal. {vix_note or 'Monitor VIX and yields for direction.'}",
        f"Data snapshot: {snapshot}",
        f"Causal chain: {regime_name} regime → {signal} cross-asset signal → monitor triggers below",
        "What is happening:",
    ]
    for i, chunk in enumerate(context_chunks[:3], start=1):
        md = chunk.get("metadata", {})
        title = md.get("title", "")
        text = " ".join((chunk.get("text") or "").split())[:160]
        entry = title or text
        if entry:
            lines.append(f"- {entry} [S{i}]")
    if bond_note:
        lines.append(f"- {bond_note}")
    lines += [
        "Market impact:",
        f"- Equities: {signal} signal; defensives preferred in {regime_name} unless growth data improves.",
        f"- Rates/Bonds: Safe-haven demand → 10Y yields fall; hawkish shock → 2Y yields spike (curve flattens).",
        f"- FX: DXY={dxy or 'N/A'} — dollar strength pressures EM currencies and commodity prices.",
        "Predicted events:",
        *pred_lines,
        "Scenarios (probabilities must add to 100%):",
        *scenario_lines,
        "What to watch:",
        "- Next CPI/PCE print versus consensus (±0.2% matters).",
        "- 10Y yield and VIX levels as regime confirmation.",
        f"Confidence: {regime_conf} - {len(context_chunks)} indexed chunks used; framework-driven reasoning.",
    ]
    return "\n".join(lines)
