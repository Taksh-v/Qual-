from __future__ import annotations

import json
import re
from typing import Any

from intelligence.market_context import (
    build_full_market_context,
    build_compact_market_context,
)
from intelligence.prompt_templates import (
    get_response_format_block,
    FINANCIAL_MECHANICS_BLOCK,
    COT_REASONING_BLOCK,
    STRICT_RULES_BLOCK,
    build_quality_rewrite_prompt as _build_quality_rewrite_prompt,
    build_citation_repair_prompt as _build_citation_repair_prompt,
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

    format_block = get_response_format_block(mode)

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

    return f"""You are a senior macro strategist at a tier-1 investment bank. Your output is read by institutional portfolio managers who need real, numbered, actionable intelligence — not commentary.
{FINANCIAL_MECHANICS_BLOCK}
{COT_REASONING_BLOCK}
{STRICT_RULES_BLOCK}
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
    return _build_quality_rewrite_prompt(section_name, draft_text)


def build_citation_repair_prompt(draft_text: str, formatted_context: str) -> str:
    return _build_citation_repair_prompt(draft_text, formatted_context)


from intelligence.response_builder import BriefResponseBuilder, DetailedResponseBuilder


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
    
    builder = DetailedResponseBuilder() if mode == "detailed" else BriefResponseBuilder()
    
    if mode == "detailed":
        builder.executive_summary("Live market data feed is unavailable. Current regime signals a transitional environment. The primary risk drivers are policy rate trajectory and growth-inflation balance.")
        builder.direct_answer("Hold balanced positioning — no high-conviction directional call is warranted without live data confirmation.")
        builder.data_snapshot("Live data unavailable — indicators below are from last known regime state.")
        builder.causal_chain(f"Uncertain macro regime → mixed growth/inflation signals → elevated cross-asset correlation → reduced diversification benefit")
        
        builder.happening([
            f"Macro regime is {regime_name} with cross-asset signal: {signal}.",
            "Central bank policy path remains data-dependent; rate expectations are volatile.",
            "Growth and inflation trends are not yet confirmed in either direction."
        ])
        
        builder.market_impact([
            "Equities: Risk of multiple compression if rates stay elevated above 4.5%; defensives outperform cyclicals.",
            "Rates/Bonds: Safe-haven demand would push 10Y yields lower; hawkish surprise pushes 2Y higher.",
            "FX: Dollar (DXY) typically strengthens in risk-off; EM currencies face outflow pressure.",
            "Commodities: Oil sensitive to growth outlook; gold rises on safe-haven flows."
        ])
    else:
        builder.direct_answer(f"Regime is {regime_name} (signal: {signal}) — no high-conviction trade without live data confirmation.")
        builder.data_snapshot("Live data unavailable — limited to regime framework.")
        builder.causal_chain(f"{regime_name} regime → mixed signals → no dominant directional catalyst identified")
        builder.happening([
            "Macro signals are mixed across growth, rates, and risk appetite — no single driver dominates.",
            "Rate sensitivity is elevated; one data surprise can shift market direction quickly.",
            "Cross-asset correlations are elevated, limiting traditional hedges."
        ])
        builder.market_impact([
            "Equities: Vulnerable to multiple compression if rates stay above 4.5%.",
            "Rates/Bonds: Flight to safety = 10Y yields fall; hawkish shock = 2Y yields spike.",
            "FX: Dollar strength in risk-off; EM currencies weaken on capital outflows."
        ])

    # Predicted Events from reasoning analysis or defaults
    scenarios = reasoning_analysis.get("scenario_generator", {}) if reasoning_analysis else {}
    if scenarios:
        order = ("base_case", "bull_case", "bear_case")
        for key in order:
            item = scenarios.get(key, {})
            if isinstance(item, dict) and item.get("narrative"):
                builder.predicted_event(
                    label=key.replace("_", " ").title(),
                    horizon=item.get("horizon", "7-30d"),
                    probability_pct=item.get("probability_pct"),
                    narrative=item.get("narrative"),
                    trigger=item.get("trigger", "N/A"),
                    invalidation=item.get("invalidation", "N/A")
                )
    
    if not builder.build().predicted_events:
        builder.predicted_event("Rate Pressure", "7-30d", 40, "Elevated rates keep pressure on growth equities", "10Y > 4.5%", "10Y < 4.2%")
        builder.predicted_event("Risk-Off", "7-30d", 35, "Risk-off episodes favor USD and defensives", "Credit spread widening", "Spread tightening")

    # Scenarios
    if scenarios:
        for key in ("base_case", "bull_case", "bear_case"):
            item = scenarios.get(key, {})
            if isinstance(item, dict) and item.get("narrative"):
                builder.scenario(key.split("_")[0].title(), item.get("probability_pct"), item.get("narrative"))
    
    if not builder.build().scenarios:
        builder.scenario("Base", 55, "Data in-line → range-bound markets, no trend breakout.")
        builder.scenario("Bull", 25, "Soft inflation data → rate cut pricing → rally.")
        builder.scenario("Bear", 20, "Hot inflation or weak growth → stagflation fear → selloff.")

    if mode == "detailed":
        builder.key_risks([
            "A data surprise (CPI, payrolls) outside ±0.2% of consensus can shift rate expectations rapidly.",
            "Central bank forward guidance divergence between Fed, ECB, BoJ can trigger FX volatility."
        ])
        builder.time_horizons([
            "24-72h: Watch any scheduled Fed speeches or PMI releases.",
            "1-4 weeks: Next CPI/PCE release will determine if rate expectations reprice.",
            "1-3 months: If inflation plateau confirmed, soft-landing narrative strengthens equities."
        ])

    builder.watch([
        "Next CPI / PCE print versus consensus (±0.2% matters).",
        "10Y Treasury yield: above 4.7% signals renewed rate pressure.",
        "VIX above 20 indicates elevated fear; below 15 confirms risk-on stability."
    ])
    builder.confidence(f"LOW - Live data unavailable; framework-only reasoning (Regime: {regime_name}).")
    
    return builder.build().to_text()


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
    mode = (response_mode or "brief").lower()
    
    builder = DetailedResponseBuilder() if mode == "detailed" else BriefResponseBuilder()

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

        why_1 = f"Current dollar context (DXY {dxy}) can materially drive short-term silver moves." if isinstance(dxy, (int, float)) else "Silver often moves with changes in the US dollar and real yields."
        why_2 = "Silver has both precious-metal and industrial-demand behavior, so growth headlines matter."
        why_3 = f"Estimated real-rate proxy ({real_rate}%) can shift opportunity cost for holding silver." if real_rate is not None else "Fast price jumps are usually flow-driven (positioning/liquidity) rather than single-factor."

        data_points = []
        if isinstance(dxy, (int, float)): data_points.append(f"dxy={dxy}")
        if isinstance(y10, (int, float)): data_points.append(f"yield_10y={y10}%")
        if isinstance(infl, (int, float)): data_points.append(f"inflation_cpi={infl}%")
        if real_rate is not None: data_points.append(f"real_rate_proxy={real_rate}%")
        snapshot = ", ".join(data_points) if data_points else "Insufficient custom evidence."

        if mode == "detailed":
            builder.executive_summary(f"Sudden silver moves are typically explained by a joint shift in dollar, real-rate, and positioning channels under a {regime_name} ({regime_conf}) regime.")
        
        builder.direct_answer("Silver move is most likely from a combination of dollar direction, real-rate repricing, and flow/positioning shock.")
        builder.data_snapshot(snapshot)
        builder.causal_chain(f"Real rate/Dollar shift → precious metal opportunity cost change → momentum flow → silver price volatility")
        
        builder.happening([
            f"Dollar move -> silver USD pricing and EM demand sensitivity -> {why_1} {c1}".strip(),
            f"Growth-demand narrative -> industrial silver demand expectations -> {why_2} {c2}".strip(),
            f"Real-rate repricing and flows -> opportunity-cost and momentum effects -> {why_3} {c3}".strip()
        ])
        
        builder.market_impact([
            f"Equities: risk-sensitive sectors may weaken when silver spike is risk-off driven (signal={signal}).",
            "Rates: real-yield shifts can dominate metal pricing in the short run.",
            "FX/Commodities: dollar trend and broader metals complex provide confirmation."
        ])
        
        builder.predicted_event("Mean Reversion", "7-30d", 45, "Silver upside fades if real-rate pressure persists", "Real yields > 2.0%", "Sharp disinflation surprise")
        builder.predicted_event("Rally Extension", "7-30d", 30, "Silver extends rally if dollar softens", "DXY < 102", "DXY re-acceleration")
        
        builder.scenario("Base", 45, "Silver consolidates as drivers neutralize.")
        builder.scenario("Bull", 30, "Dollar weakness accelerates silver gains.")
        builder.scenario("Bear", 25, "Real rate spike triggers sharp correction.")

        if mode == "detailed":
            builder.key_risks([
                "Headline-driven volatility can reverse quickly when liquidity is thin.",
                "If macro data surprises, silver can decouple from recent trend intraday."
            ])
            builder.time_horizons(["24-72h: Monitor DXY.", "1-4 weeks: Watch CPI."])

        builder.watch([
            "US dollar move (DXY) and real-yield direction over the next 24-72 hours.",
            "Gold-silver relative move and any industrial-demand headlines.",
            "Central-bank communication and inflation surprises."
        ])
        builder.confidence(f"MEDIUM - driver map is clear, but attribution remains moderate without cleaner corroboration (Sources used: {len(context_chunks)}).")
        return builder.build().to_text()

    # General Contextual Fallback
    dxy = indicators.get("dxy"); y10 = indicators.get("yield_10y"); vix = indicators.get("vix")
    gold = indicators.get("gold"); oil = indicators.get("oil_wti") or indicators.get("oil_brent")
    
    nums = []
    if dxy is not None: nums.append(f"DXY={dxy}")
    if y10 is not None: nums.append(f"10Y={y10}%")
    if vix is not None: nums.append(f"VIX={vix}")
    if gold is not None: nums.append(f"Gold=${gold}")
    if oil is not None: nums.append(f"Oil=${oil}")
    snapshot = ", ".join(nums) if nums else "Limited live data."

    if mode == "detailed":
        builder.executive_summary(f"Evidence from {len(context_chunks)} retrieved news sources. Regime: {regime_name} ({regime_conf}), signal: {signal}.")

    builder.direct_answer(f"Navigate a {regime_name} regime with {signal} cross-asset signal — monitor VIX and yields for regime confirmation.")
    builder.data_snapshot(snapshot)
    builder.causal_chain(f"{regime_name} macro regime → {signal} asset signal → transmission via rates and sentiment")
    
    happening = []
    for i, chunk in enumerate(context_chunks[:3], start=1):
        title = chunk.get("metadata", {}).get("title", "")
        text = " ".join((chunk.get("text") or "").split())[:160]
        if title or text: happening.append(f"{title or text} [S{i}]")
    builder.happening(happening)
    
    builder.market_impact([
        f"Equities: {signal} signal; defensives preferred in {regime_name} unless growth data improves.",
        "Rates/Bonds: Safe-haven demand → 10Y yields fall; hawkish shock → 2Y yields spike.",
        f"FX/Commodities: DXY={dxy or 'N/A'} — dollar strength typically pressures EM and commodities."
    ])
    
    builder.predicted_event("Regime Persistence", "7-30d", 40, f"{regime_name} regime continues with range-bound assets", "No major macro surprise", "Major policy shock")
    builder.predicted_event("Yield Pressure", "7-30d", 35, "Elevated yields sustain pressure on duration assets", "10Y holds > 4.5%", "Disinflation-led yield drop")
    
    builder.scenario("Base", 55, "Data in-line → no major trend break; market consolidates.")
    builder.scenario("Bull", 25, "Soft inflation print → rate cut expectations drive risk rally.")
    builder.scenario("Bear", 20, "Hot CPI or weak growth → stagflation fear → sell-off.")

    if mode == "detailed":
        builder.key_risks(["CPI or payrolls surprise outside ±0.2% triggers rapid repricing."])
        builder.time_horizons(["24-72h: Monitor Fed speeches.", "1-4 weeks: Next CPI print."])

    builder.watch(["Next CPI/PCE print versus consensus.", "10Y yield and VIX levels as regime confirmation."])
    builder.confidence(f"{regime_conf} - Based on {len(context_chunks)} indexed news chunks; framework-driven reasoning.")
    
    return builder.build().to_text()
