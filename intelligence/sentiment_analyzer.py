"""
sentiment_analyzer.py
----------------------
Lightweight finance-domain sentiment analysis using a tiered keyword lexicon.
Returns a structured sentiment dict: {"label": "positive|negative|neutral", "score": float, "signals": list}

No external ML model required — uses a curated financial lexicon with intensity weights.
Falls back to a neutral stance on parse failure.

Usage:
    from intelligence.sentiment_analyzer import score_sentiment, batch_score_sentiment
    result = score_sentiment("Fed signals rate cuts ahead, markets rally strongly")
    # -> {"label": "positive", "score": 0.72, "magnitude": "strong", "signals": ["rally", "cuts"]}
"""

from __future__ import annotations

import re
from typing import Any

# ─── Finance-domain lexicon ────────────────────────────────────────────────────
# Each tuple: (keyword/phrase, weight)
# Positive weights → bullish / good news;  Negative weights → bearish / bad news
# Magnitude |weight|: 0.3–0.5 = mild, 0.6–0.8 = moderate, 0.9–1.2 = strong

_BULLISH: list[tuple[str, float]] = [
    # Macro / rates
    ("rate cut", 0.9),
    ("rate cuts", 0.9),
    ("dovish", 0.8),
    ("stimulus", 0.7),
    ("quantitative easing", 0.8),
    ("accommodative", 0.6),
    ("soft landing", 0.9),
    ("disinflation", 0.7),
    ("falling inflation", 0.8),
    ("inflation easing", 0.8),
    # Activity
    ("strong gdp", 0.9),
    ("gdp growth", 0.7),
    ("record earnings", 1.0),
    ("beat expectations", 0.8),
    ("better than expected", 0.8),
    ("revenue growth", 0.7),
    ("profit surge", 0.9),
    ("margin expansion", 0.8),
    ("raised guidance", 0.9),
    ("buyback", 0.6),
    ("dividend increase", 0.7),
    ("hiring surge", 0.7),
    ("job gains", 0.6),
    ("unemployment falls", 0.7),
    # Markets
    ("rally", 0.8),
    ("surge", 0.9),
    ("breakout", 0.8),
    ("all-time high", 1.0),
    ("bull market", 0.9),
    ("risk-on", 0.7),
    ("recovery", 0.6),
    ("rebound", 0.7),
    ("outperform", 0.7),
    ("upgrade", 0.7),
    ("buy", 0.5),
    ("strong buy", 0.9),
    ("overweight", 0.6),
    ("positive outlook", 0.8),
    ("positive momentum", 0.8),
    # Deals
    ("acquisition", 0.5),
    ("merger", 0.4),
    ("strategic partnership", 0.5),
    ("ipo", 0.4),
    ("expansion", 0.6),
]

_BEARISH: list[tuple[str, float]] = [
    # Macro / rates
    ("rate hike", -0.9),
    ("rate hikes", -0.9),
    ("hawkish", -0.8),
    ("tightening", -0.7),
    ("stagflation", -1.0),
    ("recession", -1.0),
    ("contraction", -0.8),
    ("deflation", -0.7),
    ("inflation surge", -0.9),
    ("rising inflation", -0.8),
    ("hyperinflation", -1.1),
    # Activity
    ("missed expectations", -0.8),
    ("profit warning", -0.9),
    ("revenue miss", -0.8),
    ("earnings decline", -0.8),
    ("margin compression", -0.8),
    ("lowered guidance", -0.9),
    ("layoffs", -0.7),
    ("job cuts", -0.7),
    ("unemployment rises", -0.7),
    ("default", -1.0),
    ("bankruptcy", -1.1),
    ("insolvency", -1.1),
    # Markets
    ("crash", -1.1),
    ("plunge", -1.0),
    ("selloff", -0.9),
    ("sell-off", -0.9),
    ("bear market", -1.0),
    ("risk-off", -0.7),
    ("correction", -0.7),
    ("downgrade", -0.8),
    ("sell", -0.5),
    ("underweight", -0.6),
    ("negative outlook", -0.8),
    ("deteriorating", -0.7),
    ("declining", -0.6),
    ("weak", -0.5),
    ("losses", -0.6),
    ("writedown", -0.8),
    ("write-off", -0.8),
    # Geopolitical
    ("sanctions", -0.7),
    ("trade war", -0.8),
    ("tariffs", -0.6),
    ("geopolitical tensions", -0.7),
    ("conflict", -0.6),
    ("supply disruption", -0.8),
    ("supply chain crisis", -0.9),
    ("debt crisis", -1.0),
    ("credit crunch", -0.9),
]

# Build sorted list (longest phrases first to avoid partial matches)
_ALL_TERMS: list[tuple[str, float]] = sorted(
    _BULLISH + _BEARISH,
    key=lambda x: len(x[0]),
    reverse=True,
)


def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _magnitude_label(score: float) -> str:
    a = abs(score)
    if a >= 0.8:
        return "strong"
    if a >= 0.5:
        return "moderate"
    if a >= 0.2:
        return "mild"
    return "neutral"


def score_sentiment(text: str) -> dict[str, Any]:
    """
    Analyse the financial sentiment of *text*.

    Returns:
        {
            "label":      "positive" | "negative" | "neutral",
            "score":      float in [-1.0, +1.0],
            "magnitude":  "strong" | "moderate" | "mild" | "neutral",
            "signals":    list[str]  # matched keywords
        }
    """
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0, "magnitude": "neutral", "signals": []}

    normalised = _normalise(text[:2000])  # cap to save compute

    total_weight = 0.0
    matched: list[str] = []
    remaining = normalised

    for phrase, weight in _ALL_TERMS:
        if phrase in remaining:
            total_weight += weight
            matched.append(phrase)
            # Blank out matched phrase so it doesn't double-count substrings
            remaining = remaining.replace(phrase, " ", 1)

    # Normalise to [-1, +1] using a soft sigmoid-style clamp
    if matched:
        # average the weights to avoid length bias, then scale
        avg_weight = total_weight / len(matched)
        # clamp to [-1.0, +1.0]
        score = max(-1.0, min(1.0, avg_weight))
    else:
        score = 0.0

    if score > 0.10:
        label = "positive"
    elif score < -0.10:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label":     label,
        "score":     round(score, 4),
        "magnitude": _magnitude_label(score),
        "signals":   matched[:10],  # top-10 contributing keywords
    }


def batch_score_sentiment(texts: list[str]) -> list[dict[str, Any]]:
    """Score a list of texts. Returns results in the same order."""
    return [score_sentiment(t) for t in texts]


def sentiment_summary(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate sentiment across a list of RAG chunks.

    Each chunk must have a 'text' key.
    Returns:
        {
            "overall_label":     "positive" | "negative" | "neutral",
            "avg_score":         float,
            "positive_count":    int,
            "negative_count":    int,
            "neutral_count":     int,
            "top_bullish":       list[str],
            "top_bearish":       list[str],
        }
    """
    if not chunks:
        return {
            "overall_label": "neutral",
            "avg_score": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "top_bullish": [],
            "top_bearish": [],
        }

    results = [score_sentiment(c.get("text", "")) for c in chunks]
    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores)

    pos = sum(1 for r in results if r["label"] == "positive")
    neg = sum(1 for r in results if r["label"] == "negative")
    neu = sum(1 for r in results if r["label"] == "neutral")

    # Collect unique signals
    bullish_signals: list[str] = []
    bearish_signals: list[str] = []
    for r in results:
        for sig in r["signals"]:
            weight = next((w for p, w in _BULLISH if p == sig), None)
            if weight and weight > 0 and sig not in bullish_signals:
                bullish_signals.append(sig)
            weight2 = next((w for p, w in _BEARISH if p == sig), None)
            if weight2 and weight2 < 0 and sig not in bearish_signals:
                bearish_signals.append(sig)

    if avg > 0.10:
        overall = "positive"
    elif avg < -0.10:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "overall_label":  overall,
        "avg_score":      round(avg, 4),
        "positive_count": pos,
        "negative_count": neg,
        "neutral_count":  neu,
        "top_bullish":    bullish_signals[:5],
        "top_bearish":    bearish_signals[:5],
    }
