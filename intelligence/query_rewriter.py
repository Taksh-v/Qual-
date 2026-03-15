import os
import re
import requests
from functools import lru_cache

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("QUERY_REWRITE_MODEL", "phi3:mini")
# Hard cap rewrite at 10 s — if the LLM can't rewrite in 10 s, skip it.
_REWRITE_TIMEOUT = float(os.getenv("QUERY_REWRITE_TIMEOUT_SEC", "10"))

# ── Financial term synonym map for deterministic expansion ──────────────────
_TERM_SYNONYMS: dict[str, list[str]] = {
    "fed": ["Federal Reserve", "FOMC", "Federal Open Market Committee"],
    "ecb": ["European Central Bank", "ECB"],
    "boj": ["Bank of Japan", "BOJ"],
    "rate cut": ["interest rate reduction", "monetary easing", "rate decrease"],
    "rate hike": ["interest rate increase", "monetary tightening", "rate rise"],
    "cpi": ["Consumer Price Index", "inflation", "price index"],
    "pce": ["Personal Consumption Expenditures", "core inflation"],
    "gdp": ["gross domestic product", "economic growth", "economic output"],
    "pmi": ["Purchasing Managers Index", "manufacturing activity"],
    "yield curve": ["2Y-10Y spread", "Treasury yield spread", "term spread"],
    "vix": ["VIX", "volatility index", "fear gauge", "implied volatility"],
    "dxy": ["US dollar index", "dollar strength", "DXY"],
    "wti": ["West Texas Intermediate", "crude oil", "oil price"],
    "hy": ["high yield", "junk bonds", "HY credit spreads"],
    "ig": ["investment grade", "IG credit"],
    "em": ["emerging markets", "developing economies"],
    "bps": ["basis points", "bp"],
}


def _deterministic_expand(question: str) -> str:
    """Expand known financial abbreviations for better retrieval hits."""
    q = question
    q_lower = q.lower()
    for abbr, synonyms in _TERM_SYNONYMS.items():
        if abbr in q_lower and synonyms[0].lower() not in q_lower:
            # Append the primary synonym in parentheses
            q = re.sub(
                r"(?<!\w)" + re.escape(abbr) + r"(?!\w)",
                f"{abbr} ({synonyms[0]})",
                q,
                flags=re.IGNORECASE,
                count=1,
            )
    return q


@lru_cache(maxsize=256)
def rewrite_query(question: str) -> str:
    """
    Rewrite the user query for sharper retrieval using a two-stage approach:
      1. LLM-based semantic expansion (via Ollama) — adds financial context terms
      2. Deterministic abbreviation expansion — always applied as fallback

    Result is cached so repeated identical queries pay zero LLM cost.
    Returns the original question if both stages fail or produce low-quality output.
    """
    # Stage 2 is always safe — expand abbreviations first
    expanded = _deterministic_expand(question)

    prompt = (
        "You are a financial NLP assistant. Rewrite the following macro/financial question "
        "to maximise retrieval from a news + economic-report database.\n\n"
        "Rules:\n"
        "1. Expand all acronyms (e.g. 'Fed' → 'Federal Reserve FOMC', 'CPI' → 'Consumer Price Index inflation').\n"
        "2. Add 2-3 closely related financial terms or asset classes the question implies.\n"
        "3. Preserve the original intent and time frame exactly.\n"
        "4. Include geographic context if inferable (e.g. 'US', 'Eurozone', 'EM').\n"
        "5. Output ONLY the rewritten query — no explanation, no prefix, no quotes.\n\n"
        f"Original question: {question}\n\n"
        "Rewritten query:"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 80,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "repeat_penalty": 1.0,
                },
            },
            timeout=_REWRITE_TIMEOUT,
        )
        resp.raise_for_status()
        rewritten = (resp.json().get("response") or "").strip()
        # Sanity checks: must be meaningful and not too long
        if 8 <= len(rewritten) <= 400 and "\n\n" not in rewritten:
            return rewritten
    except Exception:
        pass

    # Fallback: return deterministically-expanded query
    return expanded if expanded != question else question


def extract_search_keywords(question: str) -> list[str]:
    """
    Extract 3-6 high-value search keywords from a financial question.
    Used to augment FAISS similarity search with keyword-based pre-filtering.
    """
    # Financial stop-words that add no retrieval signal
    _STOP = {
        "what", "how", "why", "when", "where", "which", "does", "will",
        "would", "could", "should", "is", "are", "the", "a", "an", "and",
        "or", "but", "in", "on", "at", "for", "to", "of", "do", "has",
        "been", "being", "have", "had", "can", "may", "might", "this",
        "that", "these", "those", "if", "impact", "affect", "effect",
        "market", "markets", "economy", "economic",
    }
    tokens = re.findall(r"[a-zA-Z0-9&%$\.\-/]{2,}", question)
    keywords = [t for t in tokens if t.lower() not in _STOP]
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for kw in keywords:
        low = kw.lower()
        if low not in seen:
            seen.add(low)
            result.append(kw)
    return result[:6]
