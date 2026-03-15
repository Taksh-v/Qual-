import re
from typing import Optional

# Minimum cleaned body length (chars) to be considered a usable article
MIN_ARTICLE_CHARS = 100

NOISE_PATTERNS = [
    # Navigation / UI boilerplate
    r"read more[^\n]{0,60}",
    r"click here[^\n]{0,60}",
    r"subscribe\s*(now|today|to (our|the))[^\n]{0,80}",
    r"sign\s*up\s*(for|to)[^\n]{0,80}",
    r"advertisement[^\n]{0,60}",
    r"sponsored\s*(content|post|by)[^\n]{0,60}",
    # Legal / rights boilerplate
    r"all rights reserved[^\n]{0,80}",
    r"copyright\s*©?\s*\d{4}[^\n]{0,120}",
    r"terms\s*(of\s*)?(use|service|conditions)[^\n]{0,80}",
    r"privacy\s*policy[^\n]{0,60}",
    # Financial disclaimers
    r"(this\s*)?(article|report|content)\s+is\s+for\s+informational\s+purposes?\s+only[^\n]{0,200}",
    r"past\s+performance\s+(is\s+)?no[t\s]+guarantee[^\n]{0,200}",
    r"not\s+(financial|investment)\s+advice[^\n]{0,200}",
    # Cookie / GDPR banners
    r"we\s+use\s+cookies[^\n]{0,200}",
    r"by\s+(continuing|using)\s+(to\s+)?use\s+this\s+site[^\n]{0,200}",
    # Paywall / registration prompts
    r"(you('ve| have) used|access) \d+ free articles?[^\n]{0,150}",
    r"(register|log\s*in)\s+to\s+(read|access|continue)[^\n]{0,150}",
]

_COMPILED_NOISE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


def normalize_text(text: str) -> str:
    """Remove boilerplate noise and normalize whitespace. Preserves financial symbols."""
    if not text:
        return ""

    text = text.strip()

    for pattern in _COMPILED_NOISE:
        text = pattern.sub("", text)

    # Collapse 3+ newlines → double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse inline whitespace runs (but keep single newlines)
    text = re.sub(r"[ \t\xa0]+", " ", text)

    return text.strip()


def structure_article(article: dict) -> Optional[dict]:
    """
    Clean and structure a raw article dict.

    Returns None if the cleaned body is too short to be useful (quality gate).
    Body text intentionally does NOT include TITLE/DATE/SOURCE headers —
    those are stored only in `metadata` to avoid polluting embedding vectors.
    """
    raw_body = normalize_text(article.get("raw_text", ""))

    # Quality gate: discard near-empty articles
    if len(raw_body) < MIN_ARTICLE_CHARS:
        return None

    return {
        "url": article.get("url", ""),
        # Plain cleaned body — no injected headers so embeddings reflect content,
        # not boilerplate metadata tokens.
        "structured_text": raw_body,
        "metadata": {
            "title": article.get("title", ""),
            "date": article.get("published_date", ""),
            "source": article.get("source", ""),
            "extracted_at": article.get("extracted_at", ""),
        },
    }

