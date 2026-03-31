from __future__ import annotations

import re

MAX_BULLET_LEN = 220
MAX_PARAGRAPH_LEN = 480

BANNED_VAGUE_PHRASES = {
    "heightened uncertainty",
    "downward pressure",
    "upward pressure",
    "various factors",
    "further monitoring required",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def enforce_sentence_case(text: str) -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return cleaned
    return cleaned[0].upper() + cleaned[1:]


def trim_bullet(text: str, max_len: int = MAX_BULLET_LEN) -> str:
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def contains_vague_phrase(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in BANNED_VAGUE_PHRASES)


def normalize_bullet(text: str) -> str:
    return trim_bullet(enforce_sentence_case(text))


def normalize_section_lines(lines: list[str]) -> list[str]:
    return [normalize_bullet(line) for line in lines if normalize_whitespace(line)]
