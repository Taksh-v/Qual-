import re

NOISE_PATTERNS = [
    r"read more.*",
    r"click here.*",
    r"subscribe.*",
    r"advertisement.*",
    r"all rights reserved.*",
]

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    # remove noise patterns
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def structure_article(article: dict) -> dict:
    structured_text = f"""
TITLE: {article.get('title', '')}
DATE: {article.get('published_date', '')}
SOURCE: {article.get('source', '')}

CONTENT:
{normalize_text(article.get('raw_text', ''))}
""".strip()

    return {
        "url": article["url"],
        "structured_text": structured_text,
        "metadata": {
            "title": article.get("title"),
            "date": article.get("published_date"),
            "source": article.get("source"),
            "extracted_at": article.get("extracted_at")
        }
    }
