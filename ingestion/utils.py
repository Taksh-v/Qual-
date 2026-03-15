import re
import unicodedata

# Control-character ranges to strip (C0 except tab/newline, C1, private-use area)
# This intentionally PRESERVES: € ₹ £ ¥ % – — • ' " and all printable Unicode
_CTRL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"   # C0 controls (keep \t=\x09, \n=\x0a)
    r"\x80-\x9f"                            # C1 controls
    r"\ufffe\uffff\ufdd0-\ufddf"            # non-characters
    r"\U000e0000-\U000effff]"               # tags / supplementary private-use
)


def clean_text(text: str) -> str:
    """Clean text while preserving financial symbols (€ ₹ £ ¥ – — % etc.)."""
    if not text:
        return ""

    # Strip invisible control characters only — NOT printable Unicode
    text = _CTRL_RE.sub("", text)

    # Normalize unicode to composed form (e.g. é rather than e + combining accent)
    text = unicodedata.normalize("NFC", text)

    # Collapse excessive whitespace (including Unicode whitespace variants)
    text = re.sub(r"[ \t\xa0\u2009\u200b]+", " ", text)   # space-like chars
    text = re.sub(r"\n{3,}", "\n\n", text)                  # triple+ newlines

    return text.strip()

