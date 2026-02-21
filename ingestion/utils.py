import re

def clean_text(text: str) -> str:
    if not text:
        return ""

    # remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # remove non-printable characters
    text = re.sub(r'[^\x20-\x7E]', '', text)

    return text.strip()
