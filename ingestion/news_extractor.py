import requests
from bs4 import BeautifulSoup
from readability import Document
from datetime import datetime
from fake_useragent import UserAgent
from ingestion.utils import clean_text

ua = UserAgent()

def extract_news(url: str) -> dict:
    headers = {
        "User-Agent": ua.random,
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    # Readability to extract main content
    doc = Document(response.text)
    html = doc.summary()

    soup = BeautifulSoup(html, "lxml")

    paragraphs = [p.get_text() for p in soup.find_all("p")]
    text = "\n".join(paragraphs)

    if len(text) < 500:
        raise ValueError("Extracted text too short")

    return {
        "url": url,
        "title": clean_text(doc.title()),
        "raw_text": clean_text(text),
        "source": url.split("/")[2],
        "published_date": None,
        "extracted_at": datetime.utcnow().isoformat()
    }
