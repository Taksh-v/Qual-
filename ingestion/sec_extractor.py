import requests
import feedparser
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Any
from datetime import datetime, timezone
import json
import os
import logging

logger = logging.getLogger(__name__)


def _fetch_with_retry(
    url: str,
    headers: dict,
    retries: int = 3,
    backoff: float = 1.5,
    timeout: float = 15.0,
) -> requests.Response:
    """
    GET *url* with exponential-backoff retry on network/5xx errors.
    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            wait = backoff ** attempt
            logger.warning(
                "[SEC] Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                attempt + 1, retries, url, exc, wait,
            )
            time.sleep(wait)
    raise last_exc


class SECExtractor:
    """
    Extracts 8-K filings from the SEC EDGAR RSS feed.
    """
    def __init__(self, user_agent: str = "Qual-Intelligence-System qual@example.com"):
        # SEC requires a descriptive user agent
        self.headers = {"User-Agent": user_agent}
        self.rss_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-k&company=&dateb=&owner=include&start=0&count=40&output=atom"
        
    def fetch_latest_8k(self) -> List[Dict[str, Any]]:
        """
        Fetches the latest 8-K filings from EDGAR.
        """
        response = _fetch_with_retry(self.rss_url, headers=self.headers)
        feed = feedparser.parse(response.content)
        extracted = []
        
        for entry in feed.entries:
            # We want the link to the actual filing
            try:
                title_parts = entry.title.split(' - ', 1)
                form_type = title_parts[0].strip()
                company_raw = title_parts[1].strip() if len(title_parts) > 1 else "Unknown"
                company_name = company_raw.split(' (')[0].strip()
                
                filing_info = {
                    "source": "SEC EDGAR",
                    "doc_type": form_type,
                    "company": company_name,
                    "title": entry.title,
                    "url": entry.link,
                    "published_at": entry.updated,
                    "summary": entry.summary if hasattr(entry, 'summary') else "",
                    "extracted_at": datetime.now(timezone.utc).isoformat()
                }
                
                # Fetch the raw text of the filing
                # EDGAR links point to an index page; we need to find the primary document
                # For this MVP, we scrape the index page, find the first matching .htm file
                document_url = self._get_primary_document_url(entry.link)
                if document_url:
                    text_content = self._extract_filing_text(document_url)
                    filing_info["raw_text"] = text_content
                    extracted.append(filing_info)
                
                # Be polite to the SEC API (0.3s between requests)
                time.sleep(0.3)

                
            except Exception as e:
                print(f"[SEC Extractor] Error parsing entry {entry.title}: {e}")
                
        return extracted
        
    def _get_primary_document_url(self, index_url: str) -> str:
        """
        Scrapes the SEC index page to find the actual .htm or .txt filing document.
        """
        try:
            response = _fetch_with_retry(index_url, headers=self.headers)
        except Exception as exc:
            logger.warning("[SEC] Could not fetch index page %s: %s", index_url, exc)
            return ""
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # The primary document is usually the first row in the document format files table
        table = soup.find('table', summary='Document Format Files')
        if not table:
            return ""
            
        rows = table.find_all('tr')
        if len(rows) > 1: # Row 0 is header
            cells = rows[1].find_all('td')
            if len(cells) >= 3:
                link = cells[2].find('a')
                if link and link.has_attr('href'):
                    # SEC links are relative
                    return f"https://www.sec.gov{link['href']}"
                    
        return ""
        
    def _extract_filing_text(self, document_url: str) -> str:
        """
        Extracts the text content from an SEC .htm filing.
        """
        try:
            response = _fetch_with_retry(document_url, headers=self.headers)
        except Exception as exc:
            logger.warning("[SEC] Could not fetch filing %s: %s", document_url, exc)
            return ""
        soup = BeautifulSoup(response.content, 'html.parser')
        # Simple extraction for now: remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator='\n')
        # Clean up excessive whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

if __name__ == "__main__":
    extractor = SECExtractor()
    filings = extractor.fetch_latest_8k()
    print(f"Extracted {len(filings)} filings.")
    if filings:
        print(json.dumps(filings[0], indent=2)[:500] + "\n...")

