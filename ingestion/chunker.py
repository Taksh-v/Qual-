import uuid
import hashlib
import re
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize
from typing import Any

try:
    from ingestion.ner_extractor import extract_entities, flat_entity_list
except ImportError:
    from ner_extractor import extract_entities, flat_entity_list

# ---------------------------------------------------------------------------
# Sector / region keyword lookup tables (Blueprint §5 metadata completeness)
# ---------------------------------------------------------------------------

_SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Technology": [
        "semiconductor", "chip", "software", "cloud", "ai", "artificial intelligence",
        "machine learning", "data center", "nvidia", "amd", "intel", "microsoft",
        "apple", "google", "alphabet", "meta", "amazon", "tech", "cyber", "saas",
        "platform", "app store", "smartphone", "5g", "broadband", "quantum",
    ],
    "Energy": [
        "oil", "gas", "petroleum", "crude", "brent", "wti", "opec", "refinery",
        "natural gas", "lng", "pipeline", "energy transition", "shale", "offshore",
        "exxon", "chevron", "bp", "shell", "total", "fossil fuel", "solar", "wind",
        "renewable", "clean energy", "carbon", "emissions",
    ],
    "Financials": [
        "bank", "banking", "interest rate", "fed", "federal reserve", "ecb",
        "central bank", "monetary policy", "credit", "loan", "mortgage", "bond",
        "treasury", "yield", "hedge fund", "private equity", "insurance", "fintech",
        "jpmorgan", "goldman sachs", "blackrock", "visa", "mastercard", "payment",
        "liquidity", "capital markets", "ipo", "debt", "equity",
    ],
    "Healthcare": [
        "pharma", "pharmaceutical", "drug", "fda", "clinical trial", "vaccine",
        "biotech", "medical device", "healthcare", "hospital", "pfizer", "moderna",
        "johnson", "merck", "abbvie", "cancer", "treatment", "therapy", "approval",
        "medicare", "medicaid", "insurance coverage",
    ],
    "Consumer": [
        "retail", "consumer spending", "walmart", "target", "costco",
        "e-commerce", "food", "beverage", "restaurant", "travel",
        "hotel", "airline", "luxury", "fashion", "apparel", "nike", "adidas",
        "consumer confidence", "household", "spending", "discretionary",
    ],
    "Industrials": [
        "manufacturing", "factory", "supply chain", "logistics", "freight",
        "aerospace", "defense", "boeing", "lockheed", "caterpillar", "deere",
        "infrastructure", "construction", "rail", "shipping", "automation",
        "pmi", "industrial production",
    ],
    "Materials": [
        "copper", "aluminum", "steel", "iron ore", "mining", "gold", "silver",
        "lithium", "cobalt", "rare earth", "metals", "commodity", "chemicals",
        "fertilizer", "agriculture", "wheat", "corn", "soybean",
    ],
    "Real Estate": [
        "real estate", "reit", "housing", "mortgage", "property", "commercial property",
        "office", "residential", "vacancy", "rent", "landlord", "home price",
    ],
    "Utilities": [
        "utility", "utilities", "electricity", "grid", "power plant", "water",
        "natural monopoly", "regulated",
    ],
}

_REGION_KEYWORDS: dict[str, list[str]] = {
    "US": [
        "united states", "u.s.", "us ", "america", "american", "federal reserve",
        "fed ", "wall street", "nasdaq", "s&p", "s&p 500", "dow jones", "new york",
        "washington", "treasury", "sec ", "white house", "congress", "senate",
    ],
    "Europe": [
        "europe", "european", "eu ", "eurozone", "ecb", "germany", "france",
        "italy", "spain", "uk", "britain", "united kingdom", "london", "paris",
        "berlin", "brussels", "boe", "bank of england", "dax", "cac", "ftse",
        "euro ", "sterling", "pound",
    ],
    "Asia": [
        "china", "chinese", "japan", "japanese", "south korea", "taiwan",
        "hong kong", "singapore", "india", "indonesia", "pboc", "boj",
        "bank of japan", "nikkei", "hang seng", "csi ", "msci asia", "yuan",
        "renminbi", "yen", "won", "rupee", "asean",
    ],
    "Emerging Markets": [
        "emerging market", "brazil", "mexico", "russia", "turkey", "south africa",
        "nigeria", "saudi arabia", "uae", "middle east", "latin america",
        "southeast asia", "brics", "em ", "developing",
    ],
    "Global": [
        "global", "worldwide", "international", "imf", "world bank", "g7", "g20",
        "bis ", "wto", "cross-border",
    ],
}


# Priority order for tie-breaking: earlier = higher priority.
# Technology is listed first because major tech companies (apple, amazon, google)
# also appear in Consumer/other lists and should resolve to Technology.
_SECTOR_PRIORITY: list[str] = [
    "Technology",
    "Financials",
    "Energy",
    "Healthcare",
    "Consumer",
    "Industrials",
    "Materials",
    "Real Estate",
    "Utilities",
]


def _infer_sector(text: str) -> str:
    """Return the best-matching sector tag from text, or 'Unknown'.

    Tie-breaking policy: when two sectors score equally, the one with higher
    priority in _SECTOR_PRIORITY wins (Technology > Financials > Energy …).
    """
    lowered = text.lower()
    best_sector = "Unknown"
    best_count = 0
    best_priority = len(_SECTOR_PRIORITY)  # lower index = higher priority

    for sector, keywords in _SECTOR_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in lowered)
        priority = _SECTOR_PRIORITY.index(sector) if sector in _SECTOR_PRIORITY else len(_SECTOR_PRIORITY)
        # Prefer higher count; break ties by priority (lower index wins)
        if count > best_count or (count == best_count and count > 0 and priority < best_priority):
            best_count = count
            best_sector = sector
            best_priority = priority

    return best_sector


def _infer_region(text: str) -> str:
    """Return the best-matching region tag from text, or 'Global'."""
    lowered = text.lower()
    best_region = "Global"
    best_count = 0
    for region, keywords in _REGION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in lowered)
        if count > best_count:
            best_count = count
            best_region = region
    return best_region


def _infer_data_type(doc_type: str) -> str | None:
    """Infer a normalized data_type from a doc_type label."""
    if not doc_type:
        return None
    lowered = doc_type.strip().lower()
    if "8-k" in lowered or "sec" in lowered or "filing" in lowered:
        return "sec"
    if "earnings" in lowered or "transcript" in lowered:
        return "earnings_transcript"
    if "macro" in lowered or "economic" in lowered:
        return "macro_commentary"
    if "research" in lowered or "report" in lowered:
        return "research_report"
    if "news" in lowered or "article" in lowered:
        return "news"
    return lowered.replace(" ", "_")


def _normalize_metadata_fields(md: dict[str, Any], sector: str, region: str) -> dict[str, Any]:
    out = dict(md) if md else {}

    if not out.get("source"):
        out["source"] = "Unknown"
    if not out.get("date"):
        out["date"] = out.get("published_at") or out.get("extracted_at") or ""
    if not out.get("company"):
        out["company"] = "Unknown"
    if not out.get("sector"):
        out["sector"] = sector
    if not out.get("region"):
        out["region"] = region

    if not out.get("data_type"):
        inferred = _infer_data_type(out.get("doc_type", ""))
        if inferred:
            out["data_type"] = inferred

    if not out.get("company_sector"):
        company = out.get("company") or ""
        if company and company != "Unknown":
            out["company_sector"] = f"{company} - {out.get('sector') or 'Unknown'}"

    return out


def _content_fingerprint(text: str) -> str:
    """
    Compute a stable fingerprint for deduplication.
    Normalises whitespace + case before hashing so minor formatting
    differences between identical articles don't create duplicates.
    """
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def chunk_text(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 250,
    with_metadata: bool = False,
    extra_metadata: dict[str, Any] | None = None,
):
    """Split *text* into overlapping chunks.

    Args:
        text:           Body text to chunk.
        chunk_size:     Target character length per chunk.
        overlap:        Approximate character overlap between adjacent chunks
                        (implemented via trailing sentence carry-over).
        with_metadata:  When True, return a list of dicts containing the chunk
                        text, a unique chunk_id, content fingerprint, and
                        auto-inferred ``sector`` / ``region`` metadata tags
                        (Blueprint §5).  When False (default), return a plain
                        ``list[str]`` for backward compatibility.
        extra_metadata: Optional caller-supplied key/value pairs merged into
                        every chunk metadata dict (only used when
                        ``with_metadata=True``).  Typical fields: ``source``,
                        ``date``, ``company``, ``url``.

    Returns:
        ``list[str]`` when *with_metadata* is False; ``list[dict]`` otherwise.
    """
    sentences = sent_tokenize(text)
    raw_chunks: list[str] = []
    seen_fingerprints: set[str] = set()  # dedup within same article

    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunk_body = " ".join(current_chunk)
            fp = _content_fingerprint(chunk_body)
            if fp not in seen_fingerprints:
                seen_fingerprints.add(fp)
                raw_chunks.append(chunk_body)

            # overlap
            overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
            current_chunk = []
            current_chunk.extend(overlap_sentences)
            current_chunk.append(sentence)
            current_length = sum(len(s) for s in current_chunk)

    if current_chunk:
        chunk_body = " ".join(current_chunk)
        fp = _content_fingerprint(chunk_body)
        if fp not in seen_fingerprints:
            raw_chunks.append(chunk_body)

    if not with_metadata:
        return raw_chunks

    # --- Enrich each chunk with metadata (Blueprint §5) ---
    base_meta: dict[str, Any] = dict(extra_metadata) if extra_metadata else {}
    result: list[dict[str, Any]] = []
    for chunk_body in raw_chunks:
        # NER: extract financial entities from this chunk
        entities = extract_entities(chunk_body)
        entity_list = flat_entity_list(entities)      # flat list for easy storage/search

        entry: dict[str, Any] = {
            "text":        chunk_body,
            "chunk_id":   str(uuid.uuid4()),
            "fingerprint": _content_fingerprint(chunk_body),
            "sector":     _infer_sector(chunk_body),
            "region":     _infer_region(chunk_body),
            "entities":   entities,      # structured: {tickers, companies, indicators, ...}
            "entity_list": entity_list,  # flat deduplicated list for storage
        }
        entry.update(base_meta)  # caller-supplied fields overwrite inferred ones
        md = entry.get("metadata")
        if isinstance(md, dict):
            entry["metadata"] = _normalize_metadata_fields(md, entry.get("sector", "Unknown"), entry.get("region", "Global"))
        else:
            entry["metadata"] = _normalize_metadata_fields({}, entry.get("sector", "Unknown"), entry.get("region", "Global"))
        result.append(entry)
    return result


def get_chunk_fingerprint(text: str) -> str:
    """Public helper — returns the content fingerprint for a chunk text.
    Use this when building metadata entries to enable cross-article dedup.
    """
    return _content_fingerprint(text)
