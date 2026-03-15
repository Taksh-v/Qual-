"""
rag/ingest.py
--------------
Build or update the FAISS vector index from pre-chunked data files.

Paths (all relative to project root):
  metadata input  : data/vector_db/metadata.json
  FAISS index out : data/vector_db/news.index
  metadata out    : data/vector_db/metadata.json
"""

import json
import os
import sys

import faiss
import numpy as np
from tqdm import tqdm

# Ensure project root is on sys.path so ingestion.* can be imported
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from ingestion.chunker import chunk_text          # shared, sentence-aware chunker
from ingestion.embeddings import get_embedding

# ---------------------------------------------------------------------------
# Paths (relative to project root, not hardcoded absolute)
# ---------------------------------------------------------------------------
_METADATA_IN  = os.path.join(_ROOT, "data", "vector_db", "metadata.json")
_INDEX_OUT    = os.path.join(_ROOT, "data", "vector_db", "news.index")
_METADATA_OUT = os.path.join(_ROOT, "data", "vector_db", "metadata.json")

# Standardised chunk settings (matches run_chunking.py and run_rss_ingest.py)
CHUNK_SIZE = 1500
OVERLAP    = 200


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Embed a list of texts using the shared embedding layer (passage role)."""
    return [get_embedding(t, role="passage") for t in texts]


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------
if not os.path.exists(_METADATA_IN):
    print(f"⚠️  Metadata file not found: {_METADATA_IN}")
    sys.exit(0)

with open(_METADATA_IN, "r", encoding="utf-8") as f:
    news = json.load(f)

texts: list[str] = []
metadata: list[dict] = []

for item in news:
    body = item.get("content") or item.get("structured_text") or item.get("text") or ""
    if not body:
        continue
    # Use the shared sentence-aware chunker (standardised 1500-char / 200-overlap)
    chunks = chunk_text(body, chunk_size=CHUNK_SIZE, overlap=OVERLAP, with_metadata=False)
    for chunk in chunks:
        texts.append(chunk)
        metadata.append({
            "source": item.get("source", ""),
            "date":   item.get("date", ""),
            "title":  item.get("title", ""),
        })

embeddings: list[np.ndarray] = []
batch_size = 20
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
    batch = texts[i : i + batch_size]
    embeddings.extend(embed_batch(batch))

if not embeddings:
    print("⚠️  No embeddings generated — nothing to index.")
    sys.exit(0)

emb_matrix = np.array(embeddings, dtype="float32")
index = faiss.IndexFlatIP(emb_matrix.shape[1])
index.add(emb_matrix)

os.makedirs(os.path.dirname(_INDEX_OUT), exist_ok=True)
faiss.write_index(index, _INDEX_OUT)

with open(_METADATA_OUT, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ RAG ingestion complete  — vectors={index.ntotal}")
print(f"   Index    → {_INDEX_OUT}")
print(f"   Metadata → {_METADATA_OUT}")

