import json
import faiss
import numpy as np
import requests
from tqdm import tqdm

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 400
OVERLAP = 80


def chunk_text(text):
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = words[i:i + CHUNK_SIZE]
        chunks.append(" ".join(chunk))
        i += CHUNK_SIZE - OVERLAP

    return chunks


def embed(text):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text}
    )
    return response.json()["embedding"]


with open("/data/vector_db/metadata.json", "r", encoding="utf-8") as f:
    news = json.load(f)

texts = []
metadata = []

for item in news:
    chunks = chunk_text(item["content"])
    for idx, chunk in enumerate(chunks):
        texts.append(chunk)
        metadata.append({
            "source": item["source"],
            "date": item["date"],
            "title": item["title"]
        })

embeddings = []
for text in tqdm(texts):
    embeddings.append(embed(text))

embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)

faiss.write_index(index, "index/faiss.index")

with open("index/metadata.json", "w") as f:
    json.dump(metadata, f)

with open("index/texts.json", "w") as f:
    json.dump(texts, f)

print("✅ RAG ingestion complete")
