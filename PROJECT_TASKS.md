# Project Task Overview

This repository implements a **news-driven financial intelligence pipeline** with both batch processing and query-time analysis.

## 1) News ingestion
- `run_news_ingestion.py` pulls a curated list of article URLs and stores raw article JSON files under `data/raw/news`.
- `ingestion/news_extractor.py` downloads each page, extracts main content via Readability + BeautifulSoup, and returns normalized article fields (title, source, raw text, extraction timestamp).

## 2) Article cleaning and structuring
- `run_cleaning.py` reads raw JSON articles and writes cleaned files into `data/processed/news`.
- `ingestion/cleaner.py` removes noise patterns and formats a structured document with metadata and cleaned body text.

## 3) Text chunking for retrieval
- `run_chunking.py` converts cleaned article text into chunk files under `data/chunks/news`.
- `ingestion/chunker.py` splits text by sentence windows with overlap so retrieval has smaller semantic units.

## 4) Embedding + vector index build
- `run_embedding_index.py` embeds each chunk and writes:
  - FAISS index to `data/vector_db/news.index`
  - retrieval metadata to `data/vector_db/metadata.json`
- `ingestion/embeddings.py` calls local Ollama embeddings endpoint (`nomic-embed-text`) and validates embedding dimensions.

## 5) Retrieval-Augmented Generation (RAG)
- `rag/query.py` handles query embedding, FAISS nearest-neighbor retrieval, prompt construction, and LLM answer generation.
- `rag/rag_core.py` wraps query execution as `ask_rag(...)`.
- `api/app.py` exposes this through a FastAPI `/ask` endpoint.

## 6) Intelligence/analysis tasks
- `intelligence/entity_extractor.py`: entity extraction over vector metadata (companies, indices, sectors, macro items).
- `intelligence/trend_analyzer.py`: market-signal aggregation (top companies/sectors/themes + sentiment distribution).
- `intelligence/macro_engine.py`: streamed multi-step macro workflow:
  1. macro reasoning prompt,
  2. sector-impact prompt,
  3. market-outlook prompt.

## 7) Optional market data helper
- `ingestion/market_data.py` fetches real-time market snapshots via `yfinance` for a predefined symbol list (indices, commodities, crypto).

## End-to-end task sequence
Typical pipeline order:
1. Ingest raw news (`run_news_ingestion.py`)
2. Clean/structure (`run_cleaning.py`)
3. Chunk text (`run_chunking.py`)
4. Build embedding index (`run_embedding_index.py`)
5. Serve and query (`api/app.py` + `rag/*`)
6. Run intelligence enrichments (`intelligence/*`) as needed.


## Quick refresh script
- `refresh_data_and_index.py` runs the update pipeline in one command (cleaning -> chunking -> embedding/index build).
- Optional: pass `--with-ingestion` to include fresh URL ingestion before rebuilding artifacts.
