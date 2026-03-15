# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — News Intelligence RAG API
# Build:  docker build -t qual-rag .
# Run:    docker-compose up  (see docker-compose.yml)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps for lxml / faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data needed by chunker
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# Copy source
COPY . .

# Create data directories if not mounted
RUN mkdir -p data/raw/news data/raw/rss \
             data/chunks/news data/chunks/rss \
             data/processed/news \
             data/vector_db \
             data/logs

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/health/quick || exit 1

# Default: run the API server
CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
