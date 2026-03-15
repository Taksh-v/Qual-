#!/bin/bash

export EMBED_PROVIDER="${EMBED_PROVIDER:-ollama}"
export EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Run the server (production mode with 4 workers)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4 --proxy-headers
