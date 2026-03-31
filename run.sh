#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
	set -a
	# shellcheck disable=SC1091
	source "$SCRIPT_DIR/.env"
	set +a
fi

export EMBED_PROVIDER="${EMBED_PROVIDER:-ollama}"
export EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
export MCP_HOST="${MCP_HOST:-127.0.0.1}"
export MCP_PORT="${MCP_PORT:-8010}"

cleanup() {
	if [[ -n "${MCP_PID:-}" ]] && kill -0 "$MCP_PID" 2>/dev/null; then
		kill "$MCP_PID" 2>/dev/null || true
		wait "$MCP_PID" 2>/dev/null || true
	fi
}
trap cleanup EXIT INT TERM

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Start MCP server in background
python run_mcp_http.py &
MCP_PID=$!
sleep 1
if ! kill -0 "$MCP_PID" 2>/dev/null; then
	echo "[ERROR] MCP server failed to start."
	exit 1
fi

# Run the server (production mode with 4 workers)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4 --proxy-headers
