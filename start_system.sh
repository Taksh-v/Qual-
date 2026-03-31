#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# start_system.sh — Full local startup for the Qual News Intelligence system
#
# What this script does:
#   1. Checks Python and pip
#   2. Installs Python dependencies
#   3. Installs Ollama if missing, starts it, pulls required models
#   4. Runs the data pipeline (ingestion → clean → chunk → embed/index)
#      — skipped if data already exists (use --fresh to force rebuild)
#   5. Starts the background scheduler (RSS ingestion daemon)
#   6. Starts the MCP HTTP server (background daemon)
#   7. Starts the FastAPI server
#
# Usage:
#   bash start_system.sh            # normal start (skips pipeline if data exists)
#   bash start_system.sh --fresh    # force full pipeline rebuild before starting
#   bash start_system.sh --no-ingest  # skip news ingestion step in pipeline
#
# Stop everything:
#   bash start_system.sh --stop
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
header()  { echo -e "\n${BOLD}${CYAN}══════ $* ══════${RESET}\n"; }

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

PID_DIR="$SCRIPT_DIR/.pids"
LOG_DIR="$SCRIPT_DIR/logs"
OLLAMA_PID_FILE="$PID_DIR/ollama.pid"
SCHEDULER_PID_FILE="$PID_DIR/scheduler.pid"
MCP_PID_FILE="$PID_DIR/mcp.pid"
API_PID_FILE="$PID_DIR/api.pid"

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
if [[ "$HOST" == "0.0.0.0" ]]; then
  warn "HOST=0.0.0.0 is blocked in this environment; switching to 127.0.0.1"
  HOST="127.0.0.1"
fi
MCP_HOST="${MCP_HOST:-127.0.0.1}"
MCP_PORT="${MCP_PORT:-8010}"
MCP_HEALTH_HOST="$MCP_HOST"
if [[ "$MCP_HEALTH_HOST" == "0.0.0.0" ]]; then
  MCP_HEALTH_HOST="127.0.0.1"
fi

EMBED_PROVIDER="${EMBED_PROVIDER:-ollama}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
export EMBED_PROVIDER EMBED_MODEL

REQUIRED_MODELS=("nomic-embed-text" "mistral")
INDEX_FILE="data/vector_db/news.index"
METADATA_FILE="data/vector_db/metadata.json"

FRESH=false
NO_INGEST=false
STOP=false

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --fresh)     FRESH=true ;;
    --no-ingest) NO_INGEST=true ;;
    --stop)      STOP=true ;;
    --resume)    ;; # alias: keep existing data, do not force rebuild
    *) warn "Unknown argument: $arg" ;;
  esac
done

# ── Stop mode ─────────────────────────────────────────────────────────────────
if $STOP; then
  header "Stopping all Qual services"
  for pf in "$API_PID_FILE" "$MCP_PID_FILE" "$SCHEDULER_PID_FILE" "$OLLAMA_PID_FILE"; do
    if [[ -f "$pf" ]]; then
      pid=$(cat "$pf")
      name=$(basename "$pf" .pid)
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" && success "Stopped $name (PID $pid)"
      else
        warn "$name PID $pid not running"
      fi
      rm -f "$pf"
    fi
  done
  # Also kill by process name as fallback
  pkill -f "uvicorn api.app:app" 2>/dev/null && success "Killed stray uvicorn process" || true
  pkill -f "run_scheduler.py"    2>/dev/null && success "Killed stray scheduler process" || true
  pkill -f "run_mcp_http.py"     2>/dev/null && success "Killed stray MCP process" || true
  success "All services stopped."
  exit 0
fi

mkdir -p "$PID_DIR" "$LOG_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Python
# ─────────────────────────────────────────────────────────────────────────────
header "Step 1 — Python"

PYTHON=$(command -v python3 || command -v python || true)
[[ -z "$PYTHON" ]] && error "Python 3 not found. Please install Python 3.9+."
PY_VER=$("$PYTHON" --version 2>&1)
success "Found $PY_VER at $PYTHON"

PIP=$(command -v pip3 || command -v pip || true)
[[ -z "$PIP" ]] && error "pip not found."

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
header "Step 2 — Python Dependencies"

# Quick check: if faiss and uvicorn already importable, skip reinstall
if "$PYTHON" -c "import faiss, uvicorn, fastapi" 2>/dev/null; then
  success "Core packages already installed — skipping pip install"
else
  info "Installing dependencies from requirements.txt..."
  $PIP install -r requirements.txt --break-system-packages --quiet \
    || $PIP install -r requirements.txt --quiet \
    || error "pip install failed. Check requirements.txt and your Python environment."
  success "Dependencies installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Ollama
# ─────────────────────────────────────────────────────────────────────────────
header "Step 3 — Ollama (Local LLM)"

# If Ollama can't start in this environment (e.g., port binding restrictions),
# we still want the rest of the system (scheduler/MCP/API) to come up.
OLLAMA_AVAILABLE=false

# Install if missing
if ! command -v ollama &>/dev/null; then
  info "Ollama not found — installing..."
  curl -fsSL https://ollama.com/install.sh | sh \
    || error "Ollama installation failed. Install manually: https://ollama.com/download"
  success "Ollama installed"
else
  success "Ollama already installed: $(ollama --version 2>/dev/null || echo 'unknown version')"
fi

# Start ollama serve if not already running
if curl -sf "$OLLAMA_URL/api/tags" &>/dev/null; then
  success "Ollama is already running at $OLLAMA_URL"
  OLLAMA_AVAILABLE=true
else
  info "Starting Ollama server..."
  ollama serve >> "$LOG_DIR/ollama.log" 2>&1 &
  echo $! > "$OLLAMA_PID_FILE"

  # Wait up to 30s for it to be ready
  for i in $(seq 1 30); do
    sleep 1
    if curl -sf "$OLLAMA_URL/api/tags" &>/dev/null; then
      success "Ollama server started (PID $(cat $OLLAMA_PID_FILE))"
      OLLAMA_AVAILABLE=true
      break
    fi
    if [[ $i -eq 30 ]]; then
      warn "Ollama did not start within 30s (continuing without Ollama). Check $LOG_DIR/ollama.log"
    fi
  done
fi

# Pull required models
for model in "${REQUIRED_MODELS[@]}"; do
  if [[ "$OLLAMA_AVAILABLE" != "true" ]]; then
    warn "Skipping model pull for '$model' because Ollama is unavailable"
    continue
  fi

  if ollama list 2>/dev/null | grep -q "^${model}"; then
    success "Model '$model' already present"
  else
    info "Pulling model '$model' (this may take a few minutes)..."
    ollama pull "$model" || warn "Failed to pull model '$model' (continuing)"
    success "Model '$model' pulled"
  fi
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Data pipeline
# ─────────────────────────────────────────────────────────────────────────────
header "Step 4 — Data Pipeline"

if [[ -f "$INDEX_FILE" && "$FRESH" == "false" ]]; then
  info "Checking embedding/index compatibility..."
  dims=$(
    INDEX_FILE="$INDEX_FILE" "$PYTHON" - <<'PY'
import os
import sys

try:
    import faiss
    from ingestion.embeddings import get_embedding

    index_path = os.environ.get("INDEX_FILE")
    index = faiss.read_index(index_path)
    emb = get_embedding("dimension check", normalize=True, role="query")
    print(f"{index.d} {len(emb)}")
except Exception as exc:
    print(f"ERR {exc}")
    sys.exit(0)
PY
  )

  if [[ "$dims" == ERR* ]]; then
    warn "Compatibility check skipped: ${dims#ERR }"
  else
    read -r idx_dim emb_dim <<< "$dims"
    if [[ -n "$idx_dim" && -n "$emb_dim" && "$idx_dim" != "$emb_dim" ]]; then
      warn "Embedding dimension ($emb_dim) does not match index dimension ($idx_dim)."
      warn "Forcing full rebuild to avoid retrieval failures."
      FRESH=true
    else
      success "Embedding/index dimensions aligned ($idx_dim)."
    fi
  fi
fi

DATA_EXISTS=false
if [[ -f "$INDEX_FILE" && -f "$METADATA_FILE" ]]; then
  DATA_EXISTS=true
fi

if $DATA_EXISTS && ! $FRESH; then
  success "Vector index already exists — skipping pipeline"
  info "   (Run with --fresh to force a full rebuild)"
else
  if $FRESH; then
    info "--fresh flag set: running full pipeline rebuild"
  else
    info "No vector index found — running pipeline for the first time"
  fi

  # 4a. News ingestion
  if $NO_INGEST; then
    warn "Skipping news ingestion (--no-ingest)"
  else
    info "Running news ingestion..."
    "$PYTHON" run_news_ingestion.py \
      || warn "News ingestion had errors (continuing — may use existing data)"
    success "News ingestion complete"
  fi

  # 4b. Clean → Chunk → Embed/Index
  info "Running clean → chunk → embed/index pipeline..."
  "$PYTHON" refresh_data_and_index.py \
    || error "Pipeline failed. Check logs above."
  success "Pipeline complete"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Scheduler (background daemon)
# ─────────────────────────────────────────────────────────────────────────────
header "Step 5 — RSS Ingestion Scheduler"

# Kill stale scheduler if pid file exists but process is dead
if [[ -f "$SCHEDULER_PID_FILE" ]]; then
  old_pid=$(cat "$SCHEDULER_PID_FILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    success "Scheduler already running (PID $old_pid)"
  else
    rm -f "$SCHEDULER_PID_FILE"
  fi
fi

if [[ ! -f "$SCHEDULER_PID_FILE" ]]; then
  info "Starting scheduler daemon..."
  nohup "$PYTHON" run_scheduler.py >> "$LOG_DIR/scheduler.log" 2>&1 &
  echo $! > "$SCHEDULER_PID_FILE"
  sleep 1
  if kill -0 "$(cat $SCHEDULER_PID_FILE)" 2>/dev/null; then
    success "Scheduler started (PID $(cat $SCHEDULER_PID_FILE))"
    info "   Logs: $LOG_DIR/scheduler.log"
  else
    warn "Scheduler may have exited early. Check $LOG_DIR/scheduler.log"
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — MCP HTTP server (background daemon)
# ─────────────────────────────────────────────────────────────────────────────
header "Step 6 — MCP HTTP Server"

# Kill stale MCP process if pid file exists but process is dead
if [[ -f "$MCP_PID_FILE" ]]; then
  old_pid=$(cat "$MCP_PID_FILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    success "MCP server already running (PID $old_pid)"
  else
    rm -f "$MCP_PID_FILE"
  fi
fi

if [[ ! -f "$MCP_PID_FILE" ]]; then
  info "Starting MCP HTTP server..."
  nohup env MCP_HOST="$MCP_HOST" MCP_PORT="$MCP_PORT" "$PYTHON" run_mcp_http.py >> "$LOG_DIR/mcp.log" 2>&1 &
  echo $! > "$MCP_PID_FILE"
  sleep 1
  if kill -0 "$(cat $MCP_PID_FILE)" 2>/dev/null; then
    # MCP may take a few seconds to fully come up; treat probe failures as informational.
    MCP_HEALTH_OK=false
    for i in $(seq 1 15); do
      if curl -sf --max-time 1 "http://${MCP_HEALTH_HOST}:${MCP_PORT}/health" &>/dev/null; then
        success "MCP server started and healthy (PID $(cat $MCP_PID_FILE))"
        MCP_HEALTH_OK=true
        break
      fi
      sleep 1
    done

    if [[ "$MCP_HEALTH_OK" != "true" ]]; then
      info "MCP process is running (PID $(cat $MCP_PID_FILE)), health probe didn't succeed yet: http://${MCP_HEALTH_HOST}:${MCP_PORT}/health"
      info "   If you still can't use MCP, check $LOG_DIR/mcp.log for details."
    fi
    info "   Logs: $LOG_DIR/mcp.log"
  else
    warn "MCP server may have exited early. Check $LOG_DIR/mcp.log"
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — API Server
# ─────────────────────────────────────────────────────────────────────────────
header "Step 7 — FastAPI Server"

# Kill stale API process if pid file exists but process is dead
if [[ -f "$API_PID_FILE" ]]; then
  old_pid=$(cat "$API_PID_FILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    warn "API server already running (PID $old_pid). Kill it first with: bash start_system.sh --stop"
    echo ""
  else
    rm -f "$API_PID_FILE"
  fi
fi

echo -e "${BOLD}${GREEN}"
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║     Qual News Intelligence — Ready           ║"
echo "  ║                                              ║"
echo "  ║   Web UI  : http://localhost:${PORT}           ║"
echo "  ║   API Docs: http://localhost:${PORT}/docs       ║"
echo "  ║   MCP    : http://localhost:${MCP_PORT}/mcp     ║"
echo "  ║   Ollama  : $OLLAMA_URL  ║"
echo "  ║                                              ║"
echo "  ║   Press Ctrl+C to stop the API server        ║"
echo "  ║   Scheduler + MCP + Ollama keep running      ║"
echo "  ╚══════════════════════════════════════════════╝"
echo -e "${RESET}"

info "Starting API server on $HOST:$PORT (workers=4)..."
info "API logs will appear below. Ctrl+C to stop.\n"

# Run API in foreground (so logs stream to terminal)
# Register cleanup for Ctrl+C
cleanup() {
  echo ""
  warn "API server stopped."
  rm -f "$API_PID_FILE"
  info "Scheduler, MCP, and Ollama are still running in background."
  info "To stop everything: bash start_system.sh --stop"
}
trap cleanup INT TERM

uvicorn api.app:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers 4 \
  --proxy-headers \
  --log-level "${LOG_LEVEL:-info}" &

API_PID=$!
echo $API_PID > "$API_PID_FILE"
wait $API_PID || true
cleanup
