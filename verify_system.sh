#!/bin/bash
# System Verification Script for the Multi-Agent Quantitative RAG Pipeline

echo "=========================================================="
echo " Starting System Verification..."
echo "=========================================================="
echo ""

# Ensure we're in the correct root directory
cd "$(dirname "$0")"

# Enforce PYTHONPATH for local module resolution
export PYTHONPATH=$(pwd)

# 1. Verify Data Source Ingestion
echo "[1/3] Testing Data Ingestion Pipelines..."
python3 run_scheduler.py --ingest-all
if [ $? -eq 0 ]; then
    echo "✅ Ingestion Pipelines: PASS"
else
    echo "❌ Ingestion Pipelines: FAIL"
    exit 1
fi
echo ""

# 2. Verify Multi-Agent Architecture
echo "[2/3] Testing Intelligent Multi-Agent Query Routing..."
python3 tests/test_multi_agent.py
if [ $? -eq 0 ]; then
    echo "✅ Multi-Agent Orchestration: PASS"
else
    echo "❌ Multi-Agent Orchestration: FAIL"
    exit 1
fi
echo ""

# 3. Interactive Mode Check
echo "[3/3] System verified successfully."
echo "If you would like to run a query interactively, run:"
echo "PYTHONPATH=. python3 rag/query.py"
echo ""
echo "=========================================================="
echo "✅ Verification Complete."
echo "=========================================================="
