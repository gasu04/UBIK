#!/bin/bash
# Quick MCP client connectivity test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "UBIK Somatic Node - Quick MCP Client Test"
echo "============================================"

# Activate venv
source /home/gasu/pytorch_env/bin/activate

# Set PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Load environment (strip carriage returns for Windows compatibility)
if [ -f config/.env ]; then
    export $(grep -v '^#' config/.env | tr -d '\r' | xargs)
fi

# Run client test
echo -e "\nRunning MCP client test..."
python3 mcp_client/hippocampal_client.py

echo -e "\n============================================"
echo "Quick test complete!"
echo "============================================"
