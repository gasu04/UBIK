#!/bin/bash
# Check inference server status

echo "=== Ubik Inference Server Status ==="

# Check if process is running
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    PID=$(pgrep -f "vllm.entrypoints.openai.api_server")
    echo "Process: Running (PID: $PID)"
else
    echo "Process: NOT RUNNING"
    exit 1
fi

# Check HTTP endpoint
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "HTTP: Responding"
else
    echo "HTTP: NOT RESPONDING"
    exit 1
fi

# Check models endpoint
MODELS=$(curl -s http://localhost:8080/v1/models)
echo "Models: $MODELS"

# GPU status
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "Server is healthy!"
