#!/bin/bash
# Start Ubik inference server

source ~/ubik/scripts/activate_ubik.sh

LOG_DIR=~/ubik/logs/inference
mkdir -p $LOG_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/vllm_$TIMESTAMP.log

echo "Starting Ubik Inference Server..."
echo "Log file: $LOG_FILE"

# Check if already running
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "Server already running. Stop it first with: ~/ubik/scripts/stop_inference.sh"
    exit 1
fi

# Start server in background
nohup python ~/ubik/somatic/inference/vllm_server.py \
    --config ~/ubik/config/models/vllm_config.yaml \
    > $LOG_FILE 2>&1 &

echo $! > ~/ubik/logs/inference/vllm.pid
echo "Server started with PID: $(cat ~/ubik/logs/inference/vllm.pid)"
echo "Waiting for server to be ready..."

# Wait for server to be ready
for i in {1..60}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "Server is ready!"
        exit 0
    fi
    sleep 2
    echo -n "."
done

echo ""
echo "Server did not become ready in time. Check logs: $LOG_FILE"
exit 1
