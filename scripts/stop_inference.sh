#!/bin/bash
# Stop Ubik inference server

PID_FILE=~/ubik/logs/inference/vllm.pid

if [ -f $PID_FILE ]; then
    PID=$(cat $PID_FILE)
    if kill -0 $PID 2>/dev/null; then
        echo "Stopping server (PID: $PID)..."
        kill $PID
        rm $PID_FILE
        echo "Server stopped."
    else
        echo "Server not running (stale PID file)."
        rm $PID_FILE
    fi
else
    echo "No PID file found. Checking for running processes..."
    pkill -f "vllm.entrypoints.openai.api_server" && echo "Killed running server." || echo "No server running."
fi
