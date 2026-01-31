#!/bin/bash
# Ubik Hippocampal Node - MCP Server Launcher
#
# Usage:
#   ./run_mcp.sh           # Run in foreground
#   ./run_mcp.sh -d        # Run in background (daemon mode)
#   ./run_mcp.sh stop      # Stop the background server (and any orphans)
#   ./run_mcp.sh restart   # Stop and start in background
#   ./run_mcp.sh status    # Check if server is running
#   ./run_mcp.sh logs      # Tail the log file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/mcp_server.log"
PID_FILE="$SCRIPT_DIR/.mcp_server.pid"
MCP_PORT=8080

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Activate virtual environment
source "/Volumes/990PRO 4T/DeepSeek/venv/bin/activate"

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    exit 1
fi

# Export environment variables
export $(grep -v '^#' .env | xargs)

# Function to check if server is running via PID file
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to find any mcp_server process (orphans included)
find_mcp_processes() {
    pgrep -f "python.*mcp_server.py" 2>/dev/null
}

# Function to find process using the MCP port
find_port_process() {
    lsof -ti:$MCP_PORT 2>/dev/null
}

# Function to stop all MCP server processes
stop_all() {
    local stopped=0

    # First, try PID file
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Stopping MCP server (PID: $PID)..."
            kill "$PID" 2>/dev/null
            stopped=1
        fi
        rm -f "$PID_FILE"
    fi

    # Kill any orphan mcp_server processes
    ORPHANS=$(find_mcp_processes)
    if [ -n "$ORPHANS" ]; then
        echo "Stopping orphan MCP server processes: $ORPHANS"
        echo "$ORPHANS" | xargs kill 2>/dev/null
        stopped=1
    fi

    # Check if port is still in use (by something else)
    sleep 0.5
    PORT_PID=$(find_port_process)
    if [ -n "$PORT_PID" ]; then
        echo "Warning: Port $MCP_PORT still in use by PID $PORT_PID"
        echo "Run 'kill $PORT_PID' to free the port if needed"
    fi

    if [ $stopped -eq 1 ]; then
        echo "Server stopped."
    else
        echo "No MCP server was running."
    fi
}

# Function to start the server in background
start_daemon() {
    # Check for any existing processes first
    EXISTING=$(find_mcp_processes)
    PORT_PID=$(find_port_process)

    if [ -n "$EXISTING" ] || [ -n "$PORT_PID" ]; then
        echo "Warning: Found existing MCP server or port conflict"
        echo "  MCP processes: ${EXISTING:-none}"
        echo "  Port $MCP_PORT used by: ${PORT_PID:-none}"
        echo "Stopping existing processes first..."
        stop_all
        sleep 1
    fi

    echo "Starting Ubik Hippocampal MCP Server in background..."
    nohup python mcp_server.py >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 1

    # Verify it started
    if ps -p $! > /dev/null 2>&1; then
        echo "Server started (PID: $!)"
        echo "Logs: $LOG_FILE"
        echo "Stop with: ./run_mcp.sh stop"
    else
        echo "Error: Server failed to start. Check logs:"
        tail -10 "$LOG_FILE"
        exit 1
    fi
}

# Handle commands
case "$1" in
    stop)
        stop_all
        exit 0
        ;;
    restart)
        echo "Restarting MCP server..."
        stop_all
        sleep 1
        start_daemon
        exit 0
        ;;
    status)
        echo "=== MCP Server Status ==="
        if is_running; then
            PID=$(cat "$PID_FILE")
            echo "PID file: running (PID: $PID)"
        else
            echo "PID file: not running"
        fi

        ORPHANS=$(find_mcp_processes)
        if [ -n "$ORPHANS" ]; then
            echo "MCP processes: $ORPHANS"
        else
            echo "MCP processes: none"
        fi

        PORT_PID=$(find_port_process)
        if [ -n "$PORT_PID" ]; then
            echo "Port $MCP_PORT: in use (PID: $PORT_PID)"
        else
            echo "Port $MCP_PORT: available"
        fi
        exit 0
        ;;
    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found at $LOG_FILE"
        fi
        exit 0
        ;;
    -d|--daemon)
        start_daemon
        exit 0
        ;;
    "")
        # Run in foreground - also check for conflicts
        EXISTING=$(find_mcp_processes)
        PORT_PID=$(find_port_process)
        if [ -n "$EXISTING" ] || [ -n "$PORT_PID" ]; then
            echo "Warning: MCP server or port already in use"
            echo "  MCP processes: ${EXISTING:-none}"
            echo "  Port $MCP_PORT used by: ${PORT_PID:-none}"
            echo "Run './run_mcp.sh stop' first, or './run_mcp.sh restart'"
            exit 1
        fi
        echo "Starting Ubik Hippocampal MCP Server..."
        echo "Tip: Use './run_mcp.sh -d' to run in background"
        python mcp_server.py
        ;;
    *)
        echo "Usage: $0 [-d|--daemon|stop|restart|status|logs]"
        exit 1
        ;;
esac
