#!/bin/bash
# =============================================================================
# Ubik Hippocampal Node - Full Restart Script
# =============================================================================
# Restarts all services required for the hippocampal node:
# - Docker daemon (if not running)
# - Neo4j database container
# - ChromaDB vector store container
# - MCP server
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UBIK_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="/Volumes/990PRO 4T/DeepSeek/venv/bin/activate"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

wait_for_docker() {
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if docker info &>/dev/null; then
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    return 1
}

wait_for_container_health() {
    local container=$1
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        local status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "not_found")

        if [ "$status" = "healthy" ]; then
            return 0
        elif [ "$status" = "unhealthy" ]; then
            # For ChromaDB, unhealthy might just mean the healthcheck command isn't working
            # Check if container is actually running
            if docker ps --filter "name=$container" --filter "status=running" -q | grep -q .; then
                return 0
            fi
            return 1
        elif [ "$status" = "not_found" ]; then
            return 1
        fi

        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    return 1
}

wait_for_port() {
    local port=$1
    local max_attempts=15
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if nc -z localhost "$port" 2>/dev/null; then
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    return 1
}

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "  Ubik Hippocampal Node - Full Restart"
echo "=============================================="
echo ""

# Step 1: Check/Start Docker
echo -e "${BLUE}[1/5]${NC} Checking Docker..."

if docker info &>/dev/null; then
    log_success "Docker is already running"
else
    log_info "Starting Docker Desktop..."
    open -a Docker

    echo -n "    Waiting for Docker to be ready"
    if wait_for_docker; then
        echo ""
        log_success "Docker is ready"
    else
        echo ""
        log_error "Docker failed to start within timeout"
        exit 1
    fi
fi

# Step 2: Stop existing MCP server
echo -e "\n${BLUE}[2/5]${NC} Stopping existing MCP server..."

MCP_PIDS=$(pgrep -f "python.*mcp_server.py" 2>/dev/null || true)
if [ -n "$MCP_PIDS" ]; then
    echo "$MCP_PIDS" | xargs kill 2>/dev/null || true
    sleep 1
    log_success "Stopped existing MCP server (PID: $MCP_PIDS)"
else
    log_info "No existing MCP server running"
fi

# Step 3: Restart database containers
echo -e "\n${BLUE}[3/5]${NC} Restarting database containers..."

cd "$UBIK_DIR"

# Stop containers if running
docker compose down 2>/dev/null || true

# Start containers
log_info "Starting Neo4j and ChromaDB..."
docker compose up -d

# Wait for Neo4j
echo -n "    Waiting for Neo4j to be healthy"
if wait_for_container_health "ubik-neo4j"; then
    echo ""
    log_success "Neo4j is healthy"
else
    echo ""
    log_warn "Neo4j health check inconclusive, checking port..."
    if wait_for_port 7687; then
        log_success "Neo4j is responding on port 7687"
    else
        log_error "Neo4j failed to start"
        exit 1
    fi
fi

# Wait for ChromaDB
echo -n "    Waiting for ChromaDB"
sleep 3  # Give it a moment
if wait_for_port 8001; then
    echo ""
    log_success "ChromaDB is responding on port 8001"
else
    echo ""
    log_error "ChromaDB failed to start"
    exit 1
fi

# Step 4: Start MCP server
echo -e "\n${BLUE}[4/5]${NC} Starting MCP server..."

cd "$SCRIPT_DIR"

# Check for .env file
if [ ! -f .env ]; then
    log_error ".env file not found in $SCRIPT_DIR"
    exit 1
fi

# Activate virtual environment and start server
source "$VENV_PATH"
export $(grep -v '^#' .env | xargs)

# Start MCP server in background with nohup
nohup python mcp_server.py > mcp_server.log 2>&1 &
MCP_PID=$!

sleep 2

# Check if it started
if ps -p $MCP_PID > /dev/null 2>&1; then
    log_success "MCP server started (PID: $MCP_PID)"
else
    log_error "MCP server failed to start. Check mcp_server.log for details"
    exit 1
fi

# Step 5: Verify all services
echo -e "\n${BLUE}[5/5]${NC} Verifying services..."

ERRORS=0

# Check Neo4j
if curl -s -u neo4j:ubik_memory_2024 http://localhost:7474/db/neo4j/tx/commit \
    -H "Content-Type: application/json" \
    -d '{"statements":[{"statement":"RETURN 1"}]}' | grep -q "results"; then
    log_success "Neo4j: Responding (port 7474/7687)"
else
    log_error "Neo4j: Not responding properly"
    ERRORS=$((ERRORS + 1))
fi

# Check ChromaDB
if curl -s http://localhost:8001/api/v2/heartbeat | grep -q "nanosecond"; then
    log_success "ChromaDB: Responding (port 8001)"
else
    # Try v1 API as fallback
    if curl -s http://localhost:8001/api/v1/heartbeat | grep -q -E "(nanosecond|Unimplemented)"; then
        log_success "ChromaDB: Responding (port 8001)"
    else
        log_error "ChromaDB: Not responding properly"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Check MCP server
if curl -s http://localhost:8080/mcp 2>/dev/null | grep -q "jsonrpc"; then
    log_success "MCP Server: Responding (port 8080)"
else
    log_error "MCP Server: Not responding properly"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo ""
echo "=============================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "  ${GREEN}All services are running!${NC}"
    echo "=============================================="
    echo ""
    echo "  Services:"
    echo "    - Neo4j:     bolt://localhost:7687"
    echo "    - Neo4j UI:  http://localhost:7474"
    echo "    - ChromaDB:  http://localhost:8001"
    echo "    - MCP:       http://localhost:8080"
    echo ""
    echo "  Logs:"
    echo "    - MCP:       $SCRIPT_DIR/mcp_server.log"
    echo "    - Neo4j:     docker logs ubik-neo4j"
    echo "    - ChromaDB:  docker logs ubik-chromadb"
    echo ""
else
    echo -e "  ${RED}Some services failed to start ($ERRORS errors)${NC}"
    echo "=============================================="
    echo ""
    echo "  Check logs for details:"
    echo "    - MCP:       $SCRIPT_DIR/mcp_server.log"
    echo "    - Neo4j:     docker logs ubik-neo4j"
    echo "    - ChromaDB:  docker logs ubik-chromadb"
    echo ""
    exit 1
fi
