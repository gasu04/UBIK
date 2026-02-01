#!/bin/bash
# =============================================================================
# UBIK Ingestion System - Quick Start
# =============================================================================
#
# This script verifies the installation and runs basic tests.
#
# Usage:
#   ./quickstart.sh
#   ./quickstart.sh --skip-tests
#   ./quickstart.sh --with-mcp
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${CYAN}"
echo "=============================================="
echo "  UBIK Content Ingestion System"
echo "  Quick Start"
echo "=============================================="
echo -e "${NC}"

# Parse arguments
SKIP_TESTS=false
WITH_MCP=false
for arg in "$@"; do
    case $arg in
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --with-mcp)
            WITH_MCP=true
            shift
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Step 1: Check Python
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/6] Checking Python...${NC}"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "  ${GREEN}✓${NC} $PYTHON_VERSION"
else
    echo -e "  ${RED}✗${NC} Python 3 not found"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: Check/Install Dependencies
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/6] Checking dependencies...${NC}"

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    # Check if key packages are installed
    MISSING=false
    for pkg in pdfplumber pyyaml httpx; do
        if ! python3 -c "import $pkg" 2>/dev/null; then
            echo -e "  ${YELLOW}!${NC} Missing: $pkg"
            MISSING=true
        fi
    done

    if [ "$MISSING" = true ]; then
        echo -e "  Installing dependencies..."
        pip install -q -r requirements.txt
        echo -e "  ${GREEN}✓${NC} Dependencies installed"
    else
        echo -e "  ${GREEN}✓${NC} All dependencies present"
    fi
else
    echo -e "  ${YELLOW}!${NC} requirements.txt not found, skipping"
fi

# -----------------------------------------------------------------------------
# Step 3: Verify Package Import
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/6] Verifying package...${NC}"

if python3 -c "from ingest import IngestPipeline, PipelineConfig; print('  \033[0;32m✓\033[0m Package imports successful')"; then
    :
else
    echo -e "  ${RED}✗${NC} Package import failed"
    exit 1
fi

# Check submodules
python3 -c "
from ingest.models import MemoryCandidate, MemoryType
from ingest.processors import ProcessorRegistry
from ingest.chunkers import SmartChunker
from ingest.classifiers import ContentClassifier
from ingest.transcript_processor import TranscriptProcessor
print('  \033[0;32m✓\033[0m All submodules loaded')
"

# -----------------------------------------------------------------------------
# Step 4: Run Tests
# -----------------------------------------------------------------------------
if [ "$SKIP_TESTS" = false ]; then
    echo -e "\n${YELLOW}[4/6] Running tests...${NC}"

    if python3 -m pytest tests/ -v --tb=short 2>&1 | tail -5; then
        echo -e "  ${GREEN}✓${NC} All tests passed"
    else
        echo -e "  ${RED}✗${NC} Some tests failed"
    fi
else
    echo -e "\n${YELLOW}[4/6] Skipping tests (--skip-tests)${NC}"
fi

# -----------------------------------------------------------------------------
# Step 5: Dry Run on Templates
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/6] Testing with templates...${NC}"

if [ -d "templates" ]; then
    python3 -m ingest.cli local templates/ --dry-run 2>&1 | grep -E "(Files:|Memories:|Episodic:|Semantic:)" | while read line; do
        echo "  $line"
    done
    echo -e "  ${GREEN}✓${NC} Template processing successful"
else
    echo -e "  ${YELLOW}!${NC} Templates directory not found"
fi

# -----------------------------------------------------------------------------
# Step 6: Check MCP Connection (optional)
# -----------------------------------------------------------------------------
if [ "$WITH_MCP" = true ]; then
    echo -e "\n${YELLOW}[6/6] Checking MCP connection...${NC}"

    python3 -c "
import asyncio
import sys
sys.path.insert(0, '../mcp_client')
try:
    from hippocampal_client import HippocampalClient
    async def test():
        async with HippocampalClient() as client:
            if client.is_connected:
                print('  \033[0;32m✓\033[0m Connected to Hippocampal Node')
            else:
                print('  \033[1;33m!\033[0m Not connected (server may be offline)')
    asyncio.run(test())
except Exception as e:
    print(f'  \033[1;33m!\033[0m MCP client not available: {e}')
"
else
    echo -e "\n${YELLOW}[6/6] Skipping MCP check (use --with-mcp to enable)${NC}"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "\n${CYAN}=============================================="
echo "  Quick Start Complete"
echo "==============================================${NC}"
echo
echo "Next steps:"
echo "  1. Prepare your source materials in ~/ubik/data/source_materials/"
echo "  2. Run a dry-run test:"
echo "     python -m ingest.cli local ~/ubik/data/source_materials/ --dry-run"
echo "  3. Use interactive mode for review:"
echo "     python interactive_ingest.py ~/ubik/data/source_materials/"
echo "  4. Connect to Hippocampal Node for storage:"
echo "     python -m ingest.cli local ~/ubik/data/source_materials/ --mcp-host <ip>"
echo
echo -e "See ${CYAN}README.md${NC} for full documentation."
