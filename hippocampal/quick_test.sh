#!/bin/bash
# Quick test of Ubik Hippocampal Node services

echo "============================================"
echo "Ubik Hippocampal Node - Quick Test"
echo "============================================"

# Somatic node config
SOMATIC_HOST="adrian-wsl"
SOMATIC_IP="100.79.166.114"

# Test Neo4j
echo -e "\n[1/5] Testing Neo4j..."
curl -s -u neo4j:ubik_memory_2024 http://localhost:7474/db/neo4j/tx/commit \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"RETURN 1 as test"}]}' | jq -r '.results[0].data[0].row[0]' && \
  echo "  ✓ Neo4j is responding" || echo "  ✗ Neo4j failed"

# Test ChromaDB
echo -e "\n[2/5] Testing ChromaDB..."
curl -s -H "Authorization: Bearer ubik_chroma_token_2024" \
  http://localhost:8001/api/v1/heartbeat | jq -r '.["nanosecond heartbeat"]' && \
  echo "  ✓ ChromaDB is responding" || echo "  ✗ ChromaDB failed"

# Test MCP Server
echo -e "\n[3/5] Testing MCP Server..."
curl -s http://localhost:8080/mcp > /dev/null && \
  echo "  ✓ MCP Server is responding" || echo "  ✗ MCP Server not running (start with ./run_mcp.sh)"

# Test Tailscale
echo -e "\n[4/5] Testing Tailscale..."
/Applications/Tailscale.app/Contents/MacOS/Tailscale status > /dev/null 2>&1 && \
  echo "  ✓ Tailscale is connected" || echo "  ✗ Tailscale not connected"

# Test Somatic Node connectivity
echo -e "\n[5/5] Testing Somatic Node ($SOMATIC_HOST @ $SOMATIC_IP)..."
ping -c 1 -W 2 $SOMATIC_IP > /dev/null 2>&1 && \
  echo "  ✓ Somatic node is reachable" || echo "  ✗ Somatic node not reachable"

# Test Ollama on Somatic Node (if reachable)
if ping -c 1 -W 2 $SOMATIC_IP > /dev/null 2>&1; then
  curl -s --connect-timeout 3 http://$SOMATIC_IP:11434/api/tags > /dev/null 2>&1 && \
    echo "  ✓ Ollama on somatic node is responding" || echo "  ⚠ Ollama not responding (may be normal)"
fi

echo -e "\n============================================"
echo "Quick test complete!"
echo "============================================"
