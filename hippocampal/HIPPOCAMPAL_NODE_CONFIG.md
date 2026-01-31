# Hippocampal Node Configuration (Mac Mini)

## MCP Server Configuration (.env)

```env
# Ubik Hippocampal Node Configuration

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=ubik_memory_2024

# ChromaDB
CHROMADB_HOST=localhost
CHROMADB_PORT=8001
CHROMADB_TOKEN=ubik_chroma_token_2024

# MCP Server
MCP_HOST=0.0.0.0
MCP_PORT=8080

# Logging
LOG_LEVEL=INFO
```

## Service Ports

| Service | Port | Protocol |
|---------|------|----------|
| MCP Server | 8080 | HTTP |
| Neo4j Bolt | 7687 | Bolt |
| Neo4j Browser | 7474 | HTTP |
| ChromaDB | 8001 | HTTP |

---

## Modifications to mcp_server.py (Phase 1)

### 1. Added `asyncio` import (line 22)

```python
import os
import json
import logging
import asyncio  # <-- ADDED
from datetime import datetime
```

### 2. Fixed `get_identity_context` tool (lines 405-468)

**Issues Fixed:**
1. Variable depth in Cypher path pattern (`$depth` doesn't work - Neo4j limitation)
2. Missing `coalesce()` for `labels(n)[0]` safety
3. Sync Neo4j driver blocking async event loop

**Updated Implementation:**

```python
@mcp.tool()
async def get_identity_context(concept: str, depth: int = 2) -> Dict[str, Any]:
    """
    Retrieve identity context from the Parfitian identity graph.

    Traverses the Neo4j graph to find related concepts, values,
    beliefs, and their relationships.

    Args:
        concept: The concept/entity to explore
        depth: How many relationship hops to traverse (1-3)

    Returns:
        Graph context with related nodes and relationships
    """
    try:
        depth = min(max(depth, 1), 3)  # Clamp between 1-3

        def _run_query():
            """Execute sync Neo4j query."""
            with db.neo4j.session() as session:
                # Note: Cypher doesn't support parameterized path lengths,
                # so we use f-string (safe since depth is clamped to 1-3)
                result = session.run(f"""
                    MATCH path = (start:Concept {{name: $concept}})-[*1..{depth}]-(related)
                    RETURN
                        start.name as source,
                        [r in relationships(path) | type(r)] as relationship_types,
                        [n in nodes(path) | {{name: n.name, type: coalesce(labels(n)[0], 'Unknown'), properties: properties(n)}}] as nodes,
                        length(path) as distance
                    ORDER BY distance
                    LIMIT 50
                """, concept=concept)

                paths = []
                for record in result:
                    paths.append({
                        "source": record["source"],
                        "relationships": record["relationship_types"],
                        "nodes": record["nodes"],
                        "distance": record["distance"]
                    })
                return paths

        # Run sync Neo4j query in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        paths = await loop.run_in_executor(None, _run_query)

        logger.info(f"Identity context retrieved for '{concept}': {len(paths)} paths")

        return {
            "status": "success",
            "concept": concept,
            "depth": depth,
            "paths_found": len(paths),
            "context": paths
        }

    except Exception as e:
        logger.error(f"Error getting identity context: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
```

### Key Changes Summary

| Change | Before | After |
|--------|--------|-------|
| asyncio import | Not present | `import asyncio` added |
| Variable depth | `$depth` parameter (broken) | f-string `{depth}` (safe, clamped 1-3) |
| Label safety | `labels(n)[0]` | `coalesce(labels(n)[0], 'Unknown')` |
| Async handling | Sync call blocking event loop | `run_in_executor()` wrapper |

---

## MCP Tools Available

| Tool | Description |
|------|-------------|
| `store_episodic` | Store new episodic memories (always writable) |
| `query_episodic` | Search episodic memories |
| `store_semantic` | Store semantic knowledge (respects freeze state) |
| `query_semantic` | Search semantic knowledge |
| `get_identity_context` | Retrieve Parfitian identity context from Neo4j |
| `update_identity_graph` | Update identity relationships |
| `get_memory_stats` | Get statistics about the memory system |
| `freeze_voice` | Freeze semantic memory (Frozen Voice mode) |
| `unfreeze_voice` | Unfreeze semantic memory (requires confirmation) |

---

## Starting Services

```bash
# Full restart script
./restart_all.sh

# Or manually:
# 1. Start Docker containers
cd /Volumes/990PRO\ 4T/ubik && docker compose up -d

# 2. Start MCP server
cd /Volumes/990PRO\ 4T/ubik/hippocampal
source /Volumes/990PRO\ 4T/DeepSeek/venv/bin/activate
python mcp_server.py
```
