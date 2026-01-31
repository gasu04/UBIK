# UBIK Project Documentation

**Digital Legacy System - Hippocampal Node**

Generated: 2026-01-17 22:53:35

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Configuration Files](#configuration-files)
5. [Source Code](#source-code)
6. [Health Check Results](#health-check-results)
7. [Setup Instructions](#setup-instructions)
8. [Usage Guide](#usage-guide)

---

## Project Overview

UBIK is a digital legacy/memory preservation system inspired by Derek Parfit's philosophical theory of personal identity and psychological continuity. The system is designed to preserve and transmit personal identity, values, memories, and wisdom to future generations (specifically grandchildren).

### Core Philosophy: "Frozen Voice, Living Mind"

- **Frozen Voice**: After voice/personality training is complete, semantic memories (beliefs, values, preferences) can be frozen to preserve the authentic voice
- **Living Mind**: Episodic memories (experiences, conversations, events) can always be added, allowing the system to continue growing

### Parfitian Identity Model

The system implements Derek Parfit's theory that personal identity persists through psychological continuity - the "Relation R" consisting of:
- Psychological connectedness (direct memory/intention links)
- Psychological continuity (overlapping chains of connections)

---

## Architecture

### Two-Node Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     UBIK SYSTEM ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────┐         ┌─────────────────────────┐
│   HIPPOCAMPAL NODE      │         │     SOMATIC NODE        │
│   (Memory Storage)      │◄───────►│     (Inference)         │
│                         │Tailscale│                         │
│   Host: minim4-2025     │ Mesh    │   Host: adrian-wsl      │
│   IP: 100.103.242.91    │         │   IP: 100.79.166.114    │
│                         │         │                         │
│   Services:             │         │   Services:             │
│   • Neo4j (7474, 7687)  │         │   • Ollama (11434)      │
│   • ChromaDB (8001)     │         │   • Inference API (8081)│
│   • MCP Server (8080)   │         │                         │
└─────────────────────────┘         └─────────────────────────┘
```

### Database Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    MEMORY SYSTEM                               │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────┐      ┌─────────────────────────────┐ │
│  │     ChromaDB        │      │          Neo4j              │ │
│  │  (Vector Store)     │      │      (Graph Database)       │ │
│  │                     │      │                             │ │
│  │  ubik_episodic:     │      │  Identity Graph:            │ │
│  │  • Letters          │      │  • Self (CoreIdentity)      │ │
│  │  • Therapy sessions │      │  • Values                   │ │
│  │  • Family meetings  │      │  • Traits                   │ │
│  │  • Conversations    │      │  • Relationships            │ │
│  │                     │      │  • TimeSlices               │ │
│  │  ubik_semantic:     │      │  • Memories                 │ │
│  │  • Beliefs          │      │                             │ │
│  │  • Values           │      │  Parfitian Relations:       │ │
│  │  • Preferences      │      │  • INFLUENCES               │ │
│  │  • Facts            │      │  • DERIVES_FROM             │ │
│  │                     │      │  • SUPPORTS                 │ │
│  └─────────────────────┘      │  • CONNECTS_TO              │ │
│                               │  • CONTINUES_AS             │ │
│                               └─────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
/Volumes/990PRO 4T/UBIK/
├── docker-compose.yml              # Docker services configuration
├── UBIK_PROJECT_DOCUMENTATION.md   # This documentation file
│
├── config/
│   └── tailscale_config.json       # Tailscale mesh network configuration
│
├── data/
│   ├── neo4j/
│   │   ├── data/                   # Neo4j database files
│   │   ├── logs/                   # Neo4j logs
│   │   ├── import/                 # Neo4j import directory
│   │   └── plugins/
│   │       └── apoc.jar            # APOC plugin
│   └── chromadb/                   # ChromaDB persistent storage
│
├── hippocampal/
│   ├── .env                        # Environment configuration
│   ├── mcp_server.py               # FastMCP server (9 tools)
│   ├── setup_chromadb.py           # ChromaDB collection initialization
│   ├── init_neo4j_schema.py        # Neo4j schema and seed data
│   ├── health_check.py             # Comprehensive health check
│   ├── run_mcp.sh                  # MCP server launcher
│   ├── quick_test.sh               # Quick connectivity test
│   ├── mcp.log                     # MCP server log file
│   └── __pycache__/                # Python bytecode cache
│
└── logs/                           # Application logs directory
```

---

## Configuration Files

### docker-compose.yml

```yaml
services:
  neo4j:
    image: neo4j:5.15-community
    container_name: ubik-neo4j
    restart: unless-stopped
    ports:
      - "7474:7474"  # HTTP (Browser interface)
      - "7687:7687"  # Bolt protocol
    volumes:
      - /Volumes/990PRO 4T/ubik/data/neo4j/data:/data
      - /Volumes/990PRO 4T/ubik/data/neo4j/logs:/logs
      - /Volumes/990PRO 4T/ubik/data/neo4j/import:/var/lib/neo4j/import
      - /Volumes/990PRO 4T/ubik/data/neo4j/plugins:/plugins
    environment:
      - NEO4J_AUTH=neo4j/ubik_memory_2024
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
      - NEO4J_dbms_memory_heap_initial__size=512m
      - NEO4J_dbms_memory_heap_max__size=2G
      - NEO4J_dbms_memory_pagecache_size=512m
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:7474"]
      interval: 30s
      timeout: 10s
      retries: 5

  chromadb:
    image: chromadb/chroma:latest
    container_name: ubik-chromadb
    restart: unless-stopped
    ports:
      - "8001:8000"
    volumes:
      - /Volumes/990PRO 4T/ubik/data/chromadb:/chroma/chroma
    environment:
      - CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token.TokenAuthServerProvider
      - CHROMA_SERVER_AUTH_CREDENTIALS=ubik_chroma_token_2024
      - CHROMA_SERVER_AUTH_TOKEN_TRANSPORT_HEADER=Authorization
      - ANONYMIZED_TELEMETRY=False
      - IS_PERSISTENT=True
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 5

networks:
  default:
    name: ubik-network
```

### config/tailscale_config.json

```json
{
    "mesh_name": "ubik-axonal-link",
    "nodes": {
        "hippocampal": {
            "hostname": "minim4-2025",
            "target_hostname": "ubik-hippocampal",
            "tailscale_ip": "100.103.242.91",
            "services": {
                "neo4j_bolt": 7687,
                "neo4j_http": 7474,
                "chromadb": 8001,
                "mcp_server": 8080
            }
        },
        "somatic": {
            "hostname": "adrian-wsl",
            "target_hostname": "ubik-somatic",
            "tailscale_ip": "100.79.166.114",
            "services": {
                "ollama": 11434,
                "inference_api": 8081
            }
        }
    },
    "security": {
        "acl_tags": ["tag:ubik-node"],
        "require_auth": true
    }
}
```

### hippocampal/.env

```bash
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

---

## Source Code

### hippocampal/mcp_server.py

MCP Server exposing 9 tools for memory management:

```python
#!/usr/bin/env python3
"""
Ubik Hippocampal Node - MCP Server

Exposes memory tools to the Somatic Node via Model Context Protocol (MCP).
Implements the "Frozen Voice, Living Mind" architecture where:
- Episodic memories can always be added (Living Mind)
- Semantic memories can be frozen after voice training (Frozen Voice)

Tools exposed:
- store_episodic: Store new episodic memories
- query_episodic: Search episodic memories
- store_semantic: Store semantic knowledge (respects freeze state)
- query_semantic: Search semantic knowledge
- get_identity_context: Retrieve Parfitian identity context from Neo4j
- update_identity_graph: Update identity relationships
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import chromadb
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ubik.hippocampal")


# =============================================================================
# Data Models
# =============================================================================

class EpisodicMemory(BaseModel):
    """Schema for episodic memory entries."""
    content: str = Field(..., description="The memory content/description")
    memory_type: str = Field(..., description="Type: letter, therapy_session, family_meeting, conversation, event")
    timestamp: Optional[str] = Field(None, description="ISO timestamp (auto-generated if not provided)")
    emotional_valence: str = Field("neutral", description="Emotional tone: positive, negative, neutral, reflective, mixed")
    importance: float = Field(0.5, ge=0, le=1, description="Importance score 0-1")
    participants: str = Field("gines", description="Comma-separated list of participants")
    themes: str = Field("", description="Comma-separated themes/tags")
    source_file: Optional[str] = Field(None, description="Source document reference if applicable")


class SemanticKnowledge(BaseModel):
    """Schema for semantic knowledge entries."""
    content: str = Field(..., description="The knowledge/belief/value statement")
    knowledge_type: str = Field(..., description="Type: belief, value, preference, fact, opinion")
    category: str = Field(..., description="Category: family, relationships, philosophy, communication, career, health")
    confidence: float = Field(0.8, ge=0, le=1, description="Confidence level 0-1")
    stability: str = Field("stable", description="Stability: core, stable, evolving")
    source: str = Field("reflection", description="Source of knowledge")


class QueryParams(BaseModel):
    """Schema for memory queries."""
    query: str = Field(..., description="Search query text")
    n_results: int = Field(5, ge=1, le=20, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class IdentityRelation(BaseModel):
    """Schema for identity graph relationships."""
    from_concept: str = Field(..., description="Source concept/entity")
    relation_type: str = Field(..., description="Relationship type: INFLUENCES, DERIVES_FROM, SUPPORTS, CONFLICTS_WITH")
    to_concept: str = Field(..., description="Target concept/entity")
    weight: float = Field(1.0, ge=0, le=1, description="Relationship strength")
    context: Optional[str] = Field(None, description="Additional context")


# =============================================================================
# Database Connections
# =============================================================================

class DatabaseManager:
    """Manages connections to ChromaDB and Neo4j."""

    def __init__(self):
        self._chroma_client = None
        self._neo4j_driver = None
        self._semantic_frozen = False

    @property
    def chroma(self):
        """Lazy-load ChromaDB client."""
        if self._chroma_client is None:
            self._chroma_client = chromadb.HttpClient(
                host=os.getenv("CHROMADB_HOST", "localhost"),
                port=int(os.getenv("CHROMADB_PORT", 8001)),
                headers={"Authorization": f"Bearer {os.getenv('CHROMADB_TOKEN')}"}
            )
            logger.info("ChromaDB client initialized")
        return self._chroma_client

    @property
    def neo4j(self):
        """Lazy-load Neo4j driver."""
        if self._neo4j_driver is None:
            self._neo4j_driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                auth=(
                    os.getenv("NEO4J_USER", "neo4j"),
                    os.getenv("NEO4J_PASSWORD")
                )
            )
            logger.info("Neo4j driver initialized")
        return self._neo4j_driver

    @property
    def semantic_frozen(self) -> bool:
        """Check if semantic memory is frozen."""
        return self._semantic_frozen

    def freeze_semantic(self):
        """Freeze semantic memory (Frozen Voice mode)."""
        self._semantic_frozen = True
        logger.warning("SEMANTIC MEMORY FROZEN - Voice preservation mode activated")

    def unfreeze_semantic(self):
        """Unfreeze semantic memory (for corrections only)."""
        self._semantic_frozen = False
        logger.warning("SEMANTIC MEMORY UNFROZEN - Use with caution")

    def close(self):
        """Close all connections."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
            logger.info("Neo4j connection closed")


# Global database manager
db = DatabaseManager()


# =============================================================================
# MCP Server Definition
# =============================================================================

mcp = FastMCP(
    name="ubik-hippocampal",
    version="1.0.0",
    instructions="Memory and identity services for the Ubik digital legacy system"
)


# =============================================================================
# Episodic Memory Tools
# =============================================================================

@mcp.tool()
async def store_episodic(memory: EpisodicMemory) -> Dict[str, Any]:
    """
    Store a new episodic memory.

    Episodic memories are always writable (Living Mind principle).
    They represent personal experiences, conversations, and events.

    Args:
        memory: The episodic memory to store

    Returns:
        Confirmation with memory ID
    """
    try:
        collection = db.chroma.get_collection("ubik_episodic")

        # Generate ID and timestamp
        memory_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = memory.timestamp or datetime.now().isoformat() + "Z"

        # Store the memory
        collection.add(
            documents=[memory.content],
            metadatas=[{
                "type": memory.memory_type,
                "timestamp": timestamp,
                "emotional_valence": memory.emotional_valence,
                "importance": memory.importance,
                "participants": memory.participants,
                "themes": memory.themes,
                "source_file": memory.source_file or "",
                "created_at": datetime.now().isoformat()
            }],
            ids=[memory_id]
        )

        logger.info(f"Stored episodic memory: {memory_id}")

        return {
            "status": "success",
            "memory_id": memory_id,
            "timestamp": timestamp,
            "message": "Episodic memory stored successfully"
        }

    except Exception as e:
        logger.error(f"Error storing episodic memory: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
async def query_episodic(params: QueryParams) -> Dict[str, Any]:
    """
    Query episodic memories using semantic search.

    Args:
        params: Query parameters including search text and filters

    Returns:
        Matching memories with metadata
    """
    try:
        collection = db.chroma.get_collection("ubik_episodic")

        # Build query kwargs
        query_kwargs = {
            "query_texts": [params.query],
            "n_results": params.n_results
        }

        # Add filters if provided
        if params.filters:
            query_kwargs["where"] = params.filters

        results = collection.query(**query_kwargs)

        # Format results
        memories = []
        if results['documents'] and results['documents'][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0] if results['distances'] else [None] * len(results['documents'][0])
            )):
                memories.append({
                    "id": results['ids'][0][i],
                    "content": doc,
                    "metadata": meta,
                    "relevance_score": 1 - dist if dist else None
                })

        logger.info(f"Episodic query returned {len(memories)} results for: {params.query[:50]}...")

        return {
            "status": "success",
            "query": params.query,
            "count": len(memories),
            "memories": memories
        }

    except Exception as e:
        logger.error(f"Error querying episodic memories: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# =============================================================================
# Semantic Memory Tools
# =============================================================================

@mcp.tool()
async def store_semantic(knowledge: SemanticKnowledge, force: bool = False) -> Dict[str, Any]:
    """
    Store semantic knowledge (beliefs, values, preferences).

    Respects the Frozen Voice principle - once frozen, semantic memory
    cannot be modified without explicit force flag.

    Args:
        knowledge: The semantic knowledge to store
        force: Override freeze protection (use with extreme caution)

    Returns:
        Confirmation with knowledge ID
    """
    try:
        # Check freeze state
        if db.semantic_frozen and not force:
            return {
                "status": "blocked",
                "message": "Semantic memory is FROZEN (Frozen Voice mode). Use force=True to override.",
                "frozen_since": "voice_training_complete"
            }

        collection = db.chroma.get_collection("ubik_semantic")

        # Generate ID
        knowledge_id = f"sem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Store the knowledge
        collection.add(
            documents=[knowledge.content],
            metadatas=[{
                "type": knowledge.knowledge_type,
                "category": knowledge.category,
                "confidence": knowledge.confidence,
                "stability": knowledge.stability,
                "source": knowledge.source,
                "frozen": db.semantic_frozen,
                "created_at": datetime.now().isoformat()
            }],
            ids=[knowledge_id]
        )

        logger.info(f"Stored semantic knowledge: {knowledge_id} (force={force})")

        return {
            "status": "success",
            "knowledge_id": knowledge_id,
            "frozen_state": db.semantic_frozen,
            "message": "Semantic knowledge stored successfully"
        }

    except Exception as e:
        logger.error(f"Error storing semantic knowledge: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
async def query_semantic(params: QueryParams) -> Dict[str, Any]:
    """
    Query semantic knowledge using semantic search.

    Args:
        params: Query parameters including search text and filters

    Returns:
        Matching knowledge entries with metadata
    """
    try:
        collection = db.chroma.get_collection("ubik_semantic")

        # Build query kwargs
        query_kwargs = {
            "query_texts": [params.query],
            "n_results": params.n_results
        }

        # Add filters if provided
        if params.filters:
            query_kwargs["where"] = params.filters

        results = collection.query(**query_kwargs)

        # Format results
        knowledge_items = []
        if results['documents'] and results['documents'][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0] if results['distances'] else [None] * len(results['documents'][0])
            )):
                knowledge_items.append({
                    "id": results['ids'][0][i],
                    "content": doc,
                    "metadata": meta,
                    "relevance_score": 1 - dist if dist else None
                })

        logger.info(f"Semantic query returned {len(knowledge_items)} results for: {params.query[:50]}...")

        return {
            "status": "success",
            "query": params.query,
            "count": len(knowledge_items),
            "frozen_state": db.semantic_frozen,
            "knowledge": knowledge_items
        }

    except Exception as e:
        logger.error(f"Error querying semantic knowledge: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# =============================================================================
# Identity Graph Tools (Neo4j)
# =============================================================================

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

        with db.neo4j.session() as session:
            result = session.run("""
                MATCH path = (start:Concept {name: $concept})-[*1..$depth]-(related)
                RETURN
                    start.name as source,
                    [r in relationships(path) | type(r)] as relationship_types,
                    [n in nodes(path) | {name: n.name, type: labels(n)[0], properties: properties(n)}] as nodes,
                    length(path) as distance
                ORDER BY distance
                LIMIT 50
            """, concept=concept, depth=depth)

            paths = []
            for record in result:
                paths.append({
                    "source": record["source"],
                    "relationships": record["relationship_types"],
                    "nodes": record["nodes"],
                    "distance": record["distance"]
                })

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


@mcp.tool()
async def update_identity_graph(relation: IdentityRelation) -> Dict[str, Any]:
    """
    Add or update a relationship in the identity graph.

    Creates nodes if they don't exist and establishes the relationship.

    Args:
        relation: The relationship to create/update

    Returns:
        Confirmation of the graph update
    """
    try:
        with db.neo4j.session() as session:
            result = session.run("""
                MERGE (from:Concept {name: $from_concept})
                MERGE (to:Concept {name: $to_concept})
                MERGE (from)-[r:$rel_type]->(to)
                SET r.weight = $weight,
                    r.context = $context,
                    r.updated_at = datetime()
                RETURN from.name as source, type(r) as relation, to.name as target
            """.replace("$rel_type", relation.relation_type),
                from_concept=relation.from_concept,
                to_concept=relation.to_concept,
                weight=relation.weight,
                context=relation.context or ""
            )

            record = result.single()

        logger.info(f"Identity graph updated: {relation.from_concept} -[{relation.relation_type}]-> {relation.to_concept}")

        return {
            "status": "success",
            "relation": {
                "source": record["source"],
                "type": record["relation"],
                "target": record["target"]
            },
            "message": "Identity graph updated successfully"
        }

    except Exception as e:
        logger.error(f"Error updating identity graph: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# =============================================================================
# Administrative Tools
# =============================================================================

@mcp.tool()
async def get_memory_stats() -> Dict[str, Any]:
    """
    Get statistics about the memory system.

    Returns:
        Statistics for episodic, semantic, and graph stores
    """
    try:
        stats = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "semantic_frozen": db.semantic_frozen
        }

        # ChromaDB stats
        try:
            episodic = db.chroma.get_collection("ubik_episodic")
            semantic = db.chroma.get_collection("ubik_semantic")
            stats["chromadb"] = {
                "episodic_count": episodic.count(),
                "semantic_count": semantic.count()
            }
        except Exception as e:
            stats["chromadb"] = {"error": str(e)}

        # Neo4j stats
        try:
            with db.neo4j.session() as session:
                result = session.run("""
                    MATCH (n)
                    WITH count(n) as nodes
                    MATCH ()-[r]->()
                    RETURN nodes, count(r) as relationships
                """)
                record = result.single()
                stats["neo4j"] = {
                    "node_count": record["nodes"] if record else 0,
                    "relationship_count": record["relationships"] if record else 0
                }
        except Exception as e:
            stats["neo4j"] = {"error": str(e)}

        return stats

    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
async def freeze_voice() -> Dict[str, Any]:
    """
    Freeze semantic memory (activate Frozen Voice mode).

    Once frozen, semantic memories cannot be modified without
    explicit force override. Use after voice training is complete.

    Returns:
        Confirmation of freeze activation
    """
    db.freeze_semantic()
    return {
        "status": "success",
        "semantic_frozen": True,
        "message": "FROZEN VOICE mode activated. Semantic memory is now read-only.",
        "timestamp": datetime.now().isoformat()
    }


@mcp.tool()
async def unfreeze_voice(confirmation: str) -> Dict[str, Any]:
    """
    Unfreeze semantic memory (requires confirmation).

    This should only be used for critical corrections.

    Args:
        confirmation: Must be exactly "I UNDERSTAND THIS MAY AFFECT VOICE PRESERVATION"

    Returns:
        Result of unfreeze attempt
    """
    expected = "I UNDERSTAND THIS MAY AFFECT VOICE PRESERVATION"

    if confirmation != expected:
        return {
            "status": "blocked",
            "message": f"Confirmation required. Please provide exact phrase: '{expected}'"
        }

    db.unfreeze_semantic()
    return {
        "status": "success",
        "semantic_frozen": False,
        "message": "WARNING: Semantic memory unfrozen. Voice preservation may be affected.",
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# Server Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", 8080))

    logger.info(f"Starting Ubik Hippocampal MCP Server on {host}:{port}")

    # Run the server
    uvicorn.run(
        mcp.http_app(),
        host=host,
        port=port,
        log_level="info"
    )
```

### hippocampal/setup_chromadb.py

```python
#!/usr/bin/env python3
"""
ChromaDB Collection Setup for Ubik Hippocampal Node

Creates two primary collections:
1. ubik_episodic - Personal experiences, conversations, time-bound memories
2. ubik_semantic - Conceptual knowledge, beliefs, values, preferences
"""

import chromadb
from chromadb.utils import embedding_functions
import time
import sys


def create_collections():
    """Initialize ChromaDB collections for Ubik memory system."""

    print("=" * 60)
    print("Ubik ChromaDB Collection Setup")
    print("=" * 60)

    # Wait for ChromaDB to be ready
    max_retries = 10
    retry_count = 0
    client = None

    while retry_count < max_retries:
        try:
            # Connect to ChromaDB with authentication (chromadb 1.x API)
            client = chromadb.HttpClient(
                host="localhost",
                port=8001,
                headers={"Authorization": "Bearer ubik_chroma_token_2024"}
            )
            # Test connection
            client.heartbeat()
            print(f"\n✓ Connected to ChromaDB")
            break
        except Exception as e:
            retry_count += 1
            print(f"  Waiting for ChromaDB... ({retry_count}/{max_retries}) - {e}")
            time.sleep(3)

    if client is None:
        print("✗ Failed to connect to ChromaDB")
        sys.exit(1)

    # Use Sentence Transformers embeddings
    try:
        from sentence_transformers import SentenceTransformer
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        print("✓ Using SentenceTransformer embeddings (all-MiniLM-L6-v2)")
    except Exception as e:
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        print("⚠ Using default embeddings (all-MiniLM-L6-v2 not available)")

    # =========================================================================
    # Collection 1: Episodic Memory
    # =========================================================================
    print("\n" + "-" * 40)
    print("Creating ubik_episodic collection...")

    try:
        # Delete if exists (for clean setup)
        try:
            client.delete_collection("ubik_episodic")
            print("  Deleted existing ubik_episodic collection")
        except:
            pass

        episodic = client.create_collection(
            name="ubik_episodic",
            embedding_function=embedding_fn,
            metadata={
                "description": "Episodic memories - personal experiences, conversations, events",
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 128,
                "hnsw:search_ef": 64,
                "hnsw:M": 16,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "version": "1.0.0",
                "node": "hippocampal"
            }
        )
        print(f"  ✓ Created ubik_episodic collection")

        # Add sample episodic memory
        episodic.add(
            documents=[
                "Today I wrote my first letter to my future grandchildren about why I started the Ubik project.",
                "In therapy today, we discussed how my values around authenticity shaped my career decisions.",
                "Family meeting: We talked about summer vacation plans and everyone's school achievements."
            ],
            metadatas=[
                {
                    "type": "letter",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "emotional_valence": "positive",
                    "importance": 0.9,
                    "participants": "gines,grandchildren",
                    "themes": "legacy,purpose,family"
                },
                {
                    "type": "therapy_session",
                    "timestamp": "2024-01-14T14:00:00Z",
                    "emotional_valence": "reflective",
                    "importance": 0.8,
                    "participants": "gines,therapist",
                    "themes": "values,career,authenticity"
                },
                {
                    "type": "family_meeting",
                    "timestamp": "2024-01-13T18:00:00Z",
                    "emotional_valence": "positive",
                    "importance": 0.7,
                    "participants": "family",
                    "themes": "planning,celebration,connection"
                }
            ],
            ids=["ep_sample_001", "ep_sample_002", "ep_sample_003"]
        )
        print(f"  ✓ Added 3 sample episodic memories")

    except Exception as e:
        print(f"  ✗ Error creating ubik_episodic: {e}")
        sys.exit(1)

    # =========================================================================
    # Collection 2: Semantic Memory
    # =========================================================================
    print("\n" + "-" * 40)
    print("Creating ubik_semantic collection...")

    try:
        # Delete if exists (for clean setup)
        try:
            client.delete_collection("ubik_semantic")
            print("  Deleted existing ubik_semantic collection")
        except:
            pass

        semantic = client.create_collection(
            name="ubik_semantic",
            embedding_function=embedding_fn,
            metadata={
                "description": "Semantic knowledge - concepts, beliefs, values, preferences",
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 128,
                "hnsw:search_ef": 64,
                "hnsw:M": 16,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "version": "1.0.0",
                "node": "hippocampal"
            }
        )
        print(f"  ✓ Created ubik_semantic collection")

        # Add sample semantic knowledge
        semantic.add(
            documents=[
                "I believe authenticity is the foundation of meaningful relationships. Being genuine matters more than being perfect.",
                "My core value: Family legacy transcends material possessions - it's about values, wisdom, and love passed down.",
                "Communication preference: I prefer thoughtful, reflective conversations over quick exchanges. Depth over breadth.",
                "Philosophical stance: Identity persists through psychological continuity - memories, values, and intentions form the self."
            ],
            metadatas=[
                {
                    "type": "belief",
                    "category": "relationships",
                    "confidence": 0.95,
                    "stability": "core",
                    "source": "life_experience",
                    "frozen": False
                },
                {
                    "type": "value",
                    "category": "family",
                    "confidence": 0.98,
                    "stability": "core",
                    "source": "reflection",
                    "frozen": False
                },
                {
                    "type": "preference",
                    "category": "communication",
                    "confidence": 0.85,
                    "stability": "stable",
                    "source": "self_observation",
                    "frozen": False
                },
                {
                    "type": "belief",
                    "category": "philosophy",
                    "confidence": 0.90,
                    "stability": "stable",
                    "source": "reading_parfitian_identity",
                    "frozen": False
                }
            ],
            ids=["sem_sample_001", "sem_sample_002", "sem_sample_003", "sem_sample_004"]
        )
        print(f"  ✓ Added 4 sample semantic memories")

    except Exception as e:
        print(f"  ✗ Error creating ubik_semantic: {e}")
        sys.exit(1)

    # =========================================================================
    # Verification
    # =========================================================================
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    collections = client.list_collections()
    print(f"\nActive collections: {len(collections)}")
    for col in collections:
        count = col.count()
        print(f"  • {col.name}: {count} documents")

    # Test query
    print("\n" + "-" * 40)
    print("Test Query: 'family values and legacy'")
    print("-" * 40)

    results = client.get_collection("ubik_semantic").query(
        query_texts=["family values and legacy"],
        n_results=2
    )

    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n  Result {i+1}:")
        print(f"    Type: {meta.get('type')}")
        print(f"    Category: {meta.get('category')}")
        print(f"    Text: {doc[:80]}...")

    print("\n" + "=" * 60)
    print("✓ ChromaDB setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    create_collections()
```

### hippocampal/init_neo4j_schema.py

```python
#!/usr/bin/env python3
"""
Neo4j Schema Initialization for Ubik Parfitian Identity Model

This schema implements a graph structure based on Derek Parfit's
theory of personal identity, emphasizing psychological continuity
and the "Relation R" (psychological connectedness + continuity).

Node Types:
- Concept: Abstract concepts, beliefs, values
- Memory: Episodic memory anchors (links to ChromaDB)
- Trait: Personality traits and characteristics
- Relationship: Interpersonal relationships
- TimeSlice: Temporal snapshots of identity state

Relationship Types:
- INFLUENCES: One concept affects another
- DERIVES_FROM: Concept originates from another
- SUPPORTS: Provides evidence/support
- CONFLICTS_WITH: Creates tension/contradiction
- CONNECTS_TO: Parfitian psychological connection
- CONTINUES_AS: Temporal continuity link
- REMEMBERS: Memory connection
- VALUES: Association with value
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import logging

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ubik.schema")


def init_schema(driver):
    """Initialize the Neo4j schema with constraints and indexes."""

    with driver.session() as session:

        # =====================================================================
        # Constraints (ensure uniqueness)
        # =====================================================================

        constraints = [
            ("concept_name", "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE"),
            ("memory_id", "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.chromadb_id IS UNIQUE"),
            ("trait_name", "CREATE CONSTRAINT trait_name IF NOT EXISTS FOR (t:Trait) REQUIRE t.name IS UNIQUE"),
            ("timeslice_id", "CREATE CONSTRAINT timeslice_id IF NOT EXISTS FOR (ts:TimeSlice) REQUIRE ts.timestamp IS UNIQUE"),
        ]

        logger.info("Creating constraints...")
        for name, query in constraints:
            try:
                session.run(query)
                logger.info(f"  ✓ Created constraint: {name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"  • Constraint exists: {name}")
                else:
                    logger.error(f"  ✗ Error creating {name}: {e}")

        # =====================================================================
        # Indexes (improve query performance)
        # =====================================================================

        indexes = [
            ("concept_fulltext", """
                CREATE FULLTEXT INDEX concept_fulltext IF NOT EXISTS
                FOR (c:Concept) ON EACH [c.name, c.description]
            """),
            ("concept_category", "CREATE INDEX concept_category IF NOT EXISTS FOR (c:Concept) ON (c.category)"),
            ("concept_stability", "CREATE INDEX concept_stability IF NOT EXISTS FOR (c:Concept) ON (c.stability)"),
            ("trait_type", "CREATE INDEX trait_type IF NOT EXISTS FOR (t:Trait) ON (t.trait_type)"),
            ("memory_type", "CREATE INDEX memory_type IF NOT EXISTS FOR (m:Memory) ON (m.memory_type)"),
            ("rel_weight", "CREATE INDEX rel_weight IF NOT EXISTS FOR ()-[r:INFLUENCES]-() ON (r.weight)"),
        ]

        logger.info("\nCreating indexes...")
        for name, query in indexes:
            try:
                session.run(query)
                logger.info(f"  ✓ Created index: {name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"  • Index exists: {name}")
                else:
                    logger.error(f"  ✗ Error creating {name}: {e}")


def create_core_identity_nodes(driver):
    """Create the foundational identity structure."""

    with driver.session() as session:
        logger.info("\nCreating core identity nodes...")

        # Core Identity Node (The "Self")
        session.run("""
            MERGE (self:Concept:CoreIdentity {name: 'Self'})
            SET self.description = 'The central identity node representing Gines',
                self.stability = 'core',
                self.frozen = false,
                self.created_at = datetime(),
                self.parfitian_note = 'The Self is not a simple, unified entity but a bundle of psychological connections'
        """)
        logger.info("  ✓ Created Self node")

        # Core Values
        core_values = [
            {"name": "Authenticity", "description": "Being genuine and true to oneself in all interactions", "category": "personal_virtue", "importance": 0.95},
            {"name": "Family Legacy", "description": "Preserving and transmitting wisdom, values, and love to future generations", "category": "life_purpose", "importance": 0.98},
            {"name": "Intellectual Curiosity", "description": "Deep drive to understand, learn, and explore ideas", "category": "personality", "importance": 0.85},
            {"name": "Meaningful Connection", "description": "Valuing deep, authentic relationships over superficial interactions", "category": "relationships", "importance": 0.90},
            {"name": "Psychological Continuity", "description": "Identity persists through connected chains of memory, intention, and character", "category": "philosophy", "importance": 0.88}
        ]

        for value in core_values:
            session.run("""
                MERGE (v:Concept:Value {name: $name})
                SET v.description = $description,
                    v.category = $category,
                    v.importance = $importance,
                    v.stability = 'core',
                    v.frozen = false,
                    v.created_at = datetime()
                WITH v
                MATCH (self:CoreIdentity {name: 'Self'})
                MERGE (self)-[r:VALUES]->(v)
                SET r.weight = $importance,
                    r.established = datetime()
            """, **value)
            logger.info(f"  ✓ Created value: {value['name']}")

        # Personality Traits
        traits = [
            {"name": "Reflective", "description": "Tendency toward deep introspection and thoughtful analysis", "trait_type": "cognitive_style", "strength": 0.85},
            {"name": "Warm", "description": "Natural warmth and genuine care in interpersonal interactions", "trait_type": "interpersonal", "strength": 0.80},
            {"name": "Methodical", "description": "Preference for systematic, organized approaches to problems", "trait_type": "behavioral", "strength": 0.75},
            {"name": "Creative", "description": "Ability to generate novel ideas and see unusual connections", "trait_type": "cognitive_style", "strength": 0.70}
        ]

        for trait in traits:
            session.run("""
                MERGE (t:Trait {name: $name})
                SET t.description = $description,
                    t.trait_type = $trait_type,
                    t.strength = $strength,
                    t.stability = 'stable',
                    t.created_at = datetime()
                WITH t
                MATCH (self:CoreIdentity {name: 'Self'})
                MERGE (self)-[r:HAS_TRAIT]->(t)
                SET r.strength = $strength
            """, **trait)
            logger.info(f"  ✓ Created trait: {trait['name']}")

        # Key Relationships
        relationships = [
            {"name": "Grandchildren", "relationship_type": "family", "description": "Future grandchildren - the intended recipients of Ubik's wisdom", "importance": 0.99},
            {"name": "Spouse", "relationship_type": "family", "description": "Life partner and closest confidant", "importance": 0.98},
            {"name": "Children", "relationship_type": "family", "description": "Children - bridge between self and grandchildren", "importance": 0.97}
        ]

        for rel in relationships:
            session.run("""
                MERGE (r:Relationship {name: $name})
                SET r.relationship_type = $relationship_type,
                    r.description = $description,
                    r.importance = $importance,
                    r.created_at = datetime()
                WITH r
                MATCH (self:CoreIdentity {name: 'Self'})
                MERGE (self)-[c:CONNECTED_TO]->(r)
                SET c.connection_type = $relationship_type,
                    c.strength = $importance
            """, **rel)
            logger.info(f"  ✓ Created relationship: {rel['name']}")


def main():
    """Main entry point."""

    print("=" * 60)
    print("Ubik Neo4j Schema Initialization")
    print("Parfitian Identity Model")
    print("=" * 60)

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        driver.verify_connectivity()
        logger.info(f"✓ Connected to Neo4j at {uri}")

        init_schema(driver)
        create_core_identity_nodes(driver)

        print("\n" + "=" * 60)
        print("✓ Schema initialization complete!")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Error during schema initialization: {e}")
        raise
    finally:
        driver.close()


if __name__ == "__main__":
    main()
```

### hippocampal/health_check.py

```python
#!/usr/bin/env python3
"""
Ubik Hippocampal Node - Health Check Script

Validates all components of the Hippocampal Node:
- Docker services (Neo4j, ChromaDB)
- MCP Server
- Tailscale connectivity
- Data integrity
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_result(name, success, details=""):
    """Print a test result."""
    status = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{status}{reset} {name}")
    if details:
        print(f"    └─ {details}")


def check_docker():
    """Check Docker and container status."""
    print_header("Docker Services")

    all_ok = True

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print_result("Docker daemon", result.returncode == 0)
        all_ok = all_ok and result.returncode == 0
    except Exception as e:
        print_result("Docker daemon", False, str(e))
        return False

    containers = {
        "ubik-neo4j": "Neo4j database",
        "ubik-chromadb": "ChromaDB vector store"
    }

    for container, description in containers.items():
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}}", container],
                capture_output=True,
                text=True,
                timeout=10
            )
            status = result.stdout.strip()
            is_running = status == "running"
            print_result(f"{description} ({container})", is_running, f"Status: {status}")
            all_ok = all_ok and is_running
        except Exception as e:
            print_result(f"{description}", False, str(e))
            all_ok = False

    return all_ok


def check_neo4j():
    """Check Neo4j connectivity and data."""
    print_header("Neo4j Database")

    try:
        from neo4j import GraphDatabase

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        driver = GraphDatabase.driver(uri, auth=(user, password))

        driver.verify_connectivity()
        print_result("Connection", True, f"Connected to {uri}")

        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            print_result("Data integrity", count > 0, f"{count} nodes in graph")

            result = session.run("MATCH (s:CoreIdentity {name: 'Self'}) RETURN s")
            has_self = result.single() is not None
            print_result("Core identity node", has_self)

        driver.close()
        return True

    except Exception as e:
        print_result("Neo4j check", False, str(e))
        return False


def check_chromadb():
    """Check ChromaDB connectivity and collections."""
    print_header("ChromaDB Vector Store")

    try:
        import chromadb

        client = chromadb.HttpClient(
            host=os.getenv("CHROMADB_HOST", "localhost"),
            port=int(os.getenv("CHROMADB_PORT", 8001)),
            headers={"Authorization": f"Bearer {os.getenv('CHROMADB_TOKEN')}"}
        )

        client.heartbeat()
        print_result("Connection", True)

        collections = client.list_collections()
        collection_names = [c.name for c in collections]

        expected = ["ubik_episodic", "ubik_semantic"]
        for name in expected:
            exists = name in collection_names
            if exists:
                count = client.get_collection(name).count()
                print_result(f"Collection: {name}", True, f"{count} documents")
            else:
                print_result(f"Collection: {name}", False, "Not found")

        if "ubik_semantic" in collection_names:
            results = client.get_collection("ubik_semantic").query(
                query_texts=["family values"],
                n_results=1
            )
            has_results = len(results['documents'][0]) > 0
            print_result("Query test", has_results, "Semantic search working")

        return True

    except Exception as e:
        print_result("ChromaDB check", False, str(e))
        return False


def check_mcp_server():
    """Check if MCP server is running and responsive."""
    print_header("MCP Server")

    try:
        import httpx

        host = os.getenv("MCP_HOST", "localhost")
        port = os.getenv("MCP_PORT", "8080")
        url = f"http://{host}:{port}"

        response = httpx.get(f"{url}/mcp", timeout=5)
        is_responding = response.status_code in [200, 400, 404, 406]
        print_result("Server status", is_responding, f"HTTP {response.status_code} (MCP server running)")

        return is_responding

    except Exception as e:
        print_result("MCP server", False, str(e))
        return False


def check_tailscale():
    """Check Tailscale connectivity."""
    print_header("Tailscale Networking")

    try:
        tailscale_cmd = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
        if not os.path.exists(tailscale_cmd):
            tailscale_cmd = "tailscale"

        result = subprocess.run(
            [tailscale_cmd, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            status = json.loads(result.stdout)

            self_info = status.get("Self", {})
            hostname = self_info.get("HostName", "unknown")
            online = self_info.get("Online", False)

            print_result("Tailscale status", online, f"Hostname: {hostname}")

            # Check for somatic node by IP
            peers = status.get("Peer", {})
            somatic_ip = "100.79.166.114"
            somatic_dns = "adrian-wsl"

            somatic_peer = None
            for peer in peers.values():
                peer_ips = peer.get("TailscaleIPs", [])
                if somatic_ip in peer_ips:
                    somatic_peer = peer
                    break

            if somatic_peer:
                peer_online = somatic_peer.get("Online", False)
                peer_hostname = somatic_peer.get("HostName", "unknown")
                print_result(f"Somatic node ({somatic_dns})", peer_online, f"Host: {peer_hostname}, IP: {somatic_ip}")
            else:
                print_result(f"Somatic node ({somatic_dns})", False, "Not found in Tailscale network")

            return online
        else:
            print_result("Tailscale", False, "Not running or not logged in")
            return False

    except Exception as e:
        print_result("Tailscale", False, str(e))
        return False


def run_all_checks():
    """Run all health checks."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " UBIK HIPPOCAMPAL NODE - HEALTH CHECK ".center(58) + "║")
    print("║" + f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    results = {
        "docker": check_docker(),
        "neo4j": check_neo4j(),
        "chromadb": check_chromadb(),
        "mcp_server": check_mcp_server(),
        "tailscale": check_tailscale()
    }

    print_header("Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for component, status in results.items():
        print_result(component.replace("_", " ").title(), status)

    print("\n" + "-" * 60)
    if passed == total:
        print(f"  \033[92m✓ All {total} checks passed!\033[0m")
        print("  Hippocampal Node is fully operational.")
    else:
        print(f"  \033[93m⚠ {passed}/{total} checks passed\033[0m")
        print("  Review failed components above.")

    print("\n")
    return passed == total


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
```

### hippocampal/run_mcp.sh

```bash
#!/bin/bash
# Ubik Hippocampal Node - MCP Server Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source "/Volumes/990PRO 4T/DeepSeek/venv/bin/activate"

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    exit 1
fi

# Export environment variables
export $(grep -v '^#' .env | xargs)

# Run the MCP server
echo "Starting Ubik Hippocampal MCP Server..."
python mcp_server.py
```

### hippocampal/quick_test.sh

```bash
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
```

---

## Health Check Results

**Last Run: 2026-01-17 22:53:35**

```
╔══════════════════════════════════════════════════════════╗
║           UBIK HIPPOCAMPAL NODE - HEALTH CHECK           ║
║                   2026-01-17 22:53:35                    ║
╚══════════════════════════════════════════════════════════╝

============================================================
 Docker Services
============================================================
  ✓ Docker daemon
  ✓ Neo4j database (ubik-neo4j)
    └─ Status: running
  ✓ ChromaDB vector store (ubik-chromadb)
    └─ Status: running

============================================================
 Neo4j Database
============================================================
  ✓ Connection
    └─ Connected to bolt://localhost:7687
  ✓ Data integrity
    └─ 18 nodes in graph
  ✓ Core identity node

============================================================
 ChromaDB Vector Store
============================================================
  ✓ Connection
  ✓ Collection: ubik_episodic
    └─ 3 documents
  ✓ Collection: ubik_semantic
    └─ 4 documents
  ✓ Query test
    └─ Semantic search working

============================================================
 MCP Server
============================================================
  ✓ Server status
    └─ HTTP 406 (MCP server running)

============================================================
 Tailscale Networking
============================================================
  ✓ Tailscale status
    └─ Hostname: MiniM4 2025
  ✓ Somatic node (adrian-wsl)
    └─ Host: Adrian, IP: 100.79.166.114

============================================================
 Summary
============================================================
  ✓ Docker
  ✓ Neo4J
  ✓ Chromadb
  ✓ Mcp Server
  ✓ Tailscale

------------------------------------------------------------
  ✓ All 5 checks passed!
  Hippocampal Node is fully operational.
```

---

## Setup Instructions

### Prerequisites

1. **Docker Desktop** installed and running
2. **Python 3.10+** with pip
3. **Tailscale** installed and authenticated
4. Virtual environment with required packages

### Step 1: Start Docker Services

```bash
cd "/Volumes/990PRO 4T/UBIK"
docker compose up -d
```

### Step 2: Initialize ChromaDB Collections

```bash
cd "/Volumes/990PRO 4T/UBIK/hippocampal"
source "/Volumes/990PRO 4T/DeepSeek/venv/bin/activate"
python setup_chromadb.py
```

### Step 3: Initialize Neo4j Schema

```bash
python init_neo4j_schema.py
```

### Step 4: Start MCP Server

```bash
./run_mcp.sh
```

### Step 5: Verify Installation

```bash
python health_check.py
```

---

## Usage Guide

### MCP Tools Available

| Tool | Description |
|------|-------------|
| `store_episodic` | Store a new episodic memory (experience, conversation, event) |
| `query_episodic` | Search episodic memories using semantic search |
| `store_semantic` | Store semantic knowledge (beliefs, values, preferences) |
| `query_semantic` | Search semantic knowledge |
| `get_identity_context` | Traverse the Parfitian identity graph from a concept |
| `update_identity_graph` | Add/update relationships in the identity graph |
| `get_memory_stats` | Get statistics about all memory stores |
| `freeze_voice` | Activate Frozen Voice mode (locks semantic memory) |
| `unfreeze_voice` | Deactivate Frozen Voice mode (requires confirmation) |

### Accessing Services

- **Neo4j Browser**: http://localhost:7474 (neo4j/ubik_memory_2024)
- **ChromaDB**: http://localhost:8001 (token: ubik_chroma_token_2024)
- **MCP Server**: http://localhost:8080

### Network Access (via Tailscale)

From the Somatic Node (adrian-wsl / 100.79.166.114):
- Neo4j Bolt: `bolt://100.103.242.91:7687`
- ChromaDB: `http://100.103.242.91:8001`
- MCP Server: `http://100.103.242.91:8080`

---

## Python Dependencies

```
chromadb>=1.0.0
neo4j>=5.0.0
fastmcp>=0.1.0
pydantic>=2.0.0
python-dotenv>=1.0.0
httpx>=0.24.0
uvicorn>=0.23.0
sentence-transformers>=2.2.0
```

---

*Generated by Claude Code - UBIK Hippocampal Node Documentation*
