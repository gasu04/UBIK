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
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import chromadb
from neo4j import GraphDatabase
import neo4j.time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ubik.hippocampal")


# =============================================================================
# Neo4j Type Serialization
# =============================================================================

def serialize_neo4j_value(value: Any) -> Any:
    """
    Convert Neo4j-specific types to JSON-serializable Python types.

    Handles DateTime, Date, Time, Duration, and nested structures.
    """
    if value is None:
        return None

    # Handle Neo4j DateTime types
    if isinstance(value, neo4j.time.DateTime):
        return value.isoformat()
    if isinstance(value, neo4j.time.Date):
        return value.isoformat()
    if isinstance(value, neo4j.time.Time):
        return value.isoformat()
    if isinstance(value, neo4j.time.Duration):
        return str(value)

    # Handle collections recursively
    if isinstance(value, dict):
        return {k: serialize_neo4j_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_neo4j_value(item) for item in value]

    # Return primitives as-is
    return value


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
# Database Connections with Connection Pooling
# =============================================================================

class DatabaseManager:
    """
    Manages connections to ChromaDB and Neo4j with connection pooling.

    Connection Pool Configuration (via environment variables):
    - NEO4J_MAX_CONNECTION_POOL_SIZE: Max connections in pool (default: 50)
    - NEO4J_CONNECTION_ACQUISITION_TIMEOUT: Seconds to wait for connection (default: 60)
    - NEO4J_MAX_CONNECTION_LIFETIME: Max lifetime of pooled connection in seconds (default: 3600)
    - CHROMADB_TIMEOUT: Request timeout in seconds (default: 30)

    The Neo4j driver uses internal connection pooling. Connections are:
    - Reused across requests to avoid TCP/TLS handshake overhead
    - Automatically validated before use
    - Released back to pool after each session

    Usage:
        db = DatabaseManager()
        with db.neo4j.session() as session:
            result = session.run("MATCH (n) RETURN n LIMIT 1")
    """

    def __init__(self):
        self._chroma_client = None
        self._neo4j_driver = None
        self._semantic_frozen = False

        # Connection pool configuration from environment
        self._neo4j_pool_size = int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50"))
        self._neo4j_acquisition_timeout = float(os.getenv("NEO4J_CONNECTION_ACQUISITION_TIMEOUT", "60"))
        self._neo4j_max_lifetime = int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "3600"))
        self._chromadb_timeout = float(os.getenv("CHROMADB_TIMEOUT", "30"))

    @property
    def chroma(self):
        """
        Lazy-load ChromaDB client with configured timeout.

        The ChromaDB HttpClient uses httpx internally which supports
        connection pooling. The client is reused across requests.
        """
        if self._chroma_client is None:
            chroma_token = os.getenv("CHROMADB_TOKEN", "")
            self._chroma_client = chromadb.HttpClient(
                host=os.getenv("CHROMADB_HOST", "localhost"),
                port=int(os.getenv("CHROMADB_PORT", 8001)),
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                    chroma_client_auth_credentials=chroma_token,
                ) if chroma_token else chromadb.Settings(
                    anonymized_telemetry=False,
                )
            )
            logger.info(
                f"ChromaDB client initialized (timeout={self._chromadb_timeout}s)"
            )
        return self._chroma_client

    @property
    def neo4j(self):
        """
        Lazy-load Neo4j driver with connection pooling.

        Pool configuration:
        - max_connection_pool_size: Maximum connections in the pool
        - connection_acquisition_timeout: Time to wait for an available connection
        - max_connection_lifetime: Time before a connection is recycled

        Connections are automatically managed:
        - Idle connections are kept alive with keep-alive packets
        - Stale connections are detected and removed
        - New connections are created on demand up to max pool size
        """
        if self._neo4j_driver is None:
            self._neo4j_driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                auth=(
                    os.getenv("NEO4J_USER", "neo4j"),
                    os.getenv("NEO4J_PASSWORD", "")
                ),
                # Connection pool configuration
                max_connection_pool_size=self._neo4j_pool_size,
                connection_acquisition_timeout=self._neo4j_acquisition_timeout,
                max_connection_lifetime=self._neo4j_max_lifetime,
                # Keep connections alive
                keep_alive=True,
            )
            logger.info(
                f"Neo4j driver initialized with connection pooling "
                f"(pool_size={self._neo4j_pool_size}, "
                f"acquisition_timeout={self._neo4j_acquisition_timeout}s, "
                f"max_lifetime={self._neo4j_max_lifetime}s)"
            )
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

    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics for monitoring.

        Returns:
            Dictionary with pool configuration and status.
        """
        return {
            "neo4j": {
                "configured": self._neo4j_driver is not None,
                "max_pool_size": self._neo4j_pool_size,
                "acquisition_timeout": self._neo4j_acquisition_timeout,
                "max_connection_lifetime": self._neo4j_max_lifetime,
            },
            "chromadb": {
                "configured": self._chroma_client is not None,
                "timeout": self._chromadb_timeout,
            },
            "semantic_frozen": self._semantic_frozen,
        }

    def close(self):
        """Close all connections and release pool resources."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
            self._neo4j_driver = None
            logger.info("Neo4j connection pool closed")
        if self._chroma_client:
            # ChromaDB HttpClient doesn't need explicit close
            self._chroma_client = None
            logger.info("ChromaDB client released")


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
                    # Serialize Neo4j DateTime objects to ISO strings
                    serialized_nodes = serialize_neo4j_value(record["nodes"])
                    paths.append({
                        "source": record["source"],
                        "relationships": record["relationship_types"],
                        "nodes": serialized_nodes,
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
