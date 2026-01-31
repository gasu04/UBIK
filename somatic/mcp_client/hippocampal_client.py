#!/usr/bin/env python3
"""
Ubik Somatic Node - MCP Client for Hippocampal Node

This client connects to the Hippocampal Node's MCP server to access
memory and identity graph services.

Services:
  - MCP Server: port configured via HIPPOCAMPAL_MCP_PORT
  - ChromaDB: port configured via HIPPOCAMPAL_CHROMA_PORT
  - Neo4j: port 7687 (direct access if needed)
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

# Import settings from config module
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings

# Configure logging using settings
_settings = get_settings()
_log_dir = _settings.logging.dir
_log_dir.mkdir(parents=True, exist_ok=True)
_log_file = _log_dir / "mcp_client.log"
_log_level = getattr(logging, _settings.logging.level, logging.INFO)

logging.basicConfig(
    level=_log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ubik.mcp_client")


@dataclass
class MemoryResult:
    """A single memory search result."""

    id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: Optional[float] = None

    def __str__(self):
        return f"[{self.id}] {self.content[:80]}..."


class HippocampalClient:
    """
    Client for connecting to the Hippocampal Node's MCP server.

    Provides access to:
    - Episodic memory (experiences, conversations)
    - Semantic memory (beliefs, values, preferences)
    - Identity graph (Parfitian psychological connections)

    Example:
        async with HippocampalClient() as client:
            context = await client.get_rag_context("family values")
            print(context)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the Hippocampal client.

        Args:
            host: Hippocampal node IP/hostname (default: from settings)
            port: MCP server port (default: from settings)
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        self.host = host or settings.hippocampal.host
        self.port = port or settings.hippocampal.mcp_port
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._session_id: Optional[str] = None
        self._initialized: bool = False

        logger.info(f"HippocampalClient initialized: {self.base_url}")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for MCP requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-load async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    async def _initialize_session(self) -> bool:
        """Initialize MCP session with the server."""
        if self._initialized:
            return True

        try:
            response = await self.client.post(
                "/mcp",
                headers=self._get_headers(),
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "id": 1,
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "ubik-somatic", "version": "1.0.0"},
                    },
                },
            )
            response.raise_for_status()

            # Extract session ID from header
            self._session_id = response.headers.get("mcp-session-id")
            if not self._session_id:
                logger.error("No session ID received from server")
                return False

            # Send initialized notification
            await self.client.post(
                "/mcp",
                headers=self._get_headers(),
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )

            self._initialized = True
            logger.info(f"MCP session initialized: {self._session_id[:16]}...")
            return True

        except Exception as e:
            logger.error(f"Session initialization failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._session_id = None
        self._initialized = False
        logger.debug("HTTP client closed")

    async def __aenter__(self):
        await self._initialize_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # =========================================================================
    # Health & Status
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Hippocampal node is healthy and get memory stats.

        Returns:
            Dict with status, memory counts, and frozen state
        """
        result = await self._call_tool("get_memory_stats", {})
        if result.get("status") == "success":
            logger.info("Health check passed")
        return result

    async def is_connected(self) -> bool:
        """Quick connectivity check."""
        health = await self.health_check()
        return health.get("status") == "success"

    # =========================================================================
    # MCP Protocol Helpers
    # =========================================================================

    def _parse_sse_response(self, text: str) -> Dict[str, Any]:
        """Parse SSE response to extract JSON data."""
        import json

        # Handle SSE format: "event: message\r\ndata: {...}\r\n"
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    return json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

        # Fallback: try parsing as plain JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": f"Failed to parse response: {text[:200]}",
            }

    def _extract_tool_result(self, mcp_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the actual result from MCP tool response."""
        import json

        # MCP returns: {"result": {"content": [{"type": "text", "text": "..."}]}}
        content = mcp_result.get("content", [])
        if content and len(content) > 0:
            text_content = content[0].get("text", "{}")
            try:
                return json.loads(text_content)
            except json.JSONDecodeError:
                return {"status": "success", "raw": text_content}
        return mcp_result

    # =========================================================================
    # MCP Tool Caller (Generic)
    # =========================================================================

    async def _call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generic MCP tool caller.

        Args:
            tool_name: Name of the MCP tool to call
            arguments: Tool arguments

        Returns:
            Tool result or error dict
        """
        # Ensure session is initialized
        if not self._initialized:
            if not await self._initialize_session():
                return {
                    "status": "error",
                    "message": "Failed to initialize MCP session",
                }

        try:
            response = await self.client.post(
                "/mcp",
                headers=self._get_headers(),
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                },
            )
            response.raise_for_status()

            # Parse SSE response
            result = self._parse_sse_response(response.text)

            if "error" in result:
                logger.error(f"Tool {tool_name} error: {result['error']}")
                return {"status": "error", "message": str(result["error"])}

            if "result" in result:
                return self._extract_tool_result(result["result"])

            return result

        except Exception as e:
            logger.error(f"Tool call failed ({tool_name}): {e}")
            return {"status": "error", "message": str(e)}

    # =========================================================================
    # Episodic Memory
    # =========================================================================

    async def store_episodic(
        self,
        content: str,
        memory_type: str,
        timestamp: Optional[str] = None,
        emotional_valence: str = "neutral",
        importance: float = 0.5,
        participants: str = "gines",
        themes: str = "",
        source_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store a new episodic memory.

        Args:
            content: The memory content
            memory_type: Type (letter, therapy_session, family_meeting, conversation, event)
            timestamp: ISO timestamp (auto-generated if not provided)
            emotional_valence: Tone (positive, negative, neutral, reflective, mixed)
            importance: Importance score 0-1
            participants: Comma-separated participants
            themes: Comma-separated themes/tags
            source_file: Source document reference
        """
        return await self._call_tool(
            "store_episodic",
            {
                "memory": {
                    "content": content,
                    "memory_type": memory_type,
                    "timestamp": timestamp or datetime.now().isoformat() + "Z",
                    "emotional_valence": emotional_valence,
                    "importance": importance,
                    "participants": participants,
                    "themes": themes,
                    "source_file": source_file or "",
                }
            },
        )

    async def query_episodic(
        self, query: str, n_results: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryResult]:
        """
        Query episodic memories using semantic search.

        Args:
            query: Search query text
            n_results: Number of results to return (1-20)
            filters: Optional metadata filters
        """
        result = await self._call_tool(
            "query_episodic",
            {"params": {"query": query, "n_results": n_results, "filters": filters}},
        )

        if result.get("status") == "success":
            return [
                MemoryResult(
                    id=m["id"],
                    content=m["content"],
                    metadata=m["metadata"],
                    relevance_score=m.get("relevance_score"),
                )
                for m in result.get("memories", [])
            ]

        logger.warning("Episodic query returned no results")
        return []

    # =========================================================================
    # Semantic Memory
    # =========================================================================

    async def store_semantic(
        self,
        content: str,
        knowledge_type: str,
        category: str,
        confidence: float = 0.8,
        stability: str = "stable",
        source: str = "reflection",
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Store semantic knowledge.

        Note: Will be blocked if semantic memory is frozen (Frozen Voice mode)
        unless force=True is specified.

        Args:
            content: The knowledge/belief/value statement
            knowledge_type: Type (belief, value, preference, fact, opinion)
            category: Category (family, relationships, philosophy, etc.)
            confidence: Confidence level 0-1
            stability: Stability (core, stable, evolving)
            source: Source of knowledge
            force: Override freeze protection
        """
        return await self._call_tool(
            "store_semantic",
            {
                "knowledge": {
                    "content": content,
                    "knowledge_type": knowledge_type,
                    "category": category,
                    "confidence": confidence,
                    "stability": stability,
                    "source": source,
                },
                "force": force,
            },
        )

    async def query_semantic(
        self, query: str, n_results: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryResult]:
        """
        Query semantic knowledge using semantic search.
        """
        result = await self._call_tool(
            "query_semantic",
            {"params": {"query": query, "n_results": n_results, "filters": filters}},
        )

        if result.get("status") == "success":
            return [
                MemoryResult(
                    id=k["id"],
                    content=k["content"],
                    metadata=k["metadata"],
                    relevance_score=k.get("relevance_score"),
                )
                for k in result.get("knowledge", [])
            ]

        logger.warning("Semantic query returned no results")
        return []

    # =========================================================================
    # Identity Graph
    # =========================================================================

    async def get_identity_context(
        self, concept: str, depth: int = 2
    ) -> Dict[str, Any]:
        """
        Retrieve identity context from the Parfitian graph.

        Args:
            concept: The concept/entity to explore (e.g., "Self", "Family Legacy")
            depth: Relationship hops to traverse (1-3)
        """
        return await self._call_tool(
            "get_identity_context", {"concept": concept, "depth": depth}
        )

    async def update_identity_graph(
        self,
        from_concept: str,
        relation_type: str,
        to_concept: str,
        weight: float = 1.0,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add or update a relationship in the identity graph.

        Args:
            from_concept: Source concept/entity
            relation_type: Relationship type (INFLUENCES, DERIVES_FROM, SUPPORTS, CONFLICTS_WITH)
            to_concept: Target concept/entity
            weight: Relationship strength 0-1
            context: Additional context
        """
        return await self._call_tool(
            "update_identity_graph",
            {
                "relation": {
                    "from_concept": from_concept,
                    "relation_type": relation_type,
                    "to_concept": to_concept,
                    "weight": weight,
                    "context": context or "",
                }
            },
        )

    # =========================================================================
    # Voice Freeze Control
    # =========================================================================

    async def freeze_voice(self) -> Dict[str, Any]:
        """Activate Frozen Voice mode (semantic memory becomes read-only)."""
        return await self._call_tool("freeze_voice", {})

    async def unfreeze_voice(self, confirmation: str) -> Dict[str, Any]:
        """
        Deactivate Frozen Voice mode.

        Args:
            confirmation: Must be exactly "I UNDERSTAND THIS MAY AFFECT VOICE PRESERVATION"
        """
        return await self._call_tool("unfreeze_voice", {"confirmation": confirmation})

    async def is_frozen(self) -> bool:
        """Check if semantic memory is frozen."""
        stats = await self.health_check()
        return stats.get("semantic_frozen", False)

    # =========================================================================
    # RAG Integration Helpers
    # =========================================================================

    async def get_rag_context(
        self,
        query: str,
        episodic_results: int = 3,
        semantic_results: int = 3,
        include_identity: bool = True,
    ) -> str:
        """
        Get combined context for RAG augmentation.

        This is the PRIMARY method for enriching LLM prompts with
        personal context from the memory system.

        Args:
            query: The user's query or topic
            episodic_results: Number of episodic memories to include
            semantic_results: Number of semantic knowledge items
            include_identity: Whether to include identity graph context

        Returns:
            Formatted context string for LLM prompt injection
        """
        context_parts = []

        # Semantic knowledge (beliefs, values, preferences)
        if semantic_results > 0:
            semantic = await self.query_semantic(query, n_results=semantic_results)
            if semantic:
                context_parts.append("## Core Beliefs & Values")
                for item in semantic:
                    category = item.metadata.get("category", "general")
                    context_parts.append(f"- [{category}] {item.content}")

        # Episodic memories (experiences)
        if episodic_results > 0:
            episodic = await self.query_episodic(query, n_results=episodic_results)
            if episodic:
                context_parts.append("\n## Relevant Experiences")
                for item in episodic:
                    date = item.metadata.get("timestamp", "")[:10]
                    mem_type = item.metadata.get("type", "memory")
                    context_parts.append(f"- [{date}] ({mem_type}) {item.content}")

        # Identity graph connections
        if include_identity:
            # Query identity around core concepts
            for concept in ["Self", "Family Legacy"]:
                try:
                    identity = await self.get_identity_context(concept, depth=1)
                    if (
                        identity.get("status") == "success"
                        and identity.get("paths_found", 0) > 0
                    ):
                        context_parts.append(f"\n## Identity Context: {concept}")
                        for path in identity.get("context", [])[:3]:
                            nodes = path.get("nodes", [])
                            rels = path.get("relationships", [])
                            if len(nodes) >= 2:
                                rel_str = rels[0] if rels else "relates to"
                                context_parts.append(
                                    f"- {nodes[0].get('name')} --[{rel_str}]--> {nodes[-1].get('name')}"
                                )
                except Exception as e:
                    logger.debug(f"Identity lookup for {concept} failed: {e}")

        context = "\n".join(context_parts) if context_parts else ""
        logger.info(
            f"Generated RAG context: {len(context)} chars for query of length {len(query)}"
        )
        return context


# =============================================================================
# Convenience Functions
# =============================================================================


async def get_context_for_inference(query: str) -> str:
    """
    Quick function to get RAG context for inference.

    Usage:
        context = await get_context_for_inference("Tell me about family traditions")
        prompt = f"<context>\\n{context}\\n</context>\\n\\nUser: {query}\\nAssistant:"
    """
    async with HippocampalClient() as client:
        return await client.get_rag_context(query)


# =============================================================================
# CLI Testing
# =============================================================================


async def main():
    """CLI test interface."""
    print("=" * 60)
    print("UBIK Hippocampal Client - Connection Test")
    print("=" * 60)

    async with HippocampalClient() as client:
        print(f"\nConnecting to: {client.base_url}")

        # Health check
        print("\n[1/4] Health Check...")
        health = await client.health_check()

        if health.get("status") == "success":
            print(f"  ✓ Connected successfully")
            print(
                f"  • Episodic memories: {health.get('chromadb', {}).get('episodic_count', 'N/A')}"
            )
            print(
                f"  • Semantic knowledge: {health.get('chromadb', {}).get('semantic_count', 'N/A')}"
            )
            print(
                f"  • Identity nodes: {health.get('neo4j', {}).get('node_count', 'N/A')}"
            )
            print(f"  • Semantic frozen: {health.get('semantic_frozen', False)}")
        else:
            print(f"  ✗ Connection failed: {health.get('message')}")
            return

        # Semantic query test
        print("\n[2/4] Semantic Query Test...")
        semantic = await client.query_semantic(
            "family values authenticity", n_results=2
        )
        for r in semantic:
            print(f"  • {r.content[:70]}...")

        # Episodic query test
        print("\n[3/4] Episodic Query Test...")
        episodic = await client.query_episodic("letter grandchildren", n_results=2)
        for r in episodic:
            print(f"  • [{r.metadata.get('type', 'memory')}] {r.content[:60]}...")

        # RAG context test
        print("\n[4/4] RAG Context Generation Test...")
        context = await client.get_rag_context("authenticity and family legacy")
        print(f"  Generated context: {len(context)} characters")
        print(f"  Preview:\n{context[:300]}...")

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
