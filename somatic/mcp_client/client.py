#!/usr/bin/env python3
"""
Production MCP client for the Hippocampal Node.

Composes resilience components (retry, circuit breaker, connection hygiene)
into a unified interface for the RAG service.

Key design decisions:
- Uses AsyncOpenAI-compatible patterns (async context manager)
- Applies retry with exponential backoff + jitter on transient failures
- Circuit breaker with Probe Latch prevents cascade failures
- Connection invalidation on errors ensures clean state
- Logs memory IDs only, never content (privacy by design)

Usage:
    async with HippocampalClientV2() as client:
        results = await client.query_semantic("family values")

    Or manual lifecycle:
        client = HippocampalClientV2()
        await client.connect()
        try:
            results = await client.query_semantic("family values")
        finally:
            await client.disconnect()

Dependencies:
    - httpx>=0.25.0
    - pydantic-settings>=2.0.0
"""

import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure parent package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings

from .connection import ManagedConnection
from .resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    RetryConfig,
    calculate_backoff_delay,
)

logger = logging.getLogger("ubik.mcp_client")


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class MemoryResult:
    """A single memory search result.

    Attributes:
        id: Unique identifier for the memory.
        content: The memory content text.
        memory_type: Category of memory (semantic, episodic, etc.).
        relevance_score: Similarity score from vector search (0.0-1.0).
        metadata: Additional metadata from the memory store.
    """

    id: str
    content: str
    memory_type: str
    relevance_score: float
    metadata: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryResult":
        """Construct from raw dictionary returned by MCP tool.

        Args:
            data: Raw result dictionary from Hippocampal Node.

        Returns:
            Parsed MemoryResult instance.
        """
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            memory_type=data.get("metadata", {}).get("type", "unknown"),
            relevance_score=data.get("relevance_score", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class IdentityContext:
    """Context from the Parfitian identity graph.

    Represents a subgraph of identity concepts and their
    relationships, retrieved from the Neo4j knowledge graph.

    Attributes:
        concept: The central concept queried.
        depth: Traversal depth used for the query.
        paths: List of relationship paths from the graph.
    """

    concept: str
    depth: int
    paths: List[Dict[str, Any]]

    def get_related_concepts(self) -> List[str]:
        """Extract all related concept names from graph paths.

        Returns:
            Deduplicated list of concept names found in paths.
        """
        concepts: set[str] = set()
        for path in self.paths:
            for node in path.get("nodes", []):
                if name := node.get("name"):
                    concepts.add(name)
        return list(concepts)


# =============================================================================
# Production MCP Client
# =============================================================================


class HippocampalClientV2:
    """Production MCP client for the Hippocampal Node.

    Integrates:
    - Retry logic with exponential backoff and jitter
    - Circuit breaker with Probe Latch mechanism
    - Proper connection lifecycle management

    The client composes existing resilience primitives from the
    ``resilience`` and ``connection`` modules into a unified
    interface suitable for production RAG workloads.

    Usage:
        async with HippocampalClientV2() as client:
            results = await client.query_semantic("family values")

        Or manual lifecycle:
            client = HippocampalClientV2()
            await client.connect()
            try:
                results = await client.query_semantic("family values")
            finally:
                await client.disconnect()

    Thread Safety:
        This class uses asyncio locks internally. Multiple coroutines
        can share a single instance within the same event loop.
    """

    def __init__(self) -> None:
        settings = get_settings()

        # Base URL from computed settings property (never hardcoded)
        self._base_url = settings.hippocampal.mcp_url

        # Resilience components wired from centralized settings
        self._retry_config = RetryConfig(
            max_attempts=settings.resilience.retry_max_attempts,
            base_delay_ms=settings.resilience.retry_base_delay_ms,
            max_delay_ms=settings.resilience.retry_max_delay_ms,
            jitter_max_ms=settings.resilience.retry_jitter_max_ms,
        )

        self._circuit_breaker = CircuitBreaker(
            name="hippocampal-mcp",
            config=CircuitBreakerConfig(
                failure_threshold=settings.resilience.circuit_breaker_failure_threshold,
                recovery_timeout_s=settings.resilience.circuit_breaker_recovery_timeout_s,
            ),
        )

        # Connection management
        self._connection = ManagedConnection(
            base_url=self._base_url,
            timeout=30.0,
        )

        # Session state
        self._session_id: Optional[str] = None
        self._initialized: bool = False

        logger.info(
            "HippocampalClientV2 created",
            extra={"base_url": self._base_url},
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for MCP requests including session ID."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse MCP response which may be SSE or plain JSON.

        The Hippocampal Node may return responses in SSE format:
            event: message
            data: {"jsonrpc": "2.0", ...}

        Or plain JSON. This method handles both.

        Args:
            text: Raw response text.

        Returns:
            Parsed JSON dictionary.
        """
        # Try SSE format first (event: ...\ndata: {...})
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
            logger.error(f"Failed to parse MCP response: {text[:200]}")
            raise ValueError(f"Invalid MCP response format: {text[:100]}")

    # =========================================================================
    # Context Manager Protocol
    # =========================================================================

    async def __aenter__(self) -> "HippocampalClientV2":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    # =========================================================================
    # Connection Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """Initialize MCP session with Hippocampal Node.

        Performs the MCP protocol handshake (initialize + initialized
        notification). Idempotent - safe to call multiple times.

        Raises:
            httpx.HTTPStatusError: If the handshake HTTP request fails.
            Exception: If MCP initialization is rejected by the server.
        """
        if self._initialized:
            return

        client = await self._connection.get_client()

        try:
            # MCP Initialize handshake
            response = await client.post(
                "/mcp",
                headers=self._get_headers(),
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "clientInfo": {
                            "name": "ubik-somatic-v2",
                            "version": "2.0.0",
                        },
                        "capabilities": {},
                    },
                    "id": 1,
                },
            )
            response.raise_for_status()

            # Capture session ID before sending initialized notification
            self._session_id = response.headers.get("mcp-session-id", "")

            # Send initialized notification (with session ID header)
            await client.post(
                "/mcp",
                headers=self._get_headers(),
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                },
            )
            self._initialized = True

            await self._circuit_breaker.record_success()

            logger.info(
                "MCP session initialized",
                extra={"session_id": self._session_id},
            )

        except Exception as e:
            await self._circuit_breaker.record_failure()
            await self._connection.invalidate()
            logger.error(f"MCP initialization failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Clean shutdown of MCP session.

        Closes the underlying HTTP connection and resets session state.
        Safe to call multiple times or on an uninitialized client.
        """
        await self._connection.close()
        self._initialized = False
        self._session_id = None
        logger.info("Disconnected from Hippocampal Node")

    # =========================================================================
    # Core Tool Calling with Resilience
    # =========================================================================

    async def _call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call an MCP tool with full resilience stack.

        Applies in order:
        1. Circuit breaker check (fail-fast if service is down)
        2. Retry with exponential backoff + jitter
        3. Connection invalidation on failure (clean state)

        Args:
            tool_name: Name of the MCP tool to invoke.
            arguments: Tool arguments dictionary.

        Returns:
            Parsed tool result as dictionary.

        Raises:
            CircuitOpenError: If circuit breaker is open.
            Exception: If all retries are exhausted.
        """
        # Check circuit breaker first (fail-fast)
        if not await self._circuit_breaker.allow_request():
            raise CircuitOpenError("hippocampal-mcp")

        last_exception: Optional[Exception] = None

        for attempt in range(self._retry_config.max_attempts):
            try:
                # Ensure connected (lazy reconnect after failure)
                if not self._initialized:
                    await self.connect()

                client = await self._connection.get_client()

                response = await client.post(
                    "/mcp",
                    headers=self._get_headers(),
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": arguments,
                        },
                        "id": int(time.time() * 1000),
                    },
                )
                response.raise_for_status()

                # Parse response (may be SSE or plain JSON)
                result = self._parse_response(response.text)

                # Debug: log parsed result structure
                logger.debug(
                    f"MCP response for {tool_name}: keys={list(result.keys())}, "
                    f"has_result={'result' in result}, has_error={'error' in result}"
                )

                # Handle MCP-level errors
                if "error" in result:
                    raise Exception(result["error"].get("message", "Unknown MCP error"))

                # Extract content from MCP response
                content = result.get("result", {}).get("content", [])

                # Debug: log content structure for troubleshooting
                if not content:
                    logger.debug(f"MCP response has no content for {tool_name}, result keys: {list(result.get('result', {}).keys())}")

                if content and isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            text_value = item.get("text", "{}")

                            # Check if server returned an error message instead of JSON
                            if text_value.startswith("Error calling tool"):
                                logger.warning(f"Server returned error for {tool_name}: {text_value[:200]}")
                                # Return empty result instead of failing
                                await self._circuit_breaker.record_success()
                                self._connection.record_request()
                                return {"status": "error", "message": text_value}

                            try:
                                parsed = json.loads(text_value)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse tool response as JSON: {e}, text preview: {text_value[:200]}")
                                raise
                            await self._circuit_breaker.record_success()
                            self._connection.record_request()
                            return parsed

                await self._circuit_breaker.record_success()
                self._connection.record_request()
                return result.get("result", {})

            except httpx.RequestError as e:
                # Network/connection error - invalidate connection
                last_exception = e
                await self._circuit_breaker.record_failure()
                await self._connection.invalidate()
                self._initialized = False

                if attempt < self._retry_config.max_attempts - 1:
                    delay = calculate_backoff_delay(attempt, self._retry_config)
                    logger.warning(
                        f"Tool call failed (network error), retrying: {e}",
                        extra={
                            "tool": tool_name,
                            "attempt": attempt + 1,
                            "max_attempts": self._retry_config.max_attempts,
                            "delay_s": delay,
                        },
                    )
                    await asyncio.sleep(delay)

            except Exception as e:
                # Parsing/logic error - don't invalidate connection, just retry
                last_exception = e
                await self._circuit_breaker.record_failure()
                # Don't invalidate - connection is fine, just parsing failed
                self._initialized = False

                # Check if we should retry
                if attempt < self._retry_config.max_attempts - 1:
                    delay = calculate_backoff_delay(attempt, self._retry_config)

                    # Log the actual error message for debugging
                    logger.warning(
                        f"Tool call failed (parsing/logic error), retrying: {type(e).__name__}: {e}",
                        extra={
                            "tool": tool_name,
                            "attempt": attempt + 1,
                            "max_attempts": self._retry_config.max_attempts,
                            "delay_s": delay,
                        },
                    )

                    await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(
            f"Tool call failed after all retries: {type(last_exception).__name__}: {last_exception}",
            extra={
                "tool": tool_name,
                "attempts": self._retry_config.max_attempts,
            },
        )
        raise last_exception  # type: ignore[misc]

    # =========================================================================
    # Memory Operations
    # =========================================================================

    async def query_semantic(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryResult]:
        """Search semantic knowledge (beliefs, values, preferences).

        Args:
            query: Search query text.
            n_results: Maximum results to return.
            filters: Optional metadata filters.

        Returns:
            List of matching knowledge entries ranked by relevance.
        """
        result = await self._call_tool(
            "query_semantic",
            {
                "params": {
                    "query": query,
                    "n_results": n_results,
                    "filters": filters,
                },
            },
        )

        knowledge = result.get("knowledge", [])

        # Log at ID level only (privacy: no content)
        logger.debug(
            "Semantic query complete",
            extra={
                "result_count": len(knowledge),
                "memory_ids": [k.get("id") for k in knowledge],
            },
        )

        return [MemoryResult.from_dict(k) for k in knowledge]

    async def query_episodic(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryResult]:
        """Search episodic memories (experiences, conversations, events).

        Args:
            query: Search query text.
            n_results: Maximum results to return.
            filters: Optional metadata filters.

        Returns:
            List of matching memories ranked by relevance.
        """
        result = await self._call_tool(
            "query_episodic",
            {
                "params": {
                    "query": query,
                    "n_results": n_results,
                    "filters": filters,
                },
            },
        )

        memories = result.get("memories", [])

        # Log at ID level only (privacy: no content)
        logger.debug(
            "Episodic query complete",
            extra={
                "result_count": len(memories),
                "memory_ids": [m.get("id") for m in memories],
            },
        )

        return [MemoryResult.from_dict(m) for m in memories]

    async def get_identity_context(
        self,
        concept: str = "Self",
        depth: int = 2,
    ) -> IdentityContext:
        """Retrieve Parfitian identity context from Neo4j graph.

        Args:
            concept: Central concept to explore (default: "Self").
            depth: Relationship traversal depth (1-3).

        Returns:
            Identity context with related concepts and paths.
        """
        result = await self._call_tool(
            "get_identity_context",
            {
                "concept": concept,
                "depth": min(max(depth, 1), 3),  # Clamp to 1-3
            },
        )

        return IdentityContext(
            concept=result.get("concept", concept),
            depth=result.get("depth", depth),
            paths=result.get("context", []),
        )

    # =========================================================================
    # Health & Diagnostics
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Check health of Hippocampal Node connection.

        Returns:
            Health status dict including circuit breaker state.
        """
        try:
            result = await self._call_tool("get_memory_stats", {})
            return {
                "status": "healthy",
                "circuit_breaker": self._circuit_breaker.state.value,
                "session_id": self._session_id,
                **result,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "circuit_breaker": self._circuit_breaker.state.value,
                "error": str(e),
            }


# =============================================================================
# Convenience Context Manager
# =============================================================================


@asynccontextmanager
async def hippocampal_session():
    """Context manager for Hippocampal client sessions.

    Handles connection setup and teardown automatically.

    Usage:
        async with hippocampal_session() as client:
            results = await client.query_semantic("family values")

    Yields:
        Connected HippocampalClientV2 instance.
    """
    client = HippocampalClientV2()
    try:
        await client.connect()
        yield client
    finally:
        await client.disconnect()
