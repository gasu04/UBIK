"""Managed HTTP connection for MCP client.

Provides proper connection lifecycle management with explicit
client closure and recreation on errors.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("ubik.mcp_client.connection")


class ManagedConnection:
    """Properly manages HTTP client lifecycle.

    On error: explicitly close and recreate the client.
    Never reuse a client that may have corrupted connection state.

    Usage:
        conn = ManagedConnection("http://localhost:8080")

        async with conn:
            client = await conn.get_client()
            try:
                response = await client.post("/endpoint", json=data)
            except httpx.RequestError:
                await conn.invalidate()  # Close and discard client
                raise

    Or without context manager:
        conn = ManagedConnection("http://localhost:8080")
        try:
            client = await conn.get_client()
            # ... use client ...
        finally:
            await conn.close()
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_connections: int = 10,
        max_keepalive_connections: int = 5,
    ):
        """Initialize managed connection.

        Args:
            base_url: Base URL for all requests.
            timeout: Request timeout in seconds.
            max_connections: Maximum concurrent connections.
            max_keepalive_connections: Maximum keepalive connections.
        """
        self._base_url = base_url
        self._timeout = timeout
        self._max_connections = max_connections
        self._max_keepalive_connections = max_keepalive_connections
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
        self._request_count = 0
        self._error_count = 0

    @property
    def base_url(self) -> str:
        """Get the base URL."""
        return self._base_url

    @property
    def is_connected(self) -> bool:
        """Check if client exists (not necessarily healthy)."""
        return self._client is not None

    @property
    def stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "base_url": self._base_url,
            "is_connected": self.is_connected,
            "request_count": self._request_count,
            "error_count": self._error_count,
        }

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create an HTTP client.

        Thread-safe client creation with connection pooling limits.

        Returns:
            Configured httpx.AsyncClient instance.
        """
        async with self._lock:
            if self._client is None:
                logger.debug(f"Creating new HTTP client for {self._base_url}")
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=httpx.Timeout(self._timeout),
                    limits=httpx.Limits(
                        max_connections=self._max_connections,
                        max_keepalive_connections=self._max_keepalive_connections,
                    ),
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                )
            return self._client

    async def invalidate(self) -> None:
        """Invalidate the current client.

        MUST explicitly close before discarding reference.
        This ensures connection pool is properly cleaned up.
        """
        async with self._lock:
            if self._client is not None:
                logger.warning(
                    f"Invalidating HTTP client for {self._base_url} "
                    f"(requests: {self._request_count}, errors: {self._error_count})"
                )
                try:
                    await self._client.aclose()
                except Exception as e:
                    logger.error(f"Error closing HTTP client: {e}")
                finally:
                    self._client = None
                    self._error_count += 1

    async def close(self) -> None:
        """Clean shutdown of the connection."""
        async with self._lock:
            if self._client is not None:
                logger.debug(f"Closing HTTP client for {self._base_url}")
                try:
                    await self._client.aclose()
                except Exception as e:
                    logger.error(f"Error during clean shutdown: {e}")
                finally:
                    self._client = None

    def record_request(self) -> None:
        """Record a successful request (for statistics)."""
        self._request_count += 1

    async def __aenter__(self) -> "ManagedConnection":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Async context manager exit - ensures cleanup."""
        if exc_type is not None:
            # Error occurred, invalidate the client
            logger.debug(
                f"Exception during request: {exc_type.__name__}, invalidating client"
            )
            await self.invalidate()
        await self.close()


class ConnectionPool:
    """Pool of managed connections for multiple endpoints.

    Usage:
        pool = ConnectionPool()
        mcp_conn = await pool.get_connection("mcp", "http://localhost:8080")
        chroma_conn = await pool.get_connection("chroma", "http://localhost:8001")

        # On error with specific connection:
        await pool.invalidate("mcp")

        # Clean shutdown:
        await pool.close_all()
    """

    def __init__(self, default_timeout: float = 30.0):
        """Initialize connection pool.

        Args:
            default_timeout: Default timeout for new connections.
        """
        self._connections: Dict[str, ManagedConnection] = {}
        self._default_timeout = default_timeout
        self._lock = asyncio.Lock()

    async def get_connection(
        self,
        name: str,
        base_url: str,
        timeout: Optional[float] = None,
    ) -> ManagedConnection:
        """Get or create a named connection.

        Args:
            name: Unique identifier for this connection.
            base_url: Base URL for the connection.
            timeout: Optional custom timeout.

        Returns:
            ManagedConnection instance.
        """
        async with self._lock:
            if name not in self._connections:
                self._connections[name] = ManagedConnection(
                    base_url=base_url,
                    timeout=timeout or self._default_timeout,
                )
                logger.debug(f"Created connection '{name}' -> {base_url}")
            return self._connections[name]

    async def invalidate(self, name: str) -> None:
        """Invalidate a specific connection.

        Args:
            name: Connection identifier.
        """
        async with self._lock:
            if name in self._connections:
                await self._connections[name].invalidate()

    async def close_all(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            for name, conn in self._connections.items():
                logger.debug(f"Closing connection '{name}'")
                await conn.close()
            self._connections.clear()

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all connections."""
        return {name: conn.stats for name, conn in self._connections.items()}
