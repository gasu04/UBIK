"""Integration tests for HippocampalClientV2.

Tests the production MCP client with mocked HTTP responses.
No live Hippocampal Node required.

Tests:
- MCP session initialization (connect/disconnect)
- Retry on transient failure (mock network error)
- Circuit breaker opens after repeated failures
- Connection cleanup on error
- query_semantic() returns MemoryResult objects
- query_episodic() returns MemoryResult objects
- get_identity_context() returns IdentityContext object
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_client.client import (
    HippocampalClientV2,
    IdentityContext,
    MemoryResult,
    hippocampal_session,
)
from mcp_client.resilience import CircuitOpenError, CircuitState


# =============================================================================
# Helpers
# =============================================================================


def _make_mcp_response(
    result_data: dict,
    status_code: int = 200,
    session_id: str = "test-session-1234",
) -> httpx.Response:
    """Create a mock httpx.Response with MCP JSON-RPC result."""
    body = json.dumps({"jsonrpc": "2.0", "result": result_data, "id": 1})
    return httpx.Response(
        status_code=status_code,
        content=body.encode(),
        headers={
            "content-type": "application/json",
            "mcp-session-id": session_id,
        },
        request=httpx.Request("POST", "http://localhost:8080/mcp"),
    )


def _make_mcp_tool_response(tool_result: dict) -> httpx.Response:
    """Create a mock MCP tool response with text content."""
    result_data = {
        "content": [
            {"type": "text", "text": json.dumps(tool_result)}
        ]
    }
    return _make_mcp_response(result_data)


def _make_init_response(session_id: str = "test-session-1234") -> httpx.Response:
    """Create a mock MCP initialize response."""
    return _make_mcp_response(
        {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "hippocampal-node", "version": "1.0.0"},
            "capabilities": {"tools": {}},
        },
        session_id=session_id,
    )


def _make_notification_response() -> httpx.Response:
    """Create a mock response for the initialized notification."""
    return httpx.Response(
        status_code=200,
        content=b"",
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "http://localhost:8080/mcp"),
    )


# =============================================================================
# Connection Lifecycle Tests
# =============================================================================


class TestClientConnection:
    """Test MCP session initialization and teardown."""

    @pytest.mark.asyncio
    async def test_connect_successful(self):
        """Client connects to Hippocampal Node successfully."""
        client = HippocampalClientV2()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=[_make_init_response(), _make_notification_response()]
        )

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ):
            await client.connect()

        assert client._initialized is True
        assert client._session_id == "test-session-123"  # Truncated to 16 chars
        assert client._circuit_breaker.state == CircuitState.CLOSED

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self):
        """Calling connect twice does not reinitialize."""
        client = HippocampalClientV2()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=[_make_init_response(), _make_notification_response()]
        )

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ):
            await client.connect()
            await client.connect()  # Second call should be no-op

        # post called exactly twice (init + notification), not four times
        assert mock_http.post.call_count == 2

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_resets_state(self):
        """Disconnect resets session state."""
        client = HippocampalClientV2()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=[_make_init_response(), _make_notification_response()]
        )

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ):
            await client.connect()

        assert client._initialized is True

        with patch.object(client._connection, "close", new_callable=AsyncMock):
            await client.disconnect()

        assert client._initialized is False
        assert client._session_id is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Async context manager connects and disconnects."""
        client = HippocampalClientV2()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=[_make_init_response(), _make_notification_response()]
        )

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ), patch.object(client._connection, "close", new_callable=AsyncMock) as mock_close:
            async with client:
                assert client._initialized is True

            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure_records_circuit_breaker(self):
        """Failed connection records circuit breaker failure."""
        client = HippocampalClientV2()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ), patch.object(
            client._connection, "invalidate", new_callable=AsyncMock
        ):
            with pytest.raises(httpx.ConnectError):
                await client.connect()

        assert client._initialized is False

        await client.disconnect()


# =============================================================================
# Retry & Circuit Breaker Integration Tests
# =============================================================================


class TestClientResilience:
    """Test retry and circuit breaker behavior in the client."""

    @pytest.fixture
    def client(self):
        """Create a client with fast retry for testing."""
        c = HippocampalClientV2()
        # Override retry config for fast tests
        c._retry_config.max_attempts = 3
        c._retry_config.base_delay_ms = 10
        c._retry_config.max_delay_ms = 50
        c._retry_config.jitter_max_ms = 0
        # Override circuit breaker for fast tests
        c._circuit_breaker.config.failure_threshold = 3
        c._circuit_breaker.config.recovery_timeout_s = 1
        # Mark as initialized (skip connect in tests)
        c._initialized = True
        return c

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, client):
        """Client retries on transient network failure."""
        mock_http = AsyncMock()
        # Fail twice, succeed on third
        mock_http.post = AsyncMock(
            side_effect=[
                httpx.ConnectError("connection refused"),
                httpx.ConnectError("connection refused"),
                _make_mcp_tool_response({"knowledge": []}),
            ]
        )

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ), patch.object(
            client._connection, "invalidate", new_callable=AsyncMock
        ):
            # Mock connect for the re-init attempts
            with patch.object(client, "connect", new_callable=AsyncMock):
                result = await client._call_tool("query_semantic", {"params": {}})

        assert result == {"knowledge": []}
        # 3 attempts total
        assert mock_http.post.call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, client):
        """Client opens circuit breaker after repeated failures."""
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ), patch.object(
            client._connection, "invalidate", new_callable=AsyncMock
        ):
            with patch.object(client, "connect", new_callable=AsyncMock):
                # First call exhausts 3 retries → 3 failures recorded
                with pytest.raises(httpx.ConnectError):
                    await client._call_tool("query_semantic", {"params": {}})

        assert client._circuit_breaker.state == CircuitState.OPEN

        # Next call should fail immediately with CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await client._call_tool("query_semantic", {"params": {}})

    @pytest.mark.asyncio
    async def test_connection_cleanup_on_error(self, client):
        """Client invalidates connection on each failure."""
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        mock_invalidate = AsyncMock()

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ), patch.object(
            client._connection, "invalidate", mock_invalidate
        ):
            with patch.object(client, "connect", new_callable=AsyncMock):
                with pytest.raises(httpx.ConnectError):
                    await client._call_tool("query_semantic", {"params": {}})

        # invalidate called once per failed attempt
        assert mock_invalidate.call_count == 3


# =============================================================================
# Memory Operation Tests
# =============================================================================


class TestMemoryOperations:
    """Test that memory operations return correct types."""

    @pytest.fixture
    def client(self):
        """Create a pre-initialized client."""
        c = HippocampalClientV2()
        c._initialized = True
        c._retry_config.max_attempts = 1
        c._retry_config.jitter_max_ms = 0
        return c

    @pytest.mark.asyncio
    async def test_query_semantic_returns_memory_results(self, client):
        """query_semantic() returns MemoryResult objects."""
        tool_response = {
            "knowledge": [
                {
                    "id": "sem-001",
                    "content": "Family is the foundation of everything.",
                    "relevance_score": 0.95,
                    "metadata": {"type": "belief", "category": "family"},
                },
                {
                    "id": "sem-002",
                    "content": "Authenticity means living your values.",
                    "relevance_score": 0.88,
                    "metadata": {"type": "value", "category": "character"},
                },
            ]
        }

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=_make_mcp_tool_response(tool_response)
        )

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ):
            results = await client.query_semantic("family values", n_results=5)

        assert len(results) == 2
        assert all(isinstance(r, MemoryResult) for r in results)

        assert results[0].id == "sem-001"
        assert results[0].content == "Family is the foundation of everything."
        assert results[0].memory_type == "belief"
        assert results[0].relevance_score == 0.95
        assert results[0].metadata["category"] == "family"

        assert results[1].id == "sem-002"
        assert results[1].memory_type == "value"

    @pytest.mark.asyncio
    async def test_query_episodic_returns_memory_results(self, client):
        """query_episodic() returns MemoryResult objects."""
        tool_response = {
            "memories": [
                {
                    "id": "epi-001",
                    "content": "Wrote a letter to my grandchildren about perseverance.",
                    "relevance_score": 0.92,
                    "metadata": {"type": "experience", "date": "2024-03-15"},
                },
                {
                    "id": "epi-002",
                    "content": "Conversation with Maria about family traditions.",
                    "relevance_score": 0.85,
                    "metadata": {"type": "conversation", "date": "2024-02-10"},
                },
            ]
        }

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=_make_mcp_tool_response(tool_response)
        )

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ):
            results = await client.query_episodic("grandchildren letter", n_results=5)

        assert len(results) == 2
        assert all(isinstance(r, MemoryResult) for r in results)

        assert results[0].id == "epi-001"
        assert results[0].memory_type == "experience"
        assert results[0].relevance_score == 0.92

        assert results[1].id == "epi-002"
        assert results[1].memory_type == "conversation"

    @pytest.mark.asyncio
    async def test_get_identity_context_returns_identity(self, client):
        """get_identity_context() returns IdentityContext object."""
        tool_response = {
            "concept": "Self",
            "depth": 2,
            "context": [
                {
                    "nodes": [
                        {"name": "Self", "type": "Identity"},
                        {"name": "Family", "type": "Value"},
                        {"name": "Authenticity", "type": "Value"},
                    ],
                    "relationships": ["VALUES", "EMBODIES"],
                },
                {
                    "nodes": [
                        {"name": "Self", "type": "Identity"},
                        {"name": "Resilience", "type": "Trait"},
                    ],
                    "relationships": ["DEMONSTRATES"],
                },
            ],
        }

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=_make_mcp_tool_response(tool_response)
        )

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ):
            result = await client.get_identity_context("Self", depth=2)

        assert isinstance(result, IdentityContext)
        assert result.concept == "Self"
        assert result.depth == 2
        assert len(result.paths) == 2

        related = result.get_related_concepts()
        assert "Self" in related
        assert "Family" in related
        assert "Authenticity" in related
        assert "Resilience" in related

    @pytest.mark.asyncio
    async def test_query_semantic_empty_results(self, client):
        """query_semantic() handles empty results gracefully."""
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=_make_mcp_tool_response({"knowledge": []})
        )

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ):
            results = await client.query_semantic("nonexistent topic")

        assert results == []

    @pytest.mark.asyncio
    async def test_identity_depth_clamped(self, client):
        """get_identity_context() clamps depth to 1-3."""
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=_make_mcp_tool_response(
                {"concept": "Self", "depth": 3, "context": []}
            )
        )

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ):
            # Depth 10 should be clamped to 3
            await client.get_identity_context("Self", depth=10)

        # Verify the clamped argument was sent
        call_args = mock_http.post.call_args
        sent_body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert sent_body["params"]["arguments"]["depth"] == 3


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Test health check endpoint."""

    @pytest.fixture
    def client(self):
        """Create a pre-initialized client."""
        c = HippocampalClientV2()
        c._initialized = True
        c._session_id = "test-session"
        c._retry_config.max_attempts = 1
        c._retry_config.jitter_max_ms = 0
        return c

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client):
        """Health check returns healthy status."""
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            return_value=_make_mcp_tool_response(
                {"episodic_count": 100, "semantic_count": 50}
            )
        )

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ):
            health = await client.health_check()

        assert health["status"] == "healthy"
        assert health["circuit_breaker"] == "closed"
        assert health["session_id"] == "test-session"
        assert health["episodic_count"] == 100

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, client):
        """Health check returns unhealthy on failure."""
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with patch.object(
            client._connection, "get_client", return_value=mock_http
        ), patch.object(
            client._connection, "invalidate", new_callable=AsyncMock
        ):
            health = await client.health_check()

        assert health["status"] == "unhealthy"
        assert "error" in health


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
