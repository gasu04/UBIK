"""Unit tests for resilience module.

Tests:
- Exponential backoff calculation with jitter
- Circuit breaker state transitions
- Probe Latch mechanism (only 1 request in HALF_OPEN)
- Connection cleanup (aclose() called on invalidation)
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_client.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    RetryConfig,
    calculate_backoff_delay,
    retry_async,
)
from mcp_client.connection import ManagedConnection


class TestExponentialBackoff:
    """Test exponential backoff calculation."""

    def test_exponential_growth_no_jitter(self):
        """Verify delay grows exponentially: base * 2^attempt."""
        config = RetryConfig(
            base_delay_ms=1000,
            max_delay_ms=60000,
            jitter_max_ms=0,  # No jitter for deterministic test
        )

        delays = [calculate_backoff_delay(i, config) for i in range(6)]

        assert delays[0] == 1.0  # 1000ms = 1s
        assert delays[1] == 2.0  # 2000ms = 2s
        assert delays[2] == 4.0  # 4000ms = 4s
        assert delays[3] == 8.0  # 8000ms = 8s
        assert delays[4] == 16.0  # 16000ms = 16s
        assert delays[5] == 32.0  # 32000ms = 32s

    def test_max_delay_cap(self):
        """Verify delay is capped at max_delay_ms."""
        config = RetryConfig(
            base_delay_ms=1000,
            max_delay_ms=10000,  # 10s cap
            jitter_max_ms=0,
        )

        # Attempt 5 would be 32s without cap
        delay = calculate_backoff_delay(5, config)
        assert delay == 10.0  # Capped at 10s

    def test_jitter_adds_randomness(self):
        """Verify jitter adds variation to delays."""
        config = RetryConfig(
            base_delay_ms=1000,
            max_delay_ms=60000,
            jitter_max_ms=500,  # Up to 500ms jitter
        )

        # Run multiple times to check for variation
        delays = [calculate_backoff_delay(0, config) for _ in range(20)]

        # Base delay is 1s, jitter adds 0-0.5s
        assert all(1.0 <= d <= 1.5 for d in delays)

        # Should have some variation (not all identical)
        unique_delays = set(delays)
        assert len(unique_delays) > 1, "Jitter should add variation"

    def test_jitter_bounded(self):
        """Verify jitter never exceeds jitter_max_ms."""
        config = RetryConfig(
            base_delay_ms=1000,
            max_delay_ms=60000,
            jitter_max_ms=100,
        )

        for _ in range(100):
            delay = calculate_backoff_delay(0, config)
            # 1s base + up to 0.1s jitter
            assert 1.0 <= delay <= 1.1


class TestCircuitBreakerStateTransitions:
    """Test circuit breaker state machine."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with test config."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_s=1,
        )
        return CircuitBreaker("test", config=config)

    @pytest.mark.asyncio
    async def test_initial_state_closed(self, breaker):
        """Circuit starts in CLOSED state."""
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_closed_allows_requests(self, breaker):
        """CLOSED state allows all requests."""
        assert await breaker.allow_request() is True
        assert await breaker.allow_request() is True
        assert await breaker.allow_request() is True

    @pytest.mark.asyncio
    async def test_failures_open_circuit(self, breaker):
        """Circuit opens after failure_threshold failures."""
        # Record failures up to threshold
        await breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        await breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        await breaker.record_failure()  # Threshold reached
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_blocks_requests(self, breaker):
        """OPEN state blocks all requests."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert await breaker.allow_request() is False
        assert await breaker.allow_request() is False

    @pytest.mark.asyncio
    async def test_open_to_half_open_after_timeout(self, breaker):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Next allow_request triggers transition
        result = await breaker.allow_request()
        assert breaker.state == CircuitState.HALF_OPEN
        assert result is True  # Probe request allowed

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self, breaker):
        """Successful probe in HALF_OPEN closes circuit."""
        # Open circuit
        for _ in range(3):
            await breaker.record_failure()

        # Wait for recovery and trigger transition
        await asyncio.sleep(1.1)
        await breaker.allow_request()
        assert breaker.state == CircuitState.HALF_OPEN

        # Record success (probe succeeded)
        await breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, breaker):
        """Failed probe in HALF_OPEN reopens circuit."""
        # Open circuit
        for _ in range(3):
            await breaker.record_failure()

        # Wait for recovery and trigger transition
        await asyncio.sleep(1.1)
        await breaker.allow_request()
        assert breaker.state == CircuitState.HALF_OPEN

        # Record failure (probe failed)
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, breaker):
        """Success in CLOSED state resets failure counter."""
        await breaker.record_failure()
        await breaker.record_failure()
        # 2 failures, one more would open

        await breaker.record_success()  # Resets counter

        await breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED  # Still closed


class TestProbeLatch:
    """Test Probe Latch mechanism in HALF_OPEN state."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with short timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_s=0.1,  # 100ms for fast tests
        )
        return CircuitBreaker("probe-test", config=config)

    @pytest.mark.asyncio
    async def test_only_one_probe_allowed(self, breaker):
        """Only ONE request (probe) allowed in HALF_OPEN."""
        # Open circuit
        await breaker.record_failure()
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # First request: becomes probe, allowed
        result1 = await breaker.allow_request()
        assert result1 is True
        assert breaker.state == CircuitState.HALF_OPEN

        # Second request: probe in flight, REJECTED
        result2 = await breaker.allow_request()
        assert result2 is False

        # Third request: still rejected
        result3 = await breaker.allow_request()
        assert result3 is False

    @pytest.mark.asyncio
    async def test_probe_success_allows_all(self, breaker):
        """After probe succeeds, all requests allowed."""
        # Open and transition to HALF_OPEN
        await breaker.record_failure()
        await breaker.record_failure()
        await asyncio.sleep(0.15)

        # Probe request
        await breaker.allow_request()

        # Probe succeeds
        await breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

        # Now all requests allowed
        assert await breaker.allow_request() is True
        assert await breaker.allow_request() is True
        assert await breaker.allow_request() is True

    @pytest.mark.asyncio
    async def test_probe_failure_blocks_again(self, breaker):
        """After probe fails, back to blocking."""
        # Open and transition to HALF_OPEN
        await breaker.record_failure()
        await breaker.record_failure()
        await asyncio.sleep(0.15)

        # Probe request
        await breaker.allow_request()

        # Probe fails
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Requests blocked immediately (no timeout wait)
        assert await breaker.allow_request() is False

    @pytest.mark.asyncio
    async def test_concurrent_requests_one_probe(self, breaker):
        """Concurrent requests in HALF_OPEN: only one becomes probe."""
        # Open and wait for recovery
        await breaker.record_failure()
        await breaker.record_failure()
        await asyncio.sleep(0.15)

        # Simulate concurrent requests
        results = await asyncio.gather(
            breaker.allow_request(),
            breaker.allow_request(),
            breaker.allow_request(),
            breaker.allow_request(),
            breaker.allow_request(),
        )

        # Exactly ONE should be True (the probe)
        assert sum(results) == 1, f"Expected 1 probe, got {sum(results)}: {results}"


class TestConnectionCleanup:
    """Test that aclose() is called on connection invalidation."""

    @pytest.mark.asyncio
    async def test_invalidate_calls_aclose(self):
        """Verify aclose() is called when invalidating connection."""
        conn = ManagedConnection("http://localhost:8080")

        # Create a mock client
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()

        # Inject mock client
        conn._client = mock_client

        # Invalidate should call aclose
        await conn.invalidate()

        mock_client.aclose.assert_called_once()
        assert conn._client is None

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self):
        """Verify aclose() is called on clean close."""
        conn = ManagedConnection("http://localhost:8080")

        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        conn._client = mock_client

        await conn.close()

        mock_client.aclose.assert_called_once()
        assert conn._client is None

    @pytest.mark.asyncio
    async def test_invalidate_increments_error_count(self):
        """Verify error count is incremented on invalidate."""
        conn = ManagedConnection("http://localhost:8080")

        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        conn._client = mock_client

        assert conn.stats["error_count"] == 0

        await conn.invalidate()

        assert conn.stats["error_count"] == 1

    @pytest.mark.asyncio
    async def test_get_client_recreates_after_invalidate(self):
        """Verify new client is created after invalidation."""
        conn = ManagedConnection("http://localhost:8002", timeout=5.0)

        # Get first client
        client1 = await conn.get_client()
        assert client1 is not None

        # Invalidate
        await conn.invalidate()
        assert conn._client is None

        # Get client again - should be new instance
        client2 = await conn.get_client()
        assert client2 is not None
        assert client1 is not client2

        # Cleanup
        await conn.close()

    @pytest.mark.asyncio
    async def test_context_manager_invalidates_on_error(self):
        """Context manager should invalidate on exception."""
        conn = ManagedConnection("http://localhost:8080")

        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()

        with patch.object(conn, "get_client", return_value=mock_client):
            with pytest.raises(ValueError):
                async with conn:
                    conn._client = mock_client
                    raise ValueError("Test error")

        # Should have called aclose via invalidate
        mock_client.aclose.assert_called()


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
