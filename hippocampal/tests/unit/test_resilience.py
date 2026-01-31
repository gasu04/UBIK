#!/usr/bin/env python3
"""
Unit tests for resilience patterns.

Tests CircuitBreaker, retry functions, and decorators
to ensure proper failure handling and recovery behavior.
"""

import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

import pytest

from resilience import (
    CircuitState,
    CircuitBreaker,
    retry_with_backoff,
    retry_sync_with_backoff,
    with_circuit_breaker,
    with_retry,
    with_timeout,
)
from exceptions import CircuitOpenError, RetryExhaustedError


# =============================================================================
# CircuitBreaker Tests
# =============================================================================

class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_states_exist(self) -> None:
        """Verify all expected states exist."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_is_closed(self) -> None:
        """Circuit starts in closed state."""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    def test_allows_requests_when_closed(self) -> None:
        """Closed circuit allows requests."""
        breaker = CircuitBreaker(name="test")
        assert breaker.allow_request() is True

    def test_success_resets_failure_count(self) -> None:
        """Recording success resets failure counter."""
        breaker = CircuitBreaker(name="test", threshold=5)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.failure_count == 2

        breaker.record_success()
        assert breaker.failure_count == 0

    def test_opens_after_threshold_failures(self) -> None:
        """Circuit opens after threshold failures."""
        breaker = CircuitBreaker(name="test", threshold=3)

        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    def test_rejects_requests_when_open(self) -> None:
        """Open circuit rejects requests."""
        breaker = CircuitBreaker(name="test", threshold=1)
        breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.allow_request() is False

    def test_transitions_to_half_open_after_timeout(self) -> None:
        """Circuit becomes half-open after timeout."""
        breaker = CircuitBreaker(name="test", threshold=1, timeout=0.1)
        breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_allows_limited_requests(self) -> None:
        """Half-open state allows limited test requests."""
        breaker = CircuitBreaker(
            name="test",
            threshold=1,
            timeout=0.01,
            half_open_max_calls=2
        )
        breaker.record_failure()
        time.sleep(0.02)

        # Should allow first 2 requests
        assert breaker.allow_request() is True
        assert breaker.allow_request() is True
        # Should reject after max calls
        assert breaker.allow_request() is False

    def test_half_open_success_closes_circuit(self) -> None:
        """Success in half-open state closes circuit."""
        breaker = CircuitBreaker(name="test", threshold=1, timeout=0.01)
        breaker.record_failure()
        time.sleep(0.02)

        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_half_open_failure_reopens_circuit(self) -> None:
        """Failure in half-open state reopens circuit."""
        breaker = CircuitBreaker(name="test", threshold=1, timeout=0.01)
        breaker.record_failure()
        time.sleep(0.02)

        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_reset_clears_all_state(self) -> None:
        """Reset returns circuit to initial state."""
        breaker = CircuitBreaker(name="test", threshold=1)
        breaker.record_failure()
        breaker.record_failure()

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_get_stats_returns_dict(self) -> None:
        """Stats method returns correct information."""
        breaker = CircuitBreaker(name="test_service", threshold=5, timeout=60.0)
        breaker.record_failure()
        breaker.record_success()

        stats = breaker.get_stats()

        assert stats["name"] == "test_service"
        assert stats["state"] == "closed"
        assert stats["threshold"] == 5
        assert stats["timeout"] == 60.0
        assert "failure_count" in stats
        assert "success_count" in stats


# =============================================================================
# Retry with Backoff Tests
# =============================================================================

class TestRetryWithBackoff:
    """Tests for async retry_with_backoff function."""

    @pytest.mark.asyncio
    async def test_returns_on_first_success(self) -> None:
        """Returns immediately on successful call."""
        mock_func = AsyncMock(return_value="success")

        result = await retry_with_backoff(mock_func, max_retries=3)

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self) -> None:
        """Retries specified number of times before success."""
        mock_func = AsyncMock(side_effect=[
            ValueError("fail 1"),
            ValueError("fail 2"),
            "success"
        ])

        result = await retry_with_backoff(
            mock_func,
            max_retries=3,
            base_delay=0.01
        )

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_retry_exhausted_after_max_retries(self) -> None:
        """Raises RetryExhaustedError after all retries fail."""
        mock_func = AsyncMock(side_effect=ValueError("always fails"))

        with pytest.raises(RetryExhaustedError) as exc_info:
            await retry_with_backoff(
                mock_func,
                max_retries=2,
                base_delay=0.01
            )

        assert exc_info.value.attempts == 3  # initial + 2 retries
        assert isinstance(exc_info.value.last_error, ValueError)

    @pytest.mark.asyncio
    async def test_respects_retryable_exceptions(self) -> None:
        """Only retries on specified exception types."""
        mock_func = AsyncMock(side_effect=TypeError("not retryable"))

        with pytest.raises(TypeError):
            await retry_with_backoff(
                mock_func,
                max_retries=3,
                base_delay=0.01,
                retryable_exceptions=(ValueError,)
            )

        # Should not have retried since TypeError is not retryable
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_delay_increases_exponentially(self) -> None:
        """Verify delay increases with each retry."""
        mock_func = AsyncMock(side_effect=[
            ValueError("fail 1"),
            ValueError("fail 2"),
            "success"
        ])

        with patch('resilience.asyncio.sleep') as mock_sleep:
            await retry_with_backoff(
                mock_func,
                max_retries=3,
                base_delay=1.0,
                jitter=False
            )

            # First retry: 1.0 * 2^0 = 1.0
            # Second retry: 1.0 * 2^1 = 2.0
            delays = [call[0][0] for call in mock_sleep.call_args_list]
            assert delays[0] == 1.0
            assert delays[1] == 2.0

    @pytest.mark.asyncio
    async def test_max_delay_caps_backoff(self) -> None:
        """Max delay caps the exponential backoff."""
        mock_func = AsyncMock(side_effect=[
            ValueError("fail"),
            ValueError("fail"),
            ValueError("fail"),
            "success"
        ])

        with patch('resilience.asyncio.sleep') as mock_sleep:
            await retry_with_backoff(
                mock_func,
                max_retries=4,
                base_delay=10.0,
                max_delay=15.0,
                jitter=False
            )

            delays = [call[0][0] for call in mock_sleep.call_args_list]
            # Should be capped at 15.0
            assert all(d <= 15.0 for d in delays)

    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs(self) -> None:
        """Passes arguments correctly to the function."""
        mock_func = AsyncMock(return_value="result")

        await retry_with_backoff(
            mock_func,
            "arg1", "arg2",
            kwarg1="value1",
            max_retries=1
        )

        mock_func.assert_called_with("arg1", "arg2", kwarg1="value1")


class TestRetrySyncWithBackoff:
    """Tests for sync retry_sync_with_backoff function."""

    def test_returns_on_first_success(self) -> None:
        """Returns immediately on successful call."""
        mock_func = Mock(return_value="success")

        result = retry_sync_with_backoff(mock_func, max_retries=3)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retries_on_failure(self) -> None:
        """Retries on failure and eventually succeeds."""
        mock_func = Mock(side_effect=[
            ValueError("fail"),
            "success"
        ])

        with patch('resilience.time.sleep'):
            result = retry_sync_with_backoff(
                mock_func,
                max_retries=3,
                base_delay=0.01
            )

        assert result == "success"
        assert mock_func.call_count == 2

    def test_raises_after_exhausted_retries(self) -> None:
        """Raises RetryExhaustedError when all retries fail."""
        mock_func = Mock(side_effect=ValueError("always fails"))

        with patch('resilience.time.sleep'):
            with pytest.raises(RetryExhaustedError) as exc_info:
                retry_sync_with_backoff(
                    mock_func,
                    max_retries=1,
                    base_delay=0.01
                )

        assert exc_info.value.attempts == 2


# =============================================================================
# Decorator Tests
# =============================================================================

class TestWithCircuitBreakerDecorator:
    """Tests for @with_circuit_breaker decorator."""

    @pytest.mark.asyncio
    async def test_allows_request_when_closed(self) -> None:
        """Decorated function executes when circuit is closed."""
        breaker = CircuitBreaker(name="test", threshold=5)

        @with_circuit_breaker(breaker)
        async def my_func():
            return "result"

        result = await my_func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_records_success(self) -> None:
        """Decorator records success on successful call."""
        breaker = CircuitBreaker(name="test", threshold=5)
        breaker.record_failure()  # Start with 1 failure

        @with_circuit_breaker(breaker)
        async def my_func():
            return "result"

        await my_func()
        assert breaker.failure_count == 0  # Reset by success

    @pytest.mark.asyncio
    async def test_records_failure(self) -> None:
        """Decorator records failure on exception."""
        breaker = CircuitBreaker(name="test", threshold=5)

        @with_circuit_breaker(breaker)
        async def my_func():
            raise ValueError("error")

        with pytest.raises(ValueError):
            await my_func()

        assert breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_raises_circuit_open_when_open(self) -> None:
        """Decorator raises CircuitOpenError when circuit is open."""
        breaker = CircuitBreaker(name="test", threshold=1)
        breaker.record_failure()  # Open the circuit

        @with_circuit_breaker(breaker)
        async def my_func():
            return "result"

        with pytest.raises(CircuitOpenError) as exc_info:
            await my_func()

        assert exc_info.value.service_name == "test"


class TestWithRetryDecorator:
    """Tests for @with_retry decorator."""

    @pytest.mark.asyncio
    async def test_retries_decorated_function(self) -> None:
        """Decorator retries the function on failure."""
        call_count = 0

        @with_retry(max_retries=2, base_delay=0.01)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("temporary failure")
            return "success"

        result = await flaky_func()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self) -> None:
        """Decorator raises after max retries exhausted."""
        @with_retry(max_retries=1, base_delay=0.01)
        async def always_fails():
            raise ValueError("always fails")

        with pytest.raises(RetryExhaustedError):
            await always_fails()


# =============================================================================
# Timeout Tests
# =============================================================================

class TestWithTimeout:
    """Tests for with_timeout function."""

    @pytest.mark.asyncio
    async def test_returns_result_within_timeout(self) -> None:
        """Returns result when operation completes in time."""
        async def quick_op():
            return "done"

        result = await with_timeout(quick_op(), timeout_seconds=1.0)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_raises_timeout_error_when_exceeded(self) -> None:
        """Raises TimeoutError when operation takes too long."""
        async def slow_op():
            await asyncio.sleep(1.0)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await with_timeout(
                slow_op(),
                timeout_seconds=0.01,
                operation_name="slow_op"
            )
