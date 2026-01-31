#!/usr/bin/env python3
"""
Ubik Hippocampal Node - Resilience Patterns

Implements resilience patterns for external service calls:
- Circuit Breaker: Prevents cascade failures
- Retry with Exponential Backoff: Handles transient failures
- Timeout wrapper: Prevents hung operations

Usage:
    from resilience import CircuitBreaker, retry_with_backoff

    # Circuit breaker for Neo4j
    neo4j_breaker = CircuitBreaker(name="neo4j", threshold=5, timeout=60.0)

    async def query_neo4j():
        if not neo4j_breaker.allow_request():
            raise CircuitOpenError("neo4j", neo4j_breaker.failure_count)

        try:
            result = await do_query()
            neo4j_breaker.record_success()
            return result
        except Exception as e:
            neo4j_breaker.record_failure()
            raise

    # Retry with backoff
    result = await retry_with_backoff(
        query_neo4j,
        max_retries=3,
        base_delay=1.0
    )
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar, Callable, Optional, Any, Awaitable
from functools import wraps

from exceptions import CircuitOpenError, RetryExhaustedError

logger = logging.getLogger("ubik.resilience")

T = TypeVar('T')


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation, requests allowed
    OPEN = "open"           # Failing, requests rejected
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker to prevent cascade failures to external services.

    When a service fails repeatedly, the circuit "opens" and rejects
    requests immediately instead of waiting for timeouts. After a
    cooldown period, it enters "half-open" state to test recovery.

    Attributes:
        name: Service name for logging.
        threshold: Number of failures before opening circuit.
        timeout: Seconds before attempting recovery (half-open).
        half_open_max_calls: Max calls allowed in half-open state.

    Example:
        breaker = CircuitBreaker(name="neo4j", threshold=5, timeout=60.0)

        if breaker.allow_request():
            try:
                result = call_service()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
                raise
        else:
            raise CircuitOpenError("neo4j", breaker.failure_count)
    """
    name: str
    threshold: int = 5
    timeout: float = 60.0
    half_open_max_calls: int = 3

    # Internal state (not part of constructor)
    _failure_count: int = field(default=0, init=False, repr=False)
    _success_count: int = field(default=0, init=False, repr=False)
    _last_failure_time: Optional[float] = field(default=None, init=False, repr=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)
    _half_open_calls: int = field(default=0, init=False, repr=False)

    @property
    def state(self) -> CircuitState:
        """
        Get current circuit state, checking for timeout transition.

        Returns:
            Current CircuitState.
        """
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.timeout:
                    logger.info(
                        f"Circuit {self.name}: OPEN -> HALF_OPEN after {elapsed:.1f}s"
                    )
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed.

        Returns:
            True if request should proceed, False if circuit is open.
        """
        state = self.state

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

        # OPEN state
        return False

    def record_success(self) -> None:
        """
        Record a successful request.

        Resets failure count and closes circuit if in half-open state.
        """
        self._success_count += 1

        if self._state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED (recovery confirmed)")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """
        Record a failed request.

        Opens circuit if threshold is reached.
        """
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            logger.warning(
                f"Circuit {self.name}: HALF_OPEN -> OPEN (recovery failed)"
            )
            self._state = CircuitState.OPEN
            self._half_open_calls = 0

        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.threshold:
                logger.warning(
                    f"Circuit {self.name}: CLOSED -> OPEN "
                    f"(threshold {self.threshold} reached)"
                )
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset circuit to initial closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        logger.info(f"Circuit {self.name}: Reset to CLOSED")

    def get_stats(self) -> dict:
        """
        Get circuit breaker statistics.

        Returns:
            Dictionary with current stats.
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "threshold": self.threshold,
            "timeout": self.timeout,
            "last_failure": self._last_failure_time
        }


# =============================================================================
# Retry with Exponential Backoff
# =============================================================================

async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    **kwargs: Any
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry.
        *args: Positional arguments for func.
        max_retries: Maximum retry attempts (0 = no retries).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Base for exponential calculation.
        jitter: Add random jitter to prevent thundering herd.
        retryable_exceptions: Tuple of exceptions to retry on.
        **kwargs: Keyword arguments for func.

    Returns:
        Result from successful function call.

    Raises:
        RetryExhaustedError: If all retries fail.

    Example:
        result = await retry_with_backoff(
            fetch_data,
            url="http://example.com",
            max_retries=3,
            base_delay=1.0
        )
    """
    import random

    last_exception: Optional[Exception] = None
    service_name = getattr(func, '__name__', 'unknown')

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)

        except retryable_exceptions as e:
            last_exception = e

            if attempt < max_retries:
                # Calculate delay with exponential backoff
                delay = min(
                    base_delay * (exponential_base ** attempt),
                    max_delay
                )

                # Add jitter (0-25% of delay)
                if jitter:
                    delay = delay * (1 + random.uniform(0, 0.25))

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed for {service_name}: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed for {service_name}: {e}"
                )

    raise RetryExhaustedError(
        service_name=service_name,
        attempts=max_retries + 1,
        last_error=last_exception
    )


def retry_sync_with_backoff(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    **kwargs: Any
) -> T:
    """
    Retry a synchronous function with exponential backoff.

    Same as retry_with_backoff but for sync functions.
    See retry_with_backoff for parameter documentation.
    """
    import random

    last_exception: Optional[Exception] = None
    service_name = getattr(func, '__name__', 'unknown')

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)

        except retryable_exceptions as e:
            last_exception = e

            if attempt < max_retries:
                delay = min(
                    base_delay * (exponential_base ** attempt),
                    max_delay
                )

                if jitter:
                    delay = delay * (1 + random.uniform(0, 0.25))

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed for {service_name}: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed for {service_name}: {e}"
                )

    raise RetryExhaustedError(
        service_name=service_name,
        attempts=max_retries + 1,
        last_error=last_exception
    )


# =============================================================================
# Decorator Versions
# =============================================================================

def with_circuit_breaker(breaker: CircuitBreaker):
    """
    Decorator to wrap a function with circuit breaker protection.

    Args:
        breaker: CircuitBreaker instance to use.

    Returns:
        Decorated function.

    Example:
        neo4j_breaker = CircuitBreaker(name="neo4j")

        @with_circuit_breaker(neo4j_breaker)
        async def query_neo4j():
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            if not breaker.allow_request():
                raise CircuitOpenError(
                    service_name=breaker.name,
                    failure_count=breaker.failure_count,
                    reset_time=breaker.timeout
                )

            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise

        return wrapper
    return decorator


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (Exception,)
):
    """
    Decorator to wrap a function with retry logic.

    Args:
        max_retries: Maximum retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap.
        retryable_exceptions: Exceptions to retry on.

    Returns:
        Decorated function.

    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        async def fetch_data():
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                retryable_exceptions=retryable_exceptions,
                **kwargs
            )
        return wrapper
    return decorator


# =============================================================================
# Timeout Wrapper
# =============================================================================

async def with_timeout(
    coro: Awaitable[T],
    timeout_seconds: float,
    operation_name: str = "operation"
) -> T:
    """
    Wrap an awaitable with a timeout.

    Args:
        coro: Awaitable to execute.
        timeout_seconds: Timeout in seconds.
        operation_name: Name for error messages.

    Returns:
        Result from the awaitable.

    Raises:
        asyncio.TimeoutError: If operation times out.

    Example:
        result = await with_timeout(
            fetch_data(),
            timeout_seconds=30.0,
            operation_name="fetch_data"
        )
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"{operation_name} timed out after {timeout_seconds}s")
        raise


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Circuit Breaker
    "CircuitState",
    "CircuitBreaker",
    # Retry
    "retry_with_backoff",
    "retry_sync_with_backoff",
    # Decorators
    "with_circuit_breaker",
    "with_retry",
    # Timeout
    "with_timeout",
]
