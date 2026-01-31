"""Resilience patterns for MCP client operations.

Implements retry logic with exponential backoff and jitter,
plus circuit breaker pattern for fault tolerance.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, Optional, Set, Type, TypeVar, cast

logger = logging.getLogger("ubik.mcp_client.resilience")

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts (including initial).
        base_delay_ms: Base delay in milliseconds for exponential backoff.
        max_delay_ms: Maximum delay cap in milliseconds.
        jitter_max_ms: Maximum random jitter in milliseconds.
        retryable_exceptions: Exception types that trigger retry.
    """

    max_attempts: int = 3
    base_delay_ms: int = 1000
    max_delay_ms: int = 30000
    jitter_max_ms: int = 500
    retryable_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {
            ConnectionError,
            TimeoutError,
            OSError,
        }
    )

    @classmethod
    def from_settings(cls) -> "RetryConfig":
        """Create RetryConfig from application settings."""
        # Import here to avoid circular imports
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import get_settings

        settings = get_settings()
        return cls(
            max_attempts=settings.resilience.retry_max_attempts,
            base_delay_ms=settings.resilience.retry_base_delay_ms,
            max_delay_ms=settings.resilience.retry_max_delay_ms,
            jitter_max_ms=settings.resilience.retry_jitter_max_ms,
        )


def calculate_backoff_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with exponential backoff and jitter.

    Formula: min(base * 2^attempt, max) + jitter

    This prevents:
    - Thundering herd (via jitter)
    - Unbounded delays (via max cap)
    - Synchronized retries (via randomization)

    Args:
        attempt: Current attempt number (0-indexed).
        config: Retry configuration.

    Returns:
        Delay in seconds.
    """
    exponential_delay = config.base_delay_ms * (2**attempt)
    capped_delay = min(exponential_delay, config.max_delay_ms)
    jitter = random.randint(0, config.jitter_max_ms)

    delay_seconds = (capped_delay + jitter) / 1000.0
    logger.debug(
        f"Backoff delay: attempt={attempt}, "
        f"exponential={exponential_delay}ms, "
        f"capped={capped_delay}ms, "
        f"jitter={jitter}ms, "
        f"total={delay_seconds:.3f}s"
    )
    return delay_seconds


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> T:
    """Execute an async function with retry logic.

    Args:
        func: Async function to execute.
        *args: Positional arguments for func.
        config: Retry configuration (defaults to settings).
        **kwargs: Keyword arguments for func.

    Returns:
        Result of successful function execution.

    Raises:
        Exception: The last exception if all retries exhausted.
    """
    if config is None:
        config = RetryConfig.from_settings()

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_attempts):
        try:
            result: T = await func(*args, **kwargs)
            return result
        except Exception as e:
            last_exception = e

            # Check if exception is retryable
            is_retryable = any(
                isinstance(e, exc_type) for exc_type in config.retryable_exceptions
            )

            if not is_retryable:
                logger.warning(f"Non-retryable exception: {type(e).__name__}: {e}")
                raise

            # Check if we have attempts remaining
            if attempt + 1 >= config.max_attempts:
                logger.error(
                    f"All {config.max_attempts} attempts exhausted. "
                    f"Last error: {type(e).__name__}: {e}"
                )
                raise

            # Calculate and apply backoff delay
            delay = calculate_backoff_delay(attempt, config)
            logger.warning(
                f"Attempt {attempt + 1}/{config.max_attempts} failed: "
                f"{type(e).__name__}: {e}. Retrying in {delay:.3f}s..."
            )
            await asyncio.sleep(delay)

    # Should not reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error: no attempts made")


def with_retry(
    config: Optional[RetryConfig] = None,
) -> Callable[[F], F]:
    """Decorator for adding retry logic to async functions.

    Usage:
        @with_retry()
        async def fetch_data():
            ...

        @with_retry(RetryConfig(max_attempts=5))
        async def fetch_with_custom_config():
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await retry_async(func, *args, config=config, **kwargs)

        return cast(F, wrapper)

    return decorator


# =============================================================================
# Circuit Breaker Pattern with Probe Latch
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failure threshold reached, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered (probe latch active)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit.
        recovery_timeout_s: Seconds to wait before attempting recovery.
    """

    failure_threshold: int = 5
    recovery_timeout_s: int = 60

    @classmethod
    def from_settings(cls) -> "CircuitBreakerConfig":
        """Create CircuitBreakerConfig from application settings."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import get_settings

        settings = get_settings()
        return cls(
            failure_threshold=settings.resilience.circuit_breaker_failure_threshold,
            recovery_timeout_s=settings.resilience.circuit_breaker_recovery_timeout_s,
        )


class CircuitBreaker:
    """Circuit breaker with Probe Latch mechanism.

    In HALF_OPEN state, only ONE request is allowed through to probe
    whether the service has recovered. All other requests are rejected
    until the probe succeeds or fails. This prevents "thundering herd"
    failures when the service is recovering.

    States:
    - CLOSED: Normal operation. Failures increment counter.
    - OPEN: Service considered down. Requests fail immediately.
    - HALF_OPEN: Testing recovery. Only probe request allowed.

    Usage:
        breaker = CircuitBreaker("hippocampal-mcp")

        async def make_request():
            if not await breaker.allow_request():
                raise CircuitOpenError(breaker.name)
            try:
                result = await actual_request()
                await breaker.record_success()
                return result
            except Exception as e:
                await breaker.record_failure()
                raise
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Identifier for this circuit (for logging).
            config: Circuit breaker configuration.
        """
        self.name = name
        self.config = config or CircuitBreakerConfig.from_settings()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._probe_in_flight = False
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state (read-only, no transitions)."""
        return self._state

    async def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Uses Probe Latch mechanism in HALF_OPEN state:
        - Only ONE request (the probe) is allowed through
        - All other requests are rejected until probe completes

        Returns:
            True if request can proceed, False if blocked.
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time is None:
                    return False

                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.config.recovery_timeout_s:
                    # Transition to HALF_OPEN
                    logger.info(
                        f"Circuit '{self.name}' transitioning OPEN -> HALF_OPEN "
                        f"after {elapsed:.1f}s"
                    )
                    self._state = CircuitState.HALF_OPEN
                    self._probe_in_flight = False
                else:
                    return False

            # HALF_OPEN state: Probe Latch mechanism
            if self._state == CircuitState.HALF_OPEN:
                if self._probe_in_flight:
                    # A probe is already in flight, reject this request
                    logger.debug(
                        f"Circuit '{self.name}' rejecting request: "
                        f"probe already in flight"
                    )
                    return False
                else:
                    # This request becomes the probe
                    self._probe_in_flight = True
                    logger.debug(f"Circuit '{self.name}' allowing probe request")
                    return True

            return False

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            self._failure_count = 0
            self._probe_in_flight = False

            if self._state == CircuitState.HALF_OPEN:
                logger.info(
                    f"Circuit '{self.name}' transitioning HALF_OPEN -> CLOSED "
                    f"(probe succeeded, service recovered)"
                )
                self._state = CircuitState.CLOSED

    async def record_failure(self) -> None:
        """Record a failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            self._probe_in_flight = False

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed, go back to OPEN
                logger.warning(
                    f"Circuit '{self.name}' transitioning HALF_OPEN -> OPEN "
                    f"(probe failed, service still unhealthy)"
                )
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"Circuit '{self.name}' transitioning CLOSED -> OPEN "
                        f"after {self._failure_count} consecutive failures"
                    )
                    self._state = CircuitState.OPEN

    async def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        async with self._lock:
            logger.info(f"Circuit '{self.name}' manually reset to CLOSED")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._probe_in_flight = False


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and request is blocked."""

    def __init__(self, circuit_name: str):
        self.circuit_name = circuit_name
        super().__init__(f"Circuit '{circuit_name}' is open - service unavailable")
