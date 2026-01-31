"""
Ubik Somatic Node - MCP Client Package

Provides connectivity to the Hippocampal Node's memory services.

Clients:
    - HippocampalClient: Original client (basic error handling)
    - HippocampalClientV2: Production client (retry, circuit breaker, connection mgmt)
"""

from .client import (
    HippocampalClientV2,
    IdentityContext,
    hippocampal_session,
)
from .client import MemoryResult as MemoryResultV2
from .connection import ConnectionPool, ManagedConnection
from .hippocampal_client import (
    HippocampalClient,
    MemoryResult,
    get_context_for_inference,
)
from .rag_integration import RAGContextBuilder
from .resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    RetryConfig,
    calculate_backoff_delay,
    retry_async,
    with_retry,
)

__all__ = [
    # Client V1
    "HippocampalClient",
    "MemoryResult",
    "get_context_for_inference",
    "RAGContextBuilder",
    # Client V2 (production)
    "HippocampalClientV2",
    "MemoryResultV2",
    "IdentityContext",
    "hippocampal_session",
    # Connection
    "ConnectionPool",
    "ManagedConnection",
    # Resilience
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "CircuitState",
    "RetryConfig",
    "calculate_backoff_delay",
    "retry_async",
    "with_retry",
]

__version__ = "2.0.0"
