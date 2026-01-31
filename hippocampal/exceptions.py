#!/usr/bin/env python3
"""
Ubik Hippocampal Node - Custom Exception Hierarchy

Provides structured error handling for the memory system with
specific exception types for different failure modes.

Usage:
    from exceptions import MemoryStoreError, DatabaseConnectionError

    try:
        result = await store_memory(data)
    except DatabaseConnectionError as e:
        logger.error(f"Database unavailable: {e.service}")
    except MemoryStoreError as e:
        logger.error(f"Failed to store memory: {e}")

Exception Hierarchy:
    HippocampalError (base)
    ├── ConfigurationError
    ├── DatabaseError
    │   ├── DatabaseConnectionError
    │   ├── DatabaseQueryError
    │   └── DatabaseTimeoutError
    ├── MemoryError
    │   ├── MemoryStoreError
    │   ├── MemoryQueryError
    │   └── MemoryValidationError
    ├── IdentityGraphError
    │   ├── GraphTraversalError
    │   └── GraphUpdateError
    └── ServiceError
        ├── CircuitOpenError
        └── RetryExhaustedError
"""

from typing import Optional, Any, Dict


# =============================================================================
# Base Exception
# =============================================================================

class HippocampalError(Exception):
    """
    Base exception for all Hippocampal Node errors.

    All custom exceptions inherit from this class, allowing
    for broad exception catching when needed.

    Attributes:
        message: Human-readable error description
        details: Additional context about the error
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(HippocampalError):
    """
    Invalid or missing configuration.

    Raised when required environment variables are missing,
    configuration values are invalid, or configuration files
    cannot be loaded.

    Attributes:
        config_key: The configuration key that caused the error
    """

    def __init__(self, message: str, config_key: Optional[str] = None):
        self.config_key = config_key
        details = {"config_key": config_key} if config_key else None
        super().__init__(message, details)


# =============================================================================
# Database Errors
# =============================================================================

class DatabaseError(HippocampalError):
    """
    Base class for database-related errors.

    Attributes:
        service: The database service (neo4j, chromadb)
    """

    def __init__(self, message: str, service: str, details: Optional[Dict[str, Any]] = None):
        self.service = service
        full_details = {"service": service}
        if details:
            full_details.update(details)
        super().__init__(message, full_details)


class DatabaseConnectionError(DatabaseError):
    """
    Failed to connect to a database service.

    Raised when the initial connection cannot be established
    or when a connection is lost during operation.

    Attributes:
        host: The database host
        port: The database port
    """

    def __init__(
        self,
        service: str,
        host: str,
        port: int,
        reason: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.reason = reason
        message = f"Failed to connect to {service} at {host}:{port}"
        if reason:
            message += f": {reason}"
        super().__init__(message, service, {"host": host, "port": port, "reason": reason})


class DatabaseQueryError(DatabaseError):
    """
    Database query execution failed.

    Raised when a query fails due to syntax errors,
    constraint violations, or other query-level issues.

    Attributes:
        query: The query that failed (may be truncated)
    """

    def __init__(self, service: str, query: str, reason: str):
        self.query = query[:200] + "..." if len(query) > 200 else query
        self.reason = reason
        message = f"Query failed on {service}: {reason}"
        super().__init__(message, service, {"query": self.query, "reason": reason})


class DatabaseTimeoutError(DatabaseError):
    """
    Database operation timed out.

    Raised when a query or connection attempt exceeds
    the configured timeout threshold.

    Attributes:
        timeout_seconds: The timeout value that was exceeded
        operation: The operation that timed out
    """

    def __init__(self, service: str, timeout_seconds: float, operation: str = "query"):
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        message = f"{service} {operation} timed out after {timeout_seconds}s"
        super().__init__(
            message, service,
            {"timeout_seconds": timeout_seconds, "operation": operation}
        )


# =============================================================================
# Memory Errors
# =============================================================================

class MemoryError(HippocampalError):
    """
    Base class for memory operation errors.

    Attributes:
        memory_type: The type of memory (episodic, semantic)
    """

    def __init__(
        self,
        message: str,
        memory_type: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.memory_type = memory_type
        full_details = {"memory_type": memory_type}
        if details:
            full_details.update(details)
        super().__init__(message, full_details)


class MemoryStoreError(MemoryError):
    """
    Failed to store a memory.

    Raised when a memory cannot be persisted to the
    vector database due to storage issues.
    """

    def __init__(self, memory_type: str, reason: str, memory_id: Optional[str] = None):
        self.reason = reason
        self.memory_id = memory_id
        message = f"Failed to store {memory_type} memory: {reason}"
        super().__init__(
            message, memory_type,
            {"reason": reason, "memory_id": memory_id}
        )


class MemoryQueryError(MemoryError):
    """
    Failed to query memories.

    Raised when a semantic search or retrieval
    operation fails.
    """

    def __init__(self, memory_type: str, query: str, reason: str):
        self.query = query
        self.reason = reason
        message = f"Failed to query {memory_type} memories: {reason}"
        super().__init__(
            message, memory_type,
            {"query": query[:100], "reason": reason}
        )


class MemoryValidationError(MemoryError):
    """
    Memory data failed validation.

    Raised when memory content or metadata does not
    meet the required schema or constraints.

    Attributes:
        field: The field that failed validation
        constraint: The constraint that was violated
    """

    def __init__(
        self,
        memory_type: str,
        field: str,
        constraint: str,
        value: Optional[Any] = None
    ):
        self.field = field
        self.constraint = constraint
        self.value = value
        message = f"Validation failed for {memory_type} memory field '{field}': {constraint}"
        super().__init__(
            message, memory_type,
            {"field": field, "constraint": constraint, "value": str(value)[:50] if value else None}
        )


# =============================================================================
# Identity Graph Errors
# =============================================================================

class IdentityGraphError(HippocampalError):
    """
    Base class for identity graph (Neo4j) errors.

    Attributes:
        concept: The concept/node involved in the error
    """

    def __init__(
        self,
        message: str,
        concept: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.concept = concept
        full_details = {"concept": concept} if concept else {}
        if details:
            full_details.update(details)
        super().__init__(message, full_details)


class GraphTraversalError(IdentityGraphError):
    """
    Failed to traverse the identity graph.

    Raised when graph traversal queries fail or
    return unexpected results.
    """

    def __init__(self, concept: str, depth: int, reason: str):
        self.depth = depth
        self.reason = reason
        message = f"Failed to traverse graph from '{concept}' (depth={depth}): {reason}"
        super().__init__(message, concept, {"depth": depth, "reason": reason})


class GraphUpdateError(IdentityGraphError):
    """
    Failed to update the identity graph.

    Raised when node or relationship creation/updates fail.
    """

    def __init__(
        self,
        from_concept: str,
        to_concept: str,
        relation: str,
        reason: str
    ):
        self.from_concept = from_concept
        self.to_concept = to_concept
        self.relation = relation
        self.reason = reason
        message = f"Failed to update graph: {from_concept} -[{relation}]-> {to_concept}: {reason}"
        super().__init__(
            message, from_concept,
            {"to_concept": to_concept, "relation": relation, "reason": reason}
        )


# =============================================================================
# Service Errors (Resilience Patterns)
# =============================================================================

class ServiceError(HippocampalError):
    """
    Base class for service-level errors.

    Used for resilience pattern failures like
    circuit breaker and retry exhaustion.

    Attributes:
        service_name: The name of the failing service
    """

    def __init__(
        self,
        message: str,
        service_name: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.service_name = service_name
        full_details = {"service_name": service_name}
        if details:
            full_details.update(details)
        super().__init__(message, full_details)


class CircuitOpenError(ServiceError):
    """
    Circuit breaker is open, rejecting requests.

    Raised when too many failures have occurred and
    the circuit breaker has tripped to prevent
    cascade failures.

    Attributes:
        failure_count: Number of failures that caused the circuit to open
        reset_time: When the circuit will attempt to close
    """

    def __init__(
        self,
        service_name: str,
        failure_count: int,
        reset_time: Optional[float] = None
    ):
        self.failure_count = failure_count
        self.reset_time = reset_time
        message = f"Circuit breaker open for {service_name} after {failure_count} failures"
        super().__init__(
            message, service_name,
            {"failure_count": failure_count, "reset_time": reset_time}
        )


class RetryExhaustedError(ServiceError):
    """
    All retry attempts have been exhausted.

    Raised when an operation has failed after all
    configured retry attempts with backoff.

    Attributes:
        attempts: Number of attempts made
        last_error: The last error encountered
    """

    def __init__(
        self,
        service_name: str,
        attempts: int,
        last_error: Optional[Exception] = None
    ):
        self.attempts = attempts
        self.last_error = last_error
        message = f"Exhausted {attempts} retry attempts for {service_name}"
        if last_error:
            message += f": {last_error}"
        super().__init__(
            message, service_name,
            {"attempts": attempts, "last_error": str(last_error) if last_error else None}
        )


# =============================================================================
# Frozen Voice Errors
# =============================================================================

class FrozenVoiceError(HippocampalError):
    """
    Operation blocked by Frozen Voice policy.

    Raised when attempting to modify semantic memories
    after the voice has been frozen (training complete).
    """

    def __init__(self, operation: str):
        self.operation = operation
        message = f"Operation '{operation}' blocked: Semantic memory is frozen"
        super().__init__(message, {"operation": operation, "policy": "frozen_voice"})


# =============================================================================
# Export All Exceptions
# =============================================================================

__all__ = [
    # Base
    "HippocampalError",
    # Configuration
    "ConfigurationError",
    # Database
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "DatabaseTimeoutError",
    # Memory
    "MemoryError",
    "MemoryStoreError",
    "MemoryQueryError",
    "MemoryValidationError",
    # Identity Graph
    "IdentityGraphError",
    "GraphTraversalError",
    "GraphUpdateError",
    # Service/Resilience
    "ServiceError",
    "CircuitOpenError",
    "RetryExhaustedError",
    # Frozen Voice
    "FrozenVoiceError",
]
