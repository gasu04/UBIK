#!/usr/bin/env python3
"""
Unit tests for custom exception hierarchy.

Tests all exception classes to ensure proper inheritance,
message formatting, and attribute handling.
"""

import pytest
from exceptions import (
    HippocampalError,
    ConfigurationError,
    DatabaseError,
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseTimeoutError,
    MemoryError,
    MemoryStoreError,
    MemoryQueryError,
    MemoryValidationError,
    IdentityGraphError,
    GraphTraversalError,
    GraphUpdateError,
    ServiceError,
    CircuitOpenError,
    RetryExhaustedError,
    FrozenVoiceError,
)


class TestHippocampalError:
    """Tests for base HippocampalError."""

    def test_basic_message(self) -> None:
        """Test basic error message."""
        error = HippocampalError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}

    def test_with_details(self) -> None:
        """Test error with additional details."""
        error = HippocampalError("Test error", {"key": "value"})
        assert "key" in str(error)
        assert error.details == {"key": "value"}


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_missing_config(self) -> None:
        """Test error for missing configuration."""
        error = ConfigurationError("Missing required config", "API_KEY")
        assert error.config_key == "API_KEY"
        assert "API_KEY" in str(error)

    def test_without_config_key(self) -> None:
        """Test error without specific config key."""
        error = ConfigurationError("Invalid configuration")
        assert error.config_key is None


class TestDatabaseErrors:
    """Tests for database-related errors."""

    def test_connection_error(self) -> None:
        """Test DatabaseConnectionError attributes."""
        error = DatabaseConnectionError(
            service="neo4j",
            host="localhost",
            port=7687,
            reason="Connection refused"
        )
        assert error.service == "neo4j"
        assert error.host == "localhost"
        assert error.port == 7687
        assert "Connection refused" in str(error)

    def test_query_error_truncation(self) -> None:
        """Test that long queries are truncated."""
        long_query = "MATCH " + "n" * 300
        error = DatabaseQueryError(
            service="neo4j",
            query=long_query,
            reason="Syntax error"
        )
        assert len(error.query) <= 203  # 200 + "..."
        assert error.query.endswith("...")

    def test_timeout_error(self) -> None:
        """Test DatabaseTimeoutError attributes."""
        error = DatabaseTimeoutError(
            service="chromadb",
            timeout_seconds=30.0,
            operation="query"
        )
        assert error.timeout_seconds == 30.0
        assert error.operation == "query"
        assert "30" in str(error)


class TestMemoryErrors:
    """Tests for memory operation errors."""

    def test_store_error(self) -> None:
        """Test MemoryStoreError attributes."""
        error = MemoryStoreError(
            memory_type="episodic",
            reason="Disk full",
            memory_id="mem_001"
        )
        assert error.memory_type == "episodic"
        assert error.memory_id == "mem_001"
        assert "episodic" in str(error)

    def test_query_error(self) -> None:
        """Test MemoryQueryError attributes."""
        error = MemoryQueryError(
            memory_type="semantic",
            query="family values",
            reason="Embedding failed"
        )
        assert error.query == "family values"
        assert error.reason == "Embedding failed"

    def test_validation_error(self) -> None:
        """Test MemoryValidationError attributes."""
        error = MemoryValidationError(
            memory_type="episodic",
            field="importance",
            constraint="must be between 0 and 1",
            value=1.5
        )
        assert error.field == "importance"
        assert error.constraint == "must be between 0 and 1"


class TestIdentityGraphErrors:
    """Tests for identity graph errors."""

    def test_traversal_error(self) -> None:
        """Test GraphTraversalError attributes."""
        error = GraphTraversalError(
            concept="Self",
            depth=3,
            reason="No path found"
        )
        assert error.concept == "Self"
        assert error.depth == 3

    def test_update_error(self) -> None:
        """Test GraphUpdateError attributes."""
        error = GraphUpdateError(
            from_concept="Self",
            to_concept="Family",
            relation="HAS_VALUE",
            reason="Constraint violation"
        )
        assert error.from_concept == "Self"
        assert error.to_concept == "Family"
        assert error.relation == "HAS_VALUE"


class TestServiceErrors:
    """Tests for service/resilience errors."""

    def test_circuit_open_error(self) -> None:
        """Test CircuitOpenError attributes."""
        error = CircuitOpenError(
            service_name="neo4j",
            failure_count=5,
            reset_time=60.0
        )
        assert error.failure_count == 5
        assert error.reset_time == 60.0

    def test_retry_exhausted_error(self) -> None:
        """Test RetryExhaustedError with last error."""
        last_error = ConnectionError("Network down")
        error = RetryExhaustedError(
            service_name="chromadb",
            attempts=3,
            last_error=last_error
        )
        assert error.attempts == 3
        assert error.last_error is last_error
        assert "Network down" in str(error)


class TestFrozenVoiceError:
    """Tests for Frozen Voice policy error."""

    def test_frozen_voice_error(self) -> None:
        """Test FrozenVoiceError attributes."""
        error = FrozenVoiceError("store_semantic")
        assert error.operation == "store_semantic"
        assert "frozen" in str(error).lower()


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_all_inherit_from_base(self) -> None:
        """Verify all exceptions inherit from HippocampalError."""
        exceptions = [
            ConfigurationError("test"),
            DatabaseError("test", "neo4j"),
            DatabaseConnectionError("neo4j", "localhost", 7687),
            MemoryError("test", "episodic"),
            MemoryStoreError("episodic", "reason"),
            IdentityGraphError("test"),
            ServiceError("test", "service"),
            FrozenVoiceError("operation"),
        ]
        for exc in exceptions:
            assert isinstance(exc, HippocampalError)

    def test_database_errors_inherit_from_database_error(self) -> None:
        """Verify database exceptions inherit correctly."""
        exceptions = [
            DatabaseConnectionError("neo4j", "localhost", 7687),
            DatabaseQueryError("neo4j", "query", "reason"),
            DatabaseTimeoutError("neo4j", 30.0),
        ]
        for exc in exceptions:
            assert isinstance(exc, DatabaseError)

    def test_memory_errors_inherit_from_memory_error(self) -> None:
        """Verify memory exceptions inherit correctly."""
        exceptions = [
            MemoryStoreError("episodic", "reason"),
            MemoryQueryError("semantic", "query", "reason"),
            MemoryValidationError("episodic", "field", "constraint"),
        ]
        for exc in exceptions:
            assert isinstance(exc, MemoryError)
