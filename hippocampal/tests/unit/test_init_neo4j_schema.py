#!/usr/bin/env python3
"""
Unit tests for Neo4j schema initialization module.

Tests configuration, schema creation, node creation, and verification
using mocked Neo4j driver.
"""

import os
from unittest.mock import Mock, patch, MagicMock, call

import pytest

from init_neo4j_schema import (
    Neo4jSchemaConfig,
    init_schema,
    create_core_identity_nodes,
    create_parfitian_structure,
    create_memory_anchors,
    verify_schema,
    main,
)
from exceptions import ConfigurationError


# =============================================================================
# Neo4jSchemaConfig Tests
# =============================================================================

class TestNeo4jSchemaConfig:
    """Tests for Neo4jSchemaConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = Neo4jSchemaConfig()

            assert config.uri == "bolt://localhost:7687"
            assert config.user == "neo4j"
            assert config.password == ""

    def test_loads_from_environment(self) -> None:
        """Test loading values from environment variables."""
        env = {
            "NEO4J_URI": "bolt://neo4j.example.com:7687",
            "NEO4J_USER": "admin",
            "NEO4J_PASSWORD": "secret123",
        }

        with patch.dict(os.environ, env, clear=True):
            config = Neo4jSchemaConfig()

            assert config.uri == "bolt://neo4j.example.com:7687"
            assert config.user == "admin"
            assert config.password == "secret123"

    def test_validate_passes_with_password(self) -> None:
        """Test validation passes when password is set."""
        config = Neo4jSchemaConfig()
        config.password = "valid_password"

        result = config.validate()

        assert result is True

    def test_validate_fails_without_password(self) -> None:
        """Test validation fails when password is empty."""
        config = Neo4jSchemaConfig()
        config.password = ""

        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()

        assert exc_info.value.config_key == "NEO4J_PASSWORD"


# =============================================================================
# Schema Initialization Tests
# =============================================================================

class TestInitSchema:
    """Tests for init_schema function."""

    def test_creates_constraints(self) -> None:
        """Test that constraints are created."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_driver.session.return_value = mock_session

        init_schema(mock_driver)

        # Verify session.run was called multiple times for constraints/indexes
        assert mock_session.run.call_count > 0

        # Check that constraint queries were run
        calls = [str(c) for c in mock_session.run.call_args_list]
        constraint_calls = [c for c in calls if "CONSTRAINT" in c]
        assert len(constraint_calls) >= 4  # At least 4 constraints

    def test_creates_indexes(self) -> None:
        """Test that indexes are created."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_driver.session.return_value = mock_session

        init_schema(mock_driver)

        calls = [str(c) for c in mock_session.run.call_args_list]
        index_calls = [c for c in calls if "INDEX" in c]
        assert len(index_calls) >= 5  # At least 5 indexes

    def test_handles_existing_constraint(self) -> None:
        """Test graceful handling when constraint already exists."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_session.run.side_effect = Exception("Constraint already exists")
        mock_driver.session.return_value = mock_session

        # Should not raise
        init_schema(mock_driver)


# =============================================================================
# Core Identity Nodes Tests
# =============================================================================

class TestCreateCoreIdentityNodes:
    """Tests for create_core_identity_nodes function."""

    def test_creates_self_node(self) -> None:
        """Test that Self node is created."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_driver.session.return_value = mock_session

        create_core_identity_nodes(mock_driver)

        # Check that a query containing 'Self' was executed
        calls = [str(c) for c in mock_session.run.call_args_list]
        self_calls = [c for c in calls if "CoreIdentity" in c and "Self" in c]
        assert len(self_calls) >= 1

    def test_creates_core_values(self) -> None:
        """Test that core value nodes are created."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_driver.session.return_value = mock_session

        create_core_identity_nodes(mock_driver)

        calls = [str(c) for c in mock_session.run.call_args_list]
        value_calls = [c for c in calls if "Value" in c]
        assert len(value_calls) >= 5  # At least 5 core values

    def test_creates_traits(self) -> None:
        """Test that trait nodes are created."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_driver.session.return_value = mock_session

        create_core_identity_nodes(mock_driver)

        calls = [str(c) for c in mock_session.run.call_args_list]
        trait_calls = [c for c in calls if "Trait" in c]
        assert len(trait_calls) >= 4  # At least 4 traits

    def test_creates_relationships(self) -> None:
        """Test that relationship nodes are created."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_driver.session.return_value = mock_session

        create_core_identity_nodes(mock_driver)

        calls = [str(c) for c in mock_session.run.call_args_list]
        rel_calls = [c for c in calls if "Relationship" in c]
        assert len(rel_calls) >= 3  # At least 3 relationships


# =============================================================================
# Parfitian Structure Tests
# =============================================================================

class TestCreateParfitianStructure:
    """Tests for create_parfitian_structure function."""

    def test_creates_timeslice_template(self) -> None:
        """Test that TimeSlice template is created."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_driver.session.return_value = mock_session

        create_parfitian_structure(mock_driver)

        calls = [str(c) for c in mock_session.run.call_args_list]
        ts_calls = [c for c in calls if "TimeSlice" in c]
        assert len(ts_calls) >= 1

    def test_creates_psychological_connections(self) -> None:
        """Test that psychological connections are created."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_driver.session.return_value = mock_session

        create_parfitian_structure(mock_driver)

        calls = [str(c) for c in mock_session.run.call_args_list]
        connect_calls = [c for c in calls if "CONNECTS_TO" in c]
        assert len(connect_calls) >= 7  # At least 7 psychological connections


# =============================================================================
# Memory Anchors Tests
# =============================================================================

class TestCreateMemoryAnchors:
    """Tests for create_memory_anchors function."""

    def test_creates_memory_nodes(self) -> None:
        """Test that memory anchor nodes are created."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_driver.session.return_value = mock_session

        create_memory_anchors(mock_driver)

        calls = [str(c) for c in mock_session.run.call_args_list]
        memory_calls = [c for c in calls if "Memory" in c]
        assert len(memory_calls) >= 3  # At least 3 memory anchors

    def test_links_memories_to_self(self) -> None:
        """Test that memories are linked to Self via REMEMBERS."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_driver.session.return_value = mock_session

        create_memory_anchors(mock_driver)

        calls = [str(c) for c in mock_session.run.call_args_list]
        remembers_calls = [c for c in calls if "REMEMBERS" in c]
        assert len(remembers_calls) >= 1

    def test_links_memories_to_concepts(self) -> None:
        """Test that memories are linked to relevant concepts."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_driver.session.return_value = mock_session

        create_memory_anchors(mock_driver)

        calls = [str(c) for c in mock_session.run.call_args_list]
        supports_calls = [c for c in calls if "SUPPORTS" in c]
        assert len(supports_calls) >= 2


# =============================================================================
# Verification Tests
# =============================================================================

class TestVerifySchema:
    """Tests for verify_schema function."""

    def test_runs_verification_queries(self) -> None:
        """Test that verification queries are executed."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        # Mock query results
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {"type": "Concept", "count": 10},
            {"type": "Value", "count": 5}
        ]))
        mock_session.run.return_value = mock_result

        mock_driver.session.return_value = mock_session

        result = verify_schema(mock_driver)

        assert result is True
        assert mock_session.run.call_count >= 3  # At least 3 verification queries

    def test_counts_nodes_by_type(self) -> None:
        """Test that node counts are retrieved."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        mock_session.run.return_value = mock_result

        mock_driver.session.return_value = mock_session

        verify_schema(mock_driver)

        calls = [str(c) for c in mock_session.run.call_args_list]
        count_calls = [c for c in calls if "count(n)" in c]
        assert len(count_calls) >= 1


# =============================================================================
# Main Function Tests
# =============================================================================

class TestMain:
    """Tests for main function."""

    @patch('init_neo4j_schema.verify_schema')
    @patch('init_neo4j_schema.create_memory_anchors')
    @patch('init_neo4j_schema.create_parfitian_structure')
    @patch('init_neo4j_schema.create_core_identity_nodes')
    @patch('init_neo4j_schema.init_schema')
    @patch('init_neo4j_schema.GraphDatabase')
    def test_successful_initialization(
        self,
        mock_gd,
        mock_init,
        mock_core,
        mock_parfit,
        mock_memory,
        mock_verify
    ) -> None:
        """Test successful schema initialization."""
        mock_driver = Mock()
        mock_gd.driver.return_value = mock_driver

        config = Neo4jSchemaConfig()
        config.password = "test_password"

        result = main(config)

        assert result is True
        mock_driver.verify_connectivity.assert_called_once()
        mock_init.assert_called_once_with(mock_driver)
        mock_core.assert_called_once_with(mock_driver)
        mock_parfit.assert_called_once_with(mock_driver)
        mock_memory.assert_called_once_with(mock_driver)
        mock_verify.assert_called_once_with(mock_driver)
        mock_driver.close.assert_called_once()

    def test_fails_without_password(self) -> None:
        """Test failure when password is missing."""
        config = Neo4jSchemaConfig()
        config.password = ""

        result = main(config)

        assert result is False

    @patch('init_neo4j_schema.GraphDatabase')
    def test_handles_connection_error(self, mock_gd) -> None:
        """Test handling of connection error."""
        mock_gd.driver.side_effect = Exception("Connection refused")

        config = Neo4jSchemaConfig()
        config.password = "test_password"

        result = main(config)

        assert result is False

    @patch('init_neo4j_schema.GraphDatabase')
    def test_closes_driver_on_error(self, mock_gd) -> None:
        """Test that driver is closed even on error."""
        mock_driver = Mock()
        mock_driver.verify_connectivity.side_effect = Exception("Auth failed")
        mock_gd.driver.return_value = mock_driver

        config = Neo4jSchemaConfig()
        config.password = "test_password"

        main(config)

        mock_driver.close.assert_called_once()

    @patch('init_neo4j_schema.verify_schema')
    @patch('init_neo4j_schema.create_memory_anchors')
    @patch('init_neo4j_schema.create_parfitian_structure')
    @patch('init_neo4j_schema.create_core_identity_nodes')
    @patch('init_neo4j_schema.init_schema')
    @patch('init_neo4j_schema.GraphDatabase')
    def test_uses_default_config(
        self,
        mock_gd,
        mock_init,
        mock_core,
        mock_parfit,
        mock_memory,
        mock_verify
    ) -> None:
        """Test that default config is used when none provided."""
        mock_driver = Mock()
        mock_gd.driver.return_value = mock_driver

        # Set environment variable for password
        with patch.dict(os.environ, {"NEO4J_PASSWORD": "env_password"}):
            result = main()  # No config provided

            assert result is True


# =============================================================================
# Integration-style Tests (Still Unit Tests with Mocks)
# =============================================================================

class TestSchemaCreationFlow:
    """Tests for the complete schema creation flow."""

    @patch('init_neo4j_schema.GraphDatabase')
    def test_full_flow_executes_in_order(self, mock_gd) -> None:
        """Test that full flow executes steps in correct order."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        mock_driver.session.return_value = mock_session
        mock_gd.driver.return_value = mock_driver

        config = Neo4jSchemaConfig()
        config.password = "test_password"

        result = main(config)

        assert result is True

        # Verify all Cypher queries were executed
        assert mock_session.run.call_count > 20  # Many queries expected
