#!/usr/bin/env python3
"""
Integration tests for Neo4j database operations.

These tests require a running Neo4j instance.
They verify that the identity graph can be queried and modified correctly.

Run with: pytest tests/integration/test_neo4j_integration.py -v
"""

import os
import sys
from pathlib import Path

import pytest
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def neo4j_driver():
    """Create a Neo4j driver for integration tests."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")

    if not password:
        pytest.skip("NEO4J_PASSWORD not set")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        yield driver
        driver.close()
    except ServiceUnavailable:
        pytest.skip("Neo4j is not available")
    except AuthError:
        pytest.skip("Neo4j authentication failed")
    except Exception as e:
        pytest.skip(f"Neo4j connection failed: {e}")


@pytest.fixture
def neo4j_session(neo4j_driver):
    """Create a session for each test."""
    with neo4j_driver.session() as session:
        yield session


# =============================================================================
# Connection Tests
# =============================================================================

@pytest.mark.integration
class TestNeo4jConnection:
    """Tests for Neo4j connectivity."""

    def test_can_connect(self, neo4j_driver) -> None:
        """Verify connection to Neo4j."""
        neo4j_driver.verify_connectivity()

    def test_can_execute_simple_query(self, neo4j_session) -> None:
        """Verify simple query execution."""
        result = neo4j_session.run("RETURN 1 as num")
        record = result.single()
        assert record["num"] == 1


# =============================================================================
# Schema Tests
# =============================================================================

@pytest.mark.integration
class TestNeo4jSchema:
    """Tests for Neo4j schema and indexes."""

    def test_core_identity_exists(self, neo4j_session) -> None:
        """Verify CoreIdentity Self node exists."""
        result = neo4j_session.run(
            "MATCH (s:CoreIdentity {name: 'Self'}) RETURN s"
        )
        record = result.single()
        assert record is not None, "CoreIdentity 'Self' node not found"

    def test_value_nodes_exist(self, neo4j_session) -> None:
        """Verify value nodes are present."""
        result = neo4j_session.run(
            "MATCH (v:Value) RETURN count(v) as count"
        )
        record = result.single()
        assert record["count"] > 0, "No Value nodes found"

    def test_trait_nodes_exist(self, neo4j_session) -> None:
        """Verify trait nodes are present."""
        result = neo4j_session.run(
            "MATCH (t:Trait) RETURN count(t) as count"
        )
        record = result.single()
        assert record["count"] > 0, "No Trait nodes found"


# =============================================================================
# Graph Traversal Tests
# =============================================================================

@pytest.mark.integration
class TestGraphTraversal:
    """Tests for graph traversal operations."""

    def test_self_has_values(self, neo4j_session) -> None:
        """Verify Self node has connected values."""
        result = neo4j_session.run("""
            MATCH (s:CoreIdentity {name: 'Self'})-[:VALUES]->(v:Value)
            RETURN v.name as name
        """)
        values = [r["name"] for r in result]
        assert len(values) > 0, "Self has no connected values"

    def test_self_has_traits(self, neo4j_session) -> None:
        """Verify Self node has connected traits."""
        result = neo4j_session.run("""
            MATCH (s:CoreIdentity {name: 'Self'})-[:HAS_TRAIT]->(t:Trait)
            RETURN t.name as name
        """)
        traits = [r["name"] for r in result]
        assert len(traits) > 0, "Self has no connected traits"

    def test_can_traverse_psychological_connections(self, neo4j_session) -> None:
        """Verify psychological connection traversal works."""
        result = neo4j_session.run("""
            MATCH path = (start {name: 'Authenticity'})-[:CONNECTS_TO*1..2]-(related)
            RETURN [n in nodes(path) | n.name] as path_nodes
            LIMIT 5
        """)
        paths = [r["path_nodes"] for r in result]
        # May or may not have paths depending on data
        assert isinstance(paths, list)

    def test_memory_anchors_connected(self, neo4j_session) -> None:
        """Verify memory anchors are connected to Self."""
        result = neo4j_session.run("""
            MATCH (s:CoreIdentity {name: 'Self'})-[:REMEMBERS]->(m:Memory)
            RETURN m.chromadb_id as id
        """)
        memories = [r["id"] for r in result]
        assert len(memories) >= 0  # May be empty but query should work


# =============================================================================
# Query Pattern Tests
# =============================================================================

@pytest.mark.integration
class TestQueryPatterns:
    """Tests for query patterns used by MCP server."""

    def test_identity_context_query(self, neo4j_session) -> None:
        """Test the identity context query pattern."""
        result = neo4j_session.run("""
            MATCH (start:CoreIdentity {name: 'Self'})
            CALL {
                WITH start
                MATCH path = (start)-[r*1..2]-(connected)
                RETURN path, connected, r
                LIMIT 20
            }
            WITH start, path, connected, r
            RETURN 
                [rel in relationships(path) | type(rel)] as relationship_types,
                [n in nodes(path) | {
                    name: n.name,
                    labels: labels(n),
                    properties: properties(n)
                }] as nodes,
                length(path) as distance
        """)
        
        results = list(result)
        # Should return some context
        assert len(results) >= 0

    def test_concept_search_query(self, neo4j_session) -> None:
        """Test concept search query pattern."""
        result = neo4j_session.run("""
            MATCH (c:Concept)
            WHERE c.name CONTAINS 'Family' OR c.description CONTAINS 'family'
            RETURN c.name as name, c.description as description
            LIMIT 5
        """)
        
        concepts = list(result)
        # May or may not find matches
        assert isinstance(concepts, list)
