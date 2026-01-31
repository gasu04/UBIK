#!/usr/bin/env python3
"""
Pytest Configuration for Ubik Hippocampal Node Tests

Provides shared fixtures for unit and integration tests.

Fixtures:
    - mock_chromadb_client: Mocked ChromaDB client
    - mock_neo4j_driver: Mocked Neo4j driver
    - sample_episodic_memory: Sample episodic memory data
    - sample_semantic_knowledge: Sample semantic knowledge data
    - test_config: Test configuration with mock values
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, MagicMock, AsyncMock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Environment Configuration
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """Set up test environment variables."""
    original_env = os.environ.copy()

    # Set test environment variables
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "test_password"
    os.environ["CHROMADB_HOST"] = "localhost"
    os.environ["CHROMADB_PORT"] = "8001"
    os.environ["CHROMADB_TOKEN"] = "test_token"
    os.environ["LOG_LEVEL"] = "DEBUG"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# =============================================================================
# Mock Database Clients
# =============================================================================

@pytest.fixture
def mock_chromadb_client() -> Mock:
    """Create a mocked ChromaDB client."""
    client = Mock()
    client.heartbeat.return_value = True

    # Mock collection
    collection = Mock()
    collection.count.return_value = 10
    collection.add.return_value = None
    collection.upsert.return_value = None
    collection.query.return_value = {
        "ids": [["mem_001", "mem_002"]],
        "documents": [["Test memory 1", "Test memory 2"]],
        "metadatas": [[
            {"type": "event", "importance": 0.8},
            {"type": "letter", "importance": 0.9}
        ]],
        "distances": [[0.1, 0.2]]
    }

    client.get_collection.return_value = collection
    client.get_or_create_collection.return_value = collection
    client.list_collections.return_value = [collection]

    return client


@pytest.fixture
def mock_neo4j_driver() -> Mock:
    """Create a mocked Neo4j driver."""
    driver = Mock()

    # Mock session context manager
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)

    # Mock query result
    result = Mock()
    result.single.return_value = {
        "source": "Self",
        "relation": "HAS_VALUE",
        "target": "Family"
    }
    result.__iter__ = Mock(return_value=iter([
        {
            "source": "Self",
            "relationship_types": ["HAS_VALUE"],
            "nodes": [
                {"name": "Self", "type": "Concept", "properties": {}},
                {"name": "Family", "type": "Value", "properties": {}}
            ],
            "distance": 1
        }
    ]))

    session.run.return_value = result
    driver.session.return_value = session

    return driver


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_episodic_memory() -> Dict[str, Any]:
    """Sample episodic memory for testing."""
    return {
        "content": "Today I had a meaningful conversation with my father about legacy.",
        "memory_type": "conversation",
        "timestamp": "2024-01-15T10:30:00Z",
        "emotional_valence": "reflective",
        "importance": 0.85,
        "participants": "gines,father",
        "themes": "legacy,family,wisdom"
    }


@pytest.fixture
def sample_semantic_knowledge() -> Dict[str, Any]:
    """Sample semantic knowledge for testing."""
    return {
        "content": "Family legacy transcends material possessions.",
        "knowledge_type": "value",
        "category": "family",
        "confidence": 0.95,
        "stability": "core",
        "source": "reflection"
    }


@pytest.fixture
def sample_identity_relation() -> Dict[str, Any]:
    """Sample identity graph relation for testing."""
    return {
        "from_concept": "Self",
        "relation_type": "HAS_VALUE",
        "to_concept": "Authenticity",
        "weight": 0.9,
        "context": "Core identity value"
    }


# =============================================================================
# Async Fixtures
# =============================================================================

@pytest.fixture
def async_mock() -> AsyncMock:
    """Create an AsyncMock for async function testing."""
    return AsyncMock()


# =============================================================================
# Test Configuration
# =============================================================================

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration values."""
    return {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "test_password",
        "chromadb_host": "localhost",
        "chromadb_port": 8001,
        "chromadb_token": "test_token",
        "mcp_host": "0.0.0.0",
        "mcp_port": 8080,
        "log_level": "DEBUG"
    }


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
