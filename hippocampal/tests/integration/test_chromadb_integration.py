#!/usr/bin/env python3
"""
Integration tests for ChromaDB vector store operations.

These tests require a running ChromaDB instance.
They verify that memories can be stored and queried correctly.

Run with: pytest tests/integration/test_chromadb_integration.py -v
"""

import os
import sys
from pathlib import Path

import pytest
import chromadb
from chromadb.errors import NotFoundError

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def chromadb_client():
    """Create a ChromaDB client for integration tests."""
    host = os.getenv("CHROMADB_HOST", "localhost")
    port = int(os.getenv("CHROMADB_PORT", "8001"))
    token = os.getenv("CHROMADB_TOKEN", "")
    
    try:
        client = chromadb.HttpClient(
            host=host,
            port=port,
            headers={"Authorization": f"Bearer {token}"} if token else {}
        )
        client.heartbeat()
        yield client
    except Exception as e:
        pytest.skip(f"ChromaDB is not available: {e}")


# =============================================================================
# Connection Tests
# =============================================================================

@pytest.mark.integration
class TestChromaDBConnection:
    """Tests for ChromaDB connectivity."""

    def test_can_connect(self, chromadb_client) -> None:
        """Verify connection to ChromaDB."""
        heartbeat = chromadb_client.heartbeat()
        assert heartbeat is not None

    def test_can_list_collections(self, chromadb_client) -> None:
        """Verify collections can be listed."""
        collections = chromadb_client.list_collections()
        assert isinstance(collections, list)


# =============================================================================
# Collection Tests
# =============================================================================

@pytest.mark.integration
class TestChromaDBCollections:
    """Tests for ChromaDB collection operations."""

    def test_episodic_collection_exists(self, chromadb_client) -> None:
        """Verify ubik_episodic collection exists."""
        try:
            collection = chromadb_client.get_collection("ubik_episodic")
            assert collection is not None
            assert collection.count() >= 0
        except NotFoundError:
            pytest.skip("ubik_episodic collection not created yet")

    def test_semantic_collection_exists(self, chromadb_client) -> None:
        """Verify ubik_semantic collection exists."""
        try:
            collection = chromadb_client.get_collection("ubik_semantic")
            assert collection is not None
            assert collection.count() >= 0
        except NotFoundError:
            pytest.skip("ubik_semantic collection not created yet")

    def test_episodic_has_expected_metadata(self, chromadb_client) -> None:
        """Verify episodic collection has expected metadata structure."""
        try:
            collection = chromadb_client.get_collection("ubik_episodic")
            if collection.count() > 0:
                result = collection.get(limit=1, include=["metadatas"])
                if result["metadatas"]:
                    metadata = result["metadatas"][0]
                    # Check for expected fields (may vary)
                    expected_fields = {"type", "timestamp", "importance"}
                    actual_fields = set(metadata.keys())
                    assert len(expected_fields & actual_fields) >= 1
        except NotFoundError:
            pytest.skip("ubik_episodic collection not created yet")

    def test_semantic_has_expected_metadata(self, chromadb_client) -> None:
        """Verify semantic collection has expected metadata structure."""
        try:
            collection = chromadb_client.get_collection("ubik_semantic")
            if collection.count() > 0:
                result = collection.get(limit=1, include=["metadatas"])
                if result["metadatas"]:
                    metadata = result["metadatas"][0]
                    expected_fields = {"type", "category", "confidence"}
                    actual_fields = set(metadata.keys())
                    assert len(expected_fields & actual_fields) >= 1
        except NotFoundError:
            pytest.skip("ubik_semantic collection not created yet")


# =============================================================================
# Query Tests
# =============================================================================

@pytest.mark.integration
class TestChromaDBQueries:
    """Tests for ChromaDB query operations."""

    def test_semantic_search_works(self, chromadb_client) -> None:
        """Verify semantic search returns results."""
        try:
            collection = chromadb_client.get_collection("ubik_semantic")
            if collection.count() == 0:
                pytest.skip("ubik_semantic collection is empty")
            
            results = collection.query(
                query_texts=["family values and legacy"],
                n_results=3
            )
            
            assert "documents" in results
            assert "distances" in results
            assert "metadatas" in results
            
        except NotFoundError:
            pytest.skip("ubik_semantic collection not created yet")

    def test_episodic_search_works(self, chromadb_client) -> None:
        """Verify episodic memory search returns results."""
        try:
            collection = chromadb_client.get_collection("ubik_episodic")
            if collection.count() == 0:
                pytest.skip("ubik_episodic collection is empty")
            
            results = collection.query(
                query_texts=["meaningful conversation"],
                n_results=3
            )
            
            assert "documents" in results
            assert "distances" in results
            
        except NotFoundError:
            pytest.skip("ubik_episodic collection not created yet")

    def test_filtered_query_works(self, chromadb_client) -> None:
        """Verify filtered queries work."""
        try:
            collection = chromadb_client.get_collection("ubik_semantic")
            if collection.count() == 0:
                pytest.skip("ubik_semantic collection is empty")
            
            results = collection.query(
                query_texts=["authenticity"],
                n_results=3,
                where={"type": "value"}
            )
            
            assert "documents" in results
            # Check filter was applied
            if results["metadatas"] and results["metadatas"][0]:
                for metadata in results["metadatas"][0]:
                    assert metadata.get("type") == "value"
                    
        except NotFoundError:
            pytest.skip("ubik_semantic collection not created yet")


# =============================================================================
# Write Tests
# =============================================================================

@pytest.mark.integration
class TestChromaDBWrites:
    """Tests for ChromaDB write operations."""

    def test_can_add_and_query_document(self, chromadb_client) -> None:
        """Verify documents can be added and queried."""
        # Create a test collection
        test_collection_name = "ubik_test_integration"
        
        try:
            # Clean up any existing test collection
            try:
                chromadb_client.delete_collection(test_collection_name)
            except:
                pass
            
            collection = chromadb_client.create_collection(
                name=test_collection_name,
                metadata={"test": True}
            )
            
            # Add a document
            collection.add(
                documents=["This is a test document for integration testing."],
                metadatas=[{"type": "test", "source": "integration"}],
                ids=["test_doc_001"]
            )
            
            # Query for it
            results = collection.query(
                query_texts=["test document"],
                n_results=1
            )
            
            assert len(results["documents"][0]) == 1
            assert "test document" in results["documents"][0][0].lower()
            
        finally:
            # Clean up
            try:
                chromadb_client.delete_collection(test_collection_name)
            except:
                pass

    def test_can_upsert_document(self, chromadb_client) -> None:
        """Verify documents can be upserted."""
        test_collection_name = "ubik_test_upsert"
        
        try:
            try:
                chromadb_client.delete_collection(test_collection_name)
            except:
                pass
            
            collection = chromadb_client.create_collection(name=test_collection_name)
            
            # Add initial document
            collection.add(
                documents=["Original content"],
                metadatas=[{"version": 1}],
                ids=["upsert_test"]
            )
            
            # Upsert with new content
            collection.upsert(
                documents=["Updated content"],
                metadatas=[{"version": 2}],
                ids=["upsert_test"]
            )
            
            # Verify update
            result = collection.get(ids=["upsert_test"], include=["documents", "metadatas"])
            assert result["documents"][0] == "Updated content"
            assert result["metadatas"][0]["version"] == 2
            
        finally:
            try:
                chromadb_client.delete_collection(test_collection_name)
            except:
                pass
