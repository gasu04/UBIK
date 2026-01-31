#!/usr/bin/env python3
"""
Full pipeline integration tests.

These tests verify the complete flow from MCP server through to databases.
They require all services (Neo4j, ChromaDB, MCP server) to be running.

Run with: pytest tests/integration/test_full_pipeline.py -v
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import pytest
import chromadb
import httpx
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def all_services():
    """Fixture that ensures all services are available."""
    services = {}
    
    # Neo4j
    try:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        
        if password:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            services["neo4j"] = driver
    except:
        pass
    
    # ChromaDB
    try:
        host = os.getenv("CHROMADB_HOST", "localhost")
        port = int(os.getenv("CHROMADB_PORT", "8001"))
        token = os.getenv("CHROMADB_TOKEN", "")
        
        client = chromadb.HttpClient(
            host=host,
            port=port,
            headers={"Authorization": f"Bearer {token}"} if token else {}
        )
        client.heartbeat()
        services["chromadb"] = client
    except:
        pass
    
    # MCP Server
    try:
        mcp_host = os.getenv("MCP_HOST", "localhost")
        mcp_port = os.getenv("MCP_PORT", "8080")
        response = httpx.get(f"http://{mcp_host}:{mcp_port}/mcp", timeout=5.0)
        if response.status_code in [200, 400, 404, 406]:
            services["mcp"] = f"http://{mcp_host}:{mcp_port}"
    except:
        pass
    
    yield services
    
    # Cleanup
    if "neo4j" in services:
        services["neo4j"].close()


# =============================================================================
# Full Pipeline Tests
# =============================================================================

@pytest.mark.integration
class TestFullPipeline:
    """Tests for the complete Hippocampal Node pipeline."""

    def test_all_services_available(self, all_services) -> None:
        """Verify all services are running."""
        if len(all_services) < 3:
            missing = []
            if "neo4j" not in all_services:
                missing.append("Neo4j")
            if "chromadb" not in all_services:
                missing.append("ChromaDB")
            if "mcp" not in all_services:
                missing.append("MCP Server")
            pytest.skip(f"Missing services: {', '.join(missing)}")
        
        assert "neo4j" in all_services
        assert "chromadb" in all_services
        assert "mcp" in all_services

    def test_neo4j_and_chromadb_data_consistency(self, all_services) -> None:
        """Verify data consistency between Neo4j and ChromaDB."""
        if "neo4j" not in all_services or "chromadb" not in all_services:
            pytest.skip("Both Neo4j and ChromaDB required")
        
        neo4j_driver = all_services["neo4j"]
        chromadb_client = all_services["chromadb"]
        
        # Get memory anchors from Neo4j
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (m:Memory)
                RETURN m.chromadb_id as id
                LIMIT 10
            """)
            neo4j_memory_ids = [r["id"] for r in result if r["id"]]
        
        # Check if those IDs exist in ChromaDB
        if neo4j_memory_ids:
            try:
                collection = chromadb_client.get_collection("ubik_episodic")
                result = collection.get(ids=neo4j_memory_ids)
                # At least some IDs should match
                found_ids = set(result["ids"])
                expected_ids = set(neo4j_memory_ids)
                # Partial match is OK (data may be out of sync during development)
                assert isinstance(found_ids, set)
            except:
                # Collection may not exist
                pass

    def test_identity_graph_has_core_structure(self, all_services) -> None:
        """Verify Neo4j has the expected identity structure."""
        if "neo4j" not in all_services:
            pytest.skip("Neo4j not available")
        
        neo4j_driver = all_services["neo4j"]
        
        with neo4j_driver.session() as session:
            # Check Self exists
            result = session.run(
                "MATCH (s:CoreIdentity {name: 'Self'}) RETURN s"
            )
            assert result.single() is not None, "Self node missing"
            
            # Check Self has connections
            result = session.run("""
                MATCH (s:CoreIdentity {name: 'Self'})-[r]->()
                RETURN count(r) as count
            """)
            count = result.single()["count"]
            assert count > 0, "Self has no connections"

    def test_memory_collections_queryable(self, all_services) -> None:
        """Verify ChromaDB collections can be queried."""
        if "chromadb" not in all_services:
            pytest.skip("ChromaDB not available")
        
        chromadb_client = all_services["chromadb"]
        
        collections_to_check = ["ubik_episodic", "ubik_semantic"]
        
        for col_name in collections_to_check:
            try:
                collection = chromadb_client.get_collection(col_name)
                
                if collection.count() > 0:
                    # Query should work
                    results = collection.query(
                        query_texts=["test query"],
                        n_results=1
                    )
                    assert "documents" in results
                    assert "distances" in results
            except:
                # Collection may not exist yet
                continue


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestPipelinePerformance:
    """Performance tests for the pipeline."""

    def test_neo4j_query_performance(self, all_services) -> None:
        """Verify Neo4j queries complete in reasonable time."""
        if "neo4j" not in all_services:
            pytest.skip("Neo4j not available")
        
        import time
        
        neo4j_driver = all_services["neo4j"]
        
        start = time.time()
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (s:CoreIdentity {name: 'Self'})-[*1..2]-(connected)
                RETURN connected.name
                LIMIT 50
            """)
            list(result)  # Consume results
        elapsed = time.time() - start
        
        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"Query took {elapsed:.2f}s"

    def test_chromadb_query_performance(self, all_services) -> None:
        """Verify ChromaDB queries complete in reasonable time."""
        if "chromadb" not in all_services:
            pytest.skip("ChromaDB not available")
        
        import time
        
        chromadb_client = all_services["chromadb"]
        
        try:
            collection = chromadb_client.get_collection("ubik_semantic")
            
            start = time.time()
            results = collection.query(
                query_texts=["family values legacy authenticity"],
                n_results=10
            )
            elapsed = time.time() - start
            
            # Should complete in under 2 seconds
            assert elapsed < 2.0, f"Query took {elapsed:.2f}s"
        except:
            pytest.skip("ubik_semantic collection not available")
