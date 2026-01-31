#!/usr/bin/env python3
"""
Ubik Somatic Node - MCP Client Connectivity Tests

Run all tests:
    python -m pytest tests/test_connectivity.py -v

Or run directly:
    python tests/test_connectivity.py
"""

import os
import sys
import asyncio
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_client import HippocampalClient


async def test_connection():
    """Test basic connectivity to Hippocampal node."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Connectivity")
    print("=" * 60)
    
    async with HippocampalClient() as client:
        print(f"  Target: {client.base_url}")
        
        connected = await client.is_connected()
        
        if connected:
            print("  ✓ PASSED - Connected to Hippocampal node")
            return True
        else:
            print("  ✗ FAILED - Cannot connect to Hippocampal node")
            return False


async def test_health_check():
    """Test health check and memory stats."""
    print("\n" + "=" * 60)
    print("TEST 2: Health Check & Memory Stats")
    print("=" * 60)
    
    async with HippocampalClient() as client:
        health = await client.health_check()
        
        if health.get("status") == "success":
            print(f"  ✓ Health check passed")
            print(f"    • Episodic count: {health.get('chromadb', {}).get('episodic_count', 'N/A')}")
            print(f"    • Semantic count: {health.get('chromadb', {}).get('semantic_count', 'N/A')}")
            print(f"    • Neo4j nodes: {health.get('neo4j', {}).get('node_count', 'N/A')}")
            print(f"    • Frozen: {health.get('semantic_frozen', False)}")
            return True
        else:
            print(f"  ✗ Health check failed: {health.get('message')}")
            return False


async def test_semantic_query():
    """Test semantic memory query."""
    print("\n" + "=" * 60)
    print("TEST 3: Semantic Memory Query")
    print("=" * 60)
    
    async with HippocampalClient() as client:
        results = await client.query_semantic("family values", n_results=3)
        
        if results:
            print(f"  ✓ Query returned {len(results)} results")
            for i, r in enumerate(results, 1):
                print(f"    {i}. [{r.metadata.get('category', 'N/A')}] {r.content[:50]}...")
            return True
        else:
            print("  ✗ Query returned no results")
            return False


async def test_episodic_query():
    """Test episodic memory query."""
    print("\n" + "=" * 60)
    print("TEST 4: Episodic Memory Query")
    print("=" * 60)
    
    async with HippocampalClient() as client:
        results = await client.query_episodic("grandchildren letter", n_results=3)
        
        if results:
            print(f"  ✓ Query returned {len(results)} results")
            for i, r in enumerate(results, 1):
                print(f"    {i}. [{r.metadata.get('type', 'N/A')}] {r.content[:50]}...")
            return True
        else:
            print("  ✗ Query returned no results")
            return False


async def test_identity_graph():
    """Test identity graph query."""
    print("\n" + "=" * 60)
    print("TEST 5: Identity Graph Query")
    print("=" * 60)
    
    async with HippocampalClient() as client:
        result = await client.get_identity_context("Self", depth=1)
        
        if result.get("status") == "success":
            paths = result.get("paths_found", 0)
            print(f"  ✓ Identity query returned {paths} paths")
            
            for path in result.get("context", [])[:3]:
                nodes = [n.get("name") for n in path.get("nodes", [])]
                print(f"    • {' -> '.join(nodes)}")
            return True
        else:
            print(f"  ✗ Identity query failed: {result.get('message')}")
            return False


async def test_rag_context():
    """Test RAG context generation."""
    print("\n" + "=" * 60)
    print("TEST 6: RAG Context Generation")
    print("=" * 60)
    
    async with HippocampalClient() as client:
        context = await client.get_rag_context(
            query="authenticity and family",
            episodic_results=2,
            semantic_results=2,
            include_identity=True
        )
        
        if context:
            print(f"  ✓ Generated {len(context)} characters of context")
            print(f"    Preview: {context[:200]}...")
            return True
        else:
            print("  ✗ No context generated")
            return False


async def run_all_tests():
    """Run all connectivity tests."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " UBIK SOMATIC NODE - MCP CLIENT TESTS ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    tests = [
        ("Connection", test_connection),
        ("Health Check", test_health_check),
        ("Semantic Query", test_semantic_query),
        ("Episodic Query", test_episodic_query),
        ("Identity Graph", test_identity_graph),
        ("RAG Context", test_rag_context),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = await test_func()
        except Exception as e:
            print(f"\n  ✗ EXCEPTION in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print(" TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, status in results.items():
        icon = "✓" if status else "✗"
        print(f"  {icon} {name}")
    
    print("\n" + "-" * 60)
    if passed == total:
        print(f"  ✓ ALL {total} TESTS PASSED")
    else:
        print(f"  ⚠ {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
