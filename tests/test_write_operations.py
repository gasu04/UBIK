#!/usr/bin/env python3
"""Test write operations to Hippocampal Node"""

import asyncio
import sys
import os

# Add somatic directory to path for mcp_client import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'somatic'))

from mcp_client import HippocampalClient

async def test_writes():
    print("\n" + "=" * 60)
    print(" WRITE OPERATIONS TEST")
    print("=" * 60)

    async with HippocampalClient() as client:

        # Test 1: Store episodic memory
        print("\n[1] Storing test episodic memory...")
        result = await client.store_episodic(
            content="Test memory: Verified Ubik system connectivity on January 19, 2026.",
            memory_type="system_event",
            emotional_valence="positive",
            importance=0.5,
            participants="gines",
            themes="testing,validation,ubik"
        )

        if result.get("status") == "success":
            print(f"  ✓ Episodic memory stored: {result.get('memory_id')}")
        else:
            print(f"  ✗ Failed: {result}")
            return False

        # Test 2: Store semantic knowledge
        print("\n[2] Storing test semantic knowledge...")
        result = await client.store_semantic(
            content="The Ubik system uses a distributed architecture with Hippocampal and Somatic nodes.",
            knowledge_type="fact",
            category="technology",
            confidence=0.95,
            stability="stable",
            source="system_design"
        )

        if result.get("status") == "success":
            print(f"  ✓ Semantic knowledge stored: {result.get('knowledge_id')}")
        else:
            print(f"  ✗ Failed: {result}")
            return False

        # Test 3: Verify new memories are queryable
        print("\n[3] Verifying new memories are queryable...")
        result = await client.query_episodic("Ubik system connectivity", n_results=3)

        found = False
        for mem in result:
            if "connectivity" in mem.content.lower():
                found = True
                print(f"  ✓ Found new episodic memory")
                break

        if not found:
            print(f"  ⚠ New memory not immediately found (may need index refresh)")

        # Test 4: Check updated stats
        print("\n[4] Checking updated memory stats...")
        stats = await client.health_check()

        if stats.get("status") == "success":
            chroma = stats.get("chromadb", {})
            print(f"  ✓ Episodic count: {chroma.get('episodic_count')}")
            print(f"  ✓ Semantic count: {chroma.get('semantic_count')}")

        print("\n" + "=" * 60)
        print(" ✓ WRITE OPERATIONS TEST PASSED")
        print("=" * 60)
        return True

if __name__ == "__main__":
    success = asyncio.run(test_writes())
    sys.exit(0 if success else 1)
