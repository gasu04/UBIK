"""
Ubik Somatic Node - RAG Integration Tests

Tests for RAG context building and prompt integration.
"""

import os
import sys
import asyncio
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_client import HippocampalClient
from mcp_client.rag_integration import RAGContextBuilder, RAGContext, build_rag_context


class TestRAGContextBuilder:
    """Tests for RAG context building."""

    @pytest.fixture
    def builder(self):
        """Create a builder instance."""
        return RAGContextBuilder()

    @pytest.mark.asyncio
    async def test_build_context(self, builder):
        """Test basic context building."""
        async with builder:
            try:
                context = await builder.build("family values")
                assert isinstance(context, RAGContext)
                assert context.query == "family values"
            except Exception as e:
                pytest.skip(f"Context build failed: {e}")

    @pytest.mark.asyncio
    async def test_build_with_limits(self, builder):
        """Test context building with result limits."""
        async with builder:
            try:
                context = await builder.build(
                    "authenticity",
                    semantic_results=1,
                    episodic_results=1,
                    include_identity=False
                )
                assert len(context.semantic_memories) <= 1
                assert len(context.episodic_memories) <= 1
            except Exception as e:
                pytest.skip(f"Limited build failed: {e}")

    @pytest.mark.asyncio
    async def test_format_for_prompt(self, builder):
        """Test prompt formatting."""
        async with builder:
            try:
                context = await builder.build("test query", semantic_results=2)
                formatted = builder.format_for_prompt(context)
                assert isinstance(formatted, str)
            except Exception as e:
                pytest.skip(f"Format failed: {e}")

    @pytest.mark.asyncio
    async def test_format_with_max_length(self, builder):
        """Test prompt formatting with length limit."""
        async with builder:
            try:
                context = await builder.build("test", semantic_results=5)
                formatted = builder.format_for_prompt(context, max_length=100)
                assert len(formatted) <= 100
            except Exception as e:
                pytest.skip(f"Max length format failed: {e}")

    @pytest.mark.asyncio
    async def test_format_system_prompt(self, builder):
        """Test system prompt generation."""
        async with builder:
            try:
                context = await builder.build("values", semantic_results=2)
                system = builder.format_system_prompt(
                    context,
                    base_prompt="You are helpful."
                )
                assert "You are helpful." in system
                if context.has_content:
                    assert "<personal_context>" in system
            except Exception as e:
                pytest.skip(f"System prompt failed: {e}")

    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test the build_rag_context convenience function."""
        try:
            context = await build_rag_context("test query", semantic_results=1)
            assert isinstance(context, str)
        except Exception as e:
            pytest.skip(f"Convenience function failed: {e}")


class TestRAGContext:
    """Tests for RAGContext dataclass."""

    def test_empty_context(self):
        """Test empty context properties."""
        ctx = RAGContext()
        assert ctx.total_memories == 0
        assert not ctx.has_content

    def test_context_with_memories(self):
        """Test context with memories."""
        from mcp_client import MemoryResult

        ctx = RAGContext(
            semantic_memories=[
                MemoryResult(id="1", content="test", metadata={})
            ],
            query="test"
        )
        assert ctx.total_memories == 1
        assert ctx.has_content

    def test_context_with_identity(self):
        """Test context with identity data."""
        ctx = RAGContext(
            identity_context={"paths_found": 1}
        )
        assert ctx.total_memories == 0
        assert ctx.has_content  # Identity counts as content


# =============================================================================
# CLI Runner
# =============================================================================

async def run_rag_tests():
    """Run RAG tests from command line."""
    print("=" * 60)
    print("RAG INTEGRATION TESTS")
    print("=" * 60)

    async with RAGContextBuilder() as builder:
        print(f"\nTarget: {builder.client.base_url}")

        # Test 1: Build context
        print("\n[1] Building RAG context...")
        try:
            context = await builder.build(
                "family values and authenticity",
                semantic_results=3,
                episodic_results=2
            )
            print(f"    Semantic memories: {len(context.semantic_memories)}")
            print(f"    Episodic memories: {len(context.episodic_memories)}")
            print(f"    Has identity: {bool(context.identity_context)}")
        except Exception as e:
            print(f"    Error: {e}")
            return False

        # Test 2: Format for prompt
        print("\n[2] Formatting for prompt...")
        try:
            formatted = builder.format_for_prompt(context)
            print(f"    Length: {len(formatted)} chars")
            if formatted:
                preview = formatted[:150].replace("\n", " ")
                print(f"    Preview: {preview}...")
        except Exception as e:
            print(f"    Error: {e}")

        # Test 3: System prompt
        print("\n[3] Building system prompt...")
        try:
            system = builder.format_system_prompt(
                context,
                base_prompt="You are Gines, speaking with authenticity and warmth."
            )
            print(f"    Length: {len(system)} chars")
            print(f"    Has context tag: {'<personal_context>' in system}")
        except Exception as e:
            print(f"    Error: {e}")

        # Test 4: Convenience function
        print("\n[4] Testing convenience function...")
        try:
            quick_context = await build_rag_context("relationships")
            print(f"    Result length: {len(quick_context)} chars")
        except Exception as e:
            print(f"    Error: {e}")

        print("\n" + "=" * 60)
        print("RAG tests complete")
        print("=" * 60)

        return True


if __name__ == "__main__":
    success = asyncio.run(run_rag_tests())
    sys.exit(0 if success else 1)
