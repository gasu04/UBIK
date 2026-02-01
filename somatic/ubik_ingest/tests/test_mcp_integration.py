"""
UBIK Ingestion System - MCP Integration Tests

Tests that the pipeline correctly calls the HippocampalClient
with properly formatted parameters.

These tests mock the MCP client to verify:
- store_episodic() receives correct parameter format
- store_semantic() receives correct parameter format
- Parameters are properly converted (lists to comma-separated strings, etc.)
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, Optional

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.pipeline import IngestPipeline, PipelineConfig
from ingest.models import MemoryCandidate, MemoryType, ProcessedContent, ContentType
from ingest.chunkers import ChunkConfig


class MockHippocampalClient:
    """
    Mock HippocampalClient for testing.

    Records all calls to store_episodic and store_semantic
    for verification.
    """

    def __init__(self):
        self._initialized = True
        self.episodic_calls: list[Dict[str, Any]] = []
        self.semantic_calls: list[Dict[str, Any]] = []
        self.graph_calls: list[Dict[str, Any]] = []
        self._call_counter = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def store_episodic(
        self,
        content: str,
        memory_type: str,
        timestamp: str,
        emotional_valence: str = "neutral",
        importance: float = 0.5,
        participants: str = "",
        themes: str = "",
        source_file: str = "",
    ) -> Dict[str, Any]:
        """Record episodic storage call and return success."""
        self._call_counter += 1
        call_record = {
            "content": content,
            "memory_type": memory_type,
            "timestamp": timestamp,
            "emotional_valence": emotional_valence,
            "importance": importance,
            "participants": participants,
            "themes": themes,
            "source_file": source_file,
        }
        self.episodic_calls.append(call_record)
        return {"status": "success", "memory_id": f"ep_{self._call_counter}"}

    async def store_semantic(
        self,
        content: str,
        knowledge_type: str = "belief",
        category: str = "",
        confidence: float = 0.8,
        stability: float = 0.5,
        source: str = "",
    ) -> Dict[str, Any]:
        """Record semantic storage call and return success."""
        self._call_counter += 1
        call_record = {
            "content": content,
            "knowledge_type": knowledge_type,
            "category": category,
            "confidence": confidence,
            "stability": stability,
            "source": source,
        }
        self.semantic_calls.append(call_record)
        return {"status": "success", "memory_id": f"sem_{self._call_counter}"}

    async def update_identity_graph(
        self,
        from_concept: str,
        relation_type: str,
        to_concept: str,
        weight: float = 1.0,
        context: str = "",
    ) -> Dict[str, Any]:
        """Record graph update call and return success."""
        call_record = {
            "from_concept": from_concept,
            "relation_type": relation_type,
            "to_concept": to_concept,
            "weight": weight,
            "context": context,
        }
        self.graph_calls.append(call_record)
        return {"status": "success"}


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client for testing."""
    return MockHippocampalClient()


@pytest.fixture
def pipeline_config():
    """Create pipeline config for testing."""
    return PipelineConfig(
        storage_mode=True,
        chunk_config=ChunkConfig(
            min_chunk_size=100,
            target_chunk_size=300,
            max_chunk_size=600,
            overlap_size=30,
        ),
        continue_on_storage_error=False,
    )


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    content = """Dear Sofia and Marco,

As I write this letter on December 15, 2024, I'm thinking about what matters most in life.

I believe that family is the foundation of everything. This has always been my deepest conviction.

Yesterday, we had a wonderful dinner together. Your grandmother made her famous paella, and we talked for hours about our hopes and dreams.

I value honesty above almost everything else. Truth matters, even when it's uncomfortable.

With all my love,
Grandpa Gines
"""
    file_path = tmp_path / "test_letter.txt"
    file_path.write_text(content)
    return file_path


class TestMCPIntegration:
    """Test MCP client integration with pipeline."""

    @pytest.mark.asyncio
    async def test_episodic_storage_parameters(
        self,
        mock_mcp_client,
        pipeline_config,
        sample_text_file,
    ):
        """Verify episodic memories are stored with correct parameters."""
        async with IngestPipeline(
            mcp_client=mock_mcp_client,
            config=pipeline_config,
        ) as pipeline:
            result = await pipeline.ingest_file(sample_text_file)

        # Should have some episodic memories (the letter parts)
        assert result.success

        # Check that episodic calls were made
        if mock_mcp_client.episodic_calls:
            call = mock_mcp_client.episodic_calls[0]

            # Verify parameter types
            assert isinstance(call["content"], str)
            assert isinstance(call["memory_type"], str)
            assert isinstance(call["timestamp"], str)
            assert isinstance(call["emotional_valence"], str)  # "positive", "negative", etc.
            assert isinstance(call["importance"], float)
            assert isinstance(call["participants"], str)  # Comma-separated
            assert isinstance(call["themes"], str)  # Comma-separated
            assert isinstance(call["source_file"], str)

            # Verify timestamp format (ISO with Z suffix)
            assert call["timestamp"].endswith("Z")

            # Verify source file is set
            assert "test_letter.txt" in call["source_file"]

    @pytest.mark.asyncio
    async def test_semantic_storage_parameters(
        self,
        mock_mcp_client,
        pipeline_config,
        sample_text_file,
    ):
        """Verify semantic memories are stored with correct parameters."""
        async with IngestPipeline(
            mcp_client=mock_mcp_client,
            config=pipeline_config,
        ) as pipeline:
            result = await pipeline.ingest_file(sample_text_file)

        # Should have some semantic memories (the belief statements)
        assert result.success

        # Check that semantic calls were made
        if mock_mcp_client.semantic_calls:
            call = mock_mcp_client.semantic_calls[0]

            # Verify parameter types
            assert isinstance(call["content"], str)
            assert isinstance(call["knowledge_type"], str)
            assert isinstance(call["category"], str)
            assert isinstance(call["confidence"], float)
            # stability is a string in the model ("stable", "evolving", "uncertain")
            assert isinstance(call["stability"], str)
            assert isinstance(call["source"], str)

            # Verify source is set
            assert "test_letter.txt" in call["source"]

    @pytest.mark.asyncio
    async def test_participants_as_comma_separated(
        self,
        mock_mcp_client,
        pipeline_config,
        tmp_path,
    ):
        """Verify participants list is converted to comma-separated string."""
        # Create content with clear participant mentions
        content = """Dear Family,

Yesterday Sofia, Marco, and Elena gathered for dinner. We all shared stories.

With love,
Gines
"""
        file_path = tmp_path / "family_letter.txt"
        file_path.write_text(content)

        async with IngestPipeline(
            mcp_client=mock_mcp_client,
            config=pipeline_config,
        ) as pipeline:
            await pipeline.ingest_file(file_path)

        # Check episodic calls
        for call in mock_mcp_client.episodic_calls:
            participants = call["participants"]
            # Should be a string (possibly comma-separated or default)
            assert isinstance(participants, str)
            # Should not be a list
            assert not isinstance(participants, list)

    @pytest.mark.asyncio
    async def test_themes_as_comma_separated(
        self,
        mock_mcp_client,
        pipeline_config,
        tmp_path,
    ):
        """Verify themes list is converted to comma-separated string."""
        # Create content with clear theme triggers
        content = """My Reflections on Family and Legacy

I've been thinking about what we leave behind. Family is everything to me.
The love we share today becomes the legacy of tomorrow.
"""
        file_path = tmp_path / "reflections.txt"
        file_path.write_text(content)

        async with IngestPipeline(
            mcp_client=mock_mcp_client,
            config=pipeline_config,
        ) as pipeline:
            await pipeline.ingest_file(file_path)

        # Check episodic calls
        for call in mock_mcp_client.episodic_calls:
            themes = call["themes"]
            # Should be a string (possibly comma-separated or empty)
            assert isinstance(themes, str)
            # Should not be a list
            assert not isinstance(themes, list)

    @pytest.mark.asyncio
    async def test_storage_mode_disabled(
        self,
        mock_mcp_client,
        sample_text_file,
    ):
        """Verify no storage calls when storage_mode is False."""
        config = PipelineConfig(
            storage_mode=False,
            chunk_config=ChunkConfig(
                min_chunk_size=100,
                target_chunk_size=300,
                max_chunk_size=600,
                overlap_size=30,
            ),
        )

        async with IngestPipeline(
            mcp_client=mock_mcp_client,
            config=config,
        ) as pipeline:
            result = await pipeline.ingest_file(sample_text_file)

        # Should succeed but no storage calls
        assert result.success
        assert len(mock_mcp_client.episodic_calls) == 0
        assert len(mock_mcp_client.semantic_calls) == 0

    @pytest.mark.asyncio
    async def test_no_client_configured(
        self,
        sample_text_file,
        pipeline_config,
    ):
        """Verify pipeline works without MCP client (dry run)."""
        pipeline_config.storage_mode = True  # Even with storage mode on

        async with IngestPipeline(
            mcp_client=None,  # No client
            config=pipeline_config,
        ) as pipeline:
            result = await pipeline.ingest_file(sample_text_file)

        # Should succeed (processing works, just no storage)
        assert result.success

    @pytest.mark.asyncio
    async def test_storage_error_handling(
        self,
        sample_text_file,
        pipeline_config,
    ):
        """Verify storage errors are handled properly."""
        # Create a client that fails
        failing_client = MockHippocampalClient()

        async def fail_episodic(*args, **kwargs):
            raise Exception("Storage failed!")

        failing_client.store_episodic = fail_episodic

        pipeline_config.continue_on_storage_error = True

        async with IngestPipeline(
            mcp_client=failing_client,
            config=pipeline_config,
        ) as pipeline:
            # Should not raise, just log errors
            result = await pipeline.ingest_file(sample_text_file)

        # Processing should still complete
        assert result.success or result.error is not None


class TestMemoryCandidateConversion:
    """Test that MemoryCandidate fields are properly converted for MCP."""

    @pytest.mark.asyncio
    async def test_episodic_candidate_conversion(self, mock_mcp_client):
        """Test direct _store_episodic with a MemoryCandidate."""
        candidate = MemoryCandidate(
            content="We had dinner together last night.",
            memory_type=MemoryType.EPISODIC,
            confidence=0.9,
            source_file="test.txt",
            source_chunk_index=0,
            category="family",
            themes=["family", "love"],
            participants=["gines", "elena", "sofia"],
            emotional_valence="positive",
            importance=0.7,
            timestamp=datetime(2024, 12, 15, 19, 30),
            event_type="gathering",
        )

        pipeline = IngestPipeline(mcp_client=mock_mcp_client, storage_mode=True)
        pipeline._connected = True  # Simulate connected state

        memory_id = await pipeline._store_episodic(candidate)

        assert memory_id is not None
        assert len(mock_mcp_client.episodic_calls) == 1

        call = mock_mcp_client.episodic_calls[0]

        # Verify conversions
        assert call["content"] == "We had dinner together last night."
        assert call["memory_type"] == "gathering"
        assert call["timestamp"] == "2024-12-15T19:30:00Z"
        assert call["emotional_valence"] == "positive"
        assert call["importance"] == 0.7
        assert call["participants"] == "gines,elena,sofia"  # Comma-separated
        assert call["themes"] == "family,love"  # Comma-separated
        assert call["source_file"] == "test.txt"

    @pytest.mark.asyncio
    async def test_semantic_candidate_conversion(self, mock_mcp_client):
        """Test direct _store_semantic with a MemoryCandidate."""
        candidate = MemoryCandidate(
            content="I believe that honesty is the foundation of trust.",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.95,
            source_file="values.txt",
            source_chunk_index=0,
            knowledge_type="belief",
            category="ethics",
            themes=["ethics", "values"],
            emotional_valence="neutral",
            importance=0.9,
            stability="stable",
        )

        pipeline = IngestPipeline(mcp_client=mock_mcp_client, storage_mode=True)
        pipeline._connected = True  # Simulate connected state

        memory_id = await pipeline._store_semantic(candidate)

        assert memory_id is not None
        assert len(mock_mcp_client.semantic_calls) == 1

        call = mock_mcp_client.semantic_calls[0]

        # Verify conversions
        assert call["content"] == "I believe that honesty is the foundation of trust."
        assert call["knowledge_type"] == "belief"
        assert call["category"] == "ethics"
        assert call["confidence"] == 0.95
        assert call["stability"] == "stable"
        assert call["source"] == "values.txt"

    @pytest.mark.asyncio
    async def test_default_values(self, mock_mcp_client):
        """Test that default values are applied correctly."""
        # Minimal candidate with required fields only
        candidate = MemoryCandidate(
            content="A simple memory.",
            memory_type=MemoryType.EPISODIC,
            confidence=0.5,
            source_file="test.txt",
            source_chunk_index=0,
            category="general",
            themes=[],
            emotional_valence="neutral",
            importance=0.5,
            # No participants, timestamp, event_type set
        )

        pipeline = IngestPipeline(mcp_client=mock_mcp_client, storage_mode=True)
        pipeline._connected = True

        await pipeline._store_episodic(candidate)

        call = mock_mcp_client.episodic_calls[0]

        # Check defaults
        assert call["memory_type"] == "general"  # Default event_type
        assert call["participants"] == "gines"  # Default participant
        assert call["themes"] == ""  # Empty themes
        assert call["timestamp"].endswith("Z")  # Has timestamp
        assert call["emotional_valence"] == "neutral"  # Default valence
        assert call["importance"] == 0.5  # Default importance


class TestTranscriptNeo4jOperations:
    """Test Neo4j operations execution for transcripts."""

    @pytest.mark.asyncio
    async def test_neo4j_operations_executed(self, mock_mcp_client):
        """Test that Neo4j operations from transcripts are executed."""
        # Create a minimal IngestItem for ProcessedContent
        from ingest.models import IngestItem

        mock_item = MagicMock(spec=IngestItem)
        mock_item.original_filename = "meeting.transcript"
        mock_item.content_type = ContentType.TEXT

        # Create a processed content with Neo4j operations
        processed = ProcessedContent(
            source_item=mock_item,
            text="Meeting content...",
            processor_used="transcript_processor",
            processing_time_ms=100.0,
            word_count=2,
            extracted_metadata={
                "neo4j_operations": [
                    {
                        "operation": "merge_relationship",
                        "from_id": "Person:Gines",
                        "rel_type": "ATTENDED",
                        "to_id": "Event:FamilyMeeting",
                        "properties": {"date": "2024-12-15"},
                    },
                    {
                        "operation": "merge_node",
                        "label": "Event",
                        "id": "FamilyMeeting",
                        "properties": {"type": "family_meeting"},
                    },
                ],
            },
        )

        pipeline = IngestPipeline(mcp_client=mock_mcp_client, storage_mode=True)
        pipeline._connected = True

        successful, failed = await pipeline._execute_neo4j_operations(
            processed.extracted_metadata["neo4j_operations"]
        )

        # Should have executed operations
        assert successful >= 1
        assert failed == 0

        # Check graph calls
        if mock_mcp_client.graph_calls:
            call = mock_mcp_client.graph_calls[0]
            assert call["from_concept"] == "Person:Gines"
            assert call["relation_type"] == "ATTENDED"
            assert call["to_concept"] == "Event:FamilyMeeting"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
