"""
UBIK Ingestion System - Integration Tests

Tests the complete ingestion pipeline including:
- Text processing
- Content chunking
- Memory classification
- Transcript parsing
- Pipeline execution

Run with: pytest tests/test_integration.py -v
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio

# Import from package
from ingest import (
    # Models
    ContentType,
    MemoryType,
    IngestItem,
    ProcessedContent,
    MemoryCandidate,
    IngestResult,
    BatchIngestResult,
    # Processors
    ProcessorConfig,
    TextProcessor,
    # Chunkers
    ChunkConfig,
    Chunk,
    SmartChunker,
    # Classifiers
    ContentClassifier,
    ClassifierConfig,
    # Transcript
    MeetingMetadata,
    TranscriptProcessor,
    TranscriptChunker,
    SpeakerTurnParser,
    parse_front_matter,
    transcript_to_memory_candidates,
    # Pipeline
    IngestPipeline,
    PipelineConfig,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory(prefix="ubik_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_letter_content():
    """Sample episodic letter content."""
    return """
Dear Sofia and Adrian,

I'm writing this letter on January 15, 2024, to share some memories with you.

Yesterday I was thinking about our trip to Lake Michigan in 2019. I remember
how we all laughed around the campfire that night. Your mother said,
"This is what life is all about."

I'll never forget the look on your faces when we saw the northern lights.
That was the moment I realized how precious our time together truly is.

With all my love,
Dad
"""


@pytest.fixture
def sample_values_content():
    """Sample semantic values content."""
    return """
# My Core Beliefs

I believe that authenticity is the foundation of a meaningful life.
My core values center around honesty, compassion, and integrity.

I think that being genuine with others matters more than appearing
successful. At my core, I am a person who values relationships
over achievements.

The meaning of life, in my view, is found in connection with others
and in living according to one's principles.
"""


@pytest.fixture
def sample_transcript_content():
    """Sample transcript with YAML front matter."""
    return """---
meeting_date: 2024-01-15
meeting_type: therapy
participants:
  - Gines
  - Dr. Smith
speakers:
  Speaker 1: Gines
  Speaker 2: Dr. Smith
emotional_tone: reflective
decisions_made:
  - Continue weekly sessions
  - Start journaling practice
---

Speaker 1  0:00
I've been thinking a lot about my family lately.

Speaker 2  0:15
Tell me more about what's been on your mind.

Speaker 1  0:30
I realized that I want to leave something meaningful for my children.
Something that captures who I really am.

Speaker 2  1:00
That's a beautiful impulse. What form do you imagine that taking?
"""


@pytest.fixture
def text_file(temp_dir, sample_letter_content):
    """Create a sample text file."""
    file_path = temp_dir / "letter.txt"
    file_path.write_text(sample_letter_content)
    return file_path


@pytest.fixture
def values_file(temp_dir, sample_values_content):
    """Create a sample values file."""
    file_path = temp_dir / "values.md"
    file_path.write_text(sample_values_content)
    return file_path


@pytest.fixture
def transcript_file(temp_dir, sample_transcript_content):
    """Create a sample transcript file."""
    file_path = temp_dir / "session.transcript"
    file_path.write_text(sample_transcript_content)
    return file_path


@pytest.fixture
def multi_file_dir(temp_dir, sample_letter_content, sample_values_content):
    """Create a directory with multiple test files."""
    (temp_dir / "letter.txt").write_text(sample_letter_content)
    (temp_dir / "values.md").write_text(sample_values_content)
    (temp_dir / "data.json").write_text(json.dumps({
        "title": "Family History",
        "author": "Gines",
        "year": 2024
    }))
    return temp_dir


# =============================================================================
# Text Processor Tests
# =============================================================================

class TestTextProcessor:
    """Tests for TextProcessor."""

    @pytest.mark.asyncio
    async def test_text_processor_extracts_content(self, text_file):
        """Test that TextProcessor extracts text content."""
        processor = TextProcessor()
        item = IngestItem.from_path(text_file)

        result = await processor.process(item, text_file)

        assert result.text is not None
        assert len(result.text) > 0
        assert "Dear Sofia" in result.text
        assert result.processor_used == "TextProcessor"

    @pytest.mark.asyncio
    async def test_text_processor_detects_title(self, text_file):
        """Test that TextProcessor extracts title from content."""
        processor = TextProcessor()
        item = IngestItem.from_path(text_file)

        result = await processor.process(item, text_file)

        # Title should be extracted from first line or filename
        assert result.title is not None

    @pytest.mark.asyncio
    async def test_text_processor_word_count(self, text_file):
        """Test that word count is calculated."""
        processor = TextProcessor()
        item = IngestItem.from_path(text_file)

        result = await processor.process(item, text_file)

        assert result.word_count > 0
        assert result.word_count == len(result.text.split())


# =============================================================================
# Smart Chunker Tests
# =============================================================================

class TestSmartChunker:
    """Tests for SmartChunker."""

    def test_chunker_respects_size_limits(self, sample_letter_content):
        """Test that chunks respect configured size limits."""
        config = ChunkConfig(
            min_chunk_size=50,
            target_chunk_size=200,
            max_chunk_size=400,
            overlap_size=20,
        )
        chunker = SmartChunker(config)

        item = IngestItem(
            source_path="/test/doc.txt",
            source_type="local",
            content_type=ContentType.TEXT,
            original_filename="doc.txt",
            file_extension=".txt",
            file_size_bytes=len(sample_letter_content),
        )
        processed = ProcessedContent(
            source_item=item,
            text=sample_letter_content,
            processor_used="test",
            processing_time_ms=0,
            word_count=len(sample_letter_content.split()),
        )

        chunks = chunker.chunk(processed)

        for chunk in chunks:
            assert chunk.char_count <= config.max_chunk_size

    def test_chunker_no_content_loss(self, sample_values_content):
        """Test that all content is preserved after chunking."""
        config = ChunkConfig(
            min_chunk_size=30,
            target_chunk_size=150,
            max_chunk_size=300,
            overlap_size=15,
        )
        chunker = SmartChunker(config)

        item = IngestItem(
            source_path="/test/doc.md",
            source_type="local",
            content_type=ContentType.TEXT,
            original_filename="doc.md",
            file_extension=".md",
            file_size_bytes=len(sample_values_content),
        )
        processed = ProcessedContent(
            source_item=item,
            text=sample_values_content,
            processor_used="test",
            processing_time_ms=0,
            word_count=len(sample_values_content.split()),
        )

        chunks = chunker.chunk(processed)

        # Key phrases should appear in at least one chunk
        key_phrases = ["authenticity", "honesty", "compassion", "integrity"]
        for phrase in key_phrases:
            found = any(phrase in chunk.text for chunk in chunks)
            assert found, f"Phrase '{phrase}' not found in any chunk"

    def test_chunker_single_chunk_for_small_content(self):
        """Test that small content produces single chunk."""
        config = ChunkConfig()
        chunker = SmartChunker(config)

        small_text = "This is a short piece of text."
        item = IngestItem(
            source_path="/test/doc.txt",
            source_type="local",
            content_type=ContentType.TEXT,
            original_filename="doc.txt",
            file_extension=".txt",
            file_size_bytes=len(small_text),
        )
        processed = ProcessedContent(
            source_item=item,
            text=small_text,
            processor_used="test",
            processing_time_ms=0,
            word_count=len(small_text.split()),
        )

        chunks = chunker.chunk(processed)

        assert len(chunks) == 1
        assert chunks[0].chunk_type == "single"


# =============================================================================
# Content Classifier Tests
# =============================================================================

class TestContentClassifier:
    """Tests for ContentClassifier."""

    def _create_chunk(self, text: str, index: int = 0) -> Chunk:
        """Helper to create a Chunk for testing."""
        return Chunk(
            text=text,
            index=index,
            start_char=0,
            end_char=len(text),
            chunk_type="test",
        )

    def _create_processed(self, text: str) -> ProcessedContent:
        """Helper to create ProcessedContent for testing."""
        item = IngestItem(
            source_path="/test/doc.txt",
            source_type="local",
            content_type=ContentType.TEXT,
            original_filename="doc.txt",
            file_extension=".txt",
            file_size_bytes=len(text),
        )
        return ProcessedContent(
            source_item=item,
            text=text,
            processor_used="test",
            processing_time_ms=0,
            word_count=len(text.split()),
        )

    def test_classifier_episodic_letter(self, sample_letter_content):
        """Test that letter content is classified as episodic."""
        classifier = ContentClassifier()

        chunk = self._create_chunk(sample_letter_content)
        processed = self._create_processed(sample_letter_content)

        candidate = classifier.classify(chunk, processed)

        assert candidate.memory_type == MemoryType.EPISODIC
        assert candidate.confidence > 0.5
        assert "family" in candidate.themes

    def test_classifier_semantic_values(self, sample_values_content):
        """Test that values content is classified as semantic."""
        classifier = ContentClassifier()

        chunk = self._create_chunk(sample_values_content)
        processed = self._create_processed(sample_values_content)

        candidate = classifier.classify(chunk, processed)

        assert candidate.memory_type == MemoryType.SEMANTIC
        assert candidate.confidence > 0.5
        assert any(t in candidate.themes for t in ["values", "authenticity", "philosophy"])

    def test_classifier_detects_themes(self):
        """Test that classifier detects themes correctly."""
        classifier = ContentClassifier()

        text = "My family gathered for Thanksgiving. Children and grandchildren all together."
        chunk = self._create_chunk(text)
        processed = self._create_processed(text)

        candidate = classifier.classify(chunk, processed)

        assert "family" in candidate.themes

    def test_classifier_detects_emotional_valence(self):
        """Test that classifier detects emotional valence."""
        classifier = ContentClassifier()

        positive_text = "I am so happy and grateful for this wonderful day!"
        chunk = self._create_chunk(positive_text)
        processed = self._create_processed(positive_text)

        candidate = classifier.classify(chunk, processed)

        assert candidate.emotional_valence == "positive"


# =============================================================================
# Transcript Processing Tests
# =============================================================================

class TestTranscriptProcessing:
    """Tests for transcript processing."""

    def test_parse_front_matter(self, sample_transcript_content):
        """Test YAML front matter parsing."""
        metadata_dict, body = parse_front_matter(sample_transcript_content)

        assert metadata_dict is not None
        assert "meeting_date" in metadata_dict
        assert metadata_dict["meeting_type"] == "therapy"
        assert "Gines" in metadata_dict["participants"]
        assert "Speaker 1" in metadata_dict["speakers"]
        assert metadata_dict["speakers"]["Speaker 1"] == "Gines"

        # Body should not contain front matter
        assert "---" not in body.split("\n")[0]
        assert "Speaker 1" in body

    def test_meeting_metadata_from_dict(self, sample_transcript_content):
        """Test MeetingMetadata creation from parsed front matter."""
        metadata_dict, _ = parse_front_matter(sample_transcript_content)
        metadata = MeetingMetadata.from_dict(metadata_dict)

        assert metadata.meeting_type == "therapy"
        assert metadata.meeting_date is not None
        assert metadata.meeting_date.year == 2024
        assert "Gines" in metadata.participants
        assert len(metadata.decisions_made) == 2

    def test_speaker_turn_parser(self, sample_transcript_content):
        """Test parsing speaker turns from transcript."""
        _, body = parse_front_matter(sample_transcript_content)

        parser = SpeakerTurnParser({
            "Speaker 1": "Gines",
            "Speaker 2": "Dr. Smith",
        })
        turns = parser.parse(body)

        assert len(turns) == 4
        assert turns[0].speaker_name == "Gines"
        assert turns[1].speaker_name == "Dr. Smith"
        assert turns[0].start_time == 0
        assert "family" in turns[0].text

    def test_transcript_chunker(self, sample_transcript_content):
        """Test transcript chunking by turns."""
        _, body = parse_front_matter(sample_transcript_content)
        metadata_dict, _ = parse_front_matter(sample_transcript_content)
        metadata = MeetingMetadata.from_dict(metadata_dict)

        parser = SpeakerTurnParser(metadata.speakers)
        turns = parser.parse(body)

        chunker = TranscriptChunker(turns_per_chunk=2, overlap_turns=1)
        chunks = chunker.chunk(turns, metadata)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.turn_count <= 2
            assert len(chunk.speakers_in_chunk) > 0


# =============================================================================
# Pipeline Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Tests for the complete ingestion pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_single_file_dry_run(self, text_file):
        """Test pipeline processes single file in dry run mode."""
        config = PipelineConfig(storage_mode=False)
        pipeline = IngestPipeline(config=config)

        result = await pipeline.ingest_file(text_file)

        assert result.success
        assert result.source_file == text_file.name
        assert result.chunks_generated > 0
        assert result.episodic_count + result.semantic_count > 0
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_pipeline_directory_dry_run(self, multi_file_dir):
        """Test pipeline processes directory in dry run mode."""
        config = PipelineConfig(storage_mode=False)
        pipeline = IngestPipeline(config=config)

        result = await pipeline.ingest_directory(multi_file_dir)

        assert result.total_files == 3
        assert result.successful == 3
        assert result.failed == 0
        assert result.success_rate == 100.0
        assert result.total_memories > 0

    @pytest.mark.asyncio
    async def test_pipeline_extension_filter(self, multi_file_dir):
        """Test pipeline filters by extension."""
        config = PipelineConfig(storage_mode=False)
        pipeline = IngestPipeline(config=config)

        result = await pipeline.ingest_directory(
            multi_file_dir,
            extensions=[".txt"],
        )

        assert result.total_files == 1
        assert result.successful == 1

    @pytest.mark.asyncio
    async def test_pipeline_memory_candidates(self, text_file):
        """Test pipeline produces valid memory candidates."""
        config = PipelineConfig(storage_mode=False)
        pipeline = IngestPipeline(config=config)

        result = await pipeline.ingest_file(text_file)

        assert len(result.memory_candidates) > 0

        for candidate in result.memory_candidates:
            assert candidate.content
            assert candidate.memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.SKIP]
            assert 0 <= candidate.confidence <= 1
            assert 0 <= candidate.importance <= 1
            assert candidate.category
            assert candidate.source_file == text_file.name

    @pytest.mark.asyncio
    async def test_pipeline_unsupported_file(self, temp_dir):
        """Test pipeline handles unsupported file types gracefully."""
        unsupported = temp_dir / "file.xyz"
        unsupported.write_text("some content")

        config = PipelineConfig(storage_mode=False)
        pipeline = IngestPipeline(config=config)

        result = await pipeline.ingest_file(unsupported)

        assert not result.success
        assert result.error is not None
        assert "Unsupported" in result.error


# =============================================================================
# Model Tests
# =============================================================================

class TestModels:
    """Tests for data models."""

    def test_ingest_item_from_path(self, text_file):
        """Test IngestItem creation from path."""
        item = IngestItem.from_path(text_file)

        assert item.source_path == str(text_file)
        assert item.source_type == "local"
        assert item.content_type == ContentType.TEXT
        assert item.file_extension == ".txt"
        assert item.file_size_bytes > 0
        assert item.created_at is not None

    def test_memory_candidate_to_episodic_dict(self):
        """Test MemoryCandidate conversion to episodic format."""
        candidate = MemoryCandidate(
            content="Test content",
            memory_type=MemoryType.EPISODIC,
            confidence=0.9,
            category="family",
            themes=["love", "memory"],
            event_type="gathering",
            participants=["Alice", "Bob"],
            emotional_valence="positive",
            source_file="test.txt",
            source_chunk_index=0,
            importance=0.8,
        )

        result = candidate.to_episodic_dict()

        assert result["content"] == "Test content"
        assert result["type"] == "episodic"
        assert result["event_type"] == "gathering"
        assert "Alice" in result["participants"]
        assert result["emotional_valence"] == "positive"

    def test_memory_candidate_to_semantic_dict(self):
        """Test MemoryCandidate conversion to semantic format."""
        candidate = MemoryCandidate(
            content="I believe in honesty",
            memory_type=MemoryType.SEMANTIC,
            confidence=0.85,
            category="values",
            themes=["authenticity"],
            knowledge_type="belief",
            emotional_valence="neutral",
            source_file="test.txt",
            source_chunk_index=0,
            importance=0.7,
        )

        result = candidate.to_semantic_dict()

        assert result["content"] == "I believe in honesty"
        assert result["type"] == "semantic"
        assert result["knowledge_type"] == "belief"
        assert result["category"] == "values"

    def test_batch_ingest_result_aggregation(self):
        """Test BatchIngestResult aggregates correctly."""
        batch = BatchIngestResult()

        result1 = IngestResult(
            source_file="file1.txt",
            success=True,
            memory_candidates=[],
            processing_time_ms=100,
            episodic_count=2,
            semantic_count=1,
        )
        result2 = IngestResult(
            source_file="file2.txt",
            success=True,
            memory_candidates=[],
            processing_time_ms=150,
            episodic_count=1,
            semantic_count=3,
        )

        batch.add_result(result1)
        batch.add_result(result2)

        assert batch.total_files == 2
        assert batch.successful == 2
        assert batch.total_episodic == 3
        assert batch.total_semantic == 4
        assert batch.total_time_ms == 250
        assert batch.success_rate == 100.0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
