"""
UBIK Content Ingestion System

Multi-format content ingestion pipeline for the UBIK Digital Legacy Project.
Handles documents, audio, images, and cloud service integrations.

Modules:
    models: Data structures (IngestItem, MemoryCandidate, etc.)
    processors: Format-specific content extractors
    chunkers: Smart content chunking
    classifiers: Memory type classification
    transcript_processor: Meeting/therapy transcript handling
    sources: Cloud storage integrations (Google Drive)
    pipeline: Main ingestion orchestrator

Usage:
    from ingest import IngestPipeline
    from pathlib import Path

    async with IngestPipeline(storage_mode=False) as pipeline:
        result = await pipeline.ingest_file(Path("document.pdf"))
        print(f"Created {result.episodic_count} episodic memories")

CLI:
    python -m ingest local ~/Documents --dry-run
    python -m ingest file ~/letter.pdf --verbose

Version: 1.0.0
"""

from .models import (
    ContentType,
    MemoryType,
    IngestItem,
    ProcessedContent,
    MemoryCandidate,
    IngestResult,
    BatchIngestResult,
)

from .processors import (
    ProcessorConfig,
    ProcessorRegistry,
    BaseProcessor,
    TextProcessor,
    PDFProcessor,
    DOCXProcessor,
    AudioProcessor,
    JSONProcessor,
)

from .chunkers import (
    ChunkConfig,
    Chunk,
    SmartChunker,
)

from .classifiers import (
    ClassifierConfig,
    ContentClassifier,
)

from .transcript_processor import (
    MeetingMetadata,
    ActionItem,
    TranscriptTurn,
    TranscriptChunk,
    SpeakerTurnParser,
    TranscriptChunker,
    TranscriptProcessor,
    parse_front_matter,
    find_companion_metadata,
    transcript_to_memory_candidates,
)

from .sources import (
    GoogleDriveSource,
    GoogleDriveConfig,
)

from .pipeline import (
    IngestPipeline,
    PipelineConfig,
)

__version__ = '1.0.0'
__all__ = [
    # Models
    'ContentType',
    'MemoryType',
    'IngestItem',
    'ProcessedContent',
    'MemoryCandidate',
    'IngestResult',
    'BatchIngestResult',
    # Processors
    'ProcessorConfig',
    'ProcessorRegistry',
    'BaseProcessor',
    'TextProcessor',
    'PDFProcessor',
    'DOCXProcessor',
    'AudioProcessor',
    'JSONProcessor',
    # Chunkers
    'ChunkConfig',
    'Chunk',
    'SmartChunker',
    # Classifiers
    'ClassifierConfig',
    'ContentClassifier',
    # Transcript Processing
    'MeetingMetadata',
    'ActionItem',
    'TranscriptTurn',
    'TranscriptChunk',
    'SpeakerTurnParser',
    'TranscriptChunker',
    'TranscriptProcessor',
    'parse_front_matter',
    'find_companion_metadata',
    'transcript_to_memory_candidates',
    # Sources
    'GoogleDriveSource',
    'GoogleDriveConfig',
    # Pipeline
    'IngestPipeline',
    'PipelineConfig',
]
