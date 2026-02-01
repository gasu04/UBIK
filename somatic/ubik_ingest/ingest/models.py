"""
UBIK Ingestion System - Data Models

Core data structures for the content ingestion pipeline.
Defines the data flow from raw files through processing to memory candidates.

Data Flow:
    IngestItem (raw file)
    → ProcessedContent (extracted text)
    → MemoryCandidate (classified for storage)
    → IngestResult (operation outcome)

Usage:
    from ingest.models import IngestItem, ProcessedContent, MemoryCandidate

    item = IngestItem.from_path(Path("document.pdf"))
    processed = ProcessedContent(source_item=item, text="...", ...)
    candidate = MemoryCandidate(content="...", memory_type=MemoryType.EPISODIC, ...)

Version: 0.1.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    'ContentType',
    'MemoryType',
    'IngestItem',
    'ProcessedContent',
    'MemoryCandidate',
    'IngestResult',
    'BatchIngestResult',
]


class ContentType(Enum):
    """
    Classification of content format.

    Used to route files to appropriate processors.

    Attributes:
        TEXT: Plain text files (.txt, .md, .rst)
        DOCUMENT: Formatted documents (.pdf, .docx, .odt)
        AUDIO: Audio recordings (.mp3, .wav, .m4a)
        STRUCTURED: Data files (.json, .yaml, .csv)
        UNKNOWN: Unrecognized format
    """
    TEXT = "text"
    DOCUMENT = "document"
    AUDIO = "audio"
    STRUCTURED = "structured"
    UNKNOWN = "unknown"


class MemoryType(Enum):
    """
    Memory classification for storage routing.

    Based on cognitive memory systems:
    - EPISODIC: Personal experiences, events, autobiographical
    - SEMANTIC: Facts, knowledge, concepts, general information
    - SKIP: Content not suitable for memory storage

    Attributes:
        EPISODIC: Event-based memories with temporal context
        SEMANTIC: Factual knowledge without temporal binding
        SKIP: Content to exclude from memory system
    """
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    SKIP = "skip"


# Extension mappings for content type detection
_CONTENT_TYPE_MAP: Dict[str, ContentType] = {
    # Text
    ".txt": ContentType.TEXT,
    ".md": ContentType.TEXT,
    ".rst": ContentType.TEXT,
    ".log": ContentType.TEXT,
    # Documents
    ".pdf": ContentType.DOCUMENT,
    ".docx": ContentType.DOCUMENT,
    ".doc": ContentType.DOCUMENT,
    ".odt": ContentType.DOCUMENT,
    ".rtf": ContentType.DOCUMENT,
    # Audio
    ".mp3": ContentType.AUDIO,
    ".wav": ContentType.AUDIO,
    ".m4a": ContentType.AUDIO,
    ".flac": ContentType.AUDIO,
    ".ogg": ContentType.AUDIO,
    ".aac": ContentType.AUDIO,
    # Structured
    ".json": ContentType.STRUCTURED,
    ".yaml": ContentType.STRUCTURED,
    ".yml": ContentType.STRUCTURED,
    ".csv": ContentType.STRUCTURED,
    ".xml": ContentType.STRUCTURED,
}

# MIME type mappings for Google Drive
_MIME_TYPE_MAP: Dict[str, ContentType] = {
    "text/plain": ContentType.TEXT,
    "text/markdown": ContentType.TEXT,
    "application/pdf": ContentType.DOCUMENT,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ContentType.DOCUMENT,
    "application/msword": ContentType.DOCUMENT,
    "audio/mpeg": ContentType.AUDIO,
    "audio/wav": ContentType.AUDIO,
    "audio/mp4": ContentType.AUDIO,
    "audio/x-m4a": ContentType.AUDIO,
    "application/json": ContentType.STRUCTURED,
    "text/csv": ContentType.STRUCTURED,
}


@dataclass
class IngestItem:
    """
    Represents a file queued for ingestion.

    Contains metadata about the source file before processing.
    Created from local paths or cloud service references.

    Attributes:
        source_path: Path or identifier for the source
        source_type: Origin type ("local", "gdrive", "dropbox")
        content_type: Detected content format
        original_filename: Original file name
        file_extension: Lowercase file extension with dot
        file_size_bytes: File size in bytes
        source_metadata: Additional source-specific metadata
        created_at: File creation timestamp
        modified_at: File modification timestamp

    Example:
        >>> item = IngestItem.from_path(Path("~/documents/letter.pdf"))
        >>> print(item.content_type)
        ContentType.DOCUMENT
    """
    source_path: str
    source_type: str
    content_type: ContentType
    original_filename: str
    file_extension: str
    file_size_bytes: int
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None

    @classmethod
    def from_path(cls, path: Path) -> "IngestItem":
        """
        Create an IngestItem from a local file path.

        Args:
            path: Path to the local file

        Returns:
            IngestItem with populated metadata

        Raises:
            FileNotFoundError: If path does not exist
            ValueError: If path is not a file
        """
        path = path.expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        stat = path.stat()
        extension = path.suffix.lower()
        content_type = _CONTENT_TYPE_MAP.get(extension, ContentType.UNKNOWN)

        return cls(
            source_path=str(path),
            source_type="local",
            content_type=content_type,
            original_filename=path.name,
            file_extension=extension,
            file_size_bytes=stat.st_size,
            source_metadata={"absolute_path": str(path)},
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
        )

    @classmethod
    def from_gdrive(
        cls,
        file_id: str,
        name: str,
        mime_type: str,
        metadata: Dict[str, Any]
    ) -> "IngestItem":
        """
        Create an IngestItem from a Google Drive file reference.

        Args:
            file_id: Google Drive file ID
            name: File name from Drive
            mime_type: MIME type from Drive
            metadata: Additional metadata from Drive API

        Returns:
            IngestItem with Drive metadata
        """
        extension = Path(name).suffix.lower()
        content_type = _MIME_TYPE_MAP.get(mime_type, ContentType.UNKNOWN)

        # Fall back to extension mapping if MIME type unknown
        if content_type == ContentType.UNKNOWN:
            content_type = _CONTENT_TYPE_MAP.get(extension, ContentType.UNKNOWN)

        # Parse timestamps from Drive metadata
        created_at = None
        modified_at = None
        if "createdTime" in metadata:
            created_at = datetime.fromisoformat(
                metadata["createdTime"].replace("Z", "+00:00")
            )
        if "modifiedTime" in metadata:
            modified_at = datetime.fromisoformat(
                metadata["modifiedTime"].replace("Z", "+00:00")
            )

        return cls(
            source_path=f"gdrive://{file_id}",
            source_type="gdrive",
            content_type=content_type,
            original_filename=name,
            file_extension=extension,
            file_size_bytes=int(metadata.get("size", 0)),
            source_metadata={
                "file_id": file_id,
                "mime_type": mime_type,
                "drive_metadata": metadata,
            },
            created_at=created_at,
            modified_at=modified_at,
        )


@dataclass
class ProcessedContent:
    """
    Content after format-specific processing.

    Contains extracted text and metadata from document processing.
    Produced by format handlers (PDF, DOCX, audio transcription, etc.).

    Attributes:
        source_item: Original IngestItem reference
        text: Extracted text content
        title: Detected or extracted title
        processor_used: Name of processor that handled this
        processing_time_ms: Processing duration in milliseconds
        page_count: Number of pages (documents only)
        word_count: Total word count
        language: Detected language code (ISO 639-1)
        audio_duration_seconds: Audio length (audio only)
        transcription_confidence: Whisper confidence (audio only)
        extracted_metadata: Additional extracted metadata

    Example:
        >>> processed = ProcessedContent(
        ...     source_item=item,
        ...     text="Dear family...",
        ...     processor_used="pdf_processor",
        ...     processing_time_ms=150.5,
        ...     word_count=500,
        ...     language="en"
        ... )
    """
    source_item: IngestItem
    text: str
    processor_used: str
    processing_time_ms: float
    word_count: int
    language: str = "en"
    title: Optional[str] = None
    page_count: Optional[int] = None
    audio_duration_seconds: Optional[float] = None
    transcription_confidence: Optional[float] = None
    extracted_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate word count if not provided."""
        if self.word_count == 0 and self.text:
            self.word_count = len(self.text.split())


@dataclass
class MemoryCandidate:
    """
    Content chunk classified and ready for memory storage.

    Represents a single memory unit after classification.
    Contains all information needed for storage in ChromaDB/Neo4j.

    Attributes:
        content: The actual text content
        memory_type: Classification (EPISODIC, SEMANTIC, SKIP)
        confidence: Classification confidence (0.0-1.0)
        category: Content category (e.g., "family", "career", "health")
        themes: Detected themes/topics
        event_type: Type of event for episodic memories
        participants: People mentioned/involved
        emotional_valence: Emotional tone ("positive", "negative", "neutral", "mixed")
        knowledge_type: Type of knowledge for semantic memories
        stability: Memory stability ("stable", "evolving", "uncertain")
        source_file: Original source filename
        source_chunk_index: Index of this chunk in source
        timestamp: Event timestamp for episodic memories
        importance: Importance score (0.0-1.0)

    Example:
        >>> candidate = MemoryCandidate(
        ...     content="My father taught me to fish at Lake Michigan...",
        ...     memory_type=MemoryType.EPISODIC,
        ...     confidence=0.92,
        ...     category="family",
        ...     themes=["childhood", "father", "outdoors"],
        ...     event_type="activity",
        ...     participants=["father"],
        ...     emotional_valence="positive",
        ...     source_file="memories.txt",
        ...     source_chunk_index=0,
        ...     importance=0.85
        ... )
    """
    content: str
    memory_type: MemoryType
    confidence: float
    category: str
    themes: List[str]
    emotional_valence: str
    source_file: str
    source_chunk_index: int
    importance: float
    event_type: Optional[str] = None
    participants: List[str] = field(default_factory=list)
    knowledge_type: Optional[str] = None
    stability: str = "stable"
    timestamp: Optional[datetime] = None

    def to_episodic_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for episodic memory storage.

        Returns:
            Dictionary with episodic memory fields formatted
            for ChromaDB/Neo4j storage.

        Raises:
            ValueError: If memory_type is not EPISODIC
        """
        if self.memory_type != MemoryType.EPISODIC:
            raise ValueError(
                f"Cannot convert {self.memory_type.value} to episodic format"
            )

        return {
            "content": self.content,
            "type": "episodic",
            "event_type": self.event_type or "general",
            "category": self.category,
            "themes": self.themes,
            "participants": self.participants,
            "emotional_valence": self.emotional_valence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "importance": self.importance,
            "confidence": self.confidence,
            "source": {
                "file": self.source_file,
                "chunk_index": self.source_chunk_index,
            },
            "metadata": {
                "stability": self.stability,
                "ingested_at": datetime.now().isoformat(),
            },
        }

    def to_semantic_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for semantic memory storage.

        Returns:
            Dictionary with semantic memory fields formatted
            for ChromaDB/Neo4j storage.

        Raises:
            ValueError: If memory_type is not SEMANTIC
        """
        if self.memory_type != MemoryType.SEMANTIC:
            raise ValueError(
                f"Cannot convert {self.memory_type.value} to semantic format"
            )

        return {
            "content": self.content,
            "type": "semantic",
            "knowledge_type": self.knowledge_type or "general",
            "category": self.category,
            "themes": self.themes,
            "stability": self.stability,
            "importance": self.importance,
            "confidence": self.confidence,
            "source": {
                "file": self.source_file,
                "chunk_index": self.source_chunk_index,
            },
            "metadata": {
                "emotional_context": self.emotional_valence,
                "ingested_at": datetime.now().isoformat(),
            },
        }


@dataclass
class IngestResult:
    """
    Result of processing a single item.

    Tracks success/failure and statistics for one ingestion operation.

    Attributes:
        source_file: Original source filename
        success: Whether ingestion succeeded
        error: Error message if failed
        memory_candidates: Generated memory candidates
        processing_time_ms: Total processing time
        chunks_generated: Number of content chunks
        episodic_count: Number of episodic memories
        semantic_count: Number of semantic memories
        skipped_count: Number of skipped chunks
    """
    source_file: str
    success: bool
    memory_candidates: List[MemoryCandidate] = field(default_factory=list)
    processing_time_ms: float = 0.0
    chunks_generated: int = 0
    episodic_count: int = 0
    semantic_count: int = 0
    skipped_count: int = 0
    error: Optional[str] = None

    @classmethod
    def failure(cls, source_file: str, error: str) -> "IngestResult":
        """Create a failure result."""
        return cls(
            source_file=source_file,
            success=False,
            error=error,
        )

    @classmethod
    def from_candidates(
        cls,
        source_file: str,
        candidates: List[MemoryCandidate],
        processing_time_ms: float
    ) -> "IngestResult":
        """Create a success result from memory candidates."""
        episodic = sum(1 for c in candidates if c.memory_type == MemoryType.EPISODIC)
        semantic = sum(1 for c in candidates if c.memory_type == MemoryType.SEMANTIC)
        skipped = sum(1 for c in candidates if c.memory_type == MemoryType.SKIP)

        return cls(
            source_file=source_file,
            success=True,
            memory_candidates=candidates,
            processing_time_ms=processing_time_ms,
            chunks_generated=len(candidates),
            episodic_count=episodic,
            semantic_count=semantic,
            skipped_count=skipped,
        )


@dataclass
class BatchIngestResult:
    """
    Aggregated results from batch ingestion.

    Summarizes outcomes across multiple files.

    Attributes:
        total_files: Number of files processed
        successful: Number of successful ingestions
        failed: Number of failed ingestions
        total_memories: Total memories generated
        total_episodic: Total episodic memories
        total_semantic: Total semantic memories
        total_skipped: Total skipped chunks
        total_time_ms: Total processing time
        results: Individual results by file
        errors: Error messages by file
    """
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    total_memories: int = 0
    total_episodic: int = 0
    total_semantic: int = 0
    total_skipped: int = 0
    total_time_ms: float = 0.0
    results: Dict[str, IngestResult] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def add_result(self, result: IngestResult) -> None:
        """Add an individual result to the batch."""
        self.total_files += 1
        self.results[result.source_file] = result

        if result.success:
            self.successful += 1
            self.total_memories += len(result.memory_candidates)
            self.total_episodic += result.episodic_count
            self.total_semantic += result.semantic_count
            self.total_skipped += result.skipped_count
            self.total_time_ms += result.processing_time_ms
        else:
            self.failed += 1
            if result.error:
                self.errors[result.source_file] = result.error

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.successful / self.total_files) * 100

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Batch Ingestion Complete\n"
            f"{'=' * 40}\n"
            f"Files: {self.successful}/{self.total_files} successful "
            f"({self.success_rate:.1f}%)\n"
            f"Memories: {self.total_memories} total\n"
            f"  - Episodic: {self.total_episodic}\n"
            f"  - Semantic: {self.total_semantic}\n"
            f"  - Skipped: {self.total_skipped}\n"
            f"Time: {self.total_time_ms:.1f}ms"
        )
