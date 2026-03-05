"""
UBIK Ingestion System - Main Pipeline

Orchestrates the complete content ingestion workflow:
    File/Drive → Process → Chunk → Classify → Store

Integrates all pipeline components into a unified interface
with support for local files, directories, and Google Drive.

Features:
    - Multi-format file processing
    - Smart content chunking
    - Memory classification (episodic/semantic)
    - MCP storage integration (HippocampalClient)
    - Neo4j graph operations for transcripts
    - Batch processing with aggregated results
    - Dry-run mode for testing

Usage:
    from ingest.pipeline import IngestPipeline

    async with IngestPipeline(mcp_client=client) as pipeline:
        # Single file
        result = await pipeline.ingest_file(Path("document.pdf"))

        # Directory
        batch = await pipeline.ingest_directory(Path("./documents"))

        # Google Drive
        batch = await pipeline.ingest_from_gdrive(
            folder_id="1ABC123",
            credentials_path="credentials.json"
        )

Version: 1.0.0
"""

import asyncio
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .chunkers import Chunk, ChunkConfig, SmartChunker
from .classifiers import ClassifierConfig, ContentClassifier
from .models import (
    BatchIngestResult,
    ContentType,
    IngestItem,
    IngestResult,
    MemoryCandidate,
    MemoryType,
    ProcessedContent,
)
from .processors import ProcessorConfig, ProcessorRegistry
from .sources import GoogleDriveConfig, GoogleDriveSource
from .tracker import FileMover, IngestionManifest, IngestionRecord, compute_file_hash
from .transcript_processor import (
    TranscriptProcessor,
    transcript_to_memory_candidates,
)

__all__ = [
    'IngestPipeline',
    'PipelineConfig',
    'StorageStats',
]

# Pipeline version — used in IngestionRecord for traceability
_PIPELINE_VERSION = "1.1.0"

logger = logging.getLogger(__name__)


@dataclass
class StorageStats:
    """
    Statistics for memory storage operations.

    Tracks successful and failed storage attempts.
    """
    stored: int = 0
    failed: int = 0
    episodic_stored: int = 0
    semantic_stored: int = 0
    intellectual_stored: int = 0
    neo4j_ops_executed: int = 0
    neo4j_ops_failed: int = 0
    memory_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """
    Configuration for the ingestion pipeline.

    Attributes:
        storage_mode: If True, store to MCP. If False, dry run.
        whisper_model: Whisper model size for audio transcription
        chunk_config: Configuration for content chunking
        classifier_config: Configuration for content classification
        processor_config: Configuration for format processors
        temp_dir: Directory for temporary files
        cleanup_temp: Whether to clean up temp files after processing
        parallel_files: Number of files to process in parallel
        continue_on_storage_error: Continue processing if storage fails
        execute_neo4j_ops: Whether to execute Neo4j graph operations
    """
    storage_mode: bool = True
    whisper_model: str = "base"
    chunk_config: Optional[ChunkConfig] = None
    classifier_config: Optional[ClassifierConfig] = None
    processor_config: Optional[ProcessorConfig] = None
    temp_dir: Optional[Path] = None
    cleanup_temp: bool = True
    parallel_files: int = 4
    continue_on_storage_error: bool = True
    execute_neo4j_ops: bool = True


class IngestPipeline:
    """
    Main ingestion pipeline orchestrating the complete workflow.

    Coordinates processing, chunking, classification, and storage
    of content from various sources.

    Attributes:
        mcp_client: Optional MCP client for memory storage
        config: Pipeline configuration
        processor_registry: Registry of format processors
        chunker: Content chunker
        classifier: Content classifier

    Example:
        # With MCP storage (using HippocampalClient)
        from mcp_client import HippocampalClient

        async with HippocampalClient() as mcp:
            async with IngestPipeline(mcp_client=mcp) as pipeline:
                result = await pipeline.ingest_file(Path("memoir.pdf"))
                print(f"Created {result.episodic_count} episodic memories")

        # Dry run (no storage)
        pipeline = IngestPipeline(storage_mode=False)
        result = await pipeline.ingest_file(Path("test.txt"))
    """

    def __init__(
        self,
        mcp_client: Optional[Any] = None,
        storage_mode: bool = True,
        whisper_model: str = "base",
        chunk_config: Optional[ChunkConfig] = None,
        classifier_config: Optional[ClassifierConfig] = None,
        config: Optional[PipelineConfig] = None,
        tracker: Optional[IngestionManifest] = None,
        file_mover: Optional[FileMover] = None,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            mcp_client: HippocampalClient instance for storage
            storage_mode: Enable/disable MCP storage
            whisper_model: Whisper model size
            chunk_config: Chunking configuration
            classifier_config: Classification configuration
            config: Full pipeline configuration (overrides other params)
            tracker: Optional IngestionManifest for dedup and audit logging.
                     When provided, already-ingested files are skipped.
            file_mover: Optional FileMover for move-after-ingest.
                        When provided, successfully processed files are
                        moved to the Ingested_data archive.
        """
        # Use config or build from params
        if config:
            self.config = config
        else:
            self.config = PipelineConfig(
                storage_mode=storage_mode,
                whisper_model=whisper_model,
                chunk_config=chunk_config,
                classifier_config=classifier_config,
            )

        self.mcp_client = mcp_client
        self.tracker = tracker
        self.file_mover = file_mover

        # Initialize processor config with whisper model
        processor_config = self.config.processor_config or ProcessorConfig()
        processor_config.whisper_model = self.config.whisper_model

        # Initialize components
        self.processor_registry = ProcessorRegistry(processor_config)

        # Register transcript processor
        transcript_processor = TranscriptProcessor(processor_config)
        self.processor_registry.register(".transcript", transcript_processor)

        self.chunker = SmartChunker(
            self.config.chunk_config or ChunkConfig()
        )

        self.classifier = ContentClassifier(
            self.config.classifier_config or ClassifierConfig()
        )

        # Track state
        self._connected = False
        self._temp_dirs: List[Path] = []
        self._storage_stats = StorageStats()

    async def __aenter__(self) -> "IngestPipeline":
        """
        Enter async context and initialize connections.

        Connects to MCP if client provided. Uses the client's
        context manager protocol if available.
        """
        if self.mcp_client is not None:
            try:
                # Try using context manager protocol
                if hasattr(self.mcp_client, '__aenter__'):
                    await self.mcp_client.__aenter__()
                    self._connected = True
                # Fallback to explicit connect method
                elif hasattr(self.mcp_client, 'connect'):
                    await self.mcp_client.connect()
                    self._connected = True
                # Check if already initialized
                elif hasattr(self.mcp_client, '_initialized'):
                    self._connected = self.mcp_client._initialized

                if self._connected:
                    logger.info("Connected to MCP server")
            except Exception as e:
                logger.warning(f"Failed to connect to MCP: {e}")
                self._connected = False

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit async context and cleanup.

        Disconnects from MCP and cleans up temp directories.
        """
        if self.mcp_client is not None and self._connected:
            try:
                # Try using context manager protocol
                if hasattr(self.mcp_client, '__aexit__'):
                    await self.mcp_client.__aexit__(exc_type, exc_val, exc_tb)
                # Fallback to explicit close/disconnect
                elif hasattr(self.mcp_client, 'close'):
                    await self.mcp_client.close()
                elif hasattr(self.mcp_client, 'disconnect'):
                    await self.mcp_client.disconnect()

                logger.info("Disconnected from MCP server")
            except Exception as e:
                logger.warning(f"Error disconnecting from MCP: {e}")

        # Cleanup temp directories
        if self.config.cleanup_temp:
            for temp_dir in self._temp_dirs:
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                        logger.debug(f"Cleaned up temp dir: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {temp_dir}: {e}")

        self._connected = False

    # =========================================================================
    # Main Ingestion Methods
    # =========================================================================

    async def ingest_file(self, file_path: Path) -> IngestResult:
        """
        Ingest a single file through the complete pipeline.

        Steps:
        1. Deduplication check (if tracker configured)
        2. Create IngestItem from path
        3. Process with appropriate processor
        4. Chunk the content
        5. Classify each chunk
        6. Store memories (if storage_mode enabled)
        7. Execute Neo4j operations (for transcripts)
        8. Move file to archive (if file_mover configured)
        9. Record ingestion to manifest (if tracker configured)

        Args:
            file_path: Path to file to ingest

        Returns:
            IngestResult with processing statistics.
            Returns a skipped_duplicate result if tracker detects the file
            has already been ingested.
        """
        start_time = time.time()
        file_path = Path(file_path).expanduser().resolve()

        logger.info(f"Ingesting file: {file_path.name}")

        # ------------------------------------------------------------------
        # Deduplication check (compute hash once; reuse for record building)
        # ------------------------------------------------------------------
        file_hash: Optional[str] = None
        if self.tracker is not None:
            try:
                file_hash, already_ingested = self.tracker.check_file(file_path)
                if already_ingested:
                    logger.info(f"Skipping already-ingested: {file_path.name}")
                    return IngestResult.skipped_duplicate_result(
                        source_file=file_path.name
                    )
            except Exception as e:
                logger.warning(
                    f"Dedup check failed for {file_path.name}: {e} — processing anyway"
                )

        try:
            # Create IngestItem
            item = IngestItem.from_path(file_path)

            # Check if we can process this file
            if not self.processor_registry.can_process(item.file_extension):
                return IngestResult.failure(
                    source_file=file_path.name,
                    error=f"Unsupported file type: {item.file_extension}"
                )

            # Process file
            processed = await self._process_item(item, file_path)

            # Chunk and classify
            candidates = await self._process_to_candidates(processed)

            # Filter out SKIP candidates
            active_candidates = [
                c for c in candidates
                if c.memory_type != MemoryType.SKIP
            ]

            # Store memories
            storage_stats = StorageStats()
            if self.config.storage_mode and active_candidates:
                storage_stats = await self._store_memories(
                    active_candidates,
                    processed
                )
                logger.info(
                    f"Stored {storage_stats.stored} memories "
                    f"({storage_stats.failed} failed)"
                )

            # Build result
            processing_time = (time.time() - start_time) * 1000
            result = IngestResult.from_candidates(
                source_file=file_path.name,
                candidates=candidates,
                processing_time_ms=processing_time,
                memory_ids=storage_stats.memory_ids,
            )

            logger.info(
                f"Completed {file_path.name}: "
                f"{result.episodic_count} episodic, "
                f"{result.semantic_count} semantic"
            )

            # ------------------------------------------------------------------
            # Post-processing: move file, then record ingestion
            # ------------------------------------------------------------------
            if self.tracker is not None and result.success:
                dest_path = ""

                # Move file BEFORE recording so destination_path is accurate
                if self.file_mover is not None:
                    try:
                        moved_to = self.file_mover.move(
                            file_path,
                            source_directory=file_path.parent.name,
                        )
                        dest_path = str(moved_to)
                        logger.info(f"Moved {file_path.name} -> {moved_to}")
                    except Exception as move_err:
                        # Log but don't fail — the memory is already stored
                        logger.error(
                            f"Failed to move {file_path.name}: {move_err} "
                            f"(memory was stored; file remains at original path)"
                        )

                # Determine storage status
                if not self.config.storage_mode:
                    storage_status = "dry_run"
                elif storage_stats.failed > 0 and storage_stats.stored > 0:
                    storage_status = "partial"
                elif storage_stats.failed > 0 and storage_stats.stored == 0:
                    storage_status = "failed"
                else:
                    storage_status = "stored"

                # Compute hash if not already done (tracker may have been None earlier)
                if file_hash is None:
                    try:
                        # File may have been moved; use dest_path if available
                        hash_source = Path(dest_path) if dest_path else file_path
                        file_hash = compute_file_hash(hash_source)
                    except Exception:
                        file_hash = ""

                error_msg: Optional[str] = None
                if storage_stats.errors:
                    error_msg = "; ".join(storage_stats.errors[:3])

                record = IngestionRecord(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    file_hash=file_hash,
                    file_size_bytes=item.file_size_bytes,
                    ingested_at=datetime.now(timezone.utc).isoformat(),
                    source_directory=file_path.parent.name,
                    destination_path=dest_path,
                    content_type=item.content_type.value,
                    chunks_generated=result.chunks_generated,
                    episodic_memories=result.episodic_count,
                    semantic_memories=result.semantic_count,
                    skipped_memories=result.skipped_count,
                    processing_time_ms=result.processing_time_ms,
                    hippocampal_connected=self._connected,
                    storage_status=storage_status,
                    pipeline_version=_PIPELINE_VERSION,
                    error=error_msg,
                    memory_ids=storage_stats.memory_ids,
                )
                self.tracker.record_ingestion(record)

            return result

        except Exception as e:
            logger.error(f"Failed to ingest {file_path.name}: {e}")
            return IngestResult.failure(
                source_file=file_path.name,
                error=str(e)
            )

    async def ingest_directory(
        self,
        directory: Path,
        recursive: bool = True,
        extensions: Optional[List[str]] = None
    ) -> BatchIngestResult:
        """
        Ingest all supported files in a directory.

        Args:
            directory: Directory to scan
            recursive: Whether to traverse subdirectories
            extensions: Optional list of extensions to filter

        Returns:
            BatchIngestResult with aggregated statistics
        """
        directory = Path(directory).expanduser().resolve()

        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        logger.info(f"Ingesting directory: {directory}")

        # Collect files
        files = self._collect_files(directory, recursive, extensions)
        logger.info(f"Found {len(files)} files to process")

        # Process files
        batch_result = BatchIngestResult()

        # Process in batches for parallel execution
        for i in range(0, len(files), self.config.parallel_files):
            batch = files[i:i + self.config.parallel_files]
            tasks = [self.ingest_file(f) for f in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    # Handle unexpected exceptions
                    batch_result.failed += 1
                    batch_result.total_files += 1
                elif isinstance(result, IngestResult):
                    batch_result.add_result(result)

        logger.info(batch_result.summary())
        return batch_result

    async def ingest_from_gdrive(
        self,
        folder_id: str,
        credentials_path: str,
        token_path: str,
        recursive: bool = True
    ) -> BatchIngestResult:
        """
        Ingest files from a Google Drive folder.

        Downloads files to a temporary directory, processes them,
        and optionally cleans up.

        Args:
            folder_id: Google Drive folder ID
            credentials_path: Path to OAuth credentials
            token_path: Path to OAuth token
            recursive: Whether to traverse subfolders

        Returns:
            BatchIngestResult with aggregated statistics
        """
        logger.info(f"Ingesting from Google Drive folder: {folder_id}")

        # Initialize Google Drive source
        gdrive = GoogleDriveSource(
            credentials_path=credentials_path,
            token_path=token_path
        )

        # Authenticate
        await gdrive.authenticate()

        # List files
        items = await gdrive.list_files(folder_id, recursive=recursive)
        logger.info(f"Found {len(items)} files in Drive")

        if not items:
            return BatchIngestResult()

        # Create temp directory for downloads
        temp_dir = Path(tempfile.mkdtemp(prefix="ubik_gdrive_"))
        self._temp_dirs.append(temp_dir)
        logger.info(f"Using temp directory: {temp_dir}")

        batch_result = BatchIngestResult()

        # Process files
        for item in items:
            try:
                # Download file
                file_id = item.source_metadata.get("file_id")
                if not file_id:
                    continue

                local_path = await gdrive.download_file(file_id, temp_dir)

                # Process the downloaded file
                result = await self.ingest_file(local_path)
                batch_result.add_result(result)

            except Exception as e:
                logger.error(f"Failed to process {item.original_filename}: {e}")
                batch_result.add_result(IngestResult.failure(
                    source_file=item.original_filename,
                    error=str(e)
                ))

        logger.info(batch_result.summary())
        return batch_result

    # =========================================================================
    # Internal Processing Methods
    # =========================================================================

    async def _process_item(
        self,
        item: IngestItem,
        file_path: Path
    ) -> ProcessedContent:
        """
        Process a file with the appropriate processor.

        Args:
            item: IngestItem with metadata
            file_path: Path to the file

        Returns:
            ProcessedContent with extracted text
        """
        return await self.processor_registry.process(item, file_path)

    async def _chunk_content(
        self,
        processed: ProcessedContent
    ) -> List[Chunk]:
        """
        Chunk processed content.

        Args:
            processed: ProcessedContent to chunk

        Returns:
            List of Chunk objects
        """
        return self.chunker.chunk(processed)

    async def _classify_chunks(
        self,
        chunks: List[Chunk],
        processed: ProcessedContent
    ) -> List[MemoryCandidate]:
        """
        Classify chunks into memory candidates.

        Args:
            chunks: List of chunks to classify
            processed: Original ProcessedContent for context

        Returns:
            List of MemoryCandidate objects
        """
        candidates = []

        for chunk in chunks:
            candidate = self.classifier.classify(chunk, processed)
            candidates.append(candidate)

        return candidates

    async def _process_to_candidates(
        self,
        processed: ProcessedContent
    ) -> List[MemoryCandidate]:
        """
        Process content to memory candidates.

        Handles special case for transcripts.

        Args:
            processed: ProcessedContent to process

        Returns:
            List of MemoryCandidate objects
        """
        # Check for transcript chunks (special handling)
        transcript_chunks = processed.extracted_metadata.get("transcript_chunks")

        if transcript_chunks:
            # Use transcript-specific conversion
            return transcript_to_memory_candidates(processed)

        # Standard chunking and classification
        chunks = await self._chunk_content(processed)
        return await self._classify_chunks(chunks, processed)

    # =========================================================================
    # MCP Storage Methods
    # =========================================================================

    async def _store_memories(
        self,
        candidates: List[MemoryCandidate],
        processed: ProcessedContent
    ) -> StorageStats:
        """
        Store memory candidates to MCP and execute Neo4j operations.

        Args:
            candidates: List of candidates to store
            processed: ProcessedContent for Neo4j operations

        Returns:
            StorageStats with success/failure counts
        """
        stats = StorageStats()

        if not self.mcp_client:
            logger.warning("No MCP client configured, skipping storage")
            return stats

        if not self._connected:
            logger.warning("MCP not connected, skipping storage")
            return stats

        # Store each memory candidate
        for candidate in candidates:
            if candidate.memory_type == MemoryType.SKIP:
                continue

            try:
                if candidate.memory_type == MemoryType.EPISODIC:
                    memory_id = await self._store_episodic(candidate)
                    if memory_id:
                        stats.episodic_stored += 1
                elif candidate.memory_type == MemoryType.INTELLECTUAL:
                    memory_id = await self._store_intellectual(candidate)
                    if memory_id:
                        stats.intellectual_stored += 1
                else:
                    memory_id = await self._store_semantic(candidate)
                    if memory_id:
                        stats.semantic_stored += 1

                if memory_id:
                    stats.stored += 1
                    stats.memory_ids.append(memory_id)
                else:
                    stats.failed += 1

            except Exception as e:
                stats.failed += 1
                stats.errors.append(str(e))
                logger.error(f"Failed to store memory: {e}")

                if not self.config.continue_on_storage_error:
                    raise

        # Execute Neo4j operations if present
        if self.config.execute_neo4j_ops:
            neo4j_ops = processed.extracted_metadata.get("neo4j_operations", [])
            if neo4j_ops:
                neo4j_stats = await self._execute_neo4j_operations(neo4j_ops)
                stats.neo4j_ops_executed = neo4j_stats[0]
                stats.neo4j_ops_failed = neo4j_stats[1]

        return stats

    async def _store_episodic(self, candidate: MemoryCandidate) -> Optional[str]:
        """
        Store an episodic memory to MCP.

        Uses HippocampalClient.store_episodic() with proper parameter mapping.

        Args:
            candidate: MemoryCandidate to store

        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            # Format timestamp
            timestamp = None
            if candidate.timestamp:
                timestamp = candidate.timestamp.isoformat() + "Z"
            else:
                timestamp = datetime.now().isoformat() + "Z"

            # Format participants as comma-separated string
            participants = ",".join(candidate.participants) if candidate.participants else "gines"

            # Format themes as comma-separated string
            themes = ",".join(candidate.themes) if candidate.themes else ""

            # Call MCP tool with HippocampalClient interface
            result = await self.mcp_client.store_episodic(
                content=candidate.content,
                memory_type=candidate.event_type or "general",
                timestamp=timestamp,
                emotional_valence=candidate.emotional_valence,
                importance=candidate.importance,
                participants=participants,
                themes=themes,
                source_file=candidate.source_file,
            )

            # Handle response
            if isinstance(result, dict):
                if result.get("status") == "error":
                    logger.warning(f"Episodic storage error: {result.get('message')}")
                    return None
                return result.get("memory_id") or result.get("id")

            return None

        except Exception as e:
            logger.error(f"Failed to store episodic memory: {e}")
            raise

    async def _store_semantic(self, candidate: MemoryCandidate) -> Optional[str]:
        """
        Store a semantic memory to MCP.

        Uses HippocampalClient.store_semantic() with proper parameter mapping.

        Args:
            candidate: MemoryCandidate to store

        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            # Call MCP tool with HippocampalClient interface
            result = await self.mcp_client.store_semantic(
                content=candidate.content,
                knowledge_type=candidate.knowledge_type or "belief",
                category=candidate.category,
                confidence=candidate.confidence,
                stability=candidate.stability,
                source=candidate.source_file,
            )

            # Handle response
            if isinstance(result, dict):
                if result.get("status") == "error":
                    logger.warning(f"Semantic storage error: {result.get('message')}")
                    return None
                return result.get("knowledge_id") or result.get("memory_id") or result.get("id")

            return None

        except Exception as e:
            logger.error(f"Failed to store semantic memory: {e}")
            raise

    async def _store_intellectual(self, candidate: MemoryCandidate) -> Optional[str]:
        """
        Store an intellectual memory (external content) via the writer.

        Routes to store_intellectual() on LocalMemoryWriter (ubik_intellectual
        collection). Falls back to store_semantic() if the client does not
        implement store_intellectual.

        Args:
            candidate: MemoryCandidate with memory_type == INTELLECTUAL

        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            themes = ",".join(candidate.themes) if candidate.themes else ""

            if hasattr(self.mcp_client, "store_intellectual"):
                result = await self.mcp_client.store_intellectual(
                    content=candidate.content,
                    importance=candidate.importance,
                    themes=themes,
                    source_file=candidate.source_file,
                )
            else:
                # Fallback for HippocampalClient which lacks store_intellectual
                result = await self.mcp_client.store_semantic(
                    content=candidate.content,
                    knowledge_type=candidate.knowledge_type or "external_insight",
                    category=candidate.category,
                    confidence=candidate.confidence,
                    stability=candidate.stability,
                    source=candidate.source_file,
                )

            if isinstance(result, dict):
                if result.get("status") == "error":
                    logger.warning(f"Intellectual storage error: {result.get('message')}")
                    return None
                return result.get("memory_id") or result.get("knowledge_id") or result.get("id")

            return None

        except Exception as e:
            logger.error(f"Failed to store intellectual memory: {e}")
            raise

    async def _execute_neo4j_operations(
        self,
        operations: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """
        Execute Neo4j graph operations from transcript processing.

        Args:
            operations: List of operation dicts from MeetingMetadata

        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not self.mcp_client:
            return (0, 0)

        if not hasattr(self.mcp_client, 'update_identity_graph'):
            logger.debug("MCP client doesn't support update_identity_graph")
            return (0, 0)

        successful = 0
        failed = 0

        for op in operations:
            try:
                op_type = op.get("operation")

                if op_type == "merge_relationship":
                    # Convert to update_identity_graph call
                    result = await self.mcp_client.update_identity_graph(
                        from_concept=op.get("from_id", ""),
                        relation_type=op.get("rel_type", "RELATES_TO"),
                        to_concept=op.get("to_id", ""),
                        weight=1.0,
                        context=str(op.get("properties", {})),
                    )

                    if result.get("status") != "error":
                        successful += 1
                    else:
                        failed += 1

                elif op_type == "merge_node":
                    # Node creation could be done via relationship to Self
                    # For now, just count as successful (nodes are created implicitly)
                    successful += 1

                else:
                    logger.debug(f"Unknown Neo4j operation type: {op_type}")

            except Exception as e:
                logger.warning(f"Neo4j operation failed: {e}")
                failed += 1

        logger.info(f"Neo4j operations: {successful} successful, {failed} failed")
        return (successful, failed)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _collect_files(
        self,
        directory: Path,
        recursive: bool,
        extensions: Optional[List[str]]
    ) -> List[Path]:
        """
        Collect files from directory.

        Args:
            directory: Directory to scan
            recursive: Whether to recurse into subdirectories
            extensions: Optional list of extensions to filter

        Returns:
            List of file paths
        """
        files = []

        # Normalize extensions
        if extensions:
            ext_set = {
                ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                for ext in extensions
            }
        else:
            ext_set = None

        # Walk directory
        if recursive:
            for path in directory.rglob("*"):
                if self._should_include(path, ext_set):
                    files.append(path)
        else:
            for path in directory.iterdir():
                if self._should_include(path, ext_set):
                    files.append(path)

        # Sort by name for consistent ordering
        return sorted(files)

    def _should_include(
        self,
        path: Path,
        extensions: Optional[Set[str]]
    ) -> bool:
        """Check if file should be included in processing."""
        if not path.is_file():
            return False

        # Skip hidden files
        if path.name.startswith('.'):
            return False

        ext = path.suffix.lower()

        # Filter by provided extensions
        if extensions and ext not in extensions:
            return False

        # Check if we can process
        return self.processor_registry.can_process(ext)

    @property
    def supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return self.processor_registry.supported_extensions

    @property
    def is_connected(self) -> bool:
        """Check if MCP client is connected."""
        return self._connected
