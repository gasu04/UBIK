"""
UBIK Ingestion Tracker - File Deduplication and Move-After-Ingest

Handles file tracking, SHA-256 deduplication, move-after-ingest logic,
and append-only JSONL manifest management.

Provides:
    compute_file_hash: SHA-256 hash utility (8 KB chunks, memory-safe for audio)
    IngestionRecord: Per-file ingestion dataclass persisted in the audit log
    IngestionManifest: Loads/saves the JSONL manifest; hash-based dedup
    FileMover: Moves processed files into ~/ubik/Ingested_data/ archive

Usage:
    from ingest.tracker import IngestionManifest, FileMover, compute_file_hash

    manifest = IngestionManifest(log_dir=Path("~/ubik/Ingested_data/ingestion_log"))
    file_hash, already_done = manifest.check_file(Path("session.transcript"))

    if not already_done:
        # ... run pipeline ...
        record = IngestionRecord(file_hash=file_hash, ...)
        manifest.record_ingestion(record)

    mover = FileMover(base_ingested_dir=Path("~/ubik/Ingested_data"))
    dest = mover.move(Path("therapy/session.transcript"), source_directory="therapy")

Version: 1.0.0
"""

import csv
import fcntl
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "IngestionRecord",
    "IngestionManifest",
    "FileMover",
    "compute_file_hash",
]


# ---------------------------------------------------------------------------
# Hash utility
# ---------------------------------------------------------------------------

def compute_file_hash(path: Path) -> str:
    """
    Compute SHA-256 hash of a file, reading in 8 KB chunks.

    Memory-safe for large audio files. Returns the same digest regardless
    of filename or path — used for content-based deduplication.

    Args:
        path: Path to the file to hash.

    Returns:
        Lowercase hex digest string (64 characters).

    Raises:
        FileNotFoundError: If path does not exist.
        OSError: If the file cannot be read.

    Example:
        >>> h = compute_file_hash(Path("session.transcript"))
        >>> len(h)
        64
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class IngestionRecord:
    """
    Immutable record of one file ingestion event.

    Stored as a single JSON line in ingestion_manifest.jsonl and as a row
    in ingestion_manifest.csv.

    Attributes:
        file_path: Original absolute path before any move.
        file_name: Basename of the source file.
        file_hash: SHA-256 hex digest of file content (dedup key).
        file_size_bytes: File size at time of ingestion.
        ingested_at: ISO 8601 UTC timestamp of the ingestion event.
        source_directory: Parent folder name (e.g. "therapy").
        destination_path: Absolute path after move; empty string if not moved.
        content_type: Detected content type string (e.g. "text", "audio").
        chunks_generated: Number of content chunks produced.
        episodic_memories: Number of episodic memories stored.
        semantic_memories: Number of semantic memories stored.
        skipped_memories: Number of chunks classified as SKIP.
        processing_time_ms: Wall-clock processing time in milliseconds.
        hippocampal_connected: Whether MCP storage was active.
        storage_status: "stored" | "dry_run" | "failed" | "partial" | "reprocessing".
        pipeline_version: ingest package version at time of ingestion.
        error: Error message if storage_status is "failed" or "partial".
        memory_ids: ChromaDB document IDs for stored memories.
    """
    file_path: str
    file_name: str
    file_hash: str
    file_size_bytes: int
    ingested_at: str
    source_directory: str
    destination_path: str
    content_type: str
    chunks_generated: int
    episodic_memories: int
    semantic_memories: int
    skipped_memories: int
    processing_time_ms: float
    hippocampal_connected: bool
    storage_status: str
    pipeline_version: str
    error: Optional[str] = None
    memory_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a plain dict suitable for JSON or CSV output.

        memory_ids is serialized as a JSON string for CSV compatibility.

        Returns:
            Dict with all fields as JSON-serializable types.
        """
        d = asdict(self)
        d["memory_ids"] = json.dumps(d["memory_ids"])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestionRecord":
        """
        Deserialize from a plain dict (loaded from a JSONL line).

        Args:
            data: Dict as returned by json.loads() on a JSONL line.

        Returns:
            IngestionRecord instance.
        """
        data = dict(data)
        # memory_ids may be a JSON string (from CSV) or already a list (from JSONL)
        raw_ids = data.get("memory_ids", [])
        if isinstance(raw_ids, str):
            try:
                data["memory_ids"] = json.loads(raw_ids)
            except (json.JSONDecodeError, TypeError):
                data["memory_ids"] = []
        # Only pass known fields to avoid TypeError on future schema extensions
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class IngestionManifest:
    """
    Append-only JSONL manifest tracking all ingested files.

    Loads existing records on init for in-memory lookup. Hash-based
    deduplication prevents re-processing identical file content.

    Re-ingestion is supported via tombstone records: calling tombstone()
    appends a record with storage_status="reprocessing". The next call to
    check_file() (or is_already_ingested()) will treat the file as new.

    File locking (fcntl LOCK_EX) prevents concurrent write corruption on
    Linux / WSL2.

    Note:
        Log rotation is not implemented in Phase 1. Consider adding rotation
        (e.g. gzip-compress and archive) if the manifest exceeds ~10,000
        records to keep load time acceptable.

    Example:
        manifest = IngestionManifest(log_dir=Path("~/ubik/Ingested_data/ingestion_log"))
        hash_val, already_done = manifest.check_file(Path("therapy/session.transcript"))
        if not already_done:
            # run pipeline ...
            manifest.record_ingestion(record)
    """

    MANIFEST_FILE = "ingestion_manifest.jsonl"

    def __init__(self, log_dir: Path) -> None:
        """
        Initialize and load the manifest from log_dir.

        Args:
            log_dir: Directory containing ingestion_manifest.jsonl.
                     Created automatically on first use.
        """
        self._log_dir = Path(log_dir).expanduser().resolve()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._log_dir / self.MANIFEST_FILE

        self._records: List[IngestionRecord] = []
        # hash -> list of records (multiple records possible after reingest cycles)
        self._hash_index: Dict[str, List[IngestionRecord]] = {}

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_file(self, path: Path) -> Tuple[str, bool]:
        """
        Compute a file's SHA-256 hash and check if it has been ingested.

        Returns both the hash and the result so callers can reuse the hash
        for record building without double-reading the file.

        Files with only tombstone records (storage_status="reprocessing")
        are treated as NOT ingested.

        Args:
            path: Absolute path to the file to check.

        Returns:
            Tuple of (sha256_hex_digest, is_already_ingested).

        Raises:
            FileNotFoundError: If path does not exist.
        """
        file_hash = compute_file_hash(path)
        records = self._hash_index.get(file_hash, [])
        already = any(r.storage_status != "reprocessing" for r in records)
        return file_hash, already

    def is_already_ingested(self, file_path: str) -> bool:
        """
        Check whether a file path has already been ingested.

        Convenience wrapper around check_file() that discards the hash.

        Args:
            file_path: Absolute path string to the file.

        Returns:
            True if a non-tombstone record with matching hash exists.
        """
        _, already = self.check_file(Path(file_path))
        return already

    def get_last_record_for_hash(self, file_hash: str) -> Optional["IngestionRecord"]:
        """
        Get the most recent non-tombstone record for a given hash.

        Args:
            file_hash: SHA-256 hex digest.

        Returns:
            Most recent IngestionRecord, or None if not found.
        """
        records = [
            r for r in self._hash_index.get(file_hash, [])
            if r.storage_status != "reprocessing"
        ]
        if not records:
            return None
        return max(records, key=lambda r: r.ingested_at)

    def record_ingestion(self, record: "IngestionRecord") -> None:
        """
        Append a new ingestion record to the JSONL manifest.

        Delegates CSV and error-log writes to IngestionLogWriter.
        Updates the in-memory index immediately.

        Args:
            record: Completed IngestionRecord to persist.
        """
        from .log_writer import IngestionLogWriter  # late import avoids circular

        self._append_jsonl(record)
        writer = IngestionLogWriter(self._log_dir)
        writer.write_record(record)

        self._records.append(record)
        self._hash_index.setdefault(record.file_hash, []).append(record)

    def get_records(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List["IngestionRecord"]:
        """
        Query manifest records with optional filters (newest first).

        All filter conditions use AND logic. String values use case-insensitive
        substring matching; non-string values require exact equality.

        Args:
            filters: Dict of field_name -> value. Pass None for all records.

        Returns:
            List of matching IngestionRecord objects, newest first.

        Example:
            records = manifest.get_records({"source_directory": "therapy"})
            errors  = manifest.get_records({"storage_status": "failed"})
        """
        results = list(self._records)
        if filters:
            for key, value in filters.items():
                filtered = []
                for r in results:
                    field_val = getattr(r, key, None)
                    if field_val is None:
                        continue
                    if isinstance(field_val, str) and isinstance(value, str):
                        if value.lower() in field_val.lower():
                            filtered.append(r)
                    elif field_val == value:
                        filtered.append(r)
                results = filtered
        return sorted(results, key=lambda r: r.ingested_at, reverse=True)

    def get_by_hash(self, file_hash: str) -> List["IngestionRecord"]:
        """
        Return all records for a specific SHA-256 hash.

        Args:
            file_hash: SHA-256 hex digest.

        Returns:
            List of IngestionRecord objects (may be empty).
        """
        return list(self._hash_index.get(file_hash, []))

    def get_stats(self) -> Dict[str, Any]:
        """
        Return summary statistics over all manifest records.

        Returns:
            Dict with keys: total_records, unique_files, total_episodic,
            total_semantic, total_chunks, error_count, last_ingested_at,
            storage_status_counts.
        """
        if not self._records:
            return {
                "total_records": 0,
                "unique_files": 0,
                "total_episodic": 0,
                "total_semantic": 0,
                "total_chunks": 0,
                "error_count": 0,
                "last_ingested_at": None,
                "storage_status_counts": {},
            }

        status_counts: Dict[str, int] = {}
        for r in self._records:
            status_counts[r.storage_status] = status_counts.get(r.storage_status, 0) + 1

        unique_hashes = {
            r.file_hash for r in self._records
            if r.storage_status != "reprocessing"
        }
        last = max(self._records, key=lambda r: r.ingested_at)

        return {
            "total_records": len(self._records),
            "unique_files": len(unique_hashes),
            "total_episodic": sum(r.episodic_memories for r in self._records),
            "total_semantic": sum(r.semantic_memories for r in self._records),
            "total_chunks": sum(r.chunks_generated for r in self._records),
            "error_count": sum(1 for r in self._records if r.error),
            "last_ingested_at": last.ingested_at,
            "storage_status_counts": status_counts,
        }

    def tombstone(self, file_hash: str, file_name: str) -> None:
        """
        Mark a file for re-ingestion by appending a tombstone record.

        After this call, check_file() and is_already_ingested() will
        return False for this hash, so the next pipeline run will process
        the file as if it were new.

        NOTE: Re-ingestion creates duplicate memories in ChromaDB. Use a
        separate purge command to remove old memories before re-ingesting
        if duplicates are undesired.

        Args:
            file_hash: SHA-256 hash of the file to tombstone.
            file_name: File name (for audit trail reference).
        """
        tombstone_record = IngestionRecord(
            file_path="",
            file_name=file_name,
            file_hash=file_hash,
            file_size_bytes=0,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            source_directory="",
            destination_path="",
            content_type="",
            chunks_generated=0,
            episodic_memories=0,
            semantic_memories=0,
            skipped_memories=0,
            processing_time_ms=0.0,
            hippocampal_connected=False,
            storage_status="reprocessing",
            pipeline_version="",
        )
        self.record_ingestion(tombstone_record)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load all records from the JSONL manifest into memory."""
        if not self._manifest_path.exists():
            return
        with open(self._manifest_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = IngestionRecord.from_dict(data)
                    self._records.append(record)
                    self._hash_index.setdefault(record.file_hash, []).append(record)
                except (json.JSONDecodeError, TypeError, KeyError):
                    # Corrupted line — skip silently; manifest remains usable
                    pass

    def _append_jsonl(self, record: "IngestionRecord") -> None:
        """
        Atomically append one JSON record to the JSONL manifest.

        Uses fcntl LOCK_EX advisory locking for concurrency safety on Linux.
        The write is a single fh.write() call, which is atomic at the OS
        level for lines shorter than PIPE_BUF (~4 KB on Linux).

        Args:
            record: IngestionRecord to append.
        """
        line = json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n"
        with open(self._manifest_path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# File mover
# ---------------------------------------------------------------------------

class FileMover:
    """
    Move processed files into the ~/ubik/Ingested_data/ archive structure.

    Creates the following layout on demand:
        {base_ingested_dir}/
        ├── {source_dir}_ingested/    <- created when first file arrives
        │   └── <filename>            <- moved file
        └── ingestion_log/            <- created on FileMover init

    Name collisions are resolved by appending _2, _3, ... before the extension.

    Example:
        mover = FileMover()
        dest = mover.move(
            Path("~/data/therapy/session_2026.transcript"),
            source_directory="therapy",
        )
        # -> ~/ubik/Ingested_data/therapy_ingested/session_2026.transcript
    """

    DEFAULT_BASE_DIR = Path("~/ubik/Ingested_data")

    def __init__(
        self,
        base_ingested_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize FileMover and create the log directory.

        Args:
            base_ingested_dir: Root archive directory.
                               Defaults to ~/ubik/Ingested_data.
            dry_run: If True, compute destination paths but do not move files.
        """
        if base_ingested_dir is None:
            base_ingested_dir = self.DEFAULT_BASE_DIR
        self.base_ingested_dir = Path(base_ingested_dir).expanduser().resolve()
        self.dry_run = dry_run

        if not dry_run:
            (self.base_ingested_dir / "ingestion_log").mkdir(parents=True, exist_ok=True)

    def compute_destination(self, source_path: Path, source_directory: str) -> Path:
        """
        Compute the destination path for a file without moving it.

        Appends _2, _3, … before the extension if the destination already
        exists. This method is idempotent: repeated calls with the same
        arguments return the same path (assuming no file is created between
        calls).

        Args:
            source_path: Original file path.
            source_directory: Parent folder name used to construct the
                              destination subdirectory (e.g. "therapy"
                              -> "therapy_ingested").

        Returns:
            Intended destination Path (may not yet exist).
        """
        dest_dir = self.base_ingested_dir / f"{source_directory}_ingested"
        dest_path = dest_dir / source_path.name
        if dest_path.exists():
            stem = source_path.stem
            suffix = source_path.suffix
            counter = 2
            while dest_path.exists():
                dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        return dest_path

    def move(self, source_path: Path, source_directory: str) -> Path:
        """
        Move a file into the ingested archive.

        Creates the destination directory if necessary. Verifies the move
        succeeded by checking source is gone and destination exists.

        In dry_run mode returns the computed destination without moving.

        Args:
            source_path: File to move (must exist).
            source_directory: Parent folder name (e.g. "therapy").

        Returns:
            Destination path where the file now lives.

        Raises:
            FileNotFoundError: If source_path does not exist.
            RuntimeError: If post-move verification fails.
        """
        source_path = Path(source_path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        dest_path = self.compute_destination(source_path, source_directory)

        if self.dry_run:
            return dest_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(dest_path))

        if not dest_path.exists():
            raise RuntimeError(
                f"Move verification failed: destination not found: {dest_path}"
            )
        if source_path.exists():
            raise RuntimeError(
                f"Move verification failed: source still exists: {source_path}"
            )

        return dest_path
