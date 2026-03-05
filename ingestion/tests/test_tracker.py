"""
UBIK Ingestion Tracker - Unit Tests

Tests for tracker.py:
    - compute_file_hash
    - IngestionManifest (first run, record/query, dedup, modified file, tombstone)
    - FileMover (basic move, name collision, dry run)
    - IngestionLogWriter (CSV mirror)
    - Pipeline integration with tracker

Run with: pytest tests/test_tracker.py -v
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ingest.tracker import (
    FileMover,
    IngestionManifest,
    IngestionRecord,
    compute_file_hash,
)
from ingest.log_writer import IngestionLogWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    file_path: str = "/tmp/test.txt",
    file_name: str = "test.txt",
    file_hash: str = "abc123",
    source_directory: str = "therapy",
    storage_status: str = "stored",
    destination_path: str = "",
    error: str = None,
) -> IngestionRecord:
    return IngestionRecord(
        file_path=file_path,
        file_name=file_name,
        file_hash=file_hash,
        file_size_bytes=100,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        source_directory=source_directory,
        destination_path=destination_path,
        content_type="text",
        chunks_generated=3,
        episodic_memories=2,
        semantic_memories=1,
        skipped_memories=0,
        processing_time_ms=42.0,
        hippocampal_connected=False,
        storage_status=storage_status,
        pipeline_version="1.1.0",
        error=error,
        memory_ids=["mem_001", "mem_002"],
    )


# ---------------------------------------------------------------------------
# 1. compute_file_hash
# ---------------------------------------------------------------------------

def test_compute_file_hash(tmp_path: Path) -> None:
    """SHA-256 produces consistent, deterministic results."""
    f = tmp_path / "sample.txt"
    f.write_text("Hello, UBIK!", encoding="utf-8")

    h1 = compute_file_hash(f)
    h2 = compute_file_hash(f)

    assert h1 == h2, "Hash must be deterministic"
    assert len(h1) == 64, "SHA-256 hex digest must be 64 chars"
    assert h1.islower(), "Hash should be lowercase hex"


def test_compute_file_hash_different_content(tmp_path: Path) -> None:
    """Different content produces different hashes."""
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("Content A", encoding="utf-8")
    f2.write_text("Content B", encoding="utf-8")

    assert compute_file_hash(f1) != compute_file_hash(f2)


# ---------------------------------------------------------------------------
# 2. IngestionManifest — first run (no manifest file)
# ---------------------------------------------------------------------------

def test_manifest_first_run(tmp_path: Path) -> None:
    """Manifest initializes cleanly when no JSONL file exists."""
    manifest = IngestionManifest(log_dir=tmp_path / "log")

    assert len(manifest.get_records()) == 0
    stats = manifest.get_stats()
    assert stats["total_records"] == 0
    assert stats["unique_files"] == 0


def test_manifest_first_run_creates_log_dir(tmp_path: Path) -> None:
    """Log directory is created automatically on first use."""
    log_dir = tmp_path / "deep" / "nested" / "log"
    assert not log_dir.exists()

    IngestionManifest(log_dir=log_dir)
    assert log_dir.is_dir()


# ---------------------------------------------------------------------------
# 3. IngestionManifest — record and query
# ---------------------------------------------------------------------------

def test_manifest_record_and_query(tmp_path: Path) -> None:
    """Write a record and read it back via get_records."""
    log_dir = tmp_path / "log"
    manifest = IngestionManifest(log_dir=log_dir)

    record = _make_record(file_name="therapy_2026.transcript", file_hash="hash001")
    manifest.record_ingestion(record)

    records = manifest.get_records()
    assert len(records) == 1
    assert records[0].file_name == "therapy_2026.transcript"
    assert records[0].file_hash == "hash001"
    assert records[0].episodic_memories == 2


def test_manifest_record_persists_to_disk(tmp_path: Path) -> None:
    """Record written to JSONL survives a fresh manifest load."""
    log_dir = tmp_path / "log"
    m1 = IngestionManifest(log_dir=log_dir)
    m1.record_ingestion(_make_record(file_hash="persist_hash"))

    # Create a fresh manifest from the same directory
    m2 = IngestionManifest(log_dir=log_dir)
    records = m2.get_records()
    assert len(records) == 1
    assert records[0].file_hash == "persist_hash"


def test_manifest_get_records_filter(tmp_path: Path) -> None:
    """Filters narrow results by substring match on string fields."""
    log_dir = tmp_path / "log"
    manifest = IngestionManifest(log_dir=log_dir)

    manifest.record_ingestion(_make_record(file_name="therapy_a.transcript",
                                           source_directory="therapy",
                                           file_hash="h1"))
    manifest.record_ingestion(_make_record(file_name="letter_b.md",
                                           source_directory="letters",
                                           file_hash="h2"))

    therapy = manifest.get_records({"source_directory": "therapy"})
    assert len(therapy) == 1
    assert therapy[0].file_name == "therapy_a.transcript"


# ---------------------------------------------------------------------------
# 4. IngestionManifest — deduplication
# ---------------------------------------------------------------------------

def test_manifest_dedup(tmp_path: Path) -> None:
    """Same file content (same hash) is detected as already ingested."""
    f = tmp_path / "file.txt"
    f.write_text("Identical content", encoding="utf-8")
    file_hash = compute_file_hash(f)

    log_dir = tmp_path / "log"
    manifest = IngestionManifest(log_dir=log_dir)
    manifest.record_ingestion(_make_record(file_hash=file_hash))

    _hash, already = manifest.check_file(f)
    assert already is True
    assert _hash == file_hash


def test_manifest_dedup_new_file(tmp_path: Path) -> None:
    """A file whose hash is not in the manifest is NOT already ingested."""
    f = tmp_path / "new_file.txt"
    f.write_text("Brand new content", encoding="utf-8")

    log_dir = tmp_path / "log"
    manifest = IngestionManifest(log_dir=log_dir)
    manifest.record_ingestion(_make_record(file_hash="totally_different_hash"))

    _hash, already = manifest.check_file(f)
    assert already is False


# ---------------------------------------------------------------------------
# 5. IngestionManifest — modified file gets new hash, re-ingested
# ---------------------------------------------------------------------------

def test_manifest_modified_file(tmp_path: Path) -> None:
    """
    Same filename but different content produces a new hash and is
    treated as a new file (not skipped).
    """
    f = tmp_path / "journal.txt"
    f.write_text("Original content", encoding="utf-8")
    original_hash = compute_file_hash(f)

    log_dir = tmp_path / "log"
    manifest = IngestionManifest(log_dir=log_dir)
    manifest.record_ingestion(_make_record(file_hash=original_hash))

    # Simulate modifying the file
    f.write_text("Modified content — different hash now", encoding="utf-8")

    new_hash, already = manifest.check_file(f)
    assert already is False
    assert new_hash != original_hash


# ---------------------------------------------------------------------------
# 6. FileMover — basic move
# ---------------------------------------------------------------------------

def test_file_mover_basic(tmp_path: Path) -> None:
    """File moves to the correct destination directory."""
    src_dir = tmp_path / "therapy"
    src_dir.mkdir()
    src_file = src_dir / "session.transcript"
    src_file.write_text("Session content")

    ingested_base = tmp_path / "Ingested_data"
    mover = FileMover(base_ingested_dir=ingested_base)

    dest = mover.move(src_file, source_directory="therapy")

    assert dest == ingested_base / "therapy_ingested" / "session.transcript"
    assert dest.exists()
    assert not src_file.exists()
    assert dest.read_text() == "Session content"


# ---------------------------------------------------------------------------
# 7. FileMover — name collision appends _2, _3, ...
# ---------------------------------------------------------------------------

def test_file_mover_name_collision(tmp_path: Path) -> None:
    """Duplicate filename at destination gets _2 suffix."""
    ingested_base = tmp_path / "Ingested_data"
    dest_dir = ingested_base / "therapy_ingested"
    dest_dir.mkdir(parents=True)

    # Pre-create the collision file
    (dest_dir / "session.transcript").write_text("Existing file")

    src_dir = tmp_path / "therapy"
    src_dir.mkdir()
    new_file = src_dir / "session.transcript"
    new_file.write_text("New session content")

    mover = FileMover(base_ingested_dir=ingested_base)
    dest = mover.move(new_file, source_directory="therapy")

    assert dest.name == "session_2.transcript"
    assert dest.exists()
    # Original collision file must still be intact
    assert (dest_dir / "session.transcript").read_text() == "Existing file"


# ---------------------------------------------------------------------------
# 8. FileMover — dry run
# ---------------------------------------------------------------------------

def test_file_mover_dry_run(tmp_path: Path) -> None:
    """Dry run computes destination but does not move or create directories."""
    src_dir = tmp_path / "letters"
    src_dir.mkdir()
    src_file = src_dir / "letter.md"
    src_file.write_text("Dear Sofia...")

    ingested_base = tmp_path / "Ingested_data"
    mover = FileMover(base_ingested_dir=ingested_base, dry_run=True)

    dest = mover.compute_destination(src_file, source_directory="letters")

    assert dest == ingested_base / "letters_ingested" / "letter.md"
    assert src_file.exists(), "Source must not be moved in dry run"
    assert not dest.exists(), "Destination must not be created in dry run"


# ---------------------------------------------------------------------------
# 9. CSV mirror
# ---------------------------------------------------------------------------

def test_csv_mirror(tmp_path: Path) -> None:
    """CSV file is created with correct headers and data row."""
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    writer = IngestionLogWriter(log_dir=log_dir)

    record = _make_record(
        file_name="memoir.pdf",
        file_hash="csvhash",
        source_directory="documents",
    )
    writer.write_record(record)

    csv_path = log_dir / IngestionLogWriter.CSV_FILE
    assert csv_path.exists()

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["file_name"] == "memoir.pdf"
    assert rows[0]["source_directory"] == "documents"
    assert rows[0]["storage_status"] == "stored"


def test_csv_mirror_multiple_rows(tmp_path: Path) -> None:
    """Subsequent writes append rows without duplicating the header."""
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    writer = IngestionLogWriter(log_dir=log_dir)

    for i in range(3):
        writer.write_record(_make_record(
            file_name=f"file_{i}.txt",
            file_hash=f"hash_{i}",
        ))

    csv_path = log_dir / IngestionLogWriter.CSV_FILE
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) == 3


def test_error_log_written_for_failed_record(tmp_path: Path) -> None:
    """Records with errors are written to ingestion_errors.jsonl."""
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    writer = IngestionLogWriter(log_dir=log_dir)

    ok_record = _make_record(file_hash="ok_hash")
    fail_record = _make_record(
        file_hash="fail_hash",
        storage_status="failed",
        error="Connection refused",
    )

    writer.write_record(ok_record)
    writer.write_record(fail_record)

    errors_path = log_dir / IngestionLogWriter.ERRORS_FILE
    assert errors_path.exists()
    lines = [l for l in errors_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["file_hash"] == "fail_hash"
    assert data["error"] == "Connection refused"


# ---------------------------------------------------------------------------
# 10. Pipeline integration with tracker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_integration_with_tracker(tmp_path: Path) -> None:
    """
    Full pipeline run with tracker: file is processed, manifest is written,
    and file is moved to the archive.
    """
    from ingest.pipeline import IngestPipeline, PipelineConfig
    from ingest.chunkers import ChunkConfig

    # Create a test file
    src_dir = tmp_path / "journal"
    src_dir.mkdir()
    test_file = src_dir / "reflection.txt"
    test_file.write_text(
        "Today I thought about what matters most. Family is everything. "
        "I believe that love and connection define a meaningful life. "
        "I want my grandchildren to remember these values. "
        "This is a genuine reflection on life and purpose."
    )

    log_dir = tmp_path / "Ingested_data" / "ingestion_log"
    ingested_base = tmp_path / "Ingested_data"

    manifest = IngestionManifest(log_dir=log_dir)
    mover = FileMover(base_ingested_dir=ingested_base)

    config = PipelineConfig(
        storage_mode=False,  # dry run — no MCP connection needed
        chunk_config=ChunkConfig(target_chunk_size=200),
    )

    async with IngestPipeline(
        mcp_client=None,
        config=config,
        tracker=manifest,
        file_mover=mover,
    ) as pipeline:
        result = await pipeline.ingest_file(test_file)

    # File was processed
    assert result.success
    assert not result.skipped_duplicate

    # Manifest was written
    records = manifest.get_records()
    assert len(records) == 1
    assert records[0].file_name == "reflection.txt"
    assert records[0].storage_status == "dry_run"

    # File was moved
    expected_dest = ingested_base / "journal_ingested" / "reflection.txt"
    assert expected_dest.exists()
    assert not test_file.exists()
    assert records[0].destination_path == str(expected_dest)


@pytest.mark.asyncio
async def test_pipeline_skips_already_ingested(tmp_path: Path) -> None:
    """
    Second pipeline run on the same file returns skipped_duplicate result.
    """
    from ingest.pipeline import IngestPipeline, PipelineConfig

    src_dir = tmp_path / "letters"
    src_dir.mkdir()
    test_file = src_dir / "letter.txt"
    test_file.write_text("Dear future self, remember what matters.")

    file_hash = compute_file_hash(test_file)
    log_dir = tmp_path / "log"

    manifest = IngestionManifest(log_dir=log_dir)
    # Pre-record as already ingested
    manifest.record_ingestion(_make_record(
        file_path=str(test_file),
        file_name="letter.txt",
        file_hash=file_hash,
        storage_status="stored",
    ))

    config = PipelineConfig(storage_mode=False)
    async with IngestPipeline(
        mcp_client=None,
        config=config,
        tracker=manifest,
        file_mover=None,
    ) as pipeline:
        result = await pipeline.ingest_file(test_file)

    assert result.skipped_duplicate is True
    assert result.source_file == "letter.txt"
    # File must NOT have been moved (file_mover=None, and we skipped anyway)
    assert test_file.exists()


@pytest.mark.asyncio
async def test_pipeline_works_without_tracker(tmp_path: Path) -> None:
    """
    Pipeline without tracker is fully backward compatible.
    """
    from ingest.pipeline import IngestPipeline, PipelineConfig

    src_dir = tmp_path / "docs"
    src_dir.mkdir()
    test_file = src_dir / "note.txt"
    test_file.write_text("A simple note for testing backward compatibility.")

    config = PipelineConfig(storage_mode=False)
    async with IngestPipeline(
        mcp_client=None,
        config=config,
        tracker=None,
        file_mover=None,
    ) as pipeline:
        result = await pipeline.ingest_file(test_file)

    assert result.success
    assert not result.skipped_duplicate
    # File must still be in place (no mover)
    assert test_file.exists()
