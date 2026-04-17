"""
UBIK Ingestion Log Writer - Structured Audit Log Output

Manages three output files for the ingestion audit trail:
    ingestion_manifest.jsonl  - append-only, every ingestion event
    ingestion_manifest.csv    - human-readable mirror (opens in Excel/Sheets)
    ingestion_errors.jsonl    - error-only events for quick diagnosis

All writes are protected by fcntl advisory locking.
The CSV header row is written atomically (temp file + rename) on first use.
Subsequent rows are appended in-place under lock.

Note:
    Log rotation is not implemented in Phase 1. If the manifest grows
    beyond ~10,000 records, consider gzip-compressing old records and
    archiving them to a dated subdirectory.

Usage:
    from ingest.log_writer import IngestionLogWriter
    from ingest.tracker import IngestionRecord

    writer = IngestionLogWriter(log_dir=Path("~/ubik/Ingested_data/ingestion_log"))
    writer.write_record(record)

Version: 1.0.0
"""

import csv
import fcntl
import json
import os
from pathlib import Path
from typing import List

from .tracker import IngestionRecord

__all__ = ["IngestionLogWriter"]

# CSV column order — optimized for human readability
_CSV_COLUMNS: List[str] = [
    "ingested_at",
    "file_name",
    "content_type",
    "chunks_generated",
    "episodic_memories",
    "semantic_memories",
    "skipped_memories",
    "storage_status",
    "processing_time_ms",
    "source_directory",
    "destination_path",
    "file_size_bytes",
    "hippocampal_connected",
    "pipeline_version",
    "file_hash",
    "file_path",
    "error",
    "memory_ids",
]


class IngestionLogWriter:
    """
    Write IngestionRecord objects to CSV and error JSONL files.

    This class handles only writes; the JSONL manifest is written directly
    by IngestionManifest (which also owns the in-memory index).

    Thread/process safety: fcntl LOCK_EX is used for all file operations.

    Attributes:
        log_dir: Directory where log files are written.
        csv_path: Path to ingestion_manifest.csv.
        errors_path: Path to ingestion_errors.jsonl.

    Example:
        writer = IngestionLogWriter(log_dir=Path("~/ubik/Ingested_data/ingestion_log"))
        writer.write_record(record)
    """

    CSV_FILE = "ingestion_manifest.csv"
    ERRORS_FILE = "ingestion_errors.jsonl"

    def __init__(self, log_dir: Path) -> None:
        """
        Initialize IngestionLogWriter.

        Args:
            log_dir: Directory for log files. Created if it does not exist.
        """
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / self.CSV_FILE
        self.errors_path = self.log_dir / self.ERRORS_FILE

    def write_record(self, record: IngestionRecord) -> None:
        """
        Write a record to the CSV mirror and, if it has an error, to
        the error JSONL.

        Args:
            record: IngestionRecord to persist.
        """
        self._write_csv_row(record)
        if record.error:
            self._append_error_jsonl(record)

    # ------------------------------------------------------------------
    # CSV mirror
    # ------------------------------------------------------------------

    def _write_csv_row(self, record: IngestionRecord) -> None:
        """
        Append one row to the CSV mirror.

        Writes the header row atomically (temp + rename) on first use,
        then appends subsequent rows under fcntl LOCK_EX.

        Args:
            record: IngestionRecord to write.
        """
        row = record.to_dict()
        values = [str(row.get(col, "")) for col in _CSV_COLUMNS]

        needs_header = not self.csv_path.exists()

        if needs_header:
            self._write_csv_header_atomic(values)
        else:
            self._append_csv_row_locked(values)

    def _write_csv_header_atomic(self, first_row_values: List[str]) -> None:
        """
        Write header + first data row to a new CSV file atomically.

        Uses a temp file + os.rename() to prevent partial writes.

        Args:
            first_row_values: Values for the first data row.
        """
        tmp_path = self.csv_path.with_suffix(".csv.tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(_CSV_COLUMNS)
            writer.writerow(first_row_values)
        os.replace(str(tmp_path), str(self.csv_path))

    def _append_csv_row_locked(self, values: List[str]) -> None:
        """
        Append one data row to the existing CSV under fcntl lock.

        Args:
            values: Row values in _CSV_COLUMNS order.
        """
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                writer = csv.writer(fh)
                writer.writerow(values)
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    # ------------------------------------------------------------------
    # Error JSONL
    # ------------------------------------------------------------------

    def _append_error_jsonl(self, record: IngestionRecord) -> None:
        """
        Append an error record to ingestion_errors.jsonl.

        Only called when record.error is not None.

        Args:
            record: Failed IngestionRecord to log.
        """
        line = json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n"
        with open(self.errors_path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
