# UBIK Ingestion Pipeline Enhancement — Claude Code Instructions

**Task:** Enhance the ingestion CLI to track processed files, move them after ingestion, and maintain a searchable audit log.

**CLAUDE.md compliance:** Follow all project coding standards. No hardcoded values. Dataclass configs. Comprehensive error handling. Type hints everywhere. Google-style docstrings.

---

## Context

The UBIK ingestion pipeline lives at `~/ubik/somatic/ubik_ingest/`. It processes `.transcript`, `.txt`, `.md`, `.pdf`, `.docx`, and audio files into episodic and semantic memories stored on the Hippocampal Node (Mac Mini, ChromaDB + Neo4j) via MCP.

Currently the pipeline has no awareness of whether a file has already been ingested. There is no audit trail. Files sit in the same directory before and after processing. This task fixes all three problems.

---

## Architecture Overview

**BEFORE INGESTION:**
```
~/ubik/data/source_materials/
├── therapy/
│   ├── therapy_2026-02-16.transcript   ← waiting
│   └── therapy_2026-03-02.transcript   ← waiting
├── letters/
│   └── letter_to_sofia.md              ← waiting
└── journal/
    └── reflection_2024.txt             ← waiting
```

**AFTER INGESTION:**
```
~/ubik/data/source_materials/
├── therapy/                            ← empty (files moved)
├── letters/                            ← empty
└── journal/                            ← empty

~/ubik/Ingested_data/
├── therapy_ingested/
│   ├── therapy_2026-02-16.transcript   ← moved here
│   └── therapy_2026-03-02.transcript   ← moved here
├── letters_ingested/
│   └── letter_to_sofia.md              ← moved here
├── journal_ingested/
│   └── reflection_2024.txt             ← moved here
└── ingestion_log/
    ├── ingestion_manifest.jsonl        ← append-only log
    ├── ingestion_manifest.csv          ← human-readable mirror
    └── ingestion_errors.jsonl          ← error-only log
```

---

## Step 1: Create the Ingestion Tracker Module

**File:** `~/ubik/somatic/ubik_ingest/ingest/tracker.py`

This module handles file tracking, deduplication, and the move-after-ingest logic.

### Requirements

1. **`IngestionRecord` dataclass** with these fields:
   - `file_path: str` — original absolute path
   - `file_name: str` — basename
   - `file_hash: str` — SHA-256 of file content (for dedup)
   - `file_size_bytes: int`
   - `ingested_at: str` — ISO 8601 timestamp
   - `source_directory: str` — parent folder name (e.g., `"therapy"`)
   - `destination_path: str` — where file was moved to
   - `content_type: str` — detected type (transcript, letter, journal, etc.)
   - `chunks_generated: int`
   - `episodic_memories: int`
   - `semantic_memories: int`
   - `skipped_memories: int`
   - `processing_time_ms: float`
   - `hippocampal_connected: bool` — whether MCP storage was active
   - `storage_status: str` — `"stored"`, `"dry_run"`, `"failed"`, `"partial"`
   - `error: Optional[str]`
   - `pipeline_version: str` — from `ingest.__version__`
   - `memory_ids: List[str]` — IDs of stored ChromaDB documents (for traceability)

2. **`IngestionManifest` class** that:
   - Loads from `ingestion_manifest.jsonl` on init (append-only JSONL)
   - Provides `is_already_ingested(file_path: str) -> bool` using SHA-256 hash lookup
   - Provides `record_ingestion(record: IngestionRecord) -> None` that appends to both JSONL and CSV
   - Provides `get_records(filters: dict) -> List[IngestionRecord]` for querying
   - Provides `get_stats() -> dict` returning summary statistics
   - Handles the manifest file not existing yet (first run)
   - Uses file locking (`fcntl` on Linux) for safe concurrent writes

3. **`FileMover` class** that:
   - Takes `base_ingested_dir: Path` (defaults to `~/ubik/Ingested_data`)
   - Computes destination: `{base_ingested_dir}/{source_folder_name}_ingested/{filename}`
   - Creates destination directories as needed
   - Moves file using `shutil.move`
   - Verifies move succeeded (destination exists, source gone)
   - Returns the destination path
   - Has a `dry_run` mode that computes but doesn't execute the move
   - Handles name collisions by appending `_2`, `_3`, etc. before the extension

4. **`compute_file_hash(path: Path) -> str`** utility function:
   - SHA-256 hash
   - Reads in 8KB chunks (memory safe for large audio files)
   - Returns hex digest

### Key Design Decisions

- **JSONL format** for the manifest because it's append-only, survives partial writes, and is grep-searchable
- **CSV mirror** because it opens in Excel/Google Sheets for non-technical review
- **SHA-256 for deduplication** — if a file is modified and re-placed in the source directory, it will be re-ingested (new hash)
- **File locking** because future phases may run parallel ingestion

---

## Step 2: Create the Ingestion Log Writer

**File:** `~/ubik/somatic/ubik_ingest/ingest/log_writer.py`

Handles writing to the structured log files.

### Requirements

1. **`IngestionLogWriter` class** that manages three output files:
   - `ingestion_manifest.jsonl` — every ingestion record, one JSON object per line
   - `ingestion_manifest.csv` — same data in CSV format for human reading
   - `ingestion_errors.jsonl` — only records where `error` is not `None`

2. **JSONL format** — each line is a complete JSON object:
   ```json
   {"file_name":"therapy_2026-02-16.transcript","file_hash":"a1b2c3...","ingested_at":"2026-03-04T20:30:00Z","chunks_generated":12,"episodic_memories":12,"semantic_memories":0,"storage_status":"stored",...}
   ```

3. **CSV format** — header row on first write, then append data rows. Columns ordered for readability:
   ```
   ingested_at,file_name,content_type,chunks_generated,episodic_memories,semantic_memories,storage_status,processing_time_ms,source_directory,file_hash
   ```

4. **Error log** — same JSONL format but only entries with errors, for quick diagnosis.

5. Log rotation is **NOT needed for Phase 1** — UBIK won't generate enough volume. But document in a comment that rotation should be added if the manifest exceeds 10,000 records.

6. All writes must be **atomic** — write to temp file, then rename (prevents corruption on crash).

---

## Step 3: Integrate Tracker into the Pipeline

**File:** `~/ubik/somatic/ubik_ingest/ingest/pipeline.py` (modify existing)

### Changes to `IngestPipeline`

1. Add `tracker: Optional[IngestionManifest]` and `file_mover: Optional[FileMover]` to the constructor. Default to `None` (backward compatible).

2. In `ingest_directory()` and `ingest_file()`:
   - Before processing, check `tracker.is_already_ingested(file_path)` using file hash
   - If already ingested, log a skip message and continue to next file
   - After successful processing, build an `IngestionRecord` from the `IngestResult`
   - Call `tracker.record_ingestion(record)`
   - Call `file_mover.move(source_path)` to relocate the processed file
   - If move fails, log error but don't fail the overall batch (the memory is already stored)

3. Add a `skipped_already_ingested: int` counter to `BatchIngestResult` and include it in the summary output.

4. The pipeline must still work **WITHOUT a tracker** (backward compatible for tests and dry runs).

---

## Step 4: Update the CLI

**File:** `~/ubik/somatic/ubik_ingest/ingest/cli.py` (modify existing)

### New CLI Flags

Add to the common argument group:
```
--track / --no-track     Enable/disable file tracking and move-on-ingest (default: --track)
--ingested-dir PATH      Directory for ingested files (default: ~/ubik/Ingested_data)
--log-dir PATH           Directory for ingestion logs (default: ~/ubik/Ingested_data/ingestion_log)
```

### New Subcommand: `status`

Add a `status` subcommand:
```bash
python -m ingest.cli status                    # Show overall ingestion stats
python -m ingest.cli status --recent 10        # Show last 10 ingestions
python -m ingest.cli status --errors           # Show only errors
python -m ingest.cli status --search "therapy" # Search by filename
```

Implementation:
- Reads from the JSONL manifest
- Formats output as a clean table using simple string formatting (no external deps)
- `--recent N` shows the last N records sorted by timestamp
- `--errors` filters to records with non-null `error` field
- `--search TERM` filters by filename substring match

### Changes to `run_local` and `run_file`

1. If `--track` is enabled (default), create `IngestionManifest` and `FileMover` instances
2. Pass them to `IngestPipeline`
3. Print a summary line at the end: `"Tracked: X new, Y skipped (already ingested), Z moved"`

### Changes to `run_local` when `--dry-run`

In dry run mode, tracking is automatically disabled (no files moved, no manifest written). But still print what WOULD happen:
```
[DRY RUN] Would ingest: therapy_2026-02-16.transcript (new file)
[DRY RUN] Would skip: therapy_2026-03-02.transcript (already ingested: 2026-03-04T20:30:00Z)
[DRY RUN] Would move 1 file to ~/ubik/Ingested_data/therapy_ingested/
```

---

## Step 5: Create the Ingested Data Directory Structure

**File:** `~/ubik/somatic/ubik_ingest/ingest/tracker.py` (part of `FileMover`)

On first run, `FileMover` should create:
```
~/ubik/Ingested_data/
└── ingestion_log/
    ├── ingestion_manifest.jsonl
    ├── ingestion_manifest.csv
    └── ingestion_errors.jsonl
```

Source subdirectory mirrors are created on demand as files are moved:
```
~/ubik/Ingested_data/therapy_ingested/       ← created when first therapy file is moved
~/ubik/Ingested_data/letters_ingested/       ← created when first letter file is moved
```

The `_ingested` suffix makes it immediately obvious these are processed files, not source files.

---

## Step 6: Add a Recovery/Reprocess Command

### New Subcommand: `reingest`

```bash
python -m ingest.cli reingest --file therapy_2026-02-16.transcript
python -m ingest.cli reingest --source-dir therapy
python -m ingest.cli reingest --all
```

This command:
1. Finds the file in `~/ubik/Ingested_data/` by name or source directory
2. Copies it back to the original source location
3. Removes its hash from the manifest (marks as "reprocessing")
4. The next `local` run will re-ingest it naturally

**Use case:** You updated the chunking algorithm or prompt templates and want to re-ingest specific content.

> **Important:** `reingest` does NOT delete memories from ChromaDB. That's a separate concern for a future `purge` command. Re-ingestion will create duplicate memories — document this limitation clearly in the help text.

---

## Step 7: Add Integrity Verification

### New Subcommand: `verify`

```bash
python -m ingest.cli verify                    # Verify all tracked files
python -m ingest.cli verify --source-dir therapy  # Verify specific source
```

This command:
1. Reads the manifest
2. For each record, checks that the file exists at `destination_path`
3. Recomputes SHA-256 and compares to stored hash
4. Reports: total checked, OK, missing, corrupted
5. Outputs a simple report

This is critical for the project's long-term survivability — it catches disk corruption, accidental deletions, or filesystem issues before they become data loss.

---

## Step 8: Write Tests

**File:** `~/ubik/somatic/ubik_ingest/tests/test_tracker.py`

### Test Cases

1. `test_compute_file_hash` — verify SHA-256 produces consistent results
2. `test_manifest_first_run` — manifest file doesn't exist, should create cleanly
3. `test_manifest_record_and_query` — write record, read it back
4. `test_manifest_dedup` — same file content detected as already ingested
5. `test_manifest_modified_file` — same filename but different content gets new hash, re-ingested
6. `test_file_mover_basic` — file moves to correct destination
7. `test_file_mover_name_collision` — duplicate filename gets `_2` suffix
8. `test_file_mover_dry_run` — computes destination but doesn't move
9. `test_csv_mirror` — CSV written with correct headers and data
10. `test_pipeline_integration_with_tracker` — full pipeline run with tracker, file moves, log written

Use `pytest` with `tmp_path` fixture for all file operations.

---

## Step 9: Update Package Exports

**File:** `~/ubik/somatic/ubik_ingest/ingest/__init__.py`

Add to exports:
```python
from .tracker import IngestionManifest, IngestionRecord, FileMover, compute_file_hash
from .log_writer import IngestionLogWriter
```

Update `__all__` and bump `__version__` to `"1.1.0"`.

---

## Step 10: Documentation

### Update `README.md`

Add a section called **"File Tracking & Audit Trail"** explaining:
- How tracking works (hash-based dedup, move-on-ingest)
- Directory structure after ingestion
- How to check status (`status` subcommand)
- How to re-ingest files (`reingest` subcommand)
- How to verify integrity (`verify` subcommand)
- Log file formats and locations

### Add Inline Documentation

Every new function gets a Google-style docstring explaining:
- What it does
- Args with types
- Returns with types
- Raises with conditions
- Example usage where helpful

---

## Verification Checklist

After implementation, verify:

- [ ] `python -m ingest.cli local ~/ubik/data/source_materials/therapy_test/ --verbose` processes files AND moves them
- [ ] Running the same command again skips already-ingested files
- [ ] `~/ubik/Ingested_data/therapy_test_ingested/` contains the moved files
- [ ] `~/ubik/Ingested_data/ingestion_log/ingestion_manifest.jsonl` has correct records
- [ ] `~/ubik/Ingested_data/ingestion_log/ingestion_manifest.csv` opens cleanly in a spreadsheet
- [ ] `python -m ingest.cli status` shows the ingestion summary
- [ ] `python -m ingest.cli status --recent 5` shows last 5 records
- [ ] `python -m ingest.cli status --errors` shows nothing (no errors)
- [ ] `python -m ingest.cli verify` reports all files OK
- [ ] `python -m ingest.cli local ... --dry-run` does NOT move files or write logs
- [ ] `python -m ingest.cli local ... --no-track` processes without tracking (backward compatible)
- [ ] `python -m pytest tests/test_tracker.py -v` all tests pass
- [ ] `grep -r "100\." ~/ubik/somatic/ubik_ingest/` returns nothing (no hardcoded IPs)
- [ ] All new files have module docstrings and type hints
- [ ] `ruff check ~/ubik/somatic/ubik_ingest/` reports no errors

---

## Important Notes for Claude Code

1. **Do not break existing functionality.** The pipeline must still work with `--no-track` and in dry-run mode exactly as before.

2. **Environment awareness.** The Somatic Node is Linux (WSL2, Ubuntu). Use `pathlib.Path` everywhere. The ingested data directory uses `~/ubik/Ingested_data/` — note the capital `I` matching the existing UBIK directory naming convention.

3. **File paths.** The project location is `~/ubik/somatic/ubik_ingest/`. The `mcp_client` package is at `~/ubik/somatic/mcp_client/`. The virtual environment is `source /home/gasu/pytorch_env/bin/activate`.

4. **Hippocampal Node connection.** MCP server at `100.103.242.91:8080` (from env var `HIPPOCAMPAL_HOST`). ChromaDB on port 8001. The CLI was recently patched to auto-create the MCP client from environment variables.

5. **The `cli.py` was recently updated.** Use the version at `~/ubik/somatic/ubik_ingest/ingest/cli.py` as the base — it includes the `create_mcp_client()` helper function added in the latest patch.

6. **One chunk per file bug.** There is a known issue where each transcript produces only 1 chunk instead of many. This is a separate bug — do not attempt to fix it in this task. The tracker should correctly record whatever the pipeline outputs.

7. **CLAUDE.md standards apply.** Resilience first. Config via dataclass. Explicit exports. Health checks. No hardcoded values.
