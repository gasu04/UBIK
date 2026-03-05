#!/usr/bin/env python3
"""
UBIK Ingestion System - Command Line Interface

Provides CLI access to the ingestion pipeline for processing
local files, directories, and Google Drive content.

Usage:
    # Process a directory (tracking enabled by default)
    python -m ingest.cli local ~/Documents/letters --recursive

    # Process from Google Drive
    python -m ingest.cli gdrive --folder-id ABC123 --credentials ~/creds.json

    # Process a single file
    python -m ingest.cli file ~/Documents/memoir.pdf

    # Dry run (preview without storing or moving)
    python -m ingest.cli local ~/Documents --dry-run --verbose

    # Overwrite: delete existing records before ingesting
    python -m ingest.cli local ~/Documents/therapy --overwrite

    # Check ingestion status
    python -m ingest.cli status --recent 10
    python -m ingest.cli status --errors
    python -m ingest.cli status --search therapy

    # Queue files for re-ingestion
    python -m ingest.cli reingest --file therapy_2026-02-16.transcript
    python -m ingest.cli reingest --source-dir therapy
    python -m ingest.cli reingest --all

    # Verify ingested file integrity
    python -m ingest.cli verify
    python -m ingest.cli verify --source-dir therapy

Version: 0.4.0
"""

import argparse
import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .chunkers import ChunkConfig
from .pipeline import IngestPipeline, PipelineConfig
from .tracker import FileMover, IngestionManifest, compute_file_hash

__all__ = ['main', 'create_parser']

# Version
VERSION = "0.4.0"

# Default paths
DEFAULT_TOKEN_PATH = Path.home() / ".ubik" / "gdrive_token.json"
DEFAULT_INGESTED_DIR = Path.home() / "ubik" / "Ingested_data"
DEFAULT_LOG_DIR = DEFAULT_INGESTED_DIR / "ingestion_log"

logger = logging.getLogger("ubik.ingest")


# =============================================================================
# Argument parser
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser with all subcommands.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="ubik-ingest",
        description="UBIK Content Ingestion System - Process files into memory storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s local ~/Documents/letters --recursive
  %(prog)s gdrive --folder-id ABC123 --credentials creds.json
  %(prog)s file ~/Documents/memoir.pdf --dry-run
  %(prog)s local ./transcripts --extensions .transcript,.txt
  %(prog)s local ./therapy --mcp-host localhost
  %(prog)s status --recent 10
  %(prog)s reingest --file therapy_2026-02-16.transcript
  %(prog)s verify

For more information, visit: https://github.com/ubik-project
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    # Common arguments (applied to all subcommands)
    common = argparse.ArgumentParser(add_help=False)

    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview ingestion without storing to MCP (disables tracking and move)",
    )

    common.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing ChromaDB records for these source files before ingesting",
    )

    common.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for audio transcription (default: base)",
    )

    common.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    common.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    common.add_argument(
        "--mcp-host",
        type=str,
        default=None,
        help="MCP server host (default: from HIPPOCAMPAL_HOST env var)",
    )

    common.add_argument(
        "--mcp-port",
        type=int,
        default=None,
        help="MCP server port (default: from HIPPOCAMPAL_PORT env var, or 8080)",
    )

    common.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in characters (default: 500)",
    )

    common.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of files to process in parallel (default: 1)",
    )

    # Tracking flags
    common.add_argument(
        "--track",
        dest="track",
        action="store_true",
        default=True,
        help="Enable file tracking and move-on-ingest (default: enabled)",
    )

    common.add_argument(
        "--no-track",
        dest="track",
        action="store_false",
        help="Disable file tracking (backward-compatible mode, no files moved)",
    )

    common.add_argument(
        "--ingested-dir",
        type=str,
        default=str(DEFAULT_INGESTED_DIR),
        help=f"Directory for ingested files (default: {DEFAULT_INGESTED_DIR})",
    )

    common.add_argument(
        "--log-dir",
        type=str,
        default=str(DEFAULT_LOG_DIR),
        help=f"Directory for ingestion logs (default: {DEFAULT_LOG_DIR})",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available ingestion commands",
        help="Use '%(prog)s <command> --help' for command-specific help",
    )

    # -------------------------------------------------------------------------
    # "local" subcommand
    # -------------------------------------------------------------------------
    local_parser = subparsers.add_parser(
        "local",
        parents=[common],
        help="Ingest files from a local directory",
        description="Process all supported files in a local directory",
    )

    local_parser.add_argument(
        "path",
        type=str,
        help="Path to directory to process",
    )

    local_parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        dest="recursive",
        help="Recursively process subdirectories (default)",
    )

    local_parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Only process files in the top-level directory",
    )

    local_parser.add_argument(
        "--extensions", "-e",
        type=str,
        default=None,
        help="Comma-separated list of extensions to process (e.g., '.pdf,.docx,.txt')",
    )

    # -------------------------------------------------------------------------
    # "gdrive" subcommand
    # -------------------------------------------------------------------------
    gdrive_parser = subparsers.add_parser(
        "gdrive",
        parents=[common],
        help="Ingest files from Google Drive",
        description="Download and process files from a Google Drive folder",
    )

    gdrive_parser.add_argument(
        "--folder-id", "-f",
        type=str,
        required=True,
        help="Google Drive folder ID to process",
    )

    gdrive_parser.add_argument(
        "--credentials", "-c",
        type=str,
        required=True,
        help="Path to Google OAuth credentials JSON file",
    )

    gdrive_parser.add_argument(
        "--token", "-t",
        type=str,
        default=str(DEFAULT_TOKEN_PATH),
        help=f"Path to store/load OAuth token (default: {DEFAULT_TOKEN_PATH})",
    )

    gdrive_parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        dest="recursive",
        help="Recursively process subfolders (default)",
    )

    gdrive_parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Only process files in the specified folder",
    )

    # -------------------------------------------------------------------------
    # "file" subcommand
    # -------------------------------------------------------------------------
    file_parser = subparsers.add_parser(
        "file",
        parents=[common],
        help="Ingest a single file",
        description="Process a single file through the ingestion pipeline",
    )

    file_parser.add_argument(
        "path",
        type=str,
        help="Path to file to process",
    )

    # -------------------------------------------------------------------------
    # "status" subcommand
    # -------------------------------------------------------------------------
    status_parser = subparsers.add_parser(
        "status",
        help="Show ingestion status and audit log",
        description="Display ingestion statistics and recent activity",
    )

    status_parser.add_argument(
        "--recent",
        type=int,
        default=None,
        metavar="N",
        help="Show last N ingestion records",
    )

    status_parser.add_argument(
        "--errors",
        action="store_true",
        help="Show only records with errors",
    )

    status_parser.add_argument(
        "--search",
        type=str,
        default=None,
        metavar="TERM",
        help="Filter records by filename substring",
    )

    status_parser.add_argument(
        "--log-dir",
        type=str,
        default=str(DEFAULT_LOG_DIR),
        help=f"Ingestion log directory (default: {DEFAULT_LOG_DIR})",
    )

    # -------------------------------------------------------------------------
    # "reingest" subcommand
    # -------------------------------------------------------------------------
    reingest_parser = subparsers.add_parser(
        "reingest",
        help="Queue ingested files for re-processing",
        description=(
            "Mark ingested files for re-processing on the next pipeline run.\n"
            "\n"
            "WARNING: Re-ingestion creates DUPLICATE memories in ChromaDB.\n"
            "Use a separate purge command to remove old memories if duplicates\n"
            "are undesired."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    reingest_group = reingest_parser.add_mutually_exclusive_group(required=True)
    reingest_group.add_argument(
        "--file",
        type=str,
        metavar="FILENAME",
        help="Re-ingest a specific file by name (basename)",
    )

    reingest_group.add_argument(
        "--source-dir",
        type=str,
        metavar="DIRNAME",
        help="Re-ingest all files from a specific source directory",
    )

    reingest_group.add_argument(
        "--all",
        action="store_true",
        help="Re-ingest all previously ingested files",
    )

    reingest_parser.add_argument(
        "--ingested-dir",
        type=str,
        default=str(DEFAULT_INGESTED_DIR),
        help=f"Base ingested data directory (default: {DEFAULT_INGESTED_DIR})",
    )

    reingest_parser.add_argument(
        "--log-dir",
        type=str,
        default=str(DEFAULT_LOG_DIR),
        help=f"Ingestion log directory (default: {DEFAULT_LOG_DIR})",
    )

    reingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be re-queued without making any changes",
    )

    # -------------------------------------------------------------------------
    # "verify" subcommand
    # -------------------------------------------------------------------------
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify integrity of ingested files",
        description=(
            "Verify that ingested files exist at their recorded destinations\n"
            "and that their SHA-256 hashes match. Catches disk corruption,\n"
            "accidental deletions, and filesystem issues."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    verify_parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        metavar="DIRNAME",
        help="Verify only files from a specific source directory",
    )

    verify_parser.add_argument(
        "--log-dir",
        type=str,
        default=str(DEFAULT_LOG_DIR),
        help=f"Ingestion log directory (default: {DEFAULT_LOG_DIR})",
    )

    return parser


# =============================================================================
# Logging
# =============================================================================

def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Configure logging based on verbosity settings.

    Args:
        verbose: Enable INFO level logging
        debug: Enable DEBUG level logging
    """
    if debug:
        level = logging.DEBUG
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    elif verbose:
        level = logging.INFO
        fmt = "%(levelname)s: %(message)s"
    else:
        level = logging.WARNING
        fmt = "%(message)s"

    logging.basicConfig(
        level=level,
        format=fmt,
        stream=sys.stderr,
    )


# =============================================================================
# Shared helpers
# =============================================================================

def parse_extensions(extensions_str: Optional[str]) -> Optional[List[str]]:
    """
    Parse comma-separated extensions string.

    Args:
        extensions_str: String like ".pdf,.docx,.txt"

    Returns:
        List of extensions or None
    """
    if not extensions_str:
        return None
    extensions = []
    for ext in extensions_str.split(","):
        ext = ext.strip()
        if ext:
            if not ext.startswith("."):
                ext = f".{ext}"
            extensions.append(ext.lower())
    return extensions if extensions else None


async def create_mcp_client(args: argparse.Namespace):
    """
    Create and connect a memory writer for Hippocampal Node storage.

    Resolution order:
      1. If --mcp-host is given (or HIPPOCAMPAL_HOST env var is set): use
         HippocampalClient (remote MCP over HTTP)
      2. Otherwise: use LocalMemoryWriter (direct ChromaDB + Neo4j on this node)

    Returns:
        Connected writer, or None if dry-run or connection fails
    """
    if getattr(args, 'dry_run', False):
        logger.info("Dry run mode - skipping storage connection")
        return None

    host = args.mcp_host
    port = args.mcp_port

    if host is None:
        host = os.environ.get("HIPPOCAMPAL_HOST")
    if port is None:
        port_str = os.environ.get("HIPPOCAMPAL_MCP_PORT") or os.environ.get("HIPPOCAMPAL_PORT")
        if port_str:
            port = int(port_str)

    # --- Local path: no explicit remote host → use LocalMemoryWriter ---
    if host is None:
        try:
            from .mcp_writer import LocalMemoryWriter

            writer = LocalMemoryWriter()
            logger.info("Using LocalMemoryWriter (direct ChromaDB + Neo4j)")
            return writer
        except Exception as e:
            logger.warning(f"LocalMemoryWriter unavailable: {e}. Running without storage.")
            return None

    # --- Remote path: explicit host → use HippocampalClient over MCP ---
    kwargs = {"host": host}
    if port is not None:
        kwargs["port"] = port

    try:
        mcp_client_dir = "/Volumes/990PRO 4T/UBIK/somatic/mcp_client"
        if mcp_client_dir not in sys.path:
            sys.path.insert(0, mcp_client_dir)

        from hippocampal_client import HippocampalClient

        client = HippocampalClient(**kwargs)
        resolved_port = getattr(client, 'port', port or '?')
        logger.info(f"Using HippocampalClient for {host}:{resolved_port}")
        return client

    except ImportError:
        logger.warning(
            "mcp_client package not found at /Volumes/990PRO 4T/UBIK/somatic/mcp_client/. "
            "Running without storage."
        )
        return None
    except Exception as e:
        display_port = port or os.environ.get("HIPPOCAMPAL_PORT", "8080")
        logger.warning(
            f"Could not create HippocampalClient for {host}:{display_port}: {e}\n"
            "Running without storage."
        )
        return None


async def cleanup_mcp_client(client) -> None:
    """Safely disconnect MCP client."""
    if client is None:
        return
    try:
        if hasattr(client, 'disconnect'):
            await client.disconnect()
        elif hasattr(client, 'close'):
            await client.close()
    except Exception as e:
        logger.debug(f"MCP client cleanup error (non-fatal): {e}")


def _load_maestro_env() -> None:
    """Load maestro .env into os.environ (values already set are not overwritten)."""
    env_path = Path.home() / "ubik" / "maestro" / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def _purge_source_files(filenames: List[str]) -> Dict[str, int]:
    """
    Delete all ChromaDB records for the given source filenames.

    Connects directly to ChromaDB on the hippocampal node using the same
    env vars as the MCP server (HIPPOCAMPAL_TAILSCALE_IP, CHROMADB_PORT,
    CHROMADB_TOKEN).

    Both metadata key names are handled:
      - "source_file" (episodic collection)
      - "source"      (semantic collection)

    Args:
        filenames: List of source filenames (basename only) to purge

    Returns:
        Dict mapping collection name to number of records deleted
    """
    try:
        import chromadb
    except ImportError:
        logger.warning("chromadb not installed — cannot purge. Skipping --overwrite.")
        return {}

    _load_maestro_env()
    host = os.environ.get("CHROMADB_HOST", os.environ.get("HIPPOCAMPAL_TAILSCALE_IP", "localhost"))
    port = int(os.environ.get("CHROMADB_PORT", "8001"))
    token = os.environ.get("CHROMADB_TOKEN", "")

    settings = chromadb.Settings(anonymized_telemetry=False)
    if token:
        settings = chromadb.Settings(
            anonymized_telemetry=False,
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=token,
        )

    try:
        client = chromadb.HttpClient(host=host, port=port, settings=settings)
        client.heartbeat()
    except Exception as e:
        logger.warning(f"Cannot connect to ChromaDB at {host}:{port} for purge: {e}")
        return {}

    filenames_set = set(filenames)
    deleted: Dict[str, int] = {}

    for coll_name in ("ubik_episodic", "ubik_semantic"):
        try:
            collection = client.get_collection(coll_name)
            result = collection.get(include=["metadatas"])
            ids_to_delete = [
                id_
                for id_, meta in zip(result["ids"], result["metadatas"])
                if (meta.get("source_file") or meta.get("source", "")) in filenames_set
            ]
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
            deleted[coll_name] = len(ids_to_delete)
        except Exception as e:
            logger.warning(f"Could not purge {coll_name}: {e}")
            deleted[coll_name] = 0

    return deleted


def _build_tracker_and_mover(
    args: argparse.Namespace,
) -> tuple:
    """
    Build IngestionManifest and FileMover from CLI args.

    Tracking is automatically disabled in dry-run mode.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Tuple of (IngestionManifest or None, FileMover or None).
    """
    if not getattr(args, 'track', True) or getattr(args, 'dry_run', False):
        return None, None

    log_dir = Path(args.log_dir).expanduser().resolve()
    ingested_dir = Path(args.ingested_dir).expanduser().resolve()

    manifest = IngestionManifest(log_dir=log_dir)
    mover = FileMover(base_ingested_dir=ingested_dir)
    return manifest, mover


def _print_dry_run_tracking_preview(
    path: Path,
    extensions: Optional[List[str]],
    recursive: bool,
    log_dir: Path,
    ingested_dir: Path,
    pipeline: "IngestPipeline",
) -> None:
    """
    Print a dry-run preview of what tracking would do.

    Shows which files would be ingested vs skipped (already ingested),
    and where they would be moved.

    Args:
        path: Source directory.
        extensions: Optional extension filter.
        recursive: Whether to recurse.
        log_dir: Manifest log directory.
        ingested_dir: Base archive directory.
        pipeline: IngestPipeline instance (for supported_extensions check).
    """
    manifest_path = log_dir / IngestionManifest.MANIFEST_FILE
    if not manifest_path.exists():
        # No manifest yet — all files are new
        manifest = None
    else:
        manifest = IngestionManifest(log_dir=log_dir)

    mover = FileMover(base_ingested_dir=ingested_dir, dry_run=True)

    files = pipeline._collect_files(path, recursive, extensions)
    new_files: List[Path] = []
    skipped_files: List[tuple] = []

    for f in files:
        if manifest is None:
            new_files.append(f)
            continue
        try:
            file_hash, already_ingested = manifest.check_file(f)
            if already_ingested:
                last = manifest.get_last_record_for_hash(file_hash)
                ts = last.ingested_at if last else "unknown"
                skipped_files.append((f, ts))
            else:
                new_files.append(f)
        except Exception:
            new_files.append(f)

    for f in new_files:
        print(f"[DRY RUN] Would ingest: {f.name} (new file)")

    for f, ts in skipped_files:
        print(f"[DRY RUN] Would skip:   {f.name} (already ingested: {ts})")

    if new_files:
        # Group by source_directory
        by_dir: Dict[str, List[Path]] = {}
        for f in new_files:
            by_dir.setdefault(f.parent.name, []).append(f)
        for src_dir, dir_files in sorted(by_dir.items()):
            dest = mover.base_ingested_dir / f"{src_dir}_ingested"
            print(f"[DRY RUN] Would move {len(dir_files)} file(s) to {dest}/")

    print()


# =============================================================================
# Command runners
# =============================================================================

async def run_local(args: argparse.Namespace) -> int:
    """
    Run local directory ingestion.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    path = Path(args.path).expanduser().resolve()

    if not path.exists():
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        return 1

    if not path.is_dir():
        print(f"Error: Path is not a directory: {path}", file=sys.stderr)
        print("Use 'file' command for single files.", file=sys.stderr)
        return 1

    extensions = parse_extensions(args.extensions)

    print(f"Ingesting directory: {path}")
    if extensions:
        print(f"Filtering extensions: {extensions}")
    print(f"Recursive: {args.recursive}")
    print(f"Dry run: {args.dry_run}")
    if not args.dry_run:
        print(f"Tracking: {args.track}")
    print()

    # Purge existing records if --overwrite
    if getattr(args, 'overwrite', False) and not args.dry_run:
        glob = "**/*" if args.recursive else "*"
        candidates = [
            f.name for f in path.glob(glob)
            if f.is_file() and (
                extensions is None or f.suffix.lower() in extensions
            )
        ]
        if candidates:
            print(f"Overwrite: purging existing records for {len(candidates)} file(s)...")
            deleted = _purge_source_files(candidates)
            for coll, n in deleted.items():
                if n:
                    print(f"  Deleted {n} records from {coll}")
            print()

    # Create MCP client for storage
    mcp_client = await create_mcp_client(args)

    # Create pipeline config
    config = PipelineConfig(
        storage_mode=not args.dry_run and mcp_client is not None,
        whisper_model=args.whisper_model,
        chunk_config=ChunkConfig(target_chunk_size=args.chunk_size),
        parallel_files=args.parallel,
    )

    # Create tracker and mover (None in dry-run mode)
    tracker, file_mover = _build_tracker_and_mover(args)

    # Run pipeline
    try:
        async with IngestPipeline(
            mcp_client=mcp_client,
            config=config,
            tracker=tracker,
            file_mover=file_mover,
        ) as pipeline:
            # Dry-run tracking preview (before running pipeline)
            if args.dry_run and args.track:
                log_dir = Path(args.log_dir).expanduser().resolve()
                ingested_dir = Path(args.ingested_dir).expanduser().resolve()
                _print_dry_run_tracking_preview(
                    path=path,
                    extensions=extensions,
                    recursive=args.recursive,
                    log_dir=log_dir,
                    ingested_dir=ingested_dir,
                    pipeline=pipeline,
                )

            result = await pipeline.ingest_directory(
                directory=path,
                recursive=args.recursive,
                extensions=extensions,
            )

        # Print summary
        print(result.summary())
        print()

        # Tracking summary line
        if tracker is not None:
            moved = result.successful  # files successfully processed were moved
            print(
                f"Tracked: {result.successful} new, "
                f"{result.skipped_already_ingested} skipped (already ingested), "
                f"{moved} moved"
            )
            print()

        # Print errors if any
        if result.errors:
            print("Errors:")
            for filename, error in result.errors.items():
                print(f"  {filename}: {error}")
            print()

        return 0 if result.failed == 0 else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


async def run_gdrive(args: argparse.Namespace) -> int:
    """
    Run Google Drive ingestion.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    credentials_path = Path(args.credentials).expanduser().resolve()
    token_path = Path(args.token).expanduser().resolve()

    if not credentials_path.exists():
        print(f"Error: Credentials file not found: {credentials_path}", file=sys.stderr)
        return 1

    token_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Ingesting from Google Drive folder: {args.folder_id}")
    print(f"Credentials: {credentials_path}")
    print(f"Token: {token_path}")
    print(f"Recursive: {args.recursive}")
    print(f"Dry run: {args.dry_run}")
    print()

    mcp_client = await create_mcp_client(args)

    config = PipelineConfig(
        storage_mode=not args.dry_run and mcp_client is not None,
        whisper_model=args.whisper_model,
        chunk_config=ChunkConfig(target_chunk_size=args.chunk_size),
        parallel_files=args.parallel,
    )

    try:
        async with IngestPipeline(mcp_client=mcp_client, config=config) as pipeline:
            result = await pipeline.ingest_from_gdrive(
                folder_id=args.folder_id,
                credentials_path=str(credentials_path),
                token_path=str(token_path),
                recursive=args.recursive,
            )

        print(result.summary())
        print()

        if result.errors:
            print("Errors:")
            for filename, error in result.errors.items():
                print(f"  {filename}: {error}")
            print()

        return 0 if result.failed == 0 else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


async def run_file(args: argparse.Namespace) -> int:
    """
    Run single file ingestion.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    path = Path(args.path).expanduser().resolve()

    if not path.exists():
        print(f"Error: File does not exist: {path}", file=sys.stderr)
        return 1

    if not path.is_file():
        print(f"Error: Path is not a file: {path}", file=sys.stderr)
        print("Use 'local' command for directories.", file=sys.stderr)
        return 1

    print(f"Ingesting file: {path}")
    print(f"Dry run: {args.dry_run}")
    if not args.dry_run:
        print(f"Tracking: {args.track}")
    print()

    # Purge existing records if --overwrite
    if getattr(args, 'overwrite', False) and not args.dry_run:
        print(f"Overwrite: purging existing records for '{path.name}'...")
        deleted = _purge_source_files([path.name])
        for coll, n in deleted.items():
            if n:
                print(f"  Deleted {n} records from {coll}")
        print()

    mcp_client = await create_mcp_client(args)

    config = PipelineConfig(
        storage_mode=not args.dry_run and mcp_client is not None,
        whisper_model=args.whisper_model,
        chunk_config=ChunkConfig(target_chunk_size=args.chunk_size),
    )

    tracker, file_mover = _build_tracker_and_mover(args)

    try:
        async with IngestPipeline(
            mcp_client=mcp_client,
            config=config,
            tracker=tracker,
            file_mover=file_mover,
        ) as pipeline:
            result = await pipeline.ingest_file(path)

        if result.skipped_duplicate:
            print(f"Skipped: {result.source_file} (already ingested)")
            return 0

        print(f"Source: {result.source_file}")
        print(f"Success: {result.success}")
        print(f"Processing time: {result.processing_time_ms:.1f}ms")
        print(f"Chunks generated: {result.chunks_generated}")
        print(f"Episodic memories: {result.episodic_count}")
        print(f"Semantic memories: {result.semantic_count}")
        print(f"Skipped: {result.skipped_count}")
        print()

        if result.error:
            print(f"Error: {result.error}", file=sys.stderr)
            return 1

        if args.verbose and result.memory_candidates:
            print("Memory Candidates:")
            for i, candidate in enumerate(result.memory_candidates):
                print(f"\n  [{i}] {candidate.memory_type.value} "
                      f"(confidence: {candidate.confidence:.2f})")
                print(f"      Category: {candidate.category}")
                print(f"      Themes: {', '.join(candidate.themes)}")
                print(f"      Importance: {candidate.importance:.2f}")
                preview = candidate.content[:80].replace('\n', ' ')
                print(f"      Content: {preview}...")

        return 0 if result.success else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def run_status(args: argparse.Namespace) -> int:
    """
    Show ingestion status and audit log entries.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 if manifest not found)
    """
    log_dir = Path(args.log_dir).expanduser().resolve()
    manifest_path = log_dir / IngestionManifest.MANIFEST_FILE

    if not manifest_path.exists():
        print("No ingestion manifest found.")
        print(f"Expected at: {manifest_path}")
        print("Run 'ubik-ingest local <path>' to start ingesting files.")
        return 0

    manifest = IngestionManifest(log_dir=log_dir)

    # Build filters
    filters = {}
    if args.errors:
        # errors filter: check error field is not None/empty
        # We filter manually below since get_records does substring match
        pass
    if args.search:
        filters["file_name"] = args.search

    records = manifest.get_records(filters=filters or None)

    # Apply errors filter manually
    if args.errors:
        records = [r for r in records if r.error]

    # Apply recent limit
    if args.recent is not None:
        records = records[:args.recent]

    if not records:
        # Show stats even if no filtered records
        stats = manifest.get_stats()
        print(f"No records match the filter. Total records: {stats['total_records']}")
        return 0

    # Print stats header
    stats = manifest.get_stats()
    print("=" * 72)
    print("UBIK Ingestion Status")
    print("=" * 72)
    print(f"Total records:    {stats['total_records']}")
    print(f"Unique files:     {stats['unique_files']}")
    print(f"Total episodic:   {stats['total_episodic']}")
    print(f"Total semantic:   {stats['total_semantic']}")
    print(f"Errors:           {stats['error_count']}")
    if stats['last_ingested_at']:
        print(f"Last ingested:    {stats['last_ingested_at']}")
    print()

    # Print record table
    col_w = {"time": 20, "name": 30, "type": 12, "ep": 6, "sem": 6, "status": 12}
    header = (
        f"{'Ingested At':<{col_w['time']}}  "
        f"{'File':<{col_w['name']}}  "
        f"{'Type':<{col_w['type']}}  "
        f"{'Ep':>{col_w['ep']}}  "
        f"{'Sem':>{col_w['sem']}}  "
        f"{'Status':<{col_w['status']}}"
    )
    print(header)
    print("-" * len(header))

    for r in records:
        ts = r.ingested_at[:19].replace("T", " ")  # trim microseconds
        name = r.file_name[:col_w['name']] if len(r.file_name) > col_w['name'] \
            else r.file_name
        ctype = r.content_type[:col_w['type']]
        status_str = r.storage_status
        if r.error:
            status_str += " !"
        print(
            f"{ts:<{col_w['time']}}  "
            f"{name:<{col_w['name']}}  "
            f"{ctype:<{col_w['type']}}  "
            f"{r.episodic_memories:>{col_w['ep']}}  "
            f"{r.semantic_memories:>{col_w['sem']}}  "
            f"{status_str:<{col_w['status']}}"
        )
        if r.error:
            print(f"  ERROR: {r.error}")

    print()
    return 0


def run_reingest(args: argparse.Namespace) -> int:
    """
    Queue ingested files for re-processing.

    Copies files back to their original source location and appends
    tombstone records to the manifest so they are picked up on the
    next pipeline run.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    log_dir = Path(args.log_dir).expanduser().resolve()
    manifest_path = log_dir / IngestionManifest.MANIFEST_FILE

    if not manifest_path.exists():
        print("No ingestion manifest found. Nothing to re-ingest.")
        return 0

    manifest = IngestionManifest(log_dir=log_dir)

    # Determine which records to reingest
    if args.file:
        records_to_process = manifest.get_records({"file_name": args.file})
        # Exact match (get_records uses substring)
        records_to_process = [r for r in records_to_process
                               if r.file_name == args.file
                               and r.storage_status != "reprocessing"]
    elif args.source_dir:
        records_to_process = manifest.get_records({"source_directory": args.source_dir})
        records_to_process = [r for r in records_to_process
                               if r.storage_status != "reprocessing"]
    else:  # --all
        records_to_process = manifest.get_records()
        records_to_process = [r for r in records_to_process
                               if r.storage_status != "reprocessing"]

    if not records_to_process:
        print("No matching records found.")
        return 0

    # Deduplicate by hash — take most recent record per hash
    seen_hashes: Dict[str, object] = {}
    for r in records_to_process:
        if r.file_hash not in seen_hashes:
            seen_hashes[r.file_hash] = r

    unique_records = list(seen_hashes.values())

    print(f"Found {len(unique_records)} file(s) to queue for re-ingestion.")
    print()
    print(
        "WARNING: Re-ingestion creates DUPLICATE memories in ChromaDB.\n"
        "         Use a purge command to remove old memories if needed.\n"
    )

    if args.dry_run:
        for r in unique_records:
            dest = r.destination_path or "(not moved)"
            orig = Path(r.file_path).parent if r.file_path else Path("?")
            print(f"[DRY RUN] Would copy: {dest}")
            print(f"          -> {orig}/{r.file_name}")
        return 0

    queued = 0
    errors = 0

    for r in unique_records:
        if not r.destination_path:
            print(f"  SKIP: {r.file_name} — no destination_path recorded")
            continue

        dest_path = Path(r.destination_path)
        if not dest_path.exists():
            print(f"  SKIP: {r.file_name} — file not found at {dest_path}")
            errors += 1
            continue

        orig_dir = Path(r.file_path).parent if r.file_path else None
        if orig_dir is None or not orig_dir.exists():
            # Try to reconstruct from source_directory and ubik data path
            ubik_data = Path.home() / "ubik" / "data" / "source_materials"
            orig_dir = ubik_data / r.source_directory
            orig_dir.mkdir(parents=True, exist_ok=True)

        target = orig_dir / r.file_name
        try:
            shutil.copy2(str(dest_path), str(target))
            manifest.tombstone(r.file_hash, r.file_name)
            print(f"  Queued: {r.file_name} -> {target}")
            queued += 1
        except Exception as e:
            print(f"  ERROR: {r.file_name}: {e}", file=sys.stderr)
            errors += 1

    print()
    print(f"Queued {queued} file(s) for re-ingestion. {errors} error(s).")
    if queued:
        print("Run 'ubik-ingest local <source_dir>' to re-ingest.")
    return 0 if errors == 0 else 1


def run_verify(args: argparse.Namespace) -> int:
    """
    Verify integrity of all tracked ingested files.

    Checks that each file exists at its recorded destination and that
    its SHA-256 hash matches the stored value. Reports any discrepancies.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 if all OK, 1 if any issues found)
    """
    log_dir = Path(args.log_dir).expanduser().resolve()
    manifest_path = log_dir / IngestionManifest.MANIFEST_FILE

    if not manifest_path.exists():
        print("No ingestion manifest found.")
        return 0

    manifest = IngestionManifest(log_dir=log_dir)

    filters = {}
    if args.source_dir:
        filters["source_directory"] = args.source_dir

    all_records = manifest.get_records(filters=filters or None)
    # Only verify successfully-ingested records
    records = [r for r in all_records if r.storage_status not in ("dry_run", "reprocessing")
               and r.destination_path]

    if not records:
        print("No ingested files to verify.")
        return 0

    print(f"Verifying {len(records)} ingested file(s)...")
    print()

    ok = 0
    missing = 0
    corrupted = 0

    for r in records:
        dest = Path(r.destination_path)
        if not dest.exists():
            print(f"  MISSING:   {r.file_name}")
            print(f"             Expected at: {dest}")
            missing += 1
            continue

        try:
            actual_hash = compute_file_hash(dest)
            if actual_hash != r.file_hash:
                print(f"  CORRUPTED: {r.file_name}")
                print(f"             Stored hash:  {r.file_hash}")
                print(f"             Actual hash:  {actual_hash}")
                corrupted += 1
            else:
                ok += 1
        except Exception as e:
            print(f"  ERROR:     {r.file_name}: {e}")
            corrupted += 1

    print()
    print("=" * 40)
    print(f"Verification complete")
    print(f"  OK:        {ok}")
    print(f"  Missing:   {missing}")
    print(f"  Corrupted: {corrupted}")
    print(f"  Total:     {len(records)}")

    return 0 if (missing == 0 and corrupted == 0) else 1


# =============================================================================
# Entry point
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command-line arguments (uses sys.argv if None)

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    setup_logging(
        verbose=getattr(args, 'verbose', False),
        debug=getattr(args, 'debug', False),
    )

    try:
        if args.command == "local":
            return asyncio.run(run_local(args))
        elif args.command == "gdrive":
            return asyncio.run(run_gdrive(args))
        elif args.command == "file":
            return asyncio.run(run_file(args))
        elif args.command == "status":
            return run_status(args)
        elif args.command == "reingest":
            return run_reingest(args)
        elif args.command == "verify":
            return run_verify(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
