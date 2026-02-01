#!/usr/bin/env python3
"""
UBIK Ingestion System - Command Line Interface

Provides CLI access to the ingestion pipeline for processing
local files, directories, and Google Drive content.

Usage:
    # Process a directory
    python -m ingest.cli local ~/Documents/letters --recursive

    # Process from Google Drive
    python -m ingest.cli gdrive --folder-id ABC123 --credentials ~/creds.json

    # Process a single file
    python -m ingest.cli file ~/Documents/memoir.pdf

    # Dry run (preview without storing)
    python -m ingest.cli local ~/Documents --dry-run --verbose

Version: 0.1.0
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .chunkers import ChunkConfig
from .pipeline import IngestPipeline, PipelineConfig

__all__ = ['main', 'create_parser']

# Version
VERSION = "0.1.0"

# Default paths
DEFAULT_TOKEN_PATH = Path.home() / ".ubik" / "gdrive_token.json"


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser with all subcommands.

    Returns:
        Configured ArgumentParser
    """
    # Main parser
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
        help="Preview ingestion without storing to MCP",
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
        default="localhost",
        help="MCP server host (default: localhost)",
    )

    common.add_argument(
        "--mcp-port",
        type=int,
        default=8080,
        help="MCP server port (default: 8080)",
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
        default=4,
        help="Number of files to process in parallel (default: 4)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available ingestion commands",
        help="Use '%(prog)s <command> --help' for command-specific help",
    )

    # -------------------------------------------------------------------------
    # "local" subcommand - Process local directory
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
    # "gdrive" subcommand - Process Google Drive folder
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
    # "file" subcommand - Process single file
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

    return parser


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
    print()

    # Create pipeline config
    config = PipelineConfig(
        storage_mode=not args.dry_run,
        whisper_model=args.whisper_model,
        chunk_config=ChunkConfig(target_chunk_size=args.chunk_size),
        parallel_files=args.parallel,
    )

    # Run pipeline
    pipeline = IngestPipeline(config=config)

    try:
        result = await pipeline.ingest_directory(
            directory=path,
            recursive=args.recursive,
            extensions=extensions,
        )

        # Print summary
        print(result.summary())
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

    # Ensure token directory exists
    token_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Ingesting from Google Drive folder: {args.folder_id}")
    print(f"Credentials: {credentials_path}")
    print(f"Token: {token_path}")
    print(f"Recursive: {args.recursive}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Create pipeline config
    config = PipelineConfig(
        storage_mode=not args.dry_run,
        whisper_model=args.whisper_model,
        chunk_config=ChunkConfig(target_chunk_size=args.chunk_size),
        parallel_files=args.parallel,
    )

    # Run pipeline
    pipeline = IngestPipeline(config=config)

    try:
        result = await pipeline.ingest_from_gdrive(
            folder_id=args.folder_id,
            credentials_path=str(credentials_path),
            token_path=str(token_path),
            recursive=args.recursive,
        )

        # Print summary
        print(result.summary())
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
    print()

    # Create pipeline config
    config = PipelineConfig(
        storage_mode=not args.dry_run,
        whisper_model=args.whisper_model,
        chunk_config=ChunkConfig(target_chunk_size=args.chunk_size),
    )

    # Run pipeline
    pipeline = IngestPipeline(config=config)

    try:
        result = await pipeline.ingest_file(path)

        # Print result
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

        # Show memory details in verbose mode
        if args.verbose and result.memory_candidates:
            print("Memory Candidates:")
            for i, candidate in enumerate(result.memory_candidates):
                print(f"\n  [{i}] {candidate.memory_type.value} "
                      f"(confidence: {candidate.confidence:.2f})")
                print(f"      Category: {candidate.category}")
                print(f"      Themes: {', '.join(candidate.themes)}")
                print(f"      Importance: {candidate.importance:.2f}")

                # Preview content
                preview = candidate.content[:80].replace('\n', ' ')
                print(f"      Content: {preview}...")

        return 0 if result.success else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


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

    # Show help if no command specified
    if not args.command:
        parser.print_help()
        return 0

    # Setup logging
    setup_logging(
        verbose=getattr(args, 'verbose', False),
        debug=getattr(args, 'debug', False),
    )

    # Run appropriate command
    try:
        if args.command == "local":
            return asyncio.run(run_local(args))
        elif args.command == "gdrive":
            return asyncio.run(run_gdrive(args))
        elif args.command == "file":
            return asyncio.run(run_file(args))
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
