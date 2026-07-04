#!/usr/bin/env python3
"""
UBIK Ingestion System - Source Acquisition

Pulls raw source files into ``sources/<type>/`` from Google Drive
(OAuth desktop flow) or from a local directory (``--local``), replacing
the old Apps Script fetch. Read-only with respect to the originals:
files are copied byte-identical, never renamed (except on name
collision, see Note) and never altered.

Idempotent: every fetched file's SHA-256 is recorded in
``sources/MANIFEST.jsonl``; a file whose content hash is already in the
manifest is skipped, so re-running a fetch is always safe.

How it fits in: this is stage 0 of the pipeline. Files landed here are
picked up by enrichment (Phase 3+) and, after Gate 1, by IngestPipeline.

Usage:
    # Local mode (manual downloads) — same manifest behavior as Drive
    python fetch_sources.py --local ~/Downloads/tactiq_batch --type tactiq
    python fetch_sources.py --local ~/Downloads/x --type letters --dry-run

    # Google Drive mode (requires OAuth credentials, see .env.example)
    python fetch_sources.py --gdrive                 # all configured folders
    python fetch_sources.py --gdrive --type tactiq   # one source type

Configuration (env / .env — see .env.example):
    GDRIVE_CREDENTIALS_PATH   OAuth client secrets JSON (never committed)
    GDRIVE_TOKEN_PATH         cached user token (never committed)
    UBIK_GDRIVE_FOLDER_<TYPE> Drive folder ID per source type, e.g.
                              UBIK_GDRIVE_FOLDER_TACTIQ=1aB2c...

Dependencies:
    stdlib only for --local mode.
    google-api-python-client + google-auth-oauthlib for --gdrive mode
    (imported lazily; a clear error explains what to install).

Tier classification: Tier 2 (standard, 80% coverage). Failures are loud
(missing dirs, bad types, and copy errors raise; nothing is deleted),
and every action is recoverable — sources can always be re-fetched.

Version: 0.1.0
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.ingestion_config import PathsConfig, load_config
from ingest.registry import UbikIngestionError, load_content_types

__all__ = [
    'FetchError',
    'SourceManifest',
    'ManifestRecord',
    'fetch_local',
    'fetch_gdrive',
    'main',
]

logger = logging.getLogger("ubik.fetch_sources")

# Valid --type values = the source buckets under sources/.
SOURCE_TYPES = (
    "tactiq", "gemini", "fireflies", "letters", "memory_notes", "constitution",
)

_HASH_CHUNK_BYTES = 1024 * 1024


class FetchError(UbikIngestionError):
    """A source-acquisition operation failed."""


def compute_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's content."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# Manifest
# =============================================================================

@dataclass(frozen=True)
class ManifestRecord:
    """
    One fetched file, as recorded in sources/MANIFEST.jsonl.

    Attributes:
        sha256: Content hash — the idempotency key.
        source: Acquisition channel ("local" or "gdrive").
        source_type: Bucket under sources/ (tactiq, letters, ...).
        original_name: Filename as provided by the source.
        dest_path: Where the copy landed, relative to the ingestion root.
        size_bytes: File size.
        fetched_at: ISO-8601 UTC timestamp of acquisition.
        drive_id: Google Drive file ID (None for local fetches).
    """
    sha256: str
    source: str
    source_type: str
    original_name: str
    dest_path: str
    size_bytes: int
    fetched_at: str
    drive_id: Optional[str] = None

    def to_json(self) -> str:
        """Serialise to a single JSONL line."""
        return json.dumps({
            "sha256": self.sha256,
            "source": self.source,
            "source_type": self.source_type,
            "original_name": self.original_name,
            "dest_path": self.dest_path,
            "size_bytes": self.size_bytes,
            "fetched_at": self.fetched_at,
            "drive_id": self.drive_id,
        }, ensure_ascii=False)


class SourceManifest:
    """
    Append-only JSONL manifest of fetched source files.

    Loads existing hashes on construction; ``has(sha)`` answers the
    idempotency question, ``append(record)`` durably records a fetch.

    Example:
        >>> manifest = SourceManifest(paths.sources_dir / "MANIFEST.jsonl")
        >>> if not manifest.has(file_hash):
        ...     manifest.append(record)

    Note:
        Malformed manifest lines raise FetchError rather than being
        skipped — a corrupted manifest must be repaired, not ignored,
        or dedup silently stops working.
    """

    def __init__(self, path: Path):
        """Load the manifest at *path* (missing file = empty manifest)."""
        self.path = path
        self._hashes: Set[str] = set()
        self._drive_ids: Set[str] = set()
        if path.exists():
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    self._hashes.add(rec["sha256"])
                    if rec.get("drive_id"):
                        self._drive_ids.add(rec["drive_id"])
                except (json.JSONDecodeError, KeyError) as e:
                    raise FetchError(
                        f"Corrupt manifest line {lineno} in {path}: {e}"
                    ) from e

    def __len__(self) -> int:
        return len(self._hashes)

    def has(self, sha256: str) -> bool:
        """True if this content hash was already fetched."""
        return sha256 in self._hashes

    def has_drive_id(self, drive_id: str) -> bool:
        """True if this Drive file ID was already fetched (pre-download check)."""
        return drive_id in self._drive_ids

    def append(self, record: ManifestRecord) -> None:
        """Append a record and update the in-memory index."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")
        self._hashes.add(record.sha256)
        if record.drive_id:
            self._drive_ids.add(record.drive_id)


# =============================================================================
# Destination handling
# =============================================================================

def _dest_for(dest_dir: Path, original_name: str, sha256: str) -> Path:
    """
    Choose a destination path, preserving the original filename.

    On a name collision with different content, suffixes the stem with
    the first 8 hash characters (the original is never overwritten).
    """
    dest = dest_dir / original_name
    if not dest.exists():
        return dest
    suffixed = dest_dir / f"{dest.stem}__{sha256[:8]}{dest.suffix}"
    return suffixed


def _copy_one(
    src: Path,
    source_type: str,
    paths: PathsConfig,
    manifest: SourceManifest,
    dry_run: bool,
) -> Optional[ManifestRecord]:
    """
    Copy one local file into sources/<type>/ if its content is new.

    Returns:
        The appended ManifestRecord, or None if skipped (duplicate).

    Raises:
        FetchError: If the copied bytes do not hash-verify against the
            source (e.g., disk error) — the bad copy is removed.
    """
    sha = compute_sha256(src)
    if manifest.has(sha):
        logger.info("skip (already fetched): %s", src.name)
        return None

    dest_dir = paths.sources_dir / source_type
    dest = _dest_for(dest_dir, src.name, sha)

    if dry_run:
        print(f"[DRY RUN] would fetch: {src.name} -> {dest}")
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    if compute_sha256(dest) != sha:
        dest.unlink(missing_ok=True)
        raise FetchError(f"Copy verification failed for {src} -> {dest}")

    record = ManifestRecord(
        sha256=sha,
        source="local",
        source_type=source_type,
        original_name=src.name,
        dest_path=str(dest.relative_to(paths.root)),
        size_bytes=dest.stat().st_size,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        drive_id=None,
    )
    manifest.append(record)
    logger.info("fetched: %s -> %s", src.name, record.dest_path)
    return record


# =============================================================================
# Local mode
# =============================================================================

def fetch_local(
    source_dir: Path,
    source_type: str,
    paths: PathsConfig,
    dry_run: bool = False,
) -> List[ManifestRecord]:
    """
    Fetch manually-downloaded files from a local directory.

    Args:
        source_dir: Directory containing the files to ingest (top level
            only; not recursive — drops are expected to be flat).
        source_type: Bucket under sources/ (one of SOURCE_TYPES).
        paths: Resolved ingestion paths.
        dry_run: Preview without copying or writing the manifest.

    Returns:
        Records for newly fetched files (empty if everything was a dup).

    Raises:
        FetchError: On unknown source_type or missing/non-directory input.

    Example:
        >>> fetch_local(Path("~/Downloads/batch"), "tactiq", paths)
    """
    if source_type not in SOURCE_TYPES:
        raise FetchError(
            f"Unknown source type {source_type!r}; valid: {list(SOURCE_TYPES)}"
        )
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FetchError(f"--local path is not a directory: {source_dir}")

    manifest = SourceManifest(paths.sources_dir / "MANIFEST.jsonl")
    fetched: List[ManifestRecord] = []
    skipped = 0

    files = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    if not files:
        print(f"No files found in {source_dir}")
        return fetched

    for src in files:
        record = _copy_one(src, source_type, paths, manifest, dry_run)
        if record:
            fetched.append(record)
        else:
            skipped += 1

    print(
        f"{'[DRY RUN] ' if dry_run else ''}"
        f"{len(fetched)} fetched, {skipped} skipped "
        f"(duplicates or dry-run) -> sources/{source_type}/"
    )
    return fetched


# =============================================================================
# Google Drive mode
# =============================================================================

# Export formats for Google-native files (raw download is impossible for
# these; the export is recorded in the manifest via the .ext on dest_path).
_GOOGLE_EXPORT = {
    "application/vnd.google-apps.document": ("text/plain", ".txt"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
}


def _drive_service(credentials_path: Path, token_path: Path):
    """
    Build an authenticated Drive v3 service (OAuth desktop flow).

    Raises:
        FetchError: If the google client libraries are not installed or
            the credentials file is missing.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as e:
        raise FetchError(
            "Google Drive mode requires: pip install "
            "google-api-python-client google-auth-oauthlib"
        ) from e

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FetchError(
                    f"OAuth credentials not found: {credentials_path}\n"
                    "Download a Desktop-app OAuth client JSON from Google "
                    "Cloud Console and set GDRIVE_CREDENTIALS_PATH."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), scopes
            )
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return build("drive", "v3", credentials=creds)


def fetch_gdrive(
    folder_ids: Dict[str, str],
    paths: PathsConfig,
    credentials_path: Path,
    token_path: Path,
    dry_run: bool = False,
) -> List[ManifestRecord]:
    """
    Fetch new files from configured Google Drive folders.

    Args:
        folder_ids: Mapping of source_type -> Drive folder ID.
        paths: Resolved ingestion paths.
        credentials_path: OAuth client secrets JSON.
        token_path: Cached user token path.
        dry_run: List what would be downloaded without downloading.

    Returns:
        Records for newly fetched files.

    Raises:
        FetchError: On missing client libraries/credentials or download
            verification failure.

    Note:
        Google-native files (Docs/Sheets) cannot be downloaded raw; they
        are exported per _GOOGLE_EXPORT and the export extension appears
        on dest_path. Binary files are downloaded byte-identical.
    """
    import io

    service = _drive_service(credentials_path, token_path)
    manifest = SourceManifest(paths.sources_dir / "MANIFEST.jsonl")
    fetched: List[ManifestRecord] = []
    skipped = 0

    from googleapiclient.http import MediaIoBaseDownload

    for source_type, folder_id in folder_ids.items():
        query = f"'{folder_id}' in parents and trashed = false"
        page_token = None
        while True:
            resp = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageToken=page_token,
            ).execute()

            for meta in resp.get("files", []):
                if meta["mimeType"] == "application/vnd.google-apps.folder":
                    continue
                if manifest.has_drive_id(meta["id"]):
                    skipped += 1
                    continue
                if dry_run:
                    print(f"[DRY RUN] would download: {meta['name']} "
                          f"({meta['id']}) -> sources/{source_type}/")
                    continue

                name = meta["name"]
                if meta["mimeType"] in _GOOGLE_EXPORT:
                    export_mime, ext = _GOOGLE_EXPORT[meta["mimeType"]]
                    request = service.files().export_media(
                        fileId=meta["id"], mimeType=export_mime
                    )
                    if not name.endswith(ext):
                        name += ext
                else:
                    request = service.files().get_media(fileId=meta["id"])

                buf = io.BytesIO()
                downloader = MediaIoBaseDownload(buf, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                content = buf.getvalue()

                sha = hashlib.sha256(content).hexdigest()
                if manifest.has(sha):
                    skipped += 1
                    continue

                dest_dir = paths.sources_dir / source_type
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = _dest_for(dest_dir, name, sha)
                dest.write_bytes(content)

                record = ManifestRecord(
                    sha256=sha,
                    source="gdrive",
                    source_type=source_type,
                    original_name=meta["name"],
                    dest_path=str(dest.relative_to(paths.root)),
                    size_bytes=len(content),
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    drive_id=meta["id"],
                )
                manifest.append(record)
                fetched.append(record)
                logger.info("fetched from Drive: %s", record.dest_path)

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    print(f"{len(fetched)} fetched, {skipped} skipped from Google Drive")
    return fetched


# =============================================================================
# CLI
# =============================================================================

def _gdrive_folders_from_env() -> Dict[str, str]:
    """Collect UBIK_GDRIVE_FOLDER_<TYPE> env vars into {type: folder_id}."""
    import os
    folders = {}
    for source_type in SOURCE_TYPES:
        folder_id = os.environ.get(f"UBIK_GDRIVE_FOLDER_{source_type.upper()}")
        if folder_id:
            folders[source_type] = folder_id
    return folders


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = argparse.ArgumentParser(
        prog="fetch_sources",
        description="Acquire raw source files into sources/<type>/ (idempotent)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", metavar="DIR",
                      help="Fetch from a local directory of manual downloads")
    mode.add_argument("--gdrive", action="store_true",
                      help="Fetch from configured Google Drive folders")
    parser.add_argument("--type", dest="source_type", choices=SOURCE_TYPES,
                        help="Source bucket (required with --local; "
                             "filters folders with --gdrive)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without copying or touching the manifest")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    config = load_config()

    try:
        if args.local:
            if not args.source_type:
                parser.error("--local requires --type")
            fetch_local(
                Path(args.local), args.source_type, config.paths,
                dry_run=args.dry_run,
            )
        else:
            import os
            folders = _gdrive_folders_from_env()
            if args.source_type:
                folders = {
                    k: v for k, v in folders.items() if k == args.source_type
                }
            if not folders:
                print(
                    "No Drive folders configured. Set UBIK_GDRIVE_FOLDER_<TYPE> "
                    f"env vars (types: {', '.join(SOURCE_TYPES)}).",
                    file=sys.stderr,
                )
                return 1
            fetch_gdrive(
                folders, config.paths,
                credentials_path=Path(
                    os.environ.get("GDRIVE_CREDENTIALS_PATH", "")
                ).expanduser(),
                token_path=Path(
                    os.environ.get("GDRIVE_TOKEN_PATH", "")
                ).expanduser(),
                dry_run=args.dry_run,
            )
        return 0
    except FetchError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
