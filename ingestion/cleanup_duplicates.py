#!/usr/bin/env python3
"""
Cleanup duplicate ingest records from ChromaDB.

Connects to the hippocampal ChromaDB node and removes all records
for specified source files, so the ingestor can run once cleanly.

Usage:
    python cleanup_duplicates.py [--source-file FILE] [--dry-run]

Examples:
    # Show counts without deleting
    python cleanup_duplicates.py --dry-run

    # Delete all therapy_test records
    python cleanup_duplicates.py --source-dir therapy_test

    # Delete specific file
    python cleanup_duplicates.py --source-file therapy_2026-02-16.transcript
"""

import argparse
import os
import sys
from pathlib import Path

import chromadb

# Load env from maestro .env
_env_path = Path(__file__).parents[2] / "maestro" / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

CHROMA_HOST = os.getenv("CHROMADB_HOST", os.getenv("HIPPOCAMPAL_TAILSCALE_IP", "localhost"))
CHROMA_PORT = int(os.getenv("CHROMADB_PORT", "8001"))
CHROMA_TOKEN = os.getenv("CHROMADB_TOKEN", "")

COLLECTIONS = ["ubik_episodic", "ubik_semantic"]


def get_client() -> chromadb.HttpClient:
    settings = chromadb.Settings(anonymized_telemetry=False)
    if CHROMA_TOKEN:
        settings = chromadb.Settings(
            anonymized_telemetry=False,
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=CHROMA_TOKEN,
        )
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=settings)


def list_all_source_files(collection) -> dict[str, int]:
    """Return {source_file: count} for all records in the collection."""
    try:
        result = collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in result["metadatas"]:
            # episodic uses "source_file", semantic uses "source"
            src = meta.get("source_file", "") or meta.get("source", "")
            if src:
                counts[src] = counts.get(src, 0) + 1
        return counts
    except Exception as e:
        print(f"  Error listing: {e}")
        return {}


def get_ids_by_source(collection, source_file: str) -> list[str]:
    """Return IDs of all records matching source_file (handles both key names)."""
    result = collection.get(include=["metadatas"])
    return [
        id_
        for id_, meta in zip(result["ids"], result["metadatas"])
        if (meta.get("source_file", "") or meta.get("source", "")) == source_file
    ]


def count_by_source(collection, source_file: str) -> int:
    return len(get_ids_by_source(collection, source_file))


def delete_by_source(collection, source_file: str) -> int:
    """Delete all records with matching source_file. Returns count deleted."""
    ids = get_ids_by_source(collection, source_file)
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def main():
    parser = argparse.ArgumentParser(description="Cleanup duplicate ingest records")
    parser.add_argument("--source-file", help="Delete records for this specific source file")
    parser.add_argument("--source-dir", help="Delete records whose source_file contains this string")
    parser.add_argument("--dry-run", action="store_true", help="Show counts, do not delete")
    parser.add_argument("--all", action="store_true", help="Show all source files in ChromaDB")
    args = parser.parse_args()

    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
    try:
        client = get_client()
        client.heartbeat()
        print("Connected.\n")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    for coll_name in COLLECTIONS:
        try:
            collection = client.get_collection(coll_name)
        except Exception as e:
            print(f"[{coll_name}] Could not open: {e}")
            continue

        total = collection.count()
        print(f"[{coll_name}] Total records: {total}")

        if args.all or (not args.source_file and not args.source_dir):
            # Show all source files
            counts = list_all_source_files(collection)
            if counts:
                for src, n in sorted(counts.items(), key=lambda x: -x[1]):
                    print(f"  {n:4d}  {src}")
            else:
                print("  (no records with source_file metadata)")
            print()
            continue

        # Collect target source files
        targets: list[str] = []
        if args.source_file:
            targets = [args.source_file]
        elif args.source_dir:
            counts = list_all_source_files(collection)
            targets = [src for src in counts if args.source_dir in src]
            if not targets:
                print(f"  No records found matching --source-dir '{args.source_dir}'")
                print()
                continue

        for src in targets:
            count = count_by_source(collection, src)
            if args.dry_run:
                print(f"  [dry-run] Would delete {count} records for '{src}'")
            else:
                deleted = delete_by_source(collection, src)
                print(f"  Deleted {deleted} records for '{src}'")

        print()


if __name__ == "__main__":
    main()
