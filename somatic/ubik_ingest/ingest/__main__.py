#!/usr/bin/env python3
"""
UBIK Ingestion System - Module Entry Point

Enables running the ingestion CLI as a module:
    python -m ingest local ~/Documents
    python -m ingest file ~/document.pdf
    python -m ingest gdrive --folder-id ABC123 --credentials creds.json

Version: 0.1.0
"""

from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
