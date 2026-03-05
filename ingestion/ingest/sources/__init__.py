"""
UBIK Ingestion System - Source Handlers

Provides integrations with various content sources:
- Local filesystem
- Google Drive
- (Future: Dropbox, OneDrive, etc.)

Usage:
    from ingest.sources import GoogleDriveSource

    gdrive = GoogleDriveSource(
        credentials_path="credentials.json",
        token_path="token.json"
    )
    await gdrive.authenticate()
    files = await gdrive.list_files(folder_id="...")

Version: 0.1.0
"""

from .gdrive import GoogleDriveSource, GoogleDriveConfig

__all__ = [
    'GoogleDriveSource',
    'GoogleDriveConfig',
]
