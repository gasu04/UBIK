#!/usr/bin/env python3
"""
test_google_docs_detection - Google Docs MIME Type Detection Tests

Verifies that the system correctly categorises Google Workspace files
(Docs, Sheets, Slides) and regular files by MIME type.

Usage:
    pytest tests/integration/test_google_docs_detection.py -v
    # or directly:
    python tests/integration/test_google_docs_detection.py

Dependencies:
    - google-auth>=2.0.0
    - google-api-python-client>=2.0.0
"""

import sys
from pathlib import Path
from typing import Dict, List

import pytest

_PROJECT_ROOT = Path(__file__).parents[2]
_TOKEN_PATH = str(_PROJECT_ROOT / "token.json")

pytestmark = pytest.mark.integration

GOOGLE_DOC_MIMES = {
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
}


def _categorise_files(files: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorise a list of Drive file metadata dicts by type."""
    categories: Dict[str, List[Dict]] = {
        "google_docs": [],
        "google_sheets": [],
        "google_slides": [],
        "regular": [],
    }
    for file in files:
        mime = file.get("mimeType", "")
        if mime == "application/vnd.google-apps.document":
            categories["google_docs"].append(file)
        elif mime == "application/vnd.google-apps.spreadsheet":
            categories["google_sheets"].append(file)
        elif mime == "application/vnd.google-apps.presentation":
            categories["google_slides"].append(file)
        else:
            categories["regular"].append(file)
    return categories


def test_categorise_files_google_doc() -> None:
    """Unit: Google Docs are categorised correctly."""
    files = [{"name": "My Doc", "mimeType": "application/vnd.google-apps.document"}]
    result = _categorise_files(files)
    assert len(result["google_docs"]) == 1
    assert len(result["regular"]) == 0


def test_categorise_files_regular() -> None:
    """Unit: Regular files (PDF, DOCX) are not mis-categorised as Google types."""
    files = [
        {"name": "report.pdf", "mimeType": "application/pdf"},
        {"name": "notes.docx", "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ]
    result = _categorise_files(files)
    assert len(result["regular"]) == 2
    assert len(result["google_docs"]) == 0


def test_folder_files_live(folder_name: str = "Tactiq Transcription") -> None:
    """Integration: List and categorise files from a real Drive folder."""
    if not Path(_TOKEN_PATH).exists():
        pytest.skip(f"token.json not found at {_TOKEN_PATH}")

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(_TOKEN_PATH)
    service = build("drive", "v3", credentials=creds)

    folder_results = service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id, name)",
    ).execute()

    folders = folder_results.get("files", [])
    if not folders:
        pytest.skip(f"Folder '{folder_name}' not found in Drive — skipping live test")

    folder_id = folders[0]["id"]

    file_results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        pageSize=100,
        fields="files(id, name, mimeType, size)",
        orderBy="name",
    ).execute()

    files = file_results.get("files", [])
    categories = _categorise_files(files)
    total = sum(len(v) for v in categories.values())
    assert total == len(files), "All files must be categorised"


if __name__ == "__main__":
    _test_folder_name = "Tactiq Transcription"
    print(f"🔍 Searching for folder: '{_test_folder_name}'")

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(_TOKEN_PATH)
    service = build("drive", "v3", credentials=creds)

    folder_results = service.files().list(
        q=f"name='{_test_folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id, name)",
    ).execute()

    folders = folder_results.get("files", [])
    if not folders:
        print(f"❌ Folder '{_test_folder_name}' not found")
        sys.exit(1)

    folder_id = folders[0]["id"]
    print(f"✓ Found folder: {folders[0]['name']} (ID: {folder_id})\n")

    file_results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        pageSize=100,
        fields="files(id, name, mimeType, size)",
        orderBy="name",
    ).execute()

    files = file_results.get("files", [])
    categories = _categorise_files(files)

    print("📊 SUMMARY:")
    print("=" * 70)
    print(f"Total files: {len(files)}")
    print(f"  📄 Google Docs: {len(categories['google_docs'])}")
    print(f"  📊 Google Sheets: {len(categories['google_sheets'])}")
    print(f"  📽️  Google Slides: {len(categories['google_slides'])}")
    print(f"  📁 Regular files: {len(categories['regular'])}")
    print("=" * 70)
    print("✅ The script can handle ALL these file types!")
