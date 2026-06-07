#!/usr/bin/env python3
"""
test_gdrive_connection - Google Drive API Connection Diagnostic

Verifies network connectivity, credential loading, and basic Drive API
calls. Run this when troubleshooting authentication or network issues.

Usage:
    pytest tests/integration/test_gdrive_connection.py -v
    # or directly:
    python tests/integration/test_gdrive_connection.py

Dependencies:
    - google-auth>=2.0.0
    - google-api-python-client>=2.0.0
"""

import sys
import socket
from pathlib import Path

import pytest

# Locate credential files relative to project root
_PROJECT_ROOT = Path(__file__).parents[2]
_TOKEN_PATH = str(_PROJECT_ROOT / "token.json")

pytestmark = pytest.mark.integration


def test_internet_connectivity() -> None:
    """Verify basic internet connectivity via DNS resolution."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
    except OSError as exc:
        pytest.fail(f"No internet connection: {exc}")


def test_credentials_load() -> None:
    """Verify token.json exists and loads as valid Google credentials."""
    from google.oauth2.credentials import Credentials

    if not Path(_TOKEN_PATH).exists():
        pytest.skip(f"token.json not found at {_TOKEN_PATH}")

    creds = Credentials.from_authorized_user_file(_TOKEN_PATH)
    assert creds is not None, "Credentials should not be None"


def test_drive_service_build() -> None:
    """Verify the Drive v3 service can be constructed from credentials."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if not Path(_TOKEN_PATH).exists():
        pytest.skip(f"token.json not found at {_TOKEN_PATH}")

    creds = Credentials.from_authorized_user_file(_TOKEN_PATH)
    service = build("drive", "v3", credentials=creds)
    assert service is not None


def test_drive_about_call() -> None:
    """Verify the Drive API 'about' endpoint returns authenticated user info."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if not Path(_TOKEN_PATH).exists():
        pytest.skip(f"token.json not found at {_TOKEN_PATH}")

    creds = Credentials.from_authorized_user_file(_TOKEN_PATH)
    service = build("drive", "v3", credentials=creds)
    about = service.about().get(fields="user").execute()
    user = about.get("user", {})
    assert "emailAddress" in user, "Response should contain emailAddress"


def test_list_folders() -> None:
    """Verify that the Drive API can list folder items without hanging."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if not Path(_TOKEN_PATH).exists():
        pytest.skip(f"token.json not found at {_TOKEN_PATH}")

    creds = Credentials.from_authorized_user_file(_TOKEN_PATH)
    service = build("drive", "v3", credentials=creds)
    results = service.files().list(
        q="mimeType='application/vnd.google-apps.folder' and trashed=false",
        pageSize=10,
        fields="files(id, name)",
        orderBy="name",
    ).execute()
    # Just assert the call succeeded; zero folders is a valid result
    assert "files" in results


if __name__ == "__main__":
    # Allow running directly as a diagnostic script
    print("=" * 60)
    print("Google Drive Connection Diagnostic")
    print("=" * 60 + "\n")

    print("1. Testing internet connection...")
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        print("   ✓ Internet connection OK\n")
    except OSError:
        print("   ❌ No internet connection!\n")
        sys.exit(1)

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    print("2. Loading credentials...")
    try:
        creds = Credentials.from_authorized_user_file(_TOKEN_PATH)
        print("   ✓ Credentials loaded\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        sys.exit(1)

    print("3. Building Drive service...")
    try:
        service = build("drive", "v3", credentials=creds)
        print("   ✓ Service created\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        sys.exit(1)

    print("4. Testing API with 'about' call...")
    try:
        about = service.about().get(fields="user").execute()
        user = about.get("user", {})
        print(f"   ✓ Connected as: {user.get('emailAddress', 'Unknown')}\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        sys.exit(1)

    print("5. Listing folders...")
    try:
        results = service.files().list(
            q="mimeType='application/vnd.google-apps.folder' and trashed=false",
            pageSize=10,
            fields="files(id, name)",
            orderBy="name",
        ).execute()
        folders = results.get("files", [])
        print(f"   ✓ Success! Found {len(folders)} folder(s):")
        for f in folders[:5]:
            print(f"      - {f['name']}")
        if len(folders) > 5:
            print(f"      ... and {len(folders) - 5} more")
        print()
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        sys.exit(1)

    print("=" * 60)
    print("✅ All tests passed! Connection is working.")
    print("=" * 60)
