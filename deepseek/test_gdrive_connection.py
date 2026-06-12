#!/usr/bin/env python3
"""Quick diagnostic script to test Google Drive API connection"""

import sys
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import socket

print("=" * 60)
print("Google Drive Connection Diagnostic")
print("=" * 60 + "\n")

# Check internet
print("1. Testing internet connection...")
try:
    socket.create_connection(("8.8.8.8", 53), timeout=3)
    print("   ✓ Internet connection OK\n")
except OSError:
    print("   ❌ No internet connection!\n")
    sys.exit(1)

# Load credentials
print("2. Loading credentials...")
try:
    creds = Credentials.from_authorized_user_file('token.json')
    print("   ✓ Credentials loaded\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    sys.exit(1)

# Build service
print("3. Building Drive service...")
try:
    service = build('drive', 'v3', credentials=creds)
    print("   ✓ Service created\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    sys.exit(1)

# Test API - About (simplest call)
print("4. Testing API with 'about' call...")
try:
    about = service.about().get(fields="user").execute()
    user = about.get('user', {})
    print(f"   ✓ Connected as: {user.get('emailAddress', 'Unknown')}\n")
except Exception as e:
    print(f"   ❌ Error: {e}")
    print(f"   Type: {type(e).__name__}\n")
    sys.exit(1)

# List folders (the problematic call)
print("5. Listing folders (this is what's hanging)...")
try:
    print("   Starting request...")
    results = service.files().list(
        q="mimeType='application/vnd.google-apps.folder' and trashed=false",
        pageSize=10,
        fields='files(id, name)',
        orderBy='name'
    ).execute()

    folders = results.get('files', [])
    print(f"   ✓ Success! Found {len(folders)} folder(s):")
    for f in folders[:5]:  # Show first 5
        print(f"      - {f['name']}")
    if len(folders) > 5:
        print(f"      ... and {len(folders) - 5} more")
    print()

except Exception as e:
    print(f"   ❌ Error: {e}")
    print(f"   Type: {type(e).__name__}\n")
    sys.exit(1)

print("=" * 60)
print("✅ All tests passed! Connection is working.")
print("=" * 60)
