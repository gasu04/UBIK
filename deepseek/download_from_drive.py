#!/usr/bin/env python3
"""
Google Drive Folder Downloader
Downloads all files from a specified Google Drive folder to a local directory.
Supports Google Docs, Sheets, and regular files.
"""

import os
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate():
    """
    Authenticate with Google Drive using OAuth 2.0.

    First time: Opens browser for authentication and saves token
    Subsequent times: Uses saved token

    Returns:
        Google Drive service object
    """
    creds = None

    # Check if we have a saved token
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print("\nERROR: credentials.json not found!")
                print("\nPlease follow these steps:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create a new project (or select existing)")
                print("3. Enable Google Drive API")
                print("4. Create OAuth 2.0 credentials (Desktop app)")
                print("5. Download the JSON file and save as 'credentials.json'")
                print("\nSee the SETUP.md file for detailed instructions.")
                exit(1)

            print("Starting authentication flow...")
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
        print("Credentials saved to token.json")

    return build('drive', 'v3', credentials=creds)

def get_folder_name(service, folder_id):
    """Get the name of a folder by its ID"""
    try:
        folder = service.files().get(fileId=folder_id, fields='name').execute()
        return folder.get('name', 'Unknown')
    except Exception as e:
        print(f"Warning: Could not get folder name: {e}")
        return 'google_drive_docs'

def download_folder(service, folder_id, local_path, recursive=False):
    """
    Download all files from a Google Drive folder.

    Args:
        service: Google Drive service object
        folder_id: ID of the Google Drive folder
        local_path: Local directory to save files
        recursive: If True, download subfolders too
    """
    os.makedirs(local_path, exist_ok=True)

    # Query all files in the folder
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)",
        pageSize=1000
    ).execute()

    files = results.get('files', [])

    if not files:
        print(f'No files found in folder.')
        return

    print(f"\nFound {len(files)} items in folder")
    print("-" * 50)

    downloaded_count = 0
    skipped_count = 0

    for file in files:
        file_id = file['id']
        file_name = file['name']
        mime_type = file['mimeType']

        # Handle folders
        if mime_type == 'application/vnd.google-apps.folder':
            if recursive:
                print(f"\n📁 Entering subfolder: {file_name}")
                subfolder_path = os.path.join(local_path, file_name)
                download_folder(service, file_id, subfolder_path, recursive)
            else:
                print(f"⏭️  Skipping folder: {file_name} (use --recursive to include)")
                skipped_count += 1
            continue

        print(f"\n📄 Downloading: {file_name}")

        try:
            # Handle Google Workspace files (Docs, Sheets, Slides)
            if mime_type == 'application/vnd.google-apps.document':
                # Export Google Doc as plain text
                request = service.files().export_media(
                    fileId=file_id,
                    mimeType='text/plain'
                )
                file_name += '.txt'
                print(f"   (Exporting Google Doc as text)")

            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                # Export Google Sheet as CSV
                request = service.files().export_media(
                    fileId=file_id,
                    mimeType='text/csv'
                )
                file_name += '.csv'
                print(f"   (Exporting Google Sheet as CSV)")

            elif mime_type == 'application/vnd.google-apps.presentation':
                # Export Google Slides as PDF
                request = service.files().export_media(
                    fileId=file_id,
                    mimeType='application/pdf'
                )
                file_name += '.pdf'
                print(f"   (Exporting Google Slides as PDF)")

            elif mime_type.startswith('application/vnd.google-apps'):
                # Other Google Workspace files - skip or export as PDF
                print(f"   ⚠️  Unsupported Google Workspace type: {mime_type}")
                skipped_count += 1
                continue

            else:
                # Regular file download
                request = service.files().get_media(fileId=file_id)

            # Download the file
            file_path = os.path.join(local_path, file_name)
            fh = io.FileIO(file_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    print(f"   Progress: {progress}%", end='\r')

            print(f"   ✅ Downloaded successfully")
            downloaded_count += 1

        except Exception as e:
            print(f"   ❌ Error: {e}")
            skipped_count += 1

    print("\n" + "=" * 50)
    print(f"✅ Downloaded: {downloaded_count} files")
    if skipped_count > 0:
        print(f"⏭️  Skipped: {skipped_count} items")
    print(f"📁 Location: {os.path.abspath(local_path)}")
    print("=" * 50)

def main():
    """Main function"""
    print("=" * 50)
    print("Google Drive Folder Downloader")
    print("=" * 50)

    # CONFIGURATION - Update these values
    FOLDER_ID = input("\nEnter Google Drive Folder ID: ").strip()

    if not FOLDER_ID:
        print("\nERROR: Folder ID is required!")
        print("\nTo get the folder ID:")
        print("1. Open the folder in Google Drive")
        print("2. Look at the URL: https://drive.google.com/drive/folders/FOLDER_ID")
        print("3. Copy the FOLDER_ID part")
        exit(1)

    # Ask for download location
    default_path = "./google_drive_docs"
    download_path = input(f"\nEnter download path (default: {default_path}): ").strip()
    if not download_path:
        download_path = default_path

    # Ask about recursive download
    recursive_input = input("\nDownload subfolders too? (y/N): ").strip().lower()
    recursive = recursive_input in ['y', 'yes']

    print("\n" + "=" * 50)
    print("Authenticating with Google Drive...")
    print("=" * 50)

    try:
        service = authenticate()
        print("✅ Authentication successful!")

        # Get folder name
        folder_name = get_folder_name(service, FOLDER_ID)
        print(f"\nFolder: {folder_name}")
        print(f"Downloading to: {os.path.abspath(download_path)}")
        if recursive:
            print("Mode: Recursive (includes subfolders)")

        input("\nPress Enter to start download...")

        download_folder(service, FOLDER_ID, download_path, recursive)

        print("\n✅ All done! Your files are ready for the RAG system.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)

if __name__ == '__main__':
    main()
