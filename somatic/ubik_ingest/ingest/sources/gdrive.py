"""
UBIK Ingestion System - Google Drive Source

Integration with Google Drive for fetching files from cloud storage.
Supports OAuth 2.0 authentication and Google Docs export.

Features:
    - OAuth 2.0 authentication with token persistence
    - Recursive folder traversal
    - Google Docs/Sheets/Slides export to Office formats
    - Pagination handling for large folders
    - IngestItem integration

Setup:
    1. Create OAuth credentials at console.cloud.google.com
    2. Download credentials.json
    3. Run authenticate() for first-time setup

Usage:
    from ingest.sources import GoogleDriveSource

    source = GoogleDriveSource(
        credentials_path="config/credentials.json",
        token_path="config/token.json"
    )
    await source.authenticate()

    # List files in folder
    items = await source.list_files("folder_id_here", recursive=True)

    # Download specific file
    local_path = await source.download_file(file_id, Path("./downloads"))

Version: 0.1.0
"""

import asyncio
import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..models import ContentType, IngestItem

__all__ = [
    'GoogleDriveSource',
    'GoogleDriveConfig',
]

logger = logging.getLogger(__name__)

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# MIME type mappings for Google Workspace files (export formats)
GOOGLE_EXPORT_MIMES: Dict[str, tuple] = {
    'application/vnd.google-apps.document': (
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.docx'
    ),
    'application/vnd.google-apps.spreadsheet': (
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.xlsx'
    ),
    'application/vnd.google-apps.presentation': (
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.pptx'
    ),
    'application/vnd.google-apps.drawing': (
        'image/png',
        '.png'
    ),
}

# Supported file extensions for ingestion
SUPPORTED_EXTENSIONS: Set[str] = {
    # Text
    '.txt', '.md', '.markdown', '.rst',
    # Documents
    '.pdf', '.docx', '.doc', '.odt', '.rtf',
    # Audio
    '.mp3', '.wav', '.m4a', '.flac', '.ogg',
    # Structured
    '.json', '.yaml', '.yml', '.csv',
    # Transcripts
    '.transcript',
}


@dataclass
class GoogleDriveConfig:
    """
    Configuration for Google Drive source.

    Attributes:
        credentials_path: Path to OAuth credentials JSON
        token_path: Path to store/load OAuth token
        download_dir: Directory for downloaded files
        page_size: Files per page in API requests
        max_results: Maximum files to return (0 = unlimited)
        include_trashed: Whether to include trashed files
        supported_extensions: File extensions to include
    """
    credentials_path: str = field(
        default_factory=lambda: os.getenv(
            "GDRIVE_CREDENTIALS_PATH",
            "config/credentials.json"
        )
    )
    token_path: str = field(
        default_factory=lambda: os.getenv(
            "GDRIVE_TOKEN_PATH",
            "config/token.json"
        )
    )
    download_dir: str = field(
        default_factory=lambda: os.getenv(
            "GDRIVE_DOWNLOAD_DIR",
            "./downloads"
        )
    )
    page_size: int = 100
    max_results: int = 0  # 0 = unlimited
    include_trashed: bool = False
    supported_extensions: Set[str] = field(
        default_factory=lambda: SUPPORTED_EXTENSIONS.copy()
    )


class GoogleDriveSource:
    """
    Google Drive integration for fetching files.

    Handles OAuth 2.0 authentication, file listing, and downloads.
    Supports Google Docs export to Office formats.

    Attributes:
        config: Source configuration
        credentials_path: Path to OAuth credentials
        token_path: Path to OAuth token

    Example:
        source = GoogleDriveSource(
            credentials_path="credentials.json",
            token_path="token.json"
        )
        await source.authenticate()

        # List all files in a folder recursively
        items = await source.list_files("1ABC123", recursive=True)
        print(f"Found {len(items)} files")

        # Download a specific file
        local = await source.download_file("file_id", Path("./output"))
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        token_path: Optional[str] = None,
        config: Optional[GoogleDriveConfig] = None
    ):
        """
        Initialize Google Drive source.

        Args:
            credentials_path: Path to OAuth credentials (overrides config)
            token_path: Path to OAuth token (overrides config)
            config: Full configuration object
        """
        self.config = config or GoogleDriveConfig()

        # Override config with explicit paths
        if credentials_path:
            self.config.credentials_path = credentials_path
        if token_path:
            self.config.token_path = token_path

        self._service = None
        self._credentials = None

    async def authenticate(self) -> bool:
        """
        Authenticate with Google Drive API.

        Performs OAuth 2.0 flow if no valid token exists.
        Opens browser for user consent on first run.

        Returns:
            True if authentication successful

        Raises:
            FileNotFoundError: If credentials file not found
            AuthenticationError: If authentication fails
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._authenticate_sync)

    def _authenticate_sync(self) -> bool:
        """Synchronous authentication (run in executor)."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as e:
            raise ImportError(
                "Google API dependencies not installed. "
                "Run: pip install google-api-python-client google-auth-oauthlib"
            ) from e

        creds = None

        # Load existing token
        token_path = Path(self.config.token_path)
        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(
                    str(token_path), SCOPES
                )
                logger.info(f"Loaded credentials from {token_path}")
            except Exception as e:
                logger.warning(f"Failed to load token: {e}")

        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    logger.info("Refreshed expired credentials")
                except Exception as e:
                    logger.warning(f"Failed to refresh token: {e}")
                    creds = None

            if not creds:
                # Run OAuth flow
                credentials_path = Path(self.config.credentials_path)
                if not credentials_path.exists():
                    raise FileNotFoundError(
                        f"Credentials file not found: {credentials_path}\n"
                        "Download from Google Cloud Console: "
                        "https://console.cloud.google.com/apis/credentials"
                    )

                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
                logger.info("Completed OAuth flow")

            # Save token for future use
            token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(token_path, 'w') as token_file:
                token_file.write(creds.to_json())
            logger.info(f"Saved credentials to {token_path}")

        self._credentials = creds
        self._service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive API service initialized")

        return True

    def _get_service(self):
        """
        Get the Drive API service.

        Returns:
            Google Drive API service resource

        Raises:
            RuntimeError: If not authenticated
        """
        if self._service is None:
            raise RuntimeError(
                "Not authenticated. Call authenticate() first."
            )
        return self._service

    async def list_files(
        self,
        folder_id: str,
        recursive: bool = False
    ) -> List[IngestItem]:
        """
        List files in a Google Drive folder.

        Args:
            folder_id: Google Drive folder ID
            recursive: Whether to traverse subfolders

        Returns:
            List of IngestItem objects for supported files
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._list_files_sync,
            folder_id,
            recursive
        )

    def _list_files_sync(
        self,
        folder_id: str,
        recursive: bool
    ) -> List[IngestItem]:
        """Synchronous file listing (run in executor)."""
        service = self._get_service()
        items: List[IngestItem] = []
        folders_to_process = [folder_id]
        processed_folders: Set[str] = set()

        while folders_to_process:
            current_folder = folders_to_process.pop(0)

            if current_folder in processed_folders:
                continue
            processed_folders.add(current_folder)

            page_token = None

            while True:
                # Query files in this folder
                response = self._query_files(current_folder, page_token)

                for file_data in response.get('files', []):
                    mime_type = file_data.get('mimeType', '')

                    # Handle subfolders
                    if mime_type == 'application/vnd.google-apps.folder':
                        if recursive:
                            folders_to_process.append(file_data['id'])
                        continue

                    # Skip unsupported files
                    if not self._is_supported(file_data):
                        continue

                    # Create IngestItem
                    item = self._file_to_ingest_item(file_data)
                    items.append(item)

                    # Check max results
                    if self.config.max_results > 0 and len(items) >= self.config.max_results:
                        logger.info(f"Reached max_results limit: {self.config.max_results}")
                        return items

                # Handle pagination
                page_token = response.get('nextPageToken')
                if not page_token:
                    break

        logger.info(f"Found {len(items)} supported files")
        return items

    def _query_files(
        self,
        folder_id: str,
        page_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query files from Drive API.

        Args:
            folder_id: Folder to query
            page_token: Pagination token

        Returns:
            API response dict with 'files' and optional 'nextPageToken'
        """
        service = self._get_service()

        # Build query
        query_parts = [f"'{folder_id}' in parents"]

        if not self.config.include_trashed:
            query_parts.append("trashed = false")

        query = " and ".join(query_parts)

        # Execute query
        result = service.files().list(
            q=query,
            pageSize=self.config.page_size,
            pageToken=page_token,
            fields=(
                "nextPageToken, files(id, name, mimeType, size, "
                "createdTime, modifiedTime, parents, webViewLink, "
                "md5Checksum, description)"
            ),
        ).execute()

        return result

    def _is_supported(self, file_data: Dict[str, Any]) -> bool:
        """Check if file is supported for ingestion."""
        name = file_data.get('name', '')
        mime_type = file_data.get('mimeType', '')

        # Google Workspace files are exportable
        if mime_type in GOOGLE_EXPORT_MIMES:
            return True

        # Check file extension
        ext = Path(name).suffix.lower()
        return ext in self.config.supported_extensions

    def _file_to_ingest_item(self, file_data: Dict[str, Any]) -> IngestItem:
        """Convert Drive API file data to IngestItem."""
        name = file_data.get('name', 'unknown')
        mime_type = file_data.get('mimeType', '')

        # Handle Google Workspace files
        if mime_type in GOOGLE_EXPORT_MIMES:
            _, ext = GOOGLE_EXPORT_MIMES[mime_type]
            name = Path(name).stem + ext

        return IngestItem.from_gdrive(
            file_id=file_data['id'],
            name=name,
            mime_type=mime_type,
            metadata={
                'size': file_data.get('size', '0'),
                'createdTime': file_data.get('createdTime'),
                'modifiedTime': file_data.get('modifiedTime'),
                'parents': file_data.get('parents', []),
                'webViewLink': file_data.get('webViewLink'),
                'md5Checksum': file_data.get('md5Checksum'),
                'description': file_data.get('description'),
            }
        )

    async def download_file(
        self,
        file_id: str,
        dest_dir: Optional[Path] = None
    ) -> Path:
        """
        Download a file from Google Drive.

        Handles Google Docs export automatically.

        Args:
            file_id: Google Drive file ID
            dest_dir: Directory to save file (uses config.download_dir if None)

        Returns:
            Path to downloaded file
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._download_file_sync,
            file_id,
            dest_dir
        )

    def _download_file_sync(
        self,
        file_id: str,
        dest_dir: Optional[Path]
    ) -> Path:
        """Synchronous file download (run in executor)."""
        from googleapiclient.http import MediaIoBaseDownload

        service = self._get_service()

        # Get file metadata
        file_meta = service.files().get(
            fileId=file_id,
            fields="id, name, mimeType, size"
        ).execute()

        name = file_meta.get('name', 'download')
        mime_type = file_meta.get('mimeType', '')

        # Determine destination
        if dest_dir is None:
            dest_dir = Path(self.config.download_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Handle Google Workspace files (export)
        if mime_type in GOOGLE_EXPORT_MIMES:
            content = self._handle_google_doc(file_id, mime_type)
            _, ext = GOOGLE_EXPORT_MIMES[mime_type]
            dest_path = dest_dir / (Path(name).stem + ext)

            with open(dest_path, 'wb') as f:
                f.write(content)

            logger.info(f"Exported {name} as {dest_path.name}")
            return dest_path

        # Regular file download
        dest_path = dest_dir / name
        request = service.files().get_media(fileId=file_id)

        with open(dest_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logger.debug(f"Download progress: {int(status.progress() * 100)}%")

        logger.info(f"Downloaded {name} to {dest_path}")
        return dest_path

    def _handle_google_doc(self, file_id: str, mime_type: str) -> bytes:
        """
        Export Google Workspace file to Office format.

        Args:
            file_id: File ID to export
            mime_type: Original Google MIME type

        Returns:
            Exported file content as bytes
        """
        service = self._get_service()

        export_mime, _ = GOOGLE_EXPORT_MIMES[mime_type]

        request = service.files().export_media(
            fileId=file_id,
            mimeType=export_mime
        )

        buffer = io.BytesIO()
        from googleapiclient.http import MediaIoBaseDownload
        downloader = MediaIoBaseDownload(buffer, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        return buffer.getvalue()

    async def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific file.

        Args:
            file_id: Google Drive file ID

        Returns:
            File metadata dictionary
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._get_file_metadata_sync,
            file_id
        )

    def _get_file_metadata_sync(self, file_id: str) -> Dict[str, Any]:
        """Synchronous metadata fetch (run in executor)."""
        service = self._get_service()

        return service.files().get(
            fileId=file_id,
            fields=(
                "id, name, mimeType, size, createdTime, modifiedTime, "
                "parents, webViewLink, md5Checksum, description, "
                "owners, lastModifyingUser"
            )
        ).execute()

    async def search_files(
        self,
        query: str,
        folder_id: Optional[str] = None
    ) -> List[IngestItem]:
        """
        Search for files matching a query.

        Args:
            query: Search query (supports Drive search syntax)
            folder_id: Optional folder to limit search

        Returns:
            List of matching IngestItem objects
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._search_files_sync,
            query,
            folder_id
        )

    def _search_files_sync(
        self,
        query: str,
        folder_id: Optional[str]
    ) -> List[IngestItem]:
        """Synchronous file search (run in executor)."""
        service = self._get_service()
        items: List[IngestItem] = []

        # Build query
        query_parts = [f"fullText contains '{query}'"]

        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")

        if not self.config.include_trashed:
            query_parts.append("trashed = false")

        full_query = " and ".join(query_parts)

        page_token = None
        while True:
            result = service.files().list(
                q=full_query,
                pageSize=self.config.page_size,
                pageToken=page_token,
                fields=(
                    "nextPageToken, files(id, name, mimeType, size, "
                    "createdTime, modifiedTime, parents)"
                ),
            ).execute()

            for file_data in result.get('files', []):
                mime_type = file_data.get('mimeType', '')

                # Skip folders
                if mime_type == 'application/vnd.google-apps.folder':
                    continue

                # Skip unsupported
                if not self._is_supported(file_data):
                    continue

                item = self._file_to_ingest_item(file_data)
                items.append(item)

                if self.config.max_results > 0 and len(items) >= self.config.max_results:
                    return items

            page_token = result.get('nextPageToken')
            if not page_token:
                break

        return items

    @property
    def is_authenticated(self) -> bool:
        """Check if source is authenticated."""
        return self._service is not None
