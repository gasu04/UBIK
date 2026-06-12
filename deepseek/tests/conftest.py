#!/usr/bin/env python3
"""
conftest - Pytest Configuration and Shared Fixtures

Provides shared fixtures and configuration for the DeepSeek RAG test suite.
Integration tests require valid Google Drive credentials (token.json in
the project root) and a live Ollama service.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on the path so imports work from any test subdirectory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Path helpers for tests that need credential files
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"
TOKEN_PATH = PROJECT_ROOT / "token.json"


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require live external services (deselect with -m 'not integration')"
    )
    config.addinivalue_line(
        "markers",
        "requires_credentials: marks tests that need Google Drive credentials"
    )
