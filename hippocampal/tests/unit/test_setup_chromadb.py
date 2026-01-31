#!/usr/bin/env python3
"""
Unit tests for ChromaDB setup module.

Tests configuration, connection retry logic, collection creation,
and sample data handling using mocked ChromaDB client.
"""

import os
from unittest.mock import Mock, patch, MagicMock

import pytest

from setup_chromadb import (
    ChromaDBSetupConfig,
    SAMPLE_EPISODIC,
    SAMPLE_SEMANTIC,
    connect_with_retry,
    get_embedding_function,
    create_collection,
    add_sample_data,
    verify_setup,
    create_collections,
)
from exceptions import DatabaseConnectionError


# =============================================================================
# ChromaDBSetupConfig Tests
# =============================================================================

class TestChromaDBSetupConfig:
    """Tests for ChromaDBSetupConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = ChromaDBSetupConfig()

            assert config.host == "localhost"
            assert config.port == 8001
            assert config.token == ""
            assert config.max_retries == 10
            assert config.retry_delay == 3.0
            assert config.embedding_model == "all-MiniLM-L6-v2"

    def test_loads_from_environment(self) -> None:
        """Test loading values from environment variables."""
        env = {
            "CHROMADB_HOST": "chromadb.example.com",
            "CHROMADB_PORT": "9000",
            "CHROMADB_TOKEN": "test_token_123",
        }

        with patch.dict(os.environ, env, clear=True):
            config = ChromaDBSetupConfig()

            assert config.host == "chromadb.example.com"
            assert config.port == 9000
            assert config.token == "test_token_123"

    def test_hnsw_defaults(self) -> None:
        """Test HNSW index parameter defaults."""
        config = ChromaDBSetupConfig()

        assert config.hnsw_space == "cosine"
        assert config.hnsw_construction_ef == 128
        assert config.hnsw_search_ef == 64
        assert config.hnsw_m == 16


# =============================================================================
# Sample Data Tests
# =============================================================================

class TestSampleData:
    """Tests for sample data constants."""

    def test_episodic_samples_valid(self) -> None:
        """Verify episodic sample data structure."""
        assert len(SAMPLE_EPISODIC) >= 1

        for sample in SAMPLE_EPISODIC:
            assert "id" in sample
            assert "document" in sample
            assert "metadata" in sample
            assert sample["id"].startswith("ep_")
            assert len(sample["document"]) > 0

            metadata = sample["metadata"]
            assert "type" in metadata
            assert "timestamp" in metadata
            assert "importance" in metadata

    def test_semantic_samples_valid(self) -> None:
        """Verify semantic sample data structure."""
        assert len(SAMPLE_SEMANTIC) >= 1

        for sample in SAMPLE_SEMANTIC:
            assert "id" in sample
            assert "document" in sample
            assert "metadata" in sample
            assert sample["id"].startswith("sem_")
            assert len(sample["document"]) > 0

            metadata = sample["metadata"]
            assert "type" in metadata
            assert "category" in metadata
            assert "confidence" in metadata


# =============================================================================
# Connection Tests
# =============================================================================

class TestConnectWithRetry:
    """Tests for connect_with_retry function."""

    @patch('setup_chromadb.chromadb')
    @patch('setup_chromadb.time.sleep')
    def test_connects_on_first_attempt(self, mock_sleep, mock_chromadb) -> None:
        """Test successful connection on first attempt."""
        mock_client = Mock()
        mock_chromadb.HttpClient.return_value = mock_client

        config = ChromaDBSetupConfig()
        result = connect_with_retry(config)

        assert result is mock_client
        mock_client.heartbeat.assert_called_once()
        mock_sleep.assert_not_called()

    @patch('setup_chromadb.chromadb')
    @patch('setup_chromadb.time.sleep')
    def test_retries_on_failure(self, mock_sleep, mock_chromadb) -> None:
        """Test retry behavior on connection failure."""
        mock_client = Mock()

        # Fail twice, then succeed
        mock_chromadb.HttpClient.side_effect = [
            Exception("Connection refused"),
            Exception("Connection refused"),
            mock_client,
        ]

        config = ChromaDBSetupConfig()
        config.max_retries = 5
        config.retry_delay = 0.01

        result = connect_with_retry(config)

        assert result is mock_client
        assert mock_sleep.call_count == 2

    @patch('setup_chromadb.chromadb')
    @patch('setup_chromadb.time.sleep')
    def test_raises_after_max_retries(self, mock_sleep, mock_chromadb) -> None:
        """Test exception after all retries exhausted."""
        mock_chromadb.HttpClient.side_effect = Exception("Connection refused")

        config = ChromaDBSetupConfig()
        config.max_retries = 3
        config.retry_delay = 0.01

        with pytest.raises(DatabaseConnectionError) as exc_info:
            connect_with_retry(config)

        assert exc_info.value.service == "chromadb"
        assert "3 attempts" in str(exc_info.value)

    @patch('setup_chromadb.chromadb')
    @patch('setup_chromadb.time.sleep')
    def test_uses_config_values(self, mock_sleep, mock_chromadb) -> None:
        """Test that config values are used for connection."""
        mock_client = Mock()
        mock_chromadb.HttpClient.return_value = mock_client

        config = ChromaDBSetupConfig()
        config.host = "custom-host"
        config.port = 9999
        config.token = "secret_token"

        connect_with_retry(config)

        mock_chromadb.HttpClient.assert_called_once_with(
            host="custom-host",
            port=9999,
            headers={"Authorization": "Bearer secret_token"}
        )


# =============================================================================
# Embedding Function Tests
# =============================================================================

class TestGetEmbeddingFunction:
    """Tests for get_embedding_function function."""

    @patch('setup_chromadb.embedding_functions')
    def test_uses_sentence_transformer(self, mock_ef) -> None:
        """Test using SentenceTransformer embeddings."""
        mock_embedding_fn = Mock()
        mock_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embedding_fn

        config = ChromaDBSetupConfig()
        config.embedding_model = "test-model"

        result = get_embedding_function(config)

        assert result is mock_embedding_fn
        mock_ef.SentenceTransformerEmbeddingFunction.assert_called_once_with(
            model_name="test-model"
        )

    @patch('setup_chromadb.embedding_functions')
    def test_falls_back_to_default(self, mock_ef) -> None:
        """Test fallback to default embeddings when SentenceTransformer fails."""
        mock_default_fn = Mock()
        mock_ef.SentenceTransformerEmbeddingFunction.side_effect = Exception("Not installed")
        mock_ef.DefaultEmbeddingFunction.return_value = mock_default_fn

        config = ChromaDBSetupConfig()
        result = get_embedding_function(config)

        assert result is mock_default_fn
        mock_ef.DefaultEmbeddingFunction.assert_called_once()


# =============================================================================
# Collection Creation Tests
# =============================================================================

class TestCreateCollection:
    """Tests for create_collection function."""

    def test_creates_collection_with_metadata(self) -> None:
        """Test collection creation with proper metadata."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.create_collection.return_value = mock_collection

        config = ChromaDBSetupConfig()
        embedding_fn = Mock()

        result = create_collection(
            client=mock_client,
            name="test_collection",
            description="Test description",
            embedding_fn=embedding_fn,
            config=config
        )

        assert result is mock_collection

        # Verify collection was created with correct args
        call_kwargs = mock_client.create_collection.call_args[1]
        assert call_kwargs["name"] == "test_collection"
        assert call_kwargs["embedding_function"] is embedding_fn
        assert call_kwargs["metadata"]["description"] == "Test description"
        assert call_kwargs["metadata"]["hnsw:space"] == config.hnsw_space

    def test_deletes_existing_collection(self) -> None:
        """Test that existing collection is deleted first."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.create_collection.return_value = mock_collection

        config = ChromaDBSetupConfig()

        create_collection(
            client=mock_client,
            name="existing_collection",
            description="Test",
            embedding_fn=Mock(),
            config=config
        )

        mock_client.delete_collection.assert_called_once_with("existing_collection")

    def test_handles_missing_collection_on_delete(self) -> None:
        """Test graceful handling when collection doesn't exist."""
        mock_client = Mock()
        mock_client.delete_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = Mock()

        config = ChromaDBSetupConfig()

        # Should not raise
        create_collection(
            client=mock_client,
            name="new_collection",
            description="Test",
            embedding_fn=Mock(),
            config=config
        )


# =============================================================================
# Sample Data Addition Tests
# =============================================================================

class TestAddSampleData:
    """Tests for add_sample_data function."""

    def test_adds_documents_correctly(self) -> None:
        """Test adding sample documents to collection."""
        mock_collection = Mock()
        mock_collection.name = "test_collection"

        samples = [
            {
                "id": "test_001",
                "document": "Test document 1",
                "metadata": {"type": "test"}
            },
            {
                "id": "test_002",
                "document": "Test document 2",
                "metadata": {"type": "test"}
            }
        ]

        result = add_sample_data(mock_collection, samples)

        assert result == 2
        mock_collection.add.assert_called_once()

        call_kwargs = mock_collection.add.call_args[1]
        assert call_kwargs["ids"] == ["test_001", "test_002"]
        assert call_kwargs["documents"] == ["Test document 1", "Test document 2"]
        assert call_kwargs["metadatas"] == [{"type": "test"}, {"type": "test"}]

    def test_returns_count(self) -> None:
        """Test that function returns count of added documents."""
        mock_collection = Mock()
        mock_collection.name = "test"

        samples = [{"id": "1", "document": "doc", "metadata": {}}] * 5

        result = add_sample_data(mock_collection, samples)

        assert result == 5


# =============================================================================
# Verification Tests
# =============================================================================

class TestVerifySetup:
    """Tests for verify_setup function."""

    def test_verification_passes_with_results(self) -> None:
        """Test verification passes when query returns results."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['result 1', 'result 2']],
            'metadatas': [[{'type': 'value'}, {'type': 'belief'}]]
        }

        mock_client.list_collections.return_value = [
            Mock(name="ubik_semantic", count=Mock(return_value=10))
        ]
        mock_client.get_collection.return_value = mock_collection

        result = verify_setup(mock_client)

        assert result is True

    def test_verification_fails_with_no_results(self) -> None:
        """Test verification fails when query returns empty results."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }

        mock_client.list_collections.return_value = []
        mock_client.get_collection.return_value = mock_collection

        result = verify_setup(mock_client)

        assert result is False

    def test_verification_fails_on_exception(self) -> None:
        """Test verification fails on query exception."""
        mock_client = Mock()
        mock_client.list_collections.return_value = []
        mock_client.get_collection.side_effect = Exception("Collection not found")

        result = verify_setup(mock_client)

        assert result is False


# =============================================================================
# Create Collections (Main Function) Tests
# =============================================================================

class TestCreateCollections:
    """Tests for create_collections main function."""

    @patch('setup_chromadb.verify_setup')
    @patch('setup_chromadb.add_sample_data')
    @patch('setup_chromadb.create_collection')
    @patch('setup_chromadb.get_embedding_function')
    @patch('setup_chromadb.connect_with_retry')
    def test_successful_setup(
        self,
        mock_connect,
        mock_embed,
        mock_create,
        mock_add,
        mock_verify,
        capsys
    ) -> None:
        """Test successful full setup flow."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['test doc']],
            'metadatas': [[{'type': 'test'}]]
        }

        mock_connect.return_value = mock_client
        mock_embed.return_value = Mock()
        mock_create.return_value = mock_collection
        mock_add.return_value = 3
        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection

        result = create_collections()

        assert result is True
        mock_connect.assert_called_once()
        mock_embed.assert_called_once()
        assert mock_create.call_count == 2  # episodic + semantic
        assert mock_add.call_count == 2

    @patch('setup_chromadb.connect_with_retry')
    def test_connection_failure(self, mock_connect, capsys) -> None:
        """Test handling of connection failure."""
        mock_connect.side_effect = DatabaseConnectionError(
            service="chromadb",
            host="localhost",
            port=8001,
            reason="Connection refused"
        )

        result = create_collections()

        assert result is False

    @patch('setup_chromadb.get_embedding_function')
    @patch('setup_chromadb.connect_with_retry')
    def test_uses_custom_config(self, mock_connect, mock_embed) -> None:
        """Test that custom config is used throughout."""
        mock_client = Mock()
        mock_connect.return_value = mock_client
        mock_embed.side_effect = Exception("Stop here")

        custom_config = ChromaDBSetupConfig()
        custom_config.host = "custom-host"
        custom_config.embedding_model = "custom-model"

        # Will fail at embedding but that's OK - we're testing config passing
        create_collections(custom_config)

        mock_connect.assert_called_once_with(custom_config)
        mock_embed.assert_called_once_with(custom_config)
