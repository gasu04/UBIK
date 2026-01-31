#!/usr/bin/env python3
"""
Unit tests for health check module.

Tests configuration, output formatting, and individual health check
functions using mocked external dependencies.
"""

import os
import subprocess
from io import StringIO
from unittest.mock import Mock, patch, MagicMock

import pytest

from health_check import (
    HealthCheckConfig,
    print_header,
    print_result,
    check_docker,
    check_neo4j,
    check_chromadb,
    check_mcp_server,
    check_tailscale,
    run_all_checks,
)


# =============================================================================
# HealthCheckConfig Tests
# =============================================================================

class TestHealthCheckConfig:
    """Tests for HealthCheckConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = HealthCheckConfig()

            assert config.neo4j_uri == "bolt://localhost:7687"
            assert config.neo4j_user == "neo4j"
            assert config.neo4j_password == ""
            assert config.chromadb_host == "localhost"
            assert config.chromadb_port == 8001
            assert config.mcp_host == "localhost"
            assert config.mcp_port == "8080"

    def test_loads_from_environment(self) -> None:
        """Test loading values from environment variables."""
        env = {
            "NEO4J_URI": "bolt://neo4j.example.com:7687",
            "NEO4J_USER": "admin",
            "NEO4J_PASSWORD": "secret123",
            "CHROMADB_HOST": "chromadb.example.com",
            "CHROMADB_PORT": "9000",
            "CHROMADB_TOKEN": "my_token",
            "MCP_HOST": "mcp.example.com",
            "MCP_PORT": "9090",
        }

        with patch.dict(os.environ, env, clear=True):
            config = HealthCheckConfig()

            assert config.neo4j_uri == "bolt://neo4j.example.com:7687"
            assert config.neo4j_user == "admin"
            assert config.neo4j_password == "secret123"
            assert config.chromadb_host == "chromadb.example.com"
            assert config.chromadb_port == 9000
            assert config.chromadb_token == "my_token"
            assert config.mcp_host == "mcp.example.com"
            assert config.mcp_port == "9090"


# =============================================================================
# Output Formatting Tests
# =============================================================================

class TestPrintHeader:
    """Tests for print_header function."""

    def test_prints_formatted_header(self, capsys) -> None:
        """Test header output formatting."""
        print_header("Test Section")
        captured = capsys.readouterr()

        assert "=" * 60 in captured.out
        assert "Test Section" in captured.out


class TestPrintResult:
    """Tests for print_result function."""

    def test_success_result(self, capsys) -> None:
        """Test successful result output."""
        print_result("Test Check", True)
        captured = capsys.readouterr()

        assert "Test Check" in captured.out

    def test_failure_result(self, capsys) -> None:
        """Test failed result output."""
        print_result("Test Check", False)
        captured = capsys.readouterr()

        assert "Test Check" in captured.out

    def test_with_details(self, capsys) -> None:
        """Test result with additional details."""
        print_result("Test Check", True, "Extra info here")
        captured = capsys.readouterr()

        assert "Test Check" in captured.out
        assert "Extra info here" in captured.out


# =============================================================================
# Docker Check Tests
# =============================================================================

class TestCheckDocker:
    """Tests for check_docker function."""

    @patch('health_check.subprocess.run')
    def test_docker_running(self, mock_run, capsys) -> None:
        """Test when Docker daemon is running."""
        # Mock docker info success
        mock_run.side_effect = [
            Mock(returncode=0),  # docker info
            Mock(returncode=0, stdout="running\n"),  # neo4j container
            Mock(returncode=0, stdout="running\n"),  # chromadb container
        ]

        result = check_docker()

        assert result is True
        assert mock_run.call_count == 3

    @patch('health_check.subprocess.run')
    def test_docker_not_running(self, mock_run, capsys) -> None:
        """Test when Docker daemon is not running."""
        mock_run.return_value = Mock(returncode=1)

        result = check_docker()

        assert result is False

    @patch('health_check.subprocess.run')
    def test_docker_not_installed(self, mock_run) -> None:
        """Test when Docker is not installed."""
        mock_run.side_effect = FileNotFoundError()

        result = check_docker()

        assert result is False

    @patch('health_check.subprocess.run')
    def test_docker_timeout(self, mock_run) -> None:
        """Test when Docker command times out."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=10)

        result = check_docker()

        assert result is False

    @patch('health_check.subprocess.run')
    def test_container_not_running(self, mock_run, capsys) -> None:
        """Test when container exists but not running."""
        mock_run.side_effect = [
            Mock(returncode=0),  # docker info OK
            Mock(returncode=0, stdout="exited\n"),  # neo4j stopped
            Mock(returncode=0, stdout="running\n"),  # chromadb OK
        ]

        result = check_docker()

        assert result is False


# =============================================================================
# Neo4j Check Tests
# =============================================================================

class TestCheckNeo4j:
    """Tests for check_neo4j function."""

    @patch('neo4j.GraphDatabase')
    def test_neo4j_healthy(self, mock_gd_class, capsys) -> None:
        """Test successful Neo4j connection."""
        # Setup mock driver
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        # Mock query results
        count_result = Mock()
        count_result.single.return_value = {"count": 42}

        self_result = Mock()
        self_result.single.return_value = {"s": {"name": "Self"}}

        mock_session.run.side_effect = [count_result, self_result]
        mock_driver.session.return_value = mock_session
        mock_gd_class.driver.return_value = mock_driver

        config = HealthCheckConfig()
        result = check_neo4j(config)

        assert result is True
        mock_driver.verify_connectivity.assert_called_once()

    @patch('neo4j.GraphDatabase')
    def test_neo4j_connection_failed(self, mock_gd_class, capsys) -> None:
        """Test Neo4j connection failure."""
        mock_gd_class.driver.side_effect = Exception("Connection refused")

        result = check_neo4j()

        assert result is False

    @patch('neo4j.GraphDatabase')
    def test_neo4j_empty_graph(self, mock_gd_class, capsys) -> None:
        """Test Neo4j with no nodes."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        count_result = Mock()
        count_result.single.return_value = {"count": 0}

        self_result = Mock()
        self_result.single.return_value = None

        mock_session.run.side_effect = [count_result, self_result]
        mock_driver.session.return_value = mock_session
        mock_gd_class.driver.return_value = mock_driver

        result = check_neo4j()

        assert result is True  # Still connects, just empty


# =============================================================================
# ChromaDB Check Tests
# =============================================================================

class TestCheckChromadb:
    """Tests for check_chromadb function."""

    def test_chromadb_healthy(self, capsys) -> None:
        """Test successful ChromaDB connection."""
        import chromadb
        with patch.object(chromadb, 'HttpClient') as mock_http_client:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.count.return_value = 10
            mock_collection.query.return_value = {
                'documents': [['test document']]
            }
            mock_collection.name = "ubik_semantic"

            mock_client.list_collections.return_value = [
                Mock(name="ubik_episodic"),
                Mock(name="ubik_semantic"),
            ]
            mock_client.get_collection.return_value = mock_collection
            mock_http_client.return_value = mock_client

            result = check_chromadb()

            assert result is True
            mock_client.heartbeat.assert_called_once()

    def test_chromadb_connection_failed(self, capsys) -> None:
        """Test ChromaDB connection failure."""
        import chromadb
        with patch.object(chromadb, 'HttpClient') as mock_http_client:
            mock_http_client.side_effect = Exception("Connection refused")

            result = check_chromadb()

            assert result is False

    def test_chromadb_missing_collections(self, capsys) -> None:
        """Test ChromaDB with missing collections."""
        import chromadb
        with patch.object(chromadb, 'HttpClient') as mock_http_client:
            mock_client = Mock()
            mock_client.list_collections.return_value = []  # No collections

            mock_http_client.return_value = mock_client

            result = check_chromadb()

            assert result is True  # Still connects


# =============================================================================
# MCP Server Check Tests
# =============================================================================

class TestCheckMcpServer:
    """Tests for check_mcp_server function."""

    def test_mcp_server_running(self, capsys) -> None:
        """Test MCP server is responding."""
        import httpx
        with patch.object(httpx, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 406  # Expected for MCP without SSE headers
            mock_get.return_value = mock_response

            result = check_mcp_server()

            assert result is True

    def test_mcp_server_not_running(self, capsys) -> None:
        """Test MCP server not responding."""
        import httpx
        with patch.object(httpx, 'get') as mock_get:
            mock_get.side_effect = Exception("ConnectError: Connection refused")

            result = check_mcp_server()

            assert result is False

    def test_mcp_server_various_status_codes(self) -> None:
        """Test various HTTP status codes are accepted."""
        import httpx
        for status_code in [200, 400, 404, 406]:
            with patch.object(httpx, 'get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_get.return_value = mock_response

                result = check_mcp_server()
                assert result is True, f"Status {status_code} should be accepted"


# =============================================================================
# Tailscale Check Tests
# =============================================================================

class TestCheckTailscale:
    """Tests for check_tailscale function."""

    @patch('health_check.os.path.exists')
    @patch('health_check.subprocess.run')
    def test_tailscale_connected(self, mock_run, mock_exists, capsys) -> None:
        """Test Tailscale is connected."""
        mock_exists.return_value = True  # macOS app exists

        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"Self": {"HostName": "mac", "Online": true}, "Peer": {}}'
        )

        result = check_tailscale()

        assert result is True

    @patch('health_check.os.path.exists')
    @patch('health_check.subprocess.run')
    def test_tailscale_not_running(self, mock_run, mock_exists) -> None:
        """Test Tailscale not running."""
        mock_exists.return_value = False
        mock_run.return_value = Mock(returncode=1, stdout="")

        result = check_tailscale()

        assert result is False

    @patch('health_check.os.path.exists')
    @patch('health_check.subprocess.run')
    def test_tailscale_not_installed(self, mock_run, mock_exists) -> None:
        """Test Tailscale not installed."""
        mock_exists.return_value = False
        mock_run.side_effect = FileNotFoundError()

        result = check_tailscale()

        assert result is False

    @patch('health_check.os.path.exists')
    @patch('health_check.subprocess.run')
    def test_tailscale_with_somatic_peer(self, mock_run, mock_exists, capsys) -> None:
        """Test Tailscale with somatic node as peer."""
        mock_exists.return_value = True

        status_json = '''{
            "Self": {"HostName": "mac", "Online": true},
            "Peer": {
                "abc123": {
                    "HostName": "adrian-wsl",
                    "TailscaleIPs": ["100.79.166.114"],
                    "Online": true
                }
            }
        }'''

        mock_run.return_value = Mock(returncode=0, stdout=status_json)

        result = check_tailscale()

        assert result is True
        captured = capsys.readouterr()
        assert "adrian-wsl" in captured.out


# =============================================================================
# Run All Checks Tests
# =============================================================================

class TestRunAllChecks:
    """Tests for run_all_checks function."""

    @patch('health_check.check_tailscale')
    @patch('health_check.check_mcp_server')
    @patch('health_check.check_chromadb')
    @patch('health_check.check_neo4j')
    @patch('health_check.check_docker')
    def test_all_checks_pass(
        self,
        mock_docker,
        mock_neo4j,
        mock_chromadb,
        mock_mcp,
        mock_tailscale,
        capsys
    ) -> None:
        """Test when all health checks pass."""
        mock_docker.return_value = True
        mock_neo4j.return_value = True
        mock_chromadb.return_value = True
        mock_mcp.return_value = True
        mock_tailscale.return_value = True

        all_passed, results = run_all_checks()

        assert all_passed is True
        assert results["docker"] is True
        assert results["neo4j"] is True
        assert results["chromadb"] is True
        assert results["mcp_server"] is True
        assert results["tailscale"] is True

    @patch('health_check.check_tailscale')
    @patch('health_check.check_mcp_server')
    @patch('health_check.check_chromadb')
    @patch('health_check.check_neo4j')
    @patch('health_check.check_docker')
    def test_some_checks_fail(
        self,
        mock_docker,
        mock_neo4j,
        mock_chromadb,
        mock_mcp,
        mock_tailscale,
        capsys
    ) -> None:
        """Test when some health checks fail."""
        mock_docker.return_value = True
        mock_neo4j.return_value = False  # Failed
        mock_chromadb.return_value = True
        mock_mcp.return_value = False  # Failed
        mock_tailscale.return_value = True

        all_passed, results = run_all_checks()

        assert all_passed is False
        assert results["docker"] is True
        assert results["neo4j"] is False
        assert results["chromadb"] is True
        assert results["mcp_server"] is False
        assert results["tailscale"] is True

    @patch('health_check.check_tailscale')
    @patch('health_check.check_mcp_server')
    @patch('health_check.check_chromadb')
    @patch('health_check.check_neo4j')
    @patch('health_check.check_docker')
    def test_uses_provided_config(
        self,
        mock_docker,
        mock_neo4j,
        mock_chromadb,
        mock_mcp,
        mock_tailscale
    ) -> None:
        """Test that custom config is passed to checks."""
        mock_docker.return_value = True
        mock_neo4j.return_value = True
        mock_chromadb.return_value = True
        mock_mcp.return_value = True
        mock_tailscale.return_value = True

        custom_config = HealthCheckConfig()
        custom_config.neo4j_uri = "bolt://custom:7687"

        run_all_checks(custom_config)

        # Verify config was passed to the checks
        mock_neo4j.assert_called_once_with(custom_config)
        mock_chromadb.assert_called_once_with(custom_config)
        mock_mcp.assert_called_once_with(custom_config)
