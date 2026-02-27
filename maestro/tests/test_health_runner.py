"""
Tests for maestro.services.health_runner

Uses unittest.mock to stub out the six individual check functions so
the runner logic can be tested without any real services.

Tests cover:
    - All healthy → ClusterHealth.overall_status == HEALTHY
    - Mixed results → correct aggregation
    - Single timeout → UNHEALTHY result injected for that service
    - Unexpected exception → UNHEALTHY result injected
    - All six service names appear in the result
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from maestro.services.health_runner import run_all_checks
from maestro.services.models import ClusterHealth, ServiceResult, ServiceStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(name: str, status: ServiceStatus) -> ServiceResult:
    return ServiceResult(service_name=name, status=status, latency_ms=1.0)


_ALL_HEALTHY = {
    "neo4j": _result("neo4j", ServiceStatus.HEALTHY),
    "chromadb": _result("chromadb", ServiceStatus.HEALTHY),
    "mcp": _result("mcp", ServiceStatus.HEALTHY),
    "vllm": _result("vllm", ServiceStatus.HEALTHY),
    "tailscale": _result("tailscale", ServiceStatus.HEALTHY),
    "docker": _result("docker", ServiceStatus.HEALTHY),
}

_PATCH_BASE = "maestro.services.health_runner"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_checks(app_config):
    """Patch all six check functions with AsyncMock returning healthy results.

    Yields the dict of mocks keyed by service name so individual tests can
    override specific mocks.

    Args:
        app_config: AppConfig fixture from conftest.

    Yields:
        dict mapping service name → AsyncMock.
    """
    mocks = {
        name: AsyncMock(return_value=_ALL_HEALTHY[name])
        for name in _ALL_HEALTHY
    }

    with (
        patch(f"{_PATCH_BASE}.check_neo4j", mocks["neo4j"]),
        patch(f"{_PATCH_BASE}.check_chromadb", mocks["chromadb"]),
        patch(f"{_PATCH_BASE}.check_mcp", mocks["mcp"]),
        patch(f"{_PATCH_BASE}.check_vllm", mocks["vllm"]),
        patch(f"{_PATCH_BASE}.check_tailscale", mocks["tailscale"]),
        patch(f"{_PATCH_BASE}.check_docker", mocks["docker"]),
    ):
        yield mocks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunAllChecks:
    @pytest.mark.asyncio
    async def test_all_healthy_returns_healthy_cluster(
        self, mock_checks, app_config
    ):
        cluster = await run_all_checks(app_config)
        assert isinstance(cluster, ClusterHealth)
        assert cluster.overall_status == ServiceStatus.HEALTHY
        assert cluster.is_healthy is True

    @pytest.mark.asyncio
    async def test_all_six_services_present(self, mock_checks, app_config):
        cluster = await run_all_checks(app_config)
        assert set(cluster.services.keys()) == {
            "neo4j", "chromadb", "mcp", "vllm", "tailscale", "docker"
        }

    @pytest.mark.asyncio
    async def test_one_unhealthy_service_degrades_cluster(
        self, mock_checks, app_config
    ):
        mock_checks["vllm"].return_value = _result("vllm", ServiceStatus.UNHEALTHY)
        cluster = await run_all_checks(app_config)
        assert cluster.overall_status == ServiceStatus.UNHEALTHY
        assert "vllm" in cluster.unhealthy_services

    @pytest.mark.asyncio
    async def test_one_degraded_service_degrades_cluster(
        self, mock_checks, app_config
    ):
        mock_checks["neo4j"].return_value = _result(
            "neo4j", ServiceStatus.DEGRADED
        )
        cluster = await run_all_checks(app_config)
        assert cluster.overall_status == ServiceStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_timeout_on_one_check_produces_unhealthy_result(
        self, mock_checks, app_config
    ):
        """A check that never resolves should be caught by wait_for."""
        async def _hang(*args, **kwargs):
            await asyncio.sleep(9999)

        mock_checks["tailscale"].side_effect = None
        mock_checks["tailscale"].return_value = None
        # Replace with a coroutine that hangs
        mock_checks["tailscale"] = AsyncMock(side_effect=_hang)

        with patch(f"{_PATCH_BASE}.check_tailscale", mock_checks["tailscale"]):
            cluster = await run_all_checks(app_config, timeout=0.05)

        assert cluster.services["tailscale"].status == ServiceStatus.UNHEALTHY
        assert "timed out" in (cluster.services["tailscale"].error or "").lower()

    @pytest.mark.asyncio
    async def test_unexpected_exception_produces_unhealthy_result(
        self, mock_checks, app_config
    ):
        mock_checks["docker"].side_effect = RuntimeError("kernel panic")

        with patch(f"{_PATCH_BASE}.check_docker", mock_checks["docker"]):
            cluster = await run_all_checks(app_config)

        result = cluster.services["docker"]
        assert result.status == ServiceStatus.UNHEALTHY
        assert "kernel panic" in (result.error or "")

    @pytest.mark.asyncio
    async def test_checked_at_is_recent_utc(self, mock_checks, app_config):
        before = datetime.now(timezone.utc)
        cluster = await run_all_checks(app_config)
        after = datetime.now(timezone.utc)
        assert before <= cluster.checked_at <= after

    @pytest.mark.asyncio
    async def test_healthy_services_list_excludes_failing(
        self, mock_checks, app_config
    ):
        mock_checks["mcp"].return_value = _result(
            "mcp", ServiceStatus.UNHEALTHY
        )
        cluster = await run_all_checks(app_config)
        assert "mcp" not in cluster.healthy_services
        assert "mcp" in cluster.unhealthy_services

    @pytest.mark.asyncio
    async def test_to_json_round_trips(self, mock_checks, app_config):
        import json
        cluster = await run_all_checks(app_config)
        parsed = json.loads(cluster.to_json())
        assert parsed["overall_status"] == "healthy"
        assert parsed["total_count"] == 6
