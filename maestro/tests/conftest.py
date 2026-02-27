"""
Maestro Test Configuration — Shared Fixtures

Provides lightweight configuration fixtures that construct AppConfig,
HippocampalConfig, and SomaticConfig objects from known test values
without touching environment variables or the filesystem.

All fixtures use pytest's function scope so each test starts clean.
"""

import pytest

from maestro.config import AppConfig, HippocampalConfig, MaestroConfig, SomaticConfig
from maestro.services.models import ClusterHealth, ServiceResult, ServiceStatus


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def hippocampal_cfg() -> HippocampalConfig:
    """HippocampalConfig populated with test values.

    Returns:
        A HippocampalConfig instance with deterministic test values.
    """
    return HippocampalConfig(
        tailscale_ip="127.0.0.1",
        tailscale_hostname="test-hippocampal",
        NEO4J_HTTP_PORT=7474,
        NEO4J_BOLT_PORT=7687,
        NEO4J_USER="neo4j",
        NEO4J_PASSWORD="test_password",
        CHROMADB_PORT=8001,
        CHROMADB_TOKEN="test_token",
        MCP_PORT=8080,
    )


@pytest.fixture()
def somatic_cfg() -> SomaticConfig:
    """SomaticConfig populated with test values.

    Returns:
        A SomaticConfig instance with deterministic test values.
    """
    return SomaticConfig(
        tailscale_ip="127.0.0.2",
        tailscale_hostname="test-somatic",
        VLLM_PORT=8002,
        VLLM_MODEL_PATH="/tmp/test-model",
    )


@pytest.fixture()
def maestro_cfg() -> MaestroConfig:
    """MaestroConfig with test defaults.

    Returns:
        A MaestroConfig instance.
    """
    return MaestroConfig(
        MAESTRO_LOG_DIR="/tmp/maestro-test-logs",
        MAESTRO_CHECK_INTERVAL_S=60,
        MAESTRO_LOG_LEVEL="DEBUG",
    )


@pytest.fixture()
def app_config(
    hippocampal_cfg: HippocampalConfig,
    somatic_cfg: SomaticConfig,
    maestro_cfg: MaestroConfig,
    tmp_path,
) -> AppConfig:
    """AppConfig assembled from test sub-configs.

    Args:
        hippocampal_cfg: Injected hippocampal fixture.
        somatic_cfg: Injected somatic fixture.
        maestro_cfg: Injected maestro fixture.
        tmp_path: pytest built-in temporary directory.

    Returns:
        A fully assembled AppConfig pointing at tmp_path.
    """
    return AppConfig(
        ubik_root=tmp_path,
        maestro=maestro_cfg,
        hippocampal=hippocampal_cfg,
        somatic=somatic_cfg,
    )


# ---------------------------------------------------------------------------
# ServiceResult factories
# ---------------------------------------------------------------------------

@pytest.fixture()
def healthy_result() -> ServiceResult:
    """A HEALTHY ServiceResult for use in tests.

    Returns:
        ServiceResult with HEALTHY status and sample details.
    """
    return ServiceResult(
        service_name="test_service",
        status=ServiceStatus.HEALTHY,
        latency_ms=5.0,
        details={"info": "all good"},
    )


@pytest.fixture()
def degraded_result() -> ServiceResult:
    """A DEGRADED ServiceResult for use in tests.

    Returns:
        ServiceResult with DEGRADED status.
    """
    return ServiceResult(
        service_name="test_service",
        status=ServiceStatus.DEGRADED,
        latency_ms=12.0,
        details={"missing": ["ubik_semantic"]},
        error="Missing collections",
    )


@pytest.fixture()
def unhealthy_result() -> ServiceResult:
    """An UNHEALTHY ServiceResult for use in tests.

    Returns:
        ServiceResult with UNHEALTHY status and an error message.
    """
    return ServiceResult(
        service_name="test_service",
        status=ServiceStatus.UNHEALTHY,
        error="Connection refused",
    )
