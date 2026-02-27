#!/usr/bin/env python3
"""
Tests for maestro/services/{neo4j,chromadb,mcp_server,vllm}_service.py
and maestro/services/__init__.py::ServiceRegistry.

All external I/O (httpx, _run_proc, asyncio.create_subprocess_exec,
_kill_port) is mocked.  No network or Docker daemon required.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from maestro.platform_detect import NodeType
from maestro.services.base import ProbeResult

# ---------------------------------------------------------------------------
# Shared node-identity mocks
# ---------------------------------------------------------------------------

def _hippo_identity():
    m = MagicMock()
    m.node_type = NodeType.HIPPOCAMPAL
    return m


def _somatic_identity():
    m = MagicMock()
    m.node_type = NodeType.SOMATIC
    return m
from maestro.services.chromadb_service import ChromaDbService
from maestro.services.docker_service import DockerService
from maestro.services.mcp_server_service import McpServerService
from maestro.services.neo4j_service import Neo4jService
from maestro.services.vllm_service import VllmService
from maestro.services import ServiceRegistry


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

_FAKE_ROOT = Path("/fake/ubik")


def _http_client_mock(status_code: int = 200, raise_exc=None):
    """Build a mock httpx.AsyncClient context manager.

    Args:
        status_code: HTTP status code to return.
        raise_exc: If given, client.get() raises this exception instead.

    Returns:
        Mock class that can be used as ``patch(..., _http_client_mock(...))``.
    """
    mock_cls = MagicMock()
    instance = AsyncMock()
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    if raise_exc is not None:
        instance.get.side_effect = raise_exc
    else:
        resp = MagicMock()
        resp.status_code = status_code
        instance.get = AsyncMock(return_value=resp)
    return mock_cls


# ---------------------------------------------------------------------------
# Neo4jService
# ---------------------------------------------------------------------------

class TestNeo4jServiceProperties:
    def test_name(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        assert svc.name == "neo4j"

    def test_node(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        assert svc.node == NodeType.HIPPOCAMPAL

    def test_ports(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        assert 7474 in svc.ports
        assert 7687 in svc.ports

    def test_depends_on(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        assert svc.depends_on == ["docker"]

    def test_custom_ports(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT, http_port=17474, bolt_port=17687)
        assert svc.ports == [17474, 17687]


class TestNeo4jServiceProbe:
    @pytest.mark.asyncio
    async def test_probe_healthy_200(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.neo4j_service.httpx.AsyncClient",
                   _http_client_mock(200)):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is True
        assert result.error is None
        assert result.details["http_status"] == 200
        assert "127.0.0.1" in result.details["url"]

    @pytest.mark.asyncio
    async def test_probe_healthy_401(self):
        # 401 means Neo4j is up but auth is required — still healthy (process is up)
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.neo4j_service.httpx.AsyncClient",
                   _http_client_mock(401)):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is True

    @pytest.mark.asyncio
    async def test_probe_unhealthy_500(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.neo4j_service.httpx.AsyncClient",
                   _http_client_mock(500)):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is False
        assert "500" in result.error

    @pytest.mark.asyncio
    async def test_probe_connection_error(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.neo4j_service.httpx.AsyncClient",
                   _http_client_mock(raise_exc=httpx.ConnectError("refused"))):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_probe_result_type(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.neo4j_service.httpx.AsyncClient",
                   _http_client_mock(200)):
            result = await svc.probe("127.0.0.1")
        assert isinstance(result, ProbeResult)
        assert result.name == "neo4j"
        assert result.node == NodeType.HIPPOCAMPAL
        assert isinstance(result.latency_ms, float)


class TestNeo4jServiceLifecycle:
    @pytest.mark.asyncio
    async def test_start_success(self, tmp_path):
        (tmp_path / "docker-compose.yml").write_text("services: {}")
        svc = Neo4jService(ubik_root=tmp_path, max_wait_s=5.0)
        healthy = ProbeResult(name="neo4j", node=NodeType.HIPPOCAMPAL,
                              healthy=True, latency_ms=5.0)
        with patch("maestro.services.neo4j_service.detect_node",
                   return_value=_hippo_identity()), \
             patch("maestro.services.neo4j_service._run_proc",
                   new_callable=AsyncMock, return_value=(0, "", "")) as mock_proc, \
             patch.object(svc, "probe_with_timeout",
                          new_callable=AsyncMock, return_value=healthy):
            result = await svc.start(tmp_path)
        assert result is True
        mock_proc.assert_called_once()
        call_args = mock_proc.call_args[0]
        assert "docker" in call_args
        assert "compose" in call_args
        assert "neo4j" in call_args

    @pytest.mark.asyncio
    async def test_start_compose_failure(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.neo4j_service._run_proc",
                   new_callable=AsyncMock,
                   return_value=(1, "", "error starting")):
            result = await svc.start(_FAKE_ROOT)
        assert result is False

    @pytest.mark.asyncio
    async def test_start_exception(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.neo4j_service._run_proc",
                   new_callable=AsyncMock,
                   side_effect=FileNotFoundError("docker not found")):
            result = await svc.start(_FAKE_ROOT)
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_success(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.neo4j_service._run_proc",
                   new_callable=AsyncMock, return_value=(0, "", "")) as mock_proc:
            result = await svc.stop()
        assert result is True
        call_args = mock_proc.call_args[0]
        assert "stop" in call_args
        assert "neo4j" in call_args

    @pytest.mark.asyncio
    async def test_stop_failure(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.neo4j_service._run_proc",
                   new_callable=AsyncMock,
                   return_value=(1, "", "compose error")):
            result = await svc.stop()
        assert result is False


# ---------------------------------------------------------------------------
# ChromaDbService
# ---------------------------------------------------------------------------

class TestChromaDbServiceProperties:
    def test_name(self):
        svc = ChromaDbService(ubik_root=_FAKE_ROOT)
        assert svc.name == "chromadb"

    def test_node(self):
        svc = ChromaDbService(ubik_root=_FAKE_ROOT)
        assert svc.node == NodeType.HIPPOCAMPAL

    def test_ports(self):
        svc = ChromaDbService(ubik_root=_FAKE_ROOT)
        assert 8001 in svc.ports

    def test_depends_on(self):
        svc = ChromaDbService(ubik_root=_FAKE_ROOT)
        assert svc.depends_on == ["docker"]

    def test_token_stored(self):
        svc = ChromaDbService(ubik_root=_FAKE_ROOT, token="secret")
        assert svc._token == "secret"


class TestChromaDbServiceProbe:
    @pytest.mark.asyncio
    async def test_probe_v2_healthy(self):
        svc = ChromaDbService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.chromadb_service.httpx.AsyncClient",
                   _http_client_mock(200)):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is True
        assert result.details["api_version"] == "v2"

    @pytest.mark.asyncio
    async def test_probe_v1_fallback(self):
        # v2 returns 404, v1 returns 200
        svc = ChromaDbService(ubik_root=_FAKE_ROOT)
        mock_cls = MagicMock()
        instance = AsyncMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=instance)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        resp_404 = MagicMock(); resp_404.status_code = 404
        resp_200 = MagicMock(); resp_200.status_code = 200
        instance.get = AsyncMock(side_effect=[resp_404, resp_200])

        with patch("maestro.services.chromadb_service.httpx.AsyncClient", mock_cls):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is True
        assert result.details["api_version"] == "v1"

    @pytest.mark.asyncio
    async def test_probe_unhealthy_503(self):
        svc = ChromaDbService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.chromadb_service.httpx.AsyncClient",
                   _http_client_mock(503)):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is False
        assert "503" in result.error

    @pytest.mark.asyncio
    async def test_probe_connection_refused(self):
        svc = ChromaDbService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.chromadb_service.httpx.AsyncClient",
                   _http_client_mock(raise_exc=httpx.ConnectError("refused"))):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is False

    @pytest.mark.asyncio
    async def test_probe_includes_url(self):
        svc = ChromaDbService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.chromadb_service.httpx.AsyncClient",
                   _http_client_mock(200)):
            result = await svc.probe("10.0.0.1")
        assert "10.0.0.1" in result.details["url"]


class TestChromaDbServiceLifecycle:
    @pytest.mark.asyncio
    async def test_start_success(self, tmp_path):
        (tmp_path / "docker-compose.yml").write_text("services: {}")
        svc = ChromaDbService(ubik_root=tmp_path, max_wait_s=5.0)
        healthy = ProbeResult(name="chromadb", node=NodeType.HIPPOCAMPAL,
                              healthy=True, latency_ms=5.0)
        with patch("maestro.services.chromadb_service.detect_node",
                   return_value=_hippo_identity()), \
             patch("maestro.services.chromadb_service._run_proc",
                   new_callable=AsyncMock, return_value=(0, "", "")) as mock_proc, \
             patch.object(svc, "probe_with_timeout",
                          new_callable=AsyncMock, return_value=healthy):
            result = await svc.start(tmp_path)
        assert result is True
        call_args = mock_proc.call_args[0]
        assert "chromadb" in call_args

    @pytest.mark.asyncio
    async def test_start_failure_nonzero_rc(self):
        svc = ChromaDbService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.chromadb_service._run_proc",
                   new_callable=AsyncMock,
                   return_value=(1, "", "some error")):
            result = await svc.start(_FAKE_ROOT)
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_success(self):
        svc = ChromaDbService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.chromadb_service._run_proc",
                   new_callable=AsyncMock, return_value=(0, "", "")) as mock_proc:
            result = await svc.stop()
        assert result is True
        call_args = mock_proc.call_args[0]
        assert "stop" in call_args
        assert "chromadb" in call_args

    @pytest.mark.asyncio
    async def test_stop_exception(self):
        svc = ChromaDbService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.chromadb_service._run_proc",
                   new_callable=AsyncMock,
                   side_effect=OSError("docker gone")):
            result = await svc.stop()
        assert result is False


# ---------------------------------------------------------------------------
# McpServerService
# ---------------------------------------------------------------------------

class TestMcpServerServiceProperties:
    def test_name(self):
        svc = McpServerService(ubik_root=_FAKE_ROOT)
        assert svc.name == "mcp"

    def test_node(self):
        svc = McpServerService(ubik_root=_FAKE_ROOT)
        assert svc.node == NodeType.HIPPOCAMPAL

    def test_ports(self):
        svc = McpServerService(ubik_root=_FAKE_ROOT)
        assert 8080 in svc.ports

    def test_depends_on(self):
        svc = McpServerService(ubik_root=_FAKE_ROOT)
        assert "neo4j" in svc.depends_on
        assert "chromadb" in svc.depends_on


class TestMcpServerServiceProbe:
    @pytest.mark.asyncio
    async def test_probe_200_healthy(self):
        svc = McpServerService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.mcp_server_service.httpx.AsyncClient",
                   _http_client_mock(200)):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is True

    @pytest.mark.asyncio
    async def test_probe_406_healthy(self):
        # 406 is the expected FastMCP liveness signal
        svc = McpServerService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.mcp_server_service.httpx.AsyncClient",
                   _http_client_mock(406)):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is True

    @pytest.mark.asyncio
    async def test_probe_404_healthy(self):
        svc = McpServerService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.mcp_server_service.httpx.AsyncClient",
                   _http_client_mock(404)):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is True

    @pytest.mark.asyncio
    async def test_probe_500_unhealthy(self):
        svc = McpServerService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.mcp_server_service.httpx.AsyncClient",
                   _http_client_mock(500)):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is False
        assert "500" in result.error

    @pytest.mark.asyncio
    async def test_probe_connect_error(self):
        svc = McpServerService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.mcp_server_service.httpx.AsyncClient",
                   _http_client_mock(raise_exc=httpx.ConnectError("refused"))):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is False

    @pytest.mark.asyncio
    async def test_probe_has_latency(self):
        svc = McpServerService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.mcp_server_service.httpx.AsyncClient",
                   _http_client_mock(200)):
            result = await svc.probe("127.0.0.1")
        assert isinstance(result.latency_ms, float)


class TestMcpServerServiceLifecycle:
    @pytest.mark.asyncio
    async def test_start_launches_detached(self, tmp_path):
        hippo_path = tmp_path / "hippocampal"
        hippo_path.mkdir()
        script = hippo_path / "run_mcp.sh"
        script.write_text("#!/bin/bash\n"); script.chmod(0o755)

        svc = McpServerService(ubik_root=tmp_path, max_wait_s=5.0)
        healthy = ProbeResult(name="mcp", node=NodeType.HIPPOCAMPAL,
                              healthy=True, latency_ms=5.0)
        mock_proc = AsyncMock()
        with patch("maestro.services.mcp_server_service.detect_node",
                   return_value=_hippo_identity()), \
             patch("maestro.services.mcp_server_service.asyncio.create_subprocess_exec",
                   new_callable=AsyncMock, return_value=mock_proc) as mock_exec, \
             patch.object(svc, "probe_with_timeout",
                          new_callable=AsyncMock, return_value=healthy):
            result = await svc.start(tmp_path)
        assert result is True
        call_args = mock_exec.call_args
        args = call_args[0]
        assert any("run_mcp.sh" in str(a) for a in args)
        assert "start" in args
        kwargs = call_args[1]
        assert kwargs.get("start_new_session") is True

    @pytest.mark.asyncio
    async def test_start_exception(self):
        svc = McpServerService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.mcp_server_service.asyncio.create_subprocess_exec",
                   side_effect=FileNotFoundError("not found")):
            result = await svc.start(_FAKE_ROOT)
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_calls_kill_port(self):
        svc = McpServerService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.mcp_server_service._kill_port",
                   new_callable=AsyncMock, return_value=True) as mock_kill:
            result = await svc.stop()
        assert result is True
        mock_kill.assert_called_once_with(8080)

    @pytest.mark.asyncio
    async def test_stop_no_process(self):
        svc = McpServerService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.mcp_server_service._kill_port",
                   new_callable=AsyncMock, return_value=False):
            result = await svc.stop()
        assert result is False


# ---------------------------------------------------------------------------
# VllmService
# ---------------------------------------------------------------------------

class TestVllmServiceProperties:
    def test_name(self):
        svc = VllmService(model_path="/fake/model")
        assert svc.name == "vllm"

    def test_node(self):
        svc = VllmService(model_path="/fake/model")
        assert svc.node == NodeType.SOMATIC

    def test_ports(self):
        svc = VllmService(model_path="/fake/model")
        assert 8002 in svc.ports

    def test_depends_on_empty(self):
        svc = VllmService(model_path="/fake/model")
        assert svc.depends_on == []

    def test_custom_port(self):
        svc = VllmService(port=19002, model_path="/fake/model")
        assert svc.ports == [19002]


class TestVllmServiceProbe:
    @pytest.mark.asyncio
    async def test_probe_200_healthy(self):
        svc = VllmService(model_path="/fake/model")
        with patch("maestro.services.vllm_service.httpx.AsyncClient",
                   _http_client_mock(200)):
            result = await svc.probe("127.0.0.2")
        assert result.healthy is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_probe_503_unhealthy(self):
        svc = VllmService(model_path="/fake/model")
        with patch("maestro.services.vllm_service.httpx.AsyncClient",
                   _http_client_mock(503)):
            result = await svc.probe("127.0.0.2")
        assert result.healthy is False
        assert "503" in result.error

    @pytest.mark.asyncio
    async def test_probe_connect_error(self):
        svc = VllmService(model_path="/fake/model")
        with patch("maestro.services.vllm_service.httpx.AsyncClient",
                   _http_client_mock(raise_exc=httpx.ConnectError("refused"))):
            result = await svc.probe("127.0.0.2")
        assert result.healthy is False

    @pytest.mark.asyncio
    async def test_probe_url_includes_health(self):
        svc = VllmService(model_path="/fake/model")
        with patch("maestro.services.vllm_service.httpx.AsyncClient",
                   _http_client_mock(200)):
            result = await svc.probe("10.0.0.2")
        assert "/health" in result.details["url"]

    @pytest.mark.asyncio
    async def test_probe_latency_set(self):
        svc = VllmService(model_path="/fake/model")
        with patch("maestro.services.vllm_service.httpx.AsyncClient",
                   _http_client_mock(200)):
            result = await svc.probe("127.0.0.2")
        assert isinstance(result.latency_ms, float)


class TestVllmServiceLifecycle:
    @pytest.mark.asyncio
    async def test_start_uses_conda_run(self):
        svc = VllmService(model_path="/fake/model", max_wait_s=5.0)
        healthy = ProbeResult(name="vllm", node=NodeType.SOMATIC,
                              healthy=True, latency_ms=5.0)
        mock_proc = AsyncMock()
        with patch("maestro.services.vllm_service.detect_node",
                   return_value=_somatic_identity()), \
             patch("maestro.services.vllm_service.asyncio.create_subprocess_exec",
                   new_callable=AsyncMock, return_value=mock_proc) as mock_exec, \
             patch.object(svc, "probe_with_timeout",
                          new_callable=AsyncMock, return_value=healthy):
            result = await svc.start(_FAKE_ROOT)
        assert result is True
        args = mock_exec.call_args[0]
        assert "conda" in args
        assert "run" in args
        assert "vllm" in args
        assert "serve" in args
        assert "/fake/model" in args
        assert mock_exec.call_args[1].get("start_new_session") is True

    @pytest.mark.asyncio
    async def test_start_exception(self):
        svc = VllmService(model_path="/fake/model")
        with patch("maestro.services.vllm_service.asyncio.create_subprocess_exec",
                   side_effect=FileNotFoundError("conda not found")):
            result = await svc.start(_FAKE_ROOT)
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_calls_kill_port(self):
        svc = VllmService(model_path="/fake/model")
        with patch("maestro.services.vllm_service._kill_port",
                   new_callable=AsyncMock, return_value=True) as mock_kill:
            result = await svc.stop()
        assert result is True
        mock_kill.assert_called_once_with(8002)

    @pytest.mark.asyncio
    async def test_stop_no_process_returns_false(self):
        svc = VllmService(model_path="/fake/model")
        with patch("maestro.services.vllm_service._kill_port",
                   new_callable=AsyncMock, return_value=False):
            result = await svc.stop()
        assert result is False


# ---------------------------------------------------------------------------
# ServiceRegistry
# ---------------------------------------------------------------------------

def _make_registry(tmp_path):
    """Create a ServiceRegistry using the test AppConfig fixture data."""
    from maestro.config import AppConfig, HippocampalConfig, MaestroConfig, SomaticConfig
    cfg = AppConfig(
        ubik_root=tmp_path,
        maestro=MaestroConfig(),
        hippocampal=HippocampalConfig(
            tailscale_ip="127.0.0.1",
            NEO4J_HTTP_PORT=7474,
            NEO4J_BOLT_PORT=7687,
            NEO4J_USER="neo4j",
            CHROMADB_PORT=8001,
            CHROMADB_TOKEN="tok",
            MCP_PORT=8080,
        ),
        somatic=SomaticConfig(
            tailscale_ip="127.0.0.2",
            VLLM_PORT=8002,
            VLLM_MODEL_PATH="/fake/model",
        ),
    )
    return ServiceRegistry(cfg=cfg)


class TestServiceRegistry:
    def test_get_all_returns_five_services(self, tmp_path):
        registry = _make_registry(tmp_path)
        assert len(registry.get_all()) == 5

    def test_all_service_names_present(self, tmp_path):
        registry = _make_registry(tmp_path)
        names = {s.name for s in registry.get_all()}
        assert names == {"docker", "neo4j", "chromadb", "mcp", "vllm"}

    def test_get_services_for_hippocampal(self, tmp_path):
        registry = _make_registry(tmp_path)
        hippo = registry.get_services_for_node(NodeType.HIPPOCAMPAL)
        names = {s.name for s in hippo}
        assert "docker" in names
        assert "neo4j" in names
        assert "chromadb" in names
        assert "mcp" in names
        assert "vllm" not in names

    def test_get_services_for_somatic(self, tmp_path):
        registry = _make_registry(tmp_path)
        somatic = registry.get_services_for_node(NodeType.SOMATIC)
        assert len(somatic) == 1
        assert somatic[0].name == "vllm"

    def test_get_startup_order_length(self, tmp_path):
        registry = _make_registry(tmp_path)
        order = registry.get_startup_order()
        assert len(order) == 5

    def test_get_startup_order_docker_first(self, tmp_path):
        registry = _make_registry(tmp_path)
        order = registry.get_startup_order()
        names = [s.name for s in order]
        assert names.index("docker") < names.index("neo4j")
        assert names.index("docker") < names.index("chromadb")

    def test_get_startup_order_neo4j_before_mcp(self, tmp_path):
        registry = _make_registry(tmp_path)
        order = registry.get_startup_order()
        names = [s.name for s in order]
        assert names.index("neo4j") < names.index("mcp")

    def test_get_startup_order_chromadb_before_mcp(self, tmp_path):
        registry = _make_registry(tmp_path)
        order = registry.get_startup_order()
        names = [s.name for s in order]
        assert names.index("chromadb") < names.index("mcp")

    def test_get_startup_order_no_cycle(self, tmp_path):
        # Should complete without raising ValueError
        registry = _make_registry(tmp_path)
        order = registry.get_startup_order()
        assert len(order) == 5

    def test_register_adds_service(self, tmp_path):
        registry = _make_registry(tmp_path)
        initial_count = len(registry.get_all())

        class FakeService(from_base := __import__(
            "maestro.services.base", fromlist=["UbikService"]
        ).UbikService):
            @property
            def name(self): return "fake"
            @property
            def node(self): return NodeType.HIPPOCAMPAL
            @property
            def ports(self): return []
            @property
            def depends_on(self): return []
            async def probe(self, host): ...
            async def start(self, ubik_root): return True
            async def stop(self): return True

        registry.register(FakeService())
        assert len(registry.get_all()) == initial_count + 1

    def test_circular_dependency_raises(self, tmp_path):
        from maestro.services.base import UbikService, ProbeResult
        from maestro.config import AppConfig, HippocampalConfig, MaestroConfig, SomaticConfig

        cfg = AppConfig(
            ubik_root=tmp_path,
            maestro=MaestroConfig(),
            hippocampal=HippocampalConfig(NEO4J_HTTP_PORT=7474, NEO4J_BOLT_PORT=7687,
                                          NEO4J_USER="neo4j", CHROMADB_PORT=8001,
                                          MCP_PORT=8080),
            somatic=SomaticConfig(VLLM_PORT=8002, VLLM_MODEL_PATH="/m"),
        )

        class SvcA(UbikService):
            @property
            def name(self): return "a"
            @property
            def node(self): return NodeType.HIPPOCAMPAL
            @property
            def ports(self): return []
            @property
            def depends_on(self): return ["b"]
            async def probe(self, h): ...
            async def start(self, r): return True
            async def stop(self): return True

        class SvcB(UbikService):
            @property
            def name(self): return "b"
            @property
            def node(self): return NodeType.HIPPOCAMPAL
            @property
            def ports(self): return []
            @property
            def depends_on(self): return ["a"]  # circular!
            async def probe(self, h): ...
            async def start(self, r): return True
            async def stop(self): return True

        registry = ServiceRegistry.__new__(ServiceRegistry)
        registry._cfg = cfg
        registry._services = [SvcA(), SvcB()]

        with pytest.raises(ValueError, match="Circular dependency"):
            registry.get_startup_order()


# ---------------------------------------------------------------------------
# probe_with_timeout integration (inherited from UbikService)
# ---------------------------------------------------------------------------

class TestProbeWithTimeout:
    @pytest.mark.asyncio
    async def test_neo4j_probe_with_timeout_healthy(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)
        with patch("maestro.services.neo4j_service.httpx.AsyncClient",
                   _http_client_mock(200)):
            result = await svc.probe_with_timeout("127.0.0.1", timeout=5.0)
        assert result.healthy is True
        assert isinstance(result.latency_ms, float)

    @pytest.mark.asyncio
    async def test_probe_with_timeout_on_slow_service(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT)

        async def _slow_probe(host):
            await asyncio.sleep(10)

        with patch.object(svc, "probe", _slow_probe):
            result = await svc.probe_with_timeout("127.0.0.1", timeout=0.01)
        assert result.healthy is False
        assert "timed out" in result.error.lower()
