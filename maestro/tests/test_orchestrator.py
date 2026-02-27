#!/usr/bin/env python3
"""
Tests for maestro/orchestrator.py and the enhanced start() methods in
the service classes (pre-flight checks + health-wait loop).

All I/O is mocked.  No network, Docker, or filesystem access required.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from maestro.config import AppConfig, HippocampalConfig, MaestroConfig, SomaticConfig
from maestro.orchestrator import Orchestrator
from maestro.platform_detect import NodeIdentity, NodeType
from maestro.services import ServiceRegistry
from maestro.services.base import ProbeResult
from maestro.services.chromadb_service import ChromaDbService
from maestro.services.docker_service import DockerService
from maestro.services.mcp_server_service import McpServerService
from maestro.services.neo4j_service import Neo4jService
from maestro.services.vllm_service import VllmService


# ---------------------------------------------------------------------------
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------

_FAKE_ROOT = Path("/fake/ubik")


def _make_app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        ubik_root=tmp_path,
        maestro=MaestroConfig(),
        hippocampal=HippocampalConfig(
            tailscale_ip="100.0.0.1",
            NEO4J_HTTP_PORT=7474,
            NEO4J_BOLT_PORT=7687,
            NEO4J_USER="neo4j",
            CHROMADB_PORT=8001,
            CHROMADB_TOKEN="tok",
            MCP_PORT=8080,
        ),
        somatic=SomaticConfig(
            tailscale_ip="100.0.0.2",
            VLLM_PORT=8002,
            VLLM_MODEL_PATH="/fake/model",
        ),
    )


def _hippocampal_identity() -> NodeIdentity:
    return NodeIdentity(
        node_type=NodeType.HIPPOCAMPAL,
        hostname="mac.lan",
        platform="darwin",
        ubik_root=_FAKE_ROOT,
        is_wsl=False,
        tailscale_ip="100.0.0.1",
        python_venv_path=None,
        python_activate_cmd=None,
    )


def _somatic_identity() -> NodeIdentity:
    return NodeIdentity(
        node_type=NodeType.SOMATIC,
        hostname="adrian",
        platform="linux",
        ubik_root=_FAKE_ROOT,
        is_wsl=True,
        tailscale_ip="100.0.0.2",
        python_venv_path=None,
        python_activate_cmd=None,
    )


def _healthy(name: str, node: NodeType) -> ProbeResult:
    return ProbeResult(name=name, node=node, healthy=True, latency_ms=5.0)


def _unhealthy(name: str, node: NodeType, error: str = "down") -> ProbeResult:
    return ProbeResult(name=name, node=node, healthy=False, latency_ms=5.0, error=error)


# ---------------------------------------------------------------------------
# Enhanced start() — pre-flight node checks
# ---------------------------------------------------------------------------

class TestStartNodeCheck:
    """All services must refuse to start on the wrong node."""

    @pytest.mark.asyncio
    async def test_docker_refuses_somatic_node(self):
        svc = DockerService(max_wait_s=0.1)
        somatic_id = NodeIdentity(
            node_type=NodeType.SOMATIC, hostname="adrian", platform="linux",
            ubik_root=_FAKE_ROOT, is_wsl=True, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        with patch("maestro.services.docker_service.detect_node", return_value=somatic_id):
            result = await svc.start(_FAKE_ROOT)
        assert result is False

    @pytest.mark.asyncio
    async def test_neo4j_refuses_somatic_node(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT, max_wait_s=0.1)
        somatic_id = NodeIdentity(
            node_type=NodeType.SOMATIC, hostname="adrian", platform="linux",
            ubik_root=_FAKE_ROOT, is_wsl=True, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        with patch("maestro.services.neo4j_service.detect_node", return_value=somatic_id):
            result = await svc.start(_FAKE_ROOT)
        assert result is False

    @pytest.mark.asyncio
    async def test_chromadb_refuses_somatic_node(self):
        svc = ChromaDbService(ubik_root=_FAKE_ROOT, max_wait_s=0.1)
        somatic_id = NodeIdentity(
            node_type=NodeType.SOMATIC, hostname="adrian", platform="linux",
            ubik_root=_FAKE_ROOT, is_wsl=True, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        with patch("maestro.services.chromadb_service.detect_node", return_value=somatic_id):
            result = await svc.start(_FAKE_ROOT)
        assert result is False

    @pytest.mark.asyncio
    async def test_mcp_refuses_somatic_node(self):
        svc = McpServerService(ubik_root=_FAKE_ROOT, max_wait_s=0.1)
        somatic_id = NodeIdentity(
            node_type=NodeType.SOMATIC, hostname="adrian", platform="linux",
            ubik_root=_FAKE_ROOT, is_wsl=True, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        with patch("maestro.services.mcp_server_service.detect_node", return_value=somatic_id):
            result = await svc.start(_FAKE_ROOT)
        assert result is False

    @pytest.mark.asyncio
    async def test_vllm_refuses_hippocampal_node(self):
        svc = VllmService(model_path="/fake/model", max_wait_s=0.1)
        hippo_id = NodeIdentity(
            node_type=NodeType.HIPPOCAMPAL, hostname="mac.lan", platform="darwin",
            ubik_root=_FAKE_ROOT, is_wsl=False, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        with patch("maestro.services.vllm_service.detect_node", return_value=hippo_id):
            result = await svc.start(_FAKE_ROOT)
        assert result is False

    @pytest.mark.asyncio
    async def test_unknown_node_allowed_for_hippocampal_service(self):
        """UNKNOWN node should not block startup (dev/test environment)."""
        svc = Neo4jService(ubik_root=_FAKE_ROOT, max_wait_s=0.1)
        unknown_id = NodeIdentity(
            node_type=NodeType.UNKNOWN, hostname="unknown", platform="linux",
            ubik_root=_FAKE_ROOT, is_wsl=False, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        # Even if the node check passes, the compose file won't exist → False
        with patch("maestro.services.neo4j_service.detect_node", return_value=unknown_id):
            result = await svc.start(_FAKE_ROOT)
        # Result is False due to missing compose file, not node check
        assert result is False


# ---------------------------------------------------------------------------
# Enhanced start() — path/file pre-flight checks
# ---------------------------------------------------------------------------

class TestStartPathCheck:
    """Services verify required files before starting."""

    @pytest.mark.asyncio
    async def test_neo4j_fails_if_compose_missing(self, tmp_path):
        svc = Neo4jService(ubik_root=tmp_path, max_wait_s=0.1)
        hippo_id = NodeIdentity(
            node_type=NodeType.HIPPOCAMPAL, hostname="mac.lan", platform="darwin",
            ubik_root=tmp_path, is_wsl=False, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        with patch("maestro.services.neo4j_service.detect_node", return_value=hippo_id):
            result = await svc.start(tmp_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_chromadb_fails_if_compose_missing(self, tmp_path):
        svc = ChromaDbService(ubik_root=tmp_path, max_wait_s=0.1)
        hippo_id = NodeIdentity(
            node_type=NodeType.HIPPOCAMPAL, hostname="mac.lan", platform="darwin",
            ubik_root=tmp_path, is_wsl=False, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        with patch("maestro.services.chromadb_service.detect_node", return_value=hippo_id):
            result = await svc.start(tmp_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_mcp_fails_if_script_missing(self, tmp_path):
        svc = McpServerService(ubik_root=tmp_path, max_wait_s=0.1)
        hippo_id = NodeIdentity(
            node_type=NodeType.HIPPOCAMPAL, hostname="mac.lan", platform="darwin",
            ubik_root=tmp_path, is_wsl=False, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        with patch("maestro.services.mcp_server_service.detect_node", return_value=hippo_id):
            result = await svc.start(tmp_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_neo4j_proceeds_when_compose_exists(self, tmp_path):
        (tmp_path / "docker-compose.yml").write_text("services: {}")
        svc = Neo4jService(ubik_root=tmp_path, max_wait_s=0.1)
        hippo_id = NodeIdentity(
            node_type=NodeType.HIPPOCAMPAL, hostname="mac.lan", platform="darwin",
            ubik_root=tmp_path, is_wsl=False, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        with patch("maestro.services.neo4j_service.detect_node", return_value=hippo_id), \
             patch("maestro.services.neo4j_service._run_proc",
                   new_callable=AsyncMock, return_value=(0, "", "")), \
             patch.object(svc, "probe_with_timeout",
                          new_callable=AsyncMock,
                          return_value=_healthy("neo4j", NodeType.HIPPOCAMPAL)):
            result = await svc.start(tmp_path)
        assert result is True


# ---------------------------------------------------------------------------
# Enhanced start() — health-wait loop
# ---------------------------------------------------------------------------

class TestStartHealthWait:
    """start() must poll probe() and return True/False accordingly."""

    @pytest.mark.asyncio
    async def test_neo4j_returns_true_when_probe_succeeds(self, tmp_path):
        (tmp_path / "docker-compose.yml").write_text("services: {}")
        svc = Neo4jService(ubik_root=tmp_path, max_wait_s=5.0)
        hippo_id = NodeIdentity(
            node_type=NodeType.HIPPOCAMPAL, hostname="mac.lan", platform="darwin",
            ubik_root=tmp_path, is_wsl=False, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        healthy = _healthy("neo4j", NodeType.HIPPOCAMPAL)
        with patch("maestro.services.neo4j_service.detect_node", return_value=hippo_id), \
             patch("maestro.services.neo4j_service._run_proc",
                   new_callable=AsyncMock, return_value=(0, "", "")), \
             patch.object(svc, "probe_with_timeout",
                          new_callable=AsyncMock, return_value=healthy):
            result = await svc.start(tmp_path)
        assert result is True

    @pytest.mark.asyncio
    async def test_neo4j_returns_false_when_probe_never_succeeds(self, tmp_path):
        (tmp_path / "docker-compose.yml").write_text("services: {}")
        # Very short max_wait_s so test doesn't hang
        svc = Neo4jService(ubik_root=tmp_path, max_wait_s=0.05)
        hippo_id = NodeIdentity(
            node_type=NodeType.HIPPOCAMPAL, hostname="mac.lan", platform="darwin",
            ubik_root=tmp_path, is_wsl=False, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        unhealthy = _unhealthy("neo4j", NodeType.HIPPOCAMPAL)
        with patch("maestro.services.neo4j_service.detect_node", return_value=hippo_id), \
             patch("maestro.services.neo4j_service._run_proc",
                   new_callable=AsyncMock, return_value=(0, "", "")), \
             patch.object(svc, "probe_with_timeout",
                          new_callable=AsyncMock, return_value=unhealthy):
            result = await svc.start(tmp_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_mcp_probe_polled_until_healthy(self, tmp_path):
        """Verify probe_with_timeout is called multiple times before success."""
        hippo_path = tmp_path / "hippocampal"
        hippo_path.mkdir()
        script = hippo_path / "run_mcp.sh"
        script.write_text("#!/bin/bash\n"); script.chmod(0o755)

        svc = McpServerService(ubik_root=tmp_path, max_wait_s=10.0)
        hippo_id = NodeIdentity(
            node_type=NodeType.HIPPOCAMPAL, hostname="mac.lan", platform="darwin",
            ubik_root=tmp_path, is_wsl=False, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        unhealthy = _unhealthy("mcp", NodeType.HIPPOCAMPAL)
        healthy = _healthy("mcp", NodeType.HIPPOCAMPAL)

        # First two calls unhealthy, third healthy
        with patch("maestro.services.mcp_server_service.detect_node",
                   return_value=hippo_id), \
             patch("maestro.services.mcp_server_service.asyncio.create_subprocess_exec",
                   new_callable=AsyncMock), \
             patch.object(
                 svc, "probe_with_timeout", new_callable=AsyncMock,
                 side_effect=[unhealthy, unhealthy, healthy]
             ) as mock_probe:
            result = await svc.start(tmp_path)
        assert result is True
        assert mock_probe.call_count == 3

    @pytest.mark.asyncio
    async def test_docker_health_wait_called_after_open(self, tmp_path):
        svc = DockerService(max_wait_s=5.0)
        hippo_id = NodeIdentity(
            node_type=NodeType.HIPPOCAMPAL, hostname="mac.lan", platform="darwin",
            ubik_root=tmp_path, is_wsl=False, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        healthy = _healthy("docker", NodeType.HIPPOCAMPAL)
        with patch("maestro.services.docker_service.detect_node", return_value=hippo_id), \
             patch("maestro.services.docker_service._run_proc",
                   new_callable=AsyncMock, return_value=(0, "", "")), \
             patch("maestro.services.docker_service.platform.system",
                   return_value="Darwin"), \
             patch.object(svc, "probe_with_timeout",
                          new_callable=AsyncMock, return_value=healthy) as mock_probe:
            result = await svc.start(tmp_path)
        assert result is True
        mock_probe.assert_called()

    @pytest.mark.asyncio
    async def test_vllm_health_wait_called_after_conda(self, tmp_path):
        svc = VllmService(model_path="/fake/model", max_wait_s=5.0)
        somatic_id = NodeIdentity(
            node_type=NodeType.SOMATIC, hostname="adrian", platform="linux",
            ubik_root=tmp_path, is_wsl=True, tailscale_ip=None,
            python_venv_path=None, python_activate_cmd=None,
        )
        healthy = _healthy("vllm", NodeType.SOMATIC)
        with patch("maestro.services.vllm_service.detect_node", return_value=somatic_id), \
             patch("maestro.services.vllm_service.asyncio.create_subprocess_exec",
                   new_callable=AsyncMock), \
             patch.object(svc, "probe_with_timeout",
                          new_callable=AsyncMock, return_value=healthy):
            result = await svc.start(tmp_path)
        assert result is True


# ---------------------------------------------------------------------------
# max_wait_s defaults
# ---------------------------------------------------------------------------

class TestMaxWaitSDefaults:
    def test_docker_default(self):
        assert DockerService().max_wait_s == 60.0

    def test_neo4j_default(self):
        assert Neo4jService(ubik_root=_FAKE_ROOT).max_wait_s == 60.0

    def test_chromadb_default(self):
        assert ChromaDbService(ubik_root=_FAKE_ROOT).max_wait_s == 30.0

    def test_mcp_default(self):
        assert McpServerService(ubik_root=_FAKE_ROOT).max_wait_s == 15.0

    def test_vllm_default(self):
        assert VllmService(model_path="/m").max_wait_s == 120.0

    def test_custom_max_wait_s(self):
        svc = Neo4jService(ubik_root=_FAKE_ROOT, max_wait_s=99.0)
        assert svc.max_wait_s == 99.0


# ---------------------------------------------------------------------------
# Orchestrator.full_status_check()
# ---------------------------------------------------------------------------

class TestFullStatusCheck:
    @pytest.mark.asyncio
    async def test_returns_all_services(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        # Patch all probes to healthy
        with patch.object(
            registry.get_all()[0], "probe_with_timeout",
            new_callable=AsyncMock,
        ):
            for svc in registry.get_all():
                svc.probe_with_timeout = AsyncMock(
                    return_value=_healthy(svc.name, svc.node)
                )
            statuses = await orch.full_status_check()

        assert set(statuses.keys()) == {"docker", "neo4j", "chromadb", "mcp", "vllm"}

    @pytest.mark.asyncio
    async def test_local_services_probed_via_localhost(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        probe_calls: dict[str, list] = {}
        for svc in registry.get_all():
            name = svc.name
            probe_calls[name] = []
            async def capture(host, *, _name=name, _node=svc.node, **kw):
                probe_calls[_name].append(host)
                return _healthy(_name, _node)
            svc.probe_with_timeout = capture

        await orch.full_status_check()

        # Hippocampal services should be probed via localhost
        for name in ("docker", "neo4j", "chromadb", "mcp"):
            assert probe_calls[name] == ["localhost"], \
                f"{name} should be probed via localhost"

    @pytest.mark.asyncio
    async def test_remote_services_probed_via_tailscale_ip(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        vllm_hosts: list[str] = []
        for svc in registry.get_all():
            if svc.name == "vllm":
                async def capture(host, **kw):
                    vllm_hosts.append(host)
                    return _healthy("vllm", NodeType.SOMATIC)
                svc.probe_with_timeout = capture
            else:
                svc.probe_with_timeout = AsyncMock(
                    return_value=_healthy(svc.name, svc.node)
                )

        await orch.full_status_check()
        assert vllm_hosts == ["100.0.0.2"]

    @pytest.mark.asyncio
    async def test_exception_in_probe_returns_unhealthy(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        for svc in registry.get_all():
            if svc.name == "neo4j":
                svc.probe_with_timeout = AsyncMock(side_effect=RuntimeError("bang"))
            else:
                svc.probe_with_timeout = AsyncMock(
                    return_value=_healthy(svc.name, svc.node)
                )

        statuses = await orch.full_status_check()
        assert statuses["neo4j"].healthy is False
        assert "bang" in statuses["neo4j"].error

    @pytest.mark.asyncio
    async def test_all_probes_run_concurrently(self, tmp_path):
        """Verify asyncio.gather is used: all probes start before any finishes."""
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        started: list[str] = []
        finished: list[str] = []

        for svc in registry.get_all():
            name = svc.name
            node = svc.node

            async def track(host, *, _n=name, _nd=node):
                started.append(_n)
                await asyncio.sleep(0.01)
                finished.append(_n)
                return _healthy(_n, _nd)

            svc.probe_with_timeout = track

        await orch.full_status_check()
        # All services should have started before any finished
        # (i.e., gather ran them concurrently)
        assert len(started) == 5
        assert len(finished) == 5


# ---------------------------------------------------------------------------
# Orchestrator.ensure_all_running()
# ---------------------------------------------------------------------------

class TestEnsureAllRunning:
    @pytest.mark.asyncio
    async def test_already_healthy_services_not_restarted(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        # All local services healthy
        for svc in registry.get_all():
            if svc.node == NodeType.HIPPOCAMPAL:
                svc.probe_with_timeout = AsyncMock(
                    return_value=_healthy(svc.name, svc.node)
                )
                svc.start = AsyncMock(return_value=True)

        failed = await orch.ensure_all_running()
        assert failed == []
        # start() should not have been called
        for svc in registry.get_all():
            if svc.node == NodeType.HIPPOCAMPAL:
                svc.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_unhealthy_service_started(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        # docker unhealthy → needs start; others healthy
        for svc in registry.get_all():
            if svc.node == NodeType.HIPPOCAMPAL:
                is_down = svc.name == "docker"
                svc.probe_with_timeout = AsyncMock(
                    return_value=(
                        _unhealthy(svc.name, svc.node) if is_down
                        else _healthy(svc.name, svc.node)
                    )
                )
                svc.start = AsyncMock(return_value=True)

        failed = await orch.ensure_all_running()
        assert failed == []
        # Only docker.start() should have been called
        for svc in registry.get_all():
            if svc.node == NodeType.HIPPOCAMPAL:
                if svc.name == "docker":
                    svc.start.assert_called_once()
                else:
                    svc.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_service_causes_dependents_to_be_skipped(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        # docker fails to start → neo4j, chromadb, mcp should all be skipped
        for svc in registry.get_all():
            if svc.node == NodeType.HIPPOCAMPAL:
                svc.probe_with_timeout = AsyncMock(
                    return_value=_unhealthy(svc.name, svc.node)
                )
                if svc.name == "docker":
                    svc.start = AsyncMock(return_value=False)
                else:
                    svc.start = AsyncMock(return_value=True)

        failed = await orch.ensure_all_running()
        assert "docker" in failed
        assert "neo4j" in failed
        assert "chromadb" in failed
        assert "mcp" in failed

    @pytest.mark.asyncio
    async def test_remote_services_skipped(self, tmp_path):
        """ensure_all_running() on Hippocampal should not touch vLLM."""
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        vllm_svc = next(s for s in registry.get_all() if s.name == "vllm")
        vllm_svc.probe_with_timeout = AsyncMock(
            return_value=_unhealthy("vllm", NodeType.SOMATIC)
        )
        vllm_svc.start = AsyncMock(return_value=True)

        for svc in registry.get_all():
            if svc.node == NodeType.HIPPOCAMPAL:
                svc.probe_with_timeout = AsyncMock(
                    return_value=_healthy(svc.name, svc.node)
                )

        await orch.ensure_all_running()
        vllm_svc.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_all_healthy(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        for svc in registry.get_all():
            if svc.node == NodeType.HIPPOCAMPAL:
                svc.probe_with_timeout = AsyncMock(
                    return_value=_healthy(svc.name, svc.node)
                )

        failed = await orch.ensure_all_running()
        assert failed == []

    @pytest.mark.asyncio
    async def test_unhealthy_dependency_blocks_start(self, tmp_path):
        """If neo4j is unhealthy and fails to start, mcp should not be attempted."""
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        for svc in registry.get_all():
            if svc.node == NodeType.HIPPOCAMPAL:
                svc.probe_with_timeout = AsyncMock(
                    return_value=_unhealthy(svc.name, svc.node)
                )
                if svc.name in ("docker", "neo4j"):
                    svc.start = AsyncMock(return_value=False)
                else:
                    svc.start = AsyncMock(return_value=True)

        failed = await orch.ensure_all_running()
        mcp_svc = next(s for s in registry.get_all() if s.name == "mcp")
        mcp_svc.start.assert_not_called()


# ---------------------------------------------------------------------------
# Orchestrator.generate_report()
# ---------------------------------------------------------------------------

class TestGenerateReport:
    @pytest.mark.asyncio
    async def test_report_contains_all_service_names(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        statuses = {
            svc.name: _healthy(svc.name, svc.node)
            for svc in registry.get_all()
        }
        report = await orch.generate_report(statuses)
        for name in ("docker", "neo4j", "chromadb", "mcp", "vllm"):
            assert name in report

    @pytest.mark.asyncio
    async def test_report_shows_healthy_summary(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        statuses = {
            svc.name: _healthy(svc.name, svc.node)
            for svc in registry.get_all()
        }
        report = await orch.generate_report(statuses)
        assert "5/5" in report

    @pytest.mark.asyncio
    async def test_report_shows_partial_count(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        statuses = {}
        for i, svc in enumerate(registry.get_all()):
            statuses[svc.name] = (
                _healthy(svc.name, svc.node) if i < 3
                else _unhealthy(svc.name, svc.node)
            )
        report = await orch.generate_report(statuses)
        assert "3/5" in report

    @pytest.mark.asyncio
    async def test_report_is_string(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        statuses = {
            svc.name: _healthy(svc.name, svc.node)
            for svc in registry.get_all()
        }
        report = await orch.generate_report(statuses)
        assert isinstance(report, str)
        assert len(report) > 0

    @pytest.mark.asyncio
    async def test_report_shows_node_identity(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        statuses = {}
        report = await orch.generate_report(statuses)
        assert "hippocampal" in report
        assert "mac.lan" in report


# ---------------------------------------------------------------------------
# Orchestrator probe host resolution
# ---------------------------------------------------------------------------

class TestProbeHostResolution:
    def test_local_service_uses_localhost(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        docker_svc = next(s for s in registry.get_all() if s.name == "docker")
        assert orch._probe_host_for(docker_svc) == "localhost"

    def test_remote_somatic_service_uses_tailscale_ip(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _hippocampal_identity())

        vllm_svc = next(s for s in registry.get_all() if s.name == "vllm")
        assert orch._probe_host_for(vllm_svc) == "100.0.0.2"

    def test_somatic_node_uses_hippocampal_ip_for_neo4j(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _somatic_identity())

        neo4j_svc = next(s for s in registry.get_all() if s.name == "neo4j")
        assert orch._probe_host_for(neo4j_svc) == "100.0.0.1"

    def test_somatic_local_services_use_localhost(self, tmp_path):
        cfg = _make_app_config(tmp_path)
        registry = ServiceRegistry(cfg=cfg)
        orch = Orchestrator(registry, _somatic_identity())

        vllm_svc = next(s for s in registry.get_all() if s.name == "vllm")
        assert orch._probe_host_for(vllm_svc) == "localhost"
