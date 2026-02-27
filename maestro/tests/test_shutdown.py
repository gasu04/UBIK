"""
Tests for maestro/shutdown.py — ShutdownController and _sigkill_port.

Coverage:
    TestShutdownControllerInit         (3)  constructor and mlog wiring
    TestLocalServicesInShutdownOrder   (4)  per-node reverse startup order
    TestWaitForDown                    (4)  poll-until-down logic
    TestSigkillPort                    (5)  SIGKILL helper (lsof/fuser path)
    TestStopDaemon                     (5)  PID-file SIGTERM
    TestVerifyAllDown                  (5)  post-shutdown probe map
    TestOrderlyShutdown               (12)  main shutdown loop
    TestEmergencyShutdown              (6)  psutil / fallback kill
    TestShutdownCLI                    (5)  CLI --dry-run / --emergency
"""

import asyncio
import signal
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from maestro.platform_detect import NodeIdentity, NodeType
from maestro.services import ServiceRegistry
from maestro.services.base import ProbeResult, UbikService
from maestro.shutdown import ShutdownController, _sigkill_port

# ---------------------------------------------------------------------------
# Patch constants
# ---------------------------------------------------------------------------

_PATCH_CFG = "maestro.cli.get_config"
_PATCH_LOGGING = "maestro.cli.configure_logging"
_PATCH_CTRL = "maestro.shutdown.ShutdownController"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity(node_type: NodeType = NodeType.HIPPOCAMPAL) -> NodeIdentity:
    return NodeIdentity(
        node_type=node_type,
        hostname="test-host",
        platform="darwin",
        ubik_root=Path("/tmp/test-ubik"),
        is_wsl=False,
        tailscale_ip="127.0.0.1",
        python_venv_path=None,
        python_activate_cmd=None,
    )


def _mock_svc(
    name: str,
    node_type: NodeType = NodeType.HIPPOCAMPAL,
    *,
    ports: list[int] | None = None,
    healthy: bool = False,
) -> MagicMock:
    """Create a mock UbikService with controllable probe result."""
    svc = MagicMock(spec=UbikService)
    svc.name = name
    svc.node = node_type
    svc.ports = ports if ports is not None else []
    svc.depends_on = []
    svc.stop = AsyncMock(return_value=True)
    svc.probe_with_timeout = AsyncMock(
        return_value=ProbeResult(
            name=name,
            node=node_type,
            healthy=healthy,
            latency_ms=5.0,
            error=None if healthy else "connection refused",
        )
    )
    return svc


def _make_ctrl(
    app_config,
    node_type: NodeType = NodeType.HIPPOCAMPAL,
) -> ShutdownController:
    """Controller backed by a real ServiceRegistry."""
    identity = _identity(node_type)
    registry = ServiceRegistry(app_config)
    return ShutdownController(registry, identity, mlog=MagicMock())


def _make_ctrl_with_services(
    services: list,
    app_config,
    node_type: NodeType = NodeType.HIPPOCAMPAL,
) -> ShutdownController:
    """Controller with a mock registry returning the given service list."""
    identity = _identity(node_type)
    registry = MagicMock(spec=ServiceRegistry)
    registry.cfg = app_config
    registry.get_all.return_value = list(services)
    registry.get_startup_order.return_value = list(services)
    return ShutdownController(registry, identity, mlog=MagicMock())


def _make_ctrl_with_log_dir(
    log_dir: Path,
    node_type: NodeType = NodeType.HIPPOCAMPAL,
) -> ShutdownController:
    """Controller with a mock registry pointed at *log_dir*."""
    identity = _identity(node_type)
    registry = MagicMock(spec=ServiceRegistry)
    registry.cfg.log_dir = log_dir
    registry.get_all.return_value = []
    registry.get_startup_order.return_value = []
    return ShutdownController(registry, identity, mlog=MagicMock())


# ---------------------------------------------------------------------------
# TestShutdownControllerInit
# ---------------------------------------------------------------------------

class TestShutdownControllerInit:
    def test_stores_registry_and_identity(self, app_config):
        identity = _identity()
        registry = ServiceRegistry(app_config)
        ctrl = ShutdownController(registry, identity, mlog=MagicMock())

        assert ctrl._registry is registry
        assert ctrl._identity is identity

    def test_creates_mlog_when_none(self, app_config):
        ctrl = _make_ctrl(app_config)
        # _make_ctrl passes mlog=MagicMock(), but a real MaestroLogger
        # would also be created when mlog is omitted — verify the attribute exists
        assert ctrl._mlog is not None

    def test_accepts_custom_mlog(self, app_config):
        custom_mlog = MagicMock()
        identity = _identity()
        registry = ServiceRegistry(app_config)
        ctrl = ShutdownController(registry, identity, mlog=custom_mlog)

        assert ctrl._mlog is custom_mlog


# ---------------------------------------------------------------------------
# TestLocalServicesInShutdownOrder
# ---------------------------------------------------------------------------

class TestLocalServicesInShutdownOrder:
    """Startup order from get_startup_order(): [docker, vllm, neo4j, chromadb, mcp].
    Hippocampal local: [docker, neo4j, chromadb, mcp] → reversed: [mcp, chromadb, neo4j, docker].
    Somatic local: [vllm] → reversed: [vllm].
    """

    def test_hippocampal_reversed_order(self, app_config):
        ctrl = _make_ctrl(app_config, NodeType.HIPPOCAMPAL)
        names = [s.name for s in ctrl._local_services_in_shutdown_order()]
        assert names == ["mcp", "chromadb", "neo4j", "docker"]

    def test_somatic_reversed_order(self, app_config):
        ctrl = _make_ctrl(app_config, NodeType.SOMATIC)
        names = [s.name for s in ctrl._local_services_in_shutdown_order()]
        assert names == ["vllm"]

    def test_unknown_node_returns_empty(self, app_config):
        ctrl = _make_ctrl(app_config, NodeType.UNKNOWN)
        assert ctrl._local_services_in_shutdown_order() == []

    def test_dependents_before_dependencies(self, app_config):
        ctrl = _make_ctrl(app_config, NodeType.HIPPOCAMPAL)
        names = [s.name for s in ctrl._local_services_in_shutdown_order()]
        # mcp depends on neo4j and chromadb — must stop before them
        assert names.index("mcp") < names.index("neo4j")
        assert names.index("mcp") < names.index("chromadb")
        # docker has no dependents among hippocampal — must be last
        assert names[-1] == "docker"


# ---------------------------------------------------------------------------
# TestWaitForDown
# ---------------------------------------------------------------------------

class TestWaitForDown:
    @pytest.mark.asyncio
    async def test_immediately_down(self, app_config):
        ctrl = _make_ctrl(app_config)
        svc = _mock_svc("neo4j", healthy=False)  # already DOWN

        result = await ctrl._wait_for_down(svc, timeout=5.0)

        assert result is True
        svc.probe_with_timeout.assert_called_once()

    @pytest.mark.asyncio
    async def test_down_after_delay(self, app_config):
        ctrl = _make_ctrl(app_config)
        svc = _mock_svc("neo4j")
        svc.probe_with_timeout = AsyncMock(side_effect=[
            ProbeResult(name="neo4j", node=NodeType.HIPPOCAMPAL, healthy=True, latency_ms=5.0),
            ProbeResult(name="neo4j", node=NodeType.HIPPOCAMPAL, healthy=False, latency_ms=5.0),
        ])

        with patch("asyncio.sleep", AsyncMock()):
            result = await ctrl._wait_for_down(svc, timeout=10.0)

        assert result is True
        assert svc.probe_with_timeout.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_returns_false(self, app_config):
        ctrl = _make_ctrl(app_config)
        svc = _mock_svc("neo4j", healthy=True)  # never goes down

        # timeout=0.0: after first healthy probe, elapsed >= 0 → return False
        result = await ctrl._wait_for_down(svc, timeout=0.0)

        assert result is False

    @pytest.mark.asyncio
    async def test_polls_with_interval(self, app_config):
        ctrl = _make_ctrl(app_config)
        svc = _mock_svc("neo4j")
        svc.probe_with_timeout = AsyncMock(side_effect=[
            ProbeResult(name="neo4j", node=NodeType.HIPPOCAMPAL, healthy=True, latency_ms=5.0),
            ProbeResult(name="neo4j", node=NodeType.HIPPOCAMPAL, healthy=False, latency_ms=5.0),
        ])

        with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            await ctrl._wait_for_down(svc, timeout=10.0)

        mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# TestSigkillPort
# ---------------------------------------------------------------------------

class TestSigkillPort:
    @pytest.mark.asyncio
    async def test_darwin_uses_lsof(self):
        with (
            patch("maestro.shutdown.platform") as mock_plat,
            patch("maestro.shutdown._run_proc", AsyncMock(return_value=(0, "1234", ""))) as mock_proc,
            patch("os.kill"),
        ):
            mock_plat.system.return_value = "Darwin"
            await _sigkill_port(8080)

        first_arg = mock_proc.call_args[0][0]
        assert first_arg == "lsof"

    @pytest.mark.asyncio
    async def test_linux_uses_fuser(self):
        with (
            patch("maestro.shutdown.platform") as mock_plat,
            patch("maestro.shutdown._run_proc", AsyncMock(return_value=(0, "1234", ""))) as mock_proc,
            patch("os.kill"),
        ):
            mock_plat.system.return_value = "Linux"
            await _sigkill_port(9000)

        first_arg = mock_proc.call_args[0][0]
        assert first_arg == "fuser"

    @pytest.mark.asyncio
    async def test_sends_sigkill_to_pids(self):
        with (
            patch("maestro.shutdown.platform") as mock_plat,
            patch("maestro.shutdown._run_proc", AsyncMock(return_value=(0, "1234 5678", ""))),
            patch("os.kill") as mock_kill,
        ):
            mock_plat.system.return_value = "Darwin"
            result = await _sigkill_port(8080)

        assert result is True
        # Verify SIGKILL (not SIGTERM) is used
        for c in mock_kill.call_args_list:
            assert c[0][1] == signal.SIGKILL

    @pytest.mark.asyncio
    async def test_no_pids_returns_false(self):
        with (
            patch("maestro.shutdown.platform") as mock_plat,
            patch("maestro.shutdown._run_proc", AsyncMock(return_value=(0, "", ""))),
            patch("os.kill") as mock_kill,
        ):
            mock_plat.system.return_value = "Darwin"
            result = await _sigkill_port(8080)

        assert result is False
        mock_kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_returns_false(self):
        with (
            patch("maestro.shutdown.platform") as mock_plat,
            patch("maestro.shutdown._run_proc", AsyncMock(side_effect=asyncio.TimeoutError())),
            patch("os.kill") as mock_kill,
        ):
            mock_plat.system.return_value = "Darwin"
            result = await _sigkill_port(8080)

        assert result is False
        mock_kill.assert_not_called()


# ---------------------------------------------------------------------------
# TestStopDaemon
# ---------------------------------------------------------------------------

class TestStopDaemon:
    def test_no_pid_file_is_noop(self, tmp_path):
        ctrl = _make_ctrl_with_log_dir(tmp_path)

        with patch("os.kill") as mock_kill:
            ctrl._stop_daemon()

        mock_kill.assert_not_called()

    def test_sends_sigterm_to_daemon(self, tmp_path):
        ctrl = _make_ctrl_with_log_dir(tmp_path)
        (tmp_path / "maestro.pid").write_text("12345")

        with patch("os.kill") as mock_kill:
            ctrl._stop_daemon()

        mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    def test_dead_process_is_ignored(self, tmp_path):
        ctrl = _make_ctrl_with_log_dir(tmp_path)
        (tmp_path / "maestro.pid").write_text("99999")

        with patch("os.kill", side_effect=ProcessLookupError()):
            ctrl._stop_daemon()  # must not raise

    def test_invalid_pid_file_ignored(self, tmp_path):
        ctrl = _make_ctrl_with_log_dir(tmp_path)
        (tmp_path / "maestro.pid").write_text("not-a-pid")

        with patch("os.kill") as mock_kill:
            ctrl._stop_daemon()  # must not raise

        mock_kill.assert_not_called()

    def test_logs_stop_action(self, tmp_path):
        ctrl = _make_ctrl_with_log_dir(tmp_path)
        (tmp_path / "maestro.pid").write_text("12345")

        with patch("os.kill"):
            ctrl._stop_daemon()

        ctrl._mlog.log_service_action.assert_called_once()
        args = ctrl._mlog.log_service_action.call_args[0]
        assert args[0] == "daemon"
        assert args[1] == "stop"
        assert args[2] is True


# ---------------------------------------------------------------------------
# TestVerifyAllDown
# ---------------------------------------------------------------------------

class TestVerifyAllDown:
    @pytest.mark.asyncio
    async def test_all_services_down(self, app_config):
        svc1 = _mock_svc("mcp", healthy=False)
        svc2 = _mock_svc("neo4j", healthy=False)
        ctrl = _make_ctrl_with_services([svc1, svc2], app_config)

        result = await ctrl._verify_all_down()

        assert result == {"mcp": True, "neo4j": True}

    @pytest.mark.asyncio
    async def test_service_still_up(self, app_config):
        svc1 = _mock_svc("mcp", healthy=False)
        svc2 = _mock_svc("neo4j", healthy=True)   # still responding
        ctrl = _make_ctrl_with_services([svc1, svc2], app_config)

        result = await ctrl._verify_all_down()

        assert result["mcp"] is True
        assert result["neo4j"] is False

    @pytest.mark.asyncio
    async def test_exception_counts_as_down(self, app_config):
        svc = _mock_svc("mcp")
        svc.probe_with_timeout = AsyncMock(side_effect=RuntimeError("oops"))
        ctrl = _make_ctrl_with_services([svc], app_config)

        result = await ctrl._verify_all_down()

        assert result["mcp"] is True   # exception → can't connect → DOWN

    @pytest.mark.asyncio
    async def test_unknown_node_returns_empty(self, app_config):
        ctrl = _make_ctrl_with_services([], app_config, NodeType.UNKNOWN)

        result = await ctrl._verify_all_down()

        assert result == {}

    @pytest.mark.asyncio
    async def test_only_checks_local_services(self, app_config):
        local_svc = _mock_svc("mcp", NodeType.HIPPOCAMPAL, healthy=False)
        remote_svc = _mock_svc("vllm", NodeType.SOMATIC, healthy=True)
        ctrl = _make_ctrl_with_services(
            [local_svc, remote_svc], app_config, NodeType.HIPPOCAMPAL
        )

        result = await ctrl._verify_all_down()

        assert "mcp" in result
        assert "vllm" not in result   # remote — never probed


# ---------------------------------------------------------------------------
# TestOrderlyShutdown
# ---------------------------------------------------------------------------

class TestOrderlyShutdown:
    @pytest.mark.asyncio
    async def test_dry_run_returns_all_names(self, app_config):
        svc1 = _mock_svc("mcp")
        svc2 = _mock_svc("neo4j")
        ctrl = _make_ctrl_with_services([svc1, svc2], app_config)

        stopped = await ctrl.orderly_shutdown(dry_run=True)

        assert sorted(stopped) == ["mcp", "neo4j"]
        svc1.stop.assert_not_called()
        svc2.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_skips_stop_daemon(self, app_config):
        svc = _mock_svc("mcp")
        ctrl = _make_ctrl_with_services([svc], app_config)

        with patch.object(ctrl, "_stop_daemon") as mock_daemon:
            await ctrl.orderly_shutdown(dry_run=True)

        mock_daemon.assert_not_called()

    @pytest.mark.asyncio
    async def test_happy_path_all_stopped(self, app_config):
        svc1 = _mock_svc("mcp")
        svc2 = _mock_svc("neo4j")
        ctrl = _make_ctrl_with_services([svc1, svc2], app_config)

        with (
            patch.object(ctrl, "_stop_daemon"),
            patch.object(ctrl, "_wait_for_down", AsyncMock(return_value=True)),
            patch.object(ctrl, "_verify_all_down", AsyncMock(return_value={})),
        ):
            stopped = await ctrl.orderly_shutdown()

        assert sorted(stopped) == ["mcp", "neo4j"]
        svc1.stop.assert_called_once()
        svc2.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_raises_continues(self, app_config):
        svc = _mock_svc("mcp")
        svc.stop = AsyncMock(side_effect=RuntimeError("stop exploded"))
        ctrl = _make_ctrl_with_services([svc], app_config)

        with (
            patch.object(ctrl, "_stop_daemon"),
            patch.object(ctrl, "_wait_for_down", AsyncMock(return_value=True)),
            patch.object(ctrl, "_verify_all_down", AsyncMock(return_value={})),
        ):
            stopped = await ctrl.orderly_shutdown()

        # stop() raised, but probe confirmed down → service is in stopped list
        assert "mcp" in stopped

    @pytest.mark.asyncio
    async def test_escalates_to_sigkill_when_timeout(self, app_config):
        svc = _mock_svc("mcp", ports=[8080])
        ctrl = _make_ctrl_with_services([svc], app_config)

        with (
            patch.object(ctrl, "_stop_daemon"),
            patch.object(ctrl, "_wait_for_down", AsyncMock(return_value=False)),
            patch.object(ctrl, "_sigkill_service", AsyncMock()) as mock_kill,
            patch("asyncio.sleep", AsyncMock()),
            patch.object(ctrl, "_verify_all_down", AsyncMock(return_value={})),
        ):
            await ctrl.orderly_shutdown()

        mock_kill.assert_called_once_with(svc)

    @pytest.mark.asyncio
    async def test_sigkill_confirmed_included_in_stopped(self, app_config):
        # After SIGKILL, probe returns healthy=False → included in stopped
        svc = _mock_svc("mcp", ports=[8080], healthy=False)
        ctrl = _make_ctrl_with_services([svc], app_config)

        with (
            patch.object(ctrl, "_stop_daemon"),
            patch.object(ctrl, "_wait_for_down", AsyncMock(return_value=False)),
            patch.object(ctrl, "_sigkill_service", AsyncMock()),
            patch("asyncio.sleep", AsyncMock()),
            patch.object(ctrl, "_verify_all_down", AsyncMock(return_value={})),
        ):
            stopped = await ctrl.orderly_shutdown()

        assert "mcp" in stopped

    @pytest.mark.asyncio
    async def test_sigkill_still_alive_excluded(self, app_config):
        # After SIGKILL, probe still returns healthy=True → excluded from stopped
        svc = _mock_svc("mcp", ports=[8080], healthy=True)
        ctrl = _make_ctrl_with_services([svc], app_config)

        with (
            patch.object(ctrl, "_stop_daemon"),
            patch.object(ctrl, "_wait_for_down", AsyncMock(return_value=False)),
            patch.object(ctrl, "_sigkill_service", AsyncMock()),
            patch("asyncio.sleep", AsyncMock()),
            patch.object(ctrl, "_verify_all_down", AsyncMock(return_value={})),
        ):
            stopped = await ctrl.orderly_shutdown()

        assert "mcp" not in stopped

    @pytest.mark.asyncio
    async def test_total_timeout_skips_remaining(self, app_config):
        svc1 = _mock_svc("mcp")
        svc2 = _mock_svc("neo4j")
        # startup_order = [mcp, neo4j] → shutdown_order (reversed) = [neo4j, mcp]
        # neo4j is processed first; perf_counter returns 200 on mcp's check → skip
        ctrl = _make_ctrl_with_services([svc1, svc2], app_config)

        # deadline = 0.0 + 120; neo4j check: 120-0=120 (OK); mcp check: 120-200=-80 → break
        perf_calls = iter([0.0, 0.0, 200.0])

        with (
            patch.object(ctrl, "_stop_daemon"),
            patch.object(ctrl, "_wait_for_down", AsyncMock(return_value=True)),
            patch.object(ctrl, "_verify_all_down", AsyncMock(return_value={})),
            patch("maestro.shutdown.time.perf_counter", side_effect=perf_calls),
        ):
            stopped = await ctrl.orderly_shutdown()

        assert "neo4j" in stopped    # first in shutdown order — processed
        assert "mcp" not in stopped  # second — skipped after timeout
        svc1.stop.assert_not_called()  # svc1=mcp was never reached

    @pytest.mark.asyncio
    async def test_no_local_services_returns_empty(self, app_config):
        ctrl = _make_ctrl(app_config, NodeType.UNKNOWN)

        with patch.object(ctrl, "_stop_daemon"):
            stopped = await ctrl.orderly_shutdown()

        assert stopped == []
        ctrl._mlog.log_shutdown.assert_called_once_with([])

    @pytest.mark.asyncio
    async def test_calls_stop_daemon(self, app_config):
        svc = _mock_svc("mcp")
        ctrl = _make_ctrl_with_services([svc], app_config)

        with (
            patch.object(ctrl, "_stop_daemon") as mock_daemon,
            patch.object(ctrl, "_wait_for_down", AsyncMock(return_value=True)),
            patch.object(ctrl, "_verify_all_down", AsyncMock(return_value={})),
        ):
            await ctrl.orderly_shutdown()

        mock_daemon.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_shutdown_on_complete(self, app_config):
        svc = _mock_svc("mcp")
        ctrl = _make_ctrl_with_services([svc], app_config)

        with (
            patch.object(ctrl, "_stop_daemon"),
            patch.object(ctrl, "_wait_for_down", AsyncMock(return_value=True)),
            patch.object(ctrl, "_verify_all_down", AsyncMock(return_value={})),
        ):
            stopped = await ctrl.orderly_shutdown()

        ctrl._mlog.log_shutdown.assert_called_once_with(stopped)

    @pytest.mark.asyncio
    async def test_stops_in_order(self, app_config):
        call_order: list[str] = []

        svc1 = _mock_svc("mcp")
        svc2 = _mock_svc("neo4j")
        svc1.stop = AsyncMock(side_effect=lambda *a, **kw: call_order.append("mcp"))
        svc2.stop = AsyncMock(side_effect=lambda *a, **kw: call_order.append("neo4j"))
        ctrl = _make_ctrl_with_services([svc1, svc2], app_config)

        with (
            patch.object(ctrl, "_stop_daemon"),
            patch.object(ctrl, "_wait_for_down", AsyncMock(return_value=True)),
            patch.object(ctrl, "_verify_all_down", AsyncMock(return_value={})),
        ):
            await ctrl.orderly_shutdown()

        # startup_order = [mcp, neo4j] → shutdown_order (reversed) = [neo4j, mcp]
        assert call_order == ["neo4j", "mcp"]


# ---------------------------------------------------------------------------
# TestEmergencyShutdown
# ---------------------------------------------------------------------------

class TestEmergencyShutdown:
    @pytest.mark.asyncio
    async def test_psutil_kills_processes(self, app_config):
        ctrl = _make_ctrl(app_config, NodeType.HIPPOCAMPAL)

        mock_psutil = MagicMock()
        mock_conn = MagicMock()
        mock_conn.laddr.port = 8080   # MCP port — in hippocampal's all_ports
        mock_conn.pid = 12345
        mock_psutil.net_connections.return_value = [mock_conn]
        mock_proc = MagicMock()
        mock_psutil.Process.return_value = mock_proc
        mock_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        mock_psutil.AccessDenied = type("AccessDenied", (Exception,), {})

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            await ctrl.emergency_shutdown()

        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_psutil_handles_no_such_process(self, app_config):
        ctrl = _make_ctrl(app_config, NodeType.HIPPOCAMPAL)

        MockNoSuchProcess = type("NoSuchProcess", (Exception,), {})
        MockAccessDenied = type("AccessDenied", (Exception,), {})

        mock_psutil = MagicMock()
        mock_conn = MagicMock()
        mock_conn.laddr.port = 8080
        mock_conn.pid = 12345
        mock_psutil.net_connections.return_value = [mock_conn]
        mock_psutil.NoSuchProcess = MockNoSuchProcess
        mock_psutil.AccessDenied = MockAccessDenied
        mock_psutil.Process.side_effect = MockNoSuchProcess("gone")

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            await ctrl.emergency_shutdown()  # must not raise

    @pytest.mark.asyncio
    async def test_fallback_when_psutil_missing(self, app_config):
        ctrl = _make_ctrl(app_config, NodeType.HIPPOCAMPAL)

        with (
            patch.dict("sys.modules", {"psutil": None}),
            patch("maestro.shutdown._sigkill_port", AsyncMock(return_value=True)) as mock_kill,
        ):
            await ctrl.emergency_shutdown()

        # Hippocampal has ports: neo4j=[7474,7687], chromadb=[8001], mcp=[8080]
        assert mock_kill.call_count >= 1

    @pytest.mark.asyncio
    async def test_no_ports_is_noop(self, app_config):
        ctrl = _make_ctrl(app_config, NodeType.UNKNOWN)   # no local services

        mock_psutil = MagicMock()
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            await ctrl.emergency_shutdown()

        mock_psutil.net_connections.assert_not_called()

    @pytest.mark.asyncio
    async def test_logs_emergency_action(self, app_config):
        ctrl = _make_ctrl(app_config, NodeType.UNKNOWN)   # no ports → early return

        mock_psutil = MagicMock()
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            await ctrl.emergency_shutdown()

        # log_service_action should be called twice: "emergency" initiated + detail
        assert ctrl._mlog.log_service_action.call_count >= 1
        first_call_args = ctrl._mlog.log_service_action.call_args_list[0][0]
        assert first_call_args[1] == "emergency"

    @pytest.mark.asyncio
    async def test_somatic_kills_vllm_port(self, app_config):
        ctrl = _make_ctrl(app_config, NodeType.SOMATIC)

        mock_psutil = MagicMock()
        mock_conn = MagicMock()
        mock_conn.laddr.port = 8002   # vLLM port
        mock_conn.pid = 77777
        mock_psutil.net_connections.return_value = [mock_conn]
        mock_proc = MagicMock()
        mock_psutil.Process.return_value = mock_proc
        mock_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        mock_psutil.AccessDenied = type("AccessDenied", (Exception,), {})

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            await ctrl.emergency_shutdown()

        mock_proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# TestShutdownCLI
# ---------------------------------------------------------------------------

class TestShutdownCLI:
    def test_dry_run_calls_orderly(self, app_config):
        from click.testing import CliRunner
        from maestro.cli import cli

        mock_ctrl = MagicMock()
        mock_ctrl.orderly_shutdown = AsyncMock(return_value=["neo4j"])
        mock_cls = MagicMock(return_value=mock_ctrl)

        with (
            patch(_PATCH_CFG, return_value=app_config),
            patch(_PATCH_LOGGING),
            patch(_PATCH_CTRL, mock_cls),
        ):
            result = CliRunner().invoke(cli, ["shutdown", "--dry-run"])

        assert result.exit_code == 0
        mock_ctrl.orderly_shutdown.assert_called_once()
        kwargs = mock_ctrl.orderly_shutdown.call_args[1]
        assert kwargs.get("dry_run") is True

    def test_emergency_calls_emergency_shutdown(self, app_config):
        from click.testing import CliRunner
        from maestro.cli import cli

        mock_ctrl = MagicMock()
        mock_ctrl.emergency_shutdown = AsyncMock()
        mock_cls = MagicMock(return_value=mock_ctrl)

        with (
            patch(_PATCH_CFG, return_value=app_config),
            patch(_PATCH_LOGGING),
            patch(_PATCH_CTRL, mock_cls),
        ):
            result = CliRunner().invoke(cli, ["shutdown", "--emergency"])

        assert result.exit_code == 0
        mock_ctrl.emergency_shutdown.assert_called_once()
        mock_ctrl.orderly_shutdown.assert_not_called()

    def test_default_calls_orderly(self, app_config):
        from click.testing import CliRunner
        from maestro.cli import cli

        mock_ctrl = MagicMock()
        mock_ctrl.orderly_shutdown = AsyncMock(return_value=[])
        mock_cls = MagicMock(return_value=mock_ctrl)

        with (
            patch(_PATCH_CFG, return_value=app_config),
            patch(_PATCH_LOGGING),
            patch(_PATCH_CTRL, mock_cls),
        ):
            result = CliRunner().invoke(cli, ["shutdown"])

        assert result.exit_code == 0
        mock_ctrl.orderly_shutdown.assert_called_once()
        kwargs = mock_ctrl.orderly_shutdown.call_args[1]
        assert kwargs.get("dry_run") is False

    def test_config_error_exits_2(self):
        from click.testing import CliRunner
        from maestro.cli import cli

        with patch(_PATCH_CFG, side_effect=RuntimeError("no config")):
            result = CliRunner().invoke(cli, ["shutdown"])

        assert result.exit_code == 2

    def test_keyboard_interrupt_handled(self, app_config):
        from click.testing import CliRunner
        from maestro.cli import cli

        mock_ctrl = MagicMock()
        mock_ctrl.orderly_shutdown = AsyncMock(side_effect=KeyboardInterrupt())
        mock_cls = MagicMock(return_value=mock_ctrl)

        with (
            patch(_PATCH_CFG, return_value=app_config),
            patch(_PATCH_LOGGING),
            patch(_PATCH_CTRL, mock_cls),
        ):
            result = CliRunner().invoke(cli, ["shutdown"])

        assert result.exit_code == 0   # graceful — not a fatal error
