"""
Tests for maestro.daemon

Coverage:
    MaestroDaemon.__init__:
        default values, custom interval, auto_restart, cfg stored

    _init_components:
        creates orchestrator/mlog/identity on first call
        idempotent (does not reinitialize)

    PID file management:
        _pid_path uses cfg.log_dir
        _write_pid writes process PID
        _remove_pid removes the file
        _check_stale_pid — no file → no-op
        _check_stale_pid — stale (dead process) → removes file
        _check_stale_pid — live process → raises RuntimeError
        _check_stale_pid — unreadable file → removes and continues
        _check_stale_pid — PermissionError → raises RuntimeError

    Signal handlers:
        _install_signal_handlers — installs SIGTERM/SIGINT/SIGHUP
        _request_reload — sets _reload_requested
        _do_reload — clears cache, rebuilds orchestrator
        _do_reload — exception path logs failure

    Interruptible sleep:
        _sleep — times out normally
        _sleep — returns early when stop event is set
        _sleep — no stop_event falls back to asyncio.sleep

    run_once:
        calls full_status_check, logs status_check, logs cycle_start/end
        increments _cycle counter
        auto_restart=True, unhealthy local → ensure_all_running called
        auto_restart=True, all healthy → ensure_all_running NOT called
        auto_restart=False → ensure_all_running never called

    stop:
        sets _shutdown
        sets _stop_event

    run (main loop):
        calls _check_stale_pid, _write_pid, _install_signal_handlers
        runs run_once in a loop until stopped
        writes and removes PID file
        reloads when _reload_requested is set
        _check_stale_pid RuntimeError propagates before PID is written

    CLI -- watch --once:
        runs one daemon cycle and exits
        config error → exit 2
        KeyboardInterrupt → prints Stopped

    CLI -- watch --auto-restart:
        launches daemon.run() instead of _watch_loop
"""

import asyncio
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from click.testing import CliRunner

from maestro.cli import cli
from maestro.daemon import MaestroDaemon
from maestro.platform_detect import NodeIdentity, NodeType
from maestro.services.base import ProbeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_probe(
    name: str,
    healthy: bool = True,
    node: NodeType = NodeType.HIPPOCAMPAL,
) -> ProbeResult:
    return ProbeResult(
        name=name,
        node=node,
        healthy=healthy,
        latency_ms=10.0,
    )


def _make_statuses(
    names=("neo4j", "chromadb", "vllm"),
    healthy=True,
) -> dict[str, ProbeResult]:
    return {n: _make_probe(n, healthy=healthy) for n in names}


def _make_identity(node_type: NodeType = NodeType.HIPPOCAMPAL) -> NodeIdentity:
    return NodeIdentity(
        node_type=node_type,
        hostname="test-host",
        platform="darwin",
        ubik_root=Path("/tmp/ubik"),
        is_wsl=False,
        tailscale_ip="127.0.0.1",
        python_venv_path=None,
        python_activate_cmd=None,
    )


def _make_daemon(
    interval: int = 60,
    auto_restart: bool = False,
    cfg=None,
) -> MaestroDaemon:
    d = MaestroDaemon(check_interval_s=interval, auto_restart=auto_restart, cfg=cfg)
    # Pre-populate internal state so tests don't need real config.
    d._identity = _make_identity()
    d._mlog = MagicMock()
    d._orchestrator = MagicMock()
    return d


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestMaestroDaemonInit:
    def test_default_values(self):
        d = MaestroDaemon(check_interval_s=300)
        assert d._interval == 300.0
        assert d._auto_restart is False
        assert d._cfg is None
        assert d._shutdown is False
        assert d._reload_requested is False
        assert d._cycle == 0

    def test_custom_interval_and_auto_restart(self):
        d = MaestroDaemon(check_interval_s=60, auto_restart=True)
        assert d._interval == 60.0
        assert d._auto_restart is True

    def test_cfg_stored(self):
        cfg = MagicMock()
        d = MaestroDaemon(check_interval_s=30, cfg=cfg)
        assert d._cfg is cfg

    def test_components_not_initialized(self):
        d = MaestroDaemon(check_interval_s=10)
        assert d._orchestrator is None
        assert d._mlog is None
        assert d._identity is None


# ---------------------------------------------------------------------------
# _init_components
# ---------------------------------------------------------------------------

class TestInitComponents:
    def test_creates_orchestrator_and_logger(self, tmp_path):
        cfg = MagicMock()
        cfg.log_dir = tmp_path
        d = MaestroDaemon(check_interval_s=30, cfg=cfg)
        with (
            patch("maestro.daemon.detect_node", return_value=_make_identity()),
            patch("maestro.daemon.MaestroLogger"),
            patch("maestro.daemon.ServiceRegistry"),
            patch("maestro.daemon.Orchestrator"),
        ):
            d._init_components()
        assert d._orchestrator is not None
        assert d._mlog is not None
        assert d._identity is not None

    def test_idempotent(self, tmp_path):
        cfg = MagicMock()
        cfg.log_dir = tmp_path
        d = MaestroDaemon(check_interval_s=30, cfg=cfg)
        sentinel = MagicMock()
        d._orchestrator = sentinel
        d._init_components()  # should not replace the existing orchestrator
        assert d._orchestrator is sentinel


# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------

class TestPidManagement:
    def test_pid_path_uses_log_dir(self, tmp_path):
        cfg = MagicMock()
        cfg.log_dir = tmp_path
        d = MaestroDaemon(check_interval_s=60, cfg=cfg)
        assert d._pid_path() == tmp_path / "maestro.pid"

    def test_write_pid_creates_file(self, tmp_path):
        cfg = MagicMock()
        cfg.log_dir = tmp_path
        d = MaestroDaemon(check_interval_s=60, cfg=cfg)
        d._write_pid()
        pid_file = tmp_path / "maestro.pid"
        assert pid_file.exists()
        assert int(pid_file.read_text()) == os.getpid()

    def test_remove_pid_deletes_file(self, tmp_path):
        cfg = MagicMock()
        cfg.log_dir = tmp_path
        d = MaestroDaemon(check_interval_s=60, cfg=cfg)
        (tmp_path / "maestro.pid").write_text("12345")
        d._remove_pid()
        assert not (tmp_path / "maestro.pid").exists()

    def test_check_stale_pid_no_file(self, tmp_path):
        cfg = MagicMock()
        cfg.log_dir = tmp_path
        d = MaestroDaemon(check_interval_s=60, cfg=cfg)
        d._check_stale_pid()  # should not raise

    def test_check_stale_pid_dead_process(self, tmp_path):
        cfg = MagicMock()
        cfg.log_dir = tmp_path
        d = MaestroDaemon(check_interval_s=60, cfg=cfg)
        pid_file = tmp_path / "maestro.pid"
        # Use a PID that is guaranteed not to exist
        pid_file.write_text("99999999")
        with patch("os.kill", side_effect=ProcessLookupError):
            d._check_stale_pid()  # should remove stale file and not raise
        assert not pid_file.exists()

    def test_check_stale_pid_live_process_raises(self, tmp_path):
        cfg = MagicMock()
        cfg.log_dir = tmp_path
        d = MaestroDaemon(check_interval_s=60, cfg=cfg)
        (tmp_path / "maestro.pid").write_text("12345")
        # os.kill(pid, 0) succeeds → process is alive
        with patch("os.kill", return_value=None):
            with pytest.raises(RuntimeError, match="already running"):
                d._check_stale_pid()

    def test_check_stale_pid_permission_error_raises(self, tmp_path):
        cfg = MagicMock()
        cfg.log_dir = tmp_path
        d = MaestroDaemon(check_interval_s=60, cfg=cfg)
        (tmp_path / "maestro.pid").write_text("12345")
        with patch("os.kill", side_effect=PermissionError):
            with pytest.raises(RuntimeError, match="appears to be running"):
                d._check_stale_pid()

    def test_check_stale_pid_unreadable_removes_and_continues(self, tmp_path):
        cfg = MagicMock()
        cfg.log_dir = tmp_path
        d = MaestroDaemon(check_interval_s=60, cfg=cfg)
        pid_file = tmp_path / "maestro.pid"
        pid_file.write_text("not-a-number")
        d._check_stale_pid()  # should remove and not raise
        assert not pid_file.exists()


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

class TestSignalHandlers:
    def test_request_reload_sets_flag(self):
        d = _make_daemon()
        assert d._reload_requested is False
        d._request_reload()
        assert d._reload_requested is True

    def test_install_signal_handlers_no_crash_outside_loop(self):
        d = _make_daemon()
        # Calling outside a running event loop should not raise.
        d._install_signal_handlers()

    def test_do_reload_clears_cache_and_rebuilds(self):
        d = _make_daemon()
        d._reload_requested = True
        cfg = MagicMock()
        with (
            patch("maestro.daemon.get_config", return_value=cfg) as mock_cfg,
            patch("maestro.daemon.ServiceRegistry"),
            patch("maestro.daemon.Orchestrator") as MockOrch,
        ):
            d._do_reload()
        assert d._reload_requested is False
        mock_cfg.cache_clear.assert_called_once()
        MockOrch.assert_called_once()

    def test_do_reload_exception_logs_failure(self):
        d = _make_daemon()
        with patch("maestro.daemon.get_config", side_effect=RuntimeError("oops")):
            d._do_reload()  # should not raise
        d._mlog.log_service_action.assert_called_with(
            "daemon", "reload", False, "oops"
        )


# ---------------------------------------------------------------------------
# Interruptible sleep
# ---------------------------------------------------------------------------

class TestSleep:
    @pytest.mark.asyncio
    async def test_sleep_times_out(self):
        d = _make_daemon()
        d._stop_event = asyncio.Event()
        # Sleep for 0.05s — should return without the event being set.
        await asyncio.wait_for(d._sleep(0.05), timeout=2.0)

    @pytest.mark.asyncio
    async def test_sleep_interrupted_by_stop(self):
        d = _make_daemon()
        d._stop_event = asyncio.Event()

        async def _set_after_delay():
            await asyncio.sleep(0.02)
            d._stop_event.set()

        asyncio.create_task(_set_after_delay())
        # Would normally sleep for 10s but stop event fires at 20ms.
        await asyncio.wait_for(d._sleep(10.0), timeout=2.0)

    @pytest.mark.asyncio
    async def test_sleep_no_event_uses_asyncio_sleep(self):
        d = _make_daemon()
        d._stop_event = None
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await d._sleep(5.0)
        mock_sleep.assert_called_once_with(5.0)


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------

class TestRunOnce:
    @pytest.mark.asyncio
    async def test_calls_full_status_check(self):
        d = _make_daemon()
        statuses = _make_statuses()
        d._orchestrator.full_status_check = AsyncMock(return_value=statuses)
        await d.run_once()
        d._orchestrator.full_status_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_logs_status_check(self):
        d = _make_daemon()
        statuses = _make_statuses()
        d._orchestrator.full_status_check = AsyncMock(return_value=statuses)
        await d.run_once()
        d._mlog.log_status_check.assert_called_once_with(statuses)

    @pytest.mark.asyncio
    async def test_increments_cycle_counter(self):
        d = _make_daemon()
        d._orchestrator.full_status_check = AsyncMock(return_value={})
        assert d._cycle == 0
        await d.run_once()
        assert d._cycle == 1
        await d.run_once()
        assert d._cycle == 2

    @pytest.mark.asyncio
    async def test_auto_restart_triggers_when_unhealthy_local(self):
        d = _make_daemon(auto_restart=True)
        # neo4j is unhealthy and on the hippocampal (local) node.
        statuses = {
            "neo4j": _make_probe("neo4j", healthy=False, node=NodeType.HIPPOCAMPAL),
        }
        d._orchestrator.full_status_check = AsyncMock(return_value=statuses)
        d._orchestrator.ensure_all_running = AsyncMock(return_value=[])
        await d.run_once()
        d._orchestrator.ensure_all_running.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_restart_skipped_when_all_healthy(self):
        d = _make_daemon(auto_restart=True)
        statuses = _make_statuses(healthy=True)
        d._orchestrator.full_status_check = AsyncMock(return_value=statuses)
        d._orchestrator.ensure_all_running = AsyncMock(return_value=[])
        await d.run_once()
        d._orchestrator.ensure_all_running.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_restart_false_never_restarts(self):
        d = _make_daemon(auto_restart=False)
        statuses = {
            "neo4j": _make_probe("neo4j", healthy=False, node=NodeType.HIPPOCAMPAL),
        }
        d._orchestrator.full_status_check = AsyncMock(return_value=statuses)
        d._orchestrator.ensure_all_running = AsyncMock(return_value=[])
        await d.run_once()
        d._orchestrator.ensure_all_running.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_restart_skips_remote_unhealthy(self):
        d = _make_daemon(auto_restart=True)
        # vllm is unhealthy but lives on the SOMATIC (remote) node.
        statuses = {
            "vllm": _make_probe("vllm", healthy=False, node=NodeType.SOMATIC),
        }
        d._orchestrator.full_status_check = AsyncMock(return_value=statuses)
        d._orchestrator.ensure_all_running = AsyncMock(return_value=[])
        await d.run_once()
        d._orchestrator.ensure_all_running.assert_not_called()

    @pytest.mark.asyncio
    async def test_logs_cycle_start_and_end(self):
        d = _make_daemon()
        d._orchestrator.full_status_check = AsyncMock(return_value={})
        await d.run_once()
        calls = [c.args[1] for c in d._mlog.log_service_action.call_args_list]
        assert "cycle_start" in calls
        assert "cycle_end" in calls

    @pytest.mark.asyncio
    async def test_returns_statuses_dict(self):
        d = _make_daemon()
        statuses = _make_statuses()
        d._orchestrator.full_status_check = AsyncMock(return_value=statuses)
        result = await d.run_once()
        assert result is statuses

    @pytest.mark.asyncio
    async def test_initializes_components_when_needed(self, tmp_path):
        cfg = MagicMock()
        cfg.log_dir = tmp_path
        d = MaestroDaemon(check_interval_s=30, cfg=cfg)
        assert d._orchestrator is None
        with (
            patch("maestro.daemon.detect_node", return_value=_make_identity()),
            patch("maestro.daemon.MaestroLogger") as MockLogger,
            patch("maestro.daemon.ServiceRegistry"),
            patch("maestro.daemon.Orchestrator") as MockOrch,
        ):
            mock_orch_inst = MagicMock()
            mock_orch_inst.full_status_check = AsyncMock(return_value={})
            MockOrch.return_value = mock_orch_inst
            MockLogger.return_value = MagicMock()
            await d.run_once()
        assert d._orchestrator is not None


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

class TestStop:
    def test_sets_shutdown_flag(self):
        d = _make_daemon()
        assert d._shutdown is False
        d.stop()
        assert d._shutdown is True

    def test_sets_stop_event(self):
        d = _make_daemon()
        event = asyncio.Event()
        d._stop_event = event
        d.stop()
        assert event.is_set()

    def test_stop_without_event_does_not_raise(self):
        d = _make_daemon()
        d._stop_event = None
        d.stop()  # should not raise
        assert d._shutdown is True


# ---------------------------------------------------------------------------
# run (main loop)
# ---------------------------------------------------------------------------

class TestRun:
    @pytest.mark.asyncio
    async def test_writes_and_removes_pid(self, tmp_path):
        cfg = MagicMock()
        cfg.log_dir = tmp_path
        d = _make_daemon(cfg=cfg)
        d._cfg = cfg

        call_count = [0]

        async def _one_shot():
            call_count[0] += 1
            d.stop()
            return {}

        with (
            patch.object(d, "_check_stale_pid"),
            patch.object(d, "_install_signal_handlers"),
            patch.object(d, "_init_components"),
            patch.object(d, "run_once", side_effect=_one_shot),
        ):
            await d.run()

        # PID file should be removed after run()
        assert not (tmp_path / "maestro.pid").exists()
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_runs_multiple_cycles(self):
        d = _make_daemon()
        cycles = [0]

        async def _mock_run_once():
            cycles[0] += 1
            if cycles[0] >= 3:
                d.stop()
            return {}

        # AsyncMock() as `new` so _sleep(seconds) is awaitable with no real delay.
        with (
            patch.object(d, "_check_stale_pid"),
            patch.object(d, "_write_pid"),
            patch.object(d, "_remove_pid"),
            patch.object(d, "_install_signal_handlers"),
            patch.object(d, "_init_components"),
            patch.object(d, "_sleep", AsyncMock()),
            patch.object(d, "run_once", side_effect=_mock_run_once),
        ):
            await d.run()

        assert cycles[0] == 3

    @pytest.mark.asyncio
    async def test_check_stale_pid_error_propagates(self):
        d = _make_daemon()
        # Keep a direct reference so we can assert after the patch exits.
        mock_write = MagicMock()
        with (
            patch.object(d, "_check_stale_pid", side_effect=RuntimeError("already running")),
            patch.object(d, "_write_pid", mock_write),
            patch.object(d, "_install_signal_handlers"),
            patch.object(d, "_init_components"),
        ):
            with pytest.raises(RuntimeError, match="already running"):
                await d.run()
        # _write_pid must not have been called — error fires before it.
        mock_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_reload_triggered_on_sighup_flag(self):
        d = _make_daemon()
        reloaded = [False]

        async def _one_shot_with_reload():
            d.stop()
            return {}

        def _mock_reload():
            reloaded[0] = True
            d._reload_requested = False

        d._reload_requested = True

        with (
            patch.object(d, "_check_stale_pid"),
            patch.object(d, "_write_pid"),
            patch.object(d, "_remove_pid"),
            patch.object(d, "_install_signal_handlers"),
            patch.object(d, "_init_components"),
            patch.object(d, "_do_reload", side_effect=_mock_reload),
            patch.object(d, "run_once", side_effect=_one_shot_with_reload),
        ):
            await d.run()

        assert reloaded[0] is True

    @pytest.mark.asyncio
    async def test_logs_shutdown_on_exit(self):
        d = _make_daemon()

        async def _stop_immediately():
            d.stop()
            return {}

        with (
            patch.object(d, "_check_stale_pid"),
            patch.object(d, "_write_pid"),
            patch.object(d, "_remove_pid"),
            patch.object(d, "_install_signal_handlers"),
            patch.object(d, "_init_components"),
            patch.object(d, "run_once", side_effect=_stop_immediately),
        ):
            await d.run()

        d._mlog.log_shutdown.assert_called_once_with([])


# ---------------------------------------------------------------------------
# CLI — watch --once
# ---------------------------------------------------------------------------

_PATCH_CFG = "maestro.cli.get_config"
_PATCH_DAEMON = "maestro.daemon.MaestroDaemon"


class TestWatchOnce:
    def test_once_flag_runs_one_cycle(self):
        runner = CliRunner()
        mock_run_once = AsyncMock(return_value={})
        mock_daemon = MagicMock()
        mock_daemon.run_once = mock_run_once
        with (
            patch(_PATCH_CFG, return_value=MagicMock()),
            patch("maestro.cli.configure_logging"),
            patch("maestro.daemon.MaestroDaemon", return_value=mock_daemon),
        ):
            result = runner.invoke(cli, ["watch", "--once"])
        assert result.exit_code == 0

    def test_once_config_error_exits_2(self):
        runner = CliRunner()
        with patch(_PATCH_CFG, side_effect=ValueError("bad config")):
            result = runner.invoke(cli, ["watch", "--once"])
        assert result.exit_code == 2
        assert "Config error" in result.output

    def test_once_keyboard_interrupt_prints_stopped(self):
        runner = CliRunner()
        mock_daemon = MagicMock()
        mock_daemon.run_once.side_effect = KeyboardInterrupt
        with (
            patch(_PATCH_CFG, return_value=MagicMock()),
            patch("maestro.cli.configure_logging"),
            patch("maestro.daemon.MaestroDaemon", return_value=mock_daemon),
        ):
            result = runner.invoke(cli, ["watch", "--once"])
        assert "Stopped" in result.output


class TestWatchAutoRestart:
    def test_auto_restart_launches_daemon_run(self):
        runner = CliRunner()
        mock_run = AsyncMock()
        mock_daemon = MagicMock()
        mock_daemon.run = mock_run
        with (
            patch(_PATCH_CFG, return_value=MagicMock()),
            patch("maestro.cli.configure_logging"),
            patch("maestro.daemon.MaestroDaemon", return_value=mock_daemon),
        ):
            result = runner.invoke(cli, ["watch", "--auto-restart"])
        assert result.exit_code == 0

    def test_auto_restart_passes_flag_to_daemon(self):
        runner = CliRunner()
        mock_daemon_cls = MagicMock()
        mock_daemon_cls.return_value.run = AsyncMock()
        with (
            patch(_PATCH_CFG, return_value=MagicMock()),
            patch("maestro.cli.configure_logging"),
            patch("maestro.daemon.MaestroDaemon", mock_daemon_cls),
        ):
            runner.invoke(cli, ["watch", "--auto-restart", "--interval", "120"])
        _, kw = mock_daemon_cls.call_args
        assert kw.get("auto_restart") is True
        assert kw.get("check_interval_s") == 120
