"""
Tests for maestro.dashboard

Coverage:
    Formatting helpers:
        _fmt_latency    — None, fast, moderate, slow
        _fmt_since      — None, seconds, minutes, hours

    Dashboard.__init__:
        defaults, custom params, identity stored

    Dashboard state tracking:
        _update_state_tracking — first check, same state, state change
        _since                 — returns onset or None

    Table builders:
        _service_table  — no cluster, healthy, unhealthy, degraded, dim rows
        _network_table  — no cluster, with cluster

    Rendering:
        render() — returns a Group; header text, section labels, footer present
        render() — checking=True and action_message paths

    Async actions:
        _do_check     — success path, exception path, checking flag toggled
        _handle_key   — q, r, s, x, unknown key
        _start_all    — success, partial failure, exception
        _shutdown_all — success, partial failure, exception

    Plain text fallback:
        _run_plain — success output, check failure

    Convenience wrapper:
        run_dashboard — delegates to Dashboard.run()

    CLI command:
        dashboard_cmd — success, config error, KeyboardInterrupt
"""

import asyncio
import io
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from maestro.cli import cli
from maestro.dashboard import (
    Dashboard,
    _fmt_latency,
    _fmt_since,
    run_dashboard,
)
from maestro.platform_detect import NodeIdentity, NodeType
from maestro.services.models import ClusterHealth, ServiceResult, ServiceStatus


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_result(
    name: str,
    status: ServiceStatus = ServiceStatus.HEALTHY,
    latency_ms: float = 20.0,
) -> ServiceResult:
    return ServiceResult(
        service_name=name,
        status=status,
        latency_ms=latency_ms,
    )


def _make_cluster(**overrides: ServiceStatus) -> ClusterHealth:
    """Build a ClusterHealth with all six services; overrides set specific statuses."""
    names = ("neo4j", "chromadb", "mcp", "vllm", "tailscale", "docker")
    services = {
        n: _make_result(n, overrides.get(n, ServiceStatus.HEALTHY))
        for n in names
    }
    return ClusterHealth(services=services, checked_at=datetime.now(timezone.utc))


def _make_identity(node_type: NodeType = NodeType.HIPPOCAMPAL) -> NodeIdentity:
    return NodeIdentity(
        node_type=node_type,
        hostname="test-host",
        platform="darwin",
        ubik_root=MagicMock(),
        is_wsl=False,
        tailscale_ip="127.0.0.1",
        python_venv_path=None,
        python_activate_cmd=None,
    )


def _make_dashboard(**kw) -> Dashboard:
    """Return a Dashboard with a fake config and deterministic identity."""
    cfg = MagicMock()
    d = Dashboard(cfg, **kw)
    d._identity = _make_identity()
    return d


def _render_to_text(dashboard: Dashboard) -> str:
    """Render the dashboard to a plain-text string (no ANSI)."""
    console = Console(record=True, no_color=True, width=120)
    console.print(dashboard.render())
    return console.export_text()


# ---------------------------------------------------------------------------
# _fmt_latency
# ---------------------------------------------------------------------------

class TestFmtLatency:
    def test_none_returns_dim_dash(self):
        t = _fmt_latency(None)
        assert t.plain == "—"
        assert "dim" in str(t._spans[0].style) if t._spans else True

    def test_fast_green(self):
        t = _fmt_latency(10.0)
        assert t.plain == "10ms"
        assert "green" in str(t.style)

    def test_boundary_50ms_yellow(self):
        t = _fmt_latency(50.0)
        assert t.plain == "50ms"
        assert "yellow" in str(t.style)

    def test_moderate_yellow(self):
        t = _fmt_latency(100.0)
        assert t.plain == "100ms"
        assert "yellow" in str(t.style)

    def test_slow_red(self):
        t = _fmt_latency(300.0)
        assert t.plain == "300ms"
        assert "red" in str(t.style)

    def test_boundary_200ms_red(self):
        t = _fmt_latency(200.0)
        assert "red" in str(t.style)


# ---------------------------------------------------------------------------
# _fmt_since
# ---------------------------------------------------------------------------

class TestFmtSince:
    def test_none_returns_dash(self):
        t = _fmt_since(None)
        assert t.plain == "—"

    def test_seconds(self):
        dt = datetime.now(timezone.utc) - timedelta(seconds=45)
        t = _fmt_since(dt)
        assert t.plain.endswith("s")
        assert "45" in t.plain

    def test_minutes(self):
        dt = datetime.now(timezone.utc) - timedelta(seconds=150)
        t = _fmt_since(dt)
        assert "2m" in t.plain

    def test_hours_and_minutes(self):
        dt = datetime.now(timezone.utc) - timedelta(hours=3, minutes=12)
        t = _fmt_since(dt)
        assert "3h" in t.plain
        assert "12m" in t.plain

    def test_exactly_one_hour(self):
        dt = datetime.now(timezone.utc) - timedelta(hours=1)
        t = _fmt_since(dt)
        assert "1h" in t.plain
        assert "0m" in t.plain


# ---------------------------------------------------------------------------
# Dashboard.__init__
# ---------------------------------------------------------------------------

class TestDashboardInit:
    def test_defaults(self):
        d = Dashboard()
        assert d._interval == 30.0
        assert d._timeout == 10.0
        assert d._cluster is None
        assert d._checking is False
        assert d._last_check is None
        assert d._action_message is None

    def test_custom_params(self):
        cfg = MagicMock()
        console = Console()
        d = Dashboard(cfg, interval=60.0, timeout=5.0, console=console)
        assert d._cfg is cfg
        assert d._interval == 60.0
        assert d._timeout == 5.0
        assert d._console is console

    def test_identity_detection_failure_is_silent(self):
        with patch("maestro.dashboard.detect_node", side_effect=RuntimeError("boom")):
            d = Dashboard()
        assert d._identity is None

    def test_state_tracking_dicts_empty(self):
        d = Dashboard()
        assert d._state_start == {}
        assert d._last_healthy == {}

    def test_cfg_stored_as_none_when_not_given(self):
        d = Dashboard()
        assert d._cfg is None


# ---------------------------------------------------------------------------
# Dashboard state tracking
# ---------------------------------------------------------------------------

class TestStateTracking:
    def test_first_check_records_onset(self):
        d = _make_dashboard()
        cluster = _make_cluster()
        d._update_state_tracking(cluster)
        for name in cluster.services:
            assert name in d._state_start
            assert name in d._last_healthy

    def test_same_state_does_not_change_onset(self):
        d = _make_dashboard()
        cluster1 = _make_cluster()
        d._update_state_tracking(cluster1)
        original_times = dict(d._state_start)

        cluster2 = _make_cluster()  # all healthy again
        d._update_state_tracking(cluster2)
        assert d._state_start == original_times

    def test_state_change_updates_onset(self):
        d = _make_dashboard()
        cluster1 = _make_cluster()
        d._update_state_tracking(cluster1)
        old_onset = d._state_start.get("neo4j")

        # neo4j becomes unhealthy in second check
        cluster2 = _make_cluster(neo4j=ServiceStatus.UNHEALTHY)
        d._update_state_tracking(cluster2)

        new_onset = d._state_start.get("neo4j")
        assert new_onset is not None
        assert new_onset != old_onset

    def test_since_returns_onset_datetime(self):
        d = _make_dashboard()
        cluster = _make_cluster()
        d._update_state_tracking(cluster)
        result = d._since("neo4j")
        assert isinstance(result, datetime)

    def test_since_returns_none_for_unknown_service(self):
        d = _make_dashboard()
        assert d._since("nonexistent") is None

    def test_unhealthy_to_healthy_resets_onset(self):
        d = _make_dashboard()
        # First check: neo4j is unhealthy
        cluster1 = _make_cluster(neo4j=ServiceStatus.UNHEALTHY)
        d._update_state_tracking(cluster1)
        down_onset = d._state_start["neo4j"]

        # Second check: neo4j recovers
        cluster2 = _make_cluster()
        d._update_state_tracking(cluster2)
        up_onset = d._state_start["neo4j"]

        assert up_onset != down_onset


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

class TestServiceTable:
    def test_no_cluster_shows_unknown(self):
        d = _make_dashboard()
        table = d._service_table(("neo4j",))
        # Render to verify content
        console = Console(record=True, no_color=True)
        console.print(table)
        output = console.export_text()
        assert "UNKNOWN" in output

    def test_healthy_row_shows_up(self):
        d = _make_dashboard()
        d._cluster = _make_cluster()
        console = Console(record=True, no_color=True)
        console.print(d._service_table(("neo4j",)))
        output = console.export_text()
        assert "UP" in output

    def test_unhealthy_row_shows_down(self):
        d = _make_dashboard()
        d._cluster = _make_cluster(neo4j=ServiceStatus.UNHEALTHY)
        console = Console(record=True, no_color=True)
        console.print(d._service_table(("neo4j",)))
        output = console.export_text()
        assert "DOWN" in output

    def test_degraded_row_shows_degraded(self):
        d = _make_dashboard()
        d._cluster = _make_cluster(neo4j=ServiceStatus.DEGRADED)
        console = Console(record=True, no_color=True)
        console.print(d._service_table(("neo4j",)))
        output = console.export_text()
        assert "DEGRADED" in output

    def test_all_hippocampal_services_rendered(self):
        d = _make_dashboard()
        d._cluster = _make_cluster()
        from maestro.dashboard import _HIPPOCAMPAL_SERVICES
        console = Console(record=True, no_color=True)
        console.print(d._service_table(_HIPPOCAMPAL_SERVICES))
        output = console.export_text()
        for name in _HIPPOCAMPAL_SERVICES:
            assert name in output

    def test_dim_rows_rendered_without_error(self):
        d = _make_dashboard()
        d._cluster = _make_cluster()
        # Should not raise
        table = d._service_table(("vllm",), dim=True)
        console = Console(record=True, no_color=True)
        console.print(table)

    def test_service_absent_from_cluster_shows_unknown(self):
        d = _make_dashboard()
        d._cluster = ClusterHealth(services={}, checked_at=datetime.now(timezone.utc))
        console = Console(record=True, no_color=True)
        console.print(d._service_table(("neo4j",)))
        output = console.export_text()
        assert "UNKNOWN" in output


class TestNetworkTable:
    def test_no_cluster_shows_unknown(self):
        d = _make_dashboard()
        console = Console(record=True, no_color=True)
        console.print(d._network_table())
        output = console.export_text()
        assert "UNKNOWN" in output

    def test_healthy_tailscale_shows_up(self):
        d = _make_dashboard()
        d._cluster = _make_cluster()
        console = Console(record=True, no_color=True)
        console.print(d._network_table())
        output = console.export_text()
        assert "UP" in output

    def test_unhealthy_tailscale_shows_down(self):
        d = _make_dashboard()
        d._cluster = _make_cluster(tailscale=ServiceStatus.UNHEALTHY)
        console = Console(record=True, no_color=True)
        console.print(d._network_table())
        output = console.export_text()
        assert "DOWN" in output

    def test_network_table_has_connection_column(self):
        d = _make_dashboard()
        console = Console(record=True, no_color=True)
        console.print(d._network_table())
        output = console.export_text()
        assert "Connection" in output


# ---------------------------------------------------------------------------
# Dashboard.render()
# ---------------------------------------------------------------------------

class TestRender:
    def test_render_returns_group(self):
        from rich.console import Group
        d = _make_dashboard()
        result = d.render()
        assert isinstance(result, Group)

    def test_render_contains_title(self):
        d = _make_dashboard()
        output = _render_to_text(d)
        assert "UBIK MAESTRO" in output

    def test_render_contains_hippocampal_section(self):
        d = _make_dashboard()
        output = _render_to_text(d)
        assert "HIPPOCAMPAL" in output

    def test_render_contains_somatic_section(self):
        d = _make_dashboard()
        output = _render_to_text(d)
        assert "SOMATIC" in output

    def test_render_contains_network_section(self):
        d = _make_dashboard()
        output = _render_to_text(d)
        assert "NETWORK" in output

    def test_render_contains_footer_hints(self):
        d = _make_dashboard()
        output = _render_to_text(d)
        assert "Quit" in output
        assert "Refresh" in output

    def test_render_checking_state_shows_indicator(self):
        d = _make_dashboard()
        d._checking = True
        output = _render_to_text(d)
        assert "checking" in output

    def test_render_action_message_shown(self):
        d = _make_dashboard()
        d._action_message = "All services started."
        output = _render_to_text(d)
        assert "All services started." in output

    def test_render_last_check_shown_when_set(self):
        d = _make_dashboard()
        d._last_check = datetime(2026, 2, 25, 10, 30, 0, tzinfo=timezone.utc)
        output = _render_to_text(d)
        assert "2026-02-25" in output

    def test_render_somatic_is_remote_when_on_hippocampal(self):
        d = _make_dashboard()
        d._identity = _make_identity(NodeType.HIPPOCAMPAL)
        output = _render_to_text(d)
        assert "Remote" in output

    def test_render_somatic_is_local_when_on_somatic(self):
        d = _make_dashboard()
        d._identity = _make_identity(NodeType.SOMATIC)
        output = _render_to_text(d)
        # Somatic section should show Local
        assert "SOMATIC NODE" in output


# ---------------------------------------------------------------------------
# Dashboard._do_check
# ---------------------------------------------------------------------------

class TestDoCheck:
    @pytest.mark.asyncio
    async def test_success_updates_cluster(self):
        d = _make_dashboard()
        mock_cluster = _make_cluster()
        with patch(
            "maestro.dashboard.run_all_checks",
            new=AsyncMock(return_value=mock_cluster),
        ):
            await d._do_check()
        assert d._cluster is mock_cluster
        assert d._last_check is not None
        assert d._checking is False

    @pytest.mark.asyncio
    async def test_exception_sets_action_message(self):
        d = _make_dashboard()
        with patch(
            "maestro.dashboard.run_all_checks",
            new=AsyncMock(side_effect=RuntimeError("network down")),
        ):
            await d._do_check()
        assert d._action_message is not None
        assert "Check failed" in d._action_message
        assert d._checking is False

    @pytest.mark.asyncio
    async def test_checking_flag_cleared_on_success(self):
        d = _make_dashboard()
        mock_cluster = _make_cluster()
        with patch(
            "maestro.dashboard.run_all_checks",
            new=AsyncMock(return_value=mock_cluster),
        ):
            await d._do_check()
        assert d._checking is False

    @pytest.mark.asyncio
    async def test_action_message_cleared_on_new_check(self):
        d = _make_dashboard()
        d._action_message = "old message"
        mock_cluster = _make_cluster()
        with patch(
            "maestro.dashboard.run_all_checks",
            new=AsyncMock(return_value=mock_cluster),
        ):
            await d._do_check()
        assert d._action_message is None


# ---------------------------------------------------------------------------
# Dashboard._handle_key
# ---------------------------------------------------------------------------

class TestHandleKey:
    @pytest.mark.asyncio
    async def test_q_returns_quit(self):
        d = _make_dashboard()
        assert await d._handle_key("q") == "quit"

    @pytest.mark.asyncio
    async def test_Q_returns_quit(self):
        d = _make_dashboard()
        assert await d._handle_key("Q") == "quit"

    @pytest.mark.asyncio
    async def test_r_returns_refresh(self):
        d = _make_dashboard()
        assert await d._handle_key("r") == "refresh"

    @pytest.mark.asyncio
    async def test_R_returns_refresh(self):
        d = _make_dashboard()
        assert await d._handle_key("R") == "refresh"

    @pytest.mark.asyncio
    async def test_r_clears_action_message(self):
        d = _make_dashboard()
        d._action_message = "old"
        await d._handle_key("r")
        assert d._action_message is None

    @pytest.mark.asyncio
    async def test_s_returns_none_and_sets_message(self):
        d = _make_dashboard()
        # Patch the target method so the spawned task does not perform I/O.
        with patch.object(d, "_start_all", new_callable=AsyncMock):
            result = await d._handle_key("s")
            await asyncio.sleep(0)  # let the created task run
        assert result is None
        assert d._action_message == "Starting unhealthy services..."

    @pytest.mark.asyncio
    async def test_x_returns_none_and_sets_message(self):
        d = _make_dashboard()
        with patch.object(d, "_shutdown_all", new_callable=AsyncMock):
            result = await d._handle_key("x")
            await asyncio.sleep(0)
        assert result is None
        assert d._action_message == "Shutting down all services..."

    @pytest.mark.asyncio
    async def test_unknown_key_returns_none(self):
        d = _make_dashboard()
        assert await d._handle_key("z") is None
        assert await d._handle_key(" ") is None
        assert await d._handle_key("\n") is None


# ---------------------------------------------------------------------------
# Dashboard._start_all
# ---------------------------------------------------------------------------

class TestStartAll:
    @pytest.mark.asyncio
    async def test_success_no_failures(self):
        d = _make_dashboard()
        mock_orch = AsyncMock()
        mock_orch.ensure_all_running.return_value = []
        with (
            patch("maestro.orchestrator.Orchestrator", return_value=mock_orch),
            patch("maestro.services.ServiceRegistry"),
        ):
            await d._start_all()
        assert d._action_message == "All services started."

    @pytest.mark.asyncio
    async def test_partial_failure_lists_services(self):
        d = _make_dashboard()
        mock_orch = AsyncMock()
        mock_orch.ensure_all_running.return_value = ["neo4j", "chromadb"]
        with (
            patch("maestro.orchestrator.Orchestrator", return_value=mock_orch),
            patch("maestro.services.ServiceRegistry"),
        ):
            await d._start_all()
        assert "neo4j" in d._action_message
        assert "chromadb" in d._action_message

    @pytest.mark.asyncio
    async def test_exception_sets_error_message(self):
        d = _make_dashboard()
        with patch(
            "maestro.orchestrator.Orchestrator", side_effect=RuntimeError("boom")
        ):
            await d._start_all()
        assert "Start error" in d._action_message


# ---------------------------------------------------------------------------
# Dashboard._shutdown_all
# ---------------------------------------------------------------------------

class TestShutdownAll:
    @pytest.mark.asyncio
    async def test_success_all_stopped(self):
        d = _make_dashboard()
        mock_svc1 = AsyncMock()
        mock_svc1.stop = AsyncMock(return_value=True)
        mock_svc1.name = "neo4j"
        mock_svc2 = AsyncMock()
        mock_svc2.stop = AsyncMock(return_value=True)
        mock_svc2.name = "chromadb"
        mock_registry = MagicMock()
        mock_registry.get_all.return_value = [mock_svc1, mock_svc2]
        with patch("maestro.services.ServiceRegistry", return_value=mock_registry):
            await d._shutdown_all()
        assert d._action_message == "All services stopped."

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        d = _make_dashboard()
        mock_svc = AsyncMock()
        mock_svc.stop = AsyncMock(return_value=False)
        mock_svc.name = "neo4j"
        mock_registry = MagicMock()
        mock_registry.get_all.return_value = [mock_svc]
        with patch("maestro.services.ServiceRegistry", return_value=mock_registry):
            await d._shutdown_all()
        assert "neo4j" in d._action_message
        assert "partial" in d._action_message

    @pytest.mark.asyncio
    async def test_exception_sets_error_message(self):
        d = _make_dashboard()
        with patch(
            "maestro.services.ServiceRegistry", side_effect=RuntimeError("boom")
        ):
            await d._shutdown_all()
        assert "Shutdown error" in d._action_message


# ---------------------------------------------------------------------------
# Dashboard._run_plain
# ---------------------------------------------------------------------------

class TestRunPlain:
    def test_success_prints_header(self, capsys):
        d = _make_dashboard()
        mock_cluster = _make_cluster()
        with patch(
            "maestro.dashboard.run_all_checks",
            new=AsyncMock(return_value=mock_cluster),
        ):
            d._run_plain()
        out = capsys.readouterr().out
        assert "UBIK MAESTRO" in out

    def test_success_prints_service_names(self, capsys):
        d = _make_dashboard()
        mock_cluster = _make_cluster()
        with patch(
            "maestro.dashboard.run_all_checks",
            new=AsyncMock(return_value=mock_cluster),
        ):
            d._run_plain()
        out = capsys.readouterr().out
        assert "neo4j" in out
        assert "vllm" in out

    def test_success_prints_healthy_count(self, capsys):
        d = _make_dashboard()
        mock_cluster = _make_cluster()
        with patch(
            "maestro.dashboard.run_all_checks",
            new=AsyncMock(return_value=mock_cluster),
        ):
            d._run_plain()
        out = capsys.readouterr().out
        assert "healthy" in out

    def test_check_failure_prints_to_stderr(self, capsys):
        d = _make_dashboard()
        with patch(
            "maestro.dashboard.run_all_checks",
            new=AsyncMock(side_effect=RuntimeError("connection refused")),
        ):
            d._run_plain()
        err = capsys.readouterr().err
        assert "Health check failed" in err


# ---------------------------------------------------------------------------
# run_dashboard convenience wrapper
# ---------------------------------------------------------------------------

class TestRunDashboard:
    def test_creates_dashboard_and_calls_run(self):
        mock_run = MagicMock()
        cfg = MagicMock()
        console = Console()
        with patch("maestro.dashboard.Dashboard") as MockDash:
            MockDash.return_value.run = mock_run
            run_dashboard(cfg, interval=45.0, timeout=5.0, console=console)
        MockDash.assert_called_once_with(
            cfg, interval=45.0, timeout=5.0, console=console
        )
        mock_run.assert_called_once()

    def test_default_params(self):
        mock_run = MagicMock()
        with patch("maestro.dashboard.Dashboard") as MockDash:
            MockDash.return_value.run = mock_run
            run_dashboard()
        _, kw = MockDash.call_args
        assert kw["interval"] == 30.0
        assert kw["timeout"] == 10.0


# ---------------------------------------------------------------------------
# CLI dashboard command
# ---------------------------------------------------------------------------

_PATCH_CFG = "maestro.cli.get_config"
# run_dashboard is a local import inside dashboard_cmd, so patch the source module.
_PATCH_RUN_DASH = "maestro.dashboard.run_dashboard"


class TestDashboardCmd:
    def test_success(self):
        runner = CliRunner()
        with (
            patch(_PATCH_CFG, return_value=MagicMock()),
            patch(_PATCH_RUN_DASH),
        ):
            result = runner.invoke(cli, ["dashboard"])
        assert result.exit_code == 0

    def test_config_error_exits_2(self):
        runner = CliRunner()
        with patch(_PATCH_CFG, side_effect=ValueError("bad config")):
            result = runner.invoke(cli, ["dashboard"])
        assert result.exit_code == 2
        assert "Config error" in result.output

    def test_keyboard_interrupt_prints_stopped(self):
        runner = CliRunner()
        with (
            patch(_PATCH_CFG, return_value=MagicMock()),
            patch(_PATCH_RUN_DASH, side_effect=KeyboardInterrupt),
        ):
            result = runner.invoke(cli, ["dashboard"])
        assert result.exit_code == 0
        assert "Stopped" in result.output
