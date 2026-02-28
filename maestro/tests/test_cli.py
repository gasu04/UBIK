"""
Tests for maestro.cli and maestro.display

All tests use Click's CliRunner and patch both ``get_config`` and
``run_selected_checks`` (or ``run_all_checks``) so no real services or
UBIK filesystem layout is needed.

Test coverage:
    CLI — check command:
        exit code 0 when all healthy
        exit code 1 when any degraded
        exit code 2 when any unhealthy
        --json flag produces valid JSON with correct overall_status
        --service flag restricts which services are checked
        config error → exit 2 with message

    CLI — watch command:
        KeyboardInterrupt stops cleanly (exit 0)
        bad config → exit 2

    Display — format_details:
        correct output for each service type
        fallback for unknown service with error

    Display — make_service_table:
        table contains all service names present in cluster
        partial cluster (filtered services) renders without error

    Display — make_dashboard:
        returns renderable when cluster is None (init state)
        returns Group when cluster is populated
        checking=True changes header text
"""

import json
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from maestro.cli import cli
from maestro.display import format_details, make_dashboard, make_service_table
from maestro.services.models import ClusterHealth, ServiceResult, ServiceStatus


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _result(name: str, status: ServiceStatus, **kw) -> ServiceResult:
    return ServiceResult(service_name=name, status=status, latency_ms=5.0, **kw)


def _make_cluster(**overrides: ServiceStatus) -> ClusterHealth:
    """Build a ClusterHealth with all six services at given statuses.

    Keyword args map service name → ServiceStatus.  Any service not
    specified defaults to HEALTHY.
    """
    names = ("neo4j", "chromadb", "mcp", "vllm", "tailscale", "docker")
    services = {
        n: _result(n, overrides.get(n, ServiceStatus.HEALTHY))
        for n in names
    }
    return ClusterHealth(
        services=services,
        checked_at=datetime.now(timezone.utc),
    )


_PATCH_CONFIG = "maestro.cli.get_config"
_PATCH_RUN = "maestro.cli.run_selected_checks"
_PATCH_RUN_ALL = "maestro.cli.run_all_checks"
_PATCH_LOGGING = "maestro.cli.configure_logging"
_PATCH_LOG_HEALTH = "maestro.cli.log_cluster_health"


# ---------------------------------------------------------------------------
# check command — exit codes
# ---------------------------------------------------------------------------

class TestCheckExitCodes:
    def test_exit_0_when_all_healthy(self, app_config):
        runner = CliRunner()
        cluster = _make_cluster()
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            result = runner.invoke(cli, ["check"])
        assert result.exit_code == 0

    def test_exit_1_when_any_degraded(self, app_config):
        runner = CliRunner()
        cluster = _make_cluster(neo4j=ServiceStatus.DEGRADED)
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            result = runner.invoke(cli, ["check"])
        assert result.exit_code == 1

    def test_exit_2_when_any_unhealthy(self, app_config):
        runner = CliRunner()
        cluster = _make_cluster(vllm=ServiceStatus.UNHEALTHY)
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            result = runner.invoke(cli, ["check"])
        assert result.exit_code == 2

    def test_exit_2_takes_precedence_over_degraded(self, app_config):
        runner = CliRunner()
        cluster = _make_cluster(
            neo4j=ServiceStatus.DEGRADED,
            vllm=ServiceStatus.UNHEALTHY,
        )
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            result = runner.invoke(cli, ["check"])
        assert result.exit_code == 2

    def test_config_error_exits_2(self):
        runner = CliRunner()
        with patch(_PATCH_CONFIG, side_effect=RuntimeError("drive not mounted")):
            result = runner.invoke(cli, ["check"])
        assert result.exit_code == 2
        assert "Config error" in result.output


# ---------------------------------------------------------------------------
# check command — JSON output
# ---------------------------------------------------------------------------

class TestCheckJsonOutput:
    def test_json_flag_produces_valid_json(self, app_config):
        runner = CliRunner()
        cluster = _make_cluster()
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            result = runner.invoke(cli, ["check", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["overall_status"] == "healthy"

    def test_json_contains_all_services(self, app_config):
        runner = CliRunner()
        cluster = _make_cluster()
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            result = runner.invoke(cli, ["check", "--json"])
        data = json.loads(result.output)
        assert set(data["services"].keys()) == {
            "neo4j", "chromadb", "mcp", "vllm", "tailscale", "docker"
        }

    def test_json_unhealthy_status_reflected(self, app_config):
        runner = CliRunner()
        cluster = _make_cluster(vllm=ServiceStatus.UNHEALTHY)
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            result = runner.invoke(cli, ["check", "--json"])
        data = json.loads(result.output)
        assert data["overall_status"] == "unhealthy"
        assert data["services"]["vllm"]["status"] == "unhealthy"

    def test_json_no_table_markup_in_output(self, app_config):
        runner = CliRunner()
        cluster = _make_cluster()
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            result = runner.invoke(cli, ["check", "--json"])
        # Output should parse cleanly as JSON — no Rich markup leaking
        json.loads(result.output)  # raises if not valid JSON


# ---------------------------------------------------------------------------
# check command — --service filter
# ---------------------------------------------------------------------------

class TestCheckServiceFilter:
    def test_service_flag_passes_correct_set(self, app_config):
        runner = CliRunner()
        cluster = ClusterHealth(
            services={"neo4j": _result("neo4j", ServiceStatus.HEALTHY)}
        )
        mock_run = AsyncMock(return_value=cluster)
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=mock_run),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            runner.invoke(cli, ["check", "--service", "neo4j"])

        call_args = mock_run.call_args
        assert call_args[0][1] == {"neo4j"}

    def test_multiple_service_flags_union(self, app_config):
        runner = CliRunner()
        cluster = ClusterHealth(services={
            "neo4j": _result("neo4j", ServiceStatus.HEALTHY),
            "chromadb": _result("chromadb", ServiceStatus.HEALTHY),
        })
        mock_run = AsyncMock(return_value=cluster)
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=mock_run),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            runner.invoke(cli, ["check", "--service", "neo4j",
                                 "--service", "chromadb"])

        call_args = mock_run.call_args
        assert call_args[0][1] == {"neo4j", "chromadb"}

    def test_invalid_service_name_rejected_by_click(self, app_config):
        runner = CliRunner()
        with patch(_PATCH_CONFIG, return_value=app_config):
            result = runner.invoke(cli, ["check", "--service", "nonexistent"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# watch command
# ---------------------------------------------------------------------------

class TestWatchCommand:
    def test_keyboard_interrupt_exits_cleanly(self, app_config):
        """Watch stops with exit 0 on KeyboardInterrupt."""
        runner = CliRunner()

        async def _interrupt(*a, **kw):
            raise KeyboardInterrupt

        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch("maestro.cli._watch_loop", new=_interrupt),
            patch(_PATCH_LOGGING),
        ):
            result = runner.invoke(cli, ["watch"])
        # KeyboardInterrupt is caught; process exits 0 (Click default)
        assert result.exit_code == 0
        assert "Stopped" in result.output

    def test_config_error_exits_2(self):
        runner = CliRunner()
        with patch(_PATCH_CONFIG, side_effect=RuntimeError("no config")):
            result = runner.invoke(cli, ["watch"])
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# display — format_details
# ---------------------------------------------------------------------------

class TestFormatDetails:
    def test_neo4j_healthy(self):
        r = ServiceResult(
            "neo4j",
            ServiceStatus.HEALTHY,
            details={"node_count": 47, "has_core_identity": True},
        )
        text = format_details(r)
        assert "47 nodes" in text
        assert "✓" in text

    def test_neo4j_missing_core_identity(self):
        r = ServiceResult(
            "neo4j",
            ServiceStatus.DEGRADED,
            details={"node_count": 0, "has_core_identity": False},
        )
        text = format_details(r)
        assert "✗" in text

    def test_chromadb_both_collections(self):
        r = ServiceResult(
            "chromadb",
            ServiceStatus.HEALTHY,
            details={
                "collections_found": ["ubik_episodic", "ubik_semantic"],
                "missing_collections": [],
            },
        )
        text = format_details(r)
        assert "2/2" in text

    def test_chromadb_missing_collection(self):
        r = ServiceResult(
            "chromadb",
            ServiceStatus.DEGRADED,
            details={
                "collections_found": ["ubik_episodic"],
                "missing_collections": ["ubik_semantic"],
            },
        )
        text = format_details(r)
        assert "missing" in text
        assert "ubik_semantic" in text

    def test_mcp_shows_http_status(self):
        r = ServiceResult(
            "mcp",
            ServiceStatus.HEALTHY,
            details={"http_status": 406},
        )
        text = format_details(r)
        assert "406" in text

    def test_vllm_with_model(self):
        r = ServiceResult(
            "vllm",
            ServiceStatus.HEALTHY,
            details={"models_loaded": 1, "model_ids": ["DeepSeek-R1"]},
        )
        text = format_details(r)
        assert "1" in text
        assert "DeepSeek" in text

    def test_vllm_no_models(self):
        r = ServiceResult(
            "vllm",
            ServiceStatus.DEGRADED,
            details={"models_loaded": 0, "model_ids": []},
        )
        text = format_details(r)
        assert "no models" in text

    def test_tailscale_both_online(self):
        r = ServiceResult(
            "tailscale",
            ServiceStatus.HEALTHY,
            details={
                "self_online": True,
                "somatic_online": True,
                "somatic_hostname": "somatic-node",
                "peer_count": 3,
            },
        )
        text = format_details(r)
        assert "online" in text
        assert "somatic-node" in text

    def test_docker_all_running(self):
        r = ServiceResult(
            "docker",
            ServiceStatus.HEALTHY,
            details={
                "daemon_ok": True,
                "containers": {
                    "ubik-neo4j": "running",
                    "ubik-chromadb": "running",
                },
            },
        )
        text = format_details(r)
        assert "2/2" in text

    def test_unknown_service_falls_back_to_error(self):
        r = ServiceResult(
            "unknown_svc",
            ServiceStatus.UNHEALTHY,
            error="Something broke",
        )
        text = format_details(r)
        assert "Something broke" in text

    def test_no_details_no_error_returns_dash(self):
        r = ServiceResult("neo4j", ServiceStatus.HEALTHY)
        text = format_details(r)
        assert text  # should be non-empty, at least a dash


# ---------------------------------------------------------------------------
# display — make_service_table
# ---------------------------------------------------------------------------

class TestMakeServiceTable:
    def test_table_contains_all_services(self):
        cluster = _make_cluster()
        table = make_service_table(cluster)
        # Rich Table stores rows; check row count
        assert table.row_count == 6

    def test_partial_cluster_renders_only_requested_services(self):
        cluster = ClusterHealth(services={
            "neo4j": _result("neo4j", ServiceStatus.HEALTHY),
            "vllm": _result("vllm", ServiceStatus.UNHEALTHY),
        })
        table = make_service_table(cluster)
        assert table.row_count == 2

    def test_empty_cluster_renders_empty_table(self):
        cluster = ClusterHealth(services={})
        table = make_service_table(cluster)
        assert table.row_count == 0


# ---------------------------------------------------------------------------
# display — make_dashboard
# ---------------------------------------------------------------------------

class TestMakeDashboard:
    def test_none_cluster_returns_renderable(self, app_config):
        renderable = make_dashboard(app_config, None, 1, 0.0)
        assert renderable is not None

    def test_populated_cluster_returns_group(self, app_config):
        from rich.console import Group
        cluster = _make_cluster()
        renderable = make_dashboard(app_config, cluster, 1, 30.0)
        assert isinstance(renderable, Group)

    def test_checking_flag_accepted(self, app_config):
        cluster = _make_cluster()
        # Should not raise
        make_dashboard(app_config, cluster, 2, 0.0, checking=True)

    def test_dashboard_renderable_by_console(self, app_config):
        """Ensure Rich can actually render the dashboard without errors."""
        from io import StringIO
        from rich.console import Console as _Console
        cluster = _make_cluster(vllm=ServiceStatus.UNHEALTHY)
        renderable = make_dashboard(app_config, cluster, 3, 12.5)
        buf = StringIO()
        c = _Console(file=buf, no_color=True, width=120)
        c.print(renderable)
        output = buf.getvalue()
        # All service names should appear somewhere in the rendered output
        for svc in ("neo4j", "chromadb", "mcp", "vllm", "tailscale", "docker"):
            assert svc in output


# ---------------------------------------------------------------------------
# status command (mirrors check command; both share the same logic)
# ---------------------------------------------------------------------------

class TestStatusCommand:
    def test_exit_0_when_all_healthy(self, app_config):
        runner = CliRunner()
        cluster = _make_cluster()
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0

    def test_exit_2_when_unhealthy(self, app_config):
        runner = CliRunner()
        cluster = _make_cluster(neo4j=ServiceStatus.UNHEALTHY)
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 2

    def test_json_flag_produces_valid_json(self, app_config):
        runner = CliRunner()
        cluster = _make_cluster()
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            result = runner.invoke(cli, ["status", "--json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert data["overall_status"] == "healthy"

    def test_config_error_exits_2(self):
        runner = CliRunner()
        with patch(_PATCH_CONFIG, side_effect=RuntimeError("drive not mounted")):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 2
        assert "Config error" in result.output

    def test_service_filter_accepted(self, app_config):
        runner = CliRunner()
        cluster = ClusterHealth(
            services={"neo4j": _result("neo4j", ServiceStatus.HEALTHY)}
        )
        mock_run = AsyncMock(return_value=cluster)
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN, new=mock_run),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
        ):
            runner.invoke(cli, ["status", "--service", "neo4j"])
        assert mock_run.call_args[0][1] == {"neo4j"}


# ---------------------------------------------------------------------------
# metrics command
# ---------------------------------------------------------------------------

_PATCH_METRICS_CLS = "maestro.metrics.MetricsCollector"
_PATCH_ORCH_CLS = "maestro.orchestrator.Orchestrator"
_PATCH_REGISTRY_CLS = "maestro.services.ServiceRegistry"
_PATCH_DETECT_NODE = "maestro.platform_detect.detect_node"


class TestMetricsCommand:
    def _mock_collector(self, fake_metrics=None, report="UBIK Metrics\nsome data"):
        from maestro.metrics import UsageMetrics
        from datetime import datetime, timezone

        if fake_metrics is None:
            fake_metrics = UsageMetrics(timestamp=datetime.now(timezone.utc))

        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_inst.collect = AsyncMock(return_value=fake_metrics)
        mock_inst.format_report = MagicMock(return_value=report)
        mock_cls.return_value = mock_inst
        return mock_cls

    def test_metrics_prints_report(self, app_config):
        runner = CliRunner()
        mock_cls = self._mock_collector()
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_METRICS_CLS, new=mock_cls),
            patch(_PATCH_ORCH_CLS),
            patch(_PATCH_REGISTRY_CLS),
            patch(_PATCH_DETECT_NODE),
        ):
            result = runner.invoke(cli, ["metrics"])
        assert result.exit_code == 0
        assert "UBIK Metrics" in result.output

    def test_config_error_exits_2(self):
        runner = CliRunner()
        with patch(_PATCH_CONFIG, side_effect=RuntimeError("no config")):
            result = runner.invoke(cli, ["metrics"])
        assert result.exit_code == 2

    def test_collect_exception_exits_2(self, app_config):
        runner = CliRunner()
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_inst.collect = AsyncMock(side_effect=RuntimeError("service down"))
        mock_cls.return_value = mock_inst
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_METRICS_CLS, new=mock_cls),
            patch(_PATCH_ORCH_CLS),
            patch(_PATCH_REGISTRY_CLS),
            patch(_PATCH_DETECT_NODE),
        ):
            result = runner.invoke(cli, ["metrics"])
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# health command
# ---------------------------------------------------------------------------

class TestHealthCommand:
    def _setup(self, app_config, cluster, fake_metrics=None, report="metrics line"):
        """Return context manager stack for health command tests."""
        from maestro.metrics import UsageMetrics
        from datetime import datetime, timezone

        if fake_metrics is None:
            fake_metrics = UsageMetrics(timestamp=datetime.now(timezone.utc))

        mock_mc_cls = MagicMock()
        mock_mc = MagicMock()
        mock_mc.collect = AsyncMock(return_value=fake_metrics)
        mock_mc.format_report = MagicMock(return_value=report)
        mock_mc_cls.return_value = mock_mc

        return (mock_mc_cls, mock_mc)

    def test_exit_0_when_all_healthy(self, app_config):
        runner = CliRunner()
        cluster = _make_cluster()
        mock_mc_cls, _ = self._setup(app_config, cluster)
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN_ALL, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
            patch(_PATCH_METRICS_CLS, new=mock_mc_cls),
            patch(_PATCH_ORCH_CLS),
            patch(_PATCH_REGISTRY_CLS),
            patch(_PATCH_DETECT_NODE),
        ):
            result = runner.invoke(cli, ["health"])
        assert result.exit_code == 0

    def test_exit_2_when_unhealthy(self, app_config):
        runner = CliRunner()
        cluster = _make_cluster(vllm=ServiceStatus.UNHEALTHY)
        mock_mc_cls, _ = self._setup(app_config, cluster)
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN_ALL, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
            patch(_PATCH_METRICS_CLS, new=mock_mc_cls),
            patch(_PATCH_ORCH_CLS),
            patch(_PATCH_REGISTRY_CLS),
            patch(_PATCH_DETECT_NODE),
        ):
            result = runner.invoke(cli, ["health"])
        assert result.exit_code == 2

    def test_json_flag_combined_output(self, app_config):
        import json
        runner = CliRunner()
        cluster = _make_cluster()
        mock_mc_cls, _ = self._setup(app_config, cluster)
        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN_ALL, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
            patch(_PATCH_METRICS_CLS, new=mock_mc_cls),
            patch(_PATCH_ORCH_CLS),
            patch(_PATCH_REGISTRY_CLS),
            patch(_PATCH_DETECT_NODE),
        ):
            result = runner.invoke(cli, ["health", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "status" in data
        assert "metrics" in data
        assert data["status"]["overall_status"] == "healthy"

    def test_config_error_exits_2(self):
        runner = CliRunner()
        with patch(_PATCH_CONFIG, side_effect=RuntimeError("bad config")):
            result = runner.invoke(cli, ["health"])
        assert result.exit_code == 2

    def test_metrics_failure_shown_as_unavailable(self, app_config):
        """When metrics collect raises, report shows 'Metrics unavailable'."""
        runner = CliRunner()
        cluster = _make_cluster()

        mock_mc_cls = MagicMock()
        mock_mc = MagicMock()
        mock_mc.collect = AsyncMock(side_effect=RuntimeError("metrics boom"))
        mock_mc_cls.return_value = mock_mc

        with (
            patch(_PATCH_CONFIG, return_value=app_config),
            patch(_PATCH_RUN_ALL, new=AsyncMock(return_value=cluster)),
            patch(_PATCH_LOGGING),
            patch(_PATCH_LOG_HEALTH),
            patch(_PATCH_METRICS_CLS, new=mock_mc_cls),
            patch(_PATCH_ORCH_CLS),
            patch(_PATCH_REGISTRY_CLS),
            patch(_PATCH_DETECT_NODE),
        ):
            result = runner.invoke(cli, ["health"])
        # Status still succeeds (cluster is healthy)
        assert result.exit_code == 0
        assert "unavailable" in result.output.lower()


# ---------------------------------------------------------------------------
# logs command
# ---------------------------------------------------------------------------

class TestLogsCommand:
    def test_missing_log_exits_0_with_message(self, app_config, tmp_path):
        runner = CliRunner()
        # app_config.log_dir points at tmp_path which has no maestro.log
        with patch(_PATCH_CONFIG, return_value=app_config):
            result = runner.invoke(cli, ["logs"])
        assert result.exit_code == 0
        assert "not found" in result.output.lower() or "run" in result.output.lower()

    def test_shows_last_n_lines(self, app_config, tmp_path):
        runner = CliRunner()
        log_file = app_config.log_dir / "maestro.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Write 10 numbered log entries (prefix avoids collision with path text)
        log_file.write_text("\n".join(f"ENTRY_{i:03d}" for i in range(1, 11)))

        with patch(_PATCH_CONFIG, return_value=app_config):
            result = runner.invoke(cli, ["logs", "--lines", "3"])
        assert result.exit_code == 0
        # Last 3 lines should appear
        assert "ENTRY_008" in result.output
        assert "ENTRY_009" in result.output
        assert "ENTRY_010" in result.output
        # Earlier lines should not appear
        assert "ENTRY_001" not in result.output

    def test_config_error_exits_2(self):
        runner = CliRunner()
        with patch(_PATCH_CONFIG, side_effect=RuntimeError("cfg fail")):
            result = runner.invoke(cli, ["logs"])
        assert result.exit_code == 2
