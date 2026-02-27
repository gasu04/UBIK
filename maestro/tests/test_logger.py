"""
Tests for maestro.logger

Verifies configure_logging, get_logger, and log_cluster_health.

Coverage:
    configure_logging:
        creates log directory when absent
        attaches RotatingFileHandler and StreamHandler to root logger
        is idempotent — no duplicate handlers on repeated calls
        file handler produces valid JSON log lines
        console handler level defaults to WARNING
        console handler level respects console_level argument

    get_logger:
        returns a non-None bound logger

    log_cluster_health:
        HEALTHY cluster → log.info called
        DEGRADED cluster → log.warning called
        UNHEALTHY cluster → log.error called
        cycle kwarg included when provided
        cycle kwarg omitted when None
        overall_status present in kwargs
        services summary present in kwargs
        error field included for unhealthy service
        error field absent for healthy service
        healthy_count / total_count present
"""

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import structlog

from maestro.config import AppConfig, HippocampalConfig, MaestroConfig, SomaticConfig
from maestro.logger import configure_logging, get_logger, log_cluster_health
from maestro.services.models import ClusterHealth, ServiceResult, ServiceStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Restore root logger handlers and level after each test."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    # Close any handlers we added to avoid ResourceWarning
    for h in root.handlers:
        if h not in saved_handlers:
            h.close()
    root.handlers = saved_handlers
    root.setLevel(saved_level)


@pytest.fixture(autouse=True)
def _reset_structlog():
    """Reset structlog to default configuration after each test."""
    yield
    structlog.reset_defaults()


@pytest.fixture()
def log_cfg(tmp_path) -> AppConfig:
    """AppConfig whose log_dir falls inside pytest's tmp_path."""
    return AppConfig(
        ubik_root=tmp_path,
        maestro=MaestroConfig(MAESTRO_LOG_LEVEL="DEBUG"),
        hippocampal=HippocampalConfig(tailscale_ip="127.0.0.1"),
        somatic=SomaticConfig(tailscale_ip="127.0.0.2"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cluster(**statuses: ServiceStatus) -> ClusterHealth:
    """Build a 6-service ClusterHealth; unspecified services default to HEALTHY."""
    names = ("neo4j", "chromadb", "mcp", "vllm", "tailscale", "docker")
    services = {
        n: ServiceResult(n, statuses.get(n, ServiceStatus.HEALTHY), latency_ms=5.0)
        for n in names
    }
    return ClusterHealth(services=services, checked_at=datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# configure_logging — directory and handler creation
# ---------------------------------------------------------------------------

class TestConfigureLogging:
    def test_creates_log_directory(self, log_cfg, tmp_path):
        expected = tmp_path / "logs" / "maestro"
        assert not expected.exists()
        configure_logging(log_cfg)
        assert expected.is_dir()

    def test_attaches_rotating_file_handler(self, log_cfg):
        configure_logging(log_cfg)
        root = logging.getLogger()
        types = [type(h).__name__ for h in root.handlers]
        assert "RotatingFileHandler" in types

    def test_attaches_stream_handler(self, log_cfg):
        configure_logging(log_cfg)
        root = logging.getLogger()
        has_bare_stream = any(type(h) is logging.StreamHandler for h in root.handlers)
        assert has_bare_stream

    def test_idempotent_no_duplicate_file_handler(self, log_cfg):
        configure_logging(log_cfg)
        count_after_first = sum(
            isinstance(h, logging.handlers.RotatingFileHandler)
            for h in logging.getLogger().handlers
        )
        configure_logging(log_cfg)
        count_after_second = sum(
            isinstance(h, logging.handlers.RotatingFileHandler)
            for h in logging.getLogger().handlers
        )
        assert count_after_first == count_after_second == 1

    def test_idempotent_no_duplicate_stream_handler(self, log_cfg):
        configure_logging(log_cfg)
        configure_logging(log_cfg)
        bare_streams = sum(
            type(h) is logging.StreamHandler
            for h in logging.getLogger().handlers
        )
        assert bare_streams == 1

    def test_console_level_defaults_to_warning(self, log_cfg):
        configure_logging(log_cfg)
        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if type(h) is logging.StreamHandler]
        assert stream_handlers[0].level == logging.WARNING

    def test_console_level_custom(self, log_cfg):
        configure_logging(log_cfg, console_level="ERROR")
        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if type(h) is logging.StreamHandler]
        assert stream_handlers[0].level == logging.ERROR

    def test_root_logger_level_set_to_debug(self, log_cfg):
        configure_logging(log_cfg)
        assert logging.getLogger().level == logging.DEBUG

    def test_json_log_file_created(self, log_cfg, tmp_path):
        configure_logging(log_cfg)
        log = get_logger("test")
        log.info("hello_world", answer=42)
        for h in logging.getLogger().handlers:
            h.flush()
        log_file = tmp_path / "logs" / "maestro" / "maestro.log"
        assert log_file.exists()

    def test_json_log_file_valid_json(self, log_cfg, tmp_path):
        configure_logging(log_cfg)
        log = get_logger("test")
        log.info("probe_event", key="value")
        for h in logging.getLogger().handlers:
            h.flush()
        log_file = tmp_path / "logs" / "maestro" / "maestro.log"
        first_line = log_file.read_text(encoding="utf-8").strip().splitlines()[0]
        data = json.loads(first_line)
        assert data["event"] == "probe_event"
        assert data["key"] == "value"

    def test_json_log_contains_level_and_timestamp(self, log_cfg, tmp_path):
        configure_logging(log_cfg)
        get_logger("t").warning("warn_event")
        for h in logging.getLogger().handlers:
            h.flush()
        log_file = tmp_path / "logs" / "maestro" / "maestro.log"
        data = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[0])
        assert "level" in data
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_returns_nonnull(self, log_cfg):
        configure_logging(log_cfg)
        assert get_logger("test.module") is not None

    def test_default_name(self, log_cfg):
        configure_logging(log_cfg)
        assert get_logger() is not None


# ---------------------------------------------------------------------------
# log_cluster_health — level selection
# ---------------------------------------------------------------------------

class TestLogClusterHealthLevels:
    def test_healthy_cluster_calls_info(self):
        cluster = _make_cluster()
        log = MagicMock()
        log_cluster_health(log, cluster)
        log.info.assert_called_once()
        log.warning.assert_not_called()
        log.error.assert_not_called()

    def test_degraded_cluster_calls_warning(self):
        cluster = _make_cluster(neo4j=ServiceStatus.DEGRADED)
        log = MagicMock()
        log_cluster_health(log, cluster)
        log.warning.assert_called_once()
        log.info.assert_not_called()
        log.error.assert_not_called()

    def test_unhealthy_cluster_calls_error(self):
        cluster = _make_cluster(vllm=ServiceStatus.UNHEALTHY)
        log = MagicMock()
        log_cluster_health(log, cluster)
        log.error.assert_called_once()
        log.info.assert_not_called()
        log.warning.assert_not_called()

    def test_unhealthy_takes_precedence_over_degraded(self):
        cluster = _make_cluster(
            neo4j=ServiceStatus.DEGRADED,
            vllm=ServiceStatus.UNHEALTHY,
        )
        log = MagicMock()
        log_cluster_health(log, cluster)
        log.error.assert_called_once()


# ---------------------------------------------------------------------------
# log_cluster_health — event payload
# ---------------------------------------------------------------------------

class TestLogClusterHealthPayload:
    def _call_kwargs(self, cluster: ClusterHealth, **kw) -> dict:
        log = MagicMock()
        log_cluster_health(log, cluster, **kw)
        # Determine which method was called based on overall status
        overall = cluster.overall_status
        if overall == ServiceStatus.HEALTHY:
            call = log.info.call_args
        elif overall == ServiceStatus.DEGRADED:
            call = log.warning.call_args
        else:
            call = log.error.call_args
        return call[1]  # keyword arguments

    def test_event_name_in_positional_arg(self):
        cluster = _make_cluster()
        log = MagicMock()
        log_cluster_health(log, cluster)
        assert log.info.call_args[0][0] == "cluster_health_check"

    def test_overall_status_in_kwargs(self):
        cluster = _make_cluster()
        kw = self._call_kwargs(cluster)
        assert kw["overall_status"] == "healthy"

    def test_overall_status_degraded(self):
        cluster = _make_cluster(chromadb=ServiceStatus.DEGRADED)
        kw = self._call_kwargs(cluster)
        assert kw["overall_status"] == "degraded"

    def test_services_dict_present(self):
        cluster = _make_cluster()
        kw = self._call_kwargs(cluster)
        assert "services" in kw
        assert isinstance(kw["services"], dict)

    def test_services_contains_all_service_names(self):
        cluster = _make_cluster()
        kw = self._call_kwargs(cluster)
        assert set(kw["services"].keys()) == {
            "neo4j", "chromadb", "mcp", "vllm", "tailscale", "docker"
        }

    def test_service_status_value_in_summary(self):
        cluster = _make_cluster(neo4j=ServiceStatus.UNHEALTHY)
        kw = self._call_kwargs(cluster)
        assert kw["services"]["neo4j"]["status"] == "unhealthy"

    def test_latency_ms_rounded(self):
        cluster = ClusterHealth(services={
            "neo4j": ServiceResult("neo4j", ServiceStatus.HEALTHY, latency_ms=12.3456),
        })
        kw = self._call_kwargs(cluster)
        assert kw["services"]["neo4j"]["latency_ms"] == 12.35

    def test_error_included_for_unhealthy(self):
        cluster = ClusterHealth(services={
            "neo4j": ServiceResult(
                "neo4j", ServiceStatus.UNHEALTHY, error="connection refused"
            ),
        })
        kw = self._call_kwargs(cluster)
        assert kw["services"]["neo4j"]["error"] == "connection refused"

    def test_error_absent_for_healthy(self):
        cluster = ClusterHealth(services={
            "neo4j": ServiceResult("neo4j", ServiceStatus.HEALTHY, latency_ms=5.0),
        })
        kw = self._call_kwargs(cluster)
        assert "error" not in kw["services"]["neo4j"]

    def test_healthy_count_correct(self):
        cluster = _make_cluster(neo4j=ServiceStatus.DEGRADED)
        kw = self._call_kwargs(cluster)
        assert kw["total_count"] == 6
        assert kw["healthy_count"] == 5

    def test_checked_at_iso_string(self):
        cluster = _make_cluster()
        kw = self._call_kwargs(cluster)
        assert "T" in kw["checked_at"]  # basic ISO-8601 check

    def test_cycle_included_when_provided(self):
        cluster = _make_cluster()
        kw = self._call_kwargs(cluster, cycle=7)
        assert kw["cycle"] == 7

    def test_cycle_absent_when_none(self):
        cluster = _make_cluster()
        kw = self._call_kwargs(cluster)
        assert "cycle" not in kw

    def test_none_latency_passes_through(self):
        cluster = ClusterHealth(services={
            "neo4j": ServiceResult("neo4j", ServiceStatus.UNHEALTHY, latency_ms=None),
        })
        kw = self._call_kwargs(cluster)
        assert kw["services"]["neo4j"]["latency_ms"] is None
