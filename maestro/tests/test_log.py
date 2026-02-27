"""
Tests for maestro.log — MaestroLogger operational event logger.

Coverage:
    _JsonFormatter:
        produces valid JSON
        includes required top-level keys (timestamp, level, event)
        merges extra fields into payload
        excludes stdlib LogRecord internal fields
        handles exc_info gracefully

    MaestroLogger.__init__:
        creates log directory when missing
        accepts custom log_dir without calling get_config
        attaches TimedRotatingFileHandler (file handler)
        attaches RichHandler (console handler)
        is idempotent — no duplicate handlers on second instantiation
        logger propagate is False
        logger name is "maestro.events"
        silently survives unwritable log directory

    log_status_check:
        emits "status_check" event
        includes node field
        healthy service → healthy: true
        unhealthy service → healthy: false + error field
        latency_ms included when present
        latency_ms absent when None
        gracefully handles object without expected attributes

    log_service_action:
        emits "service_action" event
        includes service, action, success, details fields
        success=True → INFO level
        success=False → WARNING level
        includes node field

    log_metrics_snapshot:
        emits "metrics_snapshot" event
        includes metrics dict
        includes node field

    log_shutdown:
        emits "shutdown" event
        includes services_stopped list
        includes count field

    Silent failure:
        log_service_action does not raise on any input
        log_status_check does not raise on malformed statuses
"""

import io
import json
import logging
import logging.handlers
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.logging import RichHandler

from maestro.log import (
    MaestroLogger,
    _JsonFormatter,
    _LOGGER_NAME,
    _make_namer,
)
from maestro.services.base import ProbeResult
from maestro.platform_detect import NodeType


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_event_logger():
    """Clear maestro.events handlers before and after every test."""
    lg = logging.getLogger(_LOGGER_NAME)
    for h in list(lg.handlers):
        try:
            h.close()
        except Exception:
            pass
        lg.removeHandler(h)
    yield
    for h in list(lg.handlers):
        try:
            h.close()
        except Exception:
            pass
        lg.removeHandler(h)


def _make_logger_with_capture(
    tmp_path: Path,
) -> tuple[MaestroLogger, io.StringIO]:
    """Create a MaestroLogger and attach a StringIO capture handler."""
    mlog = MaestroLogger(log_dir=tmp_path)
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(_JsonFormatter())
    h.setLevel(logging.DEBUG)
    mlog._logger.addHandler(h)
    return mlog, buf


def _last_record(buf: io.StringIO) -> dict:
    """Parse the last JSON line from a capture buffer."""
    buf.seek(0)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert lines, "No log records captured"
    return json.loads(lines[-1])


def _make_probe(
    name: str = "neo4j",
    healthy: bool = True,
    latency_ms: float = 50.0,
    error: str | None = None,
) -> ProbeResult:
    return ProbeResult(
        name=name,
        node=NodeType.HIPPOCAMPAL,
        healthy=healthy,
        latency_ms=latency_ms,
        error=error,
    )


# ---------------------------------------------------------------------------
# TestJsonFormatter
# ---------------------------------------------------------------------------

class TestJsonFormatter:

    def test_produces_valid_json(self):
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test_event", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_contains_timestamp(self):
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="evt", args=(), exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert "timestamp" in parsed
        assert "T" in parsed["timestamp"]  # ISO-8601
        assert parsed["timestamp"].endswith("Z")

    def test_contains_level(self):
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0,
            msg="evt", args=(), exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["level"] == "WARNING"

    def test_contains_event_from_message(self):
        formatter = _JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="service_action", args=(), exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["event"] == "service_action"

    def test_extra_fields_merged_into_payload(self):
        formatter = _JsonFormatter()
        logger = logging.getLogger("test.formatter")
        record = logger.makeRecord(
            "test.formatter", logging.INFO, "", 0,
            "evt", (), None,
            extra={"service": "neo4j", "latency_ms": 120},
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["service"] == "neo4j"
        assert parsed["latency_ms"] == 120

    def test_stdlib_internal_fields_excluded(self):
        formatter = _JsonFormatter()
        logger = logging.getLogger("test.formatter2")
        record = logger.makeRecord(
            "test.formatter2", logging.INFO, "", 0,
            "evt", (), None,
        )
        parsed = json.loads(formatter.format(record))
        # These stdlib internals must not appear
        for field in ("lineno", "funcName", "pathname", "filename",
                      "module", "thread", "process"):
            assert field not in parsed, f"Stdlib field '{field}' leaked into JSON"

    def test_handles_exc_info(self):
        formatter = _JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="error_event", args=(), exc_info=exc_info,
        )
        parsed = json.loads(formatter.format(record))
        assert "exc_info" in parsed
        assert "ValueError" in parsed["exc_info"]


# ---------------------------------------------------------------------------
# TestMaestroLoggerInit
# ---------------------------------------------------------------------------

class TestMaestroLoggerInit:

    def test_creates_log_directory_when_missing(self, tmp_path):
        log_dir = tmp_path / "deep" / "nested" / "logs"
        assert not log_dir.exists()
        MaestroLogger(log_dir=log_dir)
        assert log_dir.exists()

    def test_accepts_custom_log_dir_without_calling_get_config(self, tmp_path):
        with patch("maestro.log.logging.getLogger") as _mock:
            # Patch to verify get_config is NOT called
            pass
        # Simply: passing log_dir skips get_config entirely
        with patch("maestro.config.get_config") as mock_cfg:
            MaestroLogger(log_dir=tmp_path)
            mock_cfg.assert_not_called()

    def test_file_handler_attached(self, tmp_path):
        MaestroLogger(log_dir=tmp_path)
        lg = logging.getLogger(_LOGGER_NAME)
        file_handlers = [
            h for h in lg.handlers
            if isinstance(h, logging.handlers.TimedRotatingFileHandler)
        ]
        assert len(file_handlers) == 1

    def test_console_handler_is_rich_handler(self, tmp_path):
        MaestroLogger(log_dir=tmp_path)
        lg = logging.getLogger(_LOGGER_NAME)
        rich_handlers = [h for h in lg.handlers if isinstance(h, RichHandler)]
        assert len(rich_handlers) == 1

    def test_idempotent_no_duplicate_handlers(self, tmp_path):
        MaestroLogger(log_dir=tmp_path)
        MaestroLogger(log_dir=tmp_path)
        lg = logging.getLogger(_LOGGER_NAME)
        file_handlers = [
            h for h in lg.handlers
            if isinstance(h, logging.handlers.TimedRotatingFileHandler)
        ]
        assert len(file_handlers) == 1

    def test_propagate_is_false(self, tmp_path):
        MaestroLogger(log_dir=tmp_path)
        assert logging.getLogger(_LOGGER_NAME).propagate is False

    def test_logger_name_is_maestro_events(self, tmp_path):
        mlog = MaestroLogger(log_dir=tmp_path)
        assert mlog._logger.name == _LOGGER_NAME

    def test_log_file_created_in_log_dir(self, tmp_path):
        mlog, _ = _make_logger_with_capture(tmp_path)
        mlog.log_service_action("neo4j", "probe", True, "ok")
        log_file = tmp_path / "maestro.log"
        assert log_file.exists()

    def test_does_not_raise_on_unwritable_directory(self, tmp_path):
        # Pass a path where the parent doesn't exist and can't be created
        # on a read-only-like path; MaestroLogger must not raise.
        bad_path = Path("/nonexistent/readonly/path/logs")
        try:
            MaestroLogger(log_dir=bad_path)
        except Exception as exc:
            pytest.fail(f"MaestroLogger raised on bad log_dir: {exc}")


# ---------------------------------------------------------------------------
# TestMakeNamer
# ---------------------------------------------------------------------------

class TestMakeNamer:

    def test_converts_dotted_date_to_underscore_format(self, tmp_path):
        namer = _make_namer(tmp_path)
        result = namer(str(tmp_path / "maestro.log.2026-02-26"))
        assert result == str(tmp_path / "maestro_2026-02-26.log")

    def test_passthrough_when_suffix_not_date(self, tmp_path):
        namer = _make_namer(tmp_path)
        original = str(tmp_path / "maestro.log.backup")
        assert namer(original) == original


# ---------------------------------------------------------------------------
# TestLogServiceAction
# ---------------------------------------------------------------------------

class TestLogServiceAction:

    def test_emits_service_action_event(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_service_action("neo4j", "probe", True, "latency=120ms")
        rec = _last_record(buf)
        assert rec["event"] == "service_action"

    def test_includes_service_name(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_service_action("neo4j", "start", True, "ok")
        rec = _last_record(buf)
        assert rec["service"] == "neo4j"

    def test_includes_action(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_service_action("neo4j", "start", True, "ok")
        rec = _last_record(buf)
        assert rec["action"] == "start"

    def test_includes_success_true(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_service_action("neo4j", "probe", True, "ok")
        rec = _last_record(buf)
        assert rec["success"] is True

    def test_includes_success_false(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_service_action("neo4j", "probe", False, "timeout")
        rec = _last_record(buf)
        assert rec["success"] is False

    def test_includes_details(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_service_action("neo4j", "probe", True, "latency=120ms")
        rec = _last_record(buf)
        assert rec["details"] == "latency=120ms"

    def test_includes_node_field(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_service_action("neo4j", "probe", True, "ok")
        rec = _last_record(buf)
        assert "node" in rec

    def test_success_true_emits_at_info(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_service_action("neo4j", "probe", True, "ok")
        rec = _last_record(buf)
        assert rec["level"] == "INFO"

    def test_success_false_emits_at_warning(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_service_action("neo4j", "probe", False, "timeout")
        rec = _last_record(buf)
        assert rec["level"] == "WARNING"

    def test_does_not_raise_on_any_string_input(self, tmp_path):
        mlog = MaestroLogger(log_dir=tmp_path)
        try:
            mlog.log_service_action("svc", "act", True, "x" * 10_000)
        except Exception as exc:
            pytest.fail(f"log_service_action raised: {exc}")


# ---------------------------------------------------------------------------
# TestLogStatusCheck
# ---------------------------------------------------------------------------

class TestLogStatusCheck:

    def test_emits_status_check_event(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_status_check({"neo4j": _make_probe()})
        rec = _last_record(buf)
        assert rec["event"] == "status_check"

    def test_includes_node_field(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_status_check({"neo4j": _make_probe()})
        rec = _last_record(buf)
        assert "node" in rec

    def test_healthy_service_in_summary(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_status_check({"neo4j": _make_probe(healthy=True)})
        rec = _last_record(buf)
        assert rec["services"]["neo4j"]["healthy"] is True

    def test_unhealthy_service_in_summary(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_status_check({
            "neo4j": _make_probe(healthy=False, error="Connection refused"),
        })
        rec = _last_record(buf)
        assert rec["services"]["neo4j"]["healthy"] is False

    def test_error_field_present_when_unhealthy(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_status_check({
            "neo4j": _make_probe(healthy=False, error="Connection refused"),
        })
        rec = _last_record(buf)
        assert rec["services"]["neo4j"]["error"] == "Connection refused"

    def test_error_field_absent_when_healthy(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_status_check({"neo4j": _make_probe(healthy=True)})
        rec = _last_record(buf)
        assert "error" not in rec["services"]["neo4j"]

    def test_latency_ms_included_when_present(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_status_check({"neo4j": _make_probe(latency_ms=120.0)})
        rec = _last_record(buf)
        assert rec["services"]["neo4j"]["latency_ms"] == 120.0

    def test_multiple_services_in_summary(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_status_check({
            "neo4j": _make_probe("neo4j"),
            "chromadb": _make_probe("chromadb"),
        })
        rec = _last_record(buf)
        assert "neo4j" in rec["services"]
        assert "chromadb" in rec["services"]

    def test_does_not_raise_on_malformed_status_values(self, tmp_path):
        mlog = MaestroLogger(log_dir=tmp_path)
        try:
            # Object with no relevant attributes
            mlog.log_status_check({"broken": object()})
        except Exception as exc:
            pytest.fail(f"log_status_check raised on bad input: {exc}")

    def test_does_not_raise_on_empty_dict(self, tmp_path):
        mlog = MaestroLogger(log_dir=tmp_path)
        try:
            mlog.log_status_check({})
        except Exception as exc:
            pytest.fail(f"log_status_check raised on empty dict: {exc}")


# ---------------------------------------------------------------------------
# TestLogMetricsSnapshot
# ---------------------------------------------------------------------------

class TestLogMetricsSnapshot:

    def test_emits_metrics_snapshot_event(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_metrics_snapshot({"cpu_pct": 42.1})
        rec = _last_record(buf)
        assert rec["event"] == "metrics_snapshot"

    def test_includes_metrics_dict(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_metrics_snapshot({"cpu_pct": 42.1, "mem_used_gb": 12.5})
        rec = _last_record(buf)
        assert rec["metrics"]["cpu_pct"] == 42.1
        assert rec["metrics"]["mem_used_gb"] == 12.5

    def test_includes_node_field(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_metrics_snapshot({})
        rec = _last_record(buf)
        assert "node" in rec

    def test_does_not_raise_on_empty_metrics(self, tmp_path):
        mlog = MaestroLogger(log_dir=tmp_path)
        try:
            mlog.log_metrics_snapshot({})
        except Exception as exc:
            pytest.fail(f"log_metrics_snapshot raised: {exc}")


# ---------------------------------------------------------------------------
# TestLogShutdown
# ---------------------------------------------------------------------------

class TestLogShutdown:

    def test_emits_shutdown_event(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_shutdown(["neo4j", "chromadb"])
        rec = _last_record(buf)
        assert rec["event"] == "shutdown"

    def test_includes_services_stopped(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_shutdown(["neo4j", "chromadb"])
        rec = _last_record(buf)
        assert rec["services_stopped"] == ["neo4j", "chromadb"]

    def test_includes_count(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_shutdown(["neo4j", "chromadb"])
        rec = _last_record(buf)
        assert rec["count"] == 2

    def test_includes_node_field(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_shutdown([])
        rec = _last_record(buf)
        assert "node" in rec

    def test_empty_list_gives_zero_count(self, tmp_path):
        mlog, buf = _make_logger_with_capture(tmp_path)
        mlog.log_shutdown([])
        rec = _last_record(buf)
        assert rec["count"] == 0

    def test_does_not_raise_on_empty_list(self, tmp_path):
        mlog = MaestroLogger(log_dir=tmp_path)
        try:
            mlog.log_shutdown([])
        except Exception as exc:
            pytest.fail(f"log_shutdown raised: {exc}")


# ---------------------------------------------------------------------------
# TestLogFileOutput
# ---------------------------------------------------------------------------

class TestLogFileOutput:

    def test_file_contains_valid_json_lines(self, tmp_path):
        mlog = MaestroLogger(log_dir=tmp_path)
        mlog.log_service_action("neo4j", "probe", True, "ok")
        mlog.log_shutdown(["neo4j"])
        log_file = tmp_path / "maestro.log"
        lines = [ln for ln in log_file.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 2
        for line in lines:
            parsed = json.loads(line)
            assert "timestamp" in parsed
            assert "level" in parsed
            assert "event" in parsed

    def test_file_handler_uses_json_formatter(self, tmp_path):
        MaestroLogger(log_dir=tmp_path)
        lg = logging.getLogger(_LOGGER_NAME)
        timed_handlers = [
            h for h in lg.handlers
            if isinstance(h, logging.handlers.TimedRotatingFileHandler)
        ]
        assert len(timed_handlers) == 1
        assert isinstance(timed_handlers[0].formatter, _JsonFormatter)

    def test_timed_handler_rotates_at_midnight(self, tmp_path):
        MaestroLogger(log_dir=tmp_path)
        lg = logging.getLogger(_LOGGER_NAME)
        timed_handlers = [
            h for h in lg.handlers
            if isinstance(h, logging.handlers.TimedRotatingFileHandler)
        ]
        assert timed_handlers[0].when == "MIDNIGHT"

    def test_timed_handler_keeps_30_backups(self, tmp_path):
        MaestroLogger(log_dir=tmp_path)
        lg = logging.getLogger(_LOGGER_NAME)
        timed_handlers = [
            h for h in lg.handlers
            if isinstance(h, logging.handlers.TimedRotatingFileHandler)
        ]
        assert timed_handlers[0].backupCount == 30
