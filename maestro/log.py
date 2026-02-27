#!/usr/bin/env python3
"""
Maestro — Operational Event Logger

Dedicated structured logger for Maestro's own operational events.  Written
independently of the structlog-based :mod:`maestro.logger` module so that
event records are always available even when the main logging pipeline has
not been configured.

Files:
    Active   : {log_dir}/maestro.log        (written continuously)
    Archives : {log_dir}/maestro_YYYY-MM-DD.log  (daily rollover)
    Retention: 30 days

Sensitive data rules (CLAUDE.md §10):
    DO log    — service names, ports, latencies, error types, status booleans
    NEVER log — passwords, auth tokens, memory content, user query text

Public API:
    MaestroLogger                 — configured logging instance
    MaestroLogger.log_status_check      — one probe cycle result
    MaestroLogger.log_service_action    — start / stop / probe action
    MaestroLogger.log_metrics_snapshot  — arbitrary metrics dict
    MaestroLogger.log_shutdown          — graceful-shutdown record

Usage::

    from maestro.log import MaestroLogger

    mlog = MaestroLogger()
    mlog.log_service_action("neo4j", "probe", True, "latency=120ms")

Author: UBIK Project
Version: 0.1.0
"""

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_BACKUP_COUNT: int = 30  # daily archive retention

# Standard LogRecord attributes — excluded when serialising extras to JSON.
_STDLIB_RECORD_FIELDS: frozenset[str] = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text",
    "filename", "funcName", "levelname", "levelno", "lineno",
    "message", "module", "msecs", "msg", "name", "pathname",
    "process", "processName", "relativeCreated", "stack_info",
    "taskName", "thread", "threadName",
})

_LOGGER_NAME: str = "maestro.events"


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit each :class:`logging.LogRecord` as a single JSON line.

    Output schema (minimum keys)::

        {
          "timestamp": "2026-02-25T10:30:00Z",
          "level":     "INFO",
          "event":     "service_action",
          ... (extra fields injected via logging's *extra* kwarg)
        }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialise *record* to a JSON string.

        Args:
            record: Log record to serialise.

        Returns:
            Single JSON line with no trailing newline.
        """
        record.message = record.getMessage()
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "event": record.message,
        }
        # Merge extra fields injected by MaestroLogger._emit
        for key, value in record.__dict__.items():
            if key in _STDLIB_RECORD_FIELDS or key.startswith("_"):
                continue
            if key == "event":
                # Already captured above; skip to avoid duplicate
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# TimedRotatingFileHandler namer
# ---------------------------------------------------------------------------

def _make_namer(log_dir: Path):
    """Return a ``namer`` callback for :class:`~logging.handlers.TimedRotatingFileHandler`.

    Transforms the default ``maestro.log.YYYY-MM-DD`` rotation name into
    the human-friendly ``maestro_YYYY-MM-DD.log`` archive convention.

    Args:
        log_dir: Directory that contains the log files.

    Returns:
        Callable ``(default_name: str) -> str``.
    """
    def namer(default_name: str) -> str:
        # default_name: /path/to/maestro.log.2026-02-26
        # target:       /path/to/maestro_2026-02-26.log
        parts = default_name.rsplit(".", 1)
        if (
            len(parts) == 2
            and len(parts[1]) == 10
            and parts[1].count("-") == 2
        ):
            date_part = parts[1]  # "2026-02-26"
            return str(log_dir / f"maestro_{date_part}.log")
        return default_name

    return namer


# ---------------------------------------------------------------------------
# MaestroLogger
# ---------------------------------------------------------------------------

class MaestroLogger:
    """Structured operational logger for UBIK Maestro.

    Writes JSON-line events to a daily-rotating file and human-readable
    summaries to the console via :class:`rich.logging.RichHandler`.

    A named Python logger (``"maestro.events"``) with
    ``propagate = False`` is used so these records never reach the
    structlog root-logger chain configured by :mod:`maestro.logger`.

    Args:
        log_dir: Directory for log files.  When ``None``, resolved from
            :func:`~maestro.config.get_config` as
            ``{UBIK_ROOT}/logs/maestro/``.
        console_level: Minimum level for console (Rich) output.
            Defaults to ``"WARNING"`` to keep watch-mode output clean.

    Note:
        This class never raises.  Every logging failure is swallowed
        silently to respect the "fail silently" contract.

    Example::

        mlog = MaestroLogger()
        mlog.log_service_action("neo4j", "start", True, "port=7687")
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        *,
        console_level: str = "WARNING",
    ) -> None:
        if log_dir is None:
            try:
                from maestro.config import get_config
                log_dir = get_config().log_dir
            except Exception:
                log_dir = Path("logs") / "maestro"

        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # Non-fatal; writes will simply fail below

        self._log_dir: Path = log_dir

        # Detect local node once — included as "node" in every event.
        try:
            from maestro.platform_detect import detect_node
            self._node: str = detect_node().node_type.value
        except Exception:
            self._node = "unknown"

        self._logger: logging.Logger = logging.getLogger(_LOGGER_NAME)
        self._logger.setLevel(logging.DEBUG)
        # Prevent double-emission into the root / structlog chain.
        self._logger.propagate = False

        # Idempotency guard: only attach handlers once per logger instance.
        if self._logger.handlers:
            return

        self._attach_file_handler(log_dir)
        self._attach_console_handler(console_level)

    # ── Handler setup (private) ───────────────────────────────────────────

    def _attach_file_handler(self, log_dir: Path) -> None:
        """Attach a daily-rotating JSON handler to ``self._logger``.

        Args:
            log_dir: Directory where ``maestro.log`` (and its archives)
                will be written.
        """
        try:
            handler = logging.handlers.TimedRotatingFileHandler(
                filename=str(log_dir / "maestro.log"),
                when="midnight",
                backupCount=_LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            handler.namer = _make_namer(log_dir)
            handler.setFormatter(_JsonFormatter())
            handler.setLevel(logging.DEBUG)
            self._logger.addHandler(handler)
        except Exception:
            pass  # Fail silently

    def _attach_console_handler(self, console_level: str) -> None:
        """Attach a :class:`~rich.logging.RichHandler` to ``self._logger``.

        Args:
            console_level: Minimum log level for console output.
        """
        try:
            level = getattr(logging, console_level.upper(), logging.WARNING)
            handler = RichHandler(
                level=level,
                show_path=False,
                rich_tracebacks=False,
            )
            handler.setLevel(level)
            self._logger.addHandler(handler)
        except Exception:
            pass  # Fail silently

    # ── Internal emit helper ──────────────────────────────────────────────

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        """Emit a structured log record, swallowing all exceptions.

        Args:
            level: :mod:`logging` level constant (e.g. ``logging.INFO``).
            event: Short event name (e.g. ``"service_action"``).
            **fields: Additional key-value pairs serialised into the JSON
                record.  Must not include passwords, tokens, or query text.
        """
        try:
            self._logger.log(level, event, extra=fields)
        except Exception:
            pass  # Never raise on logging failures

    # ── Public helpers ────────────────────────────────────────────────────

    def log_status_check(self, statuses: dict[str, Any]) -> None:
        """Log one complete service health-check cycle.

        Builds a compact per-service summary from each value's
        ``healthy``, ``latency_ms``, and ``error`` attributes
        (duck-typed; compatible with :class:`~maestro.services.base.ProbeResult`).

        Output schema::

            {
              "event":    "status_check",
              "node":     "hippocampal",
              "services": {
                "neo4j":  {"healthy": true,  "latency_ms": 120},
                "vllm":   {"healthy": false, "error": "Connection refused"}
              }
            }

        Args:
            statuses: Mapping of service name to status object.  Values
                must expose ``healthy`` (bool), ``latency_ms`` (float|None),
                and ``error`` (str|None).

        Note:
            Never pass passwords, tokens, memory content, or query text
            inside *statuses* values.
        """
        services_summary: dict[str, Any] = {}
        for name, result in statuses.items():
            try:
                entry: dict[str, Any] = {"healthy": result.healthy}
                if getattr(result, "latency_ms", None) is not None:
                    entry["latency_ms"] = round(result.latency_ms, 2)
                if getattr(result, "error", None):
                    entry["error"] = result.error
                services_summary[name] = entry
            except Exception:
                services_summary[name] = {
                    "healthy": False,
                    "error": "summary_error",
                }

        self._emit(
            logging.INFO,
            "status_check",
            node=self._node,
            services=services_summary,
        )

    def log_service_action(
        self,
        service: str,
        action: str,
        success: bool,
        details: str,
    ) -> None:
        """Log a service lifecycle action (start, stop, probe, etc.).

        Chooses log level based on *success*: ``INFO`` on success,
        ``WARNING`` on failure.

        Output schema::

            {
              "event":   "service_action",
              "node":    "hippocampal",
              "service": "neo4j",
              "action":  "probe",
              "success": true,
              "details": "latency=120ms"
            }

        Args:
            service: Service name (e.g. ``"neo4j"``).
            action: Lifecycle action (e.g. ``"start"``, ``"stop"``,
                ``"probe"``).
            success: ``True`` if the action completed successfully.
            details: Brief human-readable context (e.g. ``"latency=120ms"``).
                Must not contain passwords, tokens, or sensitive data.
        """
        level = logging.INFO if success else logging.WARNING
        self._emit(
            level,
            "service_action",
            node=self._node,
            service=service,
            action=action,
            success=success,
            details=details,
        )

    def log_metrics_snapshot(self, metrics: dict[str, Any]) -> None:
        """Log an arbitrary operational metrics snapshot.

        Output schema::

            {
              "event":   "metrics_snapshot",
              "node":    "hippocampal",
              "metrics": {"cpu_pct": 42.1, "mem_used_gb": 12.5}
            }

        Args:
            metrics: Key-value pairs of metric names and values.

        Note:
            *metrics* must not contain passwords, tokens, memory content,
            or query text.
        """
        self._emit(
            logging.INFO,
            "metrics_snapshot",
            node=self._node,
            metrics=metrics,
        )

    def log_shutdown(self, services_stopped: list[str]) -> None:
        """Log a graceful shutdown event.

        Output schema::

            {
              "event":            "shutdown",
              "node":             "hippocampal",
              "services_stopped": ["neo4j", "chromadb"],
              "count":            2
            }

        Args:
            services_stopped: Names of services that were stopped.
        """
        self._emit(
            logging.INFO,
            "shutdown",
            node=self._node,
            services_stopped=services_stopped,
            count=len(services_stopped),
        )
