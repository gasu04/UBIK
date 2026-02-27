#!/usr/bin/env python3
"""
Maestro — Structured Logging

Configures structlog with dual output:
    File   — rotating JSON at {log_dir}/maestro.log (10 MB, 5 backups)
    Console — human-readable at WARNING+ level (keeps watch-mode clean)

Public API:
    configure_logging  — initialise structlog + stdlib logging handlers
    get_logger         — return a bound structlog logger for a module
    log_cluster_health — emit a structured health-check event

Usage:
    from maestro.logger import configure_logging, get_logger, log_cluster_health
    from maestro.config import get_config

    cfg = get_config()
    configure_logging(cfg)

    log = get_logger(__name__)
    log.info("maestro_started", version="0.4.0")

    cluster = await run_all_checks(cfg)
    log_cluster_health(log, cluster, cycle=1)

Author: UBIK Project
Version: 0.1.0
"""

import io
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import structlog

from maestro.config import AppConfig
from maestro.services.models import ClusterHealth, ServiceStatus

# ---------------------------------------------------------------------------
# Shared processor chain
# ---------------------------------------------------------------------------

# Run for every log record — both structlog-native and foreign (stdlib) records.
_SHARED_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.ExceptionRenderer(),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging(
    cfg: AppConfig,
    *,
    console_level: str = "WARNING",
) -> None:
    """Initialise structlog and stdlib logging for Maestro.

    Creates ``{log_dir}/maestro.log`` as a rotating JSON file (10 MB,
    5 backups) and attaches a stderr handler at *console_level* that
    renders events in human-readable form.

    Safe to call multiple times — duplicate handlers are not added if the
    root logger already has a ``RotatingFileHandler`` or bare
    ``StreamHandler`` from a previous call.

    Args:
        cfg: Application configuration providing ``log_dir`` and
            ``maestro.log_level``.
        console_level: Minimum level for the stderr console handler.
            Defaults to ``"WARNING"`` so watch-mode output stays clean.
    """
    log_dir: Path = cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    file_level = getattr(logging, cfg.maestro.log_level.upper(), logging.INFO)
    con_level = getattr(logging, console_level.upper(), logging.WARNING)

    # ── structlog configuration ────────────────────────────────────────────
    structlog.configure(
        processors=_SHARED_PROCESSORS + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(file_level),
        cache_logger_on_first_use=True,
    )

    # ── stdlib formatters ──────────────────────────────────────────────────
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=_SHARED_PROCESSORS,
    )

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        foreign_pre_chain=_SHARED_PROCESSORS,
    )

    # ── handlers (idempotency guard) ───────────────────────────────────────
    root = logging.getLogger()

    has_rotating = any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        for h in root.handlers
    )
    has_stream = any(
        type(h) is logging.StreamHandler
        for h in root.handlers
    )

    if not has_rotating:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "maestro.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)

    if not has_stream:
        # Use a UTF-8 stream so non-ASCII characters in log messages (e.g.
        # Unicode box-drawing chars) never raise UnicodeEncodeError on
        # ASCII-locale Linux systems (LANG=C).
        raw = getattr(sys.stderr, "buffer", None)
        if raw is not None:
            stream: io.TextIOBase = io.TextIOWrapper(
                raw, encoding="utf-8", errors="backslashreplace", line_buffering=True
            )
        else:
            stream = sys.stderr
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(con_level)
        root.addHandler(console_handler)

    root.setLevel(logging.DEBUG)


def get_logger(name: str = "maestro") -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A structlog bound logger with INFO/WARNING/ERROR/DEBUG methods.

    Example:
        log = get_logger(__name__)
        log.info("service_started", port=8080)
    """
    return structlog.get_logger(name)


def log_cluster_health(
    log: structlog.stdlib.BoundLogger,
    cluster: ClusterHealth,
    *,
    cycle: int | None = None,
) -> None:
    """Emit a structured event for one health-check cycle.

    Chooses log level based on the cluster's overall status:
        HEALTHY   → INFO
        DEGRADED  → WARNING
        UNHEALTHY → ERROR

    Args:
        log: Bound structlog logger to write to.
        cluster: Aggregated health result from the most recent check cycle.
        cycle: Optional check cycle counter for log correlation.
    """
    services_summary: dict[str, Any] = {
        name: {
            "status": result.status.value,
            "latency_ms": (
                round(result.latency_ms, 2) if result.latency_ms is not None else None
            ),
            **({"error": result.error} if result.error else {}),
        }
        for name, result in cluster.services.items()
    }

    kwargs: dict[str, Any] = dict(
        overall_status=cluster.overall_status.value,
        healthy_count=len(cluster.healthy_services),
        total_count=len(cluster.services),
        services=services_summary,
        checked_at=cluster.checked_at.isoformat(),
    )
    if cycle is not None:
        kwargs["cycle"] = cycle

    overall = cluster.overall_status
    if overall == ServiceStatus.HEALTHY:
        log.info("cluster_health_check", **kwargs)
    elif overall == ServiceStatus.DEGRADED:
        log.warning("cluster_health_check", **kwargs)
    else:
        log.error("cluster_health_check", **kwargs)
