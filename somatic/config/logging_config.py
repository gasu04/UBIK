"""Logging configuration for UBIK Somatic Node.

Provides structured JSON logging with automatic sanitization of
sensitive fields. Personal legacy data (memory content, queries,
prompts, responses) MUST never appear in logs.

What to log:
    - Memory IDs (e.g., ep_20240115_103000_123456)
    - Similarity scores
    - Retrieval counts
    - Timing metrics
    - Error types (not error details with content)

What NOT to log:
    - Memory content
    - Query text (may contain personal details)
    - Context strings
    - Full prompts
    - Response text

Usage:
    from config.logging_config import configure_logging

    configure_logging()  # Call once at startup

Dependencies:
    - config.settings (LoggingSettings)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Set

from config.settings import get_settings


class SafeJSONFormatter(logging.Formatter):
    """JSON formatter that sanitizes sensitive fields.

    Automatically redacts any log record field whose key matches
    a known sensitive name. For string values, logs the length
    instead. For all other types, replaces the value with
    ``[REDACTED]``.

    Attributes:
        SENSITIVE_KEYS: Field names that must never appear in logs.
    """

    SENSITIVE_KEYS: Set[str] = frozenset({
        "content",
        "text",
        "query",
        "context",
        "prompt",
        "response",
        "message_content",
        "memory_content",
        "raw_response",
    })

    # Standard LogRecord attributes to exclude from extra fields
    _BUILTIN_ATTRS: Set[str] = frozenset({
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message",
        "module", "msecs", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "taskName",
        "thread", "threadName",
    })

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as sanitized JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string with sensitive fields redacted.
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info and record.exc_info[1] is not None:
            log_data["exception_type"] = type(record.exc_info[1]).__name__
            log_data["exception"] = str(record.exc_info[1])

        # Add extra fields, sanitizing sensitive ones
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in self._BUILTIN_ATTRS:
                continue
            if key in log_data:
                continue

            if key.lower() in self.SENSITIVE_KEYS:
                if isinstance(value, str):
                    log_data[f"{key}_length"] = len(value)
                elif isinstance(value, (list, dict)):
                    log_data[f"{key}_count"] = len(value)
                else:
                    log_data[key] = "[REDACTED]"
            else:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class SafeTextFormatter(logging.Formatter):
    """Human-readable text formatter with sensitive field sanitization.

    Uses a standard log format but redacts sensitive extra fields.
    Intended for development/debugging use.
    """

    SENSITIVE_KEYS = SafeJSONFormatter.SENSITIVE_KEYS

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format with sensitive field redaction."""
        # Sanitize the message itself if it's in extra
        base = super().format(record)

        # Append safe extra fields
        extras = []
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in SafeJSONFormatter._BUILTIN_ATTRS:
                continue
            if key in {"message", "asctime"}:
                continue

            if key.lower() in self.SENSITIVE_KEYS:
                if isinstance(value, str):
                    extras.append(f"{key}_length={len(value)}")
                else:
                    extras.append(f"{key}=[REDACTED]")
            # Non-sensitive extras are already in the message via f-strings

        if extras:
            base += f" | {', '.join(extras)}"

        return base


def configure_logging() -> None:
    """Configure application-wide logging.

    Reads settings from ``LoggingSettings`` to determine:
    - Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Output format (json or text)
    - Log directory for file output

    Call this once at application startup before any logging occurs.
    """
    settings = get_settings()
    log_settings = settings.logging

    # Ensure log directory exists
    log_dir = Path(log_settings.dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Select formatter based on settings
    if log_settings.format == "json":
        formatter = SafeJSONFormatter()
    else:
        formatter = SafeTextFormatter()

    # Configure root logger for ubik namespace
    root_logger = logging.getLogger("ubik")
    root_logger.setLevel(getattr(logging, log_settings.level))
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_path = log_dir / "somatic.log"
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.info(
        "Logging configured",
        extra={
            "log_level": log_settings.level,
            "log_format": log_settings.format,
            "log_dir": str(log_dir),
        },
    )
