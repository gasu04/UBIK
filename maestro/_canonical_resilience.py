#!/usr/bin/env python3
"""
Maestro — Canonical Resilience Import Shim

Loads the Probe-Latch circuit breaker and retry/backoff helpers from the
canonical ``somatic/mcp_client/resilience.py`` (CLAUDE.md §2.3/§3.4 — import
this, never reimplement it) — by file path, bypassing the normal
``import somatic.mcp_client.resilience`` package import.

Why bypass the normal import: importing anything under the
``somatic.mcp_client`` *package* runs ``somatic/mcp_client/__init__.py``,
which pulls in ``hippocampal_client.py`` — and that module calls
``logging.basicConfig()`` at import time (a real, pre-existing bug: a library
configuring the global root logger as a side effect of being imported, not
something the importer asked for). That silently added stray FileHandler and
StreamHandler instances to maestro's root logger pointed at
``{ubik_root}/logs/mcp_client.log``, and broke maestro's own
``configure_logging()`` tests. It also triggers a separate pre-existing
quirk in ``client.py`` (a ``sys.path`` insert + bare ``from config import
get_settings``) that leaks a top-level ``config`` module into
``sys.modules``. ``resilience.py`` itself has no such side effects (stdlib
imports only) — loading it standalone via :mod:`importlib` sidesteps
``__init__.py`` (and therefore both bugs) entirely, while still using the
exact canonical file and classes.

Usage:
    from maestro._canonical_resilience import (
        CircuitBreaker, CircuitBreakerConfig, CircuitOpenError,
        RetryConfig, calculate_backoff_delay,
    )

Dependencies:
    None beyond the standard library — ``resilience.py`` itself only imports
    asyncio/logging/random/time/dataclasses/enum/functools/typing.

Author: UBIK Project
Version: 0.1.0
"""

import importlib.util
import sys
from pathlib import Path

_RESILIENCE_PATH = Path(__file__).parent.parent / "somatic" / "mcp_client" / "resilience.py"
_MODULE_NAME = "ubik_canonical_resilience"

if _MODULE_NAME in sys.modules:
    _module = sys.modules[_MODULE_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _RESILIENCE_PATH)
    if _spec is None or _spec.loader is None:
        raise ImportError(
            f"cannot load canonical resilience module from {_RESILIENCE_PATH}"
        )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _module
    _spec.loader.exec_module(_module)

CircuitBreaker = _module.CircuitBreaker
CircuitBreakerConfig = _module.CircuitBreakerConfig
CircuitOpenError = _module.CircuitOpenError
CircuitState = _module.CircuitState
RetryConfig = _module.RetryConfig
calculate_backoff_delay = _module.calculate_backoff_delay

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "CircuitState",
    "RetryConfig",
    "calculate_backoff_delay",
]
