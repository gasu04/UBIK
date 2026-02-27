#!/usr/bin/env python3
"""
Maestro — Abstract Base Service

Defines the shared contract that every UBIK service implementation must
satisfy, plus the :class:`ProbeResult` dataclass that all probes return.

Two layers of functionality:
    UbikService   — abstract base; subclasses implement probe/start/stop.
    ProbeResult   — structured result of a single probe call.

``probe_with_timeout`` is the recommended entry point for callers.  It
wraps ``probe`` with ``asyncio.wait_for`` and converts timeout/exception
into an appropriate error result so callers never have to handle
exceptions from probing.

Module-level helpers available to all service implementations:
    _run_proc      — run an external command with timeout, return (rc, stdout, stderr)
    _kill_port     — SIGTERM whatever process is listening on a TCP port

Author: UBIK Project
Version: 0.1.0
"""

import asyncio
import logging
import os
import platform
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from maestro.platform_detect import NodeType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProbeResult
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Structured outcome of a single service probe.

    This is the return type of both :meth:`UbikService.probe` and
    :meth:`UbikService.probe_with_timeout`.

    Attributes:
        name: Service identifier (e.g. ``"neo4j"``).
        node: Which UBIK node hosts this service.
        healthy: ``True`` when the probe confirms the service is reachable
            and operating normally.
        latency_ms: Round-trip probe latency in milliseconds.  Always
            non-``None`` — even error results record elapsed time so
            callers can use ``:.0f`` formatting safely.
        details: Arbitrary diagnostic dict (versions, counts, IPs…).
        error: Human-readable failure description; ``None`` when healthy.
        checked_at: UTC timestamp when the probe completed.
    """

    name: str
    node: NodeType
    healthy: bool
    latency_ms: float
    details: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    checked_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Subprocess helpers (shared by service implementations)
# ---------------------------------------------------------------------------

async def _run_proc(
    *args: str,
    timeout: float = 10.0,
    **popen_kwargs: Any,
) -> tuple[int, str, str]:
    """Run an external command with timeout.

    Args:
        *args: Command and arguments passed to ``asyncio.create_subprocess_exec``.
        timeout: Maximum seconds to wait for the process to finish.
        **popen_kwargs: Extra kwargs forwarded to ``create_subprocess_exec``.

    Returns:
        ``(returncode, stdout, stderr)`` as ``(int, str, str)``.

    Raises:
        asyncio.TimeoutError: Process did not complete within *timeout*.
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **popen_kwargs,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return (proc.returncode or 0,
                stdout_b.decode(errors="replace"),
                stderr_b.decode(errors="replace"))
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise


async def _kill_port(port: int, *, timeout: float = 5.0) -> bool:
    """Send SIGTERM to whatever process is listening on *port*.

    Uses ``lsof -ti`` on macOS and ``fuser`` on Linux.

    Args:
        port: TCP port number.
        timeout: Subprocess timeout in seconds.

    Returns:
        ``True`` if at least one process was signalled; ``False`` otherwise.
    """
    try:
        if platform.system() == "Darwin":
            rc, stdout, _ = await _run_proc(
                "lsof", "-ti", f":{port}", timeout=timeout
            )
        else:
            rc, stdout, _ = await _run_proc(
                "fuser", f"{port}/tcp", timeout=timeout
            )
        pids = [int(p) for p in stdout.split() if p.strip().isdigit()]
        if not pids:
            logger.debug("_kill_port(%d): no processes found", port)
            return False
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                logger.debug("_kill_port(%d): sent SIGTERM to pid %d", port, pid)
            except ProcessLookupError:
                pass
        return True
    except asyncio.TimeoutError:
        logger.warning("_kill_port(%d): lsof/fuser timed out", port)
        return False
    except Exception as exc:
        logger.warning("_kill_port(%d): failed: %s", port, exc)
        return False


# ---------------------------------------------------------------------------
# UbikService abstract base
# ---------------------------------------------------------------------------

class UbikService(ABC):
    """Abstract base class for all UBIK service implementations.

    Subclasses must implement :meth:`probe`, :meth:`start`, and
    :meth:`stop`.  Callers should use :meth:`probe_with_timeout` rather
    than calling ``probe`` directly.

    Properties ``name``, ``node``, ``ports``, and ``depends_on`` are
    abstract so that the service registry can inspect them without
    instantiating subclasses.

    Example:
        class Neo4jService(UbikService):
            @property
            def name(self) -> str:
                return "neo4j"

            async def probe(self, host: str) -> ProbeResult:
                ...
    """

    # ── Identity properties (must be overridden) ───────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical service name used in logs and the registry."""

    @property
    @abstractmethod
    def node(self) -> NodeType:
        """Which UBIK node this service runs on."""

    @property
    @abstractmethod
    def ports(self) -> list[int]:
        """TCP ports this service listens on (used by stop helpers)."""

    @property
    @abstractmethod
    def depends_on(self) -> list[str]:
        """Names of services that must be started before this one."""

    # ── Lifecycle methods (must be overridden) ─────────────────────────────

    @abstractmethod
    async def probe(self, host: str) -> ProbeResult:
        """Probe the service and return a structured result.

        Args:
            host: IP address or hostname to probe.  The caller supplies
                this so the same service class works for local and remote
                checks.

        Returns:
            :class:`ProbeResult` describing the outcome.
        """

    @abstractmethod
    async def start(self, ubik_root: Path) -> bool:
        """Attempt to start the service.

        Args:
            ubik_root: Absolute path to the UBIK project root directory
                on the *local* node.  Used to locate compose files and
                scripts.

        Returns:
            ``True`` if the start command was issued successfully.
            Does NOT wait for the service to become healthy — use
            :meth:`probe_with_timeout` after a short delay to verify.
        """

    @abstractmethod
    async def stop(self) -> bool:
        """Attempt to stop the service.

        Returns:
            ``True`` if a stop signal was sent successfully.
        """

    # ── Concrete helpers ───────────────────────────────────────────────────

    @property
    def max_wait_s(self) -> float:
        """Maximum seconds to wait for the service to become healthy after start.

        Subclasses override this with service-appropriate defaults.
        Can also be set via the constructor in concrete implementations.
        """
        return 30.0

    async def _wait_for_healthy(
        self,
        probe_host: str = "localhost",
        *,
        poll_interval: float = 3.0,
    ) -> bool:
        """Poll :meth:`probe` until healthy or :attr:`max_wait_s` exceeded.

        Logs progress every poll cycle so operators can see what's happening.
        The format "Waiting for <name>... (<elapsed>s / <limit>s)" matches
        the operational logging convention.

        Args:
            probe_host: Host argument passed to :meth:`probe_with_timeout`.
            poll_interval: Seconds between consecutive probe calls.

        Returns:
            ``True`` when the service becomes healthy within
            :attr:`max_wait_s`; ``False`` on timeout.
        """
        start_ts = time.perf_counter()
        limit = self.max_wait_s

        while True:
            elapsed = round(time.perf_counter() - start_ts)
            if elapsed >= limit:
                logger.warning(
                    "%s did not become healthy within %.0fs",
                    self.name, limit,
                )
                return False

            result = await self.probe_with_timeout(
                probe_host, timeout=min(5.0, poll_interval)
            )
            if result.healthy:
                logger.info(
                    "%s became healthy after %.0fs", self.name, elapsed
                )
                return True

            logger.info(
                "Waiting for %s... (%ds / %.0fs)", self.name, elapsed, limit
            )
            remaining = limit - (time.perf_counter() - start_ts)
            if remaining <= 0:
                logger.warning(
                    "%s did not become healthy within %.0fs",
                    self.name, limit,
                )
                return False
            await asyncio.sleep(min(poll_interval, remaining))

    async def probe_with_timeout(
        self,
        host: str,
        timeout: float = 5.0,
    ) -> ProbeResult:
        """Probe with a wall-clock timeout guard.

        Wraps :meth:`probe` with :func:`asyncio.wait_for`.  Any
        ``TimeoutError`` or unhandled exception is converted into an
        unhealthy :class:`ProbeResult` so callers never need try/except.

        Args:
            host: Passed through to :meth:`probe`.
            timeout: Maximum seconds before returning a timeout error.

        Returns:
            :class:`ProbeResult` — healthy on success, error result on
            timeout or exception.  ``latency_ms`` is always set.
        """
        start_ts = time.perf_counter()
        try:
            return await asyncio.wait_for(self.probe(host), timeout=timeout)
        except asyncio.TimeoutError:
            latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)
            logger.debug("%s probe timed out after %.1fs", self.name, timeout)
            return ProbeResult(
                name=self.name,
                node=self.node,
                healthy=False,
                latency_ms=latency_ms,
                details={},
                error=f"Probe timed out after {timeout:.0f}s",
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)
            logger.debug("%s probe raised: %s", self.name, exc)
            return ProbeResult(
                name=self.name,
                node=self.node,
                healthy=False,
                latency_ms=latency_ms,
                details={},
                error=str(exc),
            )
