#!/usr/bin/env python3
"""
Maestro — Periodic Monitor (Background Daemon)

Runs continuous health checks on a configurable interval, logs structured
results to the operational log, and optionally auto-restarts failed local
services via the :class:`~maestro.orchestrator.Orchestrator`.

Lifecycle::

    start-up
        1. _init_components() — build Orchestrator, MaestroLogger, NodeIdentity
        2. _check_stale_pid() — refuse to start if another daemon is live
        3. _write_pid()       — claim the PID file
        4. _install_signal_handlers() — SIGTERM/SIGINT → stop, SIGHUP → reload

    main loop
        while not stopped:
            run_once()   — probe → log → optionally restart
            _sleep(interval)  — interruptible; wakes immediately on stop()

    clean-up
        _remove_pid()
        log_shutdown([])

Signal handling (POSIX only):
    SIGTERM / SIGINT → graceful shutdown (current cycle finishes first)
    SIGHUP           → reload configuration without restart

PID file:
    Written to ``{UBIK_ROOT}/logs/maestro/maestro.pid``.
    Stale PID files (process dead) are silently removed on startup.
    A live PID file raises ``RuntimeError`` to prevent duplicate daemons.

Public API:
    MaestroDaemon — main daemon class

Usage::

    import asyncio
    from maestro.daemon import MaestroDaemon

    daemon = MaestroDaemon(check_interval_s=300, auto_restart=True)
    asyncio.run(daemon.run())

Author: UBIK Project
Version: 0.1.0
"""

import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Optional

from maestro.config import AppConfig, get_config
from maestro.log import MaestroLogger
from maestro.orchestrator import Orchestrator
from maestro.platform_detect import NodeIdentity, NodeType, detect_node
from maestro.services import ServiceRegistry
from maestro.services.base import ProbeResult

log = logging.getLogger(__name__)

# SIGHUP is not defined on Windows.
_SIGHUP: Optional[int] = getattr(signal, "SIGHUP", None)


class MaestroDaemon:
    """Background daemon for periodic UBIK health monitoring.

    Runs concurrent service probes on a configurable interval, logs
    structured JSON events to the operational log, and can optionally
    trigger service restarts when local services are found unhealthy.

    Args:
        check_interval_s: Seconds between consecutive check cycles.
        auto_restart: When ``True``, call
            :meth:`~Orchestrator.ensure_all_running` after any local
            service is found unhealthy.
        cfg: Application configuration.  Resolved via
            :func:`~maestro.config.get_config` on first use when ``None``.

    Example::

        import asyncio
        from maestro.daemon import MaestroDaemon

        daemon = MaestroDaemon(check_interval_s=300, auto_restart=True)
        asyncio.run(daemon.run())
    """

    def __init__(
        self,
        check_interval_s: int,
        auto_restart: bool = False,
        cfg: Optional[AppConfig] = None,
    ) -> None:
        self._interval: float = float(check_interval_s)
        self._auto_restart = auto_restart
        self._cfg = cfg

        self._shutdown: bool = False
        self._reload_requested: bool = False
        self._cycle: int = 0

        # Lazily initialized on first _init_components() call.
        self._orchestrator: Optional[Orchestrator] = None
        self._mlog: Optional[MaestroLogger] = None
        self._identity: Optional[NodeIdentity] = None

        # Asyncio event used to interrupt the between-cycle sleep.
        # Created in run() so it lives in the correct event loop.
        self._stop_event: Optional[asyncio.Event] = None

    # ── Component initialization ─────────────────────────────────────────────

    def _init_components(self) -> None:
        """Instantiate orchestrator, logger, and node identity from config.

        Idempotent — does nothing if already initialized.  Reads
        :func:`~maestro.config.get_config` when no ``cfg`` was provided to
        the constructor.
        """
        if self._orchestrator is not None:
            return
        cfg = self._cfg or get_config()
        self._cfg = cfg
        self._identity = detect_node()
        self._mlog = MaestroLogger(log_dir=cfg.log_dir)
        self._orchestrator = Orchestrator(ServiceRegistry(cfg), self._identity)

    # ── PID file management ──────────────────────────────────────────────────

    def _pid_path(self) -> Path:
        """Return the PID file path inside the configured log directory."""
        cfg = self._cfg or get_config()
        return cfg.log_dir / "maestro.pid"

    def _write_pid(self) -> None:
        """Write the current process PID to the PID file.

        Creates the parent directory if it does not exist.
        Silently ignores I/O errors — a missing PID file is non-fatal.
        """
        try:
            pid_path = self._pid_path()
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text(str(os.getpid()), encoding="utf-8")
            log.debug("daemon: wrote PID %d to %s", os.getpid(), pid_path)
        except Exception as exc:
            log.debug("daemon: failed to write PID file: %s", exc)

    def _remove_pid(self) -> None:
        """Remove the PID file.  Silently ignores errors."""
        try:
            self._pid_path().unlink(missing_ok=True)
            log.debug("daemon: removed PID file")
        except Exception as exc:
            log.debug("daemon: failed to remove PID file: %s", exc)

    def _check_stale_pid(self) -> None:
        """Check for an existing PID file and guard against duplicate daemons.

        Cleans up stale PID files from crashed daemons (process no longer
        exists).  If the recorded PID belongs to a live process, raises
        ``RuntimeError`` to prevent two daemons from running simultaneously.

        Raises:
            RuntimeError: If a live daemon process is detected.
        """
        pid_path = self._pid_path()
        if not pid_path.exists():
            return

        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except Exception:
            # Unreadable / malformed PID file — remove and start fresh.
            pid_path.unlink(missing_ok=True)
            return

        try:
            os.kill(pid, 0)  # signal 0 tests process existence
            # If no exception, the process is alive.
            raise RuntimeError(
                f"Maestro daemon is already running (PID {pid}). "
                f"Stop it first or remove {pid_path}."
            )
        except ProcessLookupError:
            # Process is dead — stale PID file.
            log.info(
                "daemon: removing stale PID file (PID %d no longer exists)", pid
            )
            pid_path.unlink(missing_ok=True)
        except PermissionError:
            # Process exists but we cannot signal it (different user).
            raise RuntimeError(
                f"Maestro daemon appears to be running (PID {pid}). "
                f"Stop it first or remove {pid_path}."
            )

    # ── Signal handling ──────────────────────────────────────────────────────

    def _install_signal_handlers(self) -> None:
        """Register signal handlers on the current asyncio event loop.

        SIGTERM and SIGINT call :meth:`stop` (graceful shutdown).
        SIGHUP calls :meth:`_request_reload` (config reload without restart).

        No-op on platforms where ``loop.add_signal_handler`` is unavailable
        (e.g. Windows) or when there is no running event loop.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGTERM, self.stop)
            loop.add_signal_handler(signal.SIGINT, self.stop)
            if _SIGHUP is not None:
                loop.add_signal_handler(_SIGHUP, self._request_reload)
        except (NotImplementedError, RuntimeError):
            # Windows or called outside a running event loop.
            pass

    def _request_reload(self) -> None:
        """Set the reload flag (invoked by the SIGHUP signal handler)."""
        self._reload_requested = True
        log.info("daemon: SIGHUP received — will reload config after current cycle")

    def _do_reload(self) -> None:
        """Reload configuration and recreate the service orchestrator.

        Clears the :func:`~maestro.config.get_config` LRU cache, re-reads
        settings from the environment / ``.env`` file, and rebuilds the
        :class:`~maestro.orchestrator.Orchestrator`.  Errors are logged but
        do not crash the daemon.
        """
        self._reload_requested = False
        try:
            get_config.cache_clear()
            cfg = get_config()
            self._cfg = cfg
            self._orchestrator = Orchestrator(
                ServiceRegistry(cfg),
                self._identity or detect_node(),
            )
            if self._mlog:
                self._mlog.log_service_action(
                    "daemon", "reload", True, "config reloaded"
                )
            log.info("daemon: configuration reloaded successfully")
        except Exception as exc:
            log.error("daemon: reload failed: %s", exc)
            if self._mlog:
                self._mlog.log_service_action(
                    "daemon", "reload", False, str(exc)
                )

    # ── Interruptible sleep ──────────────────────────────────────────────────

    async def _sleep(self, seconds: float) -> None:
        """Sleep for *seconds*, returning early when :meth:`stop` is called.

        Args:
            seconds: Maximum sleep duration in seconds.
        """
        if self._stop_event is None:
            await asyncio.sleep(seconds)
            return
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    # ── Core check logic ─────────────────────────────────────────────────────

    async def run_once(self) -> dict[str, ProbeResult]:
        """Execute a single check cycle.

        Probes all registered services concurrently, logs the results to
        the operational log, and — when :attr:`auto_restart` is ``True``
        and one or more local services are unhealthy — calls
        :meth:`~Orchestrator.ensure_all_running` to attempt recovery.

        Initializes internal components on the first call if not already done
        by :meth:`run`.

        Returns:
            Mapping of service name → :class:`~ProbeResult` for this cycle.
        """
        self._init_components()
        assert self._orchestrator is not None
        assert self._mlog is not None
        assert self._identity is not None

        self._cycle += 1
        self._mlog.log_service_action(
            "daemon", "cycle_start", True, f"cycle={self._cycle}"
        )
        log.info("daemon: starting check cycle %d", self._cycle)

        statuses = await self._orchestrator.full_status_check()
        self._mlog.log_status_check(statuses)

        if self._auto_restart:
            local_node = self._identity.node_type
            unhealthy_local = [
                name
                for name, r in statuses.items()
                if not r.healthy and r.node == local_node
            ]
            if unhealthy_local:
                log.info(
                    "daemon: unhealthy local services %s — attempting restart",
                    unhealthy_local,
                )
                failed = await self._orchestrator.ensure_all_running()
                self._mlog.log_service_action(
                    "daemon",
                    "auto_restart",
                    len(failed) == 0,
                    f"failed={failed}" if failed else "all_restarted",
                )

        healthy_count = sum(1 for r in statuses.values() if r.healthy)
        total_count = len(statuses)
        self._mlog.log_service_action(
            "daemon",
            "cycle_end",
            True,
            f"cycle={self._cycle} healthy={healthy_count}/{total_count}",
        )
        log.info(
            "daemon: cycle %d complete — %d/%d healthy",
            self._cycle,
            healthy_count,
            total_count,
        )

        return statuses

    def stop(self) -> None:
        """Request graceful shutdown.

        Sets the shutdown flag and wakes any in-progress sleep so the daemon
        exits cleanly after the current check cycle completes.
        """
        self._shutdown = True
        if self._stop_event is not None:
            self._stop_event.set()
        log.info("daemon: shutdown requested")

    async def run(self) -> None:
        """Main daemon loop — runs until :meth:`stop` is called.

        Lifecycle:

        1. Initialize components (Orchestrator, MaestroLogger, NodeIdentity).
        2. Check for stale or live PID file.
        3. Write PID file.
        4. Install signal handlers (SIGTERM/SIGINT → stop, SIGHUP → reload).
        5. Loop: :meth:`run_once` → :meth:`_sleep` until stopped.
        6. Remove PID file, emit shutdown log entry.

        Raises:
            RuntimeError: If a live daemon instance is already running.
        """
        self._init_components()
        assert self._mlog is not None

        self._check_stale_pid()
        self._write_pid()
        self._stop_event = asyncio.Event()
        self._install_signal_handlers()

        self._mlog.log_service_action(
            "daemon",
            "start",
            True,
            f"interval={self._interval:.0f}s auto_restart={self._auto_restart}",
        )
        log.info(
            "daemon: started — interval=%.0fs auto_restart=%s",
            self._interval,
            self._auto_restart,
        )

        try:
            while not self._shutdown:
                if self._reload_requested:
                    self._do_reload()

                await self.run_once()

                if not self._shutdown:
                    await self._sleep(self._interval)
        finally:
            self._remove_pid()
            if self._mlog:
                self._mlog.log_shutdown([])
            log.info("daemon: stopped")
