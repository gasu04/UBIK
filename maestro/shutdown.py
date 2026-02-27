#!/usr/bin/env python3
"""
Maestro — Orderly Shutdown Controller

Stops local UBIK services in reverse dependency order (MCP → ChromaDB →
Neo4j → Docker on Hippocampal; vLLM on Somatic), logging every step via
:class:`~maestro.log.MaestroLogger`.

Shutdown sequence:

    1. Log "MAESTRO SHUTDOWN INITIATED".
    2. Stop the background daemon (SIGTERM to PID file process if present).
    3. Log a pre-shutdown metrics snapshot.
    4. For each local service in reverse startup order:
       a. Log "Stopping <service>...".
       b. Call service.stop().
       c. Poll probe until DOWN (up to ``timeout_per_service`` seconds).
       d. Escalate to SIGKILL on each port if still alive after timeout.
       e. Log result.
    5. Post-shutdown verification: probe all local services.
    6. Log final shutdown record via :meth:`~MaestroLogger.log_shutdown`.

Public API:
    ShutdownController — orderly and emergency shutdown

Usage::

    import asyncio
    from maestro.shutdown import ShutdownController
    from maestro.services import ServiceRegistry
    from maestro.platform_detect import detect_node

    ctrl = ShutdownController(ServiceRegistry(), detect_node())
    asyncio.run(ctrl.orderly_shutdown())

Author: UBIK Project
Version: 0.1.0
"""

import asyncio
import logging
import os
import platform
import signal
import time
from typing import Optional

from maestro.log import MaestroLogger
from maestro.platform_detect import NodeIdentity, NodeType
from maestro.services import ServiceRegistry
from maestro.services.base import UbikService, _run_proc

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STOP_TIMEOUT_S: float = 30.0    # max seconds to wait for graceful stop per svc
_TOTAL_TIMEOUT_S: float = 120.0  # total shutdown budget across all services
_KILL_SETTLE_S: float = 2.0      # seconds to wait after SIGKILL before re-probe
_POLL_INTERVAL_S: float = 2.0    # probe poll interval while waiting for DOWN


# ---------------------------------------------------------------------------
# SIGKILL helper
# ---------------------------------------------------------------------------

async def _sigkill_port(port: int, *, timeout: float = 5.0) -> bool:
    """Send SIGKILL to whatever process is listening on *port*.

    Uses ``lsof -ti`` on macOS and ``fuser`` on Linux.

    Args:
        port: TCP port number.
        timeout: Subprocess timeout in seconds.

    Returns:
        ``True`` if at least one process was signalled; ``False`` otherwise.
    """
    try:
        if platform.system() == "Darwin":
            _rc, stdout, _err = await _run_proc(
                "lsof", "-ti", f":{port}", timeout=timeout
            )
        else:
            _rc, stdout, _err = await _run_proc(
                "fuser", f"{port}/tcp", timeout=timeout
            )
        pids = [int(p) for p in stdout.split() if p.strip().isdigit()]
        if not pids:
            log.debug("_sigkill_port(%d): no processes found", port)
            return False
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
                log.debug("_sigkill_port(%d): sent SIGKILL to pid %d", port, pid)
            except ProcessLookupError:
                pass
        return True
    except asyncio.TimeoutError:
        log.warning("_sigkill_port(%d): lsof/fuser timed out", port)
        return False
    except Exception as exc:
        log.warning("_sigkill_port(%d): failed: %s", port, exc)
        return False


# ---------------------------------------------------------------------------
# ShutdownController
# ---------------------------------------------------------------------------

class ShutdownController:
    """Stops local UBIK services in reverse dependency order.

    Coordinates an orderly shutdown: iterates local services in reverse
    startup order (dependents first, dependencies last), waits for each to
    confirm DOWN, and escalates to SIGKILL when graceful stops time out.

    Args:
        registry: Registry of all service instances.
        identity: Identity of the node running this controller.
            Determines which services are local (stop-eligible).
        mlog: Operational event logger.  Created from ``registry.cfg.log_dir``
            when ``None``.

    Example::

        ctrl = ShutdownController(ServiceRegistry(), detect_node())
        asyncio.run(ctrl.orderly_shutdown())
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        identity: NodeIdentity,
        *,
        mlog: Optional[MaestroLogger] = None,
    ) -> None:
        self._registry = registry
        self._identity = identity
        self._mlog: MaestroLogger = mlog or MaestroLogger(
            log_dir=registry.cfg.log_dir
        )

    # ── Service ordering ─────────────────────────────────────────────────

    def _local_services_in_shutdown_order(self) -> list[UbikService]:
        """Return local services in reverse startup (safe shutdown) order.

        Dependents (e.g. MCP) are stopped before their dependencies (Docker).

        Returns:
            Local services to stop, with dependents first and foundations last.
        """
        local_node = self._identity.node_type
        startup_order = self._registry.get_startup_order()
        local = [s for s in startup_order if s.node == local_node]
        return list(reversed(local))

    # ── Wait for service to go DOWN ──────────────────────────────────────

    async def _wait_for_down(
        self,
        svc: UbikService,
        *,
        timeout: float = _STOP_TIMEOUT_S,
    ) -> bool:
        """Poll probe until the service reports unhealthy (stopped).

        Checks immediately on the first iteration so fast-stopping services
        are confirmed without delay.

        Args:
            svc: Service to probe.
            timeout: Maximum seconds to wait.

        Returns:
            ``True`` when the probe reports ``healthy=False`` within *timeout*;
            ``False`` on timeout.
        """
        start_ts = time.perf_counter()
        while True:
            result = await svc.probe_with_timeout("localhost", timeout=5.0)
            if not result.healthy:
                return True
            elapsed = time.perf_counter() - start_ts
            if elapsed >= timeout:
                log.warning(
                    "%s still UP after %.0fs — graceful stop timeout",
                    svc.name, timeout,
                )
                return False
            remaining = timeout - elapsed
            await asyncio.sleep(min(_POLL_INTERVAL_S, remaining))

    # ── SIGKILL escalation ───────────────────────────────────────────────

    async def _sigkill_service(self, svc: UbikService) -> None:
        """Send SIGKILL to all processes listening on the service's ports.

        Args:
            svc: Service whose port(s) should be killed.
        """
        if not svc.ports:
            log.debug("shutdown: %s has no ports — skipping SIGKILL", svc.name)
            return
        for port in svc.ports:
            await _sigkill_port(port)

    # ── Daemon stop ──────────────────────────────────────────────────────

    def _stop_daemon(self) -> None:
        """Send SIGTERM to the background daemon process, if running.

        Reads the PID file at ``{log_dir}/maestro.pid``.  Silently ignores
        missing files and dead processes.
        """
        pid_path = self._registry.cfg.log_dir / "maestro.pid"
        if not pid_path.exists():
            log.debug("shutdown: no daemon PID file — daemon not running")
            return
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(pid, signal.SIGTERM)
            log.info("shutdown: sent SIGTERM to daemon PID %d", pid)
            self._mlog.log_service_action(
                "daemon", "stop", True, f"SIGTERM pid={pid}"
            )
        except (ProcessLookupError, ValueError) as exc:
            log.debug("shutdown: daemon PID not live: %s", exc)
        except Exception as exc:
            log.warning("shutdown: failed to stop daemon: %s", exc)

    # ── Post-shutdown verification ───────────────────────────────────────

    async def _verify_all_down(self) -> dict[str, bool]:
        """Probe all local services and return a name→stopped map.

        Returns:
            Dict mapping service name to ``True`` when confirmed DOWN
            (probe unhealthy or raised), ``False`` when still responding.
        """
        local_node = self._identity.node_type
        local_svcs = [
            s for s in self._registry.get_all() if s.node == local_node
        ]
        if not local_svcs:
            return {}
        raw = await asyncio.gather(
            *[svc.probe_with_timeout("localhost", timeout=5.0) for svc in local_svcs],
            return_exceptions=True,
        )
        result: dict[str, bool] = {}
        for svc, outcome in zip(local_svcs, raw):
            if isinstance(outcome, Exception):
                result[svc.name] = True   # can't connect → confirmed DOWN
            else:
                result[svc.name] = not outcome.healthy
        return result

    # ── Public API ───────────────────────────────────────────────────────

    async def orderly_shutdown(
        self,
        *,
        timeout_per_service: float = _STOP_TIMEOUT_S,
        dry_run: bool = False,
    ) -> list[str]:
        """Stop all local services in reverse dependency order.

        For each service:

        1. Log ``"Stopping <service>..."``.
        2. Call :meth:`~UbikService.stop` (skipped when *dry_run*).
        3. Poll probe until DOWN (up to *timeout_per_service* seconds).
        4. Escalate to SIGKILL on each port if still alive.
        5. Log result.

        Pre-shutdown: stop daemon (unless *dry_run*), log metrics snapshot.
        Post-shutdown: probe all local services, log verification report.
        Total wall-clock budget is capped at 120 seconds across all services.

        Args:
            timeout_per_service: Max seconds to wait per service before SIGKILL.
            dry_run: When ``True``, log planned actions without calling
                :meth:`~UbikService.stop` or sending any signals.

        Returns:
            Names of services successfully stopped (including those killed via
            SIGKILL).  Services still alive after SIGKILL are excluded.
        """
        # ── Pre-shutdown ─────────────────────────────────────────────────
        log.info("MAESTRO SHUTDOWN INITIATED (dry_run=%s)", dry_run)
        self._mlog.log_service_action(
            "shutdown", "initiated", True,
            f"node={self._identity.node_type.value} dry_run={dry_run}",
        )
        if not dry_run:
            self._stop_daemon()

        services = self._local_services_in_shutdown_order()
        if not services:
            log.info(
                "shutdown: no local services on %s node",
                self._identity.node_type.value,
            )
            self._mlog.log_shutdown([])
            return []

        self._mlog.log_metrics_snapshot({
            "pre_shutdown": [svc.name for svc in services],
            "service_count": len(services),
        })

        # ── Shutdown loop ─────────────────────────────────────────────────
        stopped: list[str] = []
        deadline = time.perf_counter() + _TOTAL_TIMEOUT_S

        for svc in services:
            remaining_total = deadline - time.perf_counter()
            if remaining_total <= 0:
                log.warning(
                    "shutdown: total timeout reached — skipping %s", svc.name
                )
                break

            effective_timeout = min(timeout_per_service, remaining_total)
            log.info("shutdown: stopping %s...", svc.name)

            if dry_run:
                log.info("shutdown: [dry-run] would stop %s", svc.name)
                stopped.append(svc.name)
                continue

            # Graceful stop
            try:
                await svc.stop()
            except Exception as exc:
                log.warning("shutdown: %s stop() raised: %s", svc.name, exc)

            # Wait for DOWN confirmation
            went_down = await self._wait_for_down(svc, timeout=effective_timeout)

            if went_down:
                log.info("shutdown: %s — stopped", svc.name)
                self._mlog.log_service_action(
                    svc.name, "stop", True, "confirmed down"
                )
                stopped.append(svc.name)
            else:
                # Escalate to SIGKILL
                log.warning(
                    "shutdown: %s still alive after %.0fs — SIGKILL",
                    svc.name, effective_timeout,
                )
                self._mlog.log_service_action(
                    svc.name, "stop", False, "escalating to SIGKILL"
                )
                await self._sigkill_service(svc)
                await asyncio.sleep(_KILL_SETTLE_S)
                verify = await svc.probe_with_timeout("localhost", timeout=5.0)
                if not verify.healthy:
                    log.info("shutdown: %s — killed", svc.name)
                    self._mlog.log_service_action(
                        svc.name, "stop", True, "SIGKILL confirmed down"
                    )
                    stopped.append(svc.name)
                else:
                    log.error(
                        "shutdown: %s still alive after SIGKILL", svc.name
                    )
                    self._mlog.log_service_action(
                        svc.name, "stop", False, "still alive after SIGKILL"
                    )

        # ── Post-shutdown verification ────────────────────────────────────
        if not dry_run:
            verification = await self._verify_all_down()
            if verification:
                report_parts = [
                    f"{n}={'DOWN' if v else 'STILL_UP'}"
                    for n, v in verification.items()
                ]
                all_down = all(verification.values())
                log.info("shutdown: verification — %s", ", ".join(report_parts))
                self._mlog.log_service_action(
                    "shutdown", "verified",
                    all_down,
                    " ".join(report_parts),
                )

        self._mlog.log_shutdown(stopped)
        log.info(
            "shutdown: complete — stopped %d/%d services",
            len(stopped),
            len(services),
        )
        return stopped

    async def emergency_shutdown(self) -> None:
        """Kill all local UBIK processes immediately via SIGKILL.

        Finds processes listening on known UBIK ports using ``psutil``
        (with a fallback to ``lsof``/``fuser`` when psutil is not installed)
        and sends SIGKILL to each.

        This is a last-resort operation — prefer :meth:`orderly_shutdown`
        whenever possible.
        """
        log.warning("MAESTRO EMERGENCY SHUTDOWN — SIGKILL all local UBIK processes")
        self._mlog.log_service_action(
            "shutdown", "emergency", True,
            f"node={self._identity.node_type.value}",
        )

        local_node = self._identity.node_type
        all_ports: set[int] = set()
        for svc in self._registry.get_all():
            if svc.node == local_node:
                all_ports.update(svc.ports)

        if not all_ports:
            log.info(
                "shutdown: emergency — no ports to kill on %s node",
                local_node.value,
            )
            return

        killed_pids: list[int] = []
        try:
            import psutil
            for conn in psutil.net_connections(kind="tcp"):
                if (
                    conn.laddr
                    and conn.laddr.port in all_ports
                    and conn.pid
                ):
                    try:
                        proc = psutil.Process(conn.pid)
                        proc.kill()  # SIGKILL
                        killed_pids.append(conn.pid)
                        log.warning(
                            "shutdown: emergency killed PID %d (port %d)",
                            conn.pid, conn.laddr.port,
                        )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except ImportError:
            log.debug("shutdown: psutil not available — using lsof/fuser fallback")
            for port in sorted(all_ports):
                await _sigkill_port(port)

        detail = (
            f"killed_pids={killed_pids}" if killed_pids else "no_processes_found"
        )
        self._mlog.log_service_action("shutdown", "emergency", True, detail)
