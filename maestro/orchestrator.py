#!/usr/bin/env python3
"""
Maestro — Orchestrator

Coordinates cluster-wide health checks and ordered service startup across
both UBIK nodes.

Public API:
    Orchestrator   — check all services, ensure local services are running,
                     generate status reports.

Design:
    full_status_check()   — concurrent probes, local via localhost, remote
                             via Tailscale IP.  Never starts or stops anything.
    ensure_all_running()  — dependency-ordered start of local services;
                             skips dependents when a dependency fails to start.
    generate_report()     — plaintext table summarising probe results.

Usage::

    import asyncio
    from maestro.orchestrator import Orchestrator
    from maestro.services import ServiceRegistry
    from maestro.platform_detect import detect_node

    orch = Orchestrator(ServiceRegistry(), detect_node())
    statuses = asyncio.run(orch.full_status_check())
    print(asyncio.run(orch.generate_report(statuses)))

Author: UBIK Project
Version: 0.1.0
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from maestro.platform_detect import NodeIdentity, NodeType
from maestro.services import ServiceRegistry
from maestro.services.base import ProbeResult, UbikService

logger = logging.getLogger(__name__)


class Orchestrator:
    """Cluster-wide service health checker and startup coordinator.

    Args:
        registry: Registry of all service instances.
        identity: Identity of the node this orchestrator is running on.
            Determines which services are local (start-eligible) and
            which probe host to use for remote services.
    """

    def __init__(self, registry: ServiceRegistry, identity: NodeIdentity) -> None:
        self._registry = registry
        self._identity = identity

    # ── Host resolution ────────────────────────────────────────────────────

    def _probe_host_for(self, svc: UbikService) -> str:
        """Return the probe host for a service.

        Local services are probed via ``"localhost"``.  Remote services are
        reached via their node's Tailscale IP from the registry config.

        Args:
            svc: Service instance to resolve a host for.

        Returns:
            Hostname or IP string to pass to :meth:`~UbikService.probe`.
        """
        local_node = self._identity.node_type
        if svc.node == local_node or local_node == NodeType.UNKNOWN:
            return "localhost"
        cfg = self._registry.cfg
        if svc.node == NodeType.HIPPOCAMPAL:
            return cfg.hippocampal.tailscale_ip
        if svc.node == NodeType.SOMATIC:
            return cfg.somatic.tailscale_ip
        return "localhost"

    # ── Public API ─────────────────────────────────────────────────────────

    async def full_status_check(self) -> dict[str, ProbeResult]:
        """Probe ALL registered services concurrently.

        Local services are probed via ``localhost``; remote services via their
        Tailscale IP.  All probes run in parallel via :func:`asyncio.gather`.

        Returns:
            Ordered dict mapping ``service_name`` → :class:`~ProbeResult`,
            in registration order.  Never raises.
        """
        services = self._registry.get_all()
        probe_coros = [
            svc.probe_with_timeout(self._probe_host_for(svc))
            for svc in services
        ]

        raw = await asyncio.gather(*probe_coros, return_exceptions=True)

        result: dict[str, ProbeResult] = {}
        for svc, outcome in zip(services, raw):
            if isinstance(outcome, Exception):
                result[svc.name] = ProbeResult(
                    name=svc.name,
                    node=svc.node,
                    healthy=False,
                    latency_ms=0.0,
                    error=f"probe raised: {outcome}",
                )
            else:
                result[svc.name] = outcome

        return result

    async def ensure_all_running(self) -> list[str]:
        """Bring all LOCAL services to a healthy state in dependency order.

        For each local service (in the topological startup order from
        :meth:`~ServiceRegistry.get_startup_order`):

        1. If a dependency previously failed to start, skip this service
           and add it to the failed list.
        2. If the service is already healthy, skip it.
        3. Verify all local dependencies are healthy.  If any are not,
           skip this service.
        4. Attempt :meth:`~UbikService.start` (which includes the health-wait
           loop).  If it returns ``False``, record the failure.

        Args:
            (none)

        Returns:
            List of service names that could not be started.  An empty list
            means all local services are healthy.
        """
        local_node = self._identity.node_type
        ubik_root = self._registry.cfg.ubik_root
        failed: list[str] = []

        for svc in self._registry.get_startup_order():
            if svc.node != local_node:
                logger.debug(
                    "%s is on %s node — skipping (not local)",
                    svc.name, svc.node.value,
                )
                continue

            # Skip if a dependency failed to start
            failed_deps = [d for d in svc.depends_on if d in failed]
            if failed_deps:
                logger.warning(
                    "%s: skipping — failed dependencies: %s",
                    svc.name, failed_deps,
                )
                failed.append(svc.name)
                continue

            # Already healthy? Nothing to do.
            current = await svc.probe_with_timeout("localhost")
            if current.healthy:
                logger.info(
                    "%s: already healthy (%.0fms)", svc.name, current.latency_ms
                )
                continue

            # Verify all local dependencies are healthy before starting
            dep_ok = True
            for dep_name in svc.depends_on:
                dep_svc = next(
                    (s for s in self._registry.get_all() if s.name == dep_name),
                    None,
                )
                if dep_svc is None or dep_svc.node != local_node:
                    continue
                dep_result = await dep_svc.probe_with_timeout("localhost")
                if not dep_result.healthy:
                    logger.error(
                        "%s: dependency '%s' is not healthy — cannot start",
                        svc.name, dep_name,
                    )
                    dep_ok = False
                    break

            if not dep_ok:
                failed.append(svc.name)
                continue

            # Attempt to start (start() blocks until healthy or timeout)
            logger.info("%s: unhealthy — attempting start", svc.name)
            success = await svc.start(ubik_root)
            if success:
                logger.info("%s: started successfully", svc.name)
            else:
                logger.error("%s: failed to start", svc.name)
                failed.append(svc.name)

        return failed

    async def generate_report(
        self, statuses: dict[str, ProbeResult]
    ) -> str:
        """Build a plaintext status report table.

        Args:
            statuses: Dict of service name → :class:`~ProbeResult`, as
                returned by :meth:`full_status_check`.

        Returns:
            Multi-line string suitable for printing to a terminal or writing
            to a log file.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        node_desc = (
            f"{self._identity.node_type.value} ({self._identity.hostname})"
        )

        lines: list[str] = []
        sep = "=" * 62
        dash = "-" * 62

        lines.append(sep)
        lines.append("UBIK MAESTRO — Cluster Status Report")
        lines.append(f"Node : {node_desc}")
        lines.append(f"Time : {now}")
        lines.append(sep)
        lines.append(
            f"{'SERVICE':<14}{'NODE':<14}{'STATUS':<14}{'LATENCY':<10}{'ERROR'}"
        )
        lines.append(dash)

        healthy_count = 0
        for name, result in statuses.items():
            if result.healthy:
                status_str = "healthy"
                healthy_count += 1
            else:
                status_str = "DOWN"
            latency_str = f"{result.latency_ms:.0f}ms"
            error_str = (result.error or "")[:30]
            lines.append(
                f"{name:<14}"
                f"{result.node.value:<14}"
                f"{status_str:<14}"
                f"{latency_str:<10}"
                f"{error_str}"
            )

        lines.append(sep)
        total = len(statuses)
        lines.append(
            f"Summary: {healthy_count}/{total} services healthy"
        )

        return "\n".join(lines)
