#!/usr/bin/env python3
"""
Maestro — Health Runner

Orchestrates concurrent health checks across all UBIK services and
returns a single :class:`~maestro.services.models.ClusterHealth` snapshot.

All six checks run in parallel via ``asyncio.gather``.  Each check is
independently guarded by ``asyncio.wait_for`` so a single hung probe
cannot stall the entire health cycle.

Services checked:
    neo4j      — Neo4j graph DB on Hippocampal Node (Bolt probe)
    chromadb   — ChromaDB vector store on Hippocampal Node (HTTP probe)
    mcp        — MCP server on Hippocampal Node (HTTP liveness)
    vllm       — vLLM inference on Somatic Node (HTTP probe)
    tailscale  — Tailscale mesh network (CLI probe)
    docker     — Docker daemon + container status (CLI probe)

Usage:
    import asyncio
    from maestro.config import get_config
    from maestro.services.health_runner import run_all_checks

    cluster = asyncio.run(run_all_checks(get_config()))
    print(cluster.to_json())

Author: UBIK Project
Version: 0.1.0
"""

import asyncio
import logging
from datetime import datetime, timezone

from maestro.config import AppConfig
from maestro.services.chromadb_check import check_chromadb
from maestro.services.docker_check import check_docker
from maestro.services.mcp_check import check_mcp
from maestro.services.models import ClusterHealth, ServiceResult, ServiceStatus
from maestro.services.neo4j_check import check_neo4j
from maestro.services.tailscale_check import check_tailscale
from maestro.services.vllm_check import check_vllm

logger = logging.getLogger(__name__)


def _timeout_result(service_name: str, timeout: float) -> ServiceResult:
    """Build an UNHEALTHY result for a check that timed out.

    Args:
        service_name: The name of the timed-out service.
        timeout: The timeout value that was exceeded.

    Returns:
        :class:`~maestro.services.models.ServiceResult` with UNHEALTHY status.
    """
    return ServiceResult(
        service_name=service_name,
        status=ServiceStatus.UNHEALTHY,
        error=f"Health check timed out after {timeout}s",
    )


def _exception_result(service_name: str, exc: BaseException) -> ServiceResult:
    """Build an UNHEALTHY result for an unexpected exception.

    Args:
        service_name: The name of the failing service.
        exc: The exception that was raised.

    Returns:
        :class:`~maestro.services.models.ServiceResult` with UNHEALTHY status.
    """
    logger.error("Unhandled exception in %s check: %s", service_name, exc)
    return ServiceResult(
        service_name=service_name,
        status=ServiceStatus.UNHEALTHY,
        error=f"Unhandled error: {exc}",
    )


async def _guarded(
    coro,
    service_name: str,
    timeout: float,
) -> ServiceResult:
    """Wrap a health check coroutine with a timeout guard.

    Args:
        coro: The check coroutine to run.
        service_name: Human-readable name for error messages.
        timeout: Maximum seconds before the check is cancelled.

    Returns:
        The check's :class:`~maestro.services.models.ServiceResult`, or a
        synthesised UNHEALTHY result on timeout or unexpected exception.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("%s check timed out after %.1fs", service_name, timeout)
        return _timeout_result(service_name, timeout)
    except Exception as exc:
        return _exception_result(service_name, exc)


async def run_all_checks(
    cfg: AppConfig,
    *,
    timeout: float = 10.0,
) -> ClusterHealth:
    """Run all six UBIK service health checks concurrently.

    All checks start simultaneously and are individually guarded by
    ``asyncio.wait_for``.  A single unresponsive service will not block
    the others.

    Args:
        cfg: Application configuration providing all node and service
            parameters.
        timeout: Per-check timeout in seconds.  This is the wall-clock
            limit for each individual check, not the total run time.

    Returns:
        :class:`~maestro.services.models.ClusterHealth` containing one
        :class:`~maestro.services.models.ServiceResult` per service, plus
        an aggregated ``overall_status``.

    Example:
        >>> import asyncio
        >>> from maestro.config import get_config
        >>> from maestro.services.health_runner import run_all_checks
        >>> cluster = asyncio.run(run_all_checks(get_config()))
        >>> print(cluster.overall_status)
        ServiceStatus.HEALTHY
    """
    h = cfg.hippocampal
    s = cfg.somatic

    checks = [
        _guarded(check_neo4j(h, timeout=timeout), "neo4j", timeout),
        _guarded(check_chromadb(h, timeout=timeout), "chromadb", timeout),
        _guarded(check_mcp(h, timeout=timeout), "mcp", timeout),
        _guarded(check_vllm(s, timeout=timeout), "vllm", timeout),
        _guarded(check_tailscale(h, s, timeout=timeout), "tailscale", timeout),
        _guarded(check_docker(timeout=timeout), "docker", timeout),
    ]

    results: list[ServiceResult] = await asyncio.gather(*checks)

    services = {r.service_name: r for r in results}

    healthy = [n for n, r in services.items() if r.is_healthy]
    unhealthy = [n for n, r in services.items() if not r.is_healthy]
    logger.info(
        "Health check complete: %d/%d healthy. Unhealthy: %s",
        len(healthy),
        len(services),
        unhealthy or "none",
    )

    return ClusterHealth(
        services=services,
        checked_at=datetime.now(timezone.utc),
    )


# Ordered canonical service names — used by CLI for validation and ordering.
ALL_SERVICE_NAMES: tuple[str, ...] = (
    "neo4j",
    "chromadb",
    "mcp",
    "vllm",
    "tailscale",
    "docker",
)


async def run_selected_checks(
    cfg: AppConfig,
    services: set[str] | None = None,
    *,
    timeout: float = 10.0,
) -> ClusterHealth:
    """Run health checks for a specific subset of services.

    When *services* is ``None`` or empty, delegates to
    :func:`run_all_checks` so callers never need to branch.

    Args:
        cfg: Application configuration providing all node and service
            parameters.
        services: Set of service names to probe.  Must be a subset of
            :data:`ALL_SERVICE_NAMES`.  Pass ``None`` to check all.
        timeout: Per-check timeout in seconds.

    Returns:
        :class:`~maestro.services.models.ClusterHealth` containing one
        result per requested service.

    Raises:
        ValueError: If *services* contains unrecognised service names.

    Example:
        >>> cluster = await run_selected_checks(cfg, {"neo4j", "chromadb"})
        >>> assert set(cluster.services.keys()) == {"neo4j", "chromadb"}
    """
    if not services:
        return await run_all_checks(cfg, timeout=timeout)

    unknown = services - set(ALL_SERVICE_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown service(s): {sorted(unknown)}.  "
            f"Valid choices: {list(ALL_SERVICE_NAMES)}"
        )

    h, s = cfg.hippocampal, cfg.somatic

    # Build coroutines only for requested services (avoids "never awaited" warnings).
    check_map: dict[str, object] = {}
    if "neo4j" in services:
        check_map["neo4j"] = check_neo4j(h, timeout=timeout)
    if "chromadb" in services:
        check_map["chromadb"] = check_chromadb(h, timeout=timeout)
    if "mcp" in services:
        check_map["mcp"] = check_mcp(h, timeout=timeout)
    if "vllm" in services:
        check_map["vllm"] = check_vllm(s, timeout=timeout)
    if "tailscale" in services:
        check_map["tailscale"] = check_tailscale(h, s, timeout=timeout)
    if "docker" in services:
        check_map["docker"] = check_docker(timeout=timeout)

    names = list(check_map.keys())
    guarded = [
        _guarded(coro, name, timeout)  # type: ignore[arg-type]
        for name, coro in check_map.items()
    ]
    results: list[ServiceResult] = await asyncio.gather(*guarded)

    logger.info(
        "Selected check complete (%s): %d/%d healthy",
        sorted(services),
        sum(1 for r in results if r.is_healthy),
        len(results),
    )

    return ClusterHealth(
        services={r.service_name: r for r in results},
        checked_at=datetime.now(timezone.utc),
    )
