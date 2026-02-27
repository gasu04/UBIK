#!/usr/bin/env python3
"""
Maestro — Docker Container Health Check (Hippocampal Node)

Infers whether the UBIK Docker containers on the Hippocampal node are
running by examining the results of the neo4j and chromadb service checks.
No direct Docker CLI or remote API access is required.

Inference logic:
    A service being HEALTHY proves its container is running.
    A service being UNHEALTHY is ambiguous (auth error ≠ container down),
    so we report DEGRADED rather than UNHEALTHY in that case.

    ubik-neo4j    — inferred from neo4j ServiceResult
    ubik-chromadb — inferred from chromadb ServiceResult

Result semantics:
    HEALTHY  — both neo4j and chromadb reported HEALTHY
               (containers confirmed running).
    DEGRADED — one or both services unhealthy; containers may be
               running but service is misconfigured, or container
               is actually stopped.
    UNHEALTHY — should not occur via inference; reserved for future
                direct checks.

Author: UBIK Project
Version: 0.3.0
"""

import time
from typing import Any

from maestro.services.models import ServiceResult, ServiceStatus

# Map from container name to the service result that witnesses it.
_CONTAINER_SERVICE_MAP: dict[str, str] = {
    "ubik-neo4j": "neo4j",
    "ubik-chromadb": "chromadb",
}


async def check_docker(
    neo4j_result: ServiceResult,
    chromadb_result: ServiceResult,
) -> ServiceResult:
    """Infer Docker container health from neo4j and chromadb service results.

    Since SSH and the Docker remote API are not available from the Somatic
    node, container liveness is inferred: a HEALTHY service probe proves
    its container is running; an UNHEALTHY probe is treated as ambiguous.

    Args:
        neo4j_result: Already-computed result from ``check_neo4j``.
        chromadb_result: Already-computed result from ``check_chromadb``.

    Returns:
        :class:`~maestro.services.models.ServiceResult` with:
            - ``details["containers"]``: mapping of container name →
              ``"running"`` (confirmed) or ``"unknown"`` (ambiguous).
            - ``details["inferred"]``: always ``True``.
    """
    start = time.perf_counter()

    service_results = {
        "neo4j": neo4j_result,
        "chromadb": chromadb_result,
    }

    containers: dict[str, str] = {}
    for container, svc in _CONTAINER_SERVICE_MAP.items():
        result = service_results[svc]
        containers[container] = "running" if result.is_healthy else "unknown"

    details: dict[str, Any] = {
        "containers": containers,
        "inferred": True,
    }

    confirmed_running = [c for c, st in containers.items() if st == "running"]
    unknown = [c for c, st in containers.items() if st == "unknown"]
    latency_ms = (time.perf_counter() - start) * 1000

    if len(confirmed_running) == len(containers):
        return ServiceResult(
            service_name="docker",
            status=ServiceStatus.HEALTHY,
            latency_ms=latency_ms,
            details=details,
        )

    missing_str = ", ".join(unknown)
    return ServiceResult(
        service_name="docker",
        status=ServiceStatus.DEGRADED,
        latency_ms=latency_ms,
        details=details,
        error=f"Cannot confirm containers running: {missing_str}",
    )
