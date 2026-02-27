#!/usr/bin/env python3
"""
Maestro — Docker Container Health Check

Verifies that the Docker daemon is running and that all UBIK containers
are in the ``running`` state.  Uses ``asyncio.to_thread`` to avoid
blocking the event loop on subprocess calls.

Containers monitored:
    ubik-neo4j    — Neo4j graph database
    ubik-chromadb — ChromaDB vector store

Result semantics:
    HEALTHY   — Docker daemon up and all containers running.
    DEGRADED  — Docker daemon up but one or more containers stopped/absent.
    UNHEALTHY — Docker daemon not running or not installed.

Author: UBIK Project
Version: 0.1.0
"""

import asyncio
import logging
import subprocess
import time
from typing import Any

from maestro.services.models import ServiceResult, ServiceStatus

logger = logging.getLogger(__name__)

# Containers that must be running for UBIK to operate.
_REQUIRED_CONTAINERS: list[str] = ["ubik-neo4j", "ubik-chromadb"]


def _check_docker_sync(timeout: float) -> dict[str, Any]:
    """Run Docker status checks synchronously.

    Args:
        timeout: Per-subprocess call timeout in seconds.

    Returns:
        Dict with keys:
            ``daemon_ok`` (bool), ``containers`` (dict[name, status str]).

    Raises:
        FileNotFoundError: Docker CLI not found.
    """
    # Verify Docker daemon is reachable.
    daemon_result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if daemon_result.returncode != 0:
        return {"daemon_ok": False, "containers": {}}

    containers: dict[str, str] = {}
    for name in _REQUIRED_CONTAINERS:
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if inspect.returncode == 0:
            containers[name] = inspect.stdout.strip()
        else:
            containers[name] = "not found"

    return {"daemon_ok": True, "containers": containers}


async def check_docker(
    *,
    timeout: float = 10.0,
) -> ServiceResult:
    """Probe Docker daemon and UBIK container status.

    Runs synchronous Docker CLI calls in a thread to keep the event loop
    free.

    Args:
        timeout: Maximum seconds for each subprocess call.

    Returns:
        :class:`~maestro.services.models.ServiceResult` with:
            - ``details["daemon_ok"]``: whether Docker daemon responded.
            - ``details["containers"]``: mapping of container name →
              status string (e.g. ``"running"``, ``"exited"``,
              ``"not found"``).
    """
    details: dict[str, Any] = {}
    start = time.perf_counter()

    try:
        result: dict[str, Any] = await asyncio.to_thread(
            _check_docker_sync, timeout
        )
        latency_ms = (time.perf_counter() - start) * 1000

        if not result["daemon_ok"]:
            details["daemon_ok"] = False
            return ServiceResult(
                service_name="docker",
                status=ServiceStatus.UNHEALTHY,
                latency_ms=latency_ms,
                details=details,
                error="Docker daemon is not running",
            )

        containers: dict[str, str] = result["containers"]
        details["daemon_ok"] = True
        details["containers"] = containers

        not_running = [
            name
            for name, status in containers.items()
            if status != "running"
        ]

        if not_running:
            return ServiceResult(
                service_name="docker",
                status=ServiceStatus.DEGRADED,
                latency_ms=latency_ms,
                details=details,
                error=f"Containers not running: {not_running}",
            )

        return ServiceResult(
            service_name="docker",
            status=ServiceStatus.HEALTHY,
            latency_ms=latency_ms,
            details=details,
        )

    except FileNotFoundError:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("docker binary not found")
        return ServiceResult(
            service_name="docker",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error="Docker is not installed",
        )
    except subprocess.TimeoutExpired:
        latency_ms = (time.perf_counter() - start) * 1000
        return ServiceResult(
            service_name="docker",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error=f"Docker check timed out after {timeout}s",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("docker check failed: %s", exc)
        return ServiceResult(
            service_name="docker",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error=str(exc),
        )
