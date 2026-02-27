#!/usr/bin/env python3
"""
Maestro — MCP Server Health Check

Probes the FastMCP server on the Hippocampal Node.  The MCP protocol
uses Server-Sent Events (SSE); hitting the ``/mcp`` endpoint without
SSE headers causes a deliberate ``406 Not Acceptable`` response, which
is the expected "I'm alive" signal from a correctly running server.

Accepted HTTP status codes (all mean "server is up"):
    200  — direct HTTP response or ping endpoint
    400  — bad request format (server received but rejected)
    404  — endpoint not found (server is up, route differs)
    406  — missing SSE Accept header (normal FastMCP liveness probe)

Result semantics:
    HEALTHY   — server responded with any of the accepted codes.
    UNHEALTHY — connection refused, timeout, or unexpected HTTP 5xx.

Author: UBIK Project
Version: 0.1.0
"""

import time
import logging
from typing import Any

import httpx

from maestro.config import HippocampalConfig
from maestro.services.models import ServiceResult, ServiceStatus

logger = logging.getLogger(__name__)

# HTTP status codes that indicate the MCP server process is alive.
_ALIVE_STATUS_CODES: frozenset[int] = frozenset({200, 400, 404, 406})


async def check_mcp(
    cfg: HippocampalConfig,
    *,
    timeout: float = 5.0,
) -> ServiceResult:
    """Probe the MCP server liveness endpoint.

    Sends a plain GET to ``{mcp_url}/mcp``.  Because FastMCP requires
    SSE headers for proper streaming, the server returns ``406`` for a
    bare GET — this is the expected liveness signal.

    Args:
        cfg: Hippocampal node configuration providing the MCP URL.
        timeout: Maximum seconds to wait for a response.

    Returns:
        :class:`~maestro.services.models.ServiceResult` with:
            - ``details["http_status"]``: the HTTP status code received.
            - ``details["url"]``: the probed URL.
    """
    probe_url = f"{cfg.mcp_url}/mcp"
    details: dict[str, Any] = {"url": probe_url}
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(probe_url)
            latency_ms = (time.perf_counter() - start) * 1000
            details["http_status"] = resp.status_code

            if resp.status_code in _ALIVE_STATUS_CODES:
                return ServiceResult(
                    service_name="mcp",
                    status=ServiceStatus.HEALTHY,
                    latency_ms=latency_ms,
                    details=details,
                )

            # Unexpected status — server is up but misbehaving.
            return ServiceResult(
                service_name="mcp",
                status=ServiceStatus.UNHEALTHY,
                latency_ms=latency_ms,
                details=details,
                error=f"Unexpected HTTP status {resp.status_code}",
            )

    except httpx.ConnectError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("mcp connect error: %s", exc)
        return ServiceResult(
            service_name="mcp",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error="Connection refused — is the MCP server running?",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("mcp check failed: %s", exc)
        return ServiceResult(
            service_name="mcp",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error=str(exc),
        )
