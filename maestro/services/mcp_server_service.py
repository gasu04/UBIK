#!/usr/bin/env python3
"""
Maestro — MCP Server Service

Probes and manages the FastMCP server on the Hippocampal node.

probe:  HTTP GET to ``http://{host}:8080/`` — FastMCP returns 200, 400, 404,
        or 406 depending on the path/headers; any of these means the server
        process is up.
start:  Launches ``{ubik_root}/hippocampal/run_mcp.sh start`` detached via
        a new process session, then waits for a healthy probe.
stop:   SIGTERM to whatever process is listening on port 8080.

Author: UBIK Project
Version: 0.2.0
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from maestro.platform_detect import NodeType, detect_node
from maestro.services.base import ProbeResult, UbikService, _kill_port

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8080
_DEFAULT_MAX_WAIT_S = 15.0

# HTTP status codes that indicate the MCP server process is alive.
_ALIVE_STATUS_CODES: frozenset[int] = frozenset({200, 400, 404, 406})


class McpServerService(UbikService):
    """MCP server health probe and lifecycle manager."""

    def __init__(
        self,
        ubik_root: Optional[Path] = None,
        *,
        port: int = _DEFAULT_PORT,
        max_wait_s: float = _DEFAULT_MAX_WAIT_S,
    ) -> None:
        if ubik_root is None:
            from maestro.config import get_config
            ubik_root = get_config().ubik_root
        self._ubik_root = ubik_root
        self._port = port
        self._max_wait_s = max_wait_s

    @property
    def max_wait_s(self) -> float:
        return self._max_wait_s

    @property
    def name(self) -> str:
        return "mcp"

    @property
    def node(self) -> NodeType:
        return NodeType.HIPPOCAMPAL

    @property
    def ports(self) -> list[int]:
        return [self._port]

    @property
    def depends_on(self) -> list[str]:
        return ["neo4j", "chromadb"]

    async def probe(self, host: str) -> ProbeResult:
        """HTTP GET to the MCP server root endpoint.

        FastMCP returns 406 for bare GET requests (missing SSE Accept header),
        which is the expected liveness signal.  200, 400, 404 are also accepted.

        Args:
            host: IP address or hostname of the node running the MCP server.

        Returns:
            :class:`~maestro.services.base.ProbeResult` with
            ``details["http_status"]`` when reachable.
        """
        url = f"http://{host}:{self._port}/"
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.debug("mcp: HTTP %d from %s", resp.status_code, url)
                healthy = resp.status_code in _ALIVE_STATUS_CODES
                return ProbeResult(
                    name=self.name, node=self.node, healthy=healthy,
                    latency_ms=latency_ms,
                    details={"http_status": resp.status_code, "url": url},
                    error=None if healthy else f"Unexpected HTTP {resp.status_code}",
                )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return ProbeResult(
                name=self.name, node=self.node, healthy=False,
                latency_ms=latency_ms,
                details={"url": url},
                error=str(exc),
            )

    async def start(self, ubik_root: Path) -> bool:
        """Launch the MCP server and wait for it to become healthy.

        Pre-flight checks:
          - Verifies this is the Hippocampal node.
          - Verifies ``hippocampal/run_mcp.sh`` exists at *ubik_root*.

        Starts ``run_mcp.sh start`` in a new process session (detached), then
        polls ``probe()`` every 3 seconds until the server is healthy or
        ``max_wait_s`` is exceeded.

        Args:
            ubik_root: UBIK project root containing ``hippocampal/run_mcp.sh``.

        Returns:
            ``True`` when the MCP server is confirmed healthy; ``False`` on
            error or timeout.
        """
        # Pre-flight: node check
        identity = detect_node()
        if identity.node_type not in (NodeType.HIPPOCAMPAL, NodeType.UNKNOWN):
            logger.error(
                "mcp: refusing to start on %s node "
                "(service belongs to hippocampal)",
                identity.node_type.value,
            )
            return False

        # Pre-flight: run_mcp.sh must exist
        run_mcp = ubik_root / "hippocampal" / "run_mcp.sh"
        if not run_mcp.exists():
            logger.error("mcp: run_mcp.sh not found: %s", run_mcp)
            return False

        logger.info("mcp: launching %s start", run_mcp)
        try:
            await asyncio.create_subprocess_exec(
                str(run_mcp), "start",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            logger.warning("mcp: start command failed: %s", exc)
            return False

        return await self._wait_for_healthy("localhost")

    async def stop(self) -> bool:
        """SIGTERM whatever process is listening on the MCP port.

        Returns:
            ``True`` if a signal was sent; ``False`` if no process was found.
        """
        logger.debug("mcp: killing process on port %d", self._port)
        return await _kill_port(self._port)
