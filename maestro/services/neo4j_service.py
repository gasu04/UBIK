#!/usr/bin/env python3
"""
Maestro — Neo4j Service

Probes and manages the Neo4j graph database on the Hippocampal node.

probe:  HTTP GET to ``http://{host}:7474`` — any non-5xx response means
        the server process is accepting connections.
start:  ``docker compose … up -d neo4j`` then waits for healthy probe.
stop:   ``docker compose … stop neo4j``

Author: UBIK Project
Version: 0.2.0
"""

import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from maestro.platform_detect import NodeType, detect_node
from maestro.services.base import ProbeResult, UbikService, _run_proc

logger = logging.getLogger(__name__)

_DEFAULT_HTTP_PORT = 7474
_DEFAULT_BOLT_PORT = 7687
_DEFAULT_MAX_WAIT_S = 60.0


class Neo4jService(UbikService):
    """Neo4j health probe and lifecycle manager."""

    def __init__(
        self,
        ubik_root: Optional[Path] = None,
        *,
        http_port: int = _DEFAULT_HTTP_PORT,
        bolt_port: int = _DEFAULT_BOLT_PORT,
        max_wait_s: float = _DEFAULT_MAX_WAIT_S,
    ) -> None:
        if ubik_root is None:
            from maestro.config import get_config
            ubik_root = get_config().ubik_root
        self._ubik_root = ubik_root
        self._http_port = http_port
        self._bolt_port = bolt_port
        self._max_wait_s = max_wait_s

    @property
    def max_wait_s(self) -> float:
        return self._max_wait_s

    @property
    def name(self) -> str:
        return "neo4j"

    @property
    def node(self) -> NodeType:
        return NodeType.HIPPOCAMPAL

    @property
    def ports(self) -> list[int]:
        return [self._http_port, self._bolt_port]

    @property
    def depends_on(self) -> list[str]:
        return ["docker"]

    async def probe(self, host: str) -> ProbeResult:
        """HTTP GET to the Neo4j browser port.

        Any non-5xx HTTP response means the server process is up.

        Args:
            host: IP address or hostname of the node running Neo4j.

        Returns:
            :class:`~maestro.services.base.ProbeResult` with
            ``details["http_status"]`` when reachable.
        """
        url = f"http://{host}:{self._http_port}"
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, follow_redirects=True)
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.debug("neo4j: HTTP %d from %s", resp.status_code, url)
                healthy = resp.status_code < 500
                return ProbeResult(
                    name=self.name, node=self.node, healthy=healthy,
                    latency_ms=latency_ms,
                    details={"http_status": resp.status_code, "url": url},
                    error=None if healthy else f"HTTP {resp.status_code}",
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
        """Start neo4j via ``docker compose up -d neo4j``.

        Pre-flight checks:
          - Verifies this is the Hippocampal node.
          - Verifies ``docker-compose.yml`` exists at *ubik_root*.

        After issuing the compose command, polls ``probe()`` every 3 seconds
        until Neo4j is healthy or ``max_wait_s`` is exceeded.

        Args:
            ubik_root: UBIK project root containing ``docker-compose.yml``.

        Returns:
            ``True`` when Neo4j is confirmed healthy; ``False`` on error or
            timeout.
        """
        # Pre-flight: node check
        identity = detect_node()
        if identity.node_type not in (NodeType.HIPPOCAMPAL, NodeType.UNKNOWN):
            logger.error(
                "neo4j: refusing to start on %s node "
                "(service belongs to hippocampal)",
                identity.node_type.value,
            )
            return False

        # Pre-flight: compose file must exist
        compose_file = ubik_root / "docker-compose.yml"
        if not compose_file.exists():
            logger.error("neo4j: docker-compose.yml not found: %s", compose_file)
            return False

        logger.info("neo4j: docker compose up -d (file=%s)", compose_file)
        try:
            rc, _, stderr = await _run_proc(
                "docker", "compose", "-f", str(compose_file),
                "up", "-d", "neo4j",
                timeout=60.0,
            )
            if rc != 0:
                logger.warning(
                    "neo4j: compose up failed (rc=%d): %s",
                    rc, stderr.strip()[:200],
                )
                return False
        except Exception as exc:
            logger.warning("neo4j: start command failed: %s", exc)
            return False

        return await self._wait_for_healthy("localhost")

    async def stop(self) -> bool:
        """Stop neo4j via ``docker compose stop neo4j``.

        Returns:
            ``True`` when the stop command completed without error.
        """
        compose_file = self._ubik_root / "docker-compose.yml"
        logger.debug("neo4j: docker compose stop (file=%s)", compose_file)
        try:
            rc, _, stderr = await _run_proc(
                "docker", "compose", "-f", str(compose_file),
                "stop", "neo4j",
                timeout=30.0,
            )
            if rc != 0:
                logger.warning(
                    "neo4j: stop failed (rc=%d): %s", rc, stderr.strip()[:200]
                )
                return False
            return True
        except Exception as exc:
            logger.warning("neo4j: stop failed: %s", exc)
            return False
