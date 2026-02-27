#!/usr/bin/env python3
"""
Maestro — ChromaDB Service

Probes and manages the ChromaDB vector store on the Hippocampal node.

probe:  HTTP GET to ``/api/v2/heartbeat`` — 200 means the server is up.
        Falls back to ``/api/v1/heartbeat`` for older ChromaDB versions.
start:  ``docker compose … up -d chromadb`` then waits for healthy probe.
stop:   ``docker compose … stop chromadb``

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

_DEFAULT_PORT = 8001
_DEFAULT_MAX_WAIT_S = 30.0


class ChromaDbService(UbikService):
    """ChromaDB health probe and lifecycle manager."""

    def __init__(
        self,
        ubik_root: Optional[Path] = None,
        *,
        port: int = _DEFAULT_PORT,
        token: Optional[str] = None,
        max_wait_s: float = _DEFAULT_MAX_WAIT_S,
    ) -> None:
        if ubik_root is None:
            from maestro.config import get_config
            cfg = get_config()
            ubik_root = cfg.ubik_root
            if token is None:
                token = cfg.hippocampal.chromadb_token
        self._ubik_root = ubik_root
        self._port = port
        self._token = token
        self._max_wait_s = max_wait_s

    @property
    def max_wait_s(self) -> float:
        return self._max_wait_s

    @property
    def name(self) -> str:
        return "chromadb"

    @property
    def node(self) -> NodeType:
        return NodeType.HIPPOCAMPAL

    @property
    def ports(self) -> list[int]:
        return [self._port]

    @property
    def depends_on(self) -> list[str]:
        return ["docker"]

    async def probe(self, host: str) -> ProbeResult:
        """HTTP GET to the ChromaDB heartbeat endpoint.

        Tries ``/api/v2/heartbeat`` first; falls back to ``/api/v1/heartbeat``
        when the server returns 404 (older ChromaDB version).

        Args:
            host: IP address or hostname of the node running ChromaDB.

        Returns:
            :class:`~maestro.services.base.ProbeResult` with
            ``details["api_version"]`` when reachable.
        """
        base_url = f"http://{host}:{self._port}"
        start = time.perf_counter()
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{base_url}/api/v2/heartbeat", headers=headers
                )
                if resp.status_code == 404:
                    resp = await client.get(
                        f"{base_url}/api/v1/heartbeat", headers=headers
                    )
                    api_version = "v1"
                else:
                    api_version = "v2"

                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.debug(
                    "chromadb: HTTP %d (api=%s) from %s",
                    resp.status_code, api_version, base_url,
                )

                if resp.status_code == 200:
                    return ProbeResult(
                        name=self.name, node=self.node, healthy=True,
                        latency_ms=latency_ms,
                        details={"api_version": api_version, "url": base_url},
                    )
                return ProbeResult(
                    name=self.name, node=self.node, healthy=False,
                    latency_ms=latency_ms,
                    details={"api_version": api_version, "url": base_url},
                    error=f"HTTP {resp.status_code}",
                )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return ProbeResult(
                name=self.name, node=self.node, healthy=False,
                latency_ms=latency_ms,
                details={"url": base_url},
                error=str(exc),
            )

    async def start(self, ubik_root: Path) -> bool:
        """Start chromadb via ``docker compose up -d chromadb``.

        Pre-flight checks:
          - Verifies this is the Hippocampal node.
          - Verifies ``docker-compose.yml`` exists at *ubik_root*.

        After issuing the compose command, polls ``probe()`` every 3 seconds
        until ChromaDB is healthy or ``max_wait_s`` is exceeded.

        Args:
            ubik_root: UBIK project root containing ``docker-compose.yml``.

        Returns:
            ``True`` when ChromaDB is confirmed healthy; ``False`` on error or
            timeout.
        """
        # Pre-flight: node check
        identity = detect_node()
        if identity.node_type not in (NodeType.HIPPOCAMPAL, NodeType.UNKNOWN):
            logger.error(
                "chromadb: refusing to start on %s node "
                "(service belongs to hippocampal)",
                identity.node_type.value,
            )
            return False

        # Pre-flight: compose file must exist
        compose_file = ubik_root / "docker-compose.yml"
        if not compose_file.exists():
            logger.error(
                "chromadb: docker-compose.yml not found: %s", compose_file
            )
            return False

        logger.info("chromadb: docker compose up -d (file=%s)", compose_file)
        try:
            rc, _, stderr = await _run_proc(
                "docker", "compose", "-f", str(compose_file),
                "up", "-d", "chromadb",
                timeout=60.0,
            )
            if rc != 0:
                logger.warning(
                    "chromadb: compose up failed (rc=%d): %s",
                    rc, stderr.strip()[:200],
                )
                return False
        except Exception as exc:
            logger.warning("chromadb: start command failed: %s", exc)
            return False

        return await self._wait_for_healthy("localhost")

    async def stop(self) -> bool:
        """Stop chromadb via ``docker compose stop chromadb``.

        Returns:
            ``True`` when the stop command completed without error.
        """
        compose_file = self._ubik_root / "docker-compose.yml"
        logger.debug("chromadb: docker compose stop (file=%s)", compose_file)
        try:
            rc, _, stderr = await _run_proc(
                "docker", "compose", "-f", str(compose_file),
                "stop", "chromadb",
                timeout=30.0,
            )
            if rc != 0:
                logger.warning(
                    "chromadb: stop failed (rc=%d): %s", rc, stderr.strip()[:200]
                )
                return False
            return True
        except Exception as exc:
            logger.warning("chromadb: stop failed: %s", exc)
            return False
