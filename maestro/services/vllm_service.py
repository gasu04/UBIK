#!/usr/bin/env python3
"""
Maestro — vLLM Service

Probes and manages the vLLM inference server on the Somatic node.

probe:  HTTP GET to ``http://{host}:8002/health`` — 200 means the server
        process is accepting requests.
start:  Launches ``conda run -n pytorch_env vllm serve <model_path>
        --port 8002`` detached in a new session, then waits for a healthy
        probe.  Model loading can take several minutes, so max_wait_s=120s.
stop:   SIGTERM to whatever process is listening on port 8002.

vLLM runs ONLY on the Somatic node.  Never call start() on the
Hippocampal node.

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

_DEFAULT_PORT = 8002
_DEFAULT_CONDA_ENV = "pytorch_env"
_DEFAULT_MAX_WAIT_S = 120.0


class VllmService(UbikService):
    """vLLM server health probe and lifecycle manager."""

    def __init__(
        self,
        *,
        port: int = _DEFAULT_PORT,
        model_path: Optional[str] = None,
        conda_env: str = _DEFAULT_CONDA_ENV,
        max_wait_s: float = _DEFAULT_MAX_WAIT_S,
    ) -> None:
        if model_path is None:
            from maestro.config import get_config
            model_path = get_config().somatic.vllm_model_path
        self._port = port
        self._model_path = model_path
        self._conda_env = conda_env
        self._max_wait_s = max_wait_s

    @property
    def max_wait_s(self) -> float:
        return self._max_wait_s

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def node(self) -> NodeType:
        return NodeType.SOMATIC

    @property
    def ports(self) -> list[int]:
        return [self._port]

    @property
    def depends_on(self) -> list[str]:
        return []

    async def probe(self, host: str) -> ProbeResult:
        """HTTP GET to ``/health``.

        Args:
            host: IP address or hostname of the Somatic node.

        Returns:
            :class:`~maestro.services.base.ProbeResult` with
            ``details["http_status"]`` when reachable.
        """
        url = f"http://{host}:{self._port}/health"
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.debug("vllm: HTTP %d from %s", resp.status_code, url)
                healthy = resp.status_code == 200
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
        """Launch vLLM and wait for it to become healthy.

        Pre-flight checks:
          - Verifies this is the Somatic node.

        Starts ``conda run -n <env> vllm serve <model_path> --port <port>``
        in a new process session (detached), then polls ``probe()`` every
        3 seconds until the server is healthy or ``max_wait_s`` is exceeded.
        Model loading typically takes 60-120 seconds.

        Args:
            ubik_root: Unused; present for interface consistency.

        Returns:
            ``True`` when vLLM is confirmed healthy; ``False`` on error or
            timeout.
        """
        # Pre-flight: node check (vLLM must run on Somatic)
        identity = detect_node()
        if identity.node_type not in (NodeType.SOMATIC, NodeType.UNKNOWN):
            logger.error(
                "vllm: refusing to start on %s node "
                "(service belongs to somatic)",
                identity.node_type.value,
            )
            return False

        logger.info(
            "vllm: launching vllm serve %s --port %d (conda env: %s)",
            self._model_path, self._port, self._conda_env,
        )
        try:
            await asyncio.create_subprocess_exec(
                "conda", "run", "-n", self._conda_env,
                "vllm", "serve", self._model_path,
                "--port", str(self._port),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            logger.warning("vllm: start command failed: %s", exc)
            return False

        return await self._wait_for_healthy("localhost")

    async def stop(self) -> bool:
        """SIGTERM whatever process is listening on the vLLM port.

        Returns:
            ``True`` if a signal was sent; ``False`` if no process was found.
        """
        logger.debug("vllm: killing process on port %d", self._port)
        return await _kill_port(self._port)
