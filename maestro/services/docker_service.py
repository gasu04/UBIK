#!/usr/bin/env python3
"""
Maestro — Docker Service

Probes and manages the Docker daemon on the Hippocampal node.

probe:  ``docker info`` via subprocess — zero exit code means daemon is up.
start:  macOS → ``open -a Docker``; Linux → ``sudo systemctl start docker``.
        Blocks until the daemon is healthy or max_wait_s is exceeded.
stop:   ``docker stop`` on all running containers.

Docker runs ONLY on the Hippocampal node.  Never call start() on the
Somatic node.

Author: UBIK Project
Version: 0.2.0
"""

import logging
import platform
import time
from pathlib import Path

from maestro.platform_detect import NodeType, detect_node
from maestro.services.base import ProbeResult, UbikService, _run_proc

logger = logging.getLogger(__name__)

_DEFAULT_MAX_WAIT_S = 60.0


class DockerService(UbikService):
    """Docker daemon health probe and lifecycle manager."""

    def __init__(self, *, max_wait_s: float = _DEFAULT_MAX_WAIT_S) -> None:
        self._max_wait_s = max_wait_s

    @property
    def max_wait_s(self) -> float:
        return self._max_wait_s

    @property
    def name(self) -> str:
        return "docker"

    @property
    def node(self) -> NodeType:
        return NodeType.HIPPOCAMPAL

    @property
    def ports(self) -> list[int]:
        return []  # Docker daemon socket, no TCP port to kill

    @property
    def depends_on(self) -> list[str]:
        return []

    async def probe(self, host: str) -> ProbeResult:
        """Run ``docker info`` and report daemon status.

        Args:
            host: Unused for Docker (daemon is always local).

        Returns:
            :class:`~maestro.services.base.ProbeResult` with
            ``details["server_version"]`` when the daemon responds.
        """
        start = time.perf_counter()
        try:
            rc, stdout, stderr = await _run_proc(
                "docker", "info", "--format", "{{.ServerVersion}}", timeout=10.0
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            if rc != 0:
                return ProbeResult(
                    name=self.name, node=self.node, healthy=False,
                    latency_ms=latency_ms,
                    details={},
                    error=f"docker info exited {rc}: {stderr.strip()[:200]}",
                )

            version = stdout.strip()
            logger.debug("docker: server_version=%s", version)
            return ProbeResult(
                name=self.name, node=self.node, healthy=True,
                latency_ms=latency_ms,
                details={"server_version": version},
            )
        except FileNotFoundError:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return ProbeResult(
                name=self.name, node=self.node, healthy=False,
                latency_ms=latency_ms,
                details={},
                error="docker binary not found",
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return ProbeResult(
                name=self.name, node=self.node, healthy=False,
                latency_ms=latency_ms,
                details={},
                error=str(exc),
            )

    async def start(self, ubik_root: Path) -> bool:
        """Launch Docker Desktop (macOS) or start the daemon via systemd (Linux).

        Pre-flight checks:
          - Verifies this is the Hippocampal node.

        After issuing the start command, polls ``probe()`` every 3 seconds
        until the daemon is healthy or ``max_wait_s`` is exceeded.

        Args:
            ubik_root: UBIK project root (unused; present for interface
                consistency).

        Returns:
            ``True`` when Docker is confirmed healthy; ``False`` on error or
            timeout.
        """
        # Pre-flight: node check
        identity = detect_node()
        if identity.node_type not in (NodeType.HIPPOCAMPAL, NodeType.UNKNOWN):
            logger.error(
                "docker: refusing to start on %s node "
                "(service belongs to hippocampal)",
                identity.node_type.value,
            )
            return False

        system = platform.system()
        try:
            if system == "Darwin":
                logger.info("docker: starting Docker Desktop via open -a Docker")
                await _run_proc("open", "-a", "Docker", timeout=10.0)
            else:
                logger.info("docker: starting daemon via systemctl")
                rc, _, stderr = await _run_proc(
                    "sudo", "systemctl", "start", "docker", timeout=30.0
                )
                if rc != 0:
                    logger.warning(
                        "docker: systemctl start failed (rc=%d): %s",
                        rc, stderr.strip()[:200],
                    )
                    return False
        except Exception as exc:
            logger.warning("docker: start command failed: %s", exc)
            return False

        return await self._wait_for_healthy("localhost")

    async def stop(self) -> bool:
        """Stop all running containers via ``docker stop``.

        Returns:
            ``True`` when the stop command completed without error.
        """
        try:
            logger.debug("docker: stopping all containers")
            rc, stdout, _ = await _run_proc("docker", "ps", "-q", timeout=10.0)
            container_ids = [
                cid.strip() for cid in stdout.splitlines() if cid.strip()
            ]
            if container_ids:
                await _run_proc("docker", "stop", *container_ids, timeout=30.0)
            return True
        except Exception as exc:
            logger.warning("docker stop failed: %s", exc)
            return False
