#!/usr/bin/env python3
"""
Maestro — Tailscale Network Health Check

Verifies that the Tailscale mesh network is operating correctly by
querying the Tailscale CLI for status JSON.  Two aspects are checked:

1. Local node  — ``Self.Online`` is ``true`` in the status JSON.
2. Remote peer — the Somatic node appears in ``Peer`` and is online.

The check is cross-platform:
    macOS  → tries the app bundle binary first, then falls back to PATH
    Linux  → expects ``tailscale`` on PATH

Result semantics:
    HEALTHY   — local node online AND Somatic peer online.
    DEGRADED  — local node online BUT Somatic peer offline or not found.
    UNHEALTHY — Tailscale not installed, not running, or local node
                offline.

Note:
    This check uses ``asyncio.to_thread`` to avoid blocking the event
    loop on the subprocess call.

Author: UBIK Project
Version: 0.1.0
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import time
from typing import Any

from maestro.config import HippocampalConfig, SomaticConfig
from maestro.services.models import ServiceResult, ServiceStatus

logger = logging.getLogger(__name__)

# macOS Tailscale app bundle path
_TAILSCALE_MACOS_APP = (
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
)


def _find_tailscale_binary() -> str:
    """Locate the Tailscale CLI binary.

    Returns:
        Full path to the Tailscale binary on macOS when the app is
        installed, otherwise ``"tailscale"`` (assumes it is on PATH).
    """
    if (
        platform.system() == "Darwin"
        and os.path.exists(_TAILSCALE_MACOS_APP)
    ):
        return _TAILSCALE_MACOS_APP
    return "tailscale"


def _run_tailscale_status(timeout: float) -> dict[str, Any]:
    """Run ``tailscale status --json`` synchronously.

    Args:
        timeout: Subprocess timeout in seconds.

    Returns:
        Parsed JSON status dict.

    Raises:
        FileNotFoundError: Tailscale binary not found.
        subprocess.TimeoutExpired: CLI call timed out.
        subprocess.CalledProcessError: Non-zero exit code.
        json.JSONDecodeError: Output is not valid JSON.
    """
    cmd = _find_tailscale_binary()
    result = subprocess.run(
        [cmd, "status", "--json"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return json.loads(result.stdout)


async def check_tailscale(
    hippocampal: HippocampalConfig,
    somatic: SomaticConfig,
    *,
    timeout: float = 10.0,
) -> ServiceResult:
    """Probe Tailscale network status.

    Runs ``tailscale status --json`` in a thread to avoid blocking the
    event loop, then inspects the JSON for local node status and the
    presence of the Somatic peer.

    Args:
        hippocampal: Hippocampal node config (used to record the local
            node's expected identity).
        somatic: Somatic node config providing the peer IP to look up.
        timeout: Maximum seconds for the subprocess call.

    Returns:
        :class:`~maestro.services.models.ServiceResult` with:
            - ``details["self_hostname"]``: local Tailscale hostname.
            - ``details["self_online"]``: whether local node is online.
            - ``details["somatic_ip"]``: the peer IP we searched for.
            - ``details["somatic_online"]``: peer online status.
            - ``details["peer_count"]``: total peers visible in mesh.
    """
    somatic_ip = somatic.tailscale_ip
    details: dict[str, Any] = {"somatic_ip": somatic_ip}
    start = time.perf_counter()

    try:
        status: dict[str, Any] = await asyncio.to_thread(
            _run_tailscale_status, timeout
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # Local node
        self_info: dict[str, Any] = status.get("Self", {})
        self_hostname: str = self_info.get("HostName", "unknown")
        self_online: bool = bool(self_info.get("Online", False))
        details["self_hostname"] = self_hostname
        details["self_online"] = self_online

        if not self_online:
            return ServiceResult(
                service_name="tailscale",
                status=ServiceStatus.UNHEALTHY,
                latency_ms=latency_ms,
                details=details,
                error="Local Tailscale node is offline",
            )

        # Somatic peer lookup by IP
        peers: dict[str, Any] = status.get("Peer", {})
        details["peer_count"] = len(peers)

        somatic_peer: dict[str, Any] | None = None
        for peer in peers.values():
            if somatic_ip in peer.get("TailscaleIPs", []):
                somatic_peer = peer
                break

        if somatic_peer is None:
            details["somatic_online"] = False
            return ServiceResult(
                service_name="tailscale",
                status=ServiceStatus.DEGRADED,
                latency_ms=latency_ms,
                details=details,
                error=(
                    f"Somatic node ({somatic_ip}) not found in Tailscale mesh "
                    "-- is the Somatic machine connected?"
                ),
            )

        somatic_online: bool = bool(somatic_peer.get("Online", False))
        somatic_hostname: str = somatic_peer.get("HostName", "unknown")
        details["somatic_online"] = somatic_online
        details["somatic_hostname"] = somatic_hostname

        if not somatic_online:
            return ServiceResult(
                service_name="tailscale",
                status=ServiceStatus.DEGRADED,
                latency_ms=latency_ms,
                details=details,
                error=(
                    f"Somatic node ({somatic_hostname} / {somatic_ip}) "
                    "is offline in Tailscale"
                ),
            )

        return ServiceResult(
            service_name="tailscale",
            status=ServiceStatus.HEALTHY,
            latency_ms=latency_ms,
            details=details,
        )

    except FileNotFoundError:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("tailscale binary not found")
        return ServiceResult(
            service_name="tailscale",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error="Tailscale is not installed",
        )
    except subprocess.TimeoutExpired:
        latency_ms = (time.perf_counter() - start) * 1000
        return ServiceResult(
            service_name="tailscale",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error=f"tailscale status timed out after {timeout}s",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("tailscale check failed: %s", exc)
        return ServiceResult(
            service_name="tailscale",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error=str(exc),
        )
