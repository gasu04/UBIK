#!/usr/bin/env python3
"""
Maestro — Tailscale Connectivity Module

Provides two independent layers of Tailscale verification:

    Layer 1 — CLI status (best-effort)
        ``tailscale status --json`` reports daemon state and peer table.
        This layer is UNRELIABLE: on macOS the CLI sometimes reports
        "Logged out" or crashes (BundleIdentifiers.swift) even when the
        daemon is running and routing correctly.

    Layer 2 — Direct TCP probe (authoritative)
        An ``asyncio.open_connection`` attempt to a known port on the peer
        IP proves end-to-end reachability independent of the CLI.  This is
        the only reliable signal when the CLI is broken.

Public API:
    TailscaleStatus            — result dataclass
    check_tailscale_status()   — runs both layers, returns TailscaleStatus
    test_peer_connectivity()   — tests a list of ports, returns {port: bool}

Usage:
    import asyncio
    from maestro.tailscale import check_tailscale_status, test_peer_connectivity

    status = asyncio.run(check_tailscale_status("100.79.166.114"))
    print(status.peer_reachable)   # True if TCP probe succeeded

    ports = asyncio.run(test_peer_connectivity("100.79.166.114", [22, 8002]))
    # {22: True, 8002: True}

Notes:
    • All CLI calls run in a thread via asyncio.to_thread to avoid blocking.
    • Subprocess timeout is capped at 10 s; TCP probe timeout defaults to 5 s.
    • No call ever raises — errors are captured in TailscaleStatus.cli_error.
    • macOS app bundle path is tried before falling back to PATH.

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
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TAILSCALE_MACOS_APP = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
_CLI_TIMEOUT_S = 10.0   # max seconds for any subprocess call
_TCP_PROBE_PORT = 8080  # default port for the TCP reachability probe


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class TailscaleStatus:
    """Result of a combined CLI + TCP Tailscale health check.

    Attributes:
        daemon_running: ``True`` when ``tailscale status --json`` reports
            ``BackendState == "Running"``.  ``False`` on CLI failure.
        cli_available: ``True`` when the Tailscale binary was found and
            produced parseable output.
        cli_error: Human-readable error from the CLI layer, or ``None``
            when the CLI succeeded.
        local_ip: This node's IPv4 Tailscale address, or ``None`` when
            both ``status --json`` and ``ip -4`` failed.
        peer_reachable: ``True`` when a direct TCP connection to
            *peer_ip*:*probe_port* succeeded.  This is the authoritative
            reachability signal — it works even when the CLI is broken.
        peer_ip: The IP address that was probed.
        peer_latency_ms: Round-trip latency of the TCP probe in
            milliseconds, or ``None`` when the probe failed.
    """

    daemon_running: bool
    cli_available: bool
    cli_error: Optional[str]
    local_ip: Optional[str]
    peer_reachable: bool
    peer_ip: Optional[str]
    peer_latency_ms: Optional[float]


# ---------------------------------------------------------------------------
# Private helpers — synchronous (called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _find_tailscale_binary() -> str:
    """Return the path to the Tailscale CLI binary.

    Prefers the macOS app-bundle path; falls back to ``"tailscale"`` on
    PATH for Linux / non-bundle macOS installs.
    """
    if platform.system() == "Darwin" and os.path.exists(_TAILSCALE_MACOS_APP):
        return _TAILSCALE_MACOS_APP
    return "tailscale"


def _run_cli_status(timeout: float = _CLI_TIMEOUT_S) -> dict[str, Any]:
    """Run ``tailscale status --json`` and return the parsed dict.

    Args:
        timeout: Subprocess timeout in seconds.

    Returns:
        Parsed JSON status dictionary.

    Raises:
        FileNotFoundError: Binary not found.
        subprocess.TimeoutExpired: CLI hung.
        subprocess.CalledProcessError: Non-zero exit.
        json.JSONDecodeError: Output not valid JSON.
    """
    cmd = _find_tailscale_binary()
    proc = subprocess.run(
        [cmd, "status", "--json"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return json.loads(proc.stdout)


def _run_cli_ip(timeout: float = _CLI_TIMEOUT_S) -> str:
    """Run ``tailscale ip -4`` and return the IPv4 address string.

    Args:
        timeout: Subprocess timeout in seconds.

    Returns:
        IPv4 Tailscale address (stripped of whitespace).

    Raises:
        FileNotFoundError: Binary not found.
        subprocess.TimeoutExpired: CLI hung.
        subprocess.CalledProcessError: Non-zero exit.
        ValueError: Output does not look like an IP address.
    """
    cmd = _find_tailscale_binary()
    proc = subprocess.run(
        [cmd, "ip", "-4"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    ip = proc.stdout.strip()
    if not ip:
        raise ValueError("tailscale ip -4 returned empty output")
    return ip


def _run_cli_ping(peer_ip: str, timeout: float = _CLI_TIMEOUT_S) -> float:
    """Run ``tailscale ping`` and return latency in milliseconds.

    Uses a single ping packet (``--c 1``) so the call is bounded.

    Args:
        peer_ip: Tailscale IP address of the peer to ping.
        timeout: Subprocess timeout in seconds.

    Returns:
        Latency in milliseconds parsed from the ping output.

    Raises:
        FileNotFoundError: Binary not found.
        subprocess.TimeoutExpired: Ping timed out at OS level.
        subprocess.CalledProcessError: Non-zero exit.
        ValueError: Could not parse latency from output.
    """
    cmd = _find_tailscale_binary()
    proc = subprocess.run(
        [cmd, "ping", "--c", "1", "--timeout", "5s", peer_ip],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    # Output contains a line like: "pong from ... in 12.34ms"
    for line in proc.stdout.splitlines():
        line_lower = line.lower()
        if "ms" in line_lower and "pong" in line_lower:
            # Extract numeric value before "ms"
            parts = line_lower.split()
            for part in reversed(parts):
                part = part.rstrip("ms").rstrip(".")
                try:
                    return float(part)
                except ValueError:
                    continue
    raise ValueError(f"Could not parse latency from ping output: {proc.stdout!r}")


# ---------------------------------------------------------------------------
# Private helpers — async
# ---------------------------------------------------------------------------

async def _cli_get_status(timeout: float = _CLI_TIMEOUT_S) -> dict[str, Any]:
    """Async wrapper: run _run_cli_status in a thread."""
    return await asyncio.to_thread(_run_cli_status, timeout)


async def _cli_get_ip(timeout: float = _CLI_TIMEOUT_S) -> str:
    """Async wrapper: run _run_cli_ip in a thread."""
    return await asyncio.to_thread(_run_cli_ip, timeout)


async def _cli_ping(peer_ip: str, timeout: float = _CLI_TIMEOUT_S) -> float:
    """Async wrapper: run _run_cli_ping in a thread."""
    return await asyncio.to_thread(_run_cli_ping, peer_ip, timeout)


async def _tcp_probe(
    ip: str,
    port: int,
    *,
    timeout: float,
) -> tuple[bool, Optional[float]]:
    """Attempt a TCP connection and return (reachable, latency_ms).

    Args:
        ip: Target IP address.
        port: Target TCP port.
        timeout: Connection timeout in seconds.

    Returns:
        ``(True, latency_ms)`` on success; ``(False, None)`` on failure.
    """
    start = time.perf_counter()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port),
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True, round(latency_ms, 2)
    except Exception:
        return False, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def check_tailscale_status(
    peer_ip: str,
    *,
    probe_port: int = _TCP_PROBE_PORT,
    cli_timeout: float = _CLI_TIMEOUT_S,
    tcp_timeout: float = 5.0,
) -> TailscaleStatus:
    """Run a two-layer Tailscale health check.

    Executes CLI probes (best-effort) and a direct TCP reachability test
    (authoritative) concurrently where possible.

    Layer 1 — CLI:
        a. ``tailscale status --json`` to determine daemon state and local IP.
        b. ``tailscale ip -4`` as a fallback if status JSON lacks the IP.
        c. ``tailscale ping <peer> --c 1`` to measure Tailscale-level latency
           (optional; failure does not affect result).

    Layer 2 — TCP probe:
        d. ``asyncio.open_connection(peer_ip, probe_port)`` with *tcp_timeout*.
           A successful connection proves end-to-end reachability regardless
           of CLI state.

    Args:
        peer_ip: Tailscale IP of the remote node to test.
        probe_port: TCP port to probe on the remote node.  Defaults to 8080
            (MCP port, always open on the Hippocampal node).
        cli_timeout: Subprocess timeout for CLI calls (seconds).
        tcp_timeout: Connection timeout for the TCP probe (seconds).

    Returns:
        :class:`TailscaleStatus` with all available diagnostic fields
        populated.  Never raises.
    """
    daemon_running = False
    cli_available = False
    cli_error: Optional[str] = None
    local_ip: Optional[str] = None

    # ── Layer 1a: tailscale status --json ──────────────────────────────────
    try:
        status_json = await _cli_get_status(cli_timeout)
        cli_available = True
        daemon_running = status_json.get("BackendState") == "Running"

        # Extract local IPv4 address from Self.TailscaleIPs
        self_ips: list[str] = status_json.get("Self", {}).get("TailscaleIPs", [])
        local_ip = next((ip for ip in self_ips if "." in ip), None)

        logger.debug(
            "tailscale: daemon=%s local_ip=%s peers=%d",
            daemon_running,
            local_ip,
            len(status_json.get("Peer", {})),
        )
    except FileNotFoundError:
        cli_error = "Tailscale binary not found"
        logger.warning("tailscale: binary not found")
    except subprocess.TimeoutExpired:
        cli_error = f"tailscale status timed out after {cli_timeout:.0f}s"
        logger.warning("tailscale: status timed out")
    except Exception as exc:
        cli_error = str(exc)
        logger.warning("tailscale: status --json failed: %s", exc)

    # ── Layer 1b: tailscale ip -4 (fallback for local IP) ─────────────────
    if local_ip is None:
        try:
            local_ip = await _cli_get_ip(cli_timeout)
            logger.debug("tailscale: local_ip from ip -4: %s", local_ip)
        except Exception as exc:
            logger.warning("tailscale: ip -4 failed: %s", exc)

    # ── Layer 2: direct TCP probe (authoritative reachability) ────────────
    peer_reachable, peer_latency_ms = await _tcp_probe(
        peer_ip, probe_port, timeout=tcp_timeout
    )
    logger.debug(
        "tailscale: TCP probe %s:%d -> reachable=%s latency=%.1fms",
        peer_ip,
        probe_port,
        peer_reachable,
        peer_latency_ms or 0.0,
    )

    # ── Layer 1c: tailscale ping (optional — best-effort latency) ─────────
    # Only attempt if TCP probe succeeded and CLI is available, to avoid
    # adding extra latency on a broken mesh.
    if peer_reachable and cli_available and peer_latency_ms is not None:
        try:
            ping_ms = await _cli_ping(peer_ip, cli_timeout)
            # Use Tailscale-level latency if ping succeeded (more accurate
            # than TCP handshake for the tunnel overhead).
            peer_latency_ms = round(ping_ms, 2)
            logger.debug("tailscale: ping latency %.1fms", ping_ms)
        except Exception as exc:
            logger.debug("tailscale: ping skipped: %s", exc)

    return TailscaleStatus(
        daemon_running=daemon_running,
        cli_available=cli_available,
        cli_error=cli_error,
        local_ip=local_ip,
        peer_reachable=peer_reachable,
        peer_ip=peer_ip,
        peer_latency_ms=peer_latency_ms,
    )


async def test_peer_connectivity(
    peer_ip: str,
    ports: list[int],
    *,
    timeout: float = 3.0,
) -> dict[int, bool]:
    """Test TCP connectivity to a list of ports on a peer node.

    All ports are probed concurrently via :func:`asyncio.gather` so the
    total wall-clock time is bounded by *timeout*, not ``timeout × len(ports)``.

    Args:
        peer_ip: IP address of the peer to probe.
        ports: List of TCP port numbers to test.
        timeout: Per-connection timeout in seconds.

    Returns:
        Dict mapping each port number to a boolean — ``True`` when a TCP
        connection was established, ``False`` on timeout or refusal.

    Example:
        >>> results = await test_peer_connectivity("100.79.166.114", [22, 8002])
        >>> results
        {22: True, 8002: True}
    """
    if not ports:
        return {}

    probes = [_tcp_probe(peer_ip, port, timeout=timeout) for port in ports]
    results = await asyncio.gather(*probes)
    return {port: reachable for port, (reachable, _) in zip(ports, results)}


# Prevent pytest from collecting this public function as a test case.
test_peer_connectivity.__test__ = False  # type: ignore[attr-defined]
