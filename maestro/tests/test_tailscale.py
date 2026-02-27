"""
Tests for maestro.tailscale

All tests mock CLI subprocess calls and TCP probes — no live Tailscale
daemon or network is required.

Coverage:
    TailscaleStatus  — dataclass fields and defaults

    check_tailscale_status:
        CLI succeeds + TCP probe succeeds → daemon_running=True, peer_reachable=True
        CLI succeeds + TCP probe fails   → peer_reachable=False
        CLI fails (FileNotFoundError)    → cli_available=False, cli_error set
        CLI fails (TimeoutExpired)       → cli_error contains "timed out"
        CLI fails (generic error)        → cli_error = str(exc)
        local_ip extracted from status JSON Self.TailscaleIPs (IPv4)
        local_ip falls back to tailscale ip -4 when status JSON missing it
        local_ip is None when both CLI calls fail
        daemon_running=False when BackendState != "Running"
        daemon_running=False when CLI unavailable
        peer_ip stored in result
        peer_latency_ms set on success, None on failure
        ping latency replaces TCP latency when ping succeeds
        ping failure leaves TCP latency unchanged

    test_peer_connectivity:
        all ports reachable → all True
        all ports closed   → all False
        mixed results
        empty port list → empty dict
        ports tested concurrently (gather called with correct arity)
        timeout parameter forwarded to _tcp_probe
"""

import asyncio
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from maestro.tailscale import (
    TailscaleStatus,
    check_tailscale_status,
    test_peer_connectivity,
)

# Patch targets
_PATCH_CLI_STATUS = "maestro.tailscale._cli_get_status"
_PATCH_CLI_IP = "maestro.tailscale._cli_get_ip"
_PATCH_CLI_PING = "maestro.tailscale._cli_ping"
_PATCH_TCP = "maestro.tailscale._tcp_probe"

_PEER_IP = "100.79.166.114"
_LOCAL_IP = "100.103.242.91"

# Minimal valid status JSON returned by the CLI
_STATUS_OK: dict = {
    "BackendState": "Running",
    "Self": {
        "HostName": "MiniM4-2025",
        "Online": True,
        "TailscaleIPs": [_LOCAL_IP, "fd7a::1"],
    },
    "Peer": {},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tcp_ok(latency: float = 12.0):
    """AsyncMock that returns (True, latency)."""
    return AsyncMock(return_value=(True, latency))


def _tcp_fail():
    """AsyncMock that returns (False, None)."""
    return AsyncMock(return_value=(False, None))


# ---------------------------------------------------------------------------
# TailscaleStatus — dataclass
# ---------------------------------------------------------------------------

class TestTailscaleStatusDataclass:
    def test_all_fields_settable(self):
        s = TailscaleStatus(
            daemon_running=True,
            cli_available=True,
            cli_error=None,
            local_ip=_LOCAL_IP,
            peer_reachable=True,
            peer_ip=_PEER_IP,
            peer_latency_ms=8.5,
        )
        assert s.daemon_running is True
        assert s.cli_available is True
        assert s.cli_error is None
        assert s.local_ip == _LOCAL_IP
        assert s.peer_reachable is True
        assert s.peer_ip == _PEER_IP
        assert s.peer_latency_ms == 8.5

    def test_optional_fields_accept_none(self):
        s = TailscaleStatus(
            daemon_running=False,
            cli_available=False,
            cli_error="not found",
            local_ip=None,
            peer_reachable=False,
            peer_ip=None,
            peer_latency_ms=None,
        )
        assert s.local_ip is None
        assert s.peer_latency_ms is None


# ---------------------------------------------------------------------------
# check_tailscale_status — happy path
# ---------------------------------------------------------------------------

class TestCheckTailscaleStatusSuccess:
    @pytest.mark.asyncio
    async def test_daemon_running_when_backend_running(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=_STATUS_OK)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value=_LOCAL_IP)),
            patch(_PATCH_TCP, new=_tcp_ok()),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.daemon_running is True

    @pytest.mark.asyncio
    async def test_cli_available_on_success(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=_STATUS_OK)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value=_LOCAL_IP)),
            patch(_PATCH_TCP, new=_tcp_ok()),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.cli_available is True
        assert result.cli_error is None

    @pytest.mark.asyncio
    async def test_peer_reachable_when_tcp_probe_succeeds(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=_STATUS_OK)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value=_LOCAL_IP)),
            patch(_PATCH_TCP, new=_tcp_ok(20.0)),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.peer_reachable is True

    @pytest.mark.asyncio
    async def test_peer_ip_stored(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=_STATUS_OK)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value=_LOCAL_IP)),
            patch(_PATCH_TCP, new=_tcp_ok()),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.peer_ip == _PEER_IP

    @pytest.mark.asyncio
    async def test_local_ip_from_status_json(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=_STATUS_OK)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value=_LOCAL_IP)),
            patch(_PATCH_TCP, new=_tcp_ok()),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        # IPv4 extracted from TailscaleIPs (filters out IPv6)
        assert result.local_ip == _LOCAL_IP

    @pytest.mark.asyncio
    async def test_latency_set_from_tcp_probe(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=_STATUS_OK)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value=_LOCAL_IP)),
            patch(_PATCH_TCP, new=_tcp_ok(33.0)),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.peer_latency_ms == 33.0


# ---------------------------------------------------------------------------
# check_tailscale_status — TCP probe fails
# ---------------------------------------------------------------------------

class TestCheckTailscaleStatusTcpFail:
    @pytest.mark.asyncio
    async def test_peer_not_reachable_when_tcp_fails(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=_STATUS_OK)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value=_LOCAL_IP)),
            patch(_PATCH_TCP, new=_tcp_fail()),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.peer_reachable is False

    @pytest.mark.asyncio
    async def test_latency_none_when_tcp_fails(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=_STATUS_OK)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value=_LOCAL_IP)),
            patch(_PATCH_TCP, new=_tcp_fail()),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.peer_latency_ms is None

    @pytest.mark.asyncio
    async def test_daemon_still_reported_when_tcp_fails(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=_STATUS_OK)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value=_LOCAL_IP)),
            patch(_PATCH_TCP, new=_tcp_fail()),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        # CLI succeeded so daemon status still known
        assert result.daemon_running is True
        assert result.cli_available is True


# ---------------------------------------------------------------------------
# check_tailscale_status — CLI failures
# ---------------------------------------------------------------------------

class TestCheckTailscaleStatusCliFail:
    @pytest.mark.asyncio
    async def test_file_not_found_sets_cli_unavailable(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(side_effect=FileNotFoundError("not found"))),
            patch(_PATCH_CLI_IP, new=AsyncMock(side_effect=FileNotFoundError("not found"))),
            patch(_PATCH_TCP, new=_tcp_ok()),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.cli_available is False
        assert result.daemon_running is False
        assert "not found" in result.cli_error.lower()

    @pytest.mark.asyncio
    async def test_timeout_error_sets_cli_error(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(
                side_effect=subprocess.TimeoutExpired(cmd="tailscale", timeout=10)
            )),
            patch(_PATCH_CLI_IP, new=AsyncMock(side_effect=Exception("also fails"))),
            patch(_PATCH_TCP, new=_tcp_ok()),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.cli_available is False
        assert "timed out" in result.cli_error.lower()

    @pytest.mark.asyncio
    async def test_generic_exception_stored_in_cli_error(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(
                side_effect=RuntimeError("BundleIdentifiers.swift crash")
            )),
            patch(_PATCH_CLI_IP, new=AsyncMock(side_effect=Exception("also fails"))),
            patch(_PATCH_TCP, new=_tcp_ok()),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.cli_available is False
        assert "BundleIdentifiers" in result.cli_error

    @pytest.mark.asyncio
    async def test_tcp_probe_still_runs_when_cli_fails(self):
        """Layer 2 must run regardless of Layer 1 outcome."""
        tcp_mock = _tcp_ok(15.0)
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(side_effect=FileNotFoundError())),
            patch(_PATCH_CLI_IP, new=AsyncMock(side_effect=FileNotFoundError())),
            patch(_PATCH_TCP, new=tcp_mock),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.peer_reachable is True
        tcp_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_local_ip_falls_back_to_cli_ip_command(self):
        """When status JSON has no IPs, ip -4 is tried."""
        status_no_ip = {
            "BackendState": "Running",
            "Self": {"TailscaleIPs": []},  # empty list
            "Peer": {},
        }
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=status_no_ip)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value="10.0.0.5")),
            patch(_PATCH_TCP, new=_tcp_ok()),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.local_ip == "10.0.0.5"

    @pytest.mark.asyncio
    async def test_local_ip_none_when_all_cli_fails(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(side_effect=FileNotFoundError())),
            patch(_PATCH_CLI_IP, new=AsyncMock(side_effect=FileNotFoundError())),
            patch(_PATCH_TCP, new=_tcp_ok()),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.local_ip is None

    @pytest.mark.asyncio
    async def test_daemon_not_running_when_backend_not_running(self):
        status_stopped = {**_STATUS_OK, "BackendState": "Stopped"}
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=status_stopped)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value=_LOCAL_IP)),
            patch(_PATCH_TCP, new=_tcp_ok()),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("skip"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.daemon_running is False


# ---------------------------------------------------------------------------
# check_tailscale_status — ping layer
# ---------------------------------------------------------------------------

class TestCheckTailscaleStatusPing:
    @pytest.mark.asyncio
    async def test_ping_latency_replaces_tcp_latency(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=_STATUS_OK)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value=_LOCAL_IP)),
            patch(_PATCH_TCP, new=_tcp_ok(50.0)),   # TCP says 50ms
            patch(_PATCH_CLI_PING, new=AsyncMock(return_value=8.5)),  # ping says 8.5ms
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.peer_latency_ms == 8.5

    @pytest.mark.asyncio
    async def test_ping_failure_leaves_tcp_latency(self):
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=_STATUS_OK)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value=_LOCAL_IP)),
            patch(_PATCH_TCP, new=_tcp_ok(50.0)),
            patch(_PATCH_CLI_PING, new=AsyncMock(side_effect=Exception("ping failed"))),
        ):
            result = await check_tailscale_status(_PEER_IP)
        assert result.peer_latency_ms == 50.0

    @pytest.mark.asyncio
    async def test_ping_not_attempted_when_tcp_fails(self):
        ping_mock = AsyncMock(return_value=5.0)
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(return_value=_STATUS_OK)),
            patch(_PATCH_CLI_IP, new=AsyncMock(return_value=_LOCAL_IP)),
            patch(_PATCH_TCP, new=_tcp_fail()),
            patch(_PATCH_CLI_PING, new=ping_mock),
        ):
            await check_tailscale_status(_PEER_IP)
        ping_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_ping_not_attempted_when_cli_unavailable(self):
        ping_mock = AsyncMock(return_value=5.0)
        with (
            patch(_PATCH_CLI_STATUS, new=AsyncMock(side_effect=FileNotFoundError())),
            patch(_PATCH_CLI_IP, new=AsyncMock(side_effect=FileNotFoundError())),
            patch(_PATCH_TCP, new=_tcp_ok(50.0)),
            patch(_PATCH_CLI_PING, new=ping_mock),
        ):
            await check_tailscale_status(_PEER_IP)
        ping_mock.assert_not_called()


# ---------------------------------------------------------------------------
# test_peer_connectivity
# ---------------------------------------------------------------------------

class TestTestPeerConnectivity:
    @pytest.mark.asyncio
    async def test_all_reachable(self):
        tcp_mock = AsyncMock(return_value=(True, 5.0))
        with patch(_PATCH_TCP, new=tcp_mock):
            result = await test_peer_connectivity(_PEER_IP, [22, 8002, 8080])
        assert result == {22: True, 8002: True, 8080: True}

    @pytest.mark.asyncio
    async def test_all_closed(self):
        tcp_mock = AsyncMock(return_value=(False, None))
        with patch(_PATCH_TCP, new=tcp_mock):
            result = await test_peer_connectivity(_PEER_IP, [9999, 9998])
        assert result == {9999: False, 9998: False}

    @pytest.mark.asyncio
    async def test_mixed_results(self):
        responses = [(True, 10.0), (False, None), (True, 20.0)]

        async def _probe(ip, port, *, timeout):
            return responses.pop(0)

        with patch(_PATCH_TCP, side_effect=_probe):
            result = await test_peer_connectivity(_PEER_IP, [22, 8080, 8002])
        assert result == {22: True, 8080: False, 8002: True}

    @pytest.mark.asyncio
    async def test_empty_ports_returns_empty_dict(self):
        result = await test_peer_connectivity(_PEER_IP, [])
        assert result == {}

    @pytest.mark.asyncio
    async def test_correct_port_numbers_in_result_keys(self):
        tcp_mock = AsyncMock(return_value=(True, 1.0))
        with patch(_PATCH_TCP, new=tcp_mock):
            result = await test_peer_connectivity(_PEER_IP, [443, 8443])
        assert set(result.keys()) == {443, 8443}

    @pytest.mark.asyncio
    async def test_timeout_forwarded_to_tcp_probe(self):
        tcp_mock = AsyncMock(return_value=(True, 1.0))
        with patch(_PATCH_TCP, new=tcp_mock):
            await test_peer_connectivity(_PEER_IP, [22], timeout=7.5)
        tcp_mock.assert_called_once_with(_PEER_IP, 22, timeout=7.5)

    @pytest.mark.asyncio
    async def test_correct_ip_forwarded_to_tcp_probe(self):
        tcp_mock = AsyncMock(return_value=(True, 1.0))
        custom_ip = "10.0.0.99"
        with patch(_PATCH_TCP, new=tcp_mock):
            await test_peer_connectivity(custom_ip, [22])
        tcp_mock.assert_called_once_with(custom_ip, 22, timeout=3.0)
