#!/usr/bin/env python3
"""
Tests for maestro.remote — the SSH remote-execution layer.

These tests mock ``asyncio.create_subprocess_exec`` so no real SSH is issued;
they verify argv construction, stdin delivery, connection-failure handling,
timeouts, and the connectivity check.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maestro.remote import RemoteExecutor, RemoteResult


# ---------------------------------------------------------------------------
# Fake subprocess helper
# ---------------------------------------------------------------------------

def _fake_proc(stdout=b"", stderr=b"", returncode=0, *, hang=False):
    """Build a mock process compatible with RemoteExecutor.run()."""
    proc = MagicMock()
    if hang:
        async def _never(*_a, **_k):
            await asyncio.sleep(3600)
        proc.communicate = AsyncMock(side_effect=_never)
    else:
        proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=None)
    return proc


# ---------------------------------------------------------------------------
# argv construction
# ---------------------------------------------------------------------------

class TestSshArgv:
    def test_wsl_wraps_bash_s(self):
        rx = RemoteExecutor("windows-server", wsl=True, connect_timeout=8.0)
        argv = rx._ssh_argv()
        assert argv[0] == "ssh"
        assert argv[-2] == "windows-server"
        assert argv[-1] == "wsl bash -s"
        # Overrides that defeat the interactive-only ssh config must be present.
        assert "BatchMode=yes" in argv
        assert "RequestTTY=no" in argv
        assert "RemoteCommand=none" in argv
        assert "ConnectTimeout=8" in argv

    def test_no_wsl_uses_plain_bash(self):
        rx = RemoteExecutor("host", wsl=False)
        assert rx._ssh_argv()[-1] == "bash -s"

    def test_host_property(self):
        assert RemoteExecutor("h").ssh_host == "h"


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

class TestRun:
    @pytest.mark.asyncio
    async def test_success_feeds_script_over_stdin(self):
        rx = RemoteExecutor("windows-server")
        proc = _fake_proc(stdout=b"hello\n", returncode=0)
        with patch(
            "maestro.remote.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            res = await rx.run("echo hello")
        assert isinstance(res, RemoteResult)
        assert res.connected is True
        assert res.returncode == 0
        assert res.stdout == "hello\n"
        # Script delivered over stdin (bytes).
        proc.communicate.assert_awaited_once_with(b"echo hello")

    @pytest.mark.asyncio
    async def test_ssh_connection_failure_returns_disconnected(self):
        rx = RemoteExecutor("windows-server")
        proc = _fake_proc(stderr=b"ssh: connect timed out", returncode=255)
        with patch(
            "maestro.remote.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            res = await rx.run("echo hi")
        assert res.connected is False
        assert res.returncode == 255

    @pytest.mark.asyncio
    async def test_nonzero_but_connected(self):
        rx = RemoteExecutor("windows-server")
        proc = _fake_proc(stdout=b"", stderr=b"boom", returncode=3)
        with patch(
            "maestro.remote.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            res = await rx.run("false")
        assert res.connected is True     # command ran, just failed
        assert res.returncode == 3

    @pytest.mark.asyncio
    async def test_timeout_kills_and_reports_disconnected(self):
        rx = RemoteExecutor("windows-server")
        proc = _fake_proc(hang=True)
        with patch(
            "maestro.remote.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            res = await rx.run("sleep 999", timeout=0.05)
        assert res.connected is False
        assert res.returncode == 124
        proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_spawn_failure_returns_disconnected(self):
        rx = RemoteExecutor("windows-server")
        with patch(
            "maestro.remote.asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=FileNotFoundError("no ssh")),
        ):
            res = await rx.run("echo hi")
        assert res.connected is False
        assert res.returncode == 255


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------

class TestCheck:
    @pytest.mark.asyncio
    async def test_check_true_on_sentinel(self):
        rx = RemoteExecutor("windows-server")
        proc = _fake_proc(stdout=b"MAESTRO_REMOTE_OK\n", returncode=0)
        with patch(
            "maestro.remote.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            assert await rx.check() is True

    @pytest.mark.asyncio
    async def test_check_false_when_disconnected(self):
        rx = RemoteExecutor("windows-server")
        proc = _fake_proc(stderr=b"timeout", returncode=255)
        with patch(
            "maestro.remote.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            assert await rx.check() is False


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_from_config_reads_somatic_fields(self):
        cfg = MagicMock()
        cfg.somatic.ssh_host = "windows-server"
        cfg.somatic.use_wsl = True
        cfg.somatic.ssh_connect_timeout = 12.0
        rx = RemoteExecutor.from_config(cfg)
        assert rx.ssh_host == "windows-server"
        argv = rx._ssh_argv()
        assert argv[-1] == "wsl bash -s"
        assert "ConnectTimeout=12" in argv


# ---------------------------------------------------------------------------
# Somatic services: remote lifecycle branch (vLLM / WhisperX)
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from maestro.platform_detect import NodeIdentity, NodeType  # noqa: E402
from maestro.services.base import ProbeResult  # noqa: E402
from maestro.services.vllm_service import VllmService  # noqa: E402
from maestro.services.whisperx_service import WhisperXService  # noqa: E402


def _hippo():
    return NodeIdentity(
        node_type=NodeType.HIPPOCAMPAL, hostname="mac.lan", platform="darwin",
        ubik_root=Path("/local/ubik"), is_wsl=False, tailscale_ip="10.0.0.1",
        python_venv_path=None, python_activate_cmd=None,
    )


def _somatic():
    return NodeIdentity(
        node_type=NodeType.SOMATIC, hostname="adrian", platform="linux",
        ubik_root=Path("/home/gasu/ubik"), is_wsl=True, tailscale_ip="10.0.0.2",
        python_venv_path=None, python_activate_cmd=None,
    )


def _remote_ok(stdout="OK"):
    rx = MagicMock(spec=RemoteExecutor)
    rx.ssh_host = "windows-server"
    rx.run = AsyncMock(return_value=RemoteResult(0, stdout, "", True))
    return rx


class TestVllmRemoteLifecycle:
    def test_is_remote_true_on_hippocampal(self):
        svc = VllmService(model_path="/m", remote=_remote_ok())
        with patch("maestro.services.vllm_service.detect_node", return_value=_hippo()):
            assert svc._is_remote() is True

    def test_is_remote_false_on_somatic(self):
        svc = VllmService(model_path="/m", remote=_remote_ok())
        with patch("maestro.services.vllm_service.detect_node", return_value=_somatic()):
            assert svc._is_remote() is False

    def test_is_remote_false_without_executor(self):
        svc = VllmService(model_path="/m")
        with patch("maestro.services.vllm_service.detect_node", return_value=_hippo()):
            assert svc._is_remote() is False

    @pytest.mark.asyncio
    async def test_remote_start_drives_vllm_server_wrapper(self):
        rx = _remote_ok("STARTED_PID=123\nLOG=/home/gasu/ubik/logs/inference/x.log")
        svc = VllmService(
            model_path="/m", remote=rx,
            remote_ubik_root="/home/gasu/ubik", probe_ip="10.0.0.2",
        )
        with patch("maestro.services.vllm_service.detect_node", return_value=_hippo()), \
             patch.object(svc, "_wait_for_healthy", new=AsyncMock(return_value=True)):
            ok = await svc.start(Path("/local/ubik"))
        assert ok is True
        script = rx.run.await_args.args[0]
        # Runs the graceful wrapper as a persistent systemd *user* unit.
        assert "systemd-run --user" in script
        assert "ubik-vllm" in script
        assert "vllm_server.py" in script
        assert "--rtx5080" in script
        # Graceful-stop window wired into the unit (VRAM cleanup time).
        assert "TimeoutStopSec=90" in script

    @pytest.mark.asyncio
    async def test_remote_stop_uses_systemctl_graceful(self):
        rx = _remote_ok("STOPPING_UNIT\nSTOP_RC=0\nVRAM_AFTER=1 MiB, 32000 MiB")
        svc = VllmService(model_path="/m", remote=rx, probe_ip="10.0.0.2")
        down = ProbeResult(name="vllm", node=NodeType.SOMATIC, healthy=False, latency_ms=1.0)
        with patch("maestro.services.vllm_service.detect_node", return_value=_hippo()), \
             patch.object(svc, "probe_with_timeout", new=AsyncMock(return_value=down)):
            ok = await svc.stop()
        assert ok is True
        script = rx.run.await_args.args[0]
        # Critical: graceful SIGTERM via systemctl stop (VRAM release), not SIGKILL first.
        assert "systemctl --user stop ubik-vllm" in script
        # nvidia-smi VRAM report for verification.
        assert "nvidia-smi" in script
        # A pkill fallback exists for a vLLM started outside systemd.
        assert "NO_UNIT_FALLBACK" in script

    @pytest.mark.asyncio
    async def test_remote_stop_fails_when_unreachable(self):
        rx = MagicMock(spec=RemoteExecutor)
        rx.ssh_host = "windows-server"
        rx.run = AsyncMock(return_value=RemoteResult(255, "", "no route", False))
        svc = VllmService(model_path="/m", remote=rx, probe_ip="10.0.0.2")
        with patch("maestro.services.vllm_service.detect_node", return_value=_hippo()):
            ok = await svc.stop()
        assert ok is False


class TestWhisperXRemoteLifecycle:
    @pytest.mark.asyncio
    async def test_remote_start_launches_server(self):
        rx = _remote_ok("STARTED_PID=99")
        svc = WhisperXService(remote=rx, remote_ubik_root="/home/gasu/ubik", probe_ip="10.0.0.2")
        with patch("maestro.services.whisperx_service.detect_node", return_value=_hippo()), \
             patch.object(svc, "_wait_for_healthy", new=AsyncMock(return_value=True)):
            ok = await svc.start(Path("/local/ubik"))
        assert ok is True
        script = rx.run.await_args.args[0]
        assert "systemd-run --user" in script
        assert "ubik-whisperx" in script
        assert "whisperx_server.py" in script

    @pytest.mark.asyncio
    async def test_remote_stop_confirms_down(self):
        rx = _remote_ok("STOPPING_UNIT\nSTOP_RC=0\nDONE")
        svc = WhisperXService(remote=rx, probe_ip="10.0.0.2")
        down = ProbeResult(name="whisperx", node=NodeType.SOMATIC, healthy=False, latency_ms=1.0)
        with patch("maestro.services.whisperx_service.detect_node", return_value=_hippo()), \
             patch.object(svc, "probe_with_timeout", new=AsyncMock(return_value=down)):
            ok = await svc.stop()
        assert ok is True
        assert "systemctl --user stop ubik-whisperx" in rx.run.await_args.args[0]
