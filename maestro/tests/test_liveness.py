"""
Tests for the out-of-band liveness hook (Layer C — "give maestro eyes").

Covers:
    TestWaitForHealthyLiveness   base `_wait_for_healthy` abort-on-death logic
    TestVllmLivenessDiagnostic   VllmService remote override behaviour

The goal of the feature under test is to turn a silent full-``max_wait_s``
timeout (300 s for vLLM) into an instant, actionable error whenever the
underlying process/unit is confirmed dead — while never false-alarming when the
process is merely still starting up or the liveness signal is unavailable.
"""

import logging
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from maestro.platform_detect import NodeType
from maestro.services.base import ProbeResult, UbikService
from maestro.remote import RemoteResult
from maestro.services.vllm_service import VllmService


# ---------------------------------------------------------------------------
# Test double: a minimal UbikService with scripted probe + liveness results
# ---------------------------------------------------------------------------

class _ScriptedService(UbikService):
    """UbikService whose probe/liveness results are supplied as sequences.

    Args:
        probe_healthy: Sequence of ``bool`` returned by successive probes.
            Exhaustion repeats the last value (so an all-``False`` service
            keeps looking unhealthy).
        liveness: Sequence of ``Optional[str]`` returned by successive
            ``_liveness_diagnostic`` calls.  Exhaustion repeats the last value.
        max_wait: Value for :attr:`max_wait_s`.
    """

    def __init__(self, *, probe_healthy, liveness, max_wait: float = 300.0):
        self._probe_healthy = list(probe_healthy)
        self._liveness = list(liveness)
        self._max_wait = max_wait
        self.probe_calls = 0
        self.liveness_calls = 0

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def node(self) -> NodeType:
        return NodeType.SOMATIC

    @property
    def ports(self) -> list[int]:
        return []

    @property
    def depends_on(self) -> list[str]:
        return []

    @property
    def max_wait_s(self) -> float:
        return self._max_wait

    async def probe(self, host: str) -> ProbeResult:
        idx = min(self.probe_calls, len(self._probe_healthy) - 1)
        healthy = self._probe_healthy[idx]
        self.probe_calls += 1
        return ProbeResult(
            name=self.name, node=self.node, healthy=healthy,
            latency_ms=1.0, error=None if healthy else "refused",
        )

    async def _liveness_diagnostic(self):
        idx = min(self.liveness_calls, len(self._liveness) - 1)
        val = self._liveness[idx]
        self.liveness_calls += 1
        return val

    async def start(self, ubik_root: Path) -> bool:  # pragma: no cover - unused
        return True

    async def stop(self) -> bool:  # pragma: no cover - unused
        return True


# ---------------------------------------------------------------------------
# base._wait_for_healthy
# ---------------------------------------------------------------------------

class TestWaitForHealthyLiveness:
    @pytest.mark.asyncio
    async def test_wait_for_healthy_aborts_on_unit_death(self, caplog):
        """A confirmed-dead process aborts the wait immediately, not at 300 s."""
        svc = _ScriptedService(
            probe_healthy=[False],                 # never healthy
            liveness=[None, "STATE=failed\njournal: Stopping ubik-vllm"],
            max_wait=300.0,                         # would be a 5-min silent wait
        )

        wall_start = time.perf_counter()
        with caplog.at_level(logging.ERROR):
            result = await svc._wait_for_healthy(
                "somatic", poll_interval=0.01, liveness_interval=0.02
            )
        elapsed = time.perf_counter() - wall_start

        assert result is False
        assert elapsed < 5.0, "should abort on unit death, not wait out max_wait_s"
        assert svc.liveness_calls >= 2
        assert any(
            "died during startup wait" in rec.message for rec in caplog.records
        )
        assert any("Stopping ubik-vllm" in rec.getMessage() for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_wait_for_healthy_keeps_waiting_while_activating(self):
        """A ``None`` liveness (still activating) must not abort — poll to healthy."""
        svc = _ScriptedService(
            probe_healthy=[False, False, True],     # healthy on 3rd probe
            liveness=[None],                         # always "alive/unknown"
            max_wait=30.0,
        )

        result = await svc._wait_for_healthy(
            "somatic", poll_interval=0.01, liveness_interval=0.01
        )

        assert result is True
        assert svc.probe_calls >= 3


# ---------------------------------------------------------------------------
# VllmService._liveness_diagnostic (remote override)
# ---------------------------------------------------------------------------

def _remote_vllm(monkeypatch, remote_result: RemoteResult) -> VllmService:
    """Build a remote VllmService whose single SSH call returns *remote_result*."""
    remote = AsyncMock()
    remote.run = AsyncMock(return_value=remote_result)
    svc = VllmService(
        model_path="/tmp/test-model",
        remote=remote,
        remote_ubik_root="/home/gasu/ubik",
        probe_ip="127.0.0.1",
        remote_venv="/home/gasu/pytorch_env_vllm024",
    )
    # Force the remote branch regardless of the node this test runs on.
    monkeypatch.setattr(svc, "_is_remote", lambda: True)
    return svc


class TestVllmLivenessDiagnostic:
    @pytest.mark.asyncio
    async def test_liveness_none_when_ssh_down(self, monkeypatch):
        """SSH unreachable → None (never false-alarm on a transient blip)."""
        svc = _remote_vllm(
            monkeypatch,
            RemoteResult(returncode=255, stdout="", stderr="conn refused",
                         connected=False),
        )
        assert await svc._liveness_diagnostic() is None

    @pytest.mark.asyncio
    async def test_liveness_none_when_unit_active(self, monkeypatch):
        """Unit active/activating (still loading) → None (keep waiting)."""
        svc = _remote_vllm(
            monkeypatch,
            RemoteResult(returncode=0, stdout="STATE=activating\n", stderr="",
                         connected=True),
        )
        assert await svc._liveness_diagnostic() is None

    @pytest.mark.asyncio
    async def test_liveness_returns_journal_when_dead(self, monkeypatch):
        """Unit not active → return the journal tail as the diagnostic."""
        journal = (
            "STATE=failed\n"
            "Jul 23 01:15:02 somatic python[123]: Started ubik-vllm\n"
            "Jul 23 01:15:17 somatic systemd[1]: Stopping ubik-vllm...\n"
        )
        svc = _remote_vllm(
            monkeypatch,
            RemoteResult(returncode=0, stdout=journal, stderr="", connected=True),
        )
        diag = await svc._liveness_diagnostic()
        assert diag is not None
        assert "Stopping ubik-vllm" in diag

    @pytest.mark.asyncio
    async def test_liveness_none_when_not_remote(self, monkeypatch):
        """Local (non-remote) start has no unit to interrogate → None."""
        remote = AsyncMock()
        remote.run = AsyncMock()
        svc = VllmService(model_path="/tmp/test-model", remote=remote)
        monkeypatch.setattr(svc, "_is_remote", lambda: False)
        assert await svc._liveness_diagnostic() is None
        remote.run.assert_not_called()
