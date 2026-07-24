"""
Tests for the persistent vLLM systemd unit (Layer A — replace transient
``systemd-run`` with an installed, enabled, self-healing unit).

Covers:
    TestRenderVllmUnit    the pure unit-file renderer (_render_vllm_unit)
    TestRemoteStart       _remote_start install-then-start behaviour

The renderer is asserted to source every value from config (no hardcoded
node-specific literals, §2.1) and to carry the self-healing / VRAM-safe
directives.  The _remote_start tests assert the generated bash installs the
unit idempotently (sha256 change detection), enables + restarts it, and no
longer uses the transient ``systemd-run`` path — and that hard failures
(missing prerequisite, SSH down) short-circuit before the health wait.
"""

from unittest.mock import AsyncMock

import pytest

from maestro.remote import RemoteResult
from maestro.services.vllm_service import (
    _BLACKWELL_ENV,
    _VLLM_UNIT,
    VllmService,
    _render_vllm_unit,
)


# ---------------------------------------------------------------------------
# _render_vllm_unit
# ---------------------------------------------------------------------------

class TestRenderVllmUnit:
    def _unit(self) -> str:
        return _render_vllm_unit(
            python="/opt/customvenv/bin/python",
            server="/srv/ubik/somatic/inference/vllm_server.py",
            config="/srv/ubik/config/models/vllm_config.yaml",
            model="/srv/models/deepseek-awq",
            port=8002,
            stop_grace_s=90,
        )

    def test_render_uses_config_values(self):
        """Model, port, venv python, server and config all come from args."""
        unit = self._unit()
        assert "/opt/customvenv/bin/python" in unit
        assert "/srv/ubik/somatic/inference/vllm_server.py" in unit
        assert "--config /srv/ubik/config/models/vllm_config.yaml" in unit
        assert "--model /srv/models/deepseek-awq" in unit
        assert "--port 8002" in unit

    def test_render_no_hardcoded_home_gasu(self):
        """The renderer injects only what it is given — no baked-in literals."""
        unit = self._unit()
        assert "/home/gasu" not in unit
        assert "pytorch_env" not in unit

    def test_render_includes_all_blackwell_env(self):
        """All four SM120 env vars are baked in as Environment= lines."""
        unit = self._unit()
        for key, val in _BLACKWELL_ENV.items():
            assert f"Environment={key}={val}" in unit
        # The FlashInfer sampler must be explicitly disabled for vLLM 0.24/SM120.
        assert "Environment=VLLM_USE_FLASHINFER_SAMPLER=0" in unit

    def test_render_self_healing_and_vram_safe_directives(self):
        """Restart policy, start-limit cap, and stop grace are present."""
        unit = self._unit()
        assert "Restart=on-failure" in unit
        assert "RestartSec=10" in unit
        assert "StartLimitBurst=4" in unit
        assert "TimeoutStopSec=90" in unit
        assert "KillSignal=SIGTERM" in unit
        assert "KillMode=mixed" in unit
        assert "WantedBy=default.target" in unit

    def test_render_ends_with_newline(self):
        """Trailing newline so the heredoc delimiter lands on its own line."""
        assert self._unit().endswith("\n")


# ---------------------------------------------------------------------------
# _remote_start
# ---------------------------------------------------------------------------

def _remote_vllm(monkeypatch, remote_result: RemoteResult) -> tuple[VllmService, AsyncMock]:
    """Build a remote VllmService whose SSH call returns *remote_result*.

    Returns the service and the ``remote.run`` AsyncMock so callers can inspect
    the generated script.  ``_wait_for_healthy`` is stubbed to return True so
    tests exercise only the install/start path, not real polling.
    """
    remote = AsyncMock()
    remote.ssh_host = "windows-server"
    remote.run = AsyncMock(return_value=remote_result)
    svc = VllmService(
        model_path="/srv/models/deepseek-awq",
        remote=remote,
        remote_ubik_root="/srv/ubik",
        probe_ip="127.0.0.1",
        remote_venv="/opt/customvenv",
    )
    monkeypatch.setattr(svc, "_is_remote", lambda: True)
    monkeypatch.setattr(svc, "_wait_for_healthy", AsyncMock(return_value=True))
    return svc, remote


class TestRemoteStart:
    @pytest.mark.asyncio
    async def test_generated_script_installs_enables_restarts(self, monkeypatch):
        """The install script contains the persistent-unit mechanism."""
        svc, remote = _remote_vllm(
            monkeypatch,
            RemoteResult(returncode=0, stdout="UNIT_INSTALLED\nSTARTED_UNIT=ubik-vllm rc=0",
                         stderr="", connected=True),
        )
        ok = await svc._remote_start()
        assert ok is True

        script = remote.run.call_args.args[0]
        # Persistent-unit install mechanism
        assert "$HOME/.config/systemd/user" in script
        assert f"{_VLLM_UNIT}.service" in script
        assert "sha256sum" in script                       # idempotent change detection
        assert "systemctl --user daemon-reload" in script
        assert f"systemctl --user enable {_VLLM_UNIT}" in script
        assert f"systemctl --user restart {_VLLM_UNIT}" in script
        # The transient path must be gone.
        assert "systemd-run" not in script
        # Rendered unit content travels in the heredoc.
        assert "Restart=on-failure" in script
        assert "Environment=VLLM_USE_FLASHINFER_SAMPLER=0" in script

    @pytest.mark.asyncio
    async def test_missing_prerequisite_short_circuits(self, monkeypatch):
        """A MISSING_ prerequisite returns False without waiting for health."""
        svc, remote = _remote_vllm(
            monkeypatch,
            RemoteResult(returncode=3, stdout="MISSING_PYTHON:/opt/customvenv/bin/python",
                         stderr="", connected=True),
        )
        ok = await svc._remote_start()
        assert ok is False
        svc._wait_for_healthy.assert_not_called()

    @pytest.mark.asyncio
    async def test_ssh_down_short_circuits(self, monkeypatch):
        """A disconnected SSH result returns False without waiting for health."""
        svc, remote = _remote_vllm(
            monkeypatch,
            RemoteResult(returncode=255, stdout="", stderr="conn refused",
                         connected=False),
        )
        ok = await svc._remote_start()
        assert ok is False
        svc._wait_for_healthy.assert_not_called()

    @pytest.mark.asyncio
    async def test_healthy_path_waits_and_returns_wait_result(self, monkeypatch):
        """On a clean launch, the result is whatever _wait_for_healthy returns."""
        svc, remote = _remote_vllm(
            monkeypatch,
            RemoteResult(returncode=0, stdout="UNIT_UNCHANGED\nSTARTED_UNIT=ubik-vllm rc=0",
                         stderr="", connected=True),
        )
        svc._wait_for_healthy = AsyncMock(return_value=False)
        ok = await svc._remote_start()
        assert ok is False
        svc._wait_for_healthy.assert_awaited_once()
