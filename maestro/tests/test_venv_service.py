#!/usr/bin/env python3
"""
Tests for maestro/services/venv_service.py

All subprocess calls and path existence checks are mocked.
No network, Docker, or real conda/venv required.
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maestro.platform_detect import NodeIdentity, NodeType
from maestro.services.venv_service import (
    _REQUIRED_PACKAGES,
    _SOMATIC_CONDA_ENV,
    _find_conda,
    _hippocampal_venv_path,
    check_venv_health,
    detect_active_venv,
    get_venv_run_prefix,
    run_in_venv,
)


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

_FAKE_UBIK_ROOT = Path("/fake/ubik")
_FAKE_VENV = _FAKE_UBIK_ROOT / "hippocampal" / "venv"


def _make_hippo(
    ubik_root: Path = _FAKE_UBIK_ROOT,
    venv_path: Path | None = None,
) -> NodeIdentity:
    return NodeIdentity(
        node_type=NodeType.HIPPOCAMPAL,
        hostname="mac.lan",
        platform="darwin",
        ubik_root=ubik_root,
        is_wsl=False,
        tailscale_ip="100.0.0.1",
        python_venv_path=venv_path or (ubik_root / "hippocampal" / "venv"),
        python_activate_cmd=f"source {ubik_root}/hippocampal/venv/bin/activate",
    )


def _make_somatic() -> NodeIdentity:
    return NodeIdentity(
        node_type=NodeType.SOMATIC,
        hostname="adrian",
        platform="linux",
        ubik_root=Path("/home/gasu/ubik"),
        is_wsl=True,
        tailscale_ip="100.0.0.2",
        python_venv_path=None,
        python_activate_cmd="conda activate pytorch_env",
    )


def _make_unknown() -> NodeIdentity:
    return NodeIdentity(
        node_type=NodeType.UNKNOWN,
        hostname="unknown",
        platform="linux",
        ubik_root=Path("/tmp"),
        is_wsl=False,
        tailscale_ip=None,
        python_venv_path=None,
        python_activate_cmd=None,
    )


# ---------------------------------------------------------------------------
# detect_active_venv()
# ---------------------------------------------------------------------------

class TestDetectActiveVenv:
    def test_returns_none_when_no_env_vars(self, monkeypatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
        assert detect_active_venv() is None

    def test_returns_virtual_env(self, monkeypatch):
        monkeypatch.setenv("VIRTUAL_ENV", "/path/to/venv")
        monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
        assert detect_active_venv() == "/path/to/venv"

    def test_returns_conda_env(self, monkeypatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "pytorch_env")
        assert detect_active_venv() == "pytorch_env"

    def test_virtual_env_takes_priority_over_conda(self, monkeypatch):
        monkeypatch.setenv("VIRTUAL_ENV", "/path/to/venv")
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "pytorch_env")
        result = detect_active_venv()
        assert result == "/path/to/venv"

    def test_empty_virtual_env_falls_through_to_conda(self, monkeypatch):
        monkeypatch.setenv("VIRTUAL_ENV", "")
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
        # Empty string is falsy, so conda wins
        assert detect_active_venv() == "base"


# ---------------------------------------------------------------------------
# get_venv_run_prefix()
# ---------------------------------------------------------------------------

class TestGetVenvRunPrefix:
    def test_hippocampal_starts_with_bash(self):
        prefix = get_venv_run_prefix(_make_hippo())
        assert prefix[0] == "bash"
        assert prefix[1] == "-c"

    def test_hippocampal_includes_source_activate(self):
        prefix = get_venv_run_prefix(_make_hippo())
        shell_str = prefix[2]
        assert "source" in shell_str
        assert "activate" in shell_str

    def test_hippocampal_includes_venv_path(self):
        node = _make_hippo()
        prefix = get_venv_run_prefix(node)
        venv_str = str(node.python_venv_path)
        assert venv_str in prefix[2]

    def test_hippocampal_ends_with_ampersand(self):
        prefix = get_venv_run_prefix(_make_hippo())
        # Prefix string ends with "&& " so it can be appended to
        assert prefix[2].endswith("&& ")

    def test_hippocampal_uses_python_venv_path_from_node(self):
        node = _make_hippo(venv_path=Path("/custom/venv"))
        prefix = get_venv_run_prefix(node)
        assert "/custom/venv" in prefix[2]

    def test_somatic_uses_conda_run(self):
        with patch("maestro.services.venv_service._find_conda", return_value="conda"):
            prefix = get_venv_run_prefix(_make_somatic())
        assert "run" in prefix
        assert "-n" in prefix
        assert _SOMATIC_CONDA_ENV in prefix

    def test_somatic_includes_no_capture_output(self):
        with patch("maestro.services.venv_service._find_conda", return_value="conda"):
            prefix = get_venv_run_prefix(_make_somatic())
        assert "--no-capture-output" in prefix

    def test_somatic_returns_list(self):
        with patch("maestro.services.venv_service._find_conda", return_value="conda"):
            prefix = get_venv_run_prefix(_make_somatic())
        assert isinstance(prefix, list)
        assert len(prefix) >= 4

    def test_unknown_returns_bash_prefix(self):
        prefix = get_venv_run_prefix(_make_unknown())
        assert "bash" in prefix

    def test_hippocampal_fallback_when_no_venv_path(self):
        node = NodeIdentity(
            node_type=NodeType.HIPPOCAMPAL,
            hostname="mac.lan",
            platform="darwin",
            ubik_root=_FAKE_UBIK_ROOT,
            is_wsl=False,
            tailscale_ip=None,
            python_venv_path=None,  # no explicit path
            python_activate_cmd=None,
        )
        prefix = get_venv_run_prefix(node)
        # Should fall back to ubik_root / hippocampal / venv
        expected = str(_FAKE_UBIK_ROOT / "hippocampal" / "venv")
        assert expected in prefix[2]


# ---------------------------------------------------------------------------
# run_in_venv()
# ---------------------------------------------------------------------------

class TestRunInVenv:
    @pytest.mark.asyncio
    async def test_hippocampal_uses_bash_source_activate(self):
        node = _make_hippo()
        with patch(
            "maestro.services.venv_service._run_proc",
            new_callable=AsyncMock,
            return_value=(0, "ok", ""),
        ) as mock_proc:
            rc, stdout, _ = await run_in_venv("echo hello", node)
        assert rc == 0
        call_args = mock_proc.call_args[0]
        assert call_args[0] == "bash"
        assert call_args[1] == "-c"
        shell_cmd = call_args[2]
        assert "source" in shell_cmd
        assert "activate" in shell_cmd
        assert "echo hello" in shell_cmd

    @pytest.mark.asyncio
    async def test_hippocampal_includes_venv_path_in_command(self):
        node = _make_hippo()
        venv_str = str(node.python_venv_path)
        with patch(
            "maestro.services.venv_service._run_proc",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ) as mock_proc:
            await run_in_venv("python --version", node)
        shell_cmd = mock_proc.call_args[0][2]
        assert venv_str in shell_cmd

    @pytest.mark.asyncio
    async def test_somatic_uses_conda_run(self):
        node = _make_somatic()
        with patch("maestro.services.venv_service._find_conda", return_value="conda"), \
             patch(
                 "maestro.services.venv_service._run_proc",
                 new_callable=AsyncMock,
                 return_value=(0, "output", ""),
             ) as mock_proc:
            rc, stdout, _ = await run_in_venv("python --version", node)
        call_args = mock_proc.call_args[0]
        assert call_args[0] == "conda"
        assert "run" in call_args
        assert "-n" in call_args
        assert _SOMATIC_CONDA_ENV in call_args
        assert "--no-capture-output" in call_args

    @pytest.mark.asyncio
    async def test_somatic_passes_command_via_bash(self):
        node = _make_somatic()
        with patch("maestro.services.venv_service._find_conda", return_value="conda"), \
             patch(
                 "maestro.services.venv_service._run_proc",
                 new_callable=AsyncMock,
                 return_value=(0, "", ""),
             ) as mock_proc:
            await run_in_venv("python -c 'print(1)'", node)
        call_args = mock_proc.call_args[0]
        # The command should be wrapped in bash -c
        assert "bash" in call_args
        assert "-c" in call_args
        assert "python -c 'print(1)'" in call_args

    @pytest.mark.asyncio
    async def test_unknown_node_runs_directly(self):
        node = _make_unknown()
        with patch(
            "maestro.services.venv_service._run_proc",
            new_callable=AsyncMock,
            return_value=(0, "hi", ""),
        ) as mock_proc:
            rc, stdout, _ = await run_in_venv("echo hi", node)
        assert rc == 0
        call_args = mock_proc.call_args[0]
        assert "bash" in call_args
        assert "echo hi" in call_args

    @pytest.mark.asyncio
    async def test_timeout_forwarded(self):
        node = _make_hippo()
        with patch(
            "maestro.services.venv_service._run_proc",
            new_callable=AsyncMock,
            return_value=(0, "", ""),
        ) as mock_proc:
            await run_in_venv("cmd", node, timeout=99.0)
        assert mock_proc.call_args[1]["timeout"] == 99.0

    @pytest.mark.asyncio
    async def test_returns_rc_stdout_stderr(self):
        node = _make_hippo()
        with patch(
            "maestro.services.venv_service._run_proc",
            new_callable=AsyncMock,
            return_value=(42, "out", "err"),
        ):
            rc, stdout, stderr = await run_in_venv("cmd", node)
        assert rc == 42
        assert stdout == "out"
        assert stderr == "err"


# ---------------------------------------------------------------------------
# _find_conda()
# ---------------------------------------------------------------------------

class TestFindConda:
    def test_uses_conda_exe_env_var_when_set(self, tmp_path, monkeypatch):
        fake_conda = tmp_path / "conda"
        fake_conda.write_text("#!/bin/sh"); fake_conda.chmod(0o755)
        monkeypatch.setenv("CONDA_EXE", str(fake_conda))
        result = _find_conda()
        assert result == str(fake_conda)

    def test_ignores_conda_exe_when_path_missing(self, monkeypatch):
        monkeypatch.setenv("CONDA_EXE", "/nonexistent/conda")
        with patch("shutil.which", return_value="/usr/bin/conda"):
            result = _find_conda()
        assert result == "conda"

    def test_falls_back_to_path_when_conda_exe_missing(self, monkeypatch):
        monkeypatch.delenv("CONDA_EXE", raising=False)
        with patch("shutil.which", return_value="/usr/bin/conda"):
            result = _find_conda()
        assert result == "conda"

    def test_tries_candidate_paths_when_not_on_path(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CONDA_EXE", raising=False)
        fake_conda = tmp_path / "conda"
        fake_conda.write_text("#!/bin/sh"); fake_conda.chmod(0o755)
        with patch("shutil.which", return_value=None), \
             patch(
                 "maestro.services.venv_service._CONDA_CANDIDATE_PATHS",
                 [fake_conda],
             ):
            result = _find_conda()
        assert result == str(fake_conda)

    def test_returns_bare_conda_when_nothing_found(self, monkeypatch):
        monkeypatch.delenv("CONDA_EXE", raising=False)
        with patch("shutil.which", return_value=None), \
             patch("maestro.services.venv_service._CONDA_CANDIDATE_PATHS", []):
            result = _find_conda()
        assert result == "conda"


# ---------------------------------------------------------------------------
# check_venv_health()
# ---------------------------------------------------------------------------

_HIPPO_PACKAGES = _REQUIRED_PACKAGES[NodeType.HIPPOCAMPAL]
_SOMATIC_PACKAGES = _REQUIRED_PACKAGES[NodeType.SOMATIC]


class TestCheckVenvHealthHippocampal:
    @pytest.mark.asyncio
    async def test_unhealthy_when_venv_path_missing(self):
        node = _make_hippo()
        with patch("maestro.services.venv_service._venv_path_exists", return_value=False):
            result = await check_venv_health(node)
        assert result.healthy is False
        assert "venv not found" in result.error

    @pytest.mark.asyncio
    async def test_unhealthy_when_python_fails(self):
        node = _make_hippo()
        with patch("maestro.services.venv_service._venv_path_exists", return_value=True), \
             patch(
                 "maestro.services.venv_service.run_in_venv",
                 new_callable=AsyncMock,
                 return_value=(1, "", "python: command not found"),
             ):
            result = await check_venv_health(node)
        assert result.healthy is False
        assert "python check failed" in result.error

    @pytest.mark.asyncio
    async def test_unhealthy_when_python_raises(self):
        node = _make_hippo()
        with patch("maestro.services.venv_service._venv_path_exists", return_value=True), \
             patch(
                 "maestro.services.venv_service.run_in_venv",
                 new_callable=AsyncMock,
                 side_effect=OSError("bash not found"),
             ):
            result = await check_venv_health(node)
        assert result.healthy is False
        assert "raised" in result.error

    @pytest.mark.asyncio
    async def test_unhealthy_when_package_missing(self):
        node = _make_hippo()

        async def side_effect(cmd, node_arg, **kw):
            if "sys.prefix" in cmd:
                return (0, "/fake/venv", "")
            # All packages fail
            return (1, "", "No module named 'fastmcp'")

        with patch("maestro.services.venv_service._venv_path_exists", return_value=True), \
             patch(
                 "maestro.services.venv_service.run_in_venv",
                 new_callable=AsyncMock,
                 side_effect=side_effect,
             ):
            result = await check_venv_health(node)
        assert result.healthy is False
        assert "Missing packages" in result.error
        assert result.details["packages_missing"] != []

    @pytest.mark.asyncio
    async def test_healthy_when_all_pass(self):
        node = _make_hippo()

        with patch("maestro.services.venv_service._venv_path_exists", return_value=True), \
             patch(
                 "maestro.services.venv_service.run_in_venv",
                 new_callable=AsyncMock,
                 return_value=(0, "/fake/venv", ""),
             ):
            result = await check_venv_health(node)
        assert result.healthy is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_healthy_result_has_sys_prefix(self):
        node = _make_hippo()
        with patch("maestro.services.venv_service._venv_path_exists", return_value=True), \
             patch(
                 "maestro.services.venv_service.run_in_venv",
                 new_callable=AsyncMock,
                 return_value=(0, "/actual/venv/prefix", ""),
             ):
            result = await check_venv_health(node)
        assert result.details.get("sys_prefix") == "/actual/venv/prefix"

    @pytest.mark.asyncio
    async def test_healthy_result_lists_packages_ok(self):
        node = _make_hippo()
        with patch("maestro.services.venv_service._venv_path_exists", return_value=True), \
             patch(
                 "maestro.services.venv_service.run_in_venv",
                 new_callable=AsyncMock,
                 return_value=(0, "/prefix", ""),
             ):
            result = await check_venv_health(node)
        assert set(result.details["packages_ok"]) == set(_HIPPO_PACKAGES)

    @pytest.mark.asyncio
    async def test_result_has_latency(self):
        node = _make_hippo()
        with patch("maestro.services.venv_service._venv_path_exists", return_value=True), \
             patch(
                 "maestro.services.venv_service.run_in_venv",
                 new_callable=AsyncMock,
                 return_value=(0, "/prefix", ""),
             ):
            result = await check_venv_health(node)
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_result_node_type_is_hippocampal(self):
        node = _make_hippo()
        with patch("maestro.services.venv_service._venv_path_exists", return_value=True), \
             patch(
                 "maestro.services.venv_service.run_in_venv",
                 new_callable=AsyncMock,
                 return_value=(0, "/prefix", ""),
             ):
            result = await check_venv_health(node)
        assert result.node == NodeType.HIPPOCAMPAL
        assert result.name == "venv"

    @pytest.mark.asyncio
    async def test_identifies_which_package_is_missing(self):
        node = _make_hippo()
        # Only fastmcp is missing
        async def side_effect(cmd, node_arg, **kw):
            if "sys.prefix" in cmd:
                return (0, "/prefix", "")
            if "fastmcp" in cmd:
                return (1, "", "No module named 'fastmcp'")
            return (0, "", "")  # chromadb, neo4j OK

        with patch("maestro.services.venv_service._venv_path_exists", return_value=True), \
             patch(
                 "maestro.services.venv_service.run_in_venv",
                 new_callable=AsyncMock,
                 side_effect=side_effect,
             ):
            result = await check_venv_health(node)
        assert result.healthy is False
        assert result.details["packages_missing"] == ["fastmcp"]

    @pytest.mark.asyncio
    async def test_venv_path_in_details(self):
        node = _make_hippo()
        with patch("maestro.services.venv_service._venv_path_exists", return_value=False):
            result = await check_venv_health(node)
        assert "venv_path" in result.details

    @pytest.mark.asyncio
    async def test_uses_actual_venv_dir(self, tmp_path):
        """With real tmp_path, check that an existing venv passes step 1."""
        venv_path = tmp_path / "hippocampal" / "venv"
        venv_path.mkdir(parents=True)
        node = _make_hippo(ubik_root=tmp_path, venv_path=venv_path)

        with patch(
            "maestro.services.venv_service.run_in_venv",
            new_callable=AsyncMock,
            return_value=(0, str(venv_path), ""),
        ):
            result = await check_venv_health(node)
        assert result.healthy is True


class TestCheckVenvHealthSomatic:
    @pytest.mark.asyncio
    async def test_somatic_no_path_check(self):
        """Somatic uses conda — no filesystem path to check."""
        node = _make_somatic()
        # No _venv_path_exists patch needed; somatic skips path check
        with patch(
            "maestro.services.venv_service.run_in_venv",
            new_callable=AsyncMock,
            return_value=(0, "/opt/conda/envs/pytorch_env", ""),
        ):
            result = await check_venv_health(node)
        assert result.healthy is True

    @pytest.mark.asyncio
    async def test_somatic_unhealthy_when_conda_fails(self):
        node = _make_somatic()
        with patch(
            "maestro.services.venv_service.run_in_venv",
            new_callable=AsyncMock,
            return_value=(1, "", "CondaError: env not found"),
        ):
            result = await check_venv_health(node)
        assert result.healthy is False

    @pytest.mark.asyncio
    async def test_somatic_checks_correct_packages(self):
        node = _make_somatic()

        async def side_effect(cmd, node_arg, **kw):
            if "sys.prefix" in cmd:
                return (0, "/opt/conda/envs/pytorch_env", "")
            # torch missing
            if "torch" in cmd:
                return (1, "", "No module named 'torch'")
            return (0, "", "")

        with patch(
            "maestro.services.venv_service.run_in_venv",
            new_callable=AsyncMock,
            side_effect=side_effect,
        ):
            result = await check_venv_health(node)
        assert result.healthy is False
        assert "torch" in result.details["packages_missing"]

    @pytest.mark.asyncio
    async def test_somatic_node_type_in_result(self):
        node = _make_somatic()
        with patch(
            "maestro.services.venv_service.run_in_venv",
            new_callable=AsyncMock,
            return_value=(0, "/prefix", ""),
        ):
            result = await check_venv_health(node)
        assert result.node == NodeType.SOMATIC


class TestCheckVenvHealthUnknown:
    @pytest.mark.asyncio
    async def test_unknown_node_skips_path_and_package_checks(self):
        """UNKNOWN node has no packages list → only python check runs."""
        node = _make_unknown()
        with patch(
            "maestro.services.venv_service.run_in_venv",
            new_callable=AsyncMock,
            return_value=(0, "/usr", ""),
        ):
            result = await check_venv_health(node)
        # No packages to check → healthy if python works
        assert result.healthy is True
        assert result.details["packages_checked"] == []


# ---------------------------------------------------------------------------
# Package constants sanity checks
# ---------------------------------------------------------------------------

class TestRequiredPackages:
    def test_hippocampal_packages_defined(self):
        pkgs = _REQUIRED_PACKAGES[NodeType.HIPPOCAMPAL]
        assert "chromadb" in pkgs
        assert "neo4j" in pkgs
        assert "fastmcp" in pkgs

    def test_somatic_packages_defined(self):
        pkgs = _REQUIRED_PACKAGES[NodeType.SOMATIC]
        assert "torch" in pkgs
        assert "vllm" in pkgs
        assert "httpx" in pkgs

    def test_unknown_not_in_required_packages(self):
        # UNKNOWN node has no required packages
        assert _REQUIRED_PACKAGES.get(NodeType.UNKNOWN) is None
