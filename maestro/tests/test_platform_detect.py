"""
Tests for maestro.platform_detect

Verifies NodeType, NodeIdentity, detect_node(), and get_remote_node_ip()
without touching the real filesystem, network, or live Tailscale state.

Coverage:
    detect_node:
        darwin platform → HIPPOCAMPAL
        linux platform  → SOMATIC
        win32 platform  → UNKNOWN
        WSL marker (/proc/version contains 'microsoft') → is_wsl=True
        hostname 'MiniM4-2025' → HIPPOCAMPAL (even on linux)
        hostname 'mac.lan'     → HIPPOCAMPAL
        hostname 'Adrian'      → SOMATIC (even on darwin)
        hostname 'adrian-wsl'  → SOMATIC
        hostname unknown + platform darwin → HIPPOCAMPAL via platform
        UBIK_ROOT path fallback when platform unknown
        ubik_root defaults when no path found
        python_venv_path set for HIPPOCAMPAL, None for SOMATIC/UNKNOWN
        python_activate_cmd set correctly for each node type
        tailscale_ip from config (mocked) when available
        tailscale_ip from defaults when config unavailable
        result is frozen (immutable)
        cache: two calls return the same object
        cache_clear forces re-detection
        exception in _detect_node_impl → UNKNOWN (never raises)

    get_remote_node_ip:
        HIPPOCAMPAL → somatic IP from config
        SOMATIC     → hippocampal IP from config
        UNKNOWN     → raises ValueError
        config unavailable → falls back to hardcoded defaults
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from maestro.platform_detect import (
    NodeIdentity,
    NodeType,
    _HIPPOCAMPAL_TAILSCALE_IP_DEFAULT,
    _LINUX_UBIK_ROOT,
    _MACOS_UBIK_ROOT,
    _SOMATIC_TAILSCALE_IP_DEFAULT,
    detect_node,
    get_remote_node_ip,
)

_PATCH_PATH_EXISTS = "maestro.platform_detect._path_exists"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_detect_cache():
    """Clear detect_node lru_cache before and after each test."""
    detect_node.cache_clear()
    yield
    detect_node.cache_clear()


def _make_node(node_type: NodeType, **overrides) -> NodeIdentity:
    """Build a NodeIdentity with sensible defaults, merging any overrides."""
    defaults: dict = dict(
        node_type=node_type,
        hostname="test-host",
        platform="darwin" if node_type == NodeType.HIPPOCAMPAL else "linux",
        ubik_root=Path("/tmp/ubik"),
        is_wsl=False,
        tailscale_ip=_HIPPOCAMPAL_TAILSCALE_IP_DEFAULT
        if node_type == NodeType.HIPPOCAMPAL
        else _SOMATIC_TAILSCALE_IP_DEFAULT,
        python_venv_path=Path("/tmp/ubik/hippocampal/venv")
        if node_type == NodeType.HIPPOCAMPAL
        else None,
        python_activate_cmd="source /tmp/ubik/hippocampal/venv/bin/activate"
        if node_type == NodeType.HIPPOCAMPAL
        else "conda activate pytorch_env",
    )
    defaults.update(overrides)
    return NodeIdentity(**defaults)


# ---------------------------------------------------------------------------
# Helpers for patching detect_node internals
# ---------------------------------------------------------------------------

def _patch_env(
    *,
    platform: str = "darwin",
    hostname: str = "test-host",
    wsl: bool = False,
    ubik_root: Path | None = None,
    config_hipp_ip: str = _HIPPOCAMPAL_TAILSCALE_IP_DEFAULT,
    config_som_ip: str = _SOMATIC_TAILSCALE_IP_DEFAULT,
):
    """Return a context manager that patches platform + filesystem + config."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        mock_cfg = MagicMock()
        mock_cfg.hippocampal.tailscale_ip = config_hipp_ip
        mock_cfg.somatic.tailscale_ip = config_som_ip

        with (
            patch("maestro.platform_detect.sys") as mock_sys,
            patch("maestro.platform_detect.socket.gethostname", return_value=hostname),
            patch("maestro.platform_detect._check_wsl", return_value=wsl),
            patch("maestro.platform_detect._resolve_ubik_root", return_value=ubik_root),
            patch("maestro.platform_detect.get_config", return_value=mock_cfg),
        ):
            mock_sys.platform = platform
            yield

    return _ctx()


# ---------------------------------------------------------------------------
# detect_node — platform signal
# ---------------------------------------------------------------------------

class TestDetectNodeByPlatform:
    def test_darwin_gives_hippocampal(self):
        with _patch_env(platform="darwin", hostname="unknown-host"):
            node = detect_node()
        assert node.node_type == NodeType.HIPPOCAMPAL
        assert node.platform == "darwin"

    def test_linux_gives_somatic(self):
        with _patch_env(platform="linux", hostname="unknown-host"):
            node = detect_node()
        assert node.node_type == NodeType.SOMATIC
        assert node.platform == "linux"

    def test_linux2_gives_somatic(self):
        with _patch_env(platform="linux2", hostname="unknown-host"):
            node = detect_node()
        assert node.node_type == NodeType.SOMATIC

    def test_win32_gives_unknown(self):
        with _patch_env(platform="win32", hostname="unknown-host"):
            node = detect_node()
        assert node.node_type == NodeType.UNKNOWN

    def test_wsl_flag_true_on_linux_with_microsoft_proc(self):
        with _patch_env(platform="linux", hostname="unknown-host", wsl=True):
            node = detect_node()
        assert node.is_wsl is True
        assert node.node_type == NodeType.SOMATIC

    def test_wsl_false_on_darwin(self):
        with _patch_env(platform="darwin", wsl=False):
            node = detect_node()
        assert node.is_wsl is False


# ---------------------------------------------------------------------------
# detect_node — hostname refinement
# ---------------------------------------------------------------------------

class TestDetectNodeByHostname:
    def test_minim4_hostname_gives_hippocampal(self):
        # Hippocampal even if platform says linux (shouldn't happen, but safe)
        with _patch_env(platform="linux", hostname="MiniM4-2025"):
            node = detect_node()
        assert node.node_type == NodeType.HIPPOCAMPAL

    def test_mac_lan_hostname_gives_hippocampal(self):
        with _patch_env(platform="darwin", hostname="mac.lan"):
            node = detect_node()
        assert node.node_type == NodeType.HIPPOCAMPAL

    def test_adrian_hostname_gives_somatic(self):
        with _patch_env(platform="darwin", hostname="Adrian"):
            node = detect_node()
        assert node.node_type == NodeType.SOMATIC

    def test_adrian_wsl_hostname_gives_somatic(self):
        with _patch_env(platform="linux", hostname="Adrian-WSL"):
            node = detect_node()
        assert node.node_type == NodeType.SOMATIC

    def test_wsl_in_hostname_gives_somatic(self):
        with _patch_env(platform="win32", hostname="my-wsl-box"):
            node = detect_node()
        assert node.node_type == NodeType.SOMATIC

    def test_unknown_hostname_falls_back_to_platform(self):
        with _patch_env(platform="darwin", hostname="mystery-machine"):
            node = detect_node()
        assert node.node_type == NodeType.HIPPOCAMPAL


# ---------------------------------------------------------------------------
# detect_node — UBIK_ROOT path fallback
# ---------------------------------------------------------------------------

class TestDetectNodeByUbikRoot:
    def test_macos_path_resolves_hippocampal_when_unknown(self):
        # _path_exists returns True only for the macOS path
        def _exists(p: Path) -> bool:
            return p == _MACOS_UBIK_ROOT

        with (
            patch("maestro.platform_detect.sys") as mock_sys,
            patch("maestro.platform_detect.socket.gethostname", return_value="mystery"),
            patch("maestro.platform_detect._check_wsl", return_value=False),
            patch("maestro.platform_detect._resolve_ubik_root", return_value=_MACOS_UBIK_ROOT),
            patch(_PATCH_PATH_EXISTS, side_effect=_exists),
            patch("maestro.platform_detect.get_config", side_effect=RuntimeError("no config")),
        ):
            mock_sys.platform = "win32"
            node = detect_node()
        assert node.node_type == NodeType.HIPPOCAMPAL

    def test_linux_path_resolves_somatic_when_unknown(self):
        # _path_exists returns True only for the Linux path
        def _exists(p: Path) -> bool:
            return p == _LINUX_UBIK_ROOT

        with (
            patch("maestro.platform_detect.sys") as mock_sys,
            patch("maestro.platform_detect.socket.gethostname", return_value="mystery"),
            patch("maestro.platform_detect._check_wsl", return_value=False),
            patch("maestro.platform_detect._resolve_ubik_root", return_value=_LINUX_UBIK_ROOT),
            patch(_PATCH_PATH_EXISTS, side_effect=_exists),
            patch("maestro.platform_detect.get_config", side_effect=RuntimeError("no config")),
        ):
            mock_sys.platform = "win32"
            node = detect_node()
        assert node.node_type == NodeType.SOMATIC


# ---------------------------------------------------------------------------
# detect_node — ubik_root assignment
# ---------------------------------------------------------------------------

class TestUbikRootAssignment:
    def test_ubik_root_from_detection(self):
        fake_root = Path("/tmp/fake-ubik")
        with _patch_env(platform="darwin", ubik_root=fake_root):
            node = detect_node()
        assert node.ubik_root == fake_root

    def test_ubik_root_defaults_to_macos_path_when_none_and_hippocampal(self):
        with _patch_env(platform="darwin", hostname="unknown-host", ubik_root=None):
            node = detect_node()
        assert node.node_type == NodeType.HIPPOCAMPAL
        assert node.ubik_root == _MACOS_UBIK_ROOT

    def test_ubik_root_defaults_to_linux_path_when_none_and_somatic(self):
        with _patch_env(platform="linux", hostname="unknown-host", ubik_root=None):
            node = detect_node()
        assert node.node_type == NodeType.SOMATIC
        assert node.ubik_root == _LINUX_UBIK_ROOT


# ---------------------------------------------------------------------------
# detect_node — venv and activate_cmd
# ---------------------------------------------------------------------------

class TestVenvPaths:
    def test_hippocampal_venv_under_ubik_root(self):
        fake_root = Path("/tmp/ubik")
        with _patch_env(platform="darwin", ubik_root=fake_root):
            node = detect_node()
        assert node.python_venv_path == fake_root / "hippocampal" / "venv"

    def test_hippocampal_activate_cmd_references_venv(self):
        fake_root = Path("/tmp/ubik")
        with _patch_env(platform="darwin", ubik_root=fake_root):
            node = detect_node()
        expected_venv = fake_root / "hippocampal" / "venv"
        assert node.python_activate_cmd == f"source {expected_venv}/bin/activate"

    def test_somatic_venv_path_is_none(self):
        with _patch_env(platform="linux", hostname="unknown-host"):
            node = detect_node()
        assert node.python_venv_path is None

    def test_somatic_activate_cmd_is_conda(self):
        with _patch_env(platform="linux", hostname="unknown-host"):
            node = detect_node()
        assert node.python_activate_cmd == "conda activate pytorch_env"

    def test_unknown_venv_path_is_none(self):
        with _patch_env(platform="win32", hostname="unknown-host"):
            node = detect_node()
        assert node.python_venv_path is None

    def test_unknown_activate_cmd_is_none(self):
        with _patch_env(platform="win32", hostname="unknown-host"):
            node = detect_node()
        assert node.python_activate_cmd is None


# ---------------------------------------------------------------------------
# detect_node — tailscale_ip
# ---------------------------------------------------------------------------

class TestTailscaleIp:
    def test_hippocampal_ip_from_config(self):
        with _patch_env(platform="darwin", config_hipp_ip="10.0.0.1"):
            node = detect_node()
        assert node.tailscale_ip == "10.0.0.1"

    def test_somatic_ip_from_config(self):
        with _patch_env(platform="linux", hostname="unknown-host", config_som_ip="10.0.0.2"):
            node = detect_node()
        assert node.tailscale_ip == "10.0.0.2"

    def test_hippocampal_ip_falls_back_to_default(self):
        with (
            patch("maestro.platform_detect.sys") as mock_sys,
            patch("maestro.platform_detect.socket.gethostname", return_value="unknown"),
            patch("maestro.platform_detect._check_wsl", return_value=False),
            patch("maestro.platform_detect._resolve_ubik_root", return_value=None),
            patch("maestro.platform_detect.get_config", side_effect=RuntimeError("no cfg")),
        ):
            mock_sys.platform = "darwin"
            node = detect_node()
        assert node.tailscale_ip == _HIPPOCAMPAL_TAILSCALE_IP_DEFAULT

    def test_unknown_tailscale_ip_is_none(self):
        with _patch_env(platform="win32", hostname="unknown-host"):
            node = detect_node()
        assert node.tailscale_ip is None


# ---------------------------------------------------------------------------
# detect_node — identity fields
# ---------------------------------------------------------------------------

class TestNodeIdentityFields:
    def test_hostname_captured(self):
        with _patch_env(platform="darwin", hostname="my-mac"):
            node = detect_node()
        assert node.hostname == "my-mac"

    def test_platform_captured(self):
        with _patch_env(platform="darwin"):
            node = detect_node()
        assert node.platform == "darwin"

    def test_is_frozen_immutable(self):
        with _patch_env(platform="darwin"):
            node = detect_node()
        with pytest.raises((AttributeError, TypeError)):
            node.node_type = NodeType.SOMATIC  # type: ignore[misc]


# ---------------------------------------------------------------------------
# detect_node — caching
# ---------------------------------------------------------------------------

class TestDetectNodeCache:
    def test_two_calls_return_same_object(self):
        with _patch_env(platform="darwin"):
            a = detect_node()
            b = detect_node()
        assert a is b

    def test_cache_clear_forces_re_detection(self):
        with _patch_env(platform="darwin"):
            a = detect_node()
        detect_node.cache_clear()
        with _patch_env(platform="linux", hostname="unknown-host"):
            b = detect_node()
        assert a.node_type == NodeType.HIPPOCAMPAL
        assert b.node_type == NodeType.SOMATIC


# ---------------------------------------------------------------------------
# detect_node — error resilience
# ---------------------------------------------------------------------------

class TestDetectNodeResilience:
    def test_exception_in_impl_returns_unknown(self):
        with patch(
            "maestro.platform_detect._detect_node_impl",
            side_effect=RuntimeError("simulated failure"),
        ):
            node = detect_node()
        assert node.node_type == NodeType.UNKNOWN

    def test_never_raises_on_exception(self):
        with patch(
            "maestro.platform_detect._detect_node_impl",
            side_effect=Exception("anything"),
        ):
            node = detect_node()  # must not raise
        assert isinstance(node, NodeIdentity)


# ---------------------------------------------------------------------------
# get_remote_node_ip
# ---------------------------------------------------------------------------

class TestGetRemoteNodeIp:
    def test_hippocampal_returns_somatic_ip(self):
        mock_cfg = MagicMock()
        mock_cfg.somatic.tailscale_ip = "10.1.2.3"
        node = _make_node(NodeType.HIPPOCAMPAL)
        with patch("maestro.platform_detect.get_config", return_value=mock_cfg):
            ip = get_remote_node_ip(node)
        assert ip == "10.1.2.3"

    def test_somatic_returns_hippocampal_ip(self):
        mock_cfg = MagicMock()
        mock_cfg.hippocampal.tailscale_ip = "10.4.5.6"
        node = _make_node(NodeType.SOMATIC)
        with patch("maestro.platform_detect.get_config", return_value=mock_cfg):
            ip = get_remote_node_ip(node)
        assert ip == "10.4.5.6"

    def test_unknown_raises_value_error(self):
        node = _make_node(NodeType.UNKNOWN)
        with pytest.raises(ValueError, match="UNKNOWN"):
            get_remote_node_ip(node)

    def test_config_unavailable_hippocampal_falls_back(self):
        node = _make_node(NodeType.HIPPOCAMPAL)
        with patch(
            "maestro.platform_detect.get_config",
            side_effect=RuntimeError("no config"),
        ):
            ip = get_remote_node_ip(node)
        assert ip == _SOMATIC_TAILSCALE_IP_DEFAULT

    def test_config_unavailable_somatic_falls_back(self):
        node = _make_node(NodeType.SOMATIC)
        with patch(
            "maestro.platform_detect.get_config",
            side_effect=RuntimeError("no config"),
        ):
            ip = get_remote_node_ip(node)
        assert ip == _HIPPOCAMPAL_TAILSCALE_IP_DEFAULT


# ---------------------------------------------------------------------------
# NodeType — enum properties
# ---------------------------------------------------------------------------

class TestNodeTypeEnum:
    def test_values_are_strings(self):
        for member in NodeType:
            assert isinstance(member.value, str)

    def test_str_subclass(self):
        assert isinstance(NodeType.HIPPOCAMPAL, str)

    def test_canonical_values(self):
        assert NodeType.HIPPOCAMPAL.value == "hippocampal"
        assert NodeType.SOMATIC.value == "somatic"
        assert NodeType.UNKNOWN.value == "unknown"
