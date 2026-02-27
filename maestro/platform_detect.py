#!/usr/bin/env python3
"""
Maestro — Platform Detection & Node Identity

Determines which UBIK node this process is running on so that Maestro can
take node-appropriate actions (start only local services, choose the right
UBIK_ROOT, find the Python venv, reach the remote node via Tailscale).

Nodes:
    HIPPOCAMPAL — Mac Mini M4 Pro (macOS, arm64)
                  UBIK root:  /Volumes/990PRO 4T/UBIK/
                  Python env: {ubik_root}/hippocampal/venv/
    SOMATIC     — PowerSpec RTX 5090 (WSL2 Linux)
                  UBIK root:  /home/gasu/ubik/
                  Python env: conda, pytorch_env
    UNKNOWN     — Detection inconclusive (e.g. CI, new machine)

Detection strategy (in priority order):
    1. sys.platform       — 'darwin' → HIPPOCAMPAL; 'linux' → SOMATIC
    2. hostname           — 'minim4*' / 'mac.lan' → HIPPOCAMPAL;
                            'adrian*' / '*wsl*' → SOMATIC
    3. WSL marker         — /proc/version contains 'microsoft' → is_wsl=True
    4. UBIK_ROOT path     — which candidate path exists on disk

The result is cached with @lru_cache so detection runs exactly once per
process; call detect_node.cache_clear() to force re-detection (tests).

Usage:
    from maestro.platform_detect import detect_node, get_remote_node_ip

    node = detect_node()
    print(node.node_type.value)          # "hippocampal"
    print(node.ubik_root)               # PosixPath('/Volumes/990PRO 4T/UBIK')
    print(node.python_activate_cmd)     # "source .../venv/bin/activate"

    remote_ip = get_remote_node_ip(node)  # IP of the other node

Author: UBIK Project
Version: 0.1.0
"""

import logging
import socket
import sys
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from maestro.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MACOS_UBIK_ROOT = Path("/Volumes/990PRO 4T/UBIK")
_LINUX_UBIK_ROOT = Path("/home/gasu/ubik")

# Relative sub-path of the Python venv inside the Hippocampal UBIK root.
_HIPPOCAMPAL_VENV_SUBPATH = Path("hippocampal/venv")

# Hostname substrings that reliably identify each node (case-insensitive).
_HIPPOCAMPAL_HOSTNAME_MARKERS: frozenset[str] = frozenset({"minim4", "mac.lan"})
_SOMATIC_HOSTNAME_MARKERS: frozenset[str] = frozenset({"adrian", "wsl"})

# Fallback Tailscale IPs (overridden by get_config() when available).
_HIPPOCAMPAL_TAILSCALE_IP_DEFAULT = "100.103.242.91"
_SOMATIC_TAILSCALE_IP_DEFAULT = "100.79.166.114"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    """Which UBIK node this process is running on.

    Inherits ``str`` for clean JSON serialisation (``node.value`` is already
    a plain string).

    Attributes:
        HIPPOCAMPAL: Mac Mini M4 Pro running macOS.
        SOMATIC: PowerSpec RTX 5090 running vLLM under WSL2 Linux.
        UNKNOWN: Detection inconclusive — treat all remote operations with
            caution and do not attempt local-only service management.
    """

    HIPPOCAMPAL = "hippocampal"
    SOMATIC = "somatic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NodeIdentity:
    """Detected identity of the local UBIK node.

    All path fields use :class:`pathlib.Path`; all optional fields are
    ``None`` when not applicable or detection failed.

    Attributes:
        node_type: Which UBIK node this is.
        hostname: Raw hostname string returned by :func:`socket.gethostname`.
        platform: ``sys.platform`` value (``'darwin'``, ``'linux'``, …).
        ubik_root: Absolute path to the UBIK project root on this node.
        is_wsl: ``True`` when running inside Windows Subsystem for Linux.
        tailscale_ip: Tailscale IP address of *this* node, or ``None`` when
            the node type is UNKNOWN.
        python_venv_path: Absolute path to the Python virtual environment for
            this node; ``None`` for the Somatic node (which uses conda) and
            for UNKNOWN.
        python_activate_cmd: Shell command to activate the Python environment
            for this node; suitable for ``subprocess`` or user display.
    """

    node_type: NodeType
    hostname: str
    platform: str
    ubik_root: Path
    is_wsl: bool
    tailscale_ip: Optional[str]
    python_venv_path: Optional[Path]
    python_activate_cmd: Optional[str]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_hostname() -> str:
    """Return the system hostname without raising."""
    try:
        return socket.gethostname()
    except Exception as exc:
        logger.debug("platform_detect: hostname lookup failed: %s", exc)
        return "unknown"


def _check_wsl() -> bool:
    """Return ``True`` when /proc/version contains 'microsoft' (WSL marker)."""
    proc_version = Path("/proc/version")
    try:
        text = proc_version.read_text(encoding="utf-8", errors="ignore")
        result = "microsoft" in text.lower()
        logger.debug("platform_detect: /proc/version wsl=%s", result)
        return result
    except OSError:
        return False


def _resolve_ubik_root() -> Optional[Path]:
    """Locate an existing UBIK root directory.

    Priority:
        1. ``UBIK_ROOT`` environment variable (already-validated path).
        2. macOS default  (``/Volumes/990PRO 4T/UBIK``).
        3. Linux default  (``/home/gasu/ubik``).

    Returns:
        A :class:`~pathlib.Path` that exists on disk, or ``None`` if none
        of the candidates exist.
    """
    import os

    env_root = os.getenv("UBIK_ROOT")
    if env_root:
        candidate = Path(env_root)
        if candidate.exists():
            logger.debug("platform_detect: ubik_root from UBIK_ROOT env: %s", candidate)
            return candidate
        logger.debug(
            "platform_detect: UBIK_ROOT env set but path absent: %s", candidate
        )

    for path in (_MACOS_UBIK_ROOT, _LINUX_UBIK_ROOT):
        if path.exists():
            logger.debug("platform_detect: ubik_root from path probe: %s", path)
            return path

    return None


def _classify_by_hostname(hostname_lower: str) -> Optional[NodeType]:
    """Return a NodeType if the hostname matches a known marker, else None."""
    for marker in _HIPPOCAMPAL_HOSTNAME_MARKERS:
        if marker in hostname_lower:
            logger.debug(
                "platform_detect: hostname %r matched hippocampal marker %r",
                hostname_lower, marker,
            )
            return NodeType.HIPPOCAMPAL

    for marker in _SOMATIC_HOSTNAME_MARKERS:
        if marker in hostname_lower:
            logger.debug(
                "platform_detect: hostname %r matched somatic marker %r",
                hostname_lower, marker,
            )
            return NodeType.SOMATIC

    return None


def _path_exists(path: Path) -> bool:
    """Return whether *path* exists; isolated for test patching."""
    return path.exists()


def _classify_by_ubik_root(ubik_root: Optional[Path]) -> Optional[NodeType]:
    """Infer node type from which UBIK root directory was found."""
    if ubik_root is None:
        return None
    if _path_exists(_MACOS_UBIK_ROOT) and ubik_root == _MACOS_UBIK_ROOT:
        logger.debug("platform_detect: ubik_root path → HIPPOCAMPAL")
        return NodeType.HIPPOCAMPAL
    if _path_exists(_LINUX_UBIK_ROOT) and ubik_root == _LINUX_UBIK_ROOT:
        logger.debug("platform_detect: ubik_root path → SOMATIC")
        return NodeType.SOMATIC
    return None


def _tailscale_ip_for(node_type: NodeType) -> Optional[str]:
    """Return this node's Tailscale IP from config (falls back to defaults)."""
    try:
        cfg = get_config()
        if node_type == NodeType.HIPPOCAMPAL:
            return cfg.hippocampal.tailscale_ip
        if node_type == NodeType.SOMATIC:
            return cfg.somatic.tailscale_ip
    except Exception as exc:
        logger.debug("platform_detect: get_config() failed, using default IP: %s", exc)

    if node_type == NodeType.HIPPOCAMPAL:
        return _HIPPOCAMPAL_TAILSCALE_IP_DEFAULT
    if node_type == NodeType.SOMATIC:
        return _SOMATIC_TAILSCALE_IP_DEFAULT
    return None


def _build_identity(
    node_type: NodeType,
    hostname: str,
    platform_str: str,
    ubik_root: Optional[Path],
    is_wsl: bool,
) -> NodeIdentity:
    """Assemble a NodeIdentity from resolved detection data."""
    # Resolve ubik_root to a concrete Path (may not exist on disk yet).
    if ubik_root is None:
        if node_type == NodeType.HIPPOCAMPAL:
            ubik_root = _MACOS_UBIK_ROOT
        elif node_type == NodeType.SOMATIC:
            ubik_root = _LINUX_UBIK_ROOT
        else:
            ubik_root = Path.cwd()
        logger.debug(
            "platform_detect: no ubik_root found, defaulting to %s", ubik_root
        )

    if node_type == NodeType.HIPPOCAMPAL:
        venv_path: Optional[Path] = ubik_root / _HIPPOCAMPAL_VENV_SUBPATH
        activate_cmd: Optional[str] = f"source {venv_path}/bin/activate"
    elif node_type == NodeType.SOMATIC:
        venv_path = None
        activate_cmd = "conda activate pytorch_env"
    else:
        venv_path = None
        activate_cmd = None

    return NodeIdentity(
        node_type=node_type,
        hostname=hostname,
        platform=platform_str,
        ubik_root=ubik_root,
        is_wsl=is_wsl,
        tailscale_ip=_tailscale_ip_for(node_type),
        python_venv_path=venv_path,
        python_activate_cmd=activate_cmd,
    )


def _detect_node_impl() -> NodeIdentity:
    """Core detection logic (not cached; use :func:`detect_node` instead)."""
    hostname = _safe_hostname()
    hostname_lower = hostname.lower()
    platform_str = sys.platform
    is_wsl = _check_wsl()
    ubik_root = _resolve_ubik_root()

    logger.debug(
        "platform_detect: platform=%s hostname=%r is_wsl=%s ubik_root=%s",
        platform_str, hostname, is_wsl, ubik_root,
    )

    # ── 1. Platform signal (most reliable primary indicator) ───────────────
    if platform_str == "darwin":
        node_type: NodeType = NodeType.HIPPOCAMPAL
        logger.debug("platform_detect: platform=darwin → HIPPOCAMPAL")
    elif platform_str.startswith("linux") or is_wsl:
        node_type = NodeType.SOMATIC
        logger.debug("platform_detect: platform=linux/wsl → SOMATIC")
    else:
        node_type = NodeType.UNKNOWN
        logger.debug("platform_detect: unrecognised platform %r → UNKNOWN", platform_str)

    # ── 2. Hostname refinement (can upgrade UNKNOWN or confirm) ────────────
    hostname_type = _classify_by_hostname(hostname_lower)
    if hostname_type is not None:
        if hostname_type != node_type:
            logger.debug(
                "platform_detect: hostname overrides platform %s → %s",
                node_type.value, hostname_type.value,
            )
        node_type = hostname_type

    # ── 3. UBIK_ROOT path fallback (if still UNKNOWN) ─────────────────────
    if node_type == NodeType.UNKNOWN:
        path_type = _classify_by_ubik_root(ubik_root)
        if path_type is not None:
            node_type = path_type

    logger.debug("platform_detect: final node_type=%s", node_type.value)

    return _build_identity(node_type, hostname, platform_str, ubik_root, is_wsl)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def detect_node() -> NodeIdentity:
    """Detect and return the identity of the local UBIK node.

    Runs the full detection chain (platform → hostname → WSL → UBIK_ROOT
    path) and caches the result so detection only happens once per process.

    Never raises — returns a :class:`NodeIdentity` with
    ``node_type=NodeType.UNKNOWN`` on any unhandled failure.

    Returns:
        :class:`NodeIdentity` describing this node.

    Example:
        >>> node = detect_node()
        >>> print(node.node_type.value)
        'hippocampal'
        >>> print(node.ubik_root)
        /Volumes/990PRO 4T/UBIK

    Note:
        Call ``detect_node.cache_clear()`` to force re-detection — useful
        after mutating environment variables or in tests.
    """
    try:
        return _detect_node_impl()
    except Exception as exc:
        logger.debug("platform_detect: detection failed unexpectedly: %s", exc)
        return NodeIdentity(
            node_type=NodeType.UNKNOWN,
            hostname=_safe_hostname(),
            platform=sys.platform,
            ubik_root=Path.cwd(),
            is_wsl=False,
            tailscale_ip=None,
            python_venv_path=None,
            python_activate_cmd=None,
        )


def get_remote_node_ip(local: NodeIdentity) -> str:
    """Return the Tailscale IP of the *other* UBIK node.

    Reads from :func:`~maestro.config.get_config` so any ``.env`` overrides
    are respected; falls back to hardcoded defaults if config is unavailable.

    Args:
        local: The :class:`NodeIdentity` of the *current* node.

    Returns:
        Tailscale IP string of the remote node.

    Raises:
        ValueError: If ``local.node_type`` is ``UNKNOWN`` — it is impossible
            to determine which node is "remote" without knowing which is local.

    Example:
        >>> node = detect_node()
        >>> get_remote_node_ip(node)
        '100.79.166.114'   # Somatic IP when called from Hippocampal
    """
    if local.node_type == NodeType.UNKNOWN:
        raise ValueError(
            "Cannot determine remote node IP: local node type is UNKNOWN. "
            "Ensure the node is correctly identified before calling get_remote_node_ip()."
        )

    try:
        cfg = get_config()
        if local.node_type == NodeType.HIPPOCAMPAL:
            return cfg.somatic.tailscale_ip
        return cfg.hippocampal.tailscale_ip
    except Exception as exc:
        logger.debug(
            "platform_detect: get_config() unavailable for remote IP lookup: %s", exc
        )

    # Hardcoded fallbacks
    if local.node_type == NodeType.HIPPOCAMPAL:
        return _SOMATIC_TAILSCALE_IP_DEFAULT
    return _HIPPOCAMPAL_TAILSCALE_IP_DEFAULT
