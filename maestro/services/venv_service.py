#!/usr/bin/env python3
"""
Maestro — Virtual Environment Manager

Detects, validates, and activates the correct Python environment on each
UBIK node, and provides a uniform interface for running commands inside it.

    Hippocampal: standard venv at ``{ubik_root}/hippocampal/venv/``
    Somatic:     conda environment named ``pytorch_env``

Public API:
    detect_active_venv()        — read VIRTUAL_ENV / CONDA_DEFAULT_ENV
    get_venv_run_prefix(node)   — command prefix list for subprocess use
    run_in_venv(cmd, node)      — run a shell command in the correct env
    check_venv_health(node)     — verify the env exists and packages load

Design notes:
  - Venv activation is per-subprocess, never global (no profile mutation).
  - Conda is found via PATH first, then common miniconda/anaconda locations.
  - All subprocess calls have a hard timeout.
  - Path existence is wrapped in ``_venv_path_exists()`` for test patching.

Author: UBIK Project
Version: 0.1.0
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

from maestro.platform_detect import NodeIdentity, NodeType
from maestro.services.base import ProbeResult, _run_proc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SOMATIC_CONDA_ENV = "pytorch_env"

_REQUIRED_PACKAGES: dict[NodeType, list[str]] = {
    NodeType.HIPPOCAMPAL: ["chromadb", "neo4j", "fastmcp"],
    NodeType.SOMATIC:     ["torch", "vllm", "httpx"],
}

# Common conda installation prefixes, tried in order when "conda" is not on PATH.
_CONDA_CANDIDATE_PATHS: list[Path] = [
    Path.home() / "miniconda3" / "bin" / "conda",
    Path.home() / "anaconda3" / "bin" / "conda",
    Path.home() / "opt" / "miniconda3" / "bin" / "conda",
    Path.home() / "opt" / "anaconda3" / "bin" / "conda",
    Path("/opt/conda/bin/conda"),
    Path("/usr/local/conda/bin/conda"),
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _venv_path_exists(p: Path) -> bool:
    """Return whether *p* exists on disk.  Extracted for test patching."""
    return p.exists()


def _find_conda() -> str:
    """Return an executable path for conda.

    Checks the ``CONDA_EXE`` environment variable first (set by conda itself
    when the shell is properly initialised), then tries PATH, then falls back
    to known miniconda/anaconda installation prefixes.

    Returns:
        The conda executable as a string.  Returns ``"conda"`` (bare name) if
        no known path is found, letting the OS resolve it at runtime.
    """
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    # Check if "conda" is on PATH
    import shutil
    if shutil.which("conda"):
        return "conda"

    # Try common installation locations
    for candidate in _CONDA_CANDIDATE_PATHS:
        if candidate.exists():
            return str(candidate)

    logger.debug("venv_service: conda not found in known locations, using 'conda'")
    return "conda"


def _hippocampal_venv_path(node: NodeIdentity) -> Path:
    """Resolve the Hippocampal venv path from NodeIdentity or ubik_root."""
    return node.python_venv_path or (node.ubik_root / "hippocampal" / "venv")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_active_venv() -> Optional[str]:
    """Return the currently-active Python environment name or path.

    Checks standard environment variables in priority order:
      1. ``VIRTUAL_ENV`` — set by ``source venv/bin/activate``
      2. ``CONDA_DEFAULT_ENV`` — set by ``conda activate <env>``

    Returns:
        The environment identifier string, or ``None`` when no virtual
        environment is active.
    """
    return (
        os.environ.get("VIRTUAL_ENV")
        or os.environ.get("CONDA_DEFAULT_ENV")
        or None
    )


def get_venv_run_prefix(node: NodeIdentity) -> list[str]:
    """Return the command prefix needed to run inside the correct Python env.

    The returned list is designed to be combined with a command string:

    Hippocampal (venv)::

        prefix = get_venv_run_prefix(node)
        # ["bash", "-c", "source /path/to/venv/bin/activate && "]
        # Append command to prefix[-1]:
        cmd = prefix[:-1] + [prefix[-1] + "python -m mymodule"]

    Somatic (conda)::

        prefix = get_venv_run_prefix(node)
        # ["conda", "run", "-n", "pytorch_env", "--no-capture-output"]
        # Append command as extra args:
        cmd = prefix + ["python", "-m", "mymodule"]

    Args:
        node: Identity of the node whose env prefix to build.

    Returns:
        Command prefix list.  Returns ``["bash", "-c", ""]`` for UNKNOWN nodes.
    """
    if node.node_type == NodeType.HIPPOCAMPAL:
        venv_path = _hippocampal_venv_path(node)
        activate = str(venv_path / "bin" / "activate")
        return ["bash", "-c", f"source {activate} && "]

    if node.node_type == NodeType.SOMATIC:
        return [_find_conda(), "run", "-n", _SOMATIC_CONDA_ENV, "--no-capture-output"]

    # UNKNOWN — no environment wrapper
    return ["bash", "-c", ""]


async def run_in_venv(
    command: str,
    node: NodeIdentity,
    *,
    timeout: float = 30.0,
) -> tuple[int, str, str]:
    """Run a shell command inside the node's Python virtual environment.

    Activation is per-subprocess: the calling process's environment is
    never modified.

    Hippocampal::

        bash -c "source {venv}/bin/activate && {command}"

    Somatic::

        conda run -n pytorch_env --no-capture-output bash -c "{command}"

    Unknown::

        bash -c "{command}"

    Args:
        command: Shell command string to run inside the environment.
        node: Node whose virtual environment to use.
        timeout: Maximum seconds to wait for the subprocess.

    Returns:
        ``(returncode, stdout, stderr)`` as ``(int, str, str)``.

    Raises:
        asyncio.TimeoutError: If the subprocess exceeds *timeout*.
    """
    if node.node_type == NodeType.HIPPOCAMPAL:
        venv_path = _hippocampal_venv_path(node)
        activate = str(venv_path / "bin" / "activate")
        return await _run_proc(
            "bash", "-c", f"source {activate} && {command}",
            timeout=timeout,
        )

    if node.node_type == NodeType.SOMATIC:
        conda = _find_conda()
        return await _run_proc(
            conda, "run", "-n", _SOMATIC_CONDA_ENV, "--no-capture-output",
            "bash", "-c", command,
            timeout=timeout,
        )

    # UNKNOWN — run directly without env wrapper
    return await _run_proc("bash", "-c", command, timeout=timeout)


async def check_venv_health(
    node: NodeIdentity,
    *,
    timeout: float = 15.0,
) -> ProbeResult:
    """Verify the virtual environment exists and required packages load.

    Steps:
      1. (Hippocampal only) Verify the venv directory exists on disk.
      2. Run ``python -c "import sys; print(sys.prefix)"`` inside the env
         to confirm the interpreter is functional.
      3. Attempt to import each required package individually.

    Required packages:
      Hippocampal: ``chromadb``, ``neo4j``, ``fastmcp``
      Somatic:     ``torch``, ``vllm``, ``httpx``

    Args:
        node: Node identity to check the environment for.
        timeout: Maximum seconds for each subprocess call.

    Returns:
        :class:`~maestro.services.base.ProbeResult` with:
          - ``healthy=True`` when all checks pass.
          - ``details["sys_prefix"]`` set to the Python interpreter prefix.
          - ``details["packages_ok"]`` listing verified packages.
          - ``details["packages_missing"]`` listing any that failed to import.
    """
    start_ts = time.perf_counter()
    node_type = node.node_type
    details: dict = {"node_type": node_type.value}

    # ── Step 1: Hippocampal venv path check ───────────────────────────────
    if node_type == NodeType.HIPPOCAMPAL:
        venv_path = _hippocampal_venv_path(node)
        details["venv_path"] = str(venv_path)
        if not _venv_path_exists(venv_path):
            logger.warning("venv: path not found: %s", venv_path)
            return ProbeResult(
                name="venv",
                node=node_type,
                healthy=False,
                latency_ms=round((time.perf_counter() - start_ts) * 1000, 2),
                details=details,
                error=f"venv not found: {venv_path}",
            )

    # ── Step 2: Verify the Python interpreter works ────────────────────────
    try:
        rc, stdout, stderr = await run_in_venv(
            'python -c "import sys; print(sys.prefix)"',
            node,
            timeout=timeout,
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)
        logger.warning("venv: python check raised: %s", exc)
        return ProbeResult(
            name="venv",
            node=node_type,
            healthy=False,
            latency_ms=latency_ms,
            details=details,
            error=f"python check raised: {exc}",
        )

    if rc != 0:
        latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)
        logger.warning("venv: python check failed (rc=%d): %s", rc, stderr.strip()[:200])
        return ProbeResult(
            name="venv",
            node=node_type,
            healthy=False,
            latency_ms=latency_ms,
            details=details,
            error=f"python check failed (rc={rc}): {stderr.strip()[:200]}",
        )

    sys_prefix = stdout.strip()
    details["sys_prefix"] = sys_prefix
    logger.debug("venv: sys.prefix=%s", sys_prefix)

    # ── Step 3: Check required packages ───────────────────────────────────
    packages = _REQUIRED_PACKAGES.get(node_type, [])
    details["packages_checked"] = packages
    missing: list[str] = []

    for pkg in packages:
        try:
            rc_pkg, _, _ = await run_in_venv(
                f"python -c 'import {pkg}'",
                node,
                timeout=min(10.0, timeout),
            )
            if rc_pkg != 0:
                missing.append(pkg)
                logger.debug("venv: package '%s' not importable (rc=%d)", pkg, rc_pkg)
        except Exception as exc:
            missing.append(pkg)
            logger.debug("venv: package '%s' check raised: %s", pkg, exc)

    details["packages_missing"] = missing
    latency_ms = round((time.perf_counter() - start_ts) * 1000, 2)

    if missing:
        logger.warning("venv: missing packages: %s", missing)
        return ProbeResult(
            name="venv",
            node=node_type,
            healthy=False,
            latency_ms=latency_ms,
            details=details,
            error=f"Missing packages: {missing}",
        )

    details["packages_ok"] = packages
    logger.info("venv: healthy (prefix=%s)", sys_prefix)
    return ProbeResult(
        name="venv",
        node=node_type,
        healthy=True,
        latency_ms=latency_ms,
        details=details,
    )
