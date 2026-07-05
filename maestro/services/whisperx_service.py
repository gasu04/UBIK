#!/usr/bin/env python3
"""
Maestro — WhisperX Transcription Service

Probes and manages the WhisperX FastAPI server on the Somatic node.

probe:  HTTP GET to ``http://{host}:9100/health`` — 200 + model_loaded=True
        means the server is accepting transcription requests.
start:  Launches ``whisperx_server.py`` in the conda pytorch_env (or venv
        fallback).  The server script is expected at
        ``{ubik_root}/somatic/whisperx_server.py``.
        Model loading typically takes 30-90 s, so max_wait_s defaults to 90 s.
stop:   SIGTERM to whatever process is listening on port 9100.

WhisperX runs ONLY on the Somatic node. Never call start() on the
Hippocampal node.

Author: UBIK Project
Version: 0.1.0
"""

import asyncio
import logging
import os
import shlex
import shutil
import time
from pathlib import Path
from typing import Optional

import httpx

from maestro.platform_detect import NodeType, detect_node
from maestro.remote import RemoteExecutor
from maestro.services.base import ProbeResult, UbikService, _kill_port

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 9100
_DEFAULT_CONDA_ENV = "pytorch_env"
_DEFAULT_MAX_WAIT_S = 90.0
_REMOTE_STOP_GRACE_S = 20

# Transient systemd *user* unit for the remote WhisperX server.
_WHISPERX_UNIT = "ubik-whisperx"

# Bash to reach the lingering user manager's bus from a non-login SSH shell.
_USER_SYSTEMD_ENV = (
    'export XDG_RUNTIME_DIR="/run/user/$(id -u)"\n'
    'export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"'
)


def _find_conda() -> Optional[str]:
    """Locate the conda executable.

    Checks CONDA_EXE, PATH, then common installation locations.

    Returns:
        Path to conda executable, or None if not found.
    """
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    if shutil.which("conda"):
        return "conda"

    candidates = [
        Path.home() / "miniconda3" / "bin" / "conda",
        Path.home() / "anaconda3" / "bin" / "conda",
        Path("/opt/conda/bin/conda"),
        Path("/usr/local/conda/bin/conda"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


class WhisperXService(UbikService):
    """WhisperX transcription server health probe and lifecycle manager.

    The server process is the FastAPI app at
    ``{ubik_root}/somatic/whisperx_server.py``.  It exposes
    ``POST /transcribe`` and ``GET /health`` on port 9100.

    Environment selection priority:
      1. Explicit ``venv_path`` parameter
      2. Conda with ``conda_env`` name (requires conda to be installed)
    """

    def __init__(
        self,
        *,
        port: int = _DEFAULT_PORT,
        conda_env: str = _DEFAULT_CONDA_ENV,
        venv_path: Optional[Path] = None,
        max_wait_s: float = _DEFAULT_MAX_WAIT_S,
        remote: Optional[RemoteExecutor] = None,
        remote_ubik_root: Optional[str] = None,
        probe_ip: Optional[str] = None,
    ) -> None:
        self._port = port
        self._conda_env = conda_env
        self._venv_path = venv_path
        self._max_wait_s = max_wait_s
        # Remote-control context (see VllmService for the rationale).
        self._remote = remote
        self._remote_ubik_root = remote_ubik_root
        self._probe_ip = probe_ip

    def _is_remote(self) -> bool:
        """Whether this service must be controlled over SSH from another node."""
        if self._remote is None:
            return False
        local = detect_node().node_type
        return local != NodeType.UNKNOWN and local != self.node

    def _lifecycle_probe_host(self) -> str:
        """Host to poll for health after a start/stop (local vs remote)."""
        if self._is_remote():
            return self._probe_ip or self.node.value
        return "localhost"

    @property
    def max_wait_s(self) -> float:
        return self._max_wait_s

    @property
    def name(self) -> str:
        return "whisperx"

    @property
    def node(self) -> NodeType:
        return NodeType.SOMATIC

    @property
    def ports(self) -> list[int]:
        return [self._port]

    @property
    def depends_on(self) -> list[str]:
        return []

    async def probe(self, host: str) -> ProbeResult:
        """HTTP GET to ``/health``.

        Args:
            host: IP address or hostname of the Somatic node.

        Returns:
            :class:`~maestro.services.base.ProbeResult` with
            ``details["model_loaded"]`` and ``details["http_status"]``
            when reachable.
        """
        url = f"http://{host}:{self._port}/health"
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                body = {}
                try:
                    body = resp.json()
                except Exception:
                    pass
                model_loaded = body.get("model_loaded", False)
                healthy = resp.status_code == 200 and model_loaded
                return ProbeResult(
                    name=self.name,
                    node=self.node,
                    healthy=healthy,
                    latency_ms=latency_ms,
                    details={
                        "http_status": resp.status_code,
                        "model_loaded": model_loaded,
                        "url": url,
                    },
                    error=None if healthy else (
                        "Model not yet loaded" if resp.status_code == 200
                        else f"HTTP {resp.status_code}"
                    ),
                )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return ProbeResult(
                name=self.name,
                node=self.node,
                healthy=False,
                latency_ms=latency_ms,
                details={"url": url},
                error=str(exc),
            )

    async def start(self, ubik_root: Path) -> bool:
        """Launch the WhisperX server and wait for it to become healthy.

        The server script is expected at
        ``{ubik_root}/somatic/whisperx_server.py``.

        Args:
            ubik_root: Path to the UBIK project root on this node.

        Returns:
            ``True`` when the server is confirmed healthy; ``False`` on error
            or timeout.
        """
        # Remote path: Maestro is on another node — start WhisperX over SSH.
        if self._is_remote():
            return await self._remote_start()

        identity = detect_node()
        if identity.node_type not in (NodeType.SOMATIC, NodeType.UNKNOWN):
            logger.error(
                "whisperx: refusing to start on %s node "
                "(service belongs to somatic)",
                identity.node_type.value,
            )
            return False

        server_script = ubik_root / "somatic" / "whisperx_server.py"
        if not server_script.exists():
            logger.error("whisperx: server script not found at %s", server_script)
            return False

        # If a server process is already listening on the port (e.g. started by
        # systemd before maestro ran), skip launching a duplicate and just wait
        # for the model to finish loading.
        try:
            async with httpx.AsyncClient(timeout=3.0) as _pre:
                r = await _pre.get(f"http://localhost:{self._port}/health")
                if r.status_code == 200:
                    logger.info(
                        "whisperx: server already listening on port %d — "
                        "waiting for model to load without re-launching",
                        self._port,
                    )
                    return await self._wait_for_healthy("localhost")
        except Exception:
            pass  # nothing on port yet, proceed with launch

        log_path = Path("/tmp/whisperx_startup.log")
        try:
            log_fh = open(log_path, "a")  # noqa: WPS515
        except OSError:
            log_fh = None

        env = {**os.environ, "WHISPERX_PORT": str(self._port)}

        try:
            venv_path = self._venv_path
            conda = _find_conda()

            # Fallback: the ubik project venv always ships whisperx on Somatic.
            # Used when no explicit venv_path was passed and conda is not installed.
            if not venv_path and not conda:
                candidate = ubik_root / "venv"
                if (candidate / "bin" / "python").exists():
                    venv_path = candidate
                    logger.info("whisperx: using ubik venv at %s", venv_path)

            if venv_path and (venv_path / "bin" / "python").exists():
                python = str(venv_path / "bin" / "python")
                logger.info(
                    "whisperx: launching %s (venv: %s, log: %s)",
                    server_script, venv_path, log_path,
                )
                await asyncio.create_subprocess_exec(
                    python, str(server_script),
                    env=env,
                    stdout=log_fh or asyncio.subprocess.DEVNULL,
                    stderr=log_fh or asyncio.subprocess.DEVNULL,
                    start_new_session=True,
                )
            elif conda:
                logger.info(
                    "whisperx: launching %s (conda env: %s, log: %s)",
                    server_script, self._conda_env, log_path,
                )
                await asyncio.create_subprocess_exec(
                    conda, "run", "-n", self._conda_env,
                    "python", str(server_script),
                    env=env,
                    stdout=log_fh or asyncio.subprocess.DEVNULL,
                    stderr=log_fh or asyncio.subprocess.DEVNULL,
                    start_new_session=True,
                )
            else:
                logger.error(
                    "whisperx: no Python environment available "
                    "(neither venv nor conda found)"
                )
                return False
        except Exception as exc:
            logger.warning("whisperx: start command failed: %s", exc)
            return False
        finally:
            if log_fh:
                log_fh.close()

        return await self._wait_for_healthy("localhost")

    async def _remote_start(self) -> bool:
        """Start WhisperX on the Somatic node over SSH as a systemd user unit.

        Registers ``somatic/whisperx_server.py`` (run in ``{root}/venv``) as a
        transient systemd *user* unit (``ubik-whisperx``) so it survives the
        SSH session — a plain ``nohup`` would be torn down when ``wsl.exe``
        exits.  Then polls ``/health`` over Tailscale.

        Returns:
            ``True`` when WhisperX becomes healthy within :attr:`max_wait_s`.
        """
        root = self._remote_ubik_root or "/home/gasu/ubik"
        server = f"{root}/somatic/whisperx_server.py"
        python = f"{root}/venv/bin/python"
        script = f"""
set -u
{_USER_SYSTEMD_ENV}
SERVER={shlex.quote(server)}
PYTHON={shlex.quote(python)}
if [ ! -f "$SERVER" ]; then echo "MISSING_SERVER:$SERVER"; exit 3; fi
if [ ! -x "$PYTHON" ]; then echo "MISSING_PYTHON:$PYTHON"; exit 3; fi
mkdir -p {shlex.quote(root)}/logs
if systemctl --user is-active --quiet {_WHISPERX_UNIT}; then echo "ALREADY_ACTIVE"; exit 0; fi
if curl -s -o /dev/null --max-time 3 http://localhost:{self._port}/health; then
    echo "ALREADY_RUNNING_UNMANAGED"; exit 0
fi
systemctl --user reset-failed {_WHISPERX_UNIT} 2>/dev/null || true
systemd-run --user --unit={_WHISPERX_UNIT} \
    --property=Type=simple \
    --property=KillSignal=SIGTERM \
    --property=KillMode=mixed \
    --property=TimeoutStopSec={_REMOTE_STOP_GRACE_S} \
    --setenv=WHISPERX_PORT={self._port} \
    "$PYTHON" "$SERVER" 2>&1
echo "STARTED_UNIT={_WHISPERX_UNIT} rc=$?"
"""
        logger.info("whisperx: remote start on %s as user unit %s", self._remote.ssh_host, _WHISPERX_UNIT)
        res = await self._remote.run(script, timeout=30.0)
        if not res.connected:
            logger.error("whisperx: remote start failed — cannot reach Somatic: %s", res.stderr.strip()[:200])
            return False
        if "MISSING_" in res.stdout:
            logger.error("whisperx: remote prerequisite missing: %s", res.stdout.strip())
            return False
        logger.info("whisperx: remote launch — %s", res.stdout.replace("\n", " ").strip())
        return await self._wait_for_healthy(self._lifecycle_probe_host())

    async def _remote_stop(self) -> bool:
        """Stop WhisperX on the Somatic node over SSH (graceful via systemd).

        Returns:
            ``True`` if the server is confirmed down.
        """
        script = f"""
set -u
{_USER_SYSTEMD_ENV}
if systemctl --user is-active --quiet {_WHISPERX_UNIT}; then
    echo "STOPPING_UNIT"
    systemctl --user stop {_WHISPERX_UNIT} 2>&1
    echo "STOP_RC=$?"
else
    echo "NO_UNIT_FALLBACK"
    pkill -TERM -f "whisperx_server.py" 2>/dev/null || true
    for i in $(seq 1 {_REMOTE_STOP_GRACE_S}); do
        pgrep -f "whisperx_server.py" >/dev/null 2>&1 || break
        sleep 1
    done
    pkill -9 -f "whisperx_server.py" 2>/dev/null || true
fi
systemctl --user reset-failed {_WHISPERX_UNIT} 2>/dev/null || true
echo "DONE"
"""
        logger.info("whisperx: remote stop on %s (systemctl --user stop)", self._remote.ssh_host)
        res = await self._remote.run(script, timeout=_REMOTE_STOP_GRACE_S + 20.0)
        if not res.connected:
            logger.error("whisperx: remote stop failed — cannot reach Somatic: %s", res.stderr.strip()[:200])
            return False
        logger.info("whisperx: remote stop result — %s", res.stdout.replace("\n", " ").strip())
        verify = await self.probe_with_timeout(self._lifecycle_probe_host(), timeout=5.0)
        return not verify.healthy

    async def stop(self) -> bool:
        """SIGTERM whatever process is listening on the WhisperX port.

        Returns:
            ``True`` if a signal was sent; ``False`` if no process was found.
        """
        # Remote path: Maestro is on another node — stop WhisperX over SSH.
        if self._is_remote():
            return await self._remote_stop()

        logger.debug("whisperx: killing process on port %d", self._port)
        return await _kill_port(self._port)
