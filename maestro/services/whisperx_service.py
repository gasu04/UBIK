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
import shutil
import time
from pathlib import Path
from typing import Optional

import httpx

from maestro.platform_detect import NodeType, detect_node
from maestro.services.base import ProbeResult, UbikService, _kill_port

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 9100
_DEFAULT_CONDA_ENV = "pytorch_env"
_DEFAULT_MAX_WAIT_S = 90.0


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
    ) -> None:
        self._port = port
        self._conda_env = conda_env
        self._venv_path = venv_path
        self._max_wait_s = max_wait_s

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

        log_path = Path("/tmp/whisperx_startup.log")
        try:
            log_fh = open(log_path, "a")  # noqa: WPS515
        except OSError:
            log_fh = None

        env = {**os.environ, "WHISPERX_PORT": str(self._port)}

        try:
            venv_path = self._venv_path
            conda = _find_conda()

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

    async def stop(self) -> bool:
        """SIGTERM whatever process is listening on the WhisperX port.

        Returns:
            ``True`` if a signal was sent; ``False`` if no process was found.
        """
        logger.debug("whisperx: killing process on port %d", self._port)
        return await _kill_port(self._port)
