#!/usr/bin/env python3
"""
Maestro — vLLM Service

Probes and manages the vLLM inference server on the Somatic node.

probe:  HTTP GET to ``http://{host}:8002/health`` — 200 means the server
        process is accepting requests.
start:  Launches vllm serve in the configured Python environment. Supports
        both conda environments and standard venvs:
        - Conda: ``conda run -n pytorch_env vllm serve ...``
        - Venv:  ``bash -c "source venv/bin/activate && vllm serve ..."``
        Flags are read from config/models/vllm_config.yaml (dtype, quantization,
        gpu_memory_utilization, etc.). RTX 5090 / Blackwell env vars are set
        automatically (VLLM_FLASH_ATTN_VERSION=2, etc.).
        Model loading typically takes 30-120s, so max_wait_s=120s.
stop:   SIGTERM to whatever process is listening on port 8002.

vLLM runs ONLY on the Somatic node.  Never call start() on the
Hippocampal node.

Author: UBIK Project
Version: 0.4.0
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
from maestro.services.base import ProbeResult, UbikService, _kill_port

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8002
_DEFAULT_CONDA_ENV = "pytorch_env"
_DEFAULT_MAX_WAIT_S = 120.0


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
        Path.home() / "opt" / "miniconda3" / "bin" / "conda",
        Path.home() / "opt" / "anaconda3" / "bin" / "conda",
        Path("/opt/conda/bin/conda"),
        Path("/usr/local/conda/bin/conda"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def _detect_venv_path() -> Optional[Path]:
    """Locate the Python venv that has vLLM installed.

    Resolution order:
      1. ``VLLM_VENV_PATH`` environment variable (explicit override).
      2. Known somatic ML venv candidates (e.g. ``~/pytorch_env``).
      3. ``VIRTUAL_ENV`` environment variable — skipped if the venv does
         not contain a ``vllm`` binary (avoids picking up the maestro
         management venv which does not run inference workloads).

    Returns:
        Path to the venv directory, or None if no suitable venv is found.
    """
    # 1. Explicit override
    explicit = os.environ.get("VLLM_VENV_PATH")
    if explicit:
        path = Path(explicit)
        if path.exists() and (path / "bin" / "activate").exists():
            return path

    # 2. Known somatic ML venv candidates (preferred over VIRTUAL_ENV)
    candidates = [
        Path.home() / "pytorch_env",
        Path("/home/gasu/pytorch_env"),
    ]
    for candidate in candidates:
        if (
            candidate.exists()
            and (candidate / "bin" / "activate").exists()
            and (candidate / "bin" / "vllm").exists()
        ):
            return candidate

    # 3. Fall back to VIRTUAL_ENV only if that venv has vllm
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        path = Path(venv)
        if (
            path.exists()
            and (path / "bin" / "activate").exists()
            and (path / "bin" / "vllm").exists()
        ):
            return path

    return None


def _build_vllm_serve_args(ubik_root: Path) -> list[str]:
    """Build extra CLI flags for ``vllm serve`` from vllm_config.yaml.

    Reads ``{ubik_root}/config/models/vllm_config.yaml`` and returns a list
    of argument strings suitable for appending to the ``vllm serve`` command.
    Falls back to hardcoded defaults matching the DeepSeek-R1 AWQ model when
    the config file is absent or unreadable.

    Args:
        ubik_root: Absolute path to the UBIK project root on this node.

    Returns:
        List of CLI argument strings (e.g. ``["--dtype", "float16", ...]``).
    """
    model_cfg: dict = {
        "dtype": "float16",
        "quantization": "awq_marlin",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 98304,
        "trust_remote_code": True,
    }
    engine_cfg: dict = {
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "max_num_seqs": 128,
    }

    config_path = ubik_root / "config" / "models" / "vllm_config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            model_cfg.update(raw.get("model", {}))
            engine_cfg.update(raw.get("engine", {}))
        except Exception as exc:
            logger.warning(
                "vllm: failed to read %s: %s — using defaults", config_path, exc
            )

    flags: list[str] = [
        "--host", "0.0.0.0",
        "--dtype", str(model_cfg["dtype"]),
        "--tensor-parallel-size", str(int(model_cfg["tensor_parallel_size"])),
        "--gpu-memory-utilization", str(float(model_cfg["gpu_memory_utilization"])),
        "--max-model-len", str(int(model_cfg["max_model_len"])),
        "--max-num-seqs", str(int(engine_cfg["max_num_seqs"])),
    ]
    if model_cfg.get("quantization"):
        flags += ["--quantization", str(model_cfg["quantization"])]
    if model_cfg.get("trust_remote_code"):
        flags.append("--trust-remote-code")
    if engine_cfg.get("enable_prefix_caching"):
        flags.append("--enable-prefix-caching")
    if engine_cfg.get("enable_chunked_prefill"):
        flags.append("--enable-chunked-prefill")

    return flags


# Environment variables required for RTX 5090 / Blackwell (SM120) compatibility.
# FA3 does not yet support SM120; FA2 must be forced.
_BLACKWELL_ENV: dict[str, str] = {
    "VLLM_FLASH_ATTN_VERSION": "2",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "CUDA_MODULE_LOADING": "LAZY",
}


class VllmService(UbikService):
    """vLLM server health probe and lifecycle manager.

    Supports two Python environment types:
      - **Conda**: Uses ``conda run -n <env> vllm serve ...``
      - **Venv**: Uses ``bash -c "source <path>/bin/activate && vllm serve ..."``

    Environment selection priority:
      1. Explicit ``venv_path`` parameter
      2. ``VIRTUAL_ENV`` environment variable (auto-detected)
      3. Conda with ``conda_env`` name (requires conda to be installed)
    """

    def __init__(
        self,
        *,
        port: int = _DEFAULT_PORT,
        model_path: Optional[str] = None,
        conda_env: str = _DEFAULT_CONDA_ENV,
        venv_path: Optional[Path] = None,
        max_wait_s: float = _DEFAULT_MAX_WAIT_S,
    ) -> None:
        if model_path is None:
            from maestro.config import get_config
            model_path = get_config().somatic.vllm_model_path
        self._port = port
        self._model_path = model_path
        self._conda_env = conda_env
        self._venv_path = venv_path
        self._max_wait_s = max_wait_s

    @property
    def max_wait_s(self) -> float:
        return self._max_wait_s

    @property
    def name(self) -> str:
        return "vllm"

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
            ``details["http_status"]`` when reachable.
        """
        url = f"http://{host}:{self._port}/health"
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.debug("vllm: HTTP %d from %s", resp.status_code, url)
                healthy = resp.status_code == 200
                return ProbeResult(
                    name=self.name, node=self.node, healthy=healthy,
                    latency_ms=latency_ms,
                    details={"http_status": resp.status_code, "url": url},
                    error=None if healthy else f"HTTP {resp.status_code}",
                )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return ProbeResult(
                name=self.name, node=self.node, healthy=False,
                latency_ms=latency_ms,
                details={"url": url},
                error=str(exc),
            )

    async def start(self, ubik_root: Path) -> bool:
        """Launch vLLM and wait for it to become healthy.

        Pre-flight checks:
          - Verifies this is the Somatic node.

        Starts vLLM in the appropriate Python environment (venv or conda),
        passing the full set of model/engine flags from vllm_config.yaml and
        setting RTX 5090 / Blackwell compatibility environment variables.
        Polls ``probe()`` every 3 seconds until the server is healthy or
        ``max_wait_s`` is exceeded. Model loading typically takes 30-120s.

        Args:
            ubik_root: Path to the UBIK project root; used to locate
                ``config/models/vllm_config.yaml``.

        Returns:
            ``True`` when vLLM is confirmed healthy; ``False`` on error or
            timeout.
        """
        # Pre-flight: node check (vLLM must run on Somatic)
        identity = detect_node()
        if identity.node_type not in (NodeType.SOMATIC, NodeType.UNKNOWN):
            logger.error(
                "vllm: refusing to start on %s node "
                "(service belongs to somatic)",
                identity.node_type.value,
            )
            return False

        # Build model/engine flags from config (with AWQ-safe defaults)
        extra_flags = _build_vllm_serve_args(ubik_root)

        # Subprocess environment: inherit current env, then overlay Blackwell vars
        env = dict(os.environ)
        for key, val in _BLACKWELL_ENV.items():
            env.setdefault(key, val)

        # Determine which environment to use: venv or conda
        venv_path = self._venv_path or _detect_venv_path()
        conda = _find_conda()

        # Log file for vLLM startup output — avoids DEVNULL which can cause
        # the APIServer process to crash in some asyncio/WSL2 environments.
        log_path = Path("/tmp/vllm_startup.log")

        try:
            log_fh = open(log_path, "a")  # noqa: WPS515
        except OSError:
            log_fh = None  # fall back silently

        try:
            if venv_path:
                # Use standard virtualenv
                activate = venv_path / "bin" / "activate"
                vllm_cmd = shlex.join([
                    "vllm", "serve", self._model_path,
                    "--port", str(self._port),
                    *extra_flags,
                ])
                cmd = f"source {shlex.quote(str(activate))} && {vllm_cmd}"
                logger.info(
                    "vllm: launching vllm serve %s --port %d (venv: %s, log: %s)",
                    self._model_path, self._port, venv_path, log_path,
                )
                await asyncio.create_subprocess_exec(
                    "bash", "-c", cmd,
                    env=env,
                    stdout=log_fh or asyncio.subprocess.DEVNULL,
                    stderr=log_fh or asyncio.subprocess.DEVNULL,
                    start_new_session=True,
                )
            elif conda:
                # Use conda environment
                logger.info(
                    "vllm: launching vllm serve %s --port %d (conda env: %s, log: %s)",
                    self._model_path, self._port, self._conda_env, log_path,
                )
                await asyncio.create_subprocess_exec(
                    conda, "run", "-n", self._conda_env,
                    "vllm", "serve", self._model_path,
                    "--port", str(self._port),
                    *extra_flags,
                    env=env,
                    stdout=log_fh or asyncio.subprocess.DEVNULL,
                    stderr=log_fh or asyncio.subprocess.DEVNULL,
                    start_new_session=True,
                )
            else:
                logger.error(
                    "vllm: no Python environment available "
                    "(neither venv nor conda found)"
                )
                return False
        except Exception as exc:
            logger.warning("vllm: start command failed: %s", exc)
            return False
        finally:
            if log_fh:
                log_fh.close()

        return await self._wait_for_healthy("localhost")

    async def stop(self) -> bool:
        """SIGTERM whatever process is listening on the vLLM port.

        Returns:
            ``True`` if a signal was sent; ``False`` if no process was found.
        """
        logger.debug("vllm: killing process on port %d", self._port)
        return await _kill_port(self._port)
