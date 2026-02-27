#!/usr/bin/env python3
"""
Maestro — Usage Metrics Collector

Gathers best-effort usage statistics for the UBIK system at the end of
each daemon check cycle.  All metrics are Optional (None when unavailable)
so a failure in any single collector never blocks the others or the caller.

Collected metrics:
    Storage:   ChromaDB collection sizes (ubik_episodic, ubik_semantic)
               Neo4j node and relationship counts
               Disk usage of the UBIK_ROOT partition
    Inference: vLLM /health liveness probe
    GPU:       nvidia-smi utilisation + VRAM used (Somatic node only)

Design notes:
    - Every sub-collector wraps its body in a broad except clause.  A
      failure is logged at DEBUG level and the field is left as None.
    - asyncio.gather(return_exceptions=True) ensures a surprise exception
      from any coroutine cannot surface to the caller.
    - GPU collection is gated on NodeType.SOMATIC so nvidia-smi is never
      invoked on the Hippocampal (macOS) node.
    - ChromaDB and Neo4j queries follow the same auth/URL logic as the
      existing health checkers (chromadb_check.py, neo4j_check.py).

Usage::

    import asyncio
    from maestro.metrics import MetricsCollector
    from maestro.orchestrator import Orchestrator
    from maestro.services import ServiceRegistry
    from maestro.platform_detect import detect_node

    orch = Orchestrator(ServiceRegistry(), detect_node())
    collector = MetricsCollector()
    metrics = asyncio.run(collector.collect(orch))
    print(collector.format_report(metrics))

Author: UBIK Project
Version: 0.1.0
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

from maestro.orchestrator import Orchestrator
from maestro.platform_detect import NodeType

logger = logging.getLogger(__name__)

# ChromaDB collection names tracked by UBIK.
_EPISODIC_COLLECTION = "ubik_episodic"
_SEMANTIC_COLLECTION = "ubik_semantic"

# ChromaDB v2 default tenant/database (mirrors chromadb_check.py).
_V2_TENANT = "default_tenant"
_V2_DATABASE = "default_database"

# Cypher queries — each returns a single ``count`` column.
_CYPHER_NODES = "MATCH (n) RETURN count(n) AS count"
_CYPHER_RELS = "MATCH ()-[r]->() RETURN count(r) AS count"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class UsageMetrics:
    """Snapshot of UBIK system usage statistics.

    All numeric fields are Optional.  ``None`` means the metric could not
    be collected this cycle (service unreachable, tool absent, etc.).
    Consumers must handle ``None`` gracefully.

    Attributes:
        timestamp: UTC datetime when collection ran.
        chromadb_episodic_count: Document count in the ``ubik_episodic``
            collection; None when ChromaDB is unreachable.
        chromadb_semantic_count: Document count in the ``ubik_semantic``
            collection; None when ChromaDB is unreachable.
        neo4j_node_count: Total nodes in the Neo4j graph; None when
            Neo4j is unreachable or the neo4j package is missing.
        neo4j_relationship_count: Total relationships in the Neo4j graph.
        vllm_running: True when vLLM ``/health`` returns HTTP 200.
        gpu_utilization_pct: GPU core utilisation in [0, 100]; Somatic
            node only.  None on other nodes or when nvidia-smi fails.
        gpu_memory_used_mb: GPU VRAM used in MiB; Somatic node only.
        disk_usage_ubik_gb: Disk space used on the UBIK_ROOT partition
            in GiB (via shutil.disk_usage — reflects the whole partition,
            not just the UBIK directory).
    """

    timestamp: datetime
    chromadb_episodic_count: Optional[int] = None
    chromadb_semantic_count: Optional[int] = None
    neo4j_node_count: Optional[int] = None
    neo4j_relationship_count: Optional[int] = None
    vllm_running: bool = False
    gpu_utilization_pct: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    disk_usage_ubik_gb: Optional[float] = None


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Pulls best-effort usage metrics from UBIK service APIs.

    All sub-collectors run concurrently via :func:`asyncio.gather`.  Any
    individual failure (network error, missing binary, auth failure) is
    logged at DEBUG level and leaves that field as None.  The remaining
    collectors are unaffected.

    Example::

        collector = MetricsCollector()
        metrics = await collector.collect(orchestrator)
        print(collector.format_report(metrics))
    """

    # ── Public API ──────────────────────────────────────────────────────────

    async def collect(self, orchestrator: Orchestrator) -> UsageMetrics:
        """Collect all metrics and return a populated :class:`UsageMetrics`.

        Queries ChromaDB collection sizes, Neo4j graph counts, vLLM
        liveness, GPU utilisation (Somatic only), and disk usage
        concurrently.  Never raises.

        Args:
            orchestrator: Running :class:`~maestro.orchestrator.Orchestrator`
                that provides access to service configuration and node
                identity.

        Returns:
            :class:`UsageMetrics` with every field populated on a
            best-effort basis; unavailable metrics are ``None``.
        """
        cfg = orchestrator._registry.cfg
        identity = orchestrator._identity
        node_type = identity.node_type
        ubik_root = identity.ubik_root

        on_somatic = node_type == NodeType.SOMATIC

        raw = await asyncio.gather(
            self._chromadb_count(cfg, _EPISODIC_COLLECTION),
            self._chromadb_count(cfg, _SEMANTIC_COLLECTION),
            self._neo4j_count(cfg, _CYPHER_NODES),
            self._neo4j_count(cfg, _CYPHER_RELS),
            self._vllm_running(cfg),
            self._gpu_utilization() if on_somatic else _noop(None),
            self._gpu_memory()      if on_somatic else _noop(None),
            self._disk_usage(ubik_root),
            return_exceptions=True,
        )

        def _safe(value, default=None):
            """Convert an Exception result from gather into default."""
            if isinstance(value, BaseException):
                logger.debug("metrics: gather caught exception: %s", value)
                return default
            return value

        (
            episodic_count,
            semantic_count,
            neo4j_nodes,
            neo4j_rels,
            vllm_up,
            gpu_util,
            gpu_mem,
            disk_gb,
        ) = [_safe(r) for r in raw]

        return UsageMetrics(
            timestamp=datetime.now(timezone.utc),
            chromadb_episodic_count=episodic_count,
            chromadb_semantic_count=semantic_count,
            neo4j_node_count=neo4j_nodes,
            neo4j_relationship_count=neo4j_rels,
            vllm_running=bool(vllm_up),
            gpu_utilization_pct=gpu_util,
            gpu_memory_used_mb=gpu_mem,
            disk_usage_ubik_gb=disk_gb,
        )

    def format_report(self, metrics: UsageMetrics) -> str:
        """Format a :class:`UsageMetrics` snapshot as a human-readable string.

        Suitable for writing to the Maestro operational log or displaying
        in the dashboard footer.  Lines are aligned in two columns and
        the GPU section is omitted entirely when no GPU data is available.

        Args:
            metrics: Snapshot as returned by :meth:`collect`.

        Returns:
            Multi-line string report (no trailing newline).
        """

        def _i(v: Optional[int]) -> str:
            return str(v) if v is not None else "N/A"

        def _f(v: Optional[float], prec: int = 1, suffix: str = "") -> str:
            return f"{v:.{prec}f}{suffix}" if v is not None else "N/A"

        ts = metrics.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        sep  = "=" * 52
        dash = "-" * 52

        lines: list[str] = [
            sep,
            "UBIK MAESTRO — Usage Metrics",
            f"Collected : {ts}",
            sep,
            "STORAGE",
            dash,
            f"  ChromaDB episodic   : {_i(metrics.chromadb_episodic_count)} vectors",
            f"  ChromaDB semantic   : {_i(metrics.chromadb_semantic_count)} vectors",
            f"  Neo4j nodes         : {_i(metrics.neo4j_node_count)}",
            f"  Neo4j relationships : {_i(metrics.neo4j_relationship_count)}",
            f"  Disk (UBIK_ROOT)    : {_f(metrics.disk_usage_ubik_gb, 2, ' GiB')}",
            "INFERENCE",
            dash,
            f"  vLLM running        : {'yes' if metrics.vllm_running else 'no'}",
        ]

        has_gpu = (
            metrics.gpu_utilization_pct is not None
            or metrics.gpu_memory_used_mb is not None
        )
        if has_gpu:
            lines += [
                "GPU (Somatic)",
                dash,
                f"  Utilisation         : {_f(metrics.gpu_utilization_pct, 1, '%')}",
                f"  VRAM used           : {_f(metrics.gpu_memory_used_mb, 0, ' MiB')}",
            ]

        lines.append(sep)
        return "\n".join(lines)

    # ── Private sub-collectors ───────────────────────────────────────────────

    async def _chromadb_count(
        self,
        cfg,
        collection_name: str,
        *,
        timeout: float = 5.0,
    ) -> Optional[int]:
        """Return the document count for a single ChromaDB collection.

        Tries the v2 ``/count`` endpoint first; falls back to v1.  Returns
        ``None`` on any connection or parsing error.

        Args:
            cfg: :class:`~maestro.config.AppConfig` providing the ChromaDB
                URL and optional bearer token.
            collection_name: Name of the collection to count.
            timeout: HTTP request timeout in seconds.

        Returns:
            Integer document count, or ``None`` on failure.
        """
        try:
            base_url = cfg.hippocampal.chromadb_url
            headers: dict[str, str] = {}
            if cfg.hippocampal.chromadb_token:
                headers["Authorization"] = (
                    f"Bearer {cfg.hippocampal.chromadb_token}"
                )

            async with httpx.AsyncClient(timeout=timeout) as client:
                # Try v2 first
                v2_url = (
                    f"{base_url}/api/v2/tenants/{_V2_TENANT}"
                    f"/databases/{_V2_DATABASE}"
                    f"/collections/{collection_name}/count"
                )
                resp = await client.get(v2_url, headers=headers)

                if resp.status_code == 200:
                    data = resp.json()
                    return int(data) if isinstance(data, (int, float)) else None

                # v2 not supported → try v1
                if resp.status_code in (404, 410):
                    v1_url = (
                        f"{base_url}/api/v1/collections/{collection_name}/count"
                    )
                    v1_resp = await client.get(v1_url, headers=headers)
                    if v1_resp.status_code == 200:
                        data = v1_resp.json()
                        return int(data) if isinstance(data, (int, float)) else None

                logger.debug(
                    "metrics: chromadb count for %r → HTTP %d",
                    collection_name, resp.status_code,
                )
        except Exception as exc:
            logger.debug(
                "metrics: chromadb_count(%r) failed: %s", collection_name, exc
            )
        return None

    async def _neo4j_count(
        self,
        cfg,
        cypher: str,
        *,
        timeout: float = 5.0,
    ) -> Optional[int]:
        """Run a COUNT Cypher query via the Neo4j Bolt driver.

        Args:
            cfg: :class:`~maestro.config.AppConfig` providing the Neo4j
                bolt URL and credentials.
            cypher: Cypher query that returns a single ``count`` column.
            timeout: Connection timeout in seconds.

        Returns:
            Integer count, or ``None`` on connection or auth failure.
        """
        try:
            from neo4j import AsyncGraphDatabase  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("metrics: neo4j package not installed — skipping")
            return None

        driver = None
        try:
            bolt_url = cfg.hippocampal.neo4j_bolt_url
            auth = (
                cfg.hippocampal.neo4j_user,
                cfg.hippocampal.neo4j_password or "",
            )
            driver = AsyncGraphDatabase.driver(
                bolt_url,
                auth=auth,
                connection_timeout=timeout,
                max_connection_lifetime=timeout * 2,
            )
            async with driver.session() as session:
                result = await session.run(cypher)
                record = await result.single()
                return int(record["count"]) if record else 0
        except Exception as exc:
            logger.debug("metrics: neo4j_count(%r) failed: %s", cypher[:40], exc)
        finally:
            if driver is not None:
                try:
                    await driver.close()
                except Exception:
                    pass
        return None

    async def _vllm_running(self, cfg, *, timeout: float = 5.0) -> bool:
        """Return ``True`` if the vLLM ``/health`` endpoint returns HTTP 200.

        Args:
            cfg: :class:`~maestro.config.AppConfig` providing the Somatic
                vLLM base URL.
            timeout: HTTP request timeout in seconds.

        Returns:
            ``True`` when healthy, ``False`` on any error or non-200 status.
        """
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{cfg.somatic.vllm_url}/health")
                return resp.status_code == 200
        except Exception as exc:
            logger.debug("metrics: vllm running check failed: %s", exc)
        return False

    async def _gpu_utilization(self) -> Optional[float]:
        """Query GPU core utilisation (%) via ``nvidia-smi``.

        Only invoked on the Somatic node.  Returns ``None`` when
        ``nvidia-smi`` is absent or returns unexpected output.

        Returns:
            GPU utilisation as a float in [0, 100], or ``None``.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=8.0)
            line = stdout_b.decode(errors="replace").strip().split("\n")[0].strip()
            return float(line)
        except Exception as exc:
            logger.debug("metrics: gpu_utilization failed: %s", exc)
        return None

    async def _gpu_memory(self) -> Optional[float]:
        """Query GPU VRAM used (MiB) via ``nvidia-smi``.

        Only invoked on the Somatic node.  Returns ``None`` when
        ``nvidia-smi`` is absent or returns unexpected output.

        Returns:
            VRAM used in MiB as a float, or ``None``.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=8.0)
            line = stdout_b.decode(errors="replace").strip().split("\n")[0].strip()
            return float(line)
        except Exception as exc:
            logger.debug("metrics: gpu_memory failed: %s", exc)
        return None

    async def _disk_usage(self, ubik_root) -> Optional[float]:
        """Return used disk space on the UBIK_ROOT partition in GiB.

        Uses :func:`shutil.disk_usage` which reflects the whole partition
        that contains *ubik_root* (cross-platform, no subprocess needed).

        Args:
            ubik_root: :class:`~pathlib.Path` to the UBIK project root.

        Returns:
            Used disk space in GiB (float), or ``None`` on failure.
        """
        try:
            usage = shutil.disk_usage(str(ubik_root))
            return usage.used / (1024 ** 3)
        except Exception as exc:
            logger.debug("metrics: disk_usage failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop(value):
    """Async identity — return *value* immediately.

    Used as a stand-in when GPU collection is not applicable on this node,
    keeping the :func:`asyncio.gather` call shape consistent.
    """
    return value
