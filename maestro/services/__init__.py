"""
Maestro Services — Health Check Integrations + Service Lifecycle Management

Provides async health-check functions for every UBIK service, plus
shared data models and the concurrent health runner.  Also exposes the
new UbikService abstraction and ServiceRegistry for lifecycle management.

Public API:
    Models:
        ServiceStatus   — HEALTHY | DEGRADED | UNHEALTHY enum
        ServiceResult   — single check outcome (status, latency, details)
        ClusterHealth   — aggregated snapshot across all services
        ProbeResult     — structured result from UbikService.probe()

    Individual checkers (legacy async functions):
        check_neo4j       — Neo4j graph DB (Bolt protocol)
        check_chromadb    — ChromaDB vector store (HTTP)
        check_mcp         — MCP server (HTTP liveness)
        check_vllm        — vLLM inference server (OpenAI-compat HTTP)
        check_tailscale   — Tailscale mesh network (CLI)
        check_docker      — Docker daemon + containers (CLI)

    Runner:
        run_all_checks  — concurrent check orchestrator → ClusterHealth

    Service classes (UbikService lifecycle management):
        DockerService     — Docker daemon (Hippocampal)
        Neo4jService      — Neo4j graph database (Hippocampal)
        ChromaDbService   — ChromaDB vector store (Hippocampal)
        McpServerService  — FastMCP server (Hippocampal)
        VllmService       — vLLM inference server (Somatic)
        ServiceRegistry   — Registry with topological startup ordering

Usage:
    import asyncio
    from maestro.config import get_config
    from maestro.services import run_all_checks, ServiceRegistry

    # Legacy health checks
    cluster = asyncio.run(run_all_checks(get_config()))
    print(cluster.to_json())

    # Service lifecycle management
    registry = ServiceRegistry()
    for svc in registry.get_startup_order():
        print(svc.name, svc.depends_on)
"""

from collections import deque
from typing import Optional

from .base import ProbeResult, UbikService
from .chromadb_check import check_chromadb
from .chromadb_service import ChromaDbService
from .docker_check import check_docker
from .docker_service import DockerService
from .health_runner import ALL_SERVICE_NAMES, run_all_checks, run_selected_checks
from .mcp_check import check_mcp
from .mcp_server_service import McpServerService
from .models import ClusterHealth, ServiceResult, ServiceStatus
from .neo4j_check import check_neo4j
from .neo4j_service import Neo4jService
from .tailscale_check import check_tailscale
from .venv_service import (
    check_venv_health,
    detect_active_venv,
    get_venv_run_prefix,
    run_in_venv,
)
from .vllm_check import check_vllm
from .vllm_service import VllmService


# ---------------------------------------------------------------------------
# ServiceRegistry
# ---------------------------------------------------------------------------

class ServiceRegistry:
    """Registry of all UBIK service instances with dependency ordering.

    Instantiates and registers all five service classes in one place.
    Provides helpers for filtering by node and computing a safe startup
    order via topological sort (Kahn's algorithm).

    Args:
        cfg: Application configuration.  When ``None``, :func:`get_config`
            is called on first access.

    Example::

        registry = ServiceRegistry()
        for svc in registry.get_startup_order():
            print(svc.name, "->", svc.depends_on)
    """

    def __init__(self, cfg=None) -> None:
        if cfg is None:
            from maestro.config import get_config
            cfg = get_config()
        self._cfg = cfg
        self._services: list[UbikService] = []
        self._register_all()

    def _register_all(self) -> None:
        """Instantiate and register all service classes."""
        ubik_root = self._cfg.ubik_root
        self.register(DockerService())
        self.register(Neo4jService(ubik_root))
        self.register(ChromaDbService(
            ubik_root,
            port=self._cfg.hippocampal.chromadb_port,
            token=self._cfg.hippocampal.chromadb_token,
        ))
        self.register(McpServerService(
            ubik_root,
            port=self._cfg.hippocampal.mcp_port,
        ))
        self.register(VllmService(
            port=self._cfg.somatic.vllm_port,
            model_path=self._cfg.somatic.vllm_model_path,
        ))

    @property
    def cfg(self):
        """The application configuration used to instantiate this registry."""
        return self._cfg

    def register(self, service: UbikService) -> None:
        """Add a service to the registry.

        Args:
            service: Any :class:`~maestro.services.base.UbikService` instance.
        """
        self._services.append(service)

    def get_all(self) -> list[UbikService]:
        """Return all registered services in registration order.

        Returns:
            List of all :class:`~maestro.services.base.UbikService` instances.
        """
        return list(self._services)

    def get_services_for_node(self, node_type) -> list[UbikService]:
        """Filter services by the node they run on.

        Args:
            node_type: :class:`~maestro.platform_detect.NodeType` value.

        Returns:
            Services whose ``node`` property matches *node_type*.
        """
        return [s for s in self._services if s.node == node_type]

    def get_startup_order(self) -> list[UbikService]:
        """Return services in dependency-safe startup order (Kahn's algorithm).

        Services with no dependencies come first; services that depend on
        others come after all their dependencies.

        Returns:
            All registered services ordered so that every service appears
            after its dependencies.

        Raises:
            ValueError: If a circular dependency is detected.
        """
        name_to_svc: dict[str, UbikService] = {s.name: s for s in self._services}

        # Build in-degree counts and reverse adjacency map
        in_degree: dict[str, int] = {s.name: 0 for s in self._services}
        dependents: dict[str, list[str]] = {s.name: [] for s in self._services}

        for svc in self._services:
            for dep in svc.depends_on:
                if dep in name_to_svc:
                    in_degree[svc.name] += 1
                    dependents[dep].append(svc.name)

        # Start with services that have no outstanding dependencies
        queue: deque[UbikService] = deque(
            s for s in self._services if in_degree[s.name] == 0
        )
        result: list[UbikService] = []

        while queue:
            svc = queue.popleft()
            result.append(svc)
            for dependent_name in dependents[svc.name]:
                in_degree[dependent_name] -= 1
                if in_degree[dependent_name] == 0:
                    queue.append(name_to_svc[dependent_name])

        if len(result) != len(self._services):
            raise ValueError(
                "Circular dependency detected in service graph. "
                f"Processed {len(result)}/{len(self._services)} services."
            )
        return result


__all__ = [
    # Models
    "ClusterHealth",
    "ProbeResult",
    "ServiceResult",
    "ServiceStatus",
    "UbikService",
    # Legacy checkers
    "check_chromadb",
    "check_docker",
    "check_mcp",
    "check_neo4j",
    "check_tailscale",
    "check_vllm",
    # Runner
    "ALL_SERVICE_NAMES",
    "run_all_checks",
    "run_selected_checks",
    # Service classes
    "ChromaDbService",
    "DockerService",
    "McpServerService",
    "Neo4jService",
    "ServiceRegistry",
    "VllmService",
    # Venv service
    "check_venv_health",
    "detect_active_venv",
    "get_venv_run_prefix",
    "run_in_venv",
]
