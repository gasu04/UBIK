"""
Maestro — UBIK Infrastructure Orchestrator

Monitors and manages UBIK services across the Hippocampal and Somatic nodes,
providing health checking, service orchestration, and operational visibility.

Nodes:
    Hippocampal Node — Mac Mini M4 Pro (macOS)
        Services: Neo4j, ChromaDB, MCP server
    Somatic Node — PowerSpec RTX 5090 (WSL2 Linux)
        Services: vLLM inference

Quick start:
    from maestro.config import get_config

    cfg = get_config()
    print(cfg.hippocampal.neo4j_http_url)   # http://100.103.242.91:7474
    print(cfg.somatic.vllm_url)             # http://100.79.166.114:8002
    print(cfg.maestro.log_level)            # INFO
"""

from .config import (
    AppConfig,
    HippocampalConfig,
    MaestroConfig,
    SomaticConfig,
    get_config,
)
from .daemon import MaestroDaemon
from .dashboard import Dashboard, run_dashboard
from .shutdown import ShutdownController
from .log import MaestroLogger
from .logger import configure_logging, get_logger, log_cluster_health
from .services import (
    ALL_SERVICE_NAMES,
    ChromaDbService,
    ClusterHealth,
    DockerService,
    McpServerService,
    Neo4jService,
    ProbeResult,
    ServiceRegistry,
    ServiceResult,
    ServiceStatus,
    UbikService,
    VllmService,
    check_venv_health,
    detect_active_venv,
    get_venv_run_prefix,
    run_all_checks,
    run_in_venv,
    run_selected_checks,
)

__version__ = "0.11.0"

__all__ = [
    # Config
    "AppConfig",
    "HippocampalConfig",
    "MaestroConfig",
    "SomaticConfig",
    "get_config",
    # Daemon
    "MaestroDaemon",
    # Shutdown
    "ShutdownController",
    # Dashboard
    "Dashboard",
    "run_dashboard",
    # Logging
    "MaestroLogger",
    "configure_logging",
    "get_logger",
    "log_cluster_health",
    # Health models
    "ClusterHealth",
    "ProbeResult",
    "ServiceResult",
    "ServiceStatus",
    "UbikService",
    # Health runners
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
