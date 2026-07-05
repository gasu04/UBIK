#!/usr/bin/env python3
"""
Maestro Web — FastAPI control panel backend.

Wraps the existing Maestro Python API (Orchestrator, ShutdownController,
ServiceRegistry, MetricsCollector, run_all_checks) in a small HTTP API and
serves a static single-page UI.  Because Maestro's lifecycle methods are
node-aware, every action works cluster-wide from the Hippocampal host — the
same one Maestro instance that controls both nodes over SSH.

Endpoints:
    GET  /                 → the control panel (static index.html)
    GET  /api/config       → node IPs + service URLs (Neo4j link, etc.)
    GET  /api/status       → full_status_check() for every service
    GET  /api/health       → run_all_checks() cluster health snapshot
    GET  /api/metrics      → usage metrics (storage / GPU / disk)
    GET  /api/logs?lines=N → tail of the Maestro log
    POST /api/start        → ensure_all_running() or start one service
    POST /api/shutdown     → orderly_shutdown() (dry-run / local-only / emergency)

Destructive actions (shutdown, emergency) require ``confirm: true`` in the body.

Author: UBIK Project
Version: 0.1.0
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from maestro.config import get_config
from maestro.orchestrator import Orchestrator
from maestro.platform_detect import detect_node
from maestro.services import ServiceRegistry, run_all_checks
from maestro.services.base import ProbeResult

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    """Body for POST /api/start."""
    service: Optional[str] = None      # single service name, or None for all
    local_only: bool = False


class ShutdownRequest(BaseModel):
    """Body for POST /api/shutdown."""
    dry_run: bool = False
    local_only: bool = False
    emergency: bool = False
    confirm: bool = False              # required for a real (non-dry-run) stop


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _probe_to_dict(p: ProbeResult) -> dict[str, Any]:
    """Convert a :class:`ProbeResult` into a JSON-friendly dict."""
    return {
        "name": p.name,
        "node": p.node.value,
        "healthy": p.healthy,
        "latency_ms": round(p.latency_ms, 1),
        "details": p.details,
        "error": p.error,
        "checked_at": p.checked_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app() -> FastAPI:
    """Construct and return the Maestro Web FastAPI application."""
    app = FastAPI(title="UBIK Maestro", docs_url="/api/docs")

    def _registry() -> ServiceRegistry:
        # Fresh registry per call so a mid-session config reload is honoured;
        # get_config() is cached so this is cheap.
        return ServiceRegistry(get_config())

    # ── Config / links ──────────────────────────────────────────────────
    @app.get("/api/config")
    async def api_config() -> dict[str, Any]:
        cfg = get_config()
        identity = detect_node()
        return {
            "node": identity.node_type.value,
            "hostname": identity.hostname,
            "hippocampal_ip": cfg.hippocampal.tailscale_ip,
            "somatic_ip": cfg.somatic.tailscale_ip,
            "links": {
                "neo4j": cfg.hippocampal.neo4j_http_url,
                "chromadb": cfg.hippocampal.chromadb_url,
                "mcp": cfg.hippocampal.mcp_url,
                "vllm": cfg.somatic.vllm_url,
            },
            "services": sorted(s.name for s in _registry().get_all()),
        }

    # ── Status ──────────────────────────────────────────────────────────
    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        registry = _registry()
        orch = Orchestrator(registry, detect_node())
        statuses = await orch.full_status_check()
        services = [_probe_to_dict(p) for p in statuses.values()]
        healthy = sum(1 for s in services if s["healthy"])
        return {
            "services": services,
            "healthy": healthy,
            "total": len(services),
        }

    # ── Health ──────────────────────────────────────────────────────────
    @app.get("/api/health")
    async def api_health() -> dict[str, Any]:
        cluster = await run_all_checks(get_config())
        # ClusterHealth exposes to_json(); fall back to a minimal dict.
        try:
            import json
            return json.loads(cluster.to_json())
        except Exception:
            return {"overall_status": getattr(cluster, "overall_status", "unknown")}

    # ── Metrics ─────────────────────────────────────────────────────────
    @app.get("/api/metrics")
    async def api_metrics() -> dict[str, Any]:
        try:
            from maestro.metrics import MetricsCollector
            registry = _registry()
            orch = Orchestrator(registry, detect_node())
            metrics = await MetricsCollector().collect(orch)
            # UsageMetrics is a dataclass → dict.
            from dataclasses import asdict, is_dataclass
            if is_dataclass(metrics):
                data = asdict(metrics)
                # datetimes → iso
                for k, v in list(data.items()):
                    if hasattr(v, "isoformat"):
                        data[k] = v.isoformat()
                return data
            return {"metrics": str(metrics)}
        except Exception as exc:
            logger.warning("metrics collection failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"metrics failed: {exc}")

    # ── Logs ────────────────────────────────────────────────────────────
    @app.get("/api/logs")
    async def api_logs(lines: int = 200) -> dict[str, Any]:
        cfg = get_config()
        log_path = cfg.log_dir / "maestro.log"
        if not log_path.exists():
            return {"path": str(log_path), "lines": [], "note": "log file not found"}
        lines = max(1, min(lines, 2000))
        try:
            content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"cannot read log: {exc}")
        return {"path": str(log_path), "lines": content[-lines:]}

    # ── Start ───────────────────────────────────────────────────────────
    @app.post("/api/start")
    async def api_start(req: StartRequest) -> dict[str, Any]:
        cfg = get_config()
        registry = _registry()
        identity = detect_node()
        if req.service:
            svc = next((s for s in registry.get_all() if s.name == req.service), None)
            if svc is None:
                raise HTTPException(status_code=404, detail=f"unknown service: {req.service}")
            ok = await svc.start(cfg.ubik_root)
            return {"action": "start", "service": req.service, "ok": ok,
                    "failed": [] if ok else [req.service]}
        orch = Orchestrator(registry, identity)
        failed = await orch.ensure_all_running(local_only=req.local_only)
        return {"action": "start", "service": None, "ok": not failed, "failed": failed}

    # ── Shutdown ────────────────────────────────────────────────────────
    @app.post("/api/shutdown")
    async def api_shutdown(req: ShutdownRequest) -> dict[str, Any]:
        registry = _registry()
        identity = detect_node()
        from maestro.shutdown import ShutdownController
        ctrl = ShutdownController(registry, identity)

        if req.emergency:
            if not req.confirm:
                raise HTTPException(status_code=400, detail="emergency shutdown requires confirm=true")
            await ctrl.emergency_shutdown()
            return {"action": "emergency_shutdown", "stopped": "all-local-ports"}

        if not req.dry_run and not req.confirm:
            raise HTTPException(status_code=400, detail="shutdown requires confirm=true (or use dry_run)")

        stopped = await ctrl.orderly_shutdown(dry_run=req.dry_run, local_only=req.local_only)
        return {"action": "shutdown", "dry_run": req.dry_run,
                "local_only": req.local_only, "stopped": stopped}

    # ── Static UI ───────────────────────────────────────────────────────
    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    return app


def run_web(host: str = "0.0.0.0", port: int = 8090, *, log_level: str = "info") -> None:
    """Launch the Maestro Web control panel with uvicorn.

    Args:
        host: Bind address (default all interfaces so it's reachable over
            Tailscale).
        port: TCP port (default 8090).
        log_level: uvicorn log level.
    """
    import uvicorn
    logger.info("Maestro Web starting on http://%s:%d", host, port)
    uvicorn.run(build_app(), host=host, port=port, log_level=log_level)
