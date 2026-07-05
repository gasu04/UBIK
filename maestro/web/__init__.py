"""
Maestro Web — browser control panel for the UBIK cluster.

Exposes every Maestro operation (status, start, shutdown, health, metrics,
logs) over a small FastAPI app plus a static single-page UI, and links out to
the Neo4j browser.  Launch with ``maestro web`` or::

    from maestro.web import run_web
    run_web(host="0.0.0.0", port=8090)
"""

from .server import build_app, run_web

__all__ = ["build_app", "run_web"]
