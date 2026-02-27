#!/usr/bin/env python3
"""
Maestro — ChromaDB Health Check

Probes the ChromaDB vector store on the Hippocampal Node over its HTTP
API using ``httpx``.  Avoids the chromadb SDK to prevent version-skew
issues; all probes are raw HTTP requests.

API version detection:
    ChromaDB >= 0.6 deprecated the v1 API (returns HTTP 410).  This
    check auto-detects the running API version:

    v2 (preferred)  GET /api/v2/heartbeat
                    GET /api/v2/tenants/default_tenant/databases/default_database/collections
    v1 (fallback)   GET /api/v1/heartbeat
                    GET /api/v1/collections

    Detection order: try v2 first; if that returns 404 the server is
    older v1-only; if v1 returns 410 the server is v2-only.

Expected collections:
    ubik_episodic  — episodic memories (letters, sessions, conversations)
    ubik_semantic  — semantic knowledge (beliefs, values, preferences)

Result semantics:
    HEALTHY   — connected, both expected collections present.
    DEGRADED  — connected, but one or both collections missing.
    UNHEALTHY — unreachable or unexpected HTTP error.

Author: UBIK Project
Version: 0.2.0
"""

import time
import logging
from typing import Any

import httpx

from maestro.config import HippocampalConfig
from maestro.services.models import ServiceResult, ServiceStatus

logger = logging.getLogger(__name__)

# Collections that must exist for the system to be fully operational.
_REQUIRED_COLLECTIONS: frozenset[str] = frozenset({"ubik_episodic", "ubik_semantic"})

# ChromaDB v2 default tenant/database identifiers.
_V2_TENANT = "default_tenant"
_V2_DATABASE = "default_database"


def _extract_collection_names(raw: list[Any]) -> set[str]:
    """Parse a collection list response into a set of names.

    Handles both dict-per-collection (v1 and v2) and bare-string formats.

    Args:
        raw: Parsed JSON array from the collections endpoint.

    Returns:
        Set of collection name strings.
    """
    found: set[str] = set()
    for item in raw:
        if isinstance(item, dict):
            name = item.get("name", "")
        else:
            name = str(item)
        if name:
            found.add(name)
    return found


async def check_chromadb(
    cfg: HippocampalConfig,
    *,
    timeout: float = 5.0,
) -> ServiceResult:
    """Probe ChromaDB health with automatic API version detection.

    Tries the v2 API first (ChromaDB >= 0.6); falls back to v1 if the
    server responds with 404 on v2 endpoints.  Handles the HTTP 410
    "v1 deprecated" response by switching to v2.

    Args:
        cfg: Hippocampal node configuration providing the ChromaDB URL
            and optional bearer token.
        timeout: Maximum seconds for each HTTP request.

    Returns:
        :class:`~maestro.services.models.ServiceResult` with:
            - ``details["api_version"]``: ``"v2"`` or ``"v1"``.
            - ``details["collections_found"]``: list of collection names.
            - ``details["missing_collections"]``: expected names absent.
            - ``details["url"]``: the probed ChromaDB base URL.
    """
    base_url = cfg.chromadb_url
    auth_headers: dict[str, str] = {}
    if cfg.chromadb_token:
        auth_headers["Authorization"] = f"Bearer {cfg.chromadb_token}"

    details: dict[str, Any] = {"url": base_url}
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # ── API version detection ──────────────────────────────────────
            api_version, collections_url = await _detect_api(
                client, base_url, auth_headers
            )
            latency_ms = (time.perf_counter() - start) * 1000
            details["api_version"] = api_version

            # ── Collection enumeration ─────────────────────────────────────
            col_resp = await client.get(
                collections_url, headers=auth_headers
            )
            col_resp.raise_for_status()

            found = _extract_collection_names(col_resp.json())
            missing = sorted(_REQUIRED_COLLECTIONS - found)
            details["collections_found"] = sorted(found)
            details["missing_collections"] = missing

            if missing:
                return ServiceResult(
                    service_name="chromadb",
                    status=ServiceStatus.DEGRADED,
                    latency_ms=latency_ms,
                    details=details,
                    error=(
                        f"Missing collections: {missing} "
                        "-- run setup_chromadb.py"
                    ),
                )

            return ServiceResult(
                service_name="chromadb",
                status=ServiceStatus.HEALTHY,
                latency_ms=latency_ms,
                details=details,
            )

    except httpx.HTTPStatusError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("chromadb HTTP error: %s", exc)
        return ServiceResult(
            service_name="chromadb",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("chromadb check failed: %s", exc)
        return ServiceResult(
            service_name="chromadb",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error=str(exc),
        )


async def _detect_api(
    client: httpx.AsyncClient,
    base_url: str,
    auth_headers: dict[str, str],
) -> tuple[str, str]:
    """Detect ChromaDB API version and return the collections endpoint URL.

    Probes v2 first; falls back to v1 if the server responds with 404.

    Args:
        client: Open httpx async client.
        base_url: ChromaDB base URL.
        auth_headers: Authentication headers to include.

    Returns:
        Tuple of (api_version, collections_url) where api_version is
        ``"v2"`` or ``"v1"``.

    Raises:
        httpx.HTTPStatusError: If the heartbeat returns an unexpected
            non-404 error status.
    """
    # Try v2 heartbeat first
    v2_hb = await client.get(f"{base_url}/api/v2/heartbeat")

    if v2_hb.status_code == 200:
        collections_url = (
            f"{base_url}/api/v2/tenants/{_V2_TENANT}"
            f"/databases/{_V2_DATABASE}/collections"
        )
        return "v2", collections_url

    if v2_hb.status_code == 404:
        # Server is older, v1-only — verify v1 heartbeat
        v1_hb = await client.get(f"{base_url}/api/v1/heartbeat")
        v1_hb.raise_for_status()
        return "v1", f"{base_url}/api/v1/collections"

    # Any other status (e.g. 401, 500) is a genuine error
    v2_hb.raise_for_status()
    # unreachable, but satisfies type checker
    raise RuntimeError("unreachable")
