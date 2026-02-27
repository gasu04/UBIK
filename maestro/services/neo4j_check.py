#!/usr/bin/env python3
"""
Maestro — Neo4j Health Check

Probes the Neo4j graph database on the Hippocampal Node via the Bolt
protocol.  Three tiers of depth:

1. Network reachability  — TCP connect to the Bolt port succeeds.
2. Auth & connectivity   — ``driver.verify_connectivity()`` passes.
3. Schema integrity      — at least one node exists AND the
   ``CoreIdentity {name: 'Self'}`` node is present.

Result semantics:
    HEALTHY   — connected, auth OK, CoreIdentity node found.
    DEGRADED  — connected, auth OK, but schema not yet initialised
                (node count == 0 or CoreIdentity missing).
    UNHEALTHY — cannot connect or auth failed.

Dependencies:
    neo4j>=5.20.0

Author: UBIK Project
Version: 0.1.0
"""

import time
import logging

from maestro.config import HippocampalConfig
from maestro.services.models import ServiceResult, ServiceStatus

logger = logging.getLogger(__name__)

# Cypher queries used in the check
_QUERY_COUNT = "MATCH (n) RETURN count(n) AS count"
_QUERY_CORE_IDENTITY = "MATCH (s:CoreIdentity {name: 'Self'}) RETURN s LIMIT 1"


async def check_neo4j(
    cfg: HippocampalConfig,
    *,
    timeout: float = 5.0,
) -> ServiceResult:
    """Probe Neo4j health via the Bolt protocol.

    Opens an async driver, verifies authentication, then runs two read
    queries to validate schema integrity.  The driver is always closed
    when the function returns, even on error.

    Args:
        cfg: Hippocampal node configuration providing bolt URL and
            credentials.
        timeout: Maximum seconds to wait for each network operation.

    Returns:
        :class:`~maestro.services.models.ServiceResult` with:
            - ``details["node_count"]``: total node count when reachable.
            - ``details["has_core_identity"]``: whether CoreIdentity
              node exists.
            - ``details["bolt_url"]``: the probed Bolt URL.
    """
    try:
        from neo4j import AsyncGraphDatabase  # type: ignore[import-untyped]
    except ImportError:
        return ServiceResult(
            service_name="neo4j",
            status=ServiceStatus.UNHEALTHY,
            error="neo4j package not installed -- run: pip install neo4j",
        )

    bolt_url = cfg.neo4j_bolt_url
    auth = (cfg.neo4j_user, cfg.neo4j_password or "")
    details: dict = {"bolt_url": bolt_url}

    start = time.perf_counter()
    driver = None
    try:
        driver = AsyncGraphDatabase.driver(
            bolt_url,
            auth=auth,
            connection_timeout=timeout,
            max_connection_lifetime=timeout * 2,
        )
        await driver.verify_connectivity()
        latency_ms = (time.perf_counter() - start) * 1000

        # Schema checks
        async with driver.session() as session:
            count_result = await session.run(_QUERY_COUNT)
            record = await count_result.single()
            node_count: int = record["count"] if record else 0
            details["node_count"] = node_count

            id_result = await session.run(_QUERY_CORE_IDENTITY)
            has_core_identity = await id_result.single() is not None
            details["has_core_identity"] = has_core_identity

        if node_count == 0 or not has_core_identity:
            return ServiceResult(
                service_name="neo4j",
                status=ServiceStatus.DEGRADED,
                latency_ms=latency_ms,
                details=details,
                error="Schema not initialised -- run init_neo4j_schema.py",
            )

        return ServiceResult(
            service_name="neo4j",
            status=ServiceStatus.HEALTHY,
            latency_ms=latency_ms,
            details=details,
        )

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("neo4j check failed: %s", exc)
        return ServiceResult(
            service_name="neo4j",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error=str(exc),
        )
    finally:
        if driver is not None:
            await driver.close()
