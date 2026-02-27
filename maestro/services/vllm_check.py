#!/usr/bin/env python3
"""
Maestro — vLLM Inference Server Health Check

Probes the vLLM server on the Somatic Node via its OpenAI-compatible
HTTP API.  vLLM exposes two useful health endpoints:

    GET /health       — basic process liveness (returns 200 when up)
    GET /v1/models    — lists loaded models (returns JSON model list)

Result semantics:
    HEALTHY   — /health returns 200 and at least one model is loaded.
    DEGRADED  — /health returns 200 but no models are loaded yet
                (server starting up or model load failed).
    UNHEALTHY — cannot connect, timeout, or /health returns non-200.

Author: UBIK Project
Version: 0.1.0
"""

import time
import logging
from typing import Any

import httpx

from maestro.config import SomaticConfig
from maestro.services.models import ServiceResult, ServiceStatus

logger = logging.getLogger(__name__)


async def check_vllm(
    cfg: SomaticConfig,
    *,
    timeout: float = 5.0,
) -> ServiceResult:
    """Probe vLLM server health and loaded model status.

    Hits ``/health`` for liveness then ``/v1/models`` to verify that at
    least one model is loaded and serving requests.

    Args:
        cfg: Somatic node configuration providing the vLLM base URL.
        timeout: Maximum seconds to wait for each HTTP request.

    Returns:
        :class:`~maestro.services.models.ServiceResult` with:
            - ``details["models_loaded"]``: number of models active.
            - ``details["model_ids"]``: list of loaded model IDs.
            - ``details["url"]``: the probed vLLM base URL.
    """
    base_url = cfg.vllm_url
    details: dict[str, Any] = {"url": base_url}
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Step 1: process liveness
            health_resp = await client.get(f"{base_url}/health")
            latency_ms = (time.perf_counter() - start) * 1000

            if health_resp.status_code != 200:
                details["health_status"] = health_resp.status_code
                return ServiceResult(
                    service_name="vllm",
                    status=ServiceStatus.UNHEALTHY,
                    latency_ms=latency_ms,
                    details=details,
                    error=f"/health returned HTTP {health_resp.status_code}",
                )

            # Step 2: model availability
            models_resp = await client.get(f"{base_url}/v1/models")
            models_resp.raise_for_status()
            models_data: dict[str, Any] = models_resp.json()

            model_list: list[dict[str, Any]] = models_data.get("data", [])
            model_ids: list[str] = [m.get("id", "") for m in model_list]
            details["models_loaded"] = len(model_ids)
            details["model_ids"] = model_ids

            if not model_ids:
                return ServiceResult(
                    service_name="vllm",
                    status=ServiceStatus.DEGRADED,
                    latency_ms=latency_ms,
                    details=details,
                    error="Server is up but no models are loaded",
                )

            return ServiceResult(
                service_name="vllm",
                status=ServiceStatus.HEALTHY,
                latency_ms=latency_ms,
                details=details,
            )

    except httpx.ConnectError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("vllm connect error: %s", exc)
        return ServiceResult(
            service_name="vllm",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error="Connection refused -- is vLLM running on the Somatic Node?",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("vllm check failed: %s", exc)
        return ServiceResult(
            service_name="vllm",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error=str(exc),
        )
