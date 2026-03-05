#!/usr/bin/env python3
"""
Maestro — WhisperX Transcription Service Health Check

Probes the WhisperX FastAPI server on the Somatic Node.

    GET /health  — returns {"status": "ok", "model_loaded": <bool>}

Result semantics:
    HEALTHY   — /health returns 200 and model_loaded is True.
    DEGRADED  — /health returns 200 but model not yet loaded (starting up).
    UNHEALTHY — cannot connect, timeout, or non-200 response.

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


async def check_whisperx(
    cfg: SomaticConfig,
    *,
    timeout: float = 5.0,
) -> ServiceResult:
    """Probe WhisperX transcription service health.

    Hits ``/health`` on the Somatic Node's WhisperX FastAPI server and
    inspects the ``model_loaded`` field to distinguish a fully-ready server
    from one that is still loading the model.

    Args:
        cfg: Somatic node configuration providing the WhisperX base URL.
        timeout: Maximum seconds to wait for the HTTP request.

    Returns:
        :class:`~maestro.services.models.ServiceResult` with:
            - ``details["model_loaded"]``: whether the model is loaded.
            - ``details["device"]``: device reported by the server (if any).
            - ``details["url"]``: the probed URL.
    """
    url = f"{cfg.whisperx_url}/health"
    details: dict[str, Any] = {"url": url}
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            latency_ms = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                details["http_status"] = resp.status_code
                return ServiceResult(
                    service_name="whisperx",
                    status=ServiceStatus.UNHEALTHY,
                    latency_ms=latency_ms,
                    details=details,
                    error=f"/health returned HTTP {resp.status_code}",
                )

            body: dict[str, Any] = resp.json()
            model_loaded: bool = body.get("model_loaded", False)
            details["model_loaded"] = model_loaded
            if "device" in body:
                details["device"] = body["device"]

            if not model_loaded:
                return ServiceResult(
                    service_name="whisperx",
                    status=ServiceStatus.DEGRADED,
                    latency_ms=latency_ms,
                    details=details,
                    error="Server is up but model is not yet loaded",
                )

            return ServiceResult(
                service_name="whisperx",
                status=ServiceStatus.HEALTHY,
                latency_ms=latency_ms,
                details=details,
            )

    except httpx.ConnectError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("whisperx connect error: %s", exc)
        return ServiceResult(
            service_name="whisperx",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error="Connection refused — is WhisperX running on the Somatic Node?",
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("whisperx check failed: %s", exc)
        return ServiceResult(
            service_name="whisperx",
            status=ServiceStatus.UNHEALTHY,
            latency_ms=latency_ms,
            details=details,
            error=str(exc),
        )
