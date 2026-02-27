#!/usr/bin/env python3
"""
Maestro Health Check Models

Shared data types for health check results.  These types flow from
individual service checkers through the health runner to any consumer
(CLI output, structured logs, dashboards, alerting).

Hierarchy:
    ServiceStatus  — enum: HEALTHY | DEGRADED | UNHEALTHY
    ServiceResult  — single service check outcome with latency & details
    ClusterHealth  — aggregated result across all monitored services

Author: UBIK Project
Version: 0.1.0
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ServiceStatus(str, Enum):
    """Health status of a single monitored service.

    Inherits from ``str`` so instances serialize correctly in JSON/logs
    without needing a custom encoder.

    Attributes:
        HEALTHY: Service is reachable and operating normally.
        DEGRADED: Service is reachable but sub-optimal — e.g. missing
            schema data, a peer node is offline, or capacity is reduced.
        UNHEALTHY: Service is unreachable or critically broken.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# Ordered worst-last for aggregation logic.
_STATUS_PRECEDENCE: list[ServiceStatus] = [
    ServiceStatus.HEALTHY,
    ServiceStatus.DEGRADED,
    ServiceStatus.UNHEALTHY,
]


@dataclass
class ServiceResult:
    """Outcome of a single service health check.

    Attributes:
        service_name: Canonical service identifier (e.g. ``"neo4j"``).
        status: Health status after evaluation.
        latency_ms: Round-trip probe latency in milliseconds; ``None`` when
            the check failed before any response was received.
        details: Structured diagnostic data (counts, versions, peer info…).
        error: Human-readable error description when status is not HEALTHY.
        checked_at: UTC timestamp when the check completed.

    Example:
        result = ServiceResult(
            service_name="neo4j",
            status=ServiceStatus.HEALTHY,
            latency_ms=12.4,
            details={"node_count": 47, "has_core_identity": True},
        )
        assert result.is_healthy
        assert result.to_dict()["status"] == "healthy"
    """

    service_name: str
    status: ServiceStatus
    latency_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    checked_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_healthy(self) -> bool:
        """``True`` when :attr:`status` is ``HEALTHY``."""
        return self.status == ServiceStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            Dict with keys: service, status, latency_ms, details, error,
            checked_at (ISO-8601 string).
        """
        return {
            "service": self.service_name,
            "status": self.status.value,
            "latency_ms": (
                round(self.latency_ms, 2) if self.latency_ms is not None else None
            ),
            "details": self.details,
            "error": self.error,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class ClusterHealth:
    """Aggregated health status across all UBIK services.

    Attributes:
        services: Mapping of service name → latest :class:`ServiceResult`.
        checked_at: UTC timestamp when this aggregate was assembled.

    Example:
        cluster = ClusterHealth(services={
            "neo4j": ServiceResult("neo4j", ServiceStatus.HEALTHY, 10.0),
            "vllm":  ServiceResult("vllm",  ServiceStatus.UNHEALTHY,
                                   error="connection refused"),
        })
        assert cluster.overall_status == ServiceStatus.UNHEALTHY
        assert cluster.unhealthy_services == ["vllm"]

    See Also:
        health_runner.run_all_checks: Populates this object concurrently.
    """

    services: dict[str, ServiceResult]
    checked_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def overall_status(self) -> ServiceStatus:
        """Worst status across all services.

        Returns:
            ``UNHEALTHY`` if any service is UNHEALTHY, ``DEGRADED`` if any
            is DEGRADED, ``HEALTHY`` only when every service is HEALTHY.
            Returns ``UNHEALTHY`` when the services dict is empty.
        """
        if not self.services:
            return ServiceStatus.UNHEALTHY
        return max(
            (r.status for r in self.services.values()),
            key=lambda s: _STATUS_PRECEDENCE.index(s),
        )

    @property
    def is_healthy(self) -> bool:
        """``True`` when every service reports HEALTHY."""
        return self.overall_status == ServiceStatus.HEALTHY

    @property
    def healthy_services(self) -> list[str]:
        """Names of services with HEALTHY status."""
        return [
            name
            for name, r in self.services.items()
            if r.status == ServiceStatus.HEALTHY
        ]

    @property
    def unhealthy_services(self) -> list[str]:
        """Names of services that are DEGRADED or UNHEALTHY."""
        return [
            name for name, r in self.services.items() if not r.is_healthy
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary.

        Returns:
            Dict with keys: overall_status, checked_at, healthy_count,
            total_count, services (nested ServiceResult dicts).
        """
        return {
            "overall_status": self.overall_status.value,
            "checked_at": self.checked_at.isoformat(),
            "healthy_count": len(self.healthy_services),
            "total_count": len(self.services),
            "services": {
                name: result.to_dict()
                for name, result in self.services.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a pretty-printed JSON string.

        Args:
            indent: JSON indentation width.

        Returns:
            Formatted JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent)
