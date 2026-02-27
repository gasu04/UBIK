"""
Tests for maestro.services.models

Pure unit tests — no I/O, no mocking, no external dependencies.
Covers ServiceStatus, ServiceResult, and ClusterHealth behaviour.
"""

import json
from datetime import datetime, timezone

import pytest

from maestro.services.models import ClusterHealth, ServiceResult, ServiceStatus


# ---------------------------------------------------------------------------
# ServiceStatus
# ---------------------------------------------------------------------------

class TestServiceStatus:
    def test_values_are_strings(self):
        assert ServiceStatus.HEALTHY == "healthy"
        assert ServiceStatus.DEGRADED == "degraded"
        assert ServiceStatus.UNHEALTHY == "unhealthy"

    def test_is_str_subclass(self):
        assert isinstance(ServiceStatus.HEALTHY, str)

    def test_json_serializable_directly(self):
        payload = {"status": ServiceStatus.HEALTHY}
        encoded = json.dumps(payload)
        assert '"healthy"' in encoded


# ---------------------------------------------------------------------------
# ServiceResult
# ---------------------------------------------------------------------------

class TestServiceResult:
    def test_is_healthy_true_when_healthy(self, healthy_result):
        assert healthy_result.is_healthy is True

    def test_is_healthy_false_when_degraded(self, degraded_result):
        assert degraded_result.is_healthy is False

    def test_is_healthy_false_when_unhealthy(self, unhealthy_result):
        assert unhealthy_result.is_healthy is False

    def test_checked_at_defaults_to_utc_now(self):
        before = datetime.now(timezone.utc)
        result = ServiceResult("svc", ServiceStatus.HEALTHY)
        after = datetime.now(timezone.utc)
        assert before <= result.checked_at <= after

    def test_to_dict_keys(self, healthy_result):
        d = healthy_result.to_dict()
        assert set(d.keys()) == {
            "service", "status", "latency_ms", "details", "error", "checked_at"
        }

    def test_to_dict_status_is_string(self, healthy_result):
        d = healthy_result.to_dict()
        assert d["status"] == "healthy"
        assert isinstance(d["status"], str)

    def test_to_dict_latency_rounded(self):
        result = ServiceResult(
            "svc", ServiceStatus.HEALTHY, latency_ms=12.3456789
        )
        assert result.to_dict()["latency_ms"] == 12.35

    def test_to_dict_latency_none_when_absent(self, unhealthy_result):
        assert unhealthy_result.to_dict()["latency_ms"] is None

    def test_to_dict_error_propagated(self, unhealthy_result):
        assert unhealthy_result.to_dict()["error"] == "Connection refused"

    def test_to_dict_checked_at_is_iso_string(self, healthy_result):
        d = healthy_result.to_dict()
        # Should be parseable as an ISO-8601 datetime
        dt = datetime.fromisoformat(d["checked_at"])
        assert dt.tzinfo is not None

    def test_details_default_empty_dict(self):
        result = ServiceResult("svc", ServiceStatus.UNHEALTHY)
        assert result.details == {}

    def test_details_mutable_default_is_independent(self):
        r1 = ServiceResult("svc", ServiceStatus.HEALTHY)
        r2 = ServiceResult("svc", ServiceStatus.HEALTHY)
        r1.details["key"] = "value"
        assert "key" not in r2.details


# ---------------------------------------------------------------------------
# ClusterHealth
# ---------------------------------------------------------------------------

class TestClusterHealth:
    def _make_result(self, name: str, status: ServiceStatus) -> ServiceResult:
        return ServiceResult(service_name=name, status=status)

    def test_overall_healthy_when_all_healthy(self):
        cluster = ClusterHealth(
            services={
                "neo4j": self._make_result("neo4j", ServiceStatus.HEALTHY),
                "chromadb": self._make_result("chromadb", ServiceStatus.HEALTHY),
            }
        )
        assert cluster.overall_status == ServiceStatus.HEALTHY
        assert cluster.is_healthy is True

    def test_overall_degraded_when_one_degraded(self):
        cluster = ClusterHealth(
            services={
                "neo4j": self._make_result("neo4j", ServiceStatus.HEALTHY),
                "chromadb": self._make_result("chromadb", ServiceStatus.DEGRADED),
            }
        )
        assert cluster.overall_status == ServiceStatus.DEGRADED
        assert cluster.is_healthy is False

    def test_overall_unhealthy_when_one_unhealthy(self):
        cluster = ClusterHealth(
            services={
                "neo4j": self._make_result("neo4j", ServiceStatus.HEALTHY),
                "vllm": self._make_result("vllm", ServiceStatus.UNHEALTHY),
            }
        )
        assert cluster.overall_status == ServiceStatus.UNHEALTHY

    def test_unhealthy_beats_degraded(self):
        """UNHEALTHY takes precedence over DEGRADED in aggregation."""
        cluster = ClusterHealth(
            services={
                "a": self._make_result("a", ServiceStatus.DEGRADED),
                "b": self._make_result("b", ServiceStatus.UNHEALTHY),
                "c": self._make_result("c", ServiceStatus.HEALTHY),
            }
        )
        assert cluster.overall_status == ServiceStatus.UNHEALTHY

    def test_overall_unhealthy_when_empty(self):
        cluster = ClusterHealth(services={})
        assert cluster.overall_status == ServiceStatus.UNHEALTHY

    def test_healthy_services_list(self):
        cluster = ClusterHealth(
            services={
                "neo4j": self._make_result("neo4j", ServiceStatus.HEALTHY),
                "vllm": self._make_result("vllm", ServiceStatus.UNHEALTHY),
            }
        )
        assert cluster.healthy_services == ["neo4j"]

    def test_unhealthy_services_list_includes_degraded(self):
        cluster = ClusterHealth(
            services={
                "neo4j": self._make_result("neo4j", ServiceStatus.DEGRADED),
                "vllm": self._make_result("vllm", ServiceStatus.UNHEALTHY),
                "mcp": self._make_result("mcp", ServiceStatus.HEALTHY),
            }
        )
        assert set(cluster.unhealthy_services) == {"neo4j", "vllm"}

    def test_to_dict_structure(self):
        cluster = ClusterHealth(
            services={
                "neo4j": self._make_result("neo4j", ServiceStatus.HEALTHY),
            }
        )
        d = cluster.to_dict()
        assert d["overall_status"] == "healthy"
        assert d["healthy_count"] == 1
        assert d["total_count"] == 1
        assert "neo4j" in d["services"]

    def test_to_json_is_valid_json(self):
        cluster = ClusterHealth(
            services={
                "neo4j": self._make_result("neo4j", ServiceStatus.HEALTHY),
            }
        )
        raw = cluster.to_json()
        parsed = json.loads(raw)
        assert parsed["overall_status"] == "healthy"

    def test_to_json_respects_indent(self):
        cluster = ClusterHealth(services={})
        raw2 = cluster.to_json(indent=2)
        raw4 = cluster.to_json(indent=4)
        # Indented JSON is longer
        assert len(raw4) >= len(raw2)
