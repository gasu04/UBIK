"""
Tests for maestro.metrics

Covers:
    UsageMetrics dataclass shape
    MetricsCollector._chromadb_count — v2 success, v1 fallback, error → None
    MetricsCollector._neo4j_count   — success, ImportError, exception → None
    MetricsCollector._vllm_running  — 200 → True, other / exception → False
    MetricsCollector._gpu_utilization — success, failure → None
    MetricsCollector._gpu_memory      — success, failure → None
    MetricsCollector._disk_usage    — success, failure → None
    MetricsCollector.collect        — Hippocampal (no GPU), Somatic (GPU),
                                      gather exception propagation
    MetricsCollector.format_report  — all fields, all-None, partial
"""

import asyncio
import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maestro.metrics import MetricsCollector, UsageMetrics, _noop
from maestro.platform_detect import NodeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(
    chromadb_url="http://127.0.0.1:8001",
    chromadb_token="tok",
    neo4j_bolt_url="bolt://127.0.0.1:7687",
    neo4j_user="neo4j",
    neo4j_password="pw",
    vllm_url="http://127.0.0.2:8002",
):
    """Return a minimal fake AppConfig-like object."""
    hc = MagicMock()
    hc.chromadb_url = chromadb_url
    hc.chromadb_token = chromadb_token
    hc.neo4j_bolt_url = neo4j_bolt_url
    hc.neo4j_user = neo4j_user
    hc.neo4j_password = neo4j_password

    sc = MagicMock()
    sc.vllm_url = vllm_url

    cfg = MagicMock()
    cfg.hippocampal = hc
    cfg.somatic = sc
    cfg.ubik_root = Path("/tmp/test-ubik")
    return cfg


def _make_orch(node_type=NodeType.HIPPOCAMPAL, ubik_root=Path("/tmp/test-ubik")):
    """Return a minimal fake Orchestrator."""
    identity = MagicMock()
    identity.node_type = node_type
    identity.ubik_root = ubik_root

    registry = MagicMock()
    registry.cfg = _make_cfg()

    orch = MagicMock()
    orch._registry = registry
    orch._identity = identity
    return orch


def _fake_response(status_code, json_body):
    """Build a fake httpx response object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    return resp


# ---------------------------------------------------------------------------
# UsageMetrics
# ---------------------------------------------------------------------------

class TestUsageMetrics:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(UsageMetrics)

    def test_required_fields(self):
        ts = datetime.now(timezone.utc)
        m = UsageMetrics(timestamp=ts)
        assert m.timestamp == ts

    def test_optional_fields_default_none(self):
        m = UsageMetrics(timestamp=datetime.now(timezone.utc))
        assert m.chromadb_episodic_count is None
        assert m.chromadb_semantic_count is None
        assert m.neo4j_node_count is None
        assert m.neo4j_relationship_count is None
        assert m.vllm_running is False
        assert m.gpu_utilization_pct is None
        assert m.gpu_memory_used_mb is None
        assert m.disk_usage_ubik_gb is None

    def test_asdict_has_timestamp(self):
        m = UsageMetrics(timestamp=datetime.now(timezone.utc), neo4j_node_count=5)
        d = dataclasses.asdict(m)
        assert "timestamp" in d
        assert d["neo4j_node_count"] == 5


# ---------------------------------------------------------------------------
# _noop helper
# ---------------------------------------------------------------------------

class TestNoop:
    @pytest.mark.asyncio
    async def test_returns_value(self):
        assert await _noop(42) == 42

    @pytest.mark.asyncio
    async def test_returns_none(self):
        assert await _noop(None) is None


# ---------------------------------------------------------------------------
# _chromadb_count
# ---------------------------------------------------------------------------

class TestChromadbCount:
    @pytest.mark.asyncio
    async def test_v2_success(self):
        resp = _fake_response(200, 99)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("maestro.metrics.httpx.AsyncClient", return_value=mock_ctx):
            result = await MetricsCollector()._chromadb_count(_make_cfg(), "ubik_episodic")

        assert result == 99

    @pytest.mark.asyncio
    async def test_v2_float_body_cast_to_int(self):
        resp = _fake_response(200, 7.0)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("maestro.metrics.httpx.AsyncClient", return_value=mock_ctx):
            result = await MetricsCollector()._chromadb_count(_make_cfg(), "ubik_semantic")

        assert result == 7

    @pytest.mark.asyncio
    async def test_v2_404_falls_back_to_v1(self):
        v2_resp = _fake_response(404, None)
        v1_resp = _fake_response(200, 42)

        call_count = 0

        async def _get(url, **_kw):
            nonlocal call_count
            call_count += 1
            return v2_resp if call_count == 1 else v1_resp

        mock_client = AsyncMock()
        mock_client.get = _get
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("maestro.metrics.httpx.AsyncClient", return_value=mock_ctx):
            result = await MetricsCollector()._chromadb_count(_make_cfg(), "ubik_episodic")

        assert result == 42

    @pytest.mark.asyncio
    async def test_v1_non_numeric_returns_none(self):
        v2_resp = _fake_response(404, None)
        v1_resp = _fake_response(200, "not-a-number")
        call_count = 0

        async def _get(url, **_kw):
            nonlocal call_count
            call_count += 1
            return v2_resp if call_count == 1 else v1_resp

        mock_client = AsyncMock()
        mock_client.get = _get
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("maestro.metrics.httpx.AsyncClient", return_value=mock_ctx):
            result = await MetricsCollector()._chromadb_count(_make_cfg(), "c")

        assert result is None

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self):
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=OSError("refused"))
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("maestro.metrics.httpx.AsyncClient", return_value=mock_ctx):
            result = await MetricsCollector()._chromadb_count(_make_cfg(), "c")

        assert result is None

    @pytest.mark.asyncio
    async def test_no_token_skips_auth_header(self):
        """When chromadb_token is None/empty, no Authorization header is sent."""
        cfg = _make_cfg(chromadb_token=None)
        sent_headers: list[dict] = []

        async def _get(url, headers=None, **_kw):
            sent_headers.append(headers or {})
            return _fake_response(200, 0)

        mock_client = AsyncMock()
        mock_client.get = _get
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("maestro.metrics.httpx.AsyncClient", return_value=mock_ctx):
            await MetricsCollector()._chromadb_count(cfg, "c")

        assert sent_headers and "Authorization" not in sent_headers[0]


# ---------------------------------------------------------------------------
# _neo4j_count
# ---------------------------------------------------------------------------

def _neo4j_mock(return_record) -> MagicMock:
    """Build a sys.modules["neo4j"] mock for _neo4j_count tests.

    Wires up: neo4j.AsyncGraphDatabase.driver(...)
                 → mock_driver with .session() context manager
                 → mock_session with .run() → result with .single()
    """
    mock_result = MagicMock()
    mock_result.single = AsyncMock(return_value=return_record)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    mock_driver.close = AsyncMock()

    mock_neo4j = MagicMock()
    # from neo4j import AsyncGraphDatabase → AsyncGraphDatabase = mock_neo4j.AsyncGraphDatabase
    mock_neo4j.AsyncGraphDatabase.driver = MagicMock(return_value=mock_driver)
    return mock_neo4j


class TestNeo4jCount:
    @pytest.mark.asyncio
    async def test_returns_count(self):
        mock_neo4j = _neo4j_mock({"count": 123})
        with patch.dict("sys.modules", {"neo4j": mock_neo4j}):
            result = await MetricsCollector()._neo4j_count(
                _make_cfg(), "MATCH (n) RETURN count(n) AS count"
            )
        assert result == 123

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_record(self):
        mock_neo4j = _neo4j_mock(None)
        with patch.dict("sys.modules", {"neo4j": mock_neo4j}):
            result = await MetricsCollector()._neo4j_count(_make_cfg(), "MATCH (n) RETURN count(n) AS count")

        assert result == 0

    @pytest.mark.asyncio
    async def test_import_error_returns_none(self):
        with patch.dict("sys.modules", {"neo4j": None}):
            result = await MetricsCollector()._neo4j_count(_make_cfg(), "MATCH (n) RETURN count(n) AS count")
        assert result is None

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self):
        mock_neo4j = MagicMock()
        # driver() itself raises — simulates connection refused
        mock_neo4j.AsyncGraphDatabase.driver = MagicMock(
            side_effect=Exception("connection refused")
        )

        with patch.dict("sys.modules", {"neo4j": mock_neo4j}):
            result = await MetricsCollector()._neo4j_count(
                _make_cfg(), "MATCH (n) RETURN count(n) AS count"
            )

        assert result is None


# ---------------------------------------------------------------------------
# _vllm_running
# ---------------------------------------------------------------------------

class TestVllmRunning:
    @pytest.mark.asyncio
    async def test_200_returns_true(self):
        resp = _fake_response(200, None)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("maestro.metrics.httpx.AsyncClient", return_value=mock_ctx):
            result = await MetricsCollector()._vllm_running(_make_cfg())

        assert result is True

    @pytest.mark.asyncio
    async def test_non_200_returns_false(self):
        resp = _fake_response(503, None)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=resp)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("maestro.metrics.httpx.AsyncClient", return_value=mock_ctx):
            result = await MetricsCollector()._vllm_running(_make_cfg())

        assert result is False

    @pytest.mark.asyncio
    async def test_connection_error_returns_false(self):
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=OSError("refused"))
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("maestro.metrics.httpx.AsyncClient", return_value=mock_ctx):
            result = await MetricsCollector()._vllm_running(_make_cfg())

        assert result is False


# ---------------------------------------------------------------------------
# _gpu_utilization
# ---------------------------------------------------------------------------

class TestGpuUtilization:
    @pytest.mark.asyncio
    async def test_returns_float(self):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"75\n", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with patch("asyncio.wait_for", new=AsyncMock(return_value=(b"75\n", b""))):
                result = await MetricsCollector()._gpu_utilization()

        assert result == 75.0

    @pytest.mark.asyncio
    async def test_subprocess_error_returns_none(self):
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("nvidia-smi not found"),
        ):
            result = await MetricsCollector()._gpu_utilization()

        assert result is None


# ---------------------------------------------------------------------------
# _gpu_memory
# ---------------------------------------------------------------------------

class TestGpuMemory:
    @pytest.mark.asyncio
    async def test_subprocess_error_returns_none(self):
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("nvidia-smi not found"),
        ):
            result = await MetricsCollector()._gpu_memory()

        assert result is None


# ---------------------------------------------------------------------------
# _disk_usage
# ---------------------------------------------------------------------------

class TestDiskUsage:
    @pytest.mark.asyncio
    async def test_returns_gib_float(self, tmp_path):
        import shutil
        usage = shutil.disk_usage(tmp_path)

        result = await MetricsCollector()._disk_usage(tmp_path)

        assert isinstance(result, float)
        assert result > 0.0

    @pytest.mark.asyncio
    async def test_existing_path_returns_float(self):
        result = await MetricsCollector()._disk_usage(Path("/tmp"))
        assert isinstance(result, float)
        assert result > 0.0

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        # Patch within the metrics module's shutil reference
        with patch("maestro.metrics.shutil.disk_usage", side_effect=OSError("perm denied")):
            result = await MetricsCollector()._disk_usage(Path("/tmp"))
        assert result is None


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

class TestCollect:
    @pytest.mark.asyncio
    async def test_hippocampal_no_gpu(self):
        """On Hippocampal node, GPU fields stay None."""
        orch = _make_orch(node_type=NodeType.HIPPOCAMPAL)

        collector = MetricsCollector()
        collector._chromadb_count = AsyncMock(side_effect=[5, 10])
        collector._neo4j_count = AsyncMock(side_effect=[100, 200])
        collector._vllm_running = AsyncMock(return_value=True)
        collector._gpu_utilization = AsyncMock(return_value=80.0)
        collector._gpu_memory = AsyncMock(return_value=12288.0)
        collector._disk_usage = AsyncMock(return_value=250.5)

        result = await collector.collect(orch)

        assert isinstance(result, UsageMetrics)
        assert result.chromadb_episodic_count == 5
        assert result.chromadb_semantic_count == 10
        assert result.neo4j_node_count == 100
        assert result.neo4j_relationship_count == 200
        assert result.vllm_running is True
        assert result.disk_usage_ubik_gb == 250.5
        # GPU not called on Hippocampal
        collector._gpu_utilization.assert_not_called()
        collector._gpu_memory.assert_not_called()
        assert result.gpu_utilization_pct is None
        assert result.gpu_memory_used_mb is None

    @pytest.mark.asyncio
    async def test_somatic_includes_gpu(self):
        """On Somatic node, GPU collectors are invoked."""
        orch = _make_orch(node_type=NodeType.SOMATIC)

        collector = MetricsCollector()
        collector._chromadb_count = AsyncMock(side_effect=[5, 10])
        collector._neo4j_count = AsyncMock(side_effect=[100, 200])
        collector._vllm_running = AsyncMock(return_value=False)
        collector._gpu_utilization = AsyncMock(return_value=60.0)
        collector._gpu_memory = AsyncMock(return_value=8192.0)
        collector._disk_usage = AsyncMock(return_value=100.0)

        result = await collector.collect(orch)

        collector._gpu_utilization.assert_called_once()
        collector._gpu_memory.assert_called_once()
        assert result.gpu_utilization_pct == 60.0
        assert result.gpu_memory_used_mb == 8192.0

    @pytest.mark.asyncio
    async def test_gather_exception_becomes_none(self):
        """An exception from a sub-collector does not propagate; field is None."""
        orch = _make_orch(node_type=NodeType.HIPPOCAMPAL)

        collector = MetricsCollector()
        collector._chromadb_count = AsyncMock(side_effect=RuntimeError("boom"))
        collector._neo4j_count = AsyncMock(side_effect=[50, 80])
        collector._vllm_running = AsyncMock(return_value=False)
        collector._gpu_utilization = AsyncMock(return_value=None)
        collector._gpu_memory = AsyncMock(return_value=None)
        collector._disk_usage = AsyncMock(return_value=10.0)

        result = await collector.collect(orch)

        assert result.chromadb_episodic_count is None
        assert result.chromadb_semantic_count is None
        assert result.neo4j_node_count == 50

    @pytest.mark.asyncio
    async def test_timestamp_is_utc(self):
        orch = _make_orch()
        collector = MetricsCollector()
        collector._chromadb_count = AsyncMock(return_value=None)
        collector._neo4j_count = AsyncMock(return_value=None)
        collector._vllm_running = AsyncMock(return_value=False)
        collector._gpu_utilization = AsyncMock(return_value=None)
        collector._gpu_memory = AsyncMock(return_value=None)
        collector._disk_usage = AsyncMock(return_value=None)

        result = await collector.collect(orch)

        assert result.timestamp.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def _ts(self) -> datetime:
        return datetime(2026, 2, 27, 12, 0, 0, tzinfo=timezone.utc)

    def test_all_fields_present(self):
        metrics = UsageMetrics(
            timestamp=self._ts(),
            chromadb_episodic_count=5,
            chromadb_semantic_count=12,
            neo4j_node_count=1234,
            neo4j_relationship_count=5678,
            vllm_running=True,
            gpu_utilization_pct=72.5,
            gpu_memory_used_mb=10240.0,
            disk_usage_ubik_gb=250.75,
        )
        report = MetricsCollector().format_report(metrics)
        assert "5" in report           # episodic count
        assert "12" in report          # semantic count
        assert "1234" in report        # neo4j nodes
        assert "5678" in report        # relationships
        assert "yes" in report         # vllm running
        assert "72.5" in report        # gpu util
        assert "10240" in report       # gpu mem
        assert "250" in report         # disk
        assert "2026-02-27" in report  # timestamp
        assert "GPU" in report         # GPU section present

    def test_all_fields_none(self):
        metrics = UsageMetrics(timestamp=self._ts())
        report = MetricsCollector().format_report(metrics)
        assert "N/A" in report
        assert "no" in report          # vllm_running=False → 'no'

    def test_no_gpu_section_when_no_gpu_data(self):
        metrics = UsageMetrics(
            timestamp=self._ts(),
            gpu_utilization_pct=None,
            gpu_memory_used_mb=None,
        )
        report = MetricsCollector().format_report(metrics)
        assert "GPU" not in report

    def test_gpu_section_when_partial_gpu_data(self):
        metrics = UsageMetrics(
            timestamp=self._ts(),
            gpu_utilization_pct=50.0,
            gpu_memory_used_mb=None,
        )
        report = MetricsCollector().format_report(metrics)
        assert "GPU" in report

    def test_returns_multiline_string(self):
        metrics = UsageMetrics(timestamp=self._ts())
        report = MetricsCollector().format_report(metrics)
        assert "\n" in report

    def test_separator_lines_present(self):
        metrics = UsageMetrics(timestamp=self._ts())
        report = MetricsCollector().format_report(metrics)
        assert "=" in report

    def test_sections_in_order(self):
        metrics = UsageMetrics(
            timestamp=self._ts(),
            chromadb_episodic_count=1,
            neo4j_node_count=2,
            vllm_running=True,
            gpu_utilization_pct=50.0,
            gpu_memory_used_mb=4096.0,
        )
        report = MetricsCollector().format_report(metrics)
        storage_pos = report.index("STORAGE")
        inference_pos = report.index("INFERENCE")
        gpu_pos = report.index("GPU")
        assert storage_pos < inference_pos < gpu_pos
