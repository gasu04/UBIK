"""
Unit tests for run_phase3.py — the Phase 3 orchestrator.

Offline stages (1 parse, 3 resolve, 4 gate) are run directly; the
service-dependent stages (2 enrich, 5 write) are exercised with injected
fakes. Also covers the stage-summary line, source iteration/limit, and the
dry-run stage filtering.
"""

import json
from pathlib import Path

import pytest

from config.ingestion_config import (
    EndpointConfig,
    GateThresholds,
    IngestionConfig,
    PathsConfig,
)
import run_phase3
from run_phase3 import (
    StageSummary,
    _iter_source_files,
    _parse_transcript_file,
    run,
    stage1_parse,
    stage3_resolve,
    stage4_gate,
    stage5_write,
)


@pytest.fixture()
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("UBIK_INGESTION_ROOT", str(tmp_path / "ingestion"))
    cfg = IngestionConfig(
        endpoints=EndpointConfig(
            sensitive_endpoint="http://test:8002/v1",
            standard_endpoint="http://test:8002/v1",
            model="test-model",
        ),
        paths=PathsConfig.from_env(),
        gates=GateThresholds(),
        prompt_version="v1.0.0",
    )
    (cfg.paths.sources_dir / "tactiq").mkdir(parents=True, exist_ok=True)
    return cfg


class _FakeProcessed:
    def __init__(self, text):
        self.text = text


class _FakeProcessors:
    def __init__(self, can=True):
        self._can = can

    def can_process(self, ext):
        return self._can

    async def process(self, item, path):
        return _FakeProcessed(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_summary_line_format():
    s = StageSummary(2, "enrich", processed=6, skipped=1, quarantined=2)
    assert s.line() == "Stage 2 (enrich) complete: 6 processed, 1 skipped, 2 quarantined"


def test_iter_source_files_respects_limit(config):
    d = config.paths.sources_dir / "tactiq"
    for i in range(5):
        (d / f"f{i}.txt").write_text("x", encoding="utf-8")
    (d / ".hidden").write_text("x", encoding="utf-8")
    got = _iter_source_files(config, limit=3)
    assert len(got) == 3
    assert all(ct == "transcript_tactiq" for _p, ct in got)


def test_parse_transcript_file_roundtrip(tmp_path):
    p = tmp_path / "x.transcript"
    p.write_text("---\nmeeting_type: therapy\n---\n\nel cuerpo aquí\n", encoding="utf-8")
    front, body = _parse_transcript_file(p)
    assert front["meeting_type"] == "therapy"
    assert body.strip() == "el cuerpo aquí"


# ---------------------------------------------------------------------------
# Stage 1 (parse)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stage1_parses_processable_files(config):
    d = config.paths.sources_dir / "tactiq"
    (d / "a.txt").write_text("Ginés: hola\nAdrián: qué tal", encoding="utf-8")
    (d / "b.txt").write_text("solo yo hablando", encoding="utf-8")
    s = await stage1_parse(config, limit=None, processors=_FakeProcessors(can=True))
    assert s.processed == 2
    assert s.quarantined == 0


@pytest.mark.asyncio
async def test_stage1_quarantines_unprocessable(config):
    d = config.paths.sources_dir / "tactiq"
    (d / "a.bin").write_text("x", encoding="utf-8")
    s = await stage1_parse(config, limit=None, processors=_FakeProcessors(can=False))
    assert s.processed == 0
    assert s.quarantined == 1


# ---------------------------------------------------------------------------
# Stage 3 (resolve)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stage3_writes_resolution_sidecar(config):
    enriched = config.paths.enriched_dir
    enriched.mkdir(parents=True, exist_ok=True)
    (enriched / "m.transcript").write_text(
        "---\n"
        "meeting_type: family_conversation\n"
        "participants_detected:\n  - Ginés\n  - Desconocido\n"
        "voice_corpus_eligible: false\n"
        "_audit:\n  source_file: m.docx\n  source_sha256: abc123\n"
        "---\n\ncuerpo\n",
        encoding="utf-8",
    )
    s = await stage3_resolve(config, limit=None)
    assert s.processed == 1
    sidecar = config.paths.pending_review_dir / "m.resolved.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["resolution_status_summary"] == "some_unresolved"
    assert data["source_sha256"] == "abc123"
    statuses = {p["raw_name"]: p["resolution_status"] for p in data["participants_resolved"]}
    assert statuses["Ginés"] == "resolved"
    assert statuses["Desconocido"] == "unresolved"


# ---------------------------------------------------------------------------
# Stage 4 (gate) — counts the queue, never auto-approves
# ---------------------------------------------------------------------------

def test_stage4_counts_pending(config):
    pending = config.paths.pending_review_dir
    pending.mkdir(parents=True, exist_ok=True)
    (pending / "a.resolved.json").write_text("{}", encoding="utf-8")
    (pending / "b.resolved.json").write_text("{}", encoding="utf-8")
    s = stage4_gate(config)
    assert s.skipped == 2
    assert s.processed == 0


# ---------------------------------------------------------------------------
# Stage 5 (write) — therapy excluded, duplicate skipped, normal written
# ---------------------------------------------------------------------------

class _FakeWriter:
    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    async def store_episodic(self, **kwargs):
        self.calls.append(kwargs)
        return self._results.pop(0)


@pytest.mark.asyncio
async def test_stage5_excludes_therapy_and_counts(config):
    approved = config.paths.approved_dir
    approved.mkdir(parents=True, exist_ok=True)
    (approved / "ok.transcript").write_text(
        "---\nmeeting_type: family_conversation\n"
        "participants_detected:\n  - Ginés\n"
        "_audit:\n  source_file: ok.docx\n  source_sha256: aaa\n---\n\ncuerpo ok\n",
        encoding="utf-8",
    )
    (approved / "ther.transcript").write_text(
        "---\nmeeting_type: therapy\n"
        "_audit:\n  source_file: t.docx\n  source_sha256: bbb\n---\n\ncuerpo therapy\n",
        encoding="utf-8",
    )
    writer = _FakeWriter([{"status": "success", "memory_id": "ep_1"}])
    s = await stage5_write(config, limit=None, writer=writer)
    assert s.processed == 1       # the family_conversation
    assert s.quarantined == 1     # therapy excluded
    # therapy never reached the writer
    assert all(c["memory_type"] != "therapy" for c in writer.calls)
    # the written call carried Phase 3 metadata + sha
    assert writer.calls[0]["source_sha256"] == "aaa"
    assert writer.calls[0]["extra_metadata"]["ingestion_phase"] == "phase3_enriched"


@pytest.mark.asyncio
async def test_stage5_duplicate_counts_as_skipped(config):
    approved = config.paths.approved_dir
    approved.mkdir(parents=True, exist_ok=True)
    (approved / "dup.transcript").write_text(
        "---\nmeeting_type: interview\n"
        "_audit:\n  source_file: d.docx\n  source_sha256: ccc\n---\n\ncuerpo\n",
        encoding="utf-8",
    )
    writer = _FakeWriter([{"status": "duplicate", "memory_id": "ep_existing"}])
    s = await stage5_write(config, limit=None, writer=writer)
    assert s.processed == 0
    assert s.skipped == 1


# ---------------------------------------------------------------------------
# run() orchestration: dry-run drops stages 4 and 5
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_dry_run_filters_stages_4_and_5(config, monkeypatch):
    ran = []

    async def _fake_stage1(cfg, limit, **kw):
        ran.append(1)
        return StageSummary(1, "parse")

    monkeypatch.setattr(run_phase3, "stage1_parse", _fake_stage1)
    summaries = await run(config, stages=[1, 4, 5], dry_run=True, limit=None)
    assert ran == [1]
    assert [s.stage for s in summaries] == [1]
