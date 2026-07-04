"""
Unit tests for LocalMemoryWriter source-document deduplication (Tier 1).

CLAUDE.md §3.4.1 entry #5: a missed dedup check writes a duplicate memory
that cannot be rolled back and silently skews retrieval forever. These tests
deliberately trigger the double-write path to prove the second write is a
no-op, plus the Phase 3 metadata merge and SHA truncation.

The writer's heavy deps (chromadb, sentence-transformers, neo4j) load only
inside connect(); here we inject fakes and exercise store_episodic directly.
"""

from typing import Any, Dict, List, Optional

import pytest

from ingest.mcp_writer import LocalMemoryWriter


class _FakeVector:
    def tolist(self) -> List[float]:
        return [0.0] * 384


class _FakeEmbed:
    def encode(self, text: str, normalize_embeddings: bool = True) -> _FakeVector:
        return _FakeVector()


class _FakeCollection:
    """Minimal stand-in for a chromadb collection (add / get / count)."""

    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {}

    def add(self, ids, embeddings, documents, metadatas) -> None:
        for i, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self.rows[i] = {"embedding": emb, "document": doc, "metadata": meta}

    def get(self, where: Optional[Dict[str, Any]] = None, **_kw) -> Dict[str, Any]:
        if where and "source_sha256" in where:
            sha = where["source_sha256"]
            ids = [
                i for i, row in self.rows.items()
                if row["metadata"].get("source_sha256") == sha
            ]
            return {"ids": ids}
        return {"ids": list(self.rows)}

    def count(self) -> int:
        return len(self.rows)


def _writer() -> LocalMemoryWriter:
    w = LocalMemoryWriter()
    w._episodic = _FakeCollection()
    w._embed_fn = _FakeEmbed()
    w._neo4j_password = ""   # force-skip the graph anchor (best-effort path)
    return w


_FULL_SHA = "a" * 64


@pytest.mark.asyncio
async def test_dedup_skips_second_write_same_source():
    """The silent-failure probe: re-ingesting the same source writes no second row."""
    w = _writer()
    first = await w.store_episodic("hola", "event", source_sha256=_FULL_SHA)
    second = await w.store_episodic("hola otra vez", "event", source_sha256=_FULL_SHA)

    assert first["status"] == "success"
    assert second["status"] == "duplicate"
    assert second["memory_id"] == first["memory_id"]
    assert w._episodic.count() == 1   # NOT 2 — the duplicate never landed


@pytest.mark.asyncio
async def test_distinct_sources_both_write():
    w = _writer()
    await w.store_episodic("uno", "event", source_sha256="a" * 64)
    await w.store_episodic("dos", "event", source_sha256="b" * 64)
    assert w._episodic.count() == 2


@pytest.mark.asyncio
async def test_no_sha_disables_dedup_backward_compatible():
    """Without source_sha256 the writer behaves like the original (always writes)."""
    w = _writer()
    await w.store_episodic("same text", "event")
    await w.store_episodic("same text", "event")
    assert w._episodic.count() == 2


@pytest.mark.asyncio
async def test_sha_stored_truncated_to_16():
    w = _writer()
    res = await w.store_episodic("x", "event", source_sha256=_FULL_SHA)
    meta = w._episodic.rows[res["memory_id"]]["metadata"]
    assert meta["source_sha256"] == _FULL_SHA[:16]
    assert len(meta["source_sha256"]) == 16


@pytest.mark.asyncio
async def test_extra_metadata_merged_and_none_dropped():
    w = _writer()
    res = await w.store_episodic(
        "x", "event", source_sha256=_FULL_SHA,
        extra_metadata={
            "voice_corpus_eligible": False,     # False must be kept
            "ingestion_phase": "phase3_enriched",
            "enrichment_confidence": 0.82,
            "meeting_date": None,               # None must be dropped
        },
    )
    meta = w._episodic.rows[res["memory_id"]]["metadata"]
    assert meta["voice_corpus_eligible"] is False
    assert meta["ingestion_phase"] == "phase3_enriched"
    assert meta["enrichment_confidence"] == 0.82
    assert "meeting_date" not in meta


@pytest.mark.asyncio
async def test_find_by_sha_none_when_absent():
    w = _writer()
    assert await w._find_episodic_by_sha("deadbeefdeadbeef") is None
