"""
Unit tests for fetch_sources — local mode, manifest, idempotency.

Covers: byte-identical copies, SHA-256 dedup across runs and across
differently-named duplicates, name-collision suffixing, dry-run
no-op behavior, corrupt-manifest loudness, and input validation.
"""

import json
from pathlib import Path

import pytest

from config.ingestion_config import PathsConfig
from fetch_sources import (
    FetchError,
    ManifestRecord,
    SourceManifest,
    compute_sha256,
    fetch_local,
)


@pytest.fixture()
def paths(tmp_path, monkeypatch):
    """PathsConfig rooted in a temp ingestion workspace."""
    monkeypatch.setenv("UBIK_INGESTION_ROOT", str(tmp_path / "ingestion"))
    return PathsConfig.from_env()


@pytest.fixture()
def drop_dir(tmp_path):
    """A directory of manually-downloaded files."""
    d = tmp_path / "downloads"
    d.mkdir()
    (d / "meeting_a.txt").write_text("contenido de la reunión A", encoding="utf-8")
    (d / "meeting_b.txt").write_text("contenido de la reunión B", encoding="utf-8")
    return d


def _manifest_lines(paths):
    manifest = paths.sources_dir / "MANIFEST.jsonl"
    if not manifest.exists():
        return []
    return [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]


# =============================================================================
# Happy path
# =============================================================================

def test_fetch_local_copies_byte_identical(paths, drop_dir):
    records = fetch_local(drop_dir, "tactiq", paths)
    assert len(records) == 2
    for rec in records:
        src = drop_dir / rec.original_name
        dest = paths.root / rec.dest_path
        assert dest.read_bytes() == src.read_bytes()
        assert rec.sha256 == compute_sha256(src)


def test_fetch_local_writes_manifest_fields(paths, drop_dir):
    fetch_local(drop_dir, "tactiq", paths)
    lines = _manifest_lines(paths)
    assert len(lines) == 2
    rec = lines[0]
    assert rec["source"] == "local"
    assert rec["source_type"] == "tactiq"
    assert rec["drive_id"] is None
    assert rec["sha256"] and rec["fetched_at"]
    assert rec["dest_path"].startswith("sources/tactiq/")


def test_original_files_untouched(paths, drop_dir):
    before = {p.name: p.read_bytes() for p in drop_dir.iterdir()}
    fetch_local(drop_dir, "tactiq", paths)
    after = {p.name: p.read_bytes() for p in drop_dir.iterdir()}
    assert before == after


# =============================================================================
# Idempotency
# =============================================================================

def test_second_run_skips_everything(paths, drop_dir):
    first = fetch_local(drop_dir, "tactiq", paths)
    second = fetch_local(drop_dir, "tactiq", paths)
    assert len(first) == 2
    assert second == []
    assert len(_manifest_lines(paths)) == 2  # no duplicate records


def test_same_content_different_name_is_skipped(paths, drop_dir, tmp_path):
    fetch_local(drop_dir, "tactiq", paths)
    other = tmp_path / "other"
    other.mkdir()
    (other / "renamed_copy.txt").write_text(
        "contenido de la reunión A", encoding="utf-8"
    )
    records = fetch_local(other, "tactiq", paths)
    assert records == []


def test_name_collision_different_content_gets_suffix(paths, drop_dir, tmp_path):
    fetch_local(drop_dir, "tactiq", paths)
    other = tmp_path / "other"
    other.mkdir()
    (other / "meeting_a.txt").write_text("contenido distinto", encoding="utf-8")
    records = fetch_local(other, "tactiq", paths)
    assert len(records) == 1
    dest = Path(records[0].dest_path)
    assert dest.name.startswith("meeting_a__")
    # the original copy is untouched
    original = paths.sources_dir / "tactiq" / "meeting_a.txt"
    assert original.read_text(encoding="utf-8") == "contenido de la reunión A"


# =============================================================================
# Dry run
# =============================================================================

def test_dry_run_copies_nothing_and_writes_no_manifest(paths, drop_dir):
    records = fetch_local(drop_dir, "tactiq", paths, dry_run=True)
    assert records == []
    assert not (paths.sources_dir / "tactiq").exists() or not any(
        f for f in (paths.sources_dir / "tactiq").iterdir()
        if f.name != ".gitkeep"
    )
    assert _manifest_lines(paths) == []


# =============================================================================
# Failure modes
# =============================================================================

def test_unknown_source_type_raises(paths, drop_dir):
    with pytest.raises(FetchError, match="Unknown source type"):
        fetch_local(drop_dir, "whatsapp", paths)


def test_missing_directory_raises(paths, tmp_path):
    with pytest.raises(FetchError, match="not a directory"):
        fetch_local(tmp_path / "nope", "tactiq", paths)


def test_corrupt_manifest_raises_loud(paths, drop_dir):
    manifest_path = paths.sources_dir / "MANIFEST.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"not valid json\n', encoding="utf-8")
    with pytest.raises(FetchError, match="Corrupt manifest"):
        fetch_local(drop_dir, "tactiq", paths)


def test_manifest_append_and_reload_roundtrip(tmp_path):
    path = tmp_path / "MANIFEST.jsonl"
    manifest = SourceManifest(path)
    record = ManifestRecord(
        sha256="ab" * 32, source="local", source_type="tactiq",
        original_name="x.txt", dest_path="sources/tactiq/x.txt",
        size_bytes=5, fetched_at="2026-06-12T00:00:00+00:00",
    )
    manifest.append(record)
    reloaded = SourceManifest(path)
    assert reloaded.has("ab" * 32)
    assert len(reloaded) == 1
