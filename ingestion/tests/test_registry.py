"""
Unit tests for ingest.registry — registry loaders and CSV conversion.

Covers: happy-path loads of the real registry files, fail-loud
validation of every malformed-entry class, alias resolution, the
voice-corpus/diarization invariant, and CSV -> YAML conversion
(including that a bad CSV cannot clobber a good registry).
"""

from pathlib import Path

import pytest
import yaml

from ingest.registry import (
    RegistryValidationError,
    convert_known_persons_csv,
    load_content_types,
    load_known_persons,
)

REGISTRY_DIR = Path(__file__).resolve().parent.parent / "registry"


def _write_persons(tmp_path: Path, persons) -> Path:
    path = tmp_path / "known_persons.yaml"
    path.write_text(yaml.safe_dump({"persons": persons}), encoding="utf-8")
    return path


def _valid_person(**overrides):
    person = {
        "canonical_name": "Test Person",
        "aliases": ["TP"],
        "relationship": "friend",
        "role": "other",
        "privacy_tier": "family",
        "ubik_access_level": "full",
    }
    person.update(overrides)
    return person


def _write_ctypes(tmp_path: Path, ctypes) -> Path:
    path = tmp_path / "content_types.yaml"
    path.write_text(yaml.safe_dump({"content_types": ctypes}), encoding="utf-8")
    return path


def _valid_ctype(**overrides):
    spec = {
        "parser": "transcript",
        "enrichment": "required",
        "target_collections": ["ubik_episodic"],
    }
    spec.update(overrides)
    return spec


# =============================================================================
# Known persons — happy path
# =============================================================================

def test_load_known_persons_real_registry_validates():
    registry = load_known_persons(REGISTRY_DIR / "known_persons.yaml")
    assert len(registry.persons) >= 2


def test_resolve_alias_case_insensitive_returns_canonical(tmp_path):
    path = _write_persons(tmp_path, [_valid_person(
        canonical_name="Leticia Zuno", aliases=["Leti"]
    )])
    registry = load_known_persons(path)
    assert registry.resolve("leti") == "Leticia Zuno"
    assert registry.resolve("LETICIA ZUNO") == "Leticia Zuno"


def test_resolve_unknown_name_returns_none(tmp_path):
    path = _write_persons(tmp_path, [_valid_person()])
    registry = load_known_persons(path)
    assert registry.resolve("A Stranger") is None


def test_alias_map_contains_canonical_and_aliases(tmp_path):
    path = _write_persons(tmp_path, [_valid_person(
        canonical_name="Leticia Zuno", aliases=["Leti", "Leticia"]
    )])
    alias_map = load_known_persons(path).alias_map()
    assert alias_map["leti"] == "Leticia Zuno"
    assert alias_map["leticia zuno"] == "Leticia Zuno"


# =============================================================================
# Known persons — fail-loud validation
# =============================================================================

def test_load_known_persons_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_known_persons(Path("/nonexistent/known_persons.yaml"))


@pytest.mark.parametrize("missing_field", [
    "canonical_name", "aliases", "relationship", "role", "privacy_tier",
    "ubik_access_level",
])
def test_missing_required_field_raises(tmp_path, missing_field):
    person = _valid_person()
    del person[missing_field]
    path = _write_persons(tmp_path, [person])
    with pytest.raises(RegistryValidationError, match=missing_field):
        load_known_persons(path)


def test_invalid_privacy_tier_raises(tmp_path):
    path = _write_persons(tmp_path, [_valid_person(privacy_tier="secret")])
    with pytest.raises(RegistryValidationError, match="privacy_tier"):
        load_known_persons(path)


def test_invalid_access_level_raises(tmp_path):
    path = _write_persons(tmp_path, [_valid_person(ubik_access_level="root")])
    with pytest.raises(RegistryValidationError, match="ubik_access_level"):
        load_known_persons(path)


def test_duplicate_alias_across_entries_raises(tmp_path):
    path = _write_persons(tmp_path, [
        _valid_person(canonical_name="Person One", aliases=["Pep"]),
        _valid_person(canonical_name="Person Two", aliases=["Pep"]),
    ])
    with pytest.raises(RegistryValidationError, match="already used"):
        load_known_persons(path)


def test_empty_canonical_name_raises(tmp_path):
    path = _write_persons(tmp_path, [_valid_person(canonical_name="  ")])
    with pytest.raises(RegistryValidationError, match="empty"):
        load_known_persons(path)


def test_non_string_alias_raises(tmp_path):
    path = _write_persons(tmp_path, [_valid_person(aliases=["ok", 42])])
    with pytest.raises(RegistryValidationError, match="aliases"):
        load_known_persons(path)


def test_empty_persons_list_raises(tmp_path):
    path = _write_persons(tmp_path, [])
    with pytest.raises(RegistryValidationError, match="non-empty"):
        load_known_persons(path)


# =============================================================================
# Content types — happy path
# =============================================================================

def test_load_content_types_real_registry_validates():
    registry = load_content_types(REGISTRY_DIR / "content_types.yaml")
    expected = {
        "transcript_tactiq", "transcript_gemini", "transcript_fireflies",
        "letter_maestro", "memory_note", "ethical_constitution",
        "qa_calibration",
    }
    assert expected == set(registry.content_types)


def test_real_registry_tactiq_is_voice_blocked():
    spec = load_content_types(REGISTRY_DIR / "content_types.yaml").get("transcript_tactiq")
    assert spec.diarization_trust == "none"
    assert spec.voice_corpus_eligible is False


def test_real_registry_constitution_overrides():
    spec = load_content_types(REGISTRY_DIR / "content_types.yaml").get("ethical_constitution")
    assert spec.metadata_overrides == {"stability": "core", "importance": 1.0}
    assert spec.enrichment == "none"


def test_get_unknown_content_type_raises():
    registry = load_content_types(REGISTRY_DIR / "content_types.yaml")
    with pytest.raises(RegistryValidationError, match="Unknown content type"):
        registry.get("transcript_zoom")


# =============================================================================
# Content types — fail-loud validation
# =============================================================================

def test_invalid_parser_raises(tmp_path):
    path = _write_ctypes(tmp_path, {"x": _valid_ctype(parser="bogus")})
    with pytest.raises(RegistryValidationError, match="parser"):
        load_content_types(path)


def test_invalid_enrichment_raises(tmp_path):
    path = _write_ctypes(tmp_path, {"x": _valid_ctype(enrichment="heavy")})
    with pytest.raises(RegistryValidationError, match="enrichment"):
        load_content_types(path)


def test_invalid_diarization_trust_raises(tmp_path):
    path = _write_ctypes(tmp_path, {"x": _valid_ctype(diarization_trust="maybe")})
    with pytest.raises(RegistryValidationError, match="diarization_trust"):
        load_content_types(path)


def test_voice_eligible_with_untrusted_diarization_raises(tmp_path):
    """The voice-corpus invariant: mono-diarized sources can never be eligible."""
    path = _write_ctypes(tmp_path, {"x": _valid_ctype(
        diarization_trust="none", voice_corpus_eligible=True
    )})
    with pytest.raises(RegistryValidationError, match="voice_corpus_eligible"):
        load_content_types(path)


def test_empty_collections_without_route_raises(tmp_path):
    path = _write_ctypes(tmp_path, {"x": _valid_ctype(target_collections=[])})
    with pytest.raises(RegistryValidationError, match="route_to"):
        load_content_types(path)


# =============================================================================
# CSV conversion
# =============================================================================

CSV_HEADER = "Canonical_Name,Aliases,Relationship,Role,Privacy_Tier,UBIK_Access_Level\n"


def test_convert_csv_roundtrip(tmp_path):
    """Header matches the real Sheet export: capitalised, with access level."""
    csv_path = tmp_path / "Known_Persons.csv"
    csv_path.write_text(
        CSV_HEADER + 'Leticia Zuno,"Leti; Leticia",therapist,professional,therapy,restricted\n',
        encoding="utf-8",
    )
    out = convert_known_persons_csv(csv_path, tmp_path / "out.yaml")
    registry = load_known_persons(out)
    assert registry.resolve("Leti") == "Leticia Zuno"
    assert registry.get("Leti").privacy_tier == "therapy"
    assert registry.get("Leti").ubik_access_level == "restricted"


def test_convert_csv_empty_aliases_cell_ok(tmp_path):
    csv_path = tmp_path / "Known_Persons.csv"
    csv_path.write_text(
        CSV_HEADER + "Elena,,wife,core_family,family,full\n", encoding="utf-8"
    )
    registry = load_known_persons(
        convert_known_persons_csv(csv_path, tmp_path / "out.yaml")
    )
    assert registry.get("Elena").aliases == ()


def test_convert_csv_missing_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("canonical_name,aliases\nA,B\n", encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="missing required columns"):
        convert_known_persons_csv(csv_path, tmp_path / "out.yaml")


def test_convert_bad_csv_does_not_clobber_existing_registry(tmp_path):
    out = tmp_path / "out.yaml"
    _write_persons(tmp_path, [_valid_person()])
    (tmp_path / "known_persons.yaml").rename(out)
    original = out.read_text()

    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        CSV_HEADER + "Someone,alias,friend,other,not_a_tier,full\n", encoding="utf-8"
    )
    with pytest.raises(RegistryValidationError):
        convert_known_persons_csv(csv_path, out)
    assert out.read_text() == original
