"""
Unit tests for ingest/person_resolver.py — 3-way participant resolution.

Covers exact/alias resolution (incl. the Gines/Gines Alberto disambiguation),
accent-normalized fallback, unresolved passthrough, ambiguity detection, and
the summary roll-up.
"""

from ingest.person_resolver import (
    AMBIGUOUS,
    RESOLVED,
    UNRESOLVED,
    PersonResolver,
    summarize_resolution,
)
from ingest.registry import KnownPerson, KnownPersonsRegistry, load_known_persons


def _resolver() -> PersonResolver:
    return PersonResolver(load_known_persons())


def test_exact_canonical_resolves_with_tier():
    r = _resolver().resolve_one("Maggie")
    assert r.resolution_status == RESOLVED
    assert r.canonical_name == "Maggie"
    assert r.privacy_tier == "family"


def test_alias_resolves_to_canonical():
    r = _resolver().resolve_one("Leti")
    assert r.resolution_status == RESOLVED
    assert r.canonical_name == "Leticia Zuno"
    assert r.privacy_tier == "therapy"


def test_bare_gines_is_the_father():
    r = _resolver().resolve_one("Gines")
    assert r.resolution_status == RESOLVED
    assert r.canonical_name == "Ginés Sanchez Urrutia"


def test_gines_alberto_is_the_son():
    r = _resolver().resolve_one("Gines Alberto")
    assert r.resolution_status == RESOLVED
    assert r.canonical_name == "Gines Alberto"


def test_gasu_alias_is_the_father():
    r = _resolver().resolve_one("GASU")
    assert r.canonical_name == "Ginés Sanchez Urrutia"


def test_accent_folded_fallback_resolves():
    # "Adrian" (no accent) is already an exact alias; "ADRIÁN " exercises fold.
    r = _resolver().resolve_one("  ADRIÁN ")
    assert r.resolution_status == RESOLVED
    assert r.canonical_name == "Adrian"


def test_unknown_name_unresolved():
    r = _resolver().resolve_one("Some Stranger")
    assert r.resolution_status == UNRESOLVED
    assert r.canonical_name is None
    assert r.privacy_tier is None


def test_empty_name_unresolved():
    r = _resolver().resolve_one("   ")
    assert r.resolution_status == UNRESOLVED


def test_ambiguous_when_fold_collides_on_two_people():
    # Two distinct canonicals that accent-fold to the same key.
    reg = KnownPersonsRegistry(persons=[
        KnownPerson("Ana", (), "friend", "personal", "family", "restricted"),
        KnownPerson("Aná", (), "colleague", "business", "business", "restricted"),
    ])
    resolver = PersonResolver(reg)
    # "anä" is not an exact label of either, but folds to "ana" -> both.
    r = resolver.resolve_one("anä")
    assert r.resolution_status == AMBIGUOUS
    assert r.canonical_name is None


def test_resolve_all_preserves_order():
    parts = _resolver().resolve_all(["Maggie", "Stranger", "Leti"])
    assert [p.raw_name for p in parts] == ["Maggie", "Stranger", "Leti"]


def test_summarize_all_resolved():
    parts = _resolver().resolve_all(["Maggie", "Adrian"])
    assert summarize_resolution(parts) == "all_resolved"


def test_summarize_some_unresolved():
    parts = _resolver().resolve_all(["Maggie", "Nobody At All"])
    assert summarize_resolution(parts) == "some_unresolved"


def test_summarize_ambiguous_wins_over_unresolved():
    reg = KnownPersonsRegistry(persons=[
        KnownPerson("Ana", (), "friend", "personal", "family", "restricted"),
        KnownPerson("Aná", (), "colleague", "business", "business", "restricted"),
    ])
    resolver = PersonResolver(reg)
    parts = resolver.resolve_all(["anä", "totally unknown"])
    assert summarize_resolution(parts) == "ambiguous"


def test_summarize_empty_is_all_resolved():
    assert summarize_resolution([]) == "all_resolved"
