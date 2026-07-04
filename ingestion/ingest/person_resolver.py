"""
UBIK Ingestion System - Participant Resolver

Resolves the raw participant names an enrichment produces
(``participants_detected``) to canonical people in the Known_Persons
registry, with a three-way outcome: ``resolved`` / ``unresolved`` /
``ambiguous``.

How it fits in: enrichment (Stage 2) yields raw name strings; this resolver
(Stage 3) attaches each to a canonical identity (or flags it) before Gate 1.
It **wraps** :class:`ingest.registry.KnownPersonsRegistry` — it does not
reimplement matching. The registry already handles case-insensitive exact and
alias matching (so the Gines/Gines Alberto disambiguation encoded in
``known_persons.yaml`` is honored: bare ``Gines``/``GASU`` → the father,
``Gines Alberto``/``Gines Hijo`` → the son). This module adds:

- an **accent-normalized** fallback (NFKD fold) so "Gines" matches "Ginés";
- **ambiguity detection**: a name that folds onto two or more distinct
  canonical people is flagged ``ambiguous`` and never auto-resolved.

Fuzzy / substring matching is intentionally NOT done: in a multi-decade
archive a wrong auto-resolution mis-attributes someone's words permanently,
so anything past an exact-or-folded match is left for a human at Gate 1.

Key classes/functions:
    ResolvedParticipant        — one name's resolution outcome
    PersonResolver             — resolve_one / resolve_all
    summarize_resolution       — roll up a list into a status summary string

Usage:
    from ingest.person_resolver import PersonResolver
    from ingest.registry import load_known_persons

    resolver = PersonResolver(load_known_persons())
    parts = resolver.resolve_all(["Ginés", "Gines Alberto", "Stranger"])
    summary = summarize_resolution(parts)   # "some_unresolved"

Dependencies: stdlib only (``unicodedata``); ``ingest.registry``.

Tier classification: Tier 2 (standard, 80% coverage). Mis-resolution is
surfaced at Gate 1 (human review), and unresolved/ambiguous never block —
they are flagged, so failure is loud and recoverable, not silent.

Version: 0.1.0
"""

import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from ingest.registry import KnownPersonsRegistry

__all__ = [
    "RESOLVED",
    "UNRESOLVED",
    "AMBIGUOUS",
    "ResolvedParticipant",
    "PersonResolver",
    "summarize_resolution",
]

RESOLVED = "resolved"
UNRESOLVED = "unresolved"
AMBIGUOUS = "ambiguous"


def _fold(label: str) -> str:
    """Accent-fold + lowercase + collapse whitespace for loose matching."""
    decomposed = unicodedata.normalize("NFKD", label.strip())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


@dataclass(frozen=True)
class ResolvedParticipant:
    """
    One participant name's resolution outcome.

    Attributes:
        raw_name: The name exactly as it appeared in the enrichment.
        canonical_id: Canonical identity key (the registry uses the canonical
            name as its identity), or None if not resolved.
        canonical_name: Canonical full name, or None if not resolved.
        resolution_status: One of ``resolved`` / ``unresolved`` / ``ambiguous``.
        privacy_tier: The resolved person's privacy tier, or None.
    """
    raw_name: str
    canonical_id: Optional[str]
    canonical_name: Optional[str]
    resolution_status: str
    privacy_tier: Optional[str]


class PersonResolver:
    """
    Resolve raw participant names against a Known_Persons registry.

    Args:
        registry: A loaded, validated KnownPersonsRegistry.

    Example:
        >>> from ingest.registry import load_known_persons
        >>> r = PersonResolver(load_known_persons())
        >>> r.resolve_one("Leti").canonical_name
        'Leticia Zuno'
        >>> r.resolve_one("nobody").resolution_status
        'unresolved'
    """

    def __init__(self, registry: KnownPersonsRegistry) -> None:
        self._registry = registry
        # Accent-folded index: folded label -> set of canonical names owning it.
        self._norm_index: Dict[str, Set[str]] = {}
        for person in registry.persons:
            for label in (person.canonical_name, *person.aliases):
                self._norm_index.setdefault(_fold(label), set()).add(
                    person.canonical_name
                )

    def resolve_one(self, raw_name: str) -> ResolvedParticipant:
        """
        Resolve a single raw name to a canonical identity or a flag.

        Resolution order: exact/alias (case-insensitive, via the registry),
        then accent-folded match. A folded form owned by exactly one canonical
        person resolves; by two or more it is ``ambiguous``; by none it is
        ``unresolved``.

        Args:
            raw_name: Name as it appears in the source/enrichment.

        Returns:
            A ResolvedParticipant describing the outcome.
        """
        raw = raw_name.strip()
        if not raw:
            return ResolvedParticipant(raw_name, None, None, UNRESOLVED, None)

        # 1. Exact / alias (case-insensitive) — owns the Gines disambiguation.
        canonical = self._registry.resolve(raw)
        if canonical:
            person = self._registry.get(raw)
            return ResolvedParticipant(
                raw, canonical, canonical, RESOLVED,
                person.privacy_tier if person else None,
            )

        # 2. Accent-folded fallback.
        candidates = self._norm_index.get(_fold(raw), set())
        if len(candidates) == 1:
            canonical = next(iter(candidates))
            person = self._registry.get(canonical)
            return ResolvedParticipant(
                raw, canonical, canonical, RESOLVED,
                person.privacy_tier if person else None,
            )
        if len(candidates) >= 2:
            return ResolvedParticipant(raw, None, None, AMBIGUOUS, None)

        return ResolvedParticipant(raw, None, None, UNRESOLVED, None)

    def resolve_all(self, raw_names: List[str]) -> List[ResolvedParticipant]:
        """Resolve a list of raw names, preserving order."""
        return [self.resolve_one(name) for name in raw_names]


def summarize_resolution(participants: List[ResolvedParticipant]) -> str:
    """
    Roll a list of resolutions into a single status summary.

    Args:
        participants: Resolved participant outcomes.

    Returns:
        ``"ambiguous"`` if any is ambiguous; else ``"some_unresolved"`` if any
        is unresolved; else ``"all_resolved"`` (also for an empty list).
    """
    statuses = {p.resolution_status for p in participants}
    if AMBIGUOUS in statuses:
        return "ambiguous"
    if UNRESOLVED in statuses:
        return "some_unresolved"
    return "all_resolved"
