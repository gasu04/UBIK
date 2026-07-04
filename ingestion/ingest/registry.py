"""
UBIK Ingestion System - Registry Loaders

Loads and validates the two ingestion registries:

- ``registry/known_persons.yaml`` — canonical identity table used to
  normalise speaker labels and mentions to one Neo4j node per person.
- ``registry/content_types.yaml`` — routing table mapping each content
  type to a parser, enrichment level, and target ChromaDB collections.

Both loaders fail loud: any structural problem (missing field, wrong
type, bad enum value, duplicate name/alias) raises
:class:`RegistryValidationError` naming the offending entry. A registry
that loads is a registry the pipeline can trust.

Also provides the one-time CSV → YAML converter for the Known_Persons
Google Sheet export::

    python -m ingest.registry --convert Known_Persons.csv

Usage:
    from ingest.registry import load_known_persons, load_content_types

    persons = load_known_persons()           # default registry path
    ctypes = load_content_types()
    canonical = persons.resolve("Leti")      # -> "Leticia Zuno"

Dependencies:
    PyYAML >= 5.0

Tier classification: Tier 2 (standard, 80% coverage). Failures are loud
by design — every validation problem raises before the pipeline runs.

Version: 0.1.0
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

__all__ = [
    'UbikIngestionError',
    'RegistryValidationError',
    'KnownPerson',
    'KnownPersonsRegistry',
    'ContentTypeSpec',
    'ContentTypeRegistry',
    'load_known_persons',
    'load_content_types',
    'convert_known_persons_csv',
    'DEFAULT_REGISTRY_DIR',
]

# Registry data lives in ingestion/registry/, one level above this package.
DEFAULT_REGISTRY_DIR = Path(__file__).resolve().parent.parent / "registry"

PRIVACY_TIERS = {"private", "therapy", "family", "business"}
ACCESS_LEVELS = {"admin", "restricted", "full", "self_and_descendants"}
PARSERS = {"transcript", "letter", "note", "constitution", "qa_pair"}
ENRICHMENT_LEVELS = {"required", "light", "none"}
DIARIZATION_TRUST_LEVELS = {"none", "partial", "full"}
DEFAULT_SENSITIVE_VALUES = {"review", True, False}


class UbikIngestionError(Exception):
    """Base exception for all UBIK ingestion errors."""


class RegistryValidationError(UbikIngestionError):
    """A registry file is malformed; message names the offending entry."""


# =============================================================================
# Known Persons
# =============================================================================

@dataclass(frozen=True)
class KnownPerson:
    """
    One validated entry from the Known_Persons registry.

    Attributes:
        canonical_name: Full name, unique across the registry.
        aliases: Alternate spellings/nicknames, unique globally.
        relationship: Relationship to Gines (self, wife, son, therapist, ...).
        role: Function in the corpus (protagonist, core_family, professional,
            next_gen, business, ...).
        privacy_tier: Content sensitivity class — one of
            "private", "therapy", "family", "business" (PRIVACY_TIERS).
        ubik_access_level: What this person may access in UBIK — one of
            "admin", "restricted", "full", "self_and_descendants"
            (ACCESS_LEVELS).
    """
    canonical_name: str
    aliases: tuple
    relationship: str
    role: str
    privacy_tier: str
    ubik_access_level: str


@dataclass
class KnownPersonsRegistry:
    """
    Validated Known_Persons registry with alias resolution.

    Attributes:
        persons: All validated entries, in file order.

    Example:
        >>> registry = load_known_persons()
        >>> registry.resolve("Leti")
        'Leticia Zuno'
        >>> registry.resolve("A Stranger") is None
        True
    """
    persons: List[KnownPerson] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_name: Dict[str, KnownPerson] = {}
        for person in self.persons:
            self._by_name[person.canonical_name.lower()] = person
            for alias in person.aliases:
                self._by_name[alias.lower()] = person

    def resolve(self, name: str) -> Optional[str]:
        """
        Resolve any name or alias to its canonical name.

        Args:
            name: Name as it appears in source content (case-insensitive).

        Returns:
            Canonical full name, or None if the name is unknown.
        """
        person = self._by_name.get(name.strip().lower())
        return person.canonical_name if person else None

    def get(self, name: str) -> Optional[KnownPerson]:
        """Return the full entry for a name or alias, or None if unknown."""
        return self._by_name.get(name.strip().lower())

    def alias_map(self) -> Dict[str, str]:
        """
        Return {alias_or_name: canonical_name} for every known label.

        Note:
            Drop-in superset of ``transcript_processor.PERSON_ALIASES``;
            pass it as ``person_aliases`` to ``to_neo4j_operations()``.
        """
        return {label: p.canonical_name for label, p in self._by_name.items()}


def _require(entry: Dict[str, Any], key: str, kind: type, where: str) -> Any:
    """Fetch a required key from a registry entry or raise loud."""
    if key not in entry:
        raise RegistryValidationError(f"{where}: missing required field '{key}'")
    value = entry[key]
    if not isinstance(value, kind):
        raise RegistryValidationError(
            f"{where}: field '{key}' must be {kind.__name__}, "
            f"got {type(value).__name__} ({value!r})"
        )
    return value


def load_known_persons(path: Optional[Path] = None) -> KnownPersonsRegistry:
    """
    Load and validate the Known_Persons registry.

    Args:
        path: Registry YAML path. Defaults to
            ``ingestion/registry/known_persons.yaml``.

    Returns:
        Validated KnownPersonsRegistry.

    Raises:
        FileNotFoundError: If the registry file does not exist.
        RegistryValidationError: On any malformed entry — missing field,
            wrong type, empty/duplicate canonical name or alias, or
            invalid privacy_tier.

    Example:
        >>> registry = load_known_persons()
        >>> registry.resolve("Leti")
        'Leticia Zuno'
    """
    path = path or (DEFAULT_REGISTRY_DIR / "known_persons.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Known persons registry not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "persons" not in data:
        raise RegistryValidationError(
            f"{path.name}: top level must be a mapping with a 'persons' list"
        )
    raw_persons = data["persons"]
    if not isinstance(raw_persons, list) or not raw_persons:
        raise RegistryValidationError(f"{path.name}: 'persons' must be a non-empty list")

    persons: List[KnownPerson] = []
    seen_labels: Dict[str, str] = {}  # lowercase label -> canonical that owns it

    for i, entry in enumerate(raw_persons):
        where = f"{path.name}: persons[{i}]"
        if not isinstance(entry, dict):
            raise RegistryValidationError(f"{where}: entry must be a mapping")

        canonical = _require(entry, "canonical_name", str, where).strip()
        if not canonical:
            raise RegistryValidationError(f"{where}: canonical_name is empty")
        where = f"{path.name}: persons[{i}] ({canonical})"

        aliases = _require(entry, "aliases", list, where)
        for alias in aliases:
            if not isinstance(alias, str) or not alias.strip():
                raise RegistryValidationError(
                    f"{where}: aliases must be non-empty strings, got {alias!r}"
                )

        relationship = _require(entry, "relationship", str, where).strip()
        role = _require(entry, "role", str, where).strip()
        if not relationship or not role:
            raise RegistryValidationError(f"{where}: relationship/role must be non-empty")

        privacy_tier = _require(entry, "privacy_tier", str, where).strip().lower()
        if privacy_tier not in PRIVACY_TIERS:
            raise RegistryValidationError(
                f"{where}: privacy_tier must be one of {sorted(PRIVACY_TIERS)}, "
                f"got {privacy_tier!r}"
            )

        access_level = _require(entry, "ubik_access_level", str, where).strip().lower()
        if access_level not in ACCESS_LEVELS:
            raise RegistryValidationError(
                f"{where}: ubik_access_level must be one of {sorted(ACCESS_LEVELS)}, "
                f"got {access_level!r}"
            )

        for label in [canonical, *aliases]:
            key = label.strip().lower()
            owner = seen_labels.get(key)
            if owner is not None and owner != canonical:
                raise RegistryValidationError(
                    f"{where}: label {label!r} already used by entry {owner!r}"
                )
            seen_labels[key] = canonical

        persons.append(KnownPerson(
            canonical_name=canonical,
            aliases=tuple(a.strip() for a in aliases),
            relationship=relationship,
            role=role,
            privacy_tier=privacy_tier,
            ubik_access_level=access_level,
        ))

    return KnownPersonsRegistry(persons=persons)


# =============================================================================
# Content Types
# =============================================================================

@dataclass(frozen=True)
class ContentTypeSpec:
    """
    One validated entry from the content_types registry.

    Attributes:
        name: Content type key (e.g., "transcript_tactiq").
        parser: Parser route (transcript, letter, note, constitution, qa_pair).
        enrichment: Enrichment level (required, light, none).
        target_collections: ChromaDB collections; empty for non-ingestion routes.
        diarization_trust: Speaker-attribution confidence, if applicable.
        default_sensitive: Gate 1 default for contains_sensitive.
        voice_corpus_eligible: Whether text may feed voice/DPO training.
        metadata_overrides: Forced metadata values at storage time.
        route_to: Alternate destination bypassing the IngestPipeline.
    """
    name: str
    parser: str
    enrichment: str
    target_collections: tuple
    diarization_trust: Optional[str] = None
    default_sensitive: Any = None
    voice_corpus_eligible: bool = False
    metadata_overrides: Dict[str, Any] = field(default_factory=dict)
    route_to: Optional[str] = None


@dataclass
class ContentTypeRegistry:
    """
    Validated content-type routing table.

    Example:
        >>> registry = load_content_types()
        >>> registry.get("transcript_tactiq").parser
        'transcript'
    """
    content_types: Dict[str, ContentTypeSpec] = field(default_factory=dict)

    def get(self, name: str) -> ContentTypeSpec:
        """
        Return the spec for a content type.

        Raises:
            RegistryValidationError: If the content type is not registered.
        """
        try:
            return self.content_types[name]
        except KeyError:
            raise RegistryValidationError(
                f"Unknown content type {name!r}; registered: "
                f"{sorted(self.content_types)}"
            ) from None


def load_content_types(path: Optional[Path] = None) -> ContentTypeRegistry:
    """
    Load and validate the content-type registry.

    Args:
        path: Registry YAML path. Defaults to
            ``ingestion/registry/content_types.yaml``.

    Returns:
        Validated ContentTypeRegistry.

    Raises:
        FileNotFoundError: If the registry file does not exist.
        RegistryValidationError: On any malformed entry — missing field,
            invalid parser/enrichment/diarization_trust value, or a
            voice-corpus-eligible source whose diarization is untrusted.

    Example:
        >>> registry = load_content_types()
        >>> registry.get("ethical_constitution").metadata_overrides
        {'stability': 'core', 'importance': 1.0}
    """
    path = path or (DEFAULT_REGISTRY_DIR / "content_types.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Content types registry not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "content_types" not in data:
        raise RegistryValidationError(
            f"{path.name}: top level must be a mapping with a 'content_types' mapping"
        )
    raw_types = data["content_types"]
    if not isinstance(raw_types, dict) or not raw_types:
        raise RegistryValidationError(
            f"{path.name}: 'content_types' must be a non-empty mapping"
        )

    specs: Dict[str, ContentTypeSpec] = {}

    for name, entry in raw_types.items():
        where = f"{path.name}: content_types.{name}"
        if not isinstance(entry, dict):
            raise RegistryValidationError(f"{where}: entry must be a mapping")

        parser = _require(entry, "parser", str, where)
        if parser not in PARSERS:
            raise RegistryValidationError(
                f"{where}: parser must be one of {sorted(PARSERS)}, got {parser!r}"
            )

        enrichment = _require(entry, "enrichment", str, where)
        if enrichment not in ENRICHMENT_LEVELS:
            raise RegistryValidationError(
                f"{where}: enrichment must be one of {sorted(ENRICHMENT_LEVELS)}, "
                f"got {enrichment!r}"
            )

        collections = _require(entry, "target_collections", list, where)
        for coll in collections:
            if not isinstance(coll, str) or not coll.strip():
                raise RegistryValidationError(
                    f"{where}: target_collections entries must be non-empty "
                    f"strings, got {coll!r}"
                )

        diarization_trust = entry.get("diarization_trust")
        if diarization_trust is not None and diarization_trust not in DIARIZATION_TRUST_LEVELS:
            raise RegistryValidationError(
                f"{where}: diarization_trust must be one of "
                f"{sorted(DIARIZATION_TRUST_LEVELS)}, got {diarization_trust!r}"
            )

        default_sensitive = entry.get("default_sensitive")
        if default_sensitive is not None and default_sensitive not in DEFAULT_SENSITIVE_VALUES:
            raise RegistryValidationError(
                f"{where}: default_sensitive must be 'review', true, or false, "
                f"got {default_sensitive!r}"
            )

        voice_eligible = entry.get("voice_corpus_eligible", False)
        if not isinstance(voice_eligible, bool):
            raise RegistryValidationError(
                f"{where}: voice_corpus_eligible must be a boolean"
            )
        # The voice corpus invariant: untrusted diarization can never feed
        # voice training, no matter what the YAML claims.
        if voice_eligible and diarization_trust in ("none", "partial"):
            raise RegistryValidationError(
                f"{where}: voice_corpus_eligible=true requires "
                f"diarization_trust='full' (got {diarization_trust!r})"
            )

        metadata_overrides = entry.get("metadata_overrides", {})
        if not isinstance(metadata_overrides, dict):
            raise RegistryValidationError(
                f"{where}: metadata_overrides must be a mapping"
            )

        route_to = entry.get("route_to")
        if route_to is not None and (not isinstance(route_to, str) or not route_to.strip()):
            raise RegistryValidationError(
                f"{where}: route_to must be a non-empty string"
            )
        if not collections and route_to is None:
            raise RegistryValidationError(
                f"{where}: empty target_collections requires a route_to destination"
            )

        specs[name] = ContentTypeSpec(
            name=name,
            parser=parser,
            enrichment=enrichment,
            target_collections=tuple(collections),
            diarization_trust=diarization_trust,
            default_sensitive=default_sensitive,
            voice_corpus_eligible=voice_eligible,
            metadata_overrides=dict(metadata_overrides),
            route_to=route_to,
        )

    return ContentTypeRegistry(content_types=specs)


# =============================================================================
# CSV -> YAML conversion (Known_Persons Google Sheet export)
# =============================================================================

def convert_known_persons_csv(csv_path: Path, yaml_path: Optional[Path] = None) -> Path:
    """
    Convert a Known_Persons Google Sheet CSV export to the YAML registry.

    Expected CSV header (case-insensitive, order-free):
        canonical_name, aliases, relationship, role, privacy_tier,
        ubik_access_level
    The aliases column is split on commas or semicolons. An empty
    aliases cell means the person has no aliases.

    Args:
        csv_path: Path to the CSV export.
        yaml_path: Output path. Defaults to
            ``ingestion/registry/known_persons.yaml``.

    Returns:
        Path of the written YAML file.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        RegistryValidationError: If the CSV is missing required columns,
            or the converted registry fails validation (the output is
            validated by load_known_persons before this function returns).

    Note:
        Writes to a temp file and validates it before replacing any
        existing registry, so a bad CSV can't clobber a good registry.
    """
    csv_path = Path(csv_path)
    yaml_path = Path(yaml_path) if yaml_path else (DEFAULT_REGISTRY_DIR / "known_persons.yaml")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV export not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RegistryValidationError(f"{csv_path.name}: empty CSV")
        columns = {name.strip().lower(): name for name in reader.fieldnames}
        required = [
            "canonical_name", "aliases", "relationship", "role",
            "privacy_tier", "ubik_access_level",
        ]
        missing = [c for c in required if c not in columns]
        if missing:
            raise RegistryValidationError(
                f"{csv_path.name}: missing required columns {missing}; "
                f"found {sorted(columns)}"
            )

        persons = []
        for row in reader:
            aliases_raw = (row[columns["aliases"]] or "").replace(";", ",")
            persons.append({
                "canonical_name": (row[columns["canonical_name"]] or "").strip(),
                "aliases": [a.strip() for a in aliases_raw.split(",") if a.strip()],
                "relationship": (row[columns["relationship"]] or "").strip(),
                "role": (row[columns["role"]] or "").strip(),
                "privacy_tier": (row[columns["privacy_tier"]] or "").strip().lower(),
                "ubik_access_level": (row[columns["ubik_access_level"]] or "").strip().lower(),
            })

    header = (
        "# UBIK Known Persons Registry — generated from "
        f"{csv_path.name} by ingest/registry.py\n"
        "# Do not edit by hand if the Google Sheet remains the source of truth.\n"
    )
    body = yaml.safe_dump({"persons": persons}, sort_keys=False, allow_unicode=True)

    tmp_path = yaml_path.with_suffix(".yaml.tmp")
    tmp_path.write_text(header + body, encoding="utf-8")
    try:
        load_known_persons(tmp_path)  # fail loud before replacing the registry
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(yaml_path)
    return yaml_path


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: validate registries or convert the persons CSV."""
    parser = argparse.ArgumentParser(
        prog="ingest.registry",
        description="Validate UBIK registries or convert Known_Persons CSV to YAML",
    )
    parser.add_argument(
        "--convert", metavar="CSV",
        help="Convert a Known_Persons CSV export to registry/known_persons.yaml",
    )
    args = parser.parse_args(argv)

    if args.convert:
        out = convert_known_persons_csv(Path(args.convert))
        print(f"Wrote and validated {out}")
        return 0

    persons = load_known_persons()
    ctypes = load_content_types()
    print(f"known_persons.yaml: {len(persons.persons)} entries OK")
    print(f"content_types.yaml: {len(ctypes.content_types)} entries OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
