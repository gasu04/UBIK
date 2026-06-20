#!/usr/bin/env python3
"""
UBIK Ingestion System - Enrichment Spike Harness (Phase 3, Checkpoint 3)

THE GATE. Runs the enrichment client over a small directory of raw
transcripts (8-10), then prints a one-screen scoring sheet per file and
writes ``spike_results.md`` for hand-scoring. Phase 4 wiring must not begin
until these results are hand-scored and approved.

For each file the sheet reports:
    - content type / classification
    - confidence
    - diarization_warning (from the enrichment, or the content-type's trust)
    - decisions count
    - participants resolved / unresolved against known_persons.yaml

Field names are read with fallbacks (see qa/schema.md "Fields the spike
scoring sheet reads") so the sheet populates regardless of which spelling
the authoritative schema settles on.

Usage:
    python spike_enrichment.py --dir sources/tactiq --type transcript_tactiq
    python spike_enrichment.py --dir <dir> --type <ct> --limit 10 --force \\
        --out spike_results.md

Logs carry file IDs, timing, and confidence ONLY — never content. The
scoring sheet itself reports participant *labels* (names), which are
metadata, not transcript content.

Tier classification: Tier 1 (critical — its output gates the project).

Version: 0.1.0
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from enrich import (
    EnrichmentResult,
    EnrichmentStatus,
    Enricher,
    PromptNotReadyError,
    SchemaNotReadyError,
)
from ingest.registry import KnownPersonsRegistry, load_known_persons

__all__ = [
    "ScoreCard",
    "build_scorecard",
    "render_terminal",
    "render_results_md",
    "run_spike",
    "main",
]

logger = logging.getLogger("ubik.spike")

# Field-name fallbacks, kept in sync with qa/schema.md.
_TYPE_FIELDS = ("content_type", "type", "classification")
_CONFIDENCE_FIELDS = ("confidence", "enrichment_confidence")
_DIARIZATION_FIELDS = ("diarization_warning", "diarization_trust")
_DECISIONS_FIELDS = ("decisions", "decisions_made")
_PARTICIPANTS_FIELDS = ("participants", "speakers")


@dataclass
class ScoreCard:
    """
    One file's spike scoring row.

    Attributes:
        source_file: Source filename.
        status: EnrichmentStatus.* for the file.
        classification: Detected content type / classification.
        confidence: Top-level confidence (or None).
        diarization_warning: True if the source/enrichment flags untrusted
            speaker attribution.
        decisions_count: Number of decision items extracted.
        participants_resolved: Participant labels that map to known_persons.
        participants_unresolved: Participant labels with no registry match.
        error: Reason for QUARANTINED / ERROR outcomes.
    """
    source_file: str
    status: str
    classification: Optional[str] = None
    confidence: Optional[float] = None
    diarization_warning: bool = False
    decisions_count: int = 0
    participants_resolved: List[str] = field(default_factory=list)
    participants_unresolved: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _first_field(data: Dict[str, Any], names: Tuple[str, ...]) -> Any:
    """Return the value of the first present key in *names*, else None."""
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return None


def _resolve_participants(
    raw_participants: Any, persons: KnownPersonsRegistry
) -> Tuple[List[str], List[str]]:
    """
    Split participant labels into resolved / unresolved against the registry.

    Handles a list of strings, a list of {name|speaker|canonical_name: ...}
    dicts, or a single string.
    """
    labels: List[str] = []
    if isinstance(raw_participants, str):
        labels = [raw_participants]
    elif isinstance(raw_participants, dict):
        labels = [str(v) for v in raw_participants.values()]
    elif isinstance(raw_participants, list):
        for item in raw_participants:
            if isinstance(item, str):
                labels.append(item)
            elif isinstance(item, dict):
                for key in ("canonical_name", "name", "speaker", "label"):
                    if key in item and item[key]:
                        labels.append(str(item[key]))
                        break

    resolved, unresolved = [], []
    for label in labels:
        label = label.strip()
        if not label:
            continue
        if persons.resolve(label):
            resolved.append(label)
        else:
            unresolved.append(label)
    return resolved, unresolved


def _diarization_warning(data: Dict[str, Any]) -> bool:
    """Interpret the diarization field as a boolean warning."""
    value = _first_field(data, _DIARIZATION_FIELDS)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        # trust levels: none/partial -> warn; full -> ok.
        return value.strip().lower() in ("none", "partial", "low", "true", "yes")
    return False


def build_scorecard(
    result: EnrichmentResult, persons: KnownPersonsRegistry
) -> ScoreCard:
    """Build a ScoreCard from an EnrichmentResult."""
    if result.status != EnrichmentStatus.ENRICHED or not result.enrichment:
        return ScoreCard(
            source_file=result.source_file,
            status=result.status,
            error=result.error,
        )
    data = result.enrichment
    decisions = _first_field(data, _DECISIONS_FIELDS)
    decisions_count = len(decisions) if isinstance(decisions, list) else (
        1 if decisions else 0
    )
    resolved, unresolved = _resolve_participants(
        _first_field(data, _PARTICIPANTS_FIELDS), persons
    )
    classification = _first_field(data, _TYPE_FIELDS)
    return ScoreCard(
        source_file=result.source_file,
        status=result.status,
        classification=str(classification) if classification is not None else None,
        confidence=result.confidence,
        diarization_warning=_diarization_warning(data),
        decisions_count=decisions_count,
        participants_resolved=resolved,
        participants_unresolved=unresolved,
    )


def _fmt_conf(conf: Optional[float]) -> str:
    return f"{conf:.2f}" if isinstance(conf, float) else "—"


def render_terminal(cards: List[ScoreCard]) -> str:
    """Render the one-screen scoring sheet for the terminal."""
    lines = [
        "=" * 78,
        "ENRICHMENT SPIKE — scoring sheet",
        "=" * 78,
        f"{'file':<28} {'status':<11} {'type':<14} {'conf':>5} "
        f"{'diar':>4} {'dec':>3} {'part(r/u)':>9}",
        "-" * 78,
    ]
    for c in cards:
        part = f"{len(c.participants_resolved)}/{len(c.participants_unresolved)}"
        diar = "WARN" if c.diarization_warning else "ok"
        lines.append(
            f"{c.source_file[:28]:<28} {c.status:<11} "
            f"{(c.classification or '—')[:14]:<14} {_fmt_conf(c.confidence):>5} "
            f"{diar:>4} {c.decisions_count:>3} {part:>9}"
        )
        if c.participants_unresolved:
            lines.append(f"    unresolved: {', '.join(c.participants_unresolved)}")
        if c.error:
            lines.append(f"    ! {c.error[:70]}")
    lines.append("-" * 78)
    enriched = sum(1 for c in cards if c.status == EnrichmentStatus.ENRICHED)
    quarantined = sum(1 for c in cards if c.status == EnrichmentStatus.QUARANTINED)
    errored = sum(1 for c in cards if c.status == EnrichmentStatus.ERROR)
    skipped = sum(1 for c in cards if c.status == EnrichmentStatus.SKIPPED)
    lines.append(
        f"{len(cards)} files: {enriched} enriched, {quarantined} quarantined, "
        f"{errored} error, {skipped} skipped"
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def render_results_md(cards: List[ScoreCard], *, source_dir: str, content_type: str) -> str:
    """Render the hand-scoring spike_results.md document."""
    now = datetime.now(timezone.utc).isoformat()
    enriched = sum(1 for c in cards if c.status == EnrichmentStatus.ENRICHED)
    quarantined = sum(1 for c in cards if c.status == EnrichmentStatus.QUARANTINED)
    errored = sum(1 for c in cards if c.status == EnrichmentStatus.ERROR)

    out: List[str] = [
        "# Enrichment Spike Results",
        "",
        f"- **Generated:** {now}",
        f"- **Source dir:** `{source_dir}`",
        f"- **Content type:** `{content_type}`",
        f"- **Files:** {len(cards)} "
        f"({enriched} enriched, {quarantined} quarantined, {errored} error)",
        "",
        "> **CHECKPOINT 3 GATE.** Hand-score each file below. Phase 4 wiring "
        "does not begin until these scores are reviewed and approved.",
        "",
        "## Scoring table",
        "",
        "| File | Status | Type | Conf | Diar | Decisions | Part R/U | "
        "Type ✓ | Conf ✓ | Persons ✓ | Decisions ✓ | Score (1-5) | Notes |",
        "|------|--------|------|------|------|-----------|----------|"
        "--------|--------|-----------|-------------|-------------|-------|",
    ]
    for c in cards:
        part = f"{len(c.participants_resolved)}/{len(c.participants_unresolved)}"
        diar = "WARN" if c.diarization_warning else "ok"
        out.append(
            f"| {c.source_file} | {c.status} | {c.classification or '—'} | "
            f"{_fmt_conf(c.confidence)} | {diar} | {c.decisions_count} | {part} | "
            "| | | | | |"
        )

    out += ["", "## Per-file detail", ""]
    for c in cards:
        out.append(f"### {c.source_file}")
        out.append("")
        out.append(f"- **Status:** {c.status}")
        if c.error:
            out.append(f"- **Error:** {c.error}")
        out.append(f"- **Classification:** {c.classification or '—'}")
        out.append(f"- **Confidence:** {_fmt_conf(c.confidence)}")
        out.append(f"- **Diarization warning:** {c.diarization_warning}")
        out.append(f"- **Decisions extracted:** {c.decisions_count}")
        out.append(
            f"- **Participants resolved ({len(c.participants_resolved)}):** "
            f"{', '.join(c.participants_resolved) or '—'}"
        )
        out.append(
            f"- **Participants unresolved ({len(c.participants_unresolved)}):** "
            f"{', '.join(c.participants_unresolved) or '—'}"
        )
        out.append("")
        out.append("**Hand-score:**")
        out.append("")
        out.append("- Classification correct? (Y/N): ")
        out.append("- Confidence calibrated? (Y/N): ")
        out.append("- Participants correct & complete? (Y/N): ")
        out.append("- Decisions accurate? (Y/N): ")
        out.append("- Overall (1-5): ")
        out.append("- Notes: ")
        out.append("")

    out += [
        "## Gate decision",
        "",
        "- [ ] **APPROVED** — proceed to Phase 4 wiring",
        "- [ ] **REJECTED** — revise prompt / schema and re-run spike",
        "",
        "Reviewer: ____________________   Date: ____________",
        "",
    ]
    return "\n".join(out)


async def run_spike(
    directory: Path,
    content_type: str,
    *,
    privacy_tier: Optional[str] = None,
    limit: Optional[int] = None,
    force: bool = False,
    enricher: Optional[Enricher] = None,
) -> List[ScoreCard]:
    """
    Run enrichment over *directory* and return scorecards.

    Args:
        directory: Directory of raw transcripts to enrich.
        content_type: Registered content-type key.
        privacy_tier: Optional routing tier override.
        limit: Max number of files to process.
        force: Re-enrich even if already enriched at this prompt version.
        enricher: Injected Enricher (built from config if omitted).

    Returns:
        One ScoreCard per file processed.
    """
    enricher = enricher or Enricher.from_config()
    persons = enricher.known_persons or load_known_persons()
    results = await enricher.enrich_directory(
        directory, content_type,
        privacy_tier=privacy_tier, force=force, limit=limit,
    )
    return [build_scorecard(r, persons) for r in results]


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = argparse.ArgumentParser(
        prog="spike_enrichment",
        description="Run the enrichment spike and write spike_results.md",
    )
    parser.add_argument("--dir", required=True, help="Directory of raw transcripts")
    parser.add_argument("--type", dest="content_type", required=True,
                        help="Registered content-type key (e.g. transcript_tactiq)")
    parser.add_argument("--privacy-tier", default=None,
                        help="Override routing tier (default: per content type)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max files to process")
    parser.add_argument("--force", action="store_true",
                        help="Re-enrich files already enriched at this prompt version")
    parser.add_argument("--out", default="spike_results.md",
                        help="Output markdown path (default: spike_results.md)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        cards = asyncio.run(run_spike(
            Path(args.dir), args.content_type,
            privacy_tier=args.privacy_tier, limit=args.limit, force=args.force,
        ))
    except (PromptNotReadyError, SchemaNotReadyError) as e:
        print(f"Spike blocked: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — CLI boundary
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(render_terminal(cards))
    md = render_results_md(cards, source_dir=args.dir, content_type=args.content_type)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"\nWrote {args.out} ({len(cards)} files) — hand-score before Phase 4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
