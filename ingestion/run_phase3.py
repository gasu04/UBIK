#!/usr/bin/env python3
"""
UBIK Ingestion System - Phase 3 Pipeline Runner

Single entry point that chains the Phase 3 stages over the source corpus and
prints a one-line summary per stage. It is a thin orchestrator: each stage
delegates to the module that already implements it — this file wires them
together and provides the ``--stage`` / ``--dry-run`` / ``--limit`` surface.

Stage map (brief stage -> implementing module):
    1 parse     ProcessorRegistry body extraction + ingest.diarization
    2 enrich    enrich.Enricher (LLM, per-tier endpoint)              [needs vLLM]
    3 resolve   ingest.person_resolver against known_persons.yaml
    4 gate1     interactive_ingest.py (human-in-the-loop)             [human]
    5 write     ingest.mcp_writer.LocalMemoryWriter (dedup + metadata) [needs ChromaDB]

Data flow uses the existing layout (sources/<type>/ -> enriched/*.transcript
-> pending_review/*.resolved.json -> approved/ -> ChromaDB); it does not
introduce a parallel staging tree.

Usage:
    python run_phase3.py                 # all stages
    python run_phase3.py --stage 1       # only stage 1 (offline parse check)
    python run_phase3.py --dry-run       # stages 1-3, no Gate 1 / no write
    python run_phase3.py --limit 8       # at most 8 files per stage

Dependencies: the ingestion package (config, enrich, ingest.*); PyYAML.

Tier classification: Tier 2 (standard, 80% coverage). The runner only
sequences other modules; the silent-failure-critical logic (dedup) lives in
mcp_writer.py and is covered there.

Version: 0.1.0
"""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.ingestion_config import IngestionConfig, load_config
from ingest.diarization import detect_diarization
from ingest.models import IngestItem
from ingest.person_resolver import PersonResolver, summarize_resolution
from ingest.processors import ProcessorRegistry
from ingest.registry import load_known_persons

logger = logging.getLogger("ubik.run_phase3")

__all__ = ["StageSummary", "SOURCE_CONTENT_TYPES", "run", "main"]

# Source subdir -> content-type key (registry). Only dirs with files are run.
SOURCE_CONTENT_TYPES: Dict[str, str] = {
    "tactiq": "transcript_tactiq",
    "gemini": "transcript_gemini",
    "fireflies": "transcript_fireflies",
    "letters": "letter_maestro",
    "memory_notes": "memory_note",
    "constitution": "ethical_constitution",
}

# meeting_type values that must never be written to a collection (quarantine).
_EXCLUDED_MEETING_TYPES = frozenset({"therapy"})

_HASH_CHUNK = 1024 * 1024


@dataclass
class StageSummary:
    """Outcome counts for one stage."""
    stage: int
    name: str
    processed: int = 0
    skipped: int = 0
    quarantined: int = 0

    def line(self) -> str:
        """The brief's one-line stage summary."""
        return (
            f"Stage {self.stage} ({self.name}) complete: "
            f"{self.processed} processed, {self.skipped} skipped, "
            f"{self.quarantined} quarantined"
        )


# =============================================================================
# Shared helpers
# =============================================================================

def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_source_files(
    config: IngestionConfig, limit: Optional[int]
) -> List[Tuple[Path, str]]:
    """Return [(path, content_type)] for every source file, capped at *limit*."""
    out: List[Tuple[Path, str]] = []
    for subdir, content_type in SOURCE_CONTENT_TYPES.items():
        d = config.paths.sources_dir / subdir
        if not d.is_dir():
            continue
        for path in sorted(d.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                out.append((path, content_type))
    return out[:limit] if limit is not None else out


def _parse_transcript_file(path: Path) -> Tuple[Dict[str, Any], str]:
    """Parse an enriched ``.transcript`` (``---\\nyaml---\\n\\nbody``) file.

    Returns:
        (front_matter_dict, body). Front matter is ``{}`` if absent.
    """
    import yaml
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    rest = text[3:]
    front_str, sep, body = rest.partition("\n---")
    if not sep:
        return {}, text
    front = yaml.safe_load(front_str) or {}
    return (front if isinstance(front, dict) else {}), body.lstrip("\n")


# =============================================================================
# Stages
# =============================================================================

async def stage1_parse(
    config: IngestionConfig, limit: Optional[int],
    processors: Optional[ProcessorRegistry] = None,
) -> StageSummary:
    """Stage 1: extract each source body and detect diarization (offline)."""
    procs = processors or ProcessorRegistry()
    summary = StageSummary(1, "parse")
    for path, _ctype in _iter_source_files(config, limit):
        ext = path.suffix.lower()
        if not procs.can_process(ext):
            summary.quarantined += 1
            logger.warning("stage1: no processor for %s", path.name)
            continue
        try:
            item = IngestItem.from_path(path)
            processed = await procs.process(item, path)
            status = detect_diarization(processed.text)
            summary.processed += 1
            logger.info("stage1: parsed %s (diarization=%s)", path.name, status)
        except Exception as exc:  # extraction failure
            summary.quarantined += 1
            logger.error("stage1: failed to parse %s: %s", path.name, exc)
    return summary


async def stage2_enrich(
    config: IngestionConfig, limit: Optional[int], enricher: Any = None,
) -> StageSummary:
    """Stage 2: enrich each source dir via the LLM (needs the vLLM endpoint)."""
    from enrich import EnrichmentStatus, Enricher

    enr = enricher or Enricher.from_config(config)
    summary = StageSummary(2, "enrich")
    for subdir, content_type in SOURCE_CONTENT_TYPES.items():
        d = config.paths.sources_dir / subdir
        if not d.is_dir() or not any(
            p.is_file() and not p.name.startswith(".") for p in d.iterdir()
        ):
            continue
        results = await enr.enrich_directory(d, content_type, limit=limit)
        for r in results:
            if r.status == EnrichmentStatus.ENRICHED:
                summary.processed += 1
            elif r.status == EnrichmentStatus.SKIPPED:
                summary.skipped += 1
            else:  # QUARANTINED or ERROR
                summary.quarantined += 1
    return summary


async def stage3_resolve(
    config: IngestionConfig, limit: Optional[int], resolver: Any = None,
) -> StageSummary:
    """Stage 3: resolve participants in each enriched file -> pending_review/."""
    res = resolver or PersonResolver(load_known_persons())
    summary = StageSummary(3, "resolve")
    enriched_dir = config.paths.enriched_dir
    if not enriched_dir.is_dir():
        return summary
    files = sorted(p for p in enriched_dir.glob("*.transcript"))
    if limit is not None:
        files = files[:limit]
    config.paths.pending_review_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        front, _body = _parse_transcript_file(path)
        if not front:
            summary.skipped += 1
            continue
        raw_names = front.get("participants_detected") or []
        if not isinstance(raw_names, list):
            raw_names = [str(raw_names)]
        parts = res.resolve_all([str(n) for n in raw_names])
        out = {
            "source_file": front.get("_audit", {}).get("source_file", path.stem),
            "source_sha256": front.get("_audit", {}).get("source_sha256", ""),
            "meeting_type": front.get("meeting_type"),
            "voice_corpus_eligible": front.get("voice_corpus_eligible"),
            "participants_resolved": [vars(p) for p in parts],
            "resolution_status_summary": summarize_resolution(parts),
        }
        dest = config.paths.pending_review_dir / f"{path.stem}.resolved.json"
        dest.write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary.processed += 1
    return summary


def stage4_gate(config: IngestionConfig) -> StageSummary:
    """Stage 4: human Gate 1 — counts the queue and points at the review CLI.

    This stage is intentionally not automated: approval is a human decision
    (interactive_ingest.py). The runner reports how many records await review.
    """
    summary = StageSummary(4, "gate1")
    pending = config.paths.pending_review_dir
    if pending.is_dir():
        summary.skipped = len(list(pending.glob("*.resolved.json")))
    logger.info(
        "stage4: %d record(s) awaiting human Gate 1 — run interactive_ingest.py "
        "to approve/flag/quarantine; this runner does not auto-approve.",
        summary.skipped,
    )
    return summary


async def stage5_write(
    config: IngestionConfig, limit: Optional[int], writer: Any = None,
) -> StageSummary:
    """Stage 5: write approved records to ChromaDB (dedup + metadata).

    Therapy is never written (quarantined). Duplicates (same source_sha256)
    count as skipped. Needs a live ChromaDB unless *writer* is injected.
    """
    summary = StageSummary(5, "write")
    approved = config.paths.approved_dir
    if not approved.is_dir():
        return summary
    files = sorted(p for p in approved.glob("*.transcript"))
    if limit is not None:
        files = files[:limit]
    if not files:
        return summary

    resolver = PersonResolver(load_known_persons())

    async def _run(w: Any) -> None:
        for path in files:
            front, body = _parse_transcript_file(path)
            audit = front.get("_audit", {})
            meeting_type = front.get("meeting_type", "unknown")
            if meeting_type in _EXCLUDED_MEETING_TYPES:
                summary.quarantined += 1
                logger.warning("stage5: excluded %s (therapy)", path.name)
                continue
            raw_names = front.get("participants_detected") or []
            if not isinstance(raw_names, list):
                raw_names = [str(raw_names)]
            parts = resolver.resolve_all([str(n) for n in raw_names])
            extra = {
                "diarization_warning": front.get("diarization_warning"),
                "type_inferred_from": front.get("type_inferred_from"),
                "enrichment_confidence": front.get("enrichment_confidence"),
                "resolution_status_summary": summarize_resolution(parts),
                "voice_corpus_eligible": front.get("voice_corpus_eligible"),
                "ingestion_phase": "phase3_enriched",
            }
            result = await w.store_episodic(
                content=body,
                memory_type=str(meeting_type),
                source_file=audit.get("source_file", path.stem),
                source_sha256=audit.get("source_sha256") or None,
                extra_metadata=extra,
            )
            if result.get("status") == "duplicate":
                summary.skipped += 1
            else:
                summary.processed += 1

    if writer is not None:
        await _run(writer)
    else:
        from ingest.mcp_writer import LocalMemoryWriter
        async with LocalMemoryWriter() as w:
            await _run(w)
    return summary


# =============================================================================
# Orchestration
# =============================================================================

async def run(
    config: IngestionConfig,
    *,
    stages: List[int],
    dry_run: bool,
    limit: Optional[int],
) -> List[StageSummary]:
    """Run the requested stages in order, printing each summary line.

    Args:
        config: Resolved ingestion configuration.
        stages: Stage numbers to run (subset of 1..5), in order.
        dry_run: If True, drop stages 4 and 5 (no Gate 1, no write).
        limit: Max files per stage, or None for all.

    Returns:
        The StageSummary for each stage that ran.
    """
    if dry_run:
        stages = [s for s in stages if s <= 3]

    summaries: List[StageSummary] = []
    for stage in stages:
        if stage == 1:
            s = await stage1_parse(config, limit)
        elif stage == 2:
            s = await stage2_enrich(config, limit)
        elif stage == 3:
            s = await stage3_resolve(config, limit)
        elif stage == 4:
            s = stage4_gate(config)
        elif stage == 5:
            s = await stage5_write(config, limit)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        summaries.append(s)
        print(s.line())
    return summaries


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="run_phase3",
        description="Run the UBIK Phase 3 enrichment pipeline.",
    )
    parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3, 4, 5],
        help="Run only this stage (default: all).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run stages 1-3 only (no Gate 1, no ChromaDB write).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N files per stage.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = load_config()
    stages = [args.stage] if args.stage else [1, 2, 3, 4, 5]
    asyncio.run(run(config, stages=stages, dry_run=args.dry_run, limit=args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
