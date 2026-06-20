"""
Unit tests for spike_enrichment.py — the Phase 3 / Checkpoint 3 gate.

Covers scorecard construction (field-name fallbacks, participant
resolution against known_persons, diarization interpretation), terminal
and markdown rendering, and an end-to-end run with a mocked Enricher.
"""

import httpx
import pytest

import spike_enrichment as spike
from config.ingestion_config import (
    EndpointConfig,
    GateThresholds,
    IngestionConfig,
    PathsConfig,
)
from enrich import EnrichmentResult, EnrichmentStatus, Enricher, _RetryPolicy
from ingest.registry import load_content_types, load_known_persons
from spike_enrichment import (
    ScoreCard,
    build_scorecard,
    render_results_md,
    render_terminal,
    run_spike,
)

PERSONS = load_known_persons()


def _enriched_result(enrichment, confidence=0.9, name="a.txt"):
    return EnrichmentResult(
        source_file=name, source_sha256="abc", content_type="transcript_tactiq",
        privacy_tier="private", status=EnrichmentStatus.ENRICHED,
        confidence=confidence, enrichment=enrichment,
    )


# =============================================================================
# build_scorecard
# =============================================================================

def test_scorecard_resolves_participants():
    card = build_scorecard(_enriched_result({
        "content_type": "transcript",
        "confidence": 0.9,
        "participants": ["Ginés", "Adrian", "Stranger Bob"],
        "decisions": ["X", "Y"],
        "diarization_warning": True,
    }), PERSONS)
    assert card.participants_resolved == ["Ginés", "Adrian"]
    assert card.participants_unresolved == ["Stranger Bob"]
    assert card.decisions_count == 2
    assert card.diarization_warning is True
    assert card.classification == "transcript"


def test_scorecard_field_fallbacks():
    """decisions_made / speakers / type spellings still populate."""
    card = build_scorecard(_enriched_result({
        "type": "therapy",
        "enrichment_confidence": 0.7,
        "speakers": ["Leti", "Maggie"],
        "decisions_made": ["a", "b", "c"],
        "diarization_trust": "partial",
    }, confidence=None), PERSONS)
    assert card.classification == "therapy"
    assert card.decisions_count == 3
    assert card.diarization_warning is True  # "partial" -> warn
    assert set(card.participants_resolved) == {"Leti", "Maggie"}


def test_scorecard_participant_dicts():
    card = build_scorecard(_enriched_result({
        "content_type": "transcript",
        "confidence": 0.8,
        "participants": [{"name": "Ginés"}, {"speaker": "Nadie"}],
        "decisions": [],
        "diarization_warning": False,
    }), PERSONS)
    assert card.participants_resolved == ["Ginés"]
    assert card.participants_unresolved == ["Nadie"]
    assert card.decisions_count == 0


def test_scorecard_quarantined_has_no_fields():
    result = EnrichmentResult(
        source_file="bad.txt", source_sha256="z", content_type="transcript_tactiq",
        privacy_tier="private", status=EnrichmentStatus.QUARANTINED,
        error="schema validation failed: <root>: ...",
    )
    card = build_scorecard(result, PERSONS)
    assert card.status == EnrichmentStatus.QUARANTINED
    assert card.classification is None
    assert card.error


# =============================================================================
# Rendering
# =============================================================================

def _sample_cards():
    return [
        ScoreCard(
            source_file="a.txt", status=EnrichmentStatus.ENRICHED,
            classification="transcript", confidence=0.91,
            diarization_warning=True, decisions_count=2,
            participants_resolved=["Ginés"], participants_unresolved=["Bob"],
        ),
        ScoreCard(
            source_file="b.txt", status=EnrichmentStatus.QUARANTINED,
            error="schema validation failed",
        ),
    ]


def test_render_terminal_contains_summary():
    out = render_terminal(_sample_cards())
    assert "ENRICHMENT SPIKE" in out
    assert "1 enriched, 1 quarantined" in out
    assert "unresolved: Bob" in out


def test_render_results_md_is_scorable():
    md = render_results_md(
        _sample_cards(), source_dir="sources/tactiq", content_type="transcript_tactiq"
    )
    assert "# Enrichment Spike Results" in md
    assert "CHECKPOINT 3 GATE" in md
    assert "Overall (1-5):" in md
    assert "APPROVED" in md and "REJECTED" in md
    assert "| a.txt |" in md


# =============================================================================
# End-to-end run with a mocked Enricher
# =============================================================================

VALID_YAML = (
    "content_type: transcript\nconfidence: 0.88\n"
    "participants:\n  - Ginés\n  - Forastero\n"
    "decisions:\n  - Hacer algo\n"
    "diarization_warning: true\n"
)

SCHEMA = {
    "type": "object",
    "required": ["content_type", "confidence", "participants",
                 "decisions", "diarization_warning"],
    "properties": {
        "content_type": {"type": "string", "enum": ["transcript", "note"]},
        "confidence": {"type": "number"},
        "participants": {"type": "array"},
        "decisions": {"type": "array"},
        "diarization_warning": {"type": "boolean"},
    },
}


@pytest.mark.asyncio
async def test_run_spike_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("UBIK_INGESTION_ROOT", str(tmp_path / "ingestion"))
    config = IngestionConfig(
        endpoints=EndpointConfig(
            sensitive_endpoint="http://test:8002/v1",
            standard_endpoint="http://test:8002/v1",
            model="test-model", request_timeout_seconds=5.0,
        ),
        paths=PathsConfig.from_env(),
        gates=GateThresholds(),
        prompt_version="v1.0.0",
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "one.txt").write_text("Ginés: hola.", encoding="utf-8")
    (raw_dir / "two.txt").write_text("Ginés: adiós.", encoding="utf-8")

    content = "```yaml\n" + VALID_YAML + "```"
    enr = Enricher(
        config=config,
        prompt_template="Persons:\n{{KNOWN_PERSONS}}\nBody:\n{{CONTENT}}",
        schema=SCHEMA,
        known_persons=load_known_persons(),
        content_types=load_content_types(),
        retry=_RetryPolicy(max_retries=2, base_seconds=0.0, max_seconds=0.0),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={
                    "choices": [{"message": {"content": content}}]
                })
            )
        ),
    )

    cards = await run_spike(raw_dir, "transcript_tactiq", enricher=enr)
    assert len(cards) == 2
    for c in cards:
        assert c.status == EnrichmentStatus.ENRICHED
        assert c.participants_resolved == ["Ginés"]
        assert c.participants_unresolved == ["Forastero"]
        assert c.decisions_count == 1
