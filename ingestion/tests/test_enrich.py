"""
Unit tests for enrich.py — the Phase 3 enrichment client.

No network: the LLM endpoint is faked with httpx.MockTransport. Covers
prompt/schema loading (fail-loud on placeholders), reasoning-block
stripping, YAML-fence parsing, schema validation, enriched-output and
quarantine routing, resumability, retry/backoff, and endpoint failure.
"""

import json
from pathlib import Path

import httpx
import pytest
import yaml

from config.ingestion_config import (
    EndpointConfig,
    GateThresholds,
    IngestionConfig,
    PathsConfig,
)
from enrich import (
    EnrichmentManifest,
    EnrichmentStatus,
    Enricher,
    PromptNotReadyError,
    SchemaNotReadyError,
    _RetryPolicy,
    load_enrichment_prompt,
    load_enrichment_schema,
)
from ingest.registry import load_content_types, load_known_persons


# =============================================================================
# Fixtures
# =============================================================================

SCHEMA = {
    "type": "object",
    "required": [
        "content_type", "confidence", "participants",
        "decisions", "diarization_warning",
    ],
    "properties": {
        "content_type": {"type": "string", "enum": ["transcript", "therapy", "note"]},
        "confidence": {"type": "number"},
        "participants": {"type": "array"},
        "decisions": {"type": "array"},
        "diarization_warning": {"type": "boolean"},
    },
    "additionalProperties": True,
}

PROMPT = (
    "You are the UBIK enrichment analyst. Work Spanish-first.\n\n"
    "Known persons:\n{{KNOWN_PERSONS}}\n\n"
    "Source hints:\n{{SOURCE_HINTS}}\n\n"
    "Schema:\n{{SCHEMA}}\n\n"
    "Transcript:\n{{CONTENT}}\n"
)

VALID_YAML = (
    "content_type: transcript\n"
    "confidence: 0.91\n"
    "participants:\n  - Ginés\n  - Adrian\n  - Stranger Bob\n"
    "decisions:\n  - Hacer X\n  - Hacer Y\n"
    "diarization_warning: true\n"
)


def _completion(content: str) -> dict:
    """A minimal OpenAI chat-completion response body."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture()
def config(tmp_path, monkeypatch):
    """IngestionConfig rooted in a temp workspace, with a model set."""
    monkeypatch.setenv("UBIK_INGESTION_ROOT", str(tmp_path / "ingestion"))
    return IngestionConfig(
        endpoints=EndpointConfig(
            sensitive_endpoint="http://test-somatic:8002/v1",
            standard_endpoint="http://test-somatic:8002/v1",
            lan_fallback_endpoint=None,
            model="test-model",
            request_timeout_seconds=5.0,
        ),
        paths=PathsConfig.from_env(),
        gates=GateThresholds(),
        prompt_version="v1.0.0",
    )


def _enricher(config, handler):
    return Enricher(
        config=config,
        prompt_template=PROMPT,
        schema=SCHEMA,
        known_persons=load_known_persons(),
        content_types=load_content_types(),
        retry=_RetryPolicy(max_retries=3, base_seconds=0.0, max_seconds=0.0),
        client=_mock_client(handler),
    )


def _raw_file(tmp_path, name="meeting.txt", text="Ginés: hola. Adrian: qué tal."):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# =============================================================================
# Prompt / schema loading
# =============================================================================

def test_load_prompt_rejects_placeholder(tmp_path):
    p = tmp_path / "prompt.md"
    p.write_text("Some text\nPLACEHOLDER\n", encoding="utf-8")
    with pytest.raises(PromptNotReadyError):
        load_enrichment_prompt(p)


def test_load_prompt_rejects_missing(tmp_path):
    with pytest.raises(PromptNotReadyError):
        load_enrichment_prompt(tmp_path / "nope.md")


def test_load_prompt_ok(tmp_path):
    p = tmp_path / "prompt.md"
    p.write_text(PROMPT, encoding="utf-8")
    assert "{{CONTENT}}" in load_enrichment_prompt(p)


def test_load_schema_extracts_json_fence(tmp_path):
    p = tmp_path / "schema.md"
    p.write_text("# Schema\n\n```json\n" + json.dumps(SCHEMA) + "\n```\n", encoding="utf-8")
    loaded = load_enrichment_schema(p)
    assert loaded["required"] == SCHEMA["required"]


def test_load_schema_rejects_placeholder(tmp_path):
    p = tmp_path / "schema.md"
    p.write_text('```json\n{"PLACEHOLDER": "x"}\n```\n', encoding="utf-8")
    with pytest.raises(SchemaNotReadyError):
        load_enrichment_schema(p)


def test_load_schema_rejects_no_fence(tmp_path):
    p = tmp_path / "schema.md"
    p.write_text("# Schema\n\nno fenced block here\n", encoding="utf-8")
    with pytest.raises(SchemaNotReadyError):
        load_enrichment_schema(p)


def test_shipped_prompt_and_schema_are_populated():
    """The shipped prompt + schema are real (Phase 3) and load cleanly.

    Inverts the former placeholder check: enrichment_v1.md and qa/schema.md
    were populated with the authoritative prompt/schema, so they must now load
    without raising and carry the Phase 3 contract.
    """
    from jsonschema import Draft202012Validator
    import enrich as enrich_mod

    root = Path(enrich_mod.__file__).resolve().parent
    prompt = load_enrichment_prompt(root / "prompts" / "enrichment_v1.md")
    assert "PLACEHOLDER" not in prompt
    for placeholder in ("{{KNOWN_PERSONS}}", "{{SOURCE_HINTS}}", "{{SCHEMA}}", "{{CONTENT}}"):
        assert placeholder in prompt

    schema = load_enrichment_schema(root / "qa" / "schema.md")
    Draft202012Validator.check_schema(schema)
    assert "meeting_type" in schema["required"]
    assert "voice_corpus_eligible" in schema["required"]


# =============================================================================
# Parsing
# =============================================================================

def test_parse_strips_think_block(config):
    enr = _enricher(config, lambda r: httpx.Response(200, json=_completion("x")))
    raw = "<think>razonando en español...</think>\n```yaml\n" + VALID_YAML + "```"
    data = enr._parse_output(raw)
    assert data["content_type"] == "transcript"
    assert data["confidence"] == 0.91


def test_parse_without_fence(config):
    enr = _enricher(config, lambda r: httpx.Response(200, json=_completion("x")))
    data = enr._parse_output(VALID_YAML)
    assert data["diarization_warning"] is True


def test_parse_non_mapping_raises(config):
    enr = _enricher(config, lambda r: httpx.Response(200, json=_completion("x")))
    with pytest.raises(Exception):
        enr._parse_output("```yaml\n- just\n- a\n- list\n```")


# =============================================================================
# enrich_file: happy path
# =============================================================================

@pytest.mark.asyncio
async def test_enrich_file_writes_enriched(config, tmp_path):
    raw = _raw_file(tmp_path)
    content = "<think>...</think>\n```yaml\n" + VALID_YAML + "```"
    enr = _enricher(config, lambda r: httpx.Response(200, json=_completion(content)))

    result = await enr.enrich_file(raw, "transcript_tactiq")

    assert result.status == EnrichmentStatus.ENRICHED
    assert result.confidence == 0.91
    out = Path(result.output_path)
    assert out.exists() and out.suffix == ".transcript"

    text = out.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front = yaml.safe_load(text.split("\n---", 1)[0].lstrip("-\n"))
    assert front["content_type"] == "transcript"
    assert front["_audit"]["prompt_version"] == "v1.0.0"
    assert front["_audit"]["source_sha256"] == result.source_sha256
    assert "Ginés" in text  # original body preserved


@pytest.mark.asyncio
async def test_enrich_file_records_manifest(config, tmp_path):
    raw = _raw_file(tmp_path)
    content = "```yaml\n" + VALID_YAML + "```"
    enr = _enricher(config, lambda r: httpx.Response(200, json=_completion(content)))
    await enr.enrich_file(raw, "transcript_tactiq")

    mpath = config.paths.enriched_dir / "ENRICHMENT_MANIFEST.jsonl"
    assert mpath.exists()
    rec = json.loads(mpath.read_text().splitlines()[0])
    assert rec["status"] == "enriched"
    assert rec["prompt_version"] == "v1.0.0"


# =============================================================================
# enrich_file: resumability
# =============================================================================

@pytest.mark.asyncio
async def test_enrich_file_skips_already_enriched(config, tmp_path):
    raw = _raw_file(tmp_path)
    content = "```yaml\n" + VALID_YAML + "```"
    enr = _enricher(config, lambda r: httpx.Response(200, json=_completion(content)))
    manifest = EnrichmentManifest(
        config.paths.enriched_dir / "ENRICHMENT_MANIFEST.jsonl"
    )

    first = await enr.enrich_file(raw, "transcript_tactiq", manifest=manifest)
    second = await enr.enrich_file(raw, "transcript_tactiq", manifest=manifest)
    assert first.status == EnrichmentStatus.ENRICHED
    assert second.status == EnrichmentStatus.SKIPPED


@pytest.mark.asyncio
async def test_enrich_file_force_reenriches(config, tmp_path):
    raw = _raw_file(tmp_path)
    content = "```yaml\n" + VALID_YAML + "```"
    enr = _enricher(config, lambda r: httpx.Response(200, json=_completion(content)))
    manifest = EnrichmentManifest(
        config.paths.enriched_dir / "ENRICHMENT_MANIFEST.jsonl"
    )
    await enr.enrich_file(raw, "transcript_tactiq", manifest=manifest)
    forced = await enr.enrich_file(
        raw, "transcript_tactiq", manifest=manifest, force=True
    )
    assert forced.status == EnrichmentStatus.ENRICHED


# =============================================================================
# enrich_file: quarantine
# =============================================================================

@pytest.mark.asyncio
async def test_invalid_enum_quarantined(config, tmp_path):
    raw = _raw_file(tmp_path)
    bad = VALID_YAML.replace("content_type: transcript", "content_type: bogus")
    content = "```yaml\n" + bad + "```"
    enr = _enricher(config, lambda r: httpx.Response(200, json=_completion(content)))

    result = await enr.enrich_file(raw, "transcript_tactiq")
    assert result.status == EnrichmentStatus.QUARANTINED
    raw_out = Path(result.output_path)
    assert raw_out.exists()
    assert "bogus" in raw_out.read_text(encoding="utf-8")
    assert raw_out.with_suffix(".txt").name.endswith("raw.txt")
    reason = raw_out.parent / raw_out.name.replace(".raw.txt", ".reason.txt")
    assert reason.exists()


@pytest.mark.asyncio
async def test_missing_required_field_quarantined(config, tmp_path):
    raw = _raw_file(tmp_path)
    bad = "content_type: transcript\nconfidence: 0.5\n"  # missing fields
    content = "```yaml\n" + bad + "```"
    enr = _enricher(config, lambda r: httpx.Response(200, json=_completion(content)))
    result = await enr.enrich_file(raw, "transcript_tactiq")
    assert result.status == EnrichmentStatus.QUARANTINED


@pytest.mark.asyncio
async def test_unparseable_output_quarantined(config, tmp_path):
    raw = _raw_file(tmp_path)
    enr = _enricher(
        config, lambda r: httpx.Response(200, json=_completion("not yaml: : :["))
    )
    result = await enr.enrich_file(raw, "transcript_tactiq")
    assert result.status == EnrichmentStatus.QUARANTINED


# =============================================================================
# Retry + failure
# =============================================================================

@pytest.mark.asyncio
async def test_retry_then_success(config, tmp_path):
    raw = _raw_file(tmp_path)
    content = "```yaml\n" + VALID_YAML + "```"
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=_completion(content))

    enr = _enricher(config, handler)
    result = await enr.enrich_file(raw, "transcript_tactiq")
    assert result.status == EnrichmentStatus.ENRICHED
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_all_attempts_fail_returns_error(config, tmp_path):
    raw = _raw_file(tmp_path)
    enr = _enricher(config, lambda r: httpx.Response(503, text="down"))
    result = await enr.enrich_file(raw, "transcript_tactiq")
    assert result.status == EnrichmentStatus.ERROR
    assert result.error


@pytest.mark.asyncio
async def test_routes_to_per_tier_endpoint(config, tmp_path):
    raw = _raw_file(tmp_path)
    content = "```yaml\n" + VALID_YAML + "```"
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json=_completion(content))

    enr = _enricher(config, handler)
    await enr.enrich_file(raw, "transcript_tactiq", privacy_tier="private")
    assert seen["url"] == "http://test-somatic:8002/v1/chat/completions"


# =============================================================================
# Hard rules: voice-corpus / diarization override (never trust the model)
# =============================================================================

def test_hard_rules_mono_forces_ineligible_over_llm_true(config):
    """A mono transcript is forced ineligible even if the LLM claimed True."""
    enr = _enricher(config, lambda r: httpx.Response(200, json=_completion("x")))
    spec = enr.content_types.get("transcript_tactiq")
    data = {
        "meeting_type": "family_conversation",
        "voice_corpus_eligible": True,   # model lied / guessed
        "diarization_warning": False,
    }
    enr._apply_hard_rules(data, spec, "mono")
    assert data["diarization_status"] == "mono"
    assert data["diarization_warning"] is True
    assert data["voice_corpus_eligible"] is False


def test_hard_rules_therapy_forces_ineligible(config):
    """meeting_type therapy forces ineligible even for a voice-eligible type."""
    enr = _enricher(config, lambda r: httpx.Response(200, json=_completion("x")))
    spec = enr.content_types.get("letter_maestro")  # voice_corpus_eligible True
    data = {"meeting_type": "therapy", "voice_corpus_eligible": True}
    enr._apply_hard_rules(data, spec, None)
    assert data["voice_corpus_eligible"] is False


def test_hard_rules_eligible_content_stays_eligible(config):
    """Voice-eligible content, non-therapy, no mono block -> stays eligible."""
    enr = _enricher(config, lambda r: httpx.Response(200, json=_completion("x")))
    spec = enr.content_types.get("letter_maestro")
    data = {"meeting_type": "personal_reflection", "voice_corpus_eligible": True}
    enr._apply_hard_rules(data, spec, None)
    assert data["voice_corpus_eligible"] is True


def test_hard_rules_multi_transcript_still_spec_gated(config):
    """A multi transcript records multi, but the spec ceiling keeps it ineligible."""
    enr = _enricher(config, lambda r: httpx.Response(200, json=_completion("x")))
    spec = enr.content_types.get("transcript_tactiq")  # voice_corpus_eligible False
    data = {"meeting_type": "family_conversation", "voice_corpus_eligible": True}
    enr._apply_hard_rules(data, spec, "multi")
    assert data["diarization_status"] == "multi"
    assert data["diarization_warning"] is False
    assert data["voice_corpus_eligible"] is False
