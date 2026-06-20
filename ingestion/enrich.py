#!/usr/bin/env python3
"""
UBIK Ingestion System - Enrichment Client (Phase 3)

Enriches raw source files with structured metadata by calling the
self-hosted, OpenAI-compatible LLM endpoint chosen by privacy tier, then
validating the model's YAML output against the enrichment schema.

Data flow:
    sources/<type>/<file>            (raw, immutable original)
    -> ProcessorRegistry             (extract body text: .docx/.pdf/.txt/...)
    -> rendered enrichment prompt    (template + known_persons + source hints)
    -> LLM /chat/completions         (per-tier endpoint, retries + LAN fallback)
    -> strip <think>/<reasoning>, parse YAML fence
    -> validate against qa/schema.md (jsonschema; every field + enums)
       valid   -> enriched/<name>.transcript   (front matter + audit + body)
       invalid -> quarantine/enrichment/        (raw model output preserved)

Resumable: every successful enrichment is recorded in
``enriched/ENRICHMENT_MANIFEST.jsonl`` keyed by (source SHA-256,
prompt_version). Re-running skips files already enriched at the current
prompt version unless ``force=True``.

Privacy: routing is fail-safe — every tier goes to the sensitive
(self-hosted Somatic) endpoint unless the tier is explicitly relaxed via
``UBIK_STANDARD_PRIVACY_TIERS`` (see config). Logs carry file IDs,
timing, and confidence scores ONLY — never source or model content.

Usage:
    from enrich import Enricher

    enricher = Enricher.from_config()
    result = await enricher.enrich_file(
        Path("sources/tactiq/meeting.docx"),
        content_type="transcript_tactiq",
    )

Dependencies: httpx, PyYAML, jsonschema (+ python-docx/pdfplumber via
ProcessorRegistry for document bodies).

Tier classification: Tier 1 (critical — writes the enriched corpus).
Failures are loud; invalid output is quarantined, never dropped.

Version: 0.1.0
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.ingestion_config import IngestionConfig, load_config
from ingest.diarization import detect_diarization
from ingest.models import IngestItem
from ingest.processors import ProcessorRegistry
from ingest.registry import (
    ContentTypeRegistry,
    ContentTypeSpec,
    KnownPersonsRegistry,
    UbikIngestionError,
    load_content_types,
    load_known_persons,
)

__all__ = [
    "EnrichmentError",
    "PromptNotReadyError",
    "SchemaNotReadyError",
    "EnrichmentStatus",
    "EnrichmentResult",
    "EnrichmentManifest",
    "Enricher",
    "load_enrichment_prompt",
    "load_enrichment_schema",
]

logger = logging.getLogger("ubik.enrich")

# Reasoning models (DeepSeek-R1) wrap their scratchpad in these tags.
_REASONING_RE = re.compile(r"<\s*(think|reasoning)\s*>.*?<\s*/\s*\1\s*>",
                           re.DOTALL | re.IGNORECASE)
# A leading, never-closed reasoning block (truncated output).
_OPEN_REASONING_RE = re.compile(r"^\s*<\s*(?:think|reasoning)\s*>.*?(?=```)",
                                re.DOTALL | re.IGNORECASE)
# First fenced YAML block (``` or ```yaml / ```yml).
_YAML_FENCE_RE = re.compile(r"```(?:ya?ml)?\s*\n(.*?)```", re.DOTALL)
# A schema fence inside qa/schema.md.
_JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\s*\n(.*?)```", re.DOTALL)

_HASH_CHUNK_BYTES = 1024 * 1024
_PLACEHOLDER_MARKER = "PLACEHOLDER"

# Default content-type -> privacy tier hint. Fail-safe: anything not listed
# (and any tier not explicitly relaxed in config) routes to the sensitive
# endpoint regardless. Used only to pick the tier label for routing.
_DEFAULT_TIER_BY_CONTENT_TYPE = {
    "transcript_tactiq": "private",
    "transcript_gemini": "private",
    "transcript_fireflies": "private",
    "memory_note": "private",
    "letter_maestro": "family",
    "ethical_constitution": "private",
}


class EnrichmentError(UbikIngestionError):
    """An enrichment operation failed."""


class PromptNotReadyError(EnrichmentError):
    """The enrichment prompt template is missing or still a placeholder."""


class SchemaNotReadyError(EnrichmentError):
    """The enrichment schema is missing or still a placeholder."""


# =============================================================================
# Prompt + schema loading
# =============================================================================

def load_enrichment_prompt(path: Path) -> str:
    """
    Load the enrichment prompt template, refusing placeholders.

    Args:
        path: Path to the prompt markdown (e.g. prompts/enrichment_v1.md).

    Returns:
        The raw template text (placeholders unsubstituted).

    Raises:
        PromptNotReadyError: If the file is missing, empty, or still
            contains the PLACEHOLDER marker.
    """
    if not path.exists():
        raise PromptNotReadyError(
            f"Enrichment prompt not found: {path}\n"
            "Create it (see prompts/README.md) or set UBIK_ENRICHMENT_PROMPT."
        )
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise PromptNotReadyError(f"Enrichment prompt is empty: {path}")
    if _PLACEHOLDER_MARKER in text:
        raise PromptNotReadyError(
            f"Enrichment prompt {path} is still a PLACEHOLDER. "
            "Paste the real prompt before enriching."
        )
    return text


def load_enrichment_schema(path: Path) -> Dict[str, Any]:
    """
    Extract the JSON Schema from qa/schema.md, refusing placeholders.

    Looks for the first fenced ```json block, then any fenced block parsed
    as JSON, then as YAML.

    Args:
        path: Path to the schema markdown (e.g. qa/schema.md).

    Returns:
        The parsed JSON Schema as a dict.

    Raises:
        SchemaNotReadyError: If the file is missing, has no parseable schema
            block, or still contains a PLACEHOLDER schema.
    """
    if not path.exists():
        raise SchemaNotReadyError(
            f"Enrichment schema not found: {path}\n"
            "Populate qa/schema.md or set UBIK_ENRICHMENT_SCHEMA."
        )
    text = path.read_text(encoding="utf-8")

    block: Optional[str] = None
    m = _JSON_FENCE_RE.search(text)
    if m:
        block = m.group(1)
    else:
        for m in _ANY_FENCE_RE.finditer(text):
            block = m.group(1)
            break
    if block is None:
        raise SchemaNotReadyError(
            f"No fenced schema block found in {path}. "
            "Paste the enrichment JSON Schema in a ```json block."
        )

    try:
        schema = json.loads(block)
    except json.JSONDecodeError:
        try:
            schema = yaml.safe_load(block)
        except yaml.YAMLError as e:
            raise SchemaNotReadyError(
                f"Schema block in {path} is neither valid JSON nor YAML: {e}"
            ) from e

    if not isinstance(schema, dict) or not schema:
        raise SchemaNotReadyError(f"Schema in {path} must be a non-empty object")
    if _PLACEHOLDER_MARKER in json.dumps(schema):
        raise SchemaNotReadyError(
            f"Schema in {path} is still a PLACEHOLDER. "
            "Paste the real enrichment schema before enriching."
        )
    return schema


# =============================================================================
# Resumability manifest
# =============================================================================

@dataclass(frozen=True)
class _ManifestRecord:
    """One enrichment outcome recorded for resume / audit."""
    source_sha256: str
    prompt_version: str
    source_file: str
    status: str
    output_path: str
    confidence: Optional[float]
    enriched_at: str

    def to_json(self) -> str:
        return json.dumps({
            "source_sha256": self.source_sha256,
            "prompt_version": self.prompt_version,
            "source_file": self.source_file,
            "status": self.status,
            "output_path": self.output_path,
            "confidence": self.confidence,
            "enriched_at": self.enriched_at,
        }, ensure_ascii=False)


class EnrichmentManifest:
    """
    Append-only JSONL record of enrichment outcomes, keyed for resume.

    The resume key is ``(source_sha256, prompt_version)``: a file already
    enriched at the current prompt version is skipped. Quarantined outcomes
    are recorded too (for audit) but do NOT block a re-run.

    Note:
        Malformed lines raise EnrichmentError — a corrupt manifest must be
        repaired, not silently ignored, or resume stops working.
    """

    def __init__(self, path: Path):
        self.path = path
        self._enriched: set = set()  # {(sha, prompt_version)} with status "enriched"
        if path.exists():
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "enriched":
                        self._enriched.add(
                            (rec["source_sha256"], rec["prompt_version"])
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    raise EnrichmentError(
                        f"Corrupt enrichment manifest line {lineno} in {path}: {e}"
                    ) from e

    def is_enriched(self, source_sha256: str, prompt_version: str) -> bool:
        """True if this content was already enriched at this prompt version."""
        return (source_sha256, prompt_version) in self._enriched

    def append(self, record: _ManifestRecord) -> None:
        """Durably record an outcome and update the resume index."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")
        if record.status == "enriched":
            self._enriched.add((record.source_sha256, record.prompt_version))


# =============================================================================
# Result
# =============================================================================

class EnrichmentStatus:
    """Outcome labels for an enrichment attempt (string constants)."""
    ENRICHED = "enriched"
    QUARANTINED = "quarantined"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class EnrichmentResult:
    """
    Outcome of enriching one source file.

    Attributes:
        source_file: Source filename (basename).
        source_sha256: Content hash of the raw source.
        content_type: Content-type key used for routing/hints.
        privacy_tier: Tier label used to select the endpoint.
        status: One of EnrichmentStatus.*.
        confidence: Top-level confidence from the enrichment, if present.
        output_path: Where output landed (enriched/ or quarantine/).
        duration_ms: Wall-clock time for the LLM call + processing.
        error: Reason string for QUARANTINED / ERROR outcomes.
        enrichment: Parsed enrichment dict (in memory only; never logged).
    """
    source_file: str
    source_sha256: str
    content_type: str
    privacy_tier: str
    status: str
    confidence: Optional[float] = None
    output_path: Optional[str] = None
    duration_ms: float = 0.0
    error: Optional[str] = None
    enrichment: Optional[Dict[str, Any]] = None


# =============================================================================
# Enricher
# =============================================================================

@dataclass
class _RetryPolicy:
    """LLM call retry policy."""
    max_retries: int = 3
    base_seconds: float = 1.0
    max_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "_RetryPolicy":
        return cls(
            max_retries=int(os.environ.get("UBIK_LLM_MAX_RETRIES", "3")),
            base_seconds=float(os.environ.get("UBIK_LLM_RETRY_BASE_SECONDS", "1.0")),
            max_seconds=float(os.environ.get("UBIK_LLM_RETRY_MAX_SECONDS", "30.0")),
        )


class Enricher:
    """
    Stateful enrichment client.

    Loads the prompt, schema, and registries once, then enriches files or
    directories. Construct via :meth:`from_config`.

    Attributes:
        config: Resolved IngestionConfig.
        prompt_template: Raw prompt template text.
        schema: Parsed JSON Schema for validation.
        known_persons: Validated Known_Persons registry.
        content_types: Validated content-type registry.
    """

    def __init__(
        self,
        *,
        config: IngestionConfig,
        prompt_template: str,
        schema: Dict[str, Any],
        known_persons: KnownPersonsRegistry,
        content_types: ContentTypeRegistry,
        retry: Optional[_RetryPolicy] = None,
        client: Optional[httpx.AsyncClient] = None,
        processor_registry: Optional[ProcessorRegistry] = None,
    ):
        self.config = config
        self.prompt_template = prompt_template
        self.schema = schema
        self.known_persons = known_persons
        self.content_types = content_types
        self._retry = retry or _RetryPolicy.from_env()
        self._client = client  # injected (tests) or built per call
        self._processors = processor_registry or ProcessorRegistry()
        self._temperature = float(os.environ.get("UBIK_LLM_TEMPERATURE", "0.1"))
        self._max_tokens = int(os.environ.get("UBIK_LLM_MAX_TOKENS", "4096"))

        # jsonschema is imported here so the rest of the module imports
        # without it; build the validator once, fail loud on a bad schema.
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(self.schema)
        self._validator = Draft202012Validator(self.schema)

    @classmethod
    def from_config(
        cls,
        config: Optional[IngestionConfig] = None,
        *,
        client: Optional[httpx.AsyncClient] = None,
        processor_registry: Optional[ProcessorRegistry] = None,
    ) -> "Enricher":
        """
        Build an Enricher from configuration and on-disk prompt/schema.

        Raises:
            PromptNotReadyError / SchemaNotReadyError: If the prompt or
                schema is missing or still a placeholder.
            ValueError: If no enrichment model is configured.
        """
        config = config or load_config()
        if not config.endpoints.model:
            raise ValueError(
                "No enrichment model configured. Set UBIK_ENRICHMENT_MODEL "
                "to the model name served by the vLLM endpoint."
            )
        prompt_path = Path(
            os.environ.get("UBIK_ENRICHMENT_PROMPT")
            or str(config.paths.root / "prompts" / "enrichment_v1.md")
        )
        schema_path = Path(
            os.environ.get("UBIK_ENRICHMENT_SCHEMA")
            or str(config.paths.qa_dir / "schema.md")
        )
        return cls(
            config=config,
            prompt_template=load_enrichment_prompt(prompt_path),
            schema=load_enrichment_schema(schema_path),
            known_persons=load_known_persons(),
            content_types=load_content_types(),
            client=client,
            processor_registry=processor_registry,
        )

    # -- public API ----------------------------------------------------------

    async def enrich_file(
        self,
        path: Path,
        content_type: str,
        *,
        privacy_tier: Optional[str] = None,
        force: bool = False,
        manifest: Optional[EnrichmentManifest] = None,
    ) -> EnrichmentResult:
        """
        Enrich a single source file.

        Args:
            path: Raw source file under sources/.
            content_type: Registered content-type key (drives parser hints
                and the routing tier default).
            privacy_tier: Override the routing tier; defaults to the
                content-type's tier, else fail-safe "private".
            force: Re-enrich even if already enriched at this prompt version.
            manifest: Shared EnrichmentManifest (created if omitted).

        Returns:
            An EnrichmentResult describing the outcome.
        """
        path = Path(path)
        spec = self.content_types.get(content_type)
        tier = (privacy_tier
                or _DEFAULT_TIER_BY_CONTENT_TYPE.get(content_type, "private"))
        manifest = manifest or EnrichmentManifest(
            self.config.paths.enriched_dir / "ENRICHMENT_MANIFEST.jsonl"
        )
        prompt_version = self.config.prompt_version
        sha = _sha256_file(path)

        if not force and manifest.is_enriched(sha, prompt_version):
            logger.info("skip (already enriched @ %s): %s", prompt_version, path.name)
            return EnrichmentResult(
                source_file=path.name, source_sha256=sha,
                content_type=content_type, privacy_tier=tier,
                status=EnrichmentStatus.SKIPPED,
            )

        started = datetime.now(timezone.utc)
        loop_start = asyncio.get_event_loop().time()
        try:
            body = await self._extract_body(path)
            # Per-file diarization is only meaningful for transcript content
            # (letters/constitution have no speaker turns). Computed before the
            # LLM call so the detected status can be surfaced in the prompt.
            detected = detect_diarization(body) if spec.parser == "transcript" else None
            messages = self._build_messages(body, spec, detected)
            raw = await self._call_llm(tier, messages)
        except Exception as e:  # network / processing failure (not validation)
            duration = (asyncio.get_event_loop().time() - loop_start) * 1000
            logger.error("enrichment call failed for %s: %s", path.name, e)
            return EnrichmentResult(
                source_file=path.name, source_sha256=sha,
                content_type=content_type, privacy_tier=tier,
                status=EnrichmentStatus.ERROR, error=str(e),
                duration_ms=duration,
            )
        duration = (asyncio.get_event_loop().time() - loop_start) * 1000

        # Parse + validate.
        parse_error: Optional[str] = None
        data: Optional[Dict[str, Any]] = None
        try:
            data = self._parse_output(raw)
            self._validate(data)
        except EnrichmentError as e:
            parse_error = str(e)

        if parse_error is not None or data is None:
            out = self._quarantine(path, sha, content_type, raw, parse_error or "parse failed")
            manifest.append(_ManifestRecord(
                source_sha256=sha, prompt_version=prompt_version,
                source_file=path.name, status=EnrichmentStatus.QUARANTINED,
                output_path=str(out), confidence=None,
                enriched_at=started.isoformat(),
            ))
            logger.warning("quarantined %s: %s", path.name, parse_error)
            return EnrichmentResult(
                source_file=path.name, source_sha256=sha,
                content_type=content_type, privacy_tier=tier,
                status=EnrichmentStatus.QUARANTINED, error=parse_error,
                output_path=str(out), duration_ms=duration,
            )

        # Hard rules: never trust the model for voice eligibility. Force the
        # safe values for mono diarization and therapy before anything is
        # written or routed. See _apply_hard_rules.
        self._apply_hard_rules(data, spec, detected)

        confidence = _coerce_confidence(
            data.get("enrichment_confidence", data.get("confidence"))
        )
        out = self._write_enriched(
            path, sha, content_type, tier, prompt_version, data, body, started
        )
        manifest.append(_ManifestRecord(
            source_sha256=sha, prompt_version=prompt_version,
            source_file=path.name, status=EnrichmentStatus.ENRICHED,
            output_path=str(out), confidence=confidence,
            enriched_at=started.isoformat(),
        ))
        logger.info(
            "enriched %s -> %s (conf=%s, %.0fms)",
            path.name, out.name, confidence, duration,
        )
        return EnrichmentResult(
            source_file=path.name, source_sha256=sha,
            content_type=content_type, privacy_tier=tier,
            status=EnrichmentStatus.ENRICHED, confidence=confidence,
            output_path=str(out), duration_ms=duration, enrichment=data,
        )

    async def enrich_directory(
        self,
        directory: Path,
        content_type: str,
        *,
        privacy_tier: Optional[str] = None,
        force: bool = False,
        limit: Optional[int] = None,
    ) -> List[EnrichmentResult]:
        """
        Enrich every file in a directory (top level only; not recursive).

        Files are processed in sorted order; a shared manifest gives
        resume across the batch. Returns one result per file attempted.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise EnrichmentError(f"Not a directory: {directory}")
        files = sorted(
            p for p in directory.iterdir()
            if p.is_file() and not p.name.startswith(".")
        )
        if limit is not None:
            files = files[:limit]
        manifest = EnrichmentManifest(
            self.config.paths.enriched_dir / "ENRICHMENT_MANIFEST.jsonl"
        )
        results: List[EnrichmentResult] = []
        for path in files:
            results.append(await self.enrich_file(
                path, content_type, privacy_tier=privacy_tier,
                force=force, manifest=manifest,
            ))
        return results

    # -- body extraction -----------------------------------------------------

    async def _extract_body(self, path: Path) -> str:
        """
        Extract the text body to enrich from a raw source file.

        Documents (.docx/.pdf/...) go through ProcessorRegistry. A
        .transcript file's YAML front matter is stripped (body only).
        Plain text is read with an encoding fallback.
        """
        ext = path.suffix.lower()
        if ext == ".transcript":
            return _strip_front_matter(_read_text_fallback(path))
        if self._processors.can_process(ext):
            item = IngestItem.from_path(path)
            processed = await self._processors.process(item, path)
            return processed.text
        # Unknown extension: best-effort plain text.
        return _strip_front_matter(_read_text_fallback(path))

    # -- prompt rendering ----------------------------------------------------

    def _build_messages(
        self, body: str, spec: ContentTypeSpec,
        detected_diarization: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Render the prompt template into OpenAI chat messages.

        If the template contains {{CONTENT}}, the fully-rendered template
        becomes a single user message. Otherwise the rendered template is
        the system message and the body is a separate user message.

        Args:
            body: Transcript/source body to enrich.
            spec: Content-type spec for routing hints.
            detected_diarization: Per-file "mono"/"multi" detection to surface
                to the model (transcript content only); None to omit.
        """
        persons_block = self._render_known_persons()
        hints_block = self._render_source_hints(spec, detected_diarization)
        schema_block = json.dumps(self.schema, ensure_ascii=False, indent=2)

        rendered = (
            self.prompt_template
            .replace("{{KNOWN_PERSONS}}", persons_block)
            .replace("{{SOURCE_HINTS}}", hints_block)
            .replace("{{SCHEMA}}", schema_block)
        )
        if "{{CONTENT}}" in rendered:
            rendered = rendered.replace("{{CONTENT}}", body)
            return [{"role": "user", "content": rendered}]
        return [
            {"role": "system", "content": rendered},
            {"role": "user", "content": body},
        ]

    def _render_known_persons(self) -> str:
        """Render the Known_Persons registry as a compact YAML block."""
        people = [
            {
                "canonical_name": p.canonical_name,
                "aliases": list(p.aliases),
                "relationship": p.relationship,
                "role": p.role,
                "privacy_tier": p.privacy_tier,
            }
            for p in self.known_persons.persons
        ]
        return yaml.safe_dump(
            {"known_persons": people}, sort_keys=False, allow_unicode=True
        ).strip()

    def _render_source_hints(
        self, spec: ContentTypeSpec, detected_diarization: Optional[str] = None
    ) -> str:
        """Render content-type routing hints as a YAML block."""
        hints: Dict[str, Any] = {
            "content_type": spec.name,
            "parser": spec.parser,
            "enrichment_level": spec.enrichment,
            "diarization_trust": spec.diarization_trust,
            "default_sensitive": spec.default_sensitive,
            "voice_corpus_eligible": spec.voice_corpus_eligible,
        }
        if detected_diarization is not None:
            hints["diarization_status"] = detected_diarization
        return yaml.safe_dump(
            {"source_hints": hints}, sort_keys=False, allow_unicode=True
        ).strip()

    def _apply_hard_rules(
        self, data: Dict[str, Any], spec: ContentTypeSpec,
        detected_diarization: Optional[str],
    ) -> None:
        """
        Force the safe values for diarization and voice eligibility in place.

        The model's answers for ``diarization_status`` / ``diarization_warning``
        / ``voice_corpus_eligible`` are advisory only. This method overwrites
        them with values that cannot be wrong in the unsafe direction:

        - For transcript content, ``diarization_status`` is set from the
          per-file detector (not the model), and ``diarization_warning`` is
          ``True`` iff mono.
        - ``voice_corpus_eligible`` is forced ``False`` whenever the content
          type is not voice-eligible, OR diarization is mono (transcript), OR
          ``meeting_type == "therapy"``. It can only ever be ``True`` when the
          content type permits it and neither block applies.

        Args:
            data: Parsed enrichment dict (mutated in place).
            spec: Content-type spec (its ``voice_corpus_eligible`` is the ceiling).
            detected_diarization: Per-file "mono"/"multi" for transcripts, else None.

        Note:
            This is the Python guarantee behind the schema contract — see
            qa/schema.md "Design notes". Tier 1 behavior: a silent failure here
            would let therapy/mono content into the voice corpus.
        """
        mono_blocks = False
        if spec.parser == "transcript" and detected_diarization is not None:
            data["diarization_status"] = detected_diarization
            data["diarization_warning"] = detected_diarization == "mono"
            mono_blocks = detected_diarization == "mono"

        is_therapy = data.get("meeting_type") == "therapy"
        data["voice_corpus_eligible"] = (
            bool(spec.voice_corpus_eligible) and not mono_blocks and not is_therapy
        )

    # -- LLM call ------------------------------------------------------------

    async def _call_llm(self, tier: str, messages: List[Dict[str, str]]) -> str:
        """
        Call the per-tier endpoint with retries and a LAN fallback.

        Returns the assistant message content (raw, unparsed).

        Raises:
            EnrichmentError: After exhausting retries on all endpoints.
        """
        primary = self.config.endpoints.for_tier(tier)
        endpoints = [primary]
        lan = self.config.endpoints.lan_fallback_endpoint
        if lan and lan != primary:
            endpoints.append(lan)

        payload = {
            "model": self.config.endpoints.model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        timeout = self.config.endpoints.request_timeout_seconds

        last_error: Optional[Exception] = None
        for endpoint in endpoints:
            url = endpoint.rstrip("/") + "/chat/completions"
            for attempt in range(1, self._retry.max_retries + 1):
                try:
                    content = await self._post_once(url, payload, timeout)
                    return content
                except _RetryableError as e:
                    last_error = e
                    if attempt < self._retry.max_retries:
                        delay = self._backoff(attempt)
                        logger.warning(
                            "LLM call attempt %d/%d failed (%s); retrying in %.1fs",
                            attempt, self._retry.max_retries, e, delay,
                        )
                        await asyncio.sleep(delay)
                except EnrichmentError as e:
                    # Non-retryable (e.g. 4xx other than 429): try next endpoint.
                    last_error = e
                    break
            logger.warning("endpoint exhausted: %s", _host_only(url))
        raise EnrichmentError(
            f"All enrichment endpoints failed: {last_error}"
        )

    async def _post_once(
        self, url: str, payload: Dict[str, Any], timeout: float
    ) -> str:
        """One POST to an OpenAI-compatible /chat/completions endpoint."""
        try:
            if self._client is not None:
                resp = await self._client.post(url, json=payload, timeout=timeout)
            else:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, json=payload)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            raise _RetryableError(f"transport error: {e}") from e

        if resp.status_code == 429 or resp.status_code >= 500:
            raise _RetryableError(f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise EnrichmentError(f"HTTP {resp.status_code} from endpoint")

        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            raise EnrichmentError(f"malformed LLM response: {e}") from e

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter, capped."""
        ceiling = min(
            self._retry.max_seconds,
            self._retry.base_seconds * (2 ** (attempt - 1)),
        )
        return random.uniform(0, ceiling)

    # -- parse + validate ----------------------------------------------------

    def _parse_output(self, raw: str) -> Dict[str, Any]:
        """
        Strip reasoning blocks, extract the YAML fence, parse to a dict.

        Raises:
            EnrichmentError: If no YAML is found or it does not parse to a
                mapping.
        """
        cleaned = _REASONING_RE.sub("", raw)
        cleaned = _OPEN_REASONING_RE.sub("", cleaned).strip()

        m = _YAML_FENCE_RE.search(cleaned)
        candidate = m.group(1) if m else cleaned
        try:
            data = yaml.safe_load(candidate)
        except yaml.YAMLError as e:
            raise EnrichmentError(f"YAML parse error: {e}") from e
        if not isinstance(data, dict):
            raise EnrichmentError(
                f"enrichment must be a YAML mapping, got {type(data).__name__}"
            )
        return data

    def _validate(self, data: Dict[str, Any]) -> None:
        """
        Validate parsed enrichment against the schema.

        Raises:
            EnrichmentError: Aggregating every schema violation found.
        """
        errors = sorted(self._validator.iter_errors(data), key=lambda e: e.path)
        if errors:
            joined = "; ".join(
                f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
                for e in errors[:10]
            )
            raise EnrichmentError(f"schema validation failed: {joined}")

    # -- output --------------------------------------------------------------

    def _write_enriched(
        self,
        path: Path,
        sha: str,
        content_type: str,
        tier: str,
        prompt_version: str,
        data: Dict[str, Any],
        body: str,
        started: datetime,
    ) -> Path:
        """Write enriched/<name>.transcript: front matter + audit + body."""
        out_dir = self.config.paths.enriched_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}.transcript"

        front = dict(data)
        front["_audit"] = {
            "enriched_by": self.config.endpoints.model,
            "prompt_version": prompt_version,
            "enriched_at": started.isoformat(),
            "source_file": path.name,
            "source_sha256": sha,
            "content_type": content_type,
            "privacy_tier": tier,
        }
        front_matter = yaml.safe_dump(
            front, sort_keys=False, allow_unicode=True
        )
        out_path.write_text(
            f"---\n{front_matter}---\n\n{body}\n", encoding="utf-8"
        )
        return out_path

    def _quarantine(
        self,
        path: Path,
        sha: str,
        content_type: str,
        raw: str,
        reason: str,
    ) -> Path:
        """
        Save invalid model output to quarantine/enrichment/ for review.

        Writes two files: the raw model output and a .reason.txt sidecar.
        """
        qdir = self.config.paths.quarantine_enrichment_dir
        qdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = qdir / f"{path.stem}__{stamp}"
        base.with_suffix(".raw.txt").write_text(raw, encoding="utf-8")
        base.with_suffix(".reason.txt").write_text(
            f"source_file: {path.name}\n"
            f"source_sha256: {sha}\n"
            f"content_type: {content_type}\n"
            f"reason: {reason}\n",
            encoding="utf-8",
        )
        return base.with_suffix(".raw.txt")


# =============================================================================
# Internal helpers
# =============================================================================

class _RetryableError(Exception):
    """A transient failure worth retrying (transport, 429, 5xx)."""


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text_fallback(path: Path) -> str:
    """Read text trying common encodings before giving up."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise EnrichmentError(f"could not decode {path} with any known encoding")


def _strip_front_matter(text: str) -> str:
    """Return the body after a leading YAML front-matter block, if present."""
    if text.startswith("---"):
        parts = text.split("\n---", 1)
        if len(parts) == 2:
            # Drop the closing fence line remainder up to the first newline.
            rest = parts[1]
            return rest.split("\n", 1)[1] if "\n" in rest else ""
    return text


def _coerce_confidence(value: Any) -> Optional[float]:
    """Best-effort float for a top-level confidence field (else None)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _host_only(url: str) -> str:
    """Strip path/query from a URL for logging (no content, just host)."""
    m = re.match(r"^(https?://[^/]+)", url)
    return m.group(1) if m else url
