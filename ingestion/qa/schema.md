# Enrichment Output Schema

The authoritative JSON Schema (Draft 2020-12) for enrichment output. The
model returns one fenced ```yaml block; `enrich.py` parses it and validates
every field against the schema below. Output that fails validation is written
to `quarantine/enrichment/` with the raw model text — never silently dropped.

## How `enrich.py` reads this file

1. It extracts the **first fenced ```json block** (falls back to a bare ```
   block parsed as JSON, then YAML).
2. That object must be a valid JSON Schema. `required` drives the
   "every field present" check; `enum` drives the "enums legal" check.
3. The same schema text is also injected into the prompt via `{{SCHEMA}}` so
   the model sees the exact contract it must satisfy.

## Design notes

- `additionalProperties` is **true** by intent: reasoning models routinely
  emit a stray extra key, and quarantining otherwise-valid enrichment over a
  harmless extra field would be a poor trade. The contract is enforced by
  `required` (presence) + `enum`/bounds (legality), matching how `enrich.py`
  is documented to read this file.
- `voice_corpus_eligible` and `diarization_warning` are validated as booleans
  here, but their **safe values are also forced in Python** after parsing
  (`enrich.py` overrides `voice_corpus_eligible=false` whenever
  `diarization_status == "mono"` or `meeting_type == "therapy"`). The schema
  is the contract with the model; the Python override is the guarantee.
- `main_topics` is bounded 3–7 per the Phase 3 spec. If live runs show this
  quarantines too many short transcripts, relax `minItems` here (one-line
  change) — it is deliberately the only "shape" constraint strict enough to
  reject thin output.

## Fields the spike scoring sheet reads

`spike_enrichment.py` reads these from each enrichment (first present name
wins). Aligned with the field names below so the spike sheet populates
without code changes.

| Scoring column        | Field names tried (in order)                              |
|-----------------------|-----------------------------------------------------------|
| type                  | `meeting_type`, `content_type`, `type`                    |
| confidence            | `enrichment_confidence`, `confidence`                     |
| diarization_warning   | `diarization_warning`, `diarization_status`               |
| decisions count       | `key_decisions`, `decisions`, `decisions_made`            |
| participants          | `participants_detected`, `participants`, `speakers`       |

## Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "UBIK Enrichment Output",
  "type": "object",
  "additionalProperties": true,
  "required": [
    "meeting_type",
    "type_inferred_from",
    "participants_detected",
    "language",
    "main_topics",
    "diarization_status",
    "diarization_warning",
    "voice_corpus_eligible",
    "enrichment_confidence"
  ],
  "properties": {
    "meeting_type": {
      "type": "string",
      "enum": ["therapy", "business", "family_conversation", "personal_reflection", "interview", "unknown"]
    },
    "type_inferred_from": {
      "type": "string",
      "minLength": 1
    },
    "meeting_date": {
      "anyOf": [
        { "type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$" },
        { "type": "null" }
      ]
    },
    "participants_detected": {
      "type": "array",
      "items": { "type": "string" }
    },
    "language": {
      "type": "string",
      "minLength": 2
    },
    "main_topics": {
      "type": "array",
      "items": { "type": "string", "minLength": 1 },
      "minItems": 3,
      "maxItems": 7
    },
    "key_decisions": {
      "type": "array",
      "items": { "type": "string" }
    },
    "diarization_status": {
      "type": "string",
      "enum": ["mono", "multi"]
    },
    "diarization_warning": {
      "type": "boolean"
    },
    "voice_corpus_eligible": {
      "type": "boolean"
    },
    "enrichment_confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    }
  }
}
```
