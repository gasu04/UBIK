# Enrichment Prompts

This directory holds the **versioned enrichment prompt templates** consumed by
`ingestion/enrich.py`. The active template is selected by
`UBIK_ENRICHMENT_PROMPT` (default: `prompts/enrichment_v1.md`).

The version tag stamped on every enrichment output comes from
`UBIK_PROMPT_VERSION` (default `v1.0.0`), **not** from the filename — keep the
two in sync deliberately (filename `enrichment_v1.md` ↔ version `v1.0.0`).

## Template contract

`enrich.py` renders the template by substituting these placeholders. All are
optional; substitute only the ones you use.

| Placeholder        | Replaced with                                                        |
|--------------------|----------------------------------------------------------------------|
| `{{KNOWN_PERSONS}}`| The Known_Persons registry block (canonical names, aliases, role, tier) |
| `{{SOURCE_HINTS}}` | Source content-type hints (parser, diarization_trust, default_sensitive, voice eligibility) |
| `{{SCHEMA}}`       | The exact output schema from `qa/schema.md` (so the model sees the contract) |
| `{{CONTENT}}`      | The transcript body to enrich                                         |

### Message construction

- If the template contains `{{CONTENT}}`, the fully-rendered template is sent
  as a **single user message** (you control exactly where the body goes).
- If it does **not** contain `{{CONTENT}}`, the rendered template becomes the
  **system** message and the transcript body is sent as a separate **user**
  message (keeps the large body out of any cached system prompt).

## Output contract (what the model must return)

- Emit the enrichment as a **single fenced YAML block** (```` ```yaml ````).
  Any `<think>…</think>` / `<reasoning>…</reasoning>` blocks a reasoning model
  produces are stripped before parsing.
- The YAML must validate against the JSON Schema in `qa/schema.md`
  (every required field present, every enum value legal). Output that fails
  validation is written to `quarantine/enrichment/` with the raw model text
  saved alongside — it is never silently dropped.

## Spanish-first

Per `qa/learned_rules.md`: the corpus is Spanish-first. Instruct the model to
work in/about Spanish, accept accented names (Ginés, Adrián, Sofía), and treat
bare `Gines`/`Ginés`/`GASU` as the protagonist (the father) — the son is always
`Gines Alberto` / `Gines Hijo`.

> **Status:** `enrichment_v1.md` is a PLACEHOLDER. `enrich.py` refuses to run
> until it is replaced with the real prompt — paste it from your Claude.ai
> session docs.
