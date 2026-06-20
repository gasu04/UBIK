# Quality Review Rubric (Gate 1)

Defines how a human adjudicates each enriched record at Gate 1 and what the
confidence bands mean. Gate 1 is the human-in-the-loop checkpoint between
enrichment and the ChromaDB write.

## Confidence bands (from `config/ingestion_config.py:GateThresholds`)

| Band                                   | Default       | Meaning                                          |
|----------------------------------------|---------------|--------------------------------------------------|
| `enrichment_confidence >= auto_approve`| ≥ 0.85        | High confidence — review is a quick sanity check |
| `quarantine_below <= conf < auto`      | 0.60 – 0.85   | Pending review — read carefully before approving |
| `enrichment_confidence < quarantine`   | < 0.60        | Low confidence — default to quarantine/flag      |

These bands *order the queue and set the default action*; the human decision
is always authoritative.

## Decisions

| Key | Action | Effect |
|-----|--------|--------|
| `A` | Approve     | Record moves to `approved/`, eligible for the ChromaDB write. |
| `F` | Flag        | Moves to `quarantine/enrichment/` with `quarantine_reason: human_flagged` for rework. |
| `Q` | Quarantine  | Moves to `quarantine/enrichment/` with `quarantine_reason: human_quarantined`. |
| `S` | Skip        | Left in place; decide later. No write. |
| `X` | Abort       | Stop the batch cleanly; no partial writes. |

## Hard rules (non-negotiable, enforced regardless of the YAML field values)

1. **Therapy is never voice-corpus eligible and is never written to a
   collection.** A record with `meeting_type == "therapy"` displays a
   prominent warning and cannot be approved into voice-eligible status. The
   writer excludes therapy from ChromaDB entirely (it is quarantined instead).
2. **Mono diarization is never voice-corpus eligible.** `diarization_status
   == "mono"` forces `voice_corpus_eligible = false` (enforced in `enrich.py`,
   re-checked at the writer). Mono records may still be approved as ordinary
   episodic memories — they just never feed voice training.
3. **Unresolved participants do not block approval.** They are surfaced for
   the reviewer; the reviewer decides whether the gaps matter for this record.

## What to check per content type

- **transcript_tactiq / gemini / fireflies:** confirm `meeting_type`,
  scan `participants_detected` for mis-resolved or `ambiguous` names, and
  honor the diarization warning. Tactiq is mono by source and structurally
  voice-corpus-blocked.
- **letter_maestro:** text is already pure protagonist voice; the review is
  about metadata accuracy, not voice eligibility.
- **ethical_constitution:** hand-curated, ingested verbatim — Gate 1 only
  confirms it routed to `ubik_semantic` with the forced `stability: core`.
