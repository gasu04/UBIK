# Ingestion Learned Rules

> **PLACEHOLDER** — paste the learned-rules content from the Claude.ai
> session docs here. This file is read by the enrichment prompt builder;
> until populated, enrichment runs with prompt defaults only.

<!-- rules accumulate below, one dated entry per lesson -->

## 2026-06-12 — Spanish is the primary language of the corpus

Per Gines: most transcripts and source material are in Spanish. All
parsing, name handling, enrichment prompts, and classification must
treat Spanish as the first choice, with English as secondary:

- Speaker/name regexes must accept accented characters (Ginés, Adrián,
  Sofía…). KNOWN ISSUE: `SpeakerTurnParser` patterns in
  `ingest/transcript_processor.py` use `[A-Za-z]` classes that break on
  accents — fix when that file is touched.
- Name disambiguation: bare "Gines"/"Ginés"/"GASU" = the protagonist
  (father). The son is always "Gines Alberto" or "Gines Hijo" — never
  bare "Gines".
- Enrichment prompts should instruct the model in/about Spanish content
  explicitly; default `language` is "es" unless detected otherwise.
- Insight/tone marker lists must lead with Spanish forms (the existing
  `_is_insight_chunk` / `_map_tone_to_valence` already do this — keep it).
