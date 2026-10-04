# CP2 Status — Phase 3 Enrichment Quality Gate

> **tl;dr** — CP2 is the checkpoint that decides whether the LLM enrichment pipeline is good enough to proceed to human Gate 1 and ChromaDB writes. It was defined as "confidence ≥ 0.7 on ≥ 6 of 8 files." We have only **3 source files** staged, and the last run scored **3/3 ≥ 0.7**, but the **denominator is unmet**. The immediate blocker is a decision, not code.

---

## 1. Background: what is CP2?

**Phase 3** of the ingestion pipeline takes raw source documents, runs them through an LLM (DeepSeek-R1-Distill-Qwen-14B-AWQ on the Somatic node), and produces structured `.transcript` files with metadata (meeting type, participants, diarization, voice eligibility, confidence, etc.).

**Checkpoint 2 (CP2)** is the *quality gate* for that enrichment step. It is documented in the original Phase 3 brief and in `ubik_sessions.md` entries from **2026-06-20** and **2026-07-19/20**.

### The gate as originally written

> Run `run_phase3.py --stage 2 --dry-run --limit 8`, then hand-score the outputs against `qa/rubric.md`. **Pass if enrichment_confidence ≥ 0.7 on at least 6 of the 8 files.**

The `--limit 8` was meant to cap the batch at 8 files. Because `enrich_directory` applies the limit **per source subdirectory**, and only `sources/tactiq/` currently has files, the effective cap is the number of files in that one directory.

### Why CP2 matters

- It is **the last automated gate before human review (Gate 1)** and before any data is written to ChromaDB/Neo4j.
- If the model's confidence or metadata quality is poor, approving it now would pollute the memory corpus with mis-tagged records.
- The rule "don't loosen a gate to manufacture a pass" was explicitly invoked in the 2026-07-19 session: we are not allowed to redefine the gate just because we don't have 8 files.

---

## 2. Current state of the pipeline

### 2.1 Key files and their roles

| File / Directory | Purpose |
|---|---|
| `ingestion/run_phase3.py` | Orchestrator for stages 1–5 (parse → enrich → resolve → Gate 1 → write). |
| `ingestion/enrich.py` | LLM client: renders prompt, strips reasoning tags, parses YAML, validates schema, applies hard rules. |
| `ingestion/ingest/diarization.py` | Per-file mono/multi detection (Unicode-safe). |
| `ingestion/ingest/person_resolver.py` | Resolves participant names against `registry/known_persons.yaml`. |
| `ingestion/ingest/mcp_writer.py` | Writes approved records to ChromaDB + Neo4j; includes SHA-256 source-document dedup. |
| `ingestion/prompts/enrichment_v1.md` | The prompt template sent to the LLM. |
| `ingestion/qa/schema.md` | JSON Schema the model output is validated against. |
| `ingestion/qa/rubric.md` | Human Gate 1 scoring rules + confidence bands. |
| `ingestion/registry/known_persons.yaml` | Canonical person registry (12 persons, Spanish-first). |
| `ingestion/registry/content_types.yaml` | Content-type definitions and diarization-trust rules. |
| `ingestion/config/ingestion_config.py` | Typed config loader (endpoints, paths, Gate thresholds). |
| `ingestion/.env` (gitignored) | Live values; currently has the **old Somatic Tailscale IP**. |
| `ingestion/.env.example` | Template; also has the old Somatic IP (needs update post-migration). |

### 2.2 Directory layout on disk

```text
ingestion/
├── sources/
│   ├── tactiq/               ← 3 .docx files (the only non-empty bucket)
│   ├── gemini/               ← empty
│   ├── fireflies/            ← empty
│   ├── letters/              ← empty
│   ├── memory_notes/         ← empty
│   └── constitution/         ← empty
├── enriched/                 ← 3 .transcript outputs + ENRICHMENT_MANIFEST.jsonl
├── pending_review/           ← empty (Gate 1 queue)
├── approved/                 ← empty
└── quarantine/enrichment/    ← 1 degenerate file quarantined from an earlier run
```

### 2.3 The 3 enriched files

| # | Source | Enriched output | `meeting_type` | `diarization_status` | `voice_corpus_eligible` | `enrichment_confidence` | Notes |
|---|--------|-----------------|----------------|----------------------|-------------------------|------------------------:|-------|
| 1 | `12 23-12-2025 Khaled Nasr... actinium 225 RFCA.docx` | `.transcript` | `business` | `mono` | `false` | **0.95** | Real multi-party clinical-isotopes dialogue; mono because Tactiq has generic "Speaker:" labels. |
| 2 | `3 Protocolo de Entregas de Equipos 03-12-2025...docx` | `.transcript` | `business` | `multi` | `false` | **0.90** ⚠️ | **Degenerate input**: Tactiq metadata stub + one AI summary paragraph, no real dialogue. `diarization: multi` is misleading. |
| 3 | `Meeting Transcription 16-12-2025.docx` | `.transcript` | `business` | `multi` | `false` | **0.95** | Real long dialogue, named participant (Fabio Robledo). |

> **Manifest:** `ingestion/enriched/ENRICHMENT_MANIFEST.jsonl` records all three with `prompt_version: v1.0.0`.

---

## 3. The CP2 decision problem

### 3.1 We have 3 quality passes, not 6

The gate requires **≥ 6 of 8 files** with confidence ≥ 0.7. We have **3 files total**, all ≥ 0.7.

So the literal gate is **not satisfied**, even though the 3 existing outputs look good.

### 3.2 Two ways to resolve the denominator

| Option | Action | Implication |
|---|---|---|
| **A. Run a proper 8-file test** | Stage 8 representative/adversarial source files into `sources/tactiq/` (or across multiple source buckets), re-run CP2, hand-score. | Gold-standard. Requires selecting or finding 8 files and takes real LLM time. |
| **B. Revisit the gate itself** | Decide that 3/3 clean enrichments, with all hard rules holding, is sufficient to proceed; lower the gate threshold or accept a smaller denominator. | Faster, but violates the original "don't loosen a gate to pass" rule unless you explicitly endorse the change. |

### 3.3 The File 2 concern

File 2 (`3 Protocolo de Entregas...`) is a **degenerate input**: it contains no verbatim transcript, only a Tactiq metadata stub and an AI summary paragraph. Yet the model returned:

- `diarization_status: multi`
- `enrichment_confidence: 0.90`

The confidence is arguably overstated for near-empty content. If CP2 is run again, this file should either:

1. Be excluded as not representative,
2. Trigger a new content-type / confidence-floor rule in `qa/rubric.md`, or
3. Be counted as a known edge case, not a clean pass.

This is a **diagnosis-only** finding; no rule was silently tuned in prior sessions.

---

## 4. Infrastructure blockers already cleared

The reasons CP2 kept getting deferred between June and September are now resolved:

| Old blocker | Status |
|---|---|
| Somatic vLLM unreachable (networking/IP confusion) | **Resolved** — Somatic migrated to native Ubuntu; Tailscale IP is now `100.92.12.89`. |
| `ingestion/.env` missing `UBIK_ENRICHMENT_MODEL` / endpoint | **Resolved** — model path and endpoint are configured. |
| WSL2 NAT/mirrored networking instability | **Resolved** — no more WSL2 in the stack. |
| vLLM 0.13.0 CVEs / fragility | **Resolved** — running vLLM 0.24.0+cu129, maestro-managed, persistent systemd unit. |
| MCP server import corruption | **Resolved** — fastmcp/fastmcp-slim reinstalled. |

> **Resolved 2026-10-03:** `ingestion/.env` and `ingestion/.env.example` both now point at `100.92.12.89` (`SOMATIC_HOST` + `SOMATIC_TAILSCALE_IP`); verified through `load_config()` — all endpoints resolve to `http://100.92.12.89:8002/v1`. The repo-wide stale-IP cleanup (maestro, hippocampal, config, docs, test fixtures) was done in the same pass.

---

## 5. Proposed next steps

### Step 0 — fix the stale Somatic IP (5 minutes)

**DONE 2026-10-03.** Applied to `ingestion/.env`:

```diff
-SOMATIC_HOST=100.92.95.39
+SOMATIC_HOST=100.92.12.89
-SOMATIC_TAILSCALE_IP=100.92.95.39
+SOMATIC_TAILSCALE_IP=100.92.12.89
```

`ingestion/.env.example` was updated in the same pass (template no longer drifts).

### Step 1 — decide which CP2 path to take

- [ ] **Path A**: run the proper 8-file test.
  - Need: select or locate 8 representative/adversarial source files.
  - Candidate sources:
    - The 7 Tactiq zip archives on `/Volumes/Seagate2T/` (documented 2026-07-28).
    - Existing `.docx` files or other raw content in the project.
  - Action: stage 8 files into `sources/tactiq/`, run `run_phase3.py --stage 2 --dry-run --limit 8`, hand-score per `qa/rubric.md`.
- [ ] **Path B**: accept 3/3 as sufficient.
  - Need: explicitly lower/revise the gate in writing (e.g., in this doc or in `qa/rubric.md`).
  - Risk: lower evidence base for trusting the pipeline.

### Step 2 — address File 2 before declaring pass

- [ ] Decide whether to exclude degenerate/no-dialogue stubs from CP2 scoring.
- [ ] Optionally add a confidence-floor rule in `qa/rubric.md` for near-empty transcripts.

### Step 3 — continue the pipeline after CP2

Once CP2 is decided:

1. Run **CP3** (`run_phase3.py --stage 3`): participant resolution against `known_persons.yaml`.
2. Run **CP4 / Gate 1** (`interactive_ingest.py`): human approve/flag/quarantine.
3. Run **CP5** (`run_phase3.py --stage 5`): write approved records to ChromaDB/Neo4j.

---

## 6. Quick reference commands

```bash
# Run just the enrichment stage, dry-run, capped at 8 files per bucket
python run_phase3.py --stage 2 --dry-run --limit 8

# Run the full offline pipeline through resolution (stages 1-3)
python run_phase3.py --dry-run --limit 8

# Check config resolves to the right endpoint
python -c "from config.ingestion_config import load_config; c=load_config(); print(c.endpoints.sensitive_endpoint, c.endpoints.model)"
```

---

## 7. Open questions for Gines

1. Do you want **Path A** (proper 8-file test) or **Path B** (accept 3/3)?
2. If Path A, which 8 files should be used? Should we extract more from the Seagate2T Tactiq archives, or use other sources?
3. How should **File 2** (the no-dialogue stub) be treated — exclude it, add a rule, or count it?
4. Should we update the confidence bands in `qa/rubric.md` or `GateThresholds` before the next run?

---

*Document generated from `ubik_sessions.md` entries 2026-06-20, 2026-07-19/20, 2026-07-28, 2026-09-20 and live filesystem inspection.*
