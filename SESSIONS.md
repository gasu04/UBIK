# SESSIONS.md — DeepSeek RAG Project Session Log

---

## Session: 2026-06-06 00:00 — [Node: Hippocampal]
**Goal:** Establish session journaling and project conventions
**Completed:**
- Confirmed Zellij smart auto-start guard in ~/.zshrc was already correctly configured — left as-is
- Added Session Journaling (MANDATORY) section to CLAUDE.md
- Created SESSIONS.md (this file)
**State left in:**
- Project is stable; no code changes made this session
**Files changed:**
- `CLAUDE.md`: Added mandatory session journaling rules and format
- `SESSIONS.md`: Created session log (this file)
**Next session should:**
- Check git status of remaining untracked files (config/, scripts/, tests/, etc.) and decide what to commit
---

## Session: 2026-06-06 18:00 — [Node: Hippocampal]
**Goal:** Establish project hygiene, session journaling, and GDrive sync across DeepSeek and UBIK
**Completed:**
- Confirmed Zellij smart auto-start guard in ~/.zshrc was already correct — left as-is
- Added Session Journaling (MANDATORY) to DeepSeek CLAUDE.md
- Created SESSIONS.md (initial entry)
- Committed and pushed SESSIONS.md and CLAUDE.md to DeepSeek repo
- Audited all untracked/modified files in DeepSeek; moved misplaced files to correct dirs (free_memory.sh → scripts/, test_google_docs_detection.py → tests/integration/)
- Added *.gguf to .gitignore to exclude large model weights
- Committed and pushed all remaining DeepSeek project files (17 files)
- Built SESSIONS.md GDrive sync: Drive API approach abandoned in favor of simple cp to GdriveMirror/Ubik (Google Drive Desktop handles upload — no OAuth needed)
- Moved SESSIONS.md canonical location to /Volumes/990PRO 4T/UBIK/SESSIONS.md
- Created scripts/sync_sessions.sh in UBIK (cp to GdriveMirror, gracefully skips if not mounted)
- Created .git/hooks/post-commit in UBIK (auto-syncs when SESSIONS.md is committed)
- Added Session Journaling (MANDATORY) to UBIK CLAUDE.md
- Updated DeepSeek CLAUDE.md to reference SESSIONS.md at its UBIK path
- Committed and pushed both repos; resolved UBIK remote divergence via rebase
**State left in:**
- Both repos clean and pushed
- SESSIONS.md syncing to GdriveMirror/Ubik on every UBIK commit
- UBIK has unstaged changes in ingestion/ and maestro/ — not touched this session
**Files changed:**
- `DeepSeek/CLAUDE.md`: session journaling rules + UBIK path reference
- `DeepSeek/.gitignore`: added *.gguf, removed token_sync.json
- `DeepSeek/config/settings.py`: removed Drive API fields
- `DeepSeek/config/.env.example`: removed Drive API section
- `UBIK/SESSIONS.md`: created and moved here from DeepSeek
- `UBIK/CLAUDE.md`: added Session Journaling (MANDATORY) section
- `UBIK/scripts/sync_sessions.sh`: created GdriveMirror sync script
**Next session should:**
- Review and commit unstaged changes in UBIK ingestion/ and maestro/ (or confirm they are intentional WIP)
---

## Session: 2026-06-06 21:05 — [Node: Hippocampal]
**Goal:** Verify and fix SESSIONS.md GDrive sync; commit UBIK Neo4j schema work; launch Docker
**Completed:**
- Discovered GdriveMirror is a stale local HFS+ volume — does NOT auto-sync to Google Drive
- Identified correct live GDrive mount: ~/Library/CloudStorage/GoogleDrive-gsanchezurrutia@gmail.com/My Drive/Ubik/
- Fixed scripts/sync_sessions.sh to copy to the live GDrive mount
- Verified SESSIONS.md landed in gsanchezurrutia@gmail.com Drive (My Drive/Ubik/)
- Committed and pushed sync script fix to UBIK
- Reviewed and committed the previously unstaged Neo4j schema upgrade (9 files, 545 insertions)
- Launched Docker Desktop (confirmed running, v28.3.3)
**State left in:**
- Both repos clean and pushed
- SESSIONS.md syncing correctly to gsanchezurrutia@gmail.com GDrive on every UBIK commit
- Docker Desktop running
- UBIK untracked: Ingested_data/, MAESTRO-0.12.0.md, UBIKParallax-source-v5/, UBIKParallax-source-v6/, UBIK_Claude_Prompts/ — not reviewed this session
**Files changed:**
- `UBIK/scripts/sync_sessions.sh`: corrected GDrive target from GdriveMirror to live CloudStorage mount
**Next session should:**
- Decide what to do with UBIK untracked dirs (Ingested_data/, UBIKParallax-source-v5/v6/, UBIK_Claude_Prompts/) — gitignore or commit
---

## Session: 2026-06-07 03:25 — [Node: Hippocampal]
**Goal:** Get maestro start working; set up daily Seagate2T backup for UBIK
**Completed:**
- Diagnosed Docker not starting: daemon socket was stale, Docker Desktop blocked on macOS admin dialog
- Resolved by user running sudo launchctl kickstart -k system/com.docker.vmnetd, then killing stale backend and restarting com.docker.backend directly — Docker daemon came up v29.4.0
- Diagnosed Neo4j container stuck in restart loop for 2 months ("Neo4j is already running (pid:7)") — stale container writable layer, not a PID file in the volume
- Fixed by force-removing ubik-neo4j and recreating via docker compose up -d neo4j — Neo4j came up clean, all data intact
- Created scripts/sync_to_seagate.sh: daily rsync of UBIK project + Docker data to /Volumes/Seagate2T/UBIK/ using --checksum, skips if drive not mounted
- Registered com.ubik.sync-to-seagate.plist as launchd agent (daily at 03:00)
- Ran initial sync: 546MB project + 531MB data transferred successfully
- Committed and pushed sync script to UBIK
**State left in:**
- Docker running (v29.4.0), Neo4j healthy, ChromaDB running
- maestro start should now work (docker + neo4j + chromadb up)
- Daily Seagate2T sync active via launchd
- UBIK untracked dirs still unresolved: Ingested_data/, MAESTRO-0.12.0.md, UBIKParallax-source-v5/v6/, UBIK_Claude_Prompts/
**Files changed:**
- `UBIK/scripts/sync_to_seagate.sh`: created rsync backup script
- `~/Library/LaunchAgents/com.ubik.sync-to-seagate.plist`: created launchd daily job (not in git — user-level agent)
**Next session should:**
- Run maestro start and confirm all services healthy
- Decide on untracked dirs in UBIK (gitignore large ones, commit docs)
---

## Session: 2026-06-12 17:20 — Node: Hippocampal
**Goal:** Consolidate the DeepSeek repo into the UBIK repo, preserving full history
**Completed:**
- Merged /Volumes/990PRO 4T/DeepSeek (github.com/gasu04/DeepSeek14B, branch main @ b82966e) into UBIK as deepseek/ via git subtree add — all 8 DeepSeek commits preserved and reachable
- Moved untracked runtime assets into UBIK/deepseek/: 8.9GB GGUF model, chroma_embeddings/, gdrive_downloads/, credentials.json, token.json, benchmark results, .cache (all covered by deepseek/.gitignore)
- Updated deepseek/config/.env.example embeddings example path to the new location
- Symlinked deepseek/venv -> /Volumes/990PRO 4T/DeepSeek/venv (venv is shared with other projects per deepseek/CLAUDE.md — must NOT be moved or deleted); changed .gitignore venv/ -> venv so the symlink is ignored
**State left in:**
- UBIK master is 3 commits ahead of origin (subtree merge + path fixes + this entry) — not pushed
- Old /Volumes/990PRO 4T/DeepSeek dir still holds: .git + tracked code (now duplicated in UBIK/deepseek/), and the SHARED venv which must stay; the rest of the dir can be archived/deleted after verification
- GitHub repo gasu04/DeepSeek14B is now superseded — consider archiving it
- UBIK untracked dirs still unresolved: Ingested_data/, MAESTRO-0.12.0.md, UBIKParallax-source-v5/v6/, UBIK_Claude_Prompts/
**Files changed:**
- deepseek/ (new): entire DeepSeek14B repo merged under this prefix with full history
- deepseek/config/.env.example: embeddings example path updated to UBIK/deepseek/
- deepseek/.gitignore: venv pattern now also matches the venv symlink
**Next session should:**
- Push UBIK to origin; verify DeepSeek RAG runs from UBIK/deepseek/ (launcher.py / start_ollama.py), then archive the old DeepSeek dir (keep its venv!)
---

## Session: 2026-06-12 18:34 — Node: Hippocampal
**Goal:** Push the DeepSeek consolidation to origin and archive the superseded GitHub repo
**Completed:**
- Pushed UBIK master to origin (1a4df75..ff08e77): subtree merge of DeepSeek14B, path fixes, and session entry now on GitHub
- Archived github.com/gasu04/DeepSeek14B via gh repo archive — verified read-only (isArchived=true); can be unarchived from repo settings if needed
**State left in:**
- UBIK master in sync with origin
- DeepSeek consolidation complete: code + history under UBIK/deepseek/, runtime assets moved, GitHub repo archived
- Old /Volumes/990PRO 4T/DeepSeek dir still on disk — safe to archive/delete EXCEPT venv/ (shared with other projects, reached via UBIK/deepseek/venv symlink)
- UBIK untracked dirs still unresolved: Ingested_data/, MAESTRO-0.12.0.md, UBIKParallax-source-v5/v6/, UBIK_Claude_Prompts/
**Files changed:**
- SESSIONS.md: this entry (no other file changes this session)
**Next session should:**
- Verify DeepSeek RAG runs from UBIK/deepseek/ (launcher.py / start_ollama.py), then clean up the old DeepSeek dir (keep its venv!); decide on UBIK untracked dirs
---

## Session: 2026-06-12 20:05 — Node: Hippocampal
**Goal:** Verify the DeepSeek RAG pipeline runs from its new home at UBIK/deepseek/
**Completed:**
- Confirmed all code paths are Path(__file__)-relative: settings.py resolves embeddings_base_dir to <project_root>/chroma_embeddings, so the moved ChromaDB data is picked up automatically at the new location — no code changes needed
- Verified the venv symlink (deepseek/venv -> old DeepSeek/venv) works: Python 3.13.7 runs, key deps (requests, chromadb, dotenv) import cleanly
- Live Ollama check (api/tags) was not run — session ended before service-level verification
**State left in:**
- Static verification passed; runtime verification incomplete: Ollama service check, model availability (deepseek-r1:14b), unit tests, and an end-to-end RAG query still pending
- UBIK master in sync with origin (before this entry)
- Old /Volumes/990PRO 4T/DeepSeek dir untouched — keep venv/, rest deletable after full verification
- UBIK untracked dirs still unresolved: Ingested_data/, MAESTRO-0.12.0.md, UBIKParallax-source-v5/v6/, UBIK_Claude_Prompts/
**Files changed:**
- SESSIONS.md: this entry (no code changes this session)
**Next session should:**
- Finish runtime verification from UBIK/deepseek/: pytest tests/unit, Ollama up + model present, one end-to-end RAG query; then clean up old DeepSeek dir (keep its venv!)
---

## Session: 2026-06-12 20:10 — Node: Hippocampal
**Goal:** Port the ingestion program into UBIK — Phase 0 (recon), Phase 1 (scaffolding/config/registries), Phase 2 (source acquisition)
**Completed:**
- Phase 0 recon: mapped ingestion/ pipeline (models, transcript_processor, cli discovery); found ChromaDB/Neo4j/MCP all DOWN (containers exited Jun 7); located raw Tactiq zips (~1,261 files, Seagate2T), 27 .transcript outputs, manifest (330 records, last ingest 2026-03-19); no _protagonist_only.txt or Known_Persons export existed locally
- CRITICAL recon finding: ChromaDB bind mount (UBIK/data/chromadb) is stale/empty (Jan 17) — real collections live in the container's writable layer (image persists to /data, compose maps /chroma/chroma). docker rm ubik-chromadb would DESTROY the 192 stored memories
- Somatic vLLM confirmed at 100.92.95.39:8002 (adrian-1, DeepSeek-R1-Distill-Qwen-14B-AWQ); 100.79.166.114:8002 is an unrelated nginx site
- Phase 1: created sources/{tactiq,gemini,fireflies,letters,memory_notes,constitution}, enriched/, pending_review/, approved/, quarantine/{source,enrichment}, registry/, qa/, config/, logs/
- registry/known_persons.yaml: converted from "Digital Twin Taxonomy and Registry Setup" CSV export, then corrected per Gines — 12 persons (Maggie wife; sons Adrian/Gines Alberto/Jaime; nueras Carissa/Delia/Sofia; nietas Elena/Mathilde; Leticia Zuno is THERAPIST not partner); schema: privacy_tier private/therapy/family/business + ubik_access_level
- registry/content_types.yaml: 7 types; voice-corpus invariant enforced by loader (diarization_trust below "full" can never be voice_corpus_eligible)
- ingest/registry.py: fail-loud loaders + CSV converter (validates before replacing)
- config/ingestion_config.py: env-driven dataclasses; per-privacy-tier endpoint routing, fail-safe (all tiers → sensitive/Somatic endpoint unless UBIK_STANDARD_PRIVACY_TIERS relaxes)
- Spanish-first directive recorded (qa/learned_rules.md + Claude memory): parsing/names/prompts Spanish primero; bare Gines/GASU = father, son = "Gines Alberto"/"Gines Hijo"
- Phase 2: fetch_sources.py — --local mode demoed (3 real Tactiq .docx, byte-identical, SHA-256 dedup, sources/MANIFEST.jsonl); --gdrive OAuth desktop flow written, awaiting credentials
- .gitignore: credentials/tokens + all raw content dirs ignored (verified); 58/58 tests pass
**State left in:**
- ChromaDB/Neo4j/MCP containers still DOWN (untouched — start them next session; fix bind-mount mismatch BEFORE any docker rm)
- 3 demo files in sources/tactiq/ + manifest; /tmp/tactiq_demo can be deleted
- qa/ docs are placeholders pending Claude.ai session docs; Google Sheet now BEHIND known_persons.yaml (update Sheet: Elena=nieta, add Maggie/Jaime/Gines Alberto/nueras)
- Known issue logged: SpeakerTurnParser regexes break on accented names ([A-Za-z]); Tactiq raw files are .docx but transcript parser expects .transcript — pre-parse step needed in Phase 3
**Files changed:**
- ingestion/{fetch_sources.py,ingest/registry.py,config/*,registry/*,qa/*,tests/test_{registry,ingestion_config,fetch_sources}.py}: new (Phases 1-2)
- ingestion/.env.example: +13 enrichment/gate/Drive vars
- .gitignore: credentials + raw-content ignore rules
- SESSIONS.md: this entry
**Next session should:**
- Phase 3 (enrichment) per Gines's spec; before that: start ubik-chromadb/ubik-neo4j containers and migrate ChromaDB data out of the container writable layer onto the bind mount (data-loss risk), and get collection counts (deferred Checkpoint 0 item)
---

## Session: 2026-06-14 12:00 — Node: Hippocampal
**Goal:** Phase C1 — ChromaDB de-containerization + data rescue (run Chroma natively off a plain dir, out of Docker). Steps 0–3 COMPLETE; native 1.3.7 proven against the test copy. Paused at Checkpoint 3 before the destructive promote/retire phase.
**Completed (Phase 1 evidence + Steps 0–2 of C1 brief):**
- Phase 1 recon CORRECTED the earlier diagnosis: container ubik-chromadb is UP (unhealthy but serving on :8001). Two sqlite files exist — `/chroma/chroma/chroma.sqlite3` (the bind mount → host `data/chromadb`) is STALE (164K, Jan 17, no segment dirs); the LIVE db is at `/data` (37M, written today, 2 vector-segment UUID dirs). Chroma persists to `/data` (IS_PERSISTENT=True, PERSIST_DIRECTORY unset → image default), NOT to the bind-mounted `/chroma/chroma`. The 37M `/data` is on the container's writable-layer trapdoor.
- Live collection counts (NEW proof-of-success target, supersedes the old "192"): ubik_episodic=1075, ubik_semantic=15, ubik_intellectual=0, **TOTAL 1090**.
- Step 0: pinned container version = **chroma 1.3.1** (`chroma --version`; image is distroless/Rust-core — no python, no dist-info; API `/v2/version` returns "1.0.0" = API schema, not pkg). Image pin: chromadb/chroma@sha256:ded4839c…ffa67ac, sha256:9f7e6e4c…d83764, built 2026-01-13.
- Step 1: quiesced writers — SIGTERM'd MCP server (PID 5890, `/Volumes/990PRO 4T/UBIK/hippocampal/mcp_server.py`, was on :8080). No ingestion/RAG/uvicorn/ollama running. Confirmed DB idle: `/data/chroma.sqlite3` mtime stable at 14:12:25 UTC across samples, no -wal/-shm. Container NOT stopped (writers only).
- Step 2: rescued live data → `/Volumes/990PRO 4T/UBIK/chromadb_data/` via `docker cp ubik-chromadb:/data/.`. Verified 37M, sqlite (28,094,464 B) + both segment UUID dirs with real HNSW files; **SHA-256 of chroma.sqlite3 matches container source exactly** (1d39a033…9e9c045d). Backed up to TWO locations (dated 20260614_115723): local `UBIK/backups/` and `/Volumes/Seagate2T/UBIK/backups/` (separate physical drive).
- VERSION RESOLUTION (Gines's call): `chromadb==1.3.1` does NOT exist on PyPI (versions jump 1.3.0 → 1.3.2 — the Docker image's Rust server version is on a different track from the PyPI package). Decision: install latest 1.3.x = **1.3.7**, since the real compatibility axis is the stable 1.3.x on-disk format, not a package-to-binary version match that never existed.
- Step 3 (COMPLETE): installed `chromadb==1.3.7` into isolated venv `~/ubik-chromadb-venv` (Python 3.13.7; note `chroma --version` self-reports "1.2.4" — same CLI/pkg skew, ignore it; `chromadb.__version__`=1.3.7 is authoritative). Recreated fresh test copy `UBIK/chromadb_data_test/` from rescue + recorded pre-boot fingerprint. Booted `chroma run --path chromadb_data_test --port 8002` (background, log /tmp/chroma_native_8002.log).
- CHECKPOINT 3 PASSED — native 1.3.7 serves the data fully:
  1. Counts on :8002 EXACT: ubik_episodic 1075, ubik_semantic 15, ubik_intellectual 0, TOTAL 1090.
  2. Sample similarity query on ubik_episodic returns 3 sane non-empty results (self-match dist=0.0000 → HNSW index genuinely read, not just SQLite metadata).
  3. Dir diff (test post-boot vs untouched rescue): all HNSW segment files byte-identical, same file set, schema identical, user_version 0=0, chroma `migrations` table identical (17=17 rows → NO format migration). The chroma.sqlite3 SHA-256 DID change (1d39a033… → b6827411…) but the sole cause is the `acquire_write` lock-bookkeeping table incrementing 17→18 on server startup. Benign, not a format migration.
- CONCLUSION: de-containerization is proven viable on 1.3.7; API export/import fallback NOT needed.
**State left in:**
- Native ChromaDB 1.3.7 STILL RUNNING on :8002 against the disposable test copy (holding the write lock — that's the acquire_write increment). Disposable; can be killed anytime.
- Container ubik-chromadb STILL UP on :8001, untouched — full authoritative fallback, holds all 1090 records. NOT yet retired.
- MCP server (:8080) STOPPED by us — must be manually restarted (`mcp_server.py` from `UBIK/hippocampal/`) to resume normal ops; no auto-restart wrapper was running.
- Canonical rescued data `UBIK/chromadb_data/` + 2 dated backups (local `UBIK/backups/` + `Seagate2T/UBIK/backups/`, stamp 20260614_115723) — all UNMODIFIED. `UBIK/chromadb_data_test/` is the boot target (sqlite mutated by the benign lock write; disposable).
**Files changed:**
- SESSIONS.md: this entry (no repo code changed; only data rescue + dirs/venv created outside git)
**Next session should:**
- Run the destructive promote/retire phase (NOT yet specified — Gines writes it): kill the :8002 test server, point native ChromaDB at the canonical `UBIK/chromadb_data/`, stand it up as a persistent service (launchd) on its final port, re-verify 1075/15/0, then retire the Docker container (ChromaDB only — Neo4j/Docker handled later) and repoint the MCP server + ingestion config at the native endpoint. Restart MCP server last. Keep backups until native is the proven sole source of truth.
---

## Session: 2026-06-14 22:15 — Node: Hippocampal
**Goal:** Phase C2 — promote native ChromaDB to canonical on :8001 (container retained as frozen fallback). Reached Checkpoint 4; launchd auto-start BLOCKED by macOS TCC — awaiting Gines's fix decision.
**Completed (C2 Steps 0–3 + Step 4 prep):**
- C2 Step 0: cleanly stopped the C1 :8002 test server (SIGTERM, lock released, :8002 free). Kept `chromadb_data_test/` as disposable safety copy.
- C2 Step 1: pre-cutover sha256 fingerprint of canonical `chromadb_data/` (sqlite 1d39a033…, migrations=17, acquire_write=17, embeddings=1090, all 9 segment files hashed).
- C2 Step 2 (cutover): `docker stop ubik-chromadb` (Exited 0, NOT removed — id d93d39a7f6c9 frozen fallback, :8001 freed) → started native chroma 1.3.7 against CANONICAL dir on :8001. Heartbeat 200.
- C2 Step 3 (integrity gate): counts EXACT 1075/15/0/1090; episodic + semantic similarity queries sane (self-match ~0.0). DIFF vs Step-1 baseline: episodic segment byte-identical, but **semantic segment `bbbb790a` had data_level0.bin + length.bin change (SAME size, content differs)** = in-place HNSW re-serialization on first native open, NOT a format migration (migrations still 17, user_version 0, same file set, data intact). sqlite went 1d39a033…→b6827411… (SAME deterministic output as the C1 test boot) — fully explained by acquire_write 17→18 + the one-time segment re-persist. NOTE: C1's "byte-identical segments" was actually SIZE-only comparison; sha256 in C2 surfaced the re-serialization.
- BASELINE CORRECTION (per Gines): re-fingerprinted canonical in its stabilized post-native state → `/Volumes/990PRO 4T/UBIK/chromadb_data.BASELINE.post-native.txt` (semantic data_level0.bin=f2bf7efb…, length.bin=7a12e561…, sqlite=b6827411…, acquire_write=18). This is the NEW reference; going forward CLEAN reopens must be sha256-IDENTICAL to it (only acquire_write may increment) — any segment shift after this = real anomaly (first-open re-serialization is one-time, must not recur per boot).
- Preserved pre-native provenance: relabeled both dated backups → `chromadb_data_PRE-NATIVE_1d39a033_20260614_115723` (local + Seagate), `chmod -R a-w` write-locked, wrote `UBIK/backups/PRE-NATIVE.README.md`. Confirmed Seagate sync (`sync_to_seagate.sh`, rsync -a --delete) mirrors source→`Seagate2T/UBIK/project/` so it only READS source (can't clobber the locked backup); the standalone `Seagate2T/UBIK/backups/` copy is outside the rsync dest.
- C2 Step 4 (launchd): wrote guard wrapper `~/ubik-chromadb-venv/start_chromadb_native.sh` (waits for `chromadb_data/chroma.sqlite3` before exec, refuses to create empty dir) + `~/Library/LaunchAgents/com.ubik.chromadb.plist` (KeepAlive+RunAtLoad). **FAILED: launchd job flapped 27× with "operation not permitted" on the external volume.**
**BLOCKER — Checkpoint 4 (macOS TCC):**
- Built a one-shot launchd probe: a LaunchAgent-spawned process (uid 501 gasu) gets EPERM on BOTH read-open and write-open of `/Volumes/990PRO 4T` (stat/-f passes); interactive Terminal context does BOTH fine. → macOS TCC denies launchd processes access to the removable volume where the data lives. The log path was a red herring; chroma-under-launchd can't read the data either. TCC grant is a GUI action only Gines can do (TCC.db is SIP-protected).
- Decision pending — options presented: (1) grant Full Disk Access to the launchd-exec'd venv Python (keeps data on 990PRO; fragile across python upgrades; manual GUI step) [recommended if 990PRO placement matters], (2) relocate the 37MB ChromaDB data to internal disk ~/ubik/chromadb_data (most robust, kills TCC + mount-timing deps, but diverges from "data on 990PRO" + needs Seagate-sync tweak), (3) LaunchDaemon/root — not recommended.
**State left in:**
- Native chroma 1.3.7 RUNNING on :8001 via MANUAL nohup (interim; interactive ctx has volume access), serving canonical dir, 1075/15/0/1090. NOT boot-persistent yet.
- `com.ubik.chromadb` launchd agent UNLOADED (not deleted — wrapper+plist stay for whichever fix Gines picks). Probe artifacts cleaned up.
- Container ubik-chromadb EXITED (0), not removed — frozen authoritative fallback.
- Pre-native backup (1d39a033…) write-locked + labeled, local + Seagate; new post-native baseline file saved. Test copy `chromadb_data_test/` retained.
- MCP server (:8080) still STOPPED since C1 Step 1 — restart `mcp_server.py` from `UBIK/hippocampal/` when resuming normal ops.
**Files changed:**
- NEW: `~/ubik-chromadb-venv/start_chromadb_native.sh`, `~/Library/LaunchAgents/com.ubik.chromadb.plist` (both local disk, not in repo)
- NEW: `UBIK/chromadb_data.BASELINE.post-native.txt`, `UBIK/backups/PRE-NATIVE.README.md`
- backups relabeled + write-locked (local + Seagate)
- SESSIONS.md: this entry
**Next session should:**
- Get Gines's TCC fix decision (#1 grant FDA vs #2 relocate to internal). Then finish C2 Step 4: load launchd agent, confirm it auto-starts native on :8001, and run the Checkpoint-4 reload/reboot test — judging segments against the NEW post-native baseline, expecting TRUE sha256-identical segments on a clean restart (if segments shift again → STOP, real anomaly). Only after launchd proven: later phase = retire container + repoint/restart MCP server.
---

## Session: 2026-06-15 21:50 — Node: Hippocampal
**Goal:** C2 Step 4 fix (Gines's decision: grant FDA, keep data on 990PRO) through Step 6 (health check) — close out the native ChromaDB cutover.
**Completed:**
- C2 Step 4 fix: copied the venv's Python (real Mach-O target, not the symlink) to a stable, private path `~/ubik-bin/ubik-chroma-python` — deliberately outside both the Homebrew Cellar (so `brew upgrade python@3.13` can't change the granted binary's identity) and the venv tree (so `python -m venv` recreation can't silently overwrite it). Verified it imports `chromadb`/`chromadb_rust_bindings` correctly via `-S` + explicit venv site-packages insertion before wiring it in. Gines granted Full Disk Access to this path via System Settings.
- Found and fixed a second blocker while wiring it in: `start_chromadb_native.sh`'s final `exec "$CHROMA" run ... >> logfile` redirect is opened by `/bin/zsh` itself (shell redirections open before `exec` replaces the process image) — confirmed live via repeating `operation not permitted` entries in the guard log that referenced the log path, not chroma's data path. That open() would still EPERM under launchd even with FDA granted only to Python. Fixed by moving log-file opening inside the Python process via `os.dup2` onto fds 1/2, so the FDA-granted binary is the only actor touching the external volume. Wrapper now invokes `chromadb.cli.cli.app()` directly via `-c`, replicating `bin/chroma`'s own entry point.
- Reloaded `com.ubik.chromadb`: running under launchd (confirmed via `launchctl list`), listening on `:8001`, heartbeat OK. Counts exact: ubik_episodic=1075, ubik_semantic=15, ubik_intellectual=0, total=1090.
- **Baseline-verified reload test result — UPDATES the Checkpoint-4 rule from the previous session:** `bbbb790a-.../data_level0.bin` (the `ubik_semantic` HNSW segment) does NOT stabilize to a fixed sha256. Hashed it across three separate opens tonight (initial launchd start, then two explicit `launchctl stop`/`start` cycles) and got three different hashes (`7ac1a2c5…`, `42199a55…`, plus the original baseline `f2bf7efb…`) — same file size (167600 B) every time, `length.bin` for the same segment unchanged, `migrations`=17, `user_version`=0 throughout, `acquire_write` incrementing by exactly 1 per restart, counts exact every time. Conclusion (Gines's call): this segment re-serializes on **every** open, not once-per-lock-release as originally assumed. **New integrity rule, supersedes `chromadb_data.BASELINE.post-native.txt`'s blanket sha256 expectation:** `bbbb790a/data_level0.bin` and `length.bin` → check size only (167600 B / 400 B), may change content; all other segment files → sha256-identical to baseline; `migrations`=17; `user_version`=0; `acquire_write` may increment; counts=1075/15/0 exact.
- C2 Step 5: restarted MCP server via `./run_mcp.sh start` (PID 10060, correct venv `/Volumes/990PRO 4T/DeepSeek/venv`). Confirmed full stack end-to-end with a real `query_semantic` call via `fastmcp.Client` against `http://localhost:8080/mcp` (the Somatic `hippocampal_client.py` couldn't be reused as-is — it pulls in Somatic's `Settings`, which requires `VLLMSettings.model_path`, irrelevant on Hippocampal). Server log shows the lazy ChromaDB client connecting to `http://localhost:8001/...` (not the container) and returning 2 real therapy-transcript-derived results with sane relevance scores.
- C2 Step 6 (health check): ran `health_check.py` — found Docker Desktop entirely down (separate, pre-existing issue, unrelated to the chroma migration; not on tonight's task list but blocked Neo4j). Started Docker Desktop + `docker start ubik-neo4j` (left `ubik-chromadb` exited on purpose — frozen fallback, untouched). Re-ran: **4/5 passed** (Neo4j: 1288 nodes, CoreIdentity present; ChromaDB, MCP Server, Tailscale all green). The one remaining red is `health_check.py`'s docker-container check expecting `ubik-chromadb` to be `running` — stale by design now that chroma is native; not a real fault.
- Self-correction logged: earlier this session I mistakenly ran `tccutil reset SystemPolicyAllFiles` while trying to read TCC.db read-only — this is destructive and reset Full Disk Access for **every app on this Mac**, not just chroma. Flagged to Gines immediately. Any app that previously had FDA (Terminal, backup tools, etc.) will need it re-granted manually; not yet audited.
**State left in:**
- Native ChromaDB 1.3.7 running under launchd on `:8001` against the canonical `chromadb_data/`, FDA-backed by `~/ubik-bin/ubik-chroma-python`. Container `ubik-chromadb` remains stopped (not removed) — frozen fallback.
- MCP server running (PID 10060) via `./run_mcp.sh`, confirmed talking to native chroma.
- Docker Desktop + `ubik-neo4j` running (started this session — were down for an unknown, unrelated duration before tonight).
- `chromadb_data.BASELINE.post-native.txt` is now PARTIALLY STALE (its sha256 for `bbbb790a/data_level0.bin` is just one of many valid values) — no replacement script written yet.
- FDA grants for apps other than `~/ubik-bin/ubik-chroma-python` are in an unknown state after tonight's accidental `tccutil reset SystemPolicyAllFiles` — not audited.
**Files changed:**
- NEW: `~/ubik-bin/ubik-chroma-python` (local disk, not in repo — the FDA-granted interpreter copy)
- `~/ubik-chromadb-venv/start_chromadb_native.sh`: switched from `bin/chroma` to the stable interpreter; moved log redirection from shell `>>` to in-process `os.dup2` (local disk, not in repo)
- SESSIONS.md: this entry
**Next session should:**
- Write `verify_chromadb_baseline.sh` (or similar) encoding the corrected integrity rule above, and regenerate `chromadb_data.BASELINE.post-native.txt` (or split it: fixed-hash files vs. size-only files) so the contract is actually checkable again.
- Audit Full Disk Access grants for other apps (Terminal at minimum) after the accidental `tccutil reset SystemPolicyAllFiles` — re-grant whatever's missing.
- Decide whether to update `health_check.py`'s docker check to stop expecting `ubik-chromadb` running (or explicitly treat `exited` as the healthy/expected state post-migration), so the script stops crying wolf.
- Investigate (low priority, data is fine either way) why `ubik_semantic`'s HNSW segment re-serializes every open while `ubik_episodic`'s (9.8MB, vs. 167KB) never has — likely just a size/threshold artifact of chromadb's own segment-loading code, not anything UBIK-specific.
- Write `START_HERE.md` succession document at UBIK repo root (identified as a Tier 0 gap; not done this session — in progress as the next immediate task).
---

## Session: 2026-06-19 — Node: Hippocampal
**Goal:** Review and close out the native ChromaDB migration (C2) after a 4-day soak; write session entry.
**Completed:**
- Confirmed system stable after 4 days running under new launchd-managed architecture: `com.ubik.chromadb` agent holding (PID 9863, exit 0), ChromaDB heartbeat OK, counts exact (ubik_episodic=1075, ubik_semantic=15), MCP server responding (HTTP 406 — correct), Neo4j container running, `ubik-chromadb` container correctly still exited (frozen fallback untouched).
- No new code changes this session. All Tier 0 tasks from the June 15 session are confirmed complete: stable-path Python copy (`~/ubik-bin/ubik-chroma-python`) + FDA grant, launchd auto-start, MCP server running, `START_HERE.md` written.
- **C2 native ChromaDB migration formally closed.** The migration path (C1 recon → data rescue → native proof → C2 cutover → launchd) is fully resolved. The container (`ubik-chromadb`, id `d93d39a7f6c9`) remains stopped but not removed — frozen fallback; may be removed in a future cleanup session once native has run satisfactorily for longer.
**State left in:**
- All services healthy: launchd → native ChromaDB (:8001) → MCP server (:8080) → Neo4j (:7687). Docker Desktop running.
- Known open items from June 15 (unchanged, still pending): write `verify_chromadb_baseline.sh` with corrected integrity rule; audit FDA grants for other apps after the accidental `tccutil reset SystemPolicyAllFiles`; update `health_check.py` to stop flagging the intentionally-exited `ubik-chromadb` container as a fault.
**Files changed:**
- SESSIONS.md: this entry
**Next session should:**
- Begin Phase 4 work (per original project plan) or address the June 15 open items above, whichever Gines prioritises.
---

## Session: 2026-06-20 — Node: Hippocampal
**Goal:** Implement the Phase 3 enrichment pipeline (brief: build stages 1–5 docx_parser/enrichment_agent/person_resolver/gate1_cli/chromadb_writer + run_phase3.py + checkpoints).
**Key finding (Think-Before-Coding): the brief described greenfield modules, but most of Phase 3 already existed, built and tested.** Recon mapped every requested module to existing code: `enrich.py` (parse+enrich, with `<think>` strip, YAML-fence parse, quarantine, SHA-256 resumable manifest, backoff+jitter); `ingest/registry.py` (resolution; Gines/Gines Alberto already encoded in `known_persons.yaml`); `interactive_ingest.py` (Gate 1 human review); `ingest/mcp_writer.py` (direct ChromaDB+Neo4j writer); `spike_enrichment.py` (Checkpoint-2 gate). Building the 5 new modules verbatim would have duplicated ~130KB of tested code into a divergent pipeline (violates CLAUDE.md §1.2/§1.3/§2.3). **Gines chose "fill gaps in existing code"** over literal build. The pipeline was *inert* because three content files were still PLACEHOLDER (`prompts/enrichment_v1.md`, `qa/schema.md`, `qa/rubric.md`) — the brief itself supplied that missing content.
**Completed (gaps filled):**
- Populated `prompts/enrichment_v1.md` (the brief's verbatim enrichment prompt, fitted to the `{{KNOWN_PERSONS}}/{{SOURCE_HINTS}}/{{SCHEMA}}/{{CONTENT}}` template contract; Spanish-first) — unblocks `enrich.py`/`spike_enrichment.py` (which fail-loud on PLACEHOLDER).
- Authored `qa/schema.md` JSON Schema (Draft 2020-12) for the brief's field set (meeting_type enum, type_inferred_from, meeting_date, participants_detected, language, main_topics 3–7, key_decisions, diarization_status enum, diarization_warning, voice_corpus_eligible, enrichment_confidence). `additionalProperties: true` by intent (avoid quarantining over harmless extras); aligned the spike sheet's field-name fallbacks to the new names.
- Populated `qa/rubric.md` with Gate 1 semantics (confidence bands → default action; A/F/Q/S/X; therapy never voice-eligible & never written; mono never voice-eligible; unresolved doesn't block).
- NEW `ingest/diarization.py` — accent-safe per-file mono/multi detection (Unicode `\w`, NOT `[A-Za-z]`), with non-speaker line-opener suppression (`Nota:`/`Fecha:`…). Deliberately a new module so the untouchable `transcript_processor.py` is left alone (its ASCII-only `SpeakerTurnParser` accent bug at lines 677/701 remains pre-existing, flagged not fixed).
- `enrich.py` — added the **Python-side hard-rule override** (`_apply_hard_rules`): never trust the LLM for voice eligibility; force `voice_corpus_eligible=false` on mono diarization or `meeting_type==therapy`; set `diarization_status`/`diarization_warning` from the detector for transcripts; plus `enrichment_confidence` read as the confidence fallback for the manifest/routing.
- `ingest/mcp_writer.py` — added **SHA-256 source-document dedup** to `store_episodic` (query by `source_sha256` metadata before add; idempotent `{"status":"duplicate"}` on re-ingest) + `extra_metadata` merge for the Phase-3 fields. This is **CLAUDE.md §3.4.1 Tier 1 entry #5** ("planned — tag once built"); module docstring now declares Tier 1; **tagged the registry row in CLAUDE.md with the real location** (the registry text explicitly invited this once built).
- NEW `ingest/person_resolver.py` — wraps the registry to add 3-way status (resolved/unresolved/**ambiguous**) + accent-normalized fallback + `summarize_resolution` (all_resolved/some_unresolved/ambiguous). No fuzzy/substring matching by design (a wrong auto-resolution mis-attributes words permanently).
- NEW `run_phase3.py` — orchestrator with the brief's CLI (`--stage`/`--dry-run`/`--limit`, per-stage summary line). Wires existing components; dry-run = stages 1–3; stage 4 counts the queue but never auto-approves (human gate); stage 5 excludes therapy, treats duplicates as skipped.
- Tests: +43 (diarization 9, mcp_writer dedup 6 incl. the deliberate silent-duplicate probe, person_resolver 14, run_phase3 10, enrich hard-rule 4; rewrote 1 obsolete placeholder test). Suite 146 → **189 collected, 185 pass**.
**Checkpoints:**
- **CP1 (parse) PASS on real data**: `run_phase3.py --stage 1` parsed all 3 real `.docx`, 0 quarantined; diarization correct (actinium RFCA = mono/unlabeled — the Tactiq case the brief warns about; the other two = multi).
- **CP6 (suite)**: 185 pass / 4 fail — the 4 are pre-existing `test_tracker.py` FileMover failures (`ingested/` vs expected `<source>_ingested/`), unrelated to Phase 3 (nothing in the tracker path imports the changed code; `test_tracker.py` unmodified). Flagged, not fixed (Surgical Changes).
- **CP2 (enrich quality spike) BLOCKED**: Somatic vLLM unreachable (`100.92.95.39:8002` and `:8002` both time out) and no `UBIK_ENRICHMENT_MODEL` in `ingestion/.env`. The brief calls this "THE GATE" before Phase 4.
- **CP3 (resolution) / CP5 (write) live runs BLOCKED downstream** (CP3 needs enriched files from CP2; CP5 needs approved files + live ChromaDB and would mutate the real corpus). Their logic is proven by unit tests (incl. dedup silent-duplicate probe).
- **CP4 (Gate 1) BLOCKED**: human-in-the-loop.
- §2.1 self-test clean (no hardcoded IPs in new code). `ruff`/`mypy` not installed in the venv → CI-style lint/type checks not run locally (code is fully type-hinted).
**State left in:**
- Phase 3 code complete and unit-tested; pipeline no longer inert. NOT committed yet (awaiting commit decision — repo is on `master`, and the brief's suggested message "checkpoint 6 green" is inaccurate given the 4 pre-existing fails).
- Native ChromaDB / MCP / Neo4j still healthy from the 2026-06-19 session.
**Files changed:**
- NEW: `ingestion/ingest/diarization.py`, `ingestion/ingest/person_resolver.py`, `ingestion/run_phase3.py`, `ingestion/tests/unit/{test_diarization,test_mcp_writer_dedup,test_person_resolver,test_run_phase3}.py`
- `ingestion/prompts/enrichment_v1.md`, `ingestion/qa/schema.md`, `ingestion/qa/rubric.md`: PLACEHOLDER → real content
- `ingestion/enrich.py`: hard-rule override + per-file diarization + confidence fallback
- `ingestion/ingest/mcp_writer.py`: SHA-256 dedup + extra_metadata + Tier-1 docstring
- `ingestion/tests/test_enrich.py`: +4 hard-rule tests, rewrote obsolete placeholder test
- `CLAUDE.md`: §3.4.1 entry #5 location tagged to `mcp_writer.py`
- SESSIONS.md: this entry
**Next session should:**
- Power on + reach the Somatic vLLM endpoint and set `UBIK_ENRICHMENT_MODEL` (+ `SOMATIC_TAILSCALE_IP`/`VLLM_PORT`) in `ingestion/.env`, then run **CP2** (`run_phase3.py --stage 2 --dry-run --limit 8`, hand-score per `qa/rubric.md`; gate = conf ≥0.7 on ≥6/8). Only then CP3 live, CP4 (human Gate 1 via `interactive_ingest.py`), CP5 (write to ChromaDB).
- Decide the commit (branch off `master`? honest message reflecting gaps-filled + 4 pre-existing tracker fails). Optionally fix the pre-existing `FileMover` `<source>_ingested/` bug separately.
---

## Session: 2026-06-29 — Node: Hippocampal
**Goal:** Fold two carried-over pending items into the canonical log so they are not lost, and reaffirm the repo (not Drive) as the source of truth for SESSIONS.md.
**Completed:**
- Verified the Drive copy (`My Drive/Ubik/SESSIONS.md`) is byte-identical to the repo copy — no divergence to reconcile. The repo remains canonical; Drive is the downstream mirror written by `scripts/sync_sessions.sh` on post-commit.
- Recorded two pending items surfaced in a prior Drive-based (Claude.ai) working session so they survive into the canonical log (see "Next session should").
**State left in:**
- No code changes this session — only this log entry added.
- Workflow note reaffirmed: decisions mirror to the repo via `sync_sessions.sh` on the next commit; Drive is a convenience mirror, the repo file is canonical. Entries drafted from Drive should be folded back into the repo file when next in Claude Code.
**Files changed:**
- SESSIONS.md: this entry
**Next session should:**
- Retire the installed `ubik-memory-sweep` skill so it does not compete with the current memory/journaling system.
- Update the ingestion loader for the new fields and the `EPISODIC` token.
---

## Session: 2026-07-04 — Node: Hippocampal
**Goal:** Session wrap-up. Commit and publish the SESSIONS.md pending-items entry to the UBIK repo. (Only UBIK-project work is logged here; other unrelated tooling done this session — worldmonitor, workspace-agent, gcloud, CrewAI, notebooklm-py — is intentionally omitted.)
**Completed:**
- Committed the 2026-06-29 pending-items entry as `46b5095` on branch `phase3-enrichment-pipeline` (1 file, 15 insertions). The `post-commit` hook ran `scripts/sync_sessions.sh` → mirrored SESSIONS.md to `gsanchezurrutia@gmail.com/Ubik`; verified repo and Drive byte-identical (repo canonical, Drive downstream mirror).
- Pushed `phase3-enrichment-pipeline` to origin — this was the branch's **first push**, so it also published the Phase 3 enrichment pipeline commit `c33e2e4` (gaps-filled build from the 2026-06-20 session) that had been local-only until now. GitHub offered a PR link; branch NOT merged to `master`.
**State left in:**
- `phase3-enrichment-pipeline` now on origin (`46b5095` confirmed on `origin/phase3-enrichment-pipeline`), not merged to `master`. Branch carries both the Phase 3 enrichment work and the SESSIONS.md pending-items entry.
- Two carried-over pending items still open (see 2026-06-29 entry): retire the `ubik-memory-sweep` skill; update the ingestion loader for the new fields + `EPISODIC` token.
**Files changed:**
- SESSIONS.md: this entry.
**Next session should:**
- Decide whether to open/merge the `phase3-enrichment-pipeline` PR into `master`.
- Address the two pending items: retire `ubik-memory-sweep`; update the ingestion loader (new fields + `EPISODIC` token).
---

## Session: 2026-07-04 (b) — Node: Hippocampal
**Goal:** Make one Maestro instance (Hippocampal) start/shutdown/manage BOTH nodes; delete Maestro from the Somatic node after testing.
**Completed (branch `maestro-remote-control`, commit `cb1f1ab`, maestro v0.13.0):**
- Recon: status was already cross-node (Tailscale probes) but `ensure_all_running` skipped remote and shutdown was local-only; Somatic vLLM stop used abrupt SIGKILL → **VRAM leak** (the imperative Gines flagged).
- Read the canonical Somatic graceful lifecycle `somatic/inference/vllm_server.py`: on SIGTERM it runs vLLM CUDA cleanup (`destroy_model_parallel`/`empty_cache`) and waits up to 60s before force-kill — that window is what frees VRAM.
- **SSH transport** — the `windows-server` ssh config forces `RequestTTY yes`+`RemoteCommand wsl ~`; overrode with `BatchMode/RequestTTY=no/RemoteCommand=none` and deliver the bash script over **stdin** to `wsl bash -s` (sidesteps cmd.exe→WSL quoting). Somatic reachable non-interactively (key-based).
- **Persistence discovery**: a plain `nohup … &` over `wsl bash -s` is torn down when `wsl.exe` exits. WSL has systemd (PID 1) and **user lingering is enabled for gasu** → launch services as transient **systemd user units** via `systemd-run --user` (bus at `/run/user/1000`); they survive the SSH session. No sudo (none available; passwordless sudo = NO).
- New `maestro/remote.py` (`RemoteExecutor.from_config`, `run`, `check`). `config.py`: SomaticConfig `ssh_host=windows-server`, `use_wsl`, `ssh_connect_timeout`, `remote_ubik_root=/home/gasu/ubik`.
- `vllm_service`/`whisperx_service`: node-aware `start`/`stop`. Remote start = `systemd-run --user --unit=ubik-{vllm,whisperx}` (vLLM via `vllm_server.py --rtx5080 --skip-checks --config … --model … --port …`, `KillSignal=SIGTERM`, `KillMode=mixed`, `TimeoutStopSec=90`). Remote stop = `systemctl --user stop` (graceful → VRAM release), pkill fallback for unmanaged procs, reports `nvidia-smi`. vLLM `max_wait_s` 120→300.
- `orchestrator.ensure_all_running` + `shutdown.orderly_shutdown`: default cluster-wide, probe each service at its real host, remote services skip local SIGKILL; `--local-only` flag added to `maestro start`/`shutdown`.
- **Live test PASSED (real hardware)**: remote start → vLLM healthy (HTTP 200), VRAM 32077 MiB used; graceful stop → **VRAM 746 MiB (fully released)**. Imperative satisfied.
- Fixed a **pre-existing** vLLM config bug (blocked startup, unrelated to Maestro): only WSLg `/Xwayland` (746 MiB) uses the GPU; `gpu_memory_utilization: 0.90` left KV cache short for 98K ctx and `0.95` exceeded free VRAM → set **0.93** on Somatic (`config/models/vllm_config.yaml`; gitignored on Hippocampal). vLLM now uses ~full 32 GB.
- Deleted `~/ubik/maestro` on the Somatic node (git-tracked & clean → recoverable via `git checkout`); remote control depends only on `somatic/inference/vllm_server.py`, `somatic/whisperx_server.py`, `config/`, venvs, systemd — none in `maestro/`.
- Tests: new `test_remote.py`; updated orchestrator/shutdown tests for cluster-wide semantics. Affected suites green (159 passed). Remaining failures are all **pre-existing** and env-specific (no conda on the Mac; venv-path detection; `whisperx` registry-count debt) — verified identical on stashed original code; plus a pre-existing `test_logger` stderr-teardown crash and `test_health_runner` network-timeout tests.
**State left in:**
- `maestro-remote-control` committed locally (`cb1f1ab`), **not pushed / not merged**.
- Somatic: `maestro/` removed (working tree shows it deleted); vLLM config at 0.93; vLLM currently **stopped** (test left it down, VRAM released); WhisperX server process was running on CPU.
- Native ChromaDB/MCP/Neo4j on Hippocampal untouched (live test used a scoped script, never `maestro shutdown`).
**Files changed:**
- NEW: `maestro/remote.py`, `maestro/tests/test_remote.py`
- `maestro/{config,cli,orchestrator,shutdown,__init__}.py`, `maestro/services/{__init__,vllm_service,whisperx_service}.py`, `maestro/tests/{test_orchestrator,test_shutdown}.py`
- Somatic-only (gitignored here): `config/models/vllm_config.yaml` 0.90→0.93
- SESSIONS.md: this entry
**Next session should:**
- Push `maestro-remote-control` and open a PR into `master` if approved.
- Optional: add per-service `maestro shutdown --service NAME` for scoped remote stops; wire WhisperX health tests (deferred by Gines).
- Consider persisting the vLLM/WhisperX systemd user units as installed `.service` files (currently transient via `systemd-run`).
---

## Session: 2026-07-05 — Node: Hippocampal
**Goal:** Ship the maestro cross-node work: push, PR, merge, sync local, clean up.
**Completed:**
- Pushed `maestro-remote-control` to origin; opened **PR #3** (base `master`, +1116/−139), mergeable/CLEAN.
- Merged PR #3 via merge commit **`e4f11c3`** (mergedAt 2026-07-05 02:47 UTC) — maestro v0.13.0 cross-node control + the 2026-07-04(b) SESSIONS entry now on `master`.
- Fast-forwarded local `master` `724c8d8 → e4f11c3` (in sync with origin).
- Deleted branch `maestro-remote-control` (local `git branch -d`, merge-verified, was `4535134`; and `origin`). No such branches remain.
**State left in:**
- `master` = `e4f11c3` on both local and origin; working tree clean; on `master`.
- Somatic still has `maestro/` removed and vLLM config at 0.93 (vLLM stopped, VRAM released).
**Files changed:**
- SESSIONS.md: this entry (no code changes this session — release/merge only).
**Next session should:**
- Optional follow-ups still open from 2026-07-04(b): per-service `maestro shutdown --service NAME`; WhisperX health tests; persist the systemd user units as installed `.service` files.
- Still pending from earlier: retire `ubik-memory-sweep`; update the ingestion loader (new fields + `EPISODIC` token).
---

## Session: 2026-07-05 (b) — Node: Hippocampal
**Goal:** Web control panel for UBIK (all Maestro commands in the browser + Neo4j link, styled after the PKD Ubik book); make it persistent with launchd.
**Completed (maestro v0.14.0; PR #4 merged → `20875a9`):**
- New `maestro/web/` package: FastAPI backend `server.py` reusing the Maestro Python API (Orchestrator, ShutdownController, registry, metrics, health). Endpoints `/api/{config,status,health,metrics,logs,start,shutdown}`; destructive ops require `confirm=true`.
- Static SPA (`index.html`/`style.css`/`app.js`/`ubik-logo.svg`): live auto-refreshing cross-node service grid; Start (all / per-service / local-only); Shutdown (dry-run / local-only / emergency, confirm modal); Health/Metrics/Logs to a console; links to Neo4j browser (`:7474`) + ChromaDB/MCP/vLLM/API-docs.
- Logo is **original** (aerosol can + wordmark — not the copyrighted cover; swap `static/ubik-logo.svg`). Retro-commercial theme with rotating in-book Ubik ad epigraphs ("Safe when used as directed").
- New CLI `maestro web [--host --port]` (default `0.0.0.0:8090`); requirements += fastapi, uvicorn.
- Verified via screenshot + curl (all endpoints, static assets, live `/api/status`).
- **launchd persistence** `com.ubik.maestro-web` (mirrors `com.ubik.chromadb`; all artifacts local-disk, not in repo): plist (RunAtLoad+KeepAlive, logs → `~/Library/Logs/com.ubik.maestro-web.*`), wrapper `~/ubik-bin/start_maestro_web.sh` (volume-wait guard, sets PYTHONPATH + MAESTRO_LOG_DIR), FDA interpreter copy `~/ubik-bin/ubik-maestro-python`. Read+write of the external volume under launchd both work. KeepAlive auto-restart verified (killed pid → respawned → HTTP 200).
- **Bug fixed**: local `maestro/.env` had `MAESTRO_LOG_DIR=<spaces># comment` → pydantic-settings took the comment as the value (crashed under launchd cwd=/, silently made a garbage dir on manual runs). Cleaned the `.env` line, deleted the garbage dir, pinned `MAESTRO_LOG_DIR` in the wrapper. `.env.example` was already correct.
**State left in:**
- `master` = `20875a9` (local + origin, in sync); `maestro-web` branch deleted (local + remote).
- Web panel **live** at `http://100.103.242.91:8090` via launchd (persists across reboots / login).
**Files changed:**
- NEW: `maestro/web/{__init__,server}.py`, `maestro/web/static/{index.html,style.css,app.js,ubik-logo.svg}`
- `maestro/cli.py` (`web` command), `maestro/requirements.txt` (fastapi/uvicorn), `maestro/__init__.py` (v0.14.0)
- Local-disk only (not repo): `~/ubik-bin/{start_maestro_web.sh,ubik-maestro-python}`, `~/Library/LaunchAgents/com.ubik.maestro-web.plist`, `maestro/.env` (gitignored) log-dir fix
- SESSIONS.md: this entry
**Next session should:**
- Optional: swap in a licensed Ubik cover image at `static/ubik-logo.svg`; add background-job progress for long start/shutdown ops; auth in front of the panel if exposed beyond the tailnet.
- Carried over: per-service `maestro shutdown --service NAME`; WhisperX health tests; persist vLLM/WhisperX systemd user units as `.service` files; retire `ubik-memory-sweep`; update ingestion loader (new fields + `EPISODIC` token).
---

## Session: 2026-07-05 (c) — Node: Hippocampal
**Goal:** Dependency audit across all UBIK components; evict non-UBIK finance research projects; fix maestro requirement floors.
**Completed:**
- Full dependency audit across all 16 manifest files in the UBIK tree: surfaced chromadb v0→v1 floor mismatch, neo4j v5→v6 floor mismatch, Python version fragmentation (3.10/3.11/3.12/3.13 across projects), FinRobot entirely stale (langchain 0.1.20, aiohttp 3.8.5, pdfkit 1.0.0 unmaintained), pyautogen archived, Python 3.10 EOL October 2026.
- Identified 4 directories that are external finance-research projects and do not belong in UBIK core: `FinRobot/`, `TradingAgents/`, `aihedgefund/`, `my-project/` (a TradingAgents duplicate).
- Moved all four to `/Volumes/990PRO 4T/FinanceAI/` (26 MB + 1.5 GB + 5 MB + 684 MB). UBIK root is now clean.
- Created `/Volumes/990PRO 4T/FinanceAI/venv/` (Python 3.12) with TradingAgents/aihedgefund deps installed (langchain-core 1.4.8, langgraph 1.2.7, chromadb 1.5.9). FinRobot kept isolated from the shared venv — its pinned versions conflict; requires a requirements refresh before it can be installed cleanly.
- Wrote `FinanceAI/README.md` documenting the venv, per-project notes, and the FinRobot isolation warning.
- Bumped `maestro/requirements.txt` floors: `chromadb>=0.5.0` → `>=1.4.0`; `neo4j>=5.20.0` → `>=6.1.0` (both now match installed venv versions). Committed `a3359cf` and pushed to `master`.
**State left in:**
- `master` = `a3359cf` on both local and origin; working tree clean.
- FinanceAI projects at `/Volumes/990PRO 4T/FinanceAI/`; their TradingAgents embedded venv (Python 3.10) is still present inside `TradingAgents/` — usable but EOL in October 2026.
**Files changed:**
- `maestro/requirements.txt`: chromadb and neo4j floor bumps
- SESSIONS.md: this entry
**Next session should:**
- Upgrade TradingAgents Python from 3.10 → 3.12 before October 2026 EOL (update `.python-version` and recreate the embedded venv).
- Refresh FinRobot's requirements.txt (drop pinned stale versions; langchain 0.1.20 → 1.x, aiohttp 3.8.5 → 3.11+, drop pdfkit/pyautogen/unstructured 0.8.1).
- Run `pip-audit` against the active DeepSeek venv to surface any CVE-flagged packages.
- Carried over: per-service `maestro shutdown --service NAME`; WhisperX health tests; persist vLLM/WhisperX systemd user units; retire `ubik-memory-sweep`; update ingestion loader (new fields + `EPISODIC` token).
---

## Session: 2026-07-05 (d) — Node: Hippocampal
**Goal:** Security audit and CVE remediation across all UBIK Python environments (Hippocampal DeepSeek venv + Somatic pytorch_env / ubik/venv).
**Completed:**
- **Hippocampal DeepSeek venv** (`/Volumes/990PRO 4T/DeepSeek/venv/`, Python 3.13.7): installed pip-audit; found 167 vulnerabilities across 40 packages. Upgraded all fixable packages in two passes (35-package main pass + torch/transformers separately). Resolved post-upgrade conflicts: fastapi 0.128→0.139 (starlette 1.x compatibility — critical for maestro web panel), protobuf pinned back to 6.33.6 (avoid breaking Google API packages), marshmallow/sentence-transformers/langchain-huggingface upgraded. Restarted `com.ubik.maestro-web` launchd agent to pick up new starlette/fastapi; confirmed HTTP 200. Final state: **4 CVEs remaining** (chromadb, diskcache, lupa, nltk — all have no upstream fix yet).
- **Somatic discovery**: SSH'd via `ssh -o BatchMode… windows-server "wsl bash -s"`. Found two envs: `~/pytorch_env` (primary, has vLLM/torch/ray) and `~/ubik/venv` (built with `include-system-site-packages=true` on top of pytorch_env — effectively the same package set). No conda. Both Python 3.12.3.
- **Somatic audit**: both envs identical — ~110+ CVEs. Key findings: vLLM 0.13.0 (15 CVEs, needs 0.22.1), torch 2.9.0 (4 CVEs), ray 2.54.0 (1 CVE), plus the same aiohttp/starlette/cryptography/jupyter stack issues as Hippocampal.
- **Somatic step 1 upgrades** (all packages except vLLM/torch/ray): upgraded both envs in parallel. Fixed post-upgrade conflicts: fastapi upgraded for starlette 1.x; protobuf pinned to 6.33.6; datasets/prometheus-fastapi-instrumentator upgraded. Final state: **27 CVEs remaining** — all confined to vllm 0.13.0 (15), torch 2.9.0 (4), ray 2.54.0 (1), chromadb/diskcache/nltk (3 no-fix). Step 2 (vLLM+torch+ray coordinated upgrade) deferred pending compatibility research.
**State left in:**
- Hippocampal venv: 4 CVEs (no fix available), all services healthy, web panel on new fastapi/starlette.
- Somatic both envs: 27 CVEs, all in vLLM/torch/ray — deferred to Step 2.
- `master` = `35f0990` unchanged (no code changes this session).
**Files changed:**
- SESSIONS.md: this entry (no repo code changes — venv upgrades are outside git)
**Next session should:**
- **Step 2 — vLLM upgrade** (Somatic): research vLLM 0.13.0→0.22.1 breaking changes for RTX 5090 + AWQ model; then upgrade `vllm + torch + ray` together and re-validate `vllm_server.py` startup with the current `vllm_config.yaml`. This will clear the remaining 16 fixable Somatic CVEs.
- Upgrade `ray 2.54.0` → 2.55.0 (can be done independently of vLLM if needed).
- Carried over: TradingAgents Python 3.10→3.12 before EOL; FinRobot requirements refresh; per-service `maestro shutdown --service NAME`; WhisperX health tests; persist systemd units; retire `ubik-memory-sweep`; update ingestion loader.
---

## Session: 2026-07-05 (e) — Node: Hippocampal
**Goal:** Upgrade ray on Somatic; plan the vLLM upgrade.
**Completed:**
- **Ray upgraded** on Somatic `pytorch_env`: 2.54.0 → **2.56.0** (pip picked the latest stable; ubik/venv inherits via include-system-site-packages). Clears CVE-2026-41486. Ray CVE is now resolved.
- **vLLM upgrade plan produced** (deferred to next dedicated session). Key findings from reading the actual `vllm_server.py` and `vllm_config.yaml`:
  - torch is **2.9.0+cu128** (CUDA 12.8) — already very new; the plan does NOT downgrade torch.
  - `vllm_config.yaml` requires **no changes** — all flags compatible with 0.22.x.
  - **Critical risk**: `vllm_server.py` lines 79–84 import `destroy_model_parallel` / `cleanup_dist_env_and_memory` from `vllm.distributed.parallel_state` — this is what frees VRAM on shutdown. The import path likely moved between 0.13 and 0.22. Silent `ImportError` fallback means VRAM leaks if not fixed. Must verify and patch before go-live.
  - **FA3 risk**: line 45 forces `VLLM_FLASH_ATTN_VERSION=2` (Blackwell workaround from 0.13 era). 0.22.x supports FA3 on Blackwell natively; workaround may now be a performance penalty — test both ways.
  - **CUDA wheel risk**: vLLM PyPI wheels are built for CUDA 12.4/12.6; torch is cu128. Should work at runtime but monitor for CUDA errors on first startup.
  - Rollback: `pip install vllm==0.13.0 xgrammar==0.1.27` (xgrammar must be re-pinned to 0.1.27 for 0.13.0).
**State left in:**
- Somatic `pytorch_env`: ray=2.56.0, vLLM still at 0.13.0 (upgrade deferred). Remaining CVEs: vLLM 0.13.0 (15 CVEs), torch 2.9.0 (4 CVEs, tied to vLLM upgrade), chromadb/diskcache/nltk (3 no-fix).
- `master` = `e690715` unchanged; no code changes this session.
**Files changed:**
- SESSIONS.md: this entry
**Next session should (vLLM upgrade — plan ready):**
  1. SSH to Somatic; freeze current state: `~/pytorch_env/bin/pip freeze > ~/pytorch_env_vllm_0.13_freeze.txt`
  2. `pip install vllm==0.22.0` (add `--extra-index-url https://download.pytorch.org/whl/cu124` if wheel mismatch)
  3. Verify cleanup API: `python3 -c "from vllm.distributed.parallel_state import destroy_model_parallel, cleanup_dist_env_and_memory; print('OK')"` — if it fails, find new path and patch `vllm_server.py` lines 79–84
  4. Test FA3: comment out `VLLM_FLASH_ATTN_VERSION=2` (line 45) and test; restore if Blackwell kernel fails
  5. Start server with `--skip-checks`, confirm it listens on :8002 and AWQ Marlin kernel loads
  6. Send SIGTERM; confirm VRAM drops to ~0.7 GB (WSLg only) — if not, cleanup API path is wrong
  7. Rollback if needed: `pip install vllm==0.13.0 xgrammar==0.1.27`
- Also carried over: TradingAgents Python 3.10→3.12; FinRobot requirements refresh; per-service `maestro shutdown --service NAME`; WhisperX health tests; persist systemd units; retire `ubik-memory-sweep`; update ingestion loader.
---

## Session: 2026-07-06 — Node: Hippocampal
**Goal:** Package version / deprecation-status assessment across both UBIK nodes (Hippocampal + Somatic), as a refresh of the 2026-07-05 CVE remediation work.
**Completed:**
- Dispatched parallel read-only audits of every dependency manifest/venv under UBIK on both nodes.
- **Somatic connectivity incident**: the `windows-server` SSH alias resolved to Tailscale IP `100.92.95.39` (`adrian-1`, identity `gasu04@`), which `tailscale status` showed offline ~22h — genuinely unreachable, not a credentials/syntax issue. Investigation of `tailscale status` revealed this tailnet has **multiple `adrian*`-named devices under two different identities** (`gasu04@` and `acefesan@`), some belonging to a different person's devices sharing the tailnet. Confirmed with the user that the correct current device is `100.75.228.46` (`adrian`, Windows, `acefesan@`); updated `~/.ssh/config` `windows-server` `HostName` to this IP and re-verified SSH + `wsl bash -s` non-interactive access works. The old IP `100.79.166.114` (previously logged as "Somatic") belongs to yet another device (`adrian-wsl`, `acefesan@`) and was never actually Somatic — that mapping in memory was stale.
- **Hippocampal audit**: shared DeepSeek venv (Python 3.13.7, serves `hippocampal/`/`maestro/`/`ingestion/`/`deepseek/`) still carries the same 4 no-fix-available CVEs from 2026-07-05 (`chromadb 1.4.1`, `diskcache 5.6.3`, `lupa 2.6`, `nltk 3.9.4`); `ubik-chromadb-venv` (native Chroma server) runs an older `chromadb 1.3.7`. `rag_env` confirmed **dead**: shebangs point at a moved/nonexistent path, nothing runs, zero references anywhere in the tree — same disposition as the FinRobot/TradingAgents eviction. Vendored third-party dirs skimmed: `evernote-sdk-python` is dead (Python-2-only, last commit 2024), pulled in transitively by `Cross-Platform-Workflow-Orchestrator/geeknote`. Confirmed `UBIKParallax-source-v6` is the user's real frontend (`github.com/gasu04/UBIKParallax`) with real pending majors (vite 7→8, typescript 5.6→6.0, express 4→5); `-v5` is a redundant 9-minutes-older duplicate.
- **Somatic audit — corrects a standing error in the 2026-07-05(d) log**: `~/ubik/venv` is a **symlink to `~/pytorch_env`**, not a second layered environment — there has only ever been ONE venv on Somatic. vLLM's CVE exposure **grew since 07-05**: 15→**18** distinct CVEs on the still-installed `vllm 0.13.0`; latest upstream is now **0.24.0** (the 07-05e plan's "0.22.x" target is stale). `torch` is installed at **2.9.0** even though `requirements-frozen.txt` already pins `2.9.1` — an env/manifest drift that trivially closes one CVE (`CVE-2025-2999`) once applied. `ray` confirmed holding at 2.56.0/0 CVEs (07-05e upgrade stuck). No `vllm`/`whisperx` process running during the audit; GPU idle (602 MiB/32607 MiB) — nothing was started, stopped, or modified.
**State left in:**
- Read-only audit only — no packages, services, or repo code changed on either node.
- `~/.ssh/config` `windows-server` `HostName` corrected to `100.75.228.46` (local machine config, not in git).
- Local Claude Code memory (`MEMORY.md`) updated with the corrected Tailscale device mapping and a note to always re-verify via `tailscale status` before trusting a cached IP for Somatic.
- All findings above (torch drift, vllm CVE growth, `rag_env` eviction, chromadb version misalignment across 3 envs, single-venv correction) are unactioned — captured here for the next session.
**Files changed:**
- SESSIONS.md: this entry (no repo code changes this session — audit + local SSH config only)
**Next session should:**
1. Bump Somatic `torch` 2.9.0→2.9.1 (matches the existing `requirements-frozen.txt` pin, closes CVE-2025-2999, no compatibility risk, independent of the vLLM decision).
2. Re-scope the vLLM upgrade plan from 0.22.x to **0.24.0** given the CVE growth, then execute the existing 7-step procedure (freeze state → install → verify `destroy_model_parallel`/`cleanup_dist_env_and_memory` import path → test FA3 → validate startup → confirm graceful VRAM release → rollback path if needed).
3. Evict `rag_env` on Hippocampal (dead, zero references, same treatment as FinRobot/TradingAgents).
4. Align `chromadb` versions across the three environments (Somatic 1.5.1, Hippocampal shared 1.4.1, `ubik-chromadb-venv` 1.3.7).
5. Correct the "two venvs on Somatic" claim wherever else it may be documented (e.g. `platform_detect.py` comments/memory referencing Somatic conda/pytorch_env as distinct from `ubik/venv`).
6. Carried over: TradingAgents Python 3.10→3.12 before Oct 2026 EOL; FinRobot requirements refresh; per-service `maestro shutdown --service NAME`; WhisperX health tests; persist systemd units as installed `.service` files; retire `ubik-memory-sweep`; update ingestion loader (new fields + `EPISODIC` token); consider evicting `evernote-sdk-python` and the redundant `UBIKParallax-source-v5`.
---

## Session: 2026-07-06 (b) — Node: Hippocampal
**Goal:** Act on two items from the same-day audit: evict the dead `rag_env` venv (Hippocampal) and close the free-win torch CVE on Somatic.
**Completed:**
- **`rag_env` evicted**: moved `/Volumes/990PRO 4T/UBIK/rag_env` (718 MB, untracked by git) to `~/.Trash/rag_env_evicted_20260706` rather than a hard delete, in case anything unexpected surfaces. Confirmed it's gone from the UBIK tree.
- **Somatic torch upgrade**: confirmed via SSH that nothing (`vllm`/`whisperx`) was running and the GPU was idle (602 MiB / 2%) before touching the shared env. Upgraded `~/pytorch_env` (== `~/ubik/venv`) from `torch 2.9.0`/`torchaudio 2.9.0`/`torchvision 0.24.0` to the versions already pinned in `requirements-frozen.txt`: `torch==2.9.1+cu128`, `torchaudio==2.9.1+cu128`, `torchvision==0.24.1+cu128` (via `pip install --extra-index-url https://download.pytorch.org/whl/cu128`; pulled in `triton 3.5.1` as a dependency, up from 3.5.0). Closes `CVE-2025-2999`.
- **Compatibility check surfaced by pip's resolver** (not by us): `vllm 0.13.0` hard-pins `torch==2.9.0` / `torchaudio==2.9.0` / `torchvision==0.24.0` exactly — the upgrade is technically a declared-dependency violation for vllm. Verified functional impact: `python -c "import torch"` → OK, `cuda available: True`; `python -c "import vllm"` → OK, reports `0.13.0`. Patch-level torch bumps are normally ABI-stable within a minor version (2.9.x), consistent with the clean import. **Not verified**: an actual vLLM server start / model load / inference call — that's a heavier operation with VRAM allocation and was out of scope for this pass. The pip resolver also flagged two **pre-existing** mismatches unrelated to this change (already present before today): `transformers 5.13.0` vs vllm's declared `<5,>=4.56.0`, and `xgrammar 0.2.3` vs vllm's declared `==0.1.27`.
**State left in:**
- `rag_env` no longer in the UBIK tree (recoverable from Trash until emptied).
- Somatic `pytorch_env`/`ubik/venv`: torch stack now matches `requirements-frozen.txt` exactly; `CVE-2025-2999` closed. The other 3 torch CVEs, all 18 vllm CVEs, and the 3 no-fix-available CVEs (chromadb/diskcache/nltk) are unchanged — this pass only touched the one free-win item.
- vLLM has NOT been started/tested end-to-end since the torch bump — first real use should be watched for import/runtime errors, even though the risk is assessed as low.
**Files changed:**
- SESSIONS.md: this entry (no repo code changes — venv eviction + remote pip upgrade, both outside git)
**Next session should:**
- Before/at next real Somatic inference use: confirm `vllm_server.py` starts cleanly and serves a request under torch 2.9.1; watch specifically for CUDA-extension ABI errors on first load. If it breaks, rollback is `pip install torch==2.9.0+cu128 torchaudio==2.9.0+cu128 torchvision==0.24.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128`.
- Everything else carried over from the 2026-07-06 audit entry above (vLLM upgrade re-scoped to 0.24.0, chromadb version alignment across 3 envs, single-venv doc correction, plus the older backlog).
---

## Session: 2026-07-06 (c) — Node: Hippocampal
**Goal:** Evict two more dead-weight items flagged by the same-day audit: `evernote-sdk-python` and the redundant `UBIKParallax-source-v5`.
**Completed:**
- Verified both untracked by the UBIK git repo and no cross-references to `evernote-sdk-python` elsewhere in the tree (its use is via the `evernote` PyPI package pulled in transitively by `Cross-Platform-Workflow-Orchestrator/geeknote` — a separate dependency, unaffected by removing this vendored SDK checkout).
- Confirmed `UBIKParallax-source-v5` and `-v6` are independent git clones of `github.com/gasu04/UBIKParallax.git`; `diff -qr` showed differences confined to `.git` internals, `.DS_Store`, and one extra file only in v6 (`scripts/clean-html.py`) — v6 is a strict superset, v5 has nothing unique.
- Moved both to Trash (not hard-deleted, for reversibility): `evernote-sdk-python` (2.7 MB) → `~/.Trash/evernote-sdk-python_evicted_20260706`; `UBIKParallax-source-v5` (1.5 MB) → `~/.Trash/UBIKParallax-source-v5_evicted_20260706`. Confirmed both gone from the UBIK tree.
**State left in:**
- UBIK tree is smaller by ~4.2 MB; no functional impact (both were vendored/duplicate, unreferenced).
- `UBIKParallax-source-v6` remains as the sole, current copy of the actual UBIK frontend.
**Files changed:**
- SESSIONS.md: this entry (no repo code changes — Trash moves only, both outside git)
**Next session should:**
- Carried over: vLLM upgrade re-scoped to 0.24.0 (+ live server verification after the torch 2.9.1 bump); chromadb version alignment across 3 envs; single-venv doc correction; TradingAgents Python 3.10→3.12; FinRobot requirements refresh; per-service `maestro shutdown --service NAME`; WhisperX health tests; persist systemd units; retire `ubik-memory-sweep`; update ingestion loader (new fields + `EPISODIC` token).
---

## Session: 2026-07-06 (d) — Node: Hippocampal
**Goal:** Evict two more items flagged by the same-day audit: `AutoGPT` and `Cross-Platform-Workflow-Orchestrator`.
**Completed:**
- Verified both untracked by the UBIK git repo before touching anything.
- **`AutoGPT`** (620 MB): confirmed its git remote is the upstream third-party repo (`Significant-Gravitas/AutoGPT`) — a pure vendored checkout, contains both the deprecated `classic/` tree and the current `autogpt_platform/`, neither imported or referenced by any UBIK code. Only loose reference found was the top-level `install.sh` (AutoGPT's own upstream setup/clone script, copied to the UBIK root) — now orphaned but harmless (would just re-clone AutoGPT if ever run); left in place, not evicted.
- **`Cross-Platform-Workflow-Orchestrator`** (456 MB): **not vendored third-party** — checked and found it's the user's own project, already pushed to its own GitHub repo (`gasu04/Cross-Platform-Workflow-Orchestrator`, one commit) via a `gh repo create` recorded in `.claude/settings.local.json`. Fully recoverable via `git clone` independent of the local Trash copy. Its bundled `geeknote` dependency on the (already-evicted) `evernote` SDK is moot now.
- Moved both to Trash (not hard-deleted): `~/.Trash/AutoGPT_evicted_20260706`, `~/.Trash/Cross-Platform-Workflow-Orchestrator_evicted_20260706`. Confirmed both gone from the UBIK tree.
**State left in:**
- UBIK tree is smaller by ~1.07 GB.
- `install.sh` at the UBIK root is now an orphaned AutoGPT setup script — flagged, not removed.
- `Cross-Platform-Workflow-Orchestrator` still exists as its own repo on GitHub under `gasu04` — the eviction here only removed the local co-located copy, not the project itself.
**Files changed:**
- SESSIONS.md: this entry (no repo code changes — Trash moves only, both outside git)
**Next session should:**
- Decide whether to remove the now-orphaned root `install.sh`.
- Carried over: vLLM upgrade re-scoped to 0.24.0 (+ live server verification after the torch 2.9.1 bump); chromadb version alignment across 3 envs; single-venv doc correction; TradingAgents Python 3.10→3.12; FinRobot requirements refresh; per-service `maestro shutdown --service NAME`; WhisperX health tests; persist systemd units; retire `ubik-memory-sweep`; update ingestion loader (new fields + `EPISODIC` token).
---

## Session: 2026-07-06 (e) — Node: Hippocampal
**Goal:** Clean up the last remnant from the AutoGPT eviction: the now-orphaned root `install.sh`.
**Completed:**
- Confirmed `install.sh` was untracked by the UBIK git repo, then moved it to `~/.Trash/install.sh_evicted_20260706` (not hard-deleted). It was AutoGPT's own upstream setup/clone script, harmless but pointless without `AutoGPT/` present.
**State left in:**
- UBIK root no longer has any AutoGPT-related files.
**Files changed:**
- SESSIONS.md: this entry (no repo code changes — Trash move only, outside git)
**Next session should:**
- Carried over: vLLM upgrade re-scoped to 0.24.0 (+ live server verification after the torch 2.9.1 bump); chromadb version alignment across 3 envs; single-venv doc correction; TradingAgents Python 3.10→3.12; FinRobot requirements refresh; per-service `maestro shutdown --service NAME`; WhisperX health tests; persist systemd units; retire `ubik-memory-sweep`; update ingestion loader (new fields + `EPISODIC` token).
---
