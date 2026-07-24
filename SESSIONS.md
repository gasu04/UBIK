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

## Session: 2026-07-06 (f) — Node: Hippocampal
**Goal:** Evict two more vendored checkouts flagged by the same-day audit: `nifi` and `automatisch`.
**Completed:**
- Verified both untracked by the UBIK git repo. Both have `gasu04`-owned GitHub remotes (`github.com/gasu04/nifi.git`, `github.com/gasu04/automatisch.git`) — likely forks, backed up independently of the local Trash copy.
- Checked for real cross-references: an initial substring grep for "nifi"/"automatisch" returned dozens of hits, but a word-boundary re-check showed those were false positives (the substring "nifi" appears inside words like "unified", "significant", "magnificent"). The only genuine word-boundary hits for "nifi" were two files in `.tmp.driveupload/` (Google Drive Desktop's own upload-staging cache) — Java source snippets that are remnants of nifi's own files mid-sync, not references from UBIK code. No real dependency on either project found anywhere.
- Moved both to Trash (not hard-deleted): `~/.Trash/nifi_evicted_20260706` (470 MB), `~/.Trash/automatisch_evicted_20260706` (34 MB). Confirmed both gone from the UBIK tree.
**State left in:**
- UBIK tree is smaller by ~504 MB.
- Both projects remain recoverable via `git clone` from their `gasu04` GitHub remotes even after Trash empties.
**Files changed:**
- SESSIONS.md: this entry (no repo code changes — Trash moves only, both outside git)
**Next session should:**
- Carried over: vLLM upgrade re-scoped to 0.24.0 (+ live server verification after the torch 2.9.1 bump); chromadb version alignment across 3 envs; single-venv doc correction; TradingAgents Python 3.10→3.12; FinRobot requirements refresh; per-service `maestro shutdown --service NAME`; WhisperX health tests; persist systemd units; retire `ubik-memory-sweep`; update ingestion loader (new fields + `EPISODIC` token).
---

## Session: 2026-07-06 (g) — Node: Hippocampal
**Goal:** Evict two more vendored checkouts flagged by the same-day audit: `claude-code` and `gastos-promed`.
**Completed:**
- Verified both untracked by the UBIK git repo. Both have `gasu04`-owned GitHub remotes (`github.com/gasu04/claude-code.git`, `github.com/gasu04/gastos-promed.git`) — recoverable via `git clone` independent of the local Trash copy.
- `claude-code`: a vendored checkout of `@anthropic-ai/claude-code` pinned at v1.0.83 (stale relative to current releases). The only word-boundary match for "claude-code" elsewhere in the tree was a table-of-contents heading in `Claudemdv2.md` ("Claude Code Session Management") referring to the tool itself, not this directory — not a real dependency.
- `gastos-promed`: a single personal script (`analyze_expenses.py`) with no manifest; no references found elsewhere.
- Moved both to Trash (not hard-deleted): `~/.Trash/claude-code_evicted_20260706` (229 MB), `~/.Trash/gastos-promed_evicted_20260706` (132 KB). Confirmed both gone from the UBIK tree.
**State left in:**
- UBIK tree is smaller by ~229 MB.
- Both projects remain recoverable via `git clone` from their `gasu04` GitHub remotes even after Trash empties.
**Files changed:**
- SESSIONS.md: this entry (no repo code changes — Trash moves only, both outside git)
**Next session should:**
- Carried over: vLLM upgrade re-scoped to 0.24.0 (+ live server verification after the torch 2.9.1 bump); chromadb version alignment across 3 envs; single-venv doc correction; TradingAgents Python 3.10→3.12; FinRobot requirements refresh; per-service `maestro shutdown --service NAME`; WhisperX health tests; persist systemd units; retire `ubik-memory-sweep`; update ingestion loader (new fields + `EPISODIC` token).
---

## Session: 2026-07-06 (h) — Node: Hippocampal
**Goal:** Evict the last three low-concern items flagged by the same-day audit: `helloworld`, `pythonProject`, `test-setup`.
**Completed:**
- Verified all three untracked by the UBIK git repo. Unlike the last several evictions, none of these three have their own git remote (no nested `.git` at all — plain scratch folders) — Trash is the only safety net here, appropriate given the content.
- Confirmed all trivial: `helloworld/main.py` is a PyCharm default template ("Hi, PyCharm"); `pythonProject/{helloworld,ProjectEuler,santiago}` are scratch/practice subfolders (10–12 MB each, no manifests); `test-setup/system_info.py` is a one-off script that prints platform info and pings httpbin.org. Zero word-boundary references to any of the three names found elsewhere in the tree.
- Moved all three to Trash (not hard-deleted): `~/.Trash/helloworld_evicted_20260706` (10 MB), `~/.Trash/pythonProject_evicted_20260706` (34 MB), `~/.Trash/test-setup_evicted_20260706` (4 KB). Confirmed all gone from the UBIK tree.
**State left in:**
- UBIK tree is smaller by ~44 MB.
- Remaining untracked items in the tree are now limited to UBIK's own legitimate artifacts (`chromadb_data/`, `backups/`, `Ingested_data/`, etc.), `open-notebook` (Docker-based, no local manifest), and the pre-existing unstaged `maestro/services/docker_service.py` / `maestro/tests/test_orchestrator.py` edits.
**Files changed:**
- SESSIONS.md: this entry (no repo code changes — Trash moves only, all outside git)
**Next session should:**
- Carried over: vLLM upgrade re-scoped to 0.24.0 (+ live server verification after the torch 2.9.1 bump); chromadb version alignment across 3 envs; single-venv doc correction; TradingAgents Python 3.10→3.12; FinRobot requirements refresh; per-service `maestro shutdown --service NAME`; WhisperX health tests; persist systemd units; retire `ubik-memory-sweep`; update ingestion loader (new fields + `EPISODIC` token).
---

## Session: 2026-07-06 (i) — Node: Hippocampal
**Goal:** Session wrap-up. Consolidate the day's pending backlog into one place for the next working session.
**Completed:**
- Committed and pushed one last loose end unrelated to the audit: `maestro/services/docker_service.py`'s `DockerService` default `max_wait_s` 60.0→180.0 (Docker Desktop can take longer than 60s to report healthy on cold start), with the matching `test_orchestrator.py` assertion update. Verified `TestMaxWaitSDefaults` passes before committing (`85d1d2e`).
- No new investigation this session — just closing out the day's pending list below so nothing from today's audit/eviction work gets lost before the next session picks it up.
**State left in:**
- `master` = `85d1d2e` on both local and origin; working tree clean except legitimate untracked items (`chromadb_data/`, `backups/`, `Ingested_data/`, `UBIKParallax-source-v6/`, `open-notebook/`, etc.).
- Somatic: torch stack at 2.9.1+cu128 (matches `requirements-frozen.txt`), `vllm` still 0.13.0 (import-verified OK post-torch-bump, but no live server/inference test run), `ray` at 2.56.0. No `vllm`/`whisperx` process running; GPU idle.
- Hippocampal UBIK tree substantially decluttered today: evicted `rag_env`, `evernote-sdk-python`, `UBIKParallax-source-v5`, `AutoGPT`, `Cross-Platform-Workflow-Orchestrator`, `nifi`, `automatisch`, `claude-code`, `gastos-promed`, `helloworld`, `pythonProject`, `test-setup`, and the orphaned root `install.sh` — all moved to dated folders under `~/.Trash/`, none hard-deleted.
**Files changed:**
- SESSIONS.md: this entry
**Pending — to work on next session (nothing actioned yet on any of these):**
1. **vLLM upgrade (Somatic)** — re-scoped target **0.24.0** (was 0.22.x; CVE count grew 15→18 while the plan sat unexecuted). Execute the existing 7-step procedure: freeze state → install → verify `destroy_model_parallel`/`cleanup_dist_env_and_memory` import path (version-sensitive, critical for VRAM release on shutdown) → test FA3 → validate startup → confirm graceful VRAM release → rollback path (`pip install vllm==0.13.0 xgrammar==0.1.27`) if needed.
2. **Live-verify vLLM under torch 2.9.1** — import checks passed today, but no actual server start / model load / inference call has been run since the torch bump. Rollback if it breaks: `pip install torch==2.9.0+cu128 torchaudio==2.9.0+cu128 torchvision==0.24.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128`.
3. **Align chromadb versions** across the three environments: Somatic `pytorch_env` (1.5.1), Hippocampal shared DeepSeek venv (1.4.1), `ubik-chromadb-venv` (1.3.7).
4. **Correct the "two venvs on Somatic" documentation** — confirmed today `~/ubik/venv` is a symlink to `~/pytorch_env`, not a separate layered environment (contradicts the 2026-07-05(d) log and `platform_detect.py` comments referencing Somatic conda/pytorch_env as distinct from `ubik/venv`).
5. **No-fix CVEs, monitor only** (no upstream fix exists as of 2026-07-06): Hippocampal `chromadb`/`diskcache`/`lupa`/`nltk`; Somatic `chromadb`/`diskcache`/`nltk`.
6. **Older backlog, still open**: TradingAgents Python 3.10→3.12 before Oct 2026 EOL; FinRobot requirements refresh (stale langchain/aiohttp/pdfkit); per-service `maestro shutdown --service NAME`; WhisperX health tests; persist vLLM/WhisperX systemd user units as installed `.service` files; retire the `ubik-memory-sweep` skill; update the ingestion loader (new fields + `EPISODIC` token).
7. **Lower priority, not yet decided**: `UBIKParallax-source-v6` major-version bumps (Vite 7→8, TypeScript 5.6→6.0, Express 4→5, Vitest 2→4, Recharts 2→3); `pandas 2.3→3.0` on the Hippocampal shared venv (breaking-change risk, needs testing before committing to it).
---

## Session: 2026-07-09 — Node: Hippocampal
**Goal:** Diagnose and fix the SESSIONS.md → Google Drive sync (claude.ai reported reading a stale copy ending at 2026-07-05(e)), and fix a stale Somatic IP found along the way.
**Completed:**
- **Root-caused the Drive sync break**: the `gsanchezurrutia@gmail.com` Google Drive Desktop account hit a recurring OAuth token-refresh failure (`DEADLINE_EXCEEDED`, logged repeatedly from 2026-07-06 23:20 through 07-07 03:01) shortly after our last sync that evening, which meant none of that day's 9 commits' worth of SESSIONS.md revisions reached Google's cloud — confirmed independently by claude.ai reading the cloud-side `modifiedTime` via the Drive API (frozen at 2026-07-05T04:37:38Z).
- By 07-09, the account had recovered and Drive Desktop ran a large catch-up resync (thousands of queued changes). That resync's own conflict-resolution **renamed the target folder** (same underlying item-id `1VrGIgeTmjL-8vFMERV7w9DS_w3Mnv1a1`) from `Ubik` to **`Ubik_drive`** — silently breaking `scripts/sync_sessions.sh`'s hardcoded path (its own directory-exists guard made it no-op quietly rather than error).
- Also surfaced, not yet acted on: the same resync appears to have created `(1)`-suffix duplicates of dozens of unrelated top-level folders across the whole Drive account (`Promed`, `Correspondencia`, `23andme DNA data`, etc.) — a wider issue flagged to the user, out of scope for this fix.
- Chased a red herring: the user pointed at a Drive folder link (`1e3Y_dwXVHNtzm15xRooX4mSqHyyU8Brs`) that turned out to be `Computers > Mini M1 2022 > UBIK` — Google Drive's legacy Backup-and-Sync "Computers" section for a different, retired machine, not locally writable/mounted on this Mac under either Google account. Not used as the sync target.
- **Fix applied**: repointed `scripts/sync_sessions.sh`'s `DST` to `.../My Drive/Ubik_drive/SESSIONS.md` (commit `8f1ba90`). Ran it manually and verified byte-identical to the canonical file. claude.ai independently re-read the same file/folder (file ID `107eFrHa1HBTEU0Eh3hDWN8ipAL4tki3g`, parent `1VrGIgeTmjL...`) and confirmed it now sees the 07-06(i) wrap-up correctly, updating its own memory to match.
- **Side fix, flagged by claude.ai while re-reading the log**: `ingestion/.env` and `.env.example` had `SOMATIC_HOST=100.79.166.114` — the same stale/wrong Tailscale IP identified on 2026-07-06 (belongs to `adrian-wsl` under a different tailnet identity, never actually Somatic). Corrected both to `100.75.228.46` (commit `1a0a74b`). Left the separate, older CP2 blocker alone per user's explicit choice: `ingestion/.env` still has no `SOMATIC_VLLM_URL`/`SOMATIC_TAILSCALE_IP`/`VLLM_PORT`/`UBIK_ENRICHMENT_MODEL` — that gap has been open since 2026-06-20 and needs a model-name decision from the user, not just an IP fix.
**State left in:**
- `master` = `1a0a74b` on both local and origin; working tree clean except legitimate untracked items.
- SESSIONS.md → Drive sync confirmed working end-to-end (local write → repo commit → post-commit hook → `Ubik_drive` copy → claude.ai read), verified from both the Hippocampal side and claude.ai's independent Drive API read.
- The account-wide `(1)`-duplicate-folder issue on Drive is unresolved and un-investigated beyond the initial discovery.
- CP2/Phase 3 enrichment still blocked on Somatic vLLM endpoint config in `ingestion/.env` (unchanged by this session, by choice).
**Files changed:**
- `scripts/sync_sessions.sh`: `DST` path `Ubik` → `Ubik_drive` (`8f1ba90`)
- `ingestion/.env.example`: `SOMATIC_HOST` stale IP fix (`1a0a74b`); local `ingestion/.env` updated to match (gitignored, not committed)
- SESSIONS.md: this entry
**Next session should:**
- Investigate the wider Drive `(1)`-duplicate-folder issue (dozens of top-level folders affected) — separate from anything UBIK-code-related, but worth a look via Drive's web UI before more content is added anywhere in that account.
- Decide on the CP2 enrichment endpoint config (`SOMATIC_VLLM_URL`/`SOMATIC_TAILSCALE_IP`=`100.75.228.46`/`VLLM_PORT=8002`/`UBIK_ENRICHMENT_MODEL`=?) to unblock Phase 3 enrichment, dormant since 2026-06-20.
- Everything else carried over from the 2026-07-06(i) pending list above (vLLM 0.24.0 upgrade, live vLLM verification under torch 2.9.1, chromadb version alignment, single-venv doc correction, no-fix CVEs, older backlog).
---

## Session: 2026-07-11 — Node: Hippocampal
**Goal:** Investigate the Drive `(1)`-duplicate-folder issue flagged on 2026-07-09, then set the CP2 enrichment model/endpoint to unblock Phase 3 enrichment.
**Completed:**
- **Correction to the 2026-07-09 entry**: that entry wrongly attributed the Drive `(1)`-duplicate folders to "the same resync" that renamed `Ubik`→`Ubik_drive`. Timestamp evidence disproves this — none of the 59 duplicated top-level folders in `gsanchezurrutia@gmail.com`'s "My Drive" have a July 2026 mtime; they range from 2012 to 2026-05-28, long before this month's OAuth incident. It's a separate, long-standing client-side issue (likely from Google's Backup-and-Sync → Drive-for-Desktop transition years ago), affects only that one account (the `gsanchez@promed-sa.com` account has just 1 such pair), and is unrelated to UBIK. Sampled 8 pairs: some are exact mirrors (identical item counts), others have genuinely diverged content (different item counts) — not safe to bulk-delete either side. Flagged to the user as personal Drive hygiene, out of scope for UBIK development; no further action taken.
- **CP2 enrichment config, unblocked** (dormant since 2026-06-20): determined the actual model being served — `vllm_server.py` never sets `--served-model-name`, so the model ID vLLM reports is exactly its `--model` argument, which resolves (via `os.path.expanduser`) to `/home/gasu/ubik/models/deepseek-awq/DeepSeek-R1-Distill-Qwen-14B-AWQ` (confirmed this path exists on Somatic). Set `ingestion/.env` and `.env.example`: `SOMATIC_TAILSCALE_IP=100.75.228.46`, `VLLM_PORT=8002`, `UBIK_ENRICHMENT_MODEL=/home/gasu/ubik/models/deepseek-awq/DeepSeek-R1-Distill-Qwen-14B-AWQ` (commit `15730e9`).
- **Real bug found and fixed while wiring this up**: `_load_env_file` (`config/ingestion_config.py`) uses `os.environ.setdefault`, so a key present in `.env` with an *empty* value still wins over the intended composed fallback (`os.environ.get(key, default)` treats present-but-blank as set, not absent) — silently breaking the documented `SOMATIC_VLLM_URL > SOMATIC_TAILSCALE_IP > SOMATIC_LAN_IP > hostname` priority chain. Fixed by commenting out (not blanking) the unused optional vars (`SOMATIC_VLLM_URL`, `SOMATIC_LAN_IP`, `SOMATIC_VLLM_LAN_URL`, `UBIK_SENSITIVE_LLM_URL`, `UBIK_STANDARD_LLM_URL`) in both `.env` and `.env.example`, with an explanatory comment so it isn't reintroduced. Verified via `EndpointConfig.from_env()` (the app's real code path): `sensitive_endpoint`/`standard_endpoint` now correctly resolve to `http://100.75.228.46:8002/v1`, `model` resolves correctly, and `for_tier()` returns the right URL for both therapy and business tiers.
**State left in:**
- `master` = `15730e9` on both local and origin; working tree clean except legitimate untracked items.
- CP2's config blocker is resolved, but **CP2 has not actually been run** — vLLM is not currently running on Somatic (no process, no systemd unit), so `run_phase3.py --stage 2 --dry-run --limit 8` would still fail on connection, not config. Starting vLLM + running the dry-run is a separate next step, not done this session.
- Drive `(1)`-duplicate-folder issue remains uninvestigated further beyond characterization; no cleanup attempted (personal data, needs user's own per-pair judgment).
**Files changed:**
- `ingestion/.env.example`: enrichment endpoint/model filled in + blank-var fallback bug fixed (`15730e9`); local `ingestion/.env` updated to match (gitignored, not committed)
- SESSIONS.md: this entry
**Next session should:**
- Start vLLM on Somatic and run `run_phase3.py --stage 2 --dry-run --limit 8`, hand-score per `qa/rubric.md` (gate: confidence ≥0.7 on ≥6/8) — the original CP2 gate from 2026-06-20, now finally unblocked config-wise.
- Everything else carried over from the 2026-07-06(i)/07-09 pending lists (vLLM 0.24.0 upgrade, live vLLM verification under torch 2.9.1, chromadb version alignment, single-venv doc correction, no-fix CVEs, older backlog).
---

## Session: 2026-07-11 (b) — Node: Hippocampal
**Goal:** Actually start vLLM on Somatic and run the CP2 dry-run now that config is unblocked; fix whatever breaks along the way.
**Completed:**
- **WSL idle-shutdown fixed**: first two `maestro start --service vllm` attempts got the unit killed ~15-20s after start. Root-caused to WSL2's default VM idle-timeout tearing down the *entire* VM (not just the vllm unit) between short-lived SSH sessions — the systemd `--user` manager and its lingering session don't survive a full VM shutdown, only a login-session logout. Fixed by adding `vmIdleTimeout=-1` to `/mnt/c/Users/gasu.Adrian/.wslconfig` (backed up original first) and forcing a `wsl --shutdown` via the Windows-shell SSH path (`RemoteCommand=none` override) to apply it.
- **Separate, transient WSL bug hit on the next attempt**: vLLM still got killed ~20s in, but this time the *whole system* did a clean `poweroff.target` shutdown (not an idle-timeout — happened only 2 min after the fresh restart). Journal showed the actual trigger: `WSL (427) ERROR: CheckConnection: getaddrinfo() failed: -5` / `Operation canceled @p9io.cpp:258 (AcceptAsync)` immediately followed by `systemd-logind: The system will power off now!` — WSL2's own internal Windows↔VM health-check (p9io/Plan9 transport) lost its connection and the VM shut itself down in response. This is a Windows-host/WSL-platform issue, not fixable from inside the VM. Retried once more and it worked cleanly — 3rd attempt succeeded, so this specific failure mode appears to be intermittent (tied to boot-time network/DNS settling?), not fixed, just not hit again this session. Flagged for awareness, not resolved.
- **vLLM confirmed healthy**: `/health` → 200, `/v1/models` reports exactly the configured `UBIK_ENRICHMENT_MODEL` path, GPU shows 31.2 GB/32.6 GB loaded, systemd unit `active`, VM uptime stayed stable afterward (24+ min, no further unplanned restarts).
- **MCP server found down, root-caused, and fixed** (a separate, pre-existing break unrelated to today's other work): `mcp_server.py` was crash-failing on `from fastmcp import FastMCP` with `ImportError: cannot import name 'FastMCP' from 'fastmcp' (unknown location)`. Diagnosed as a genuinely corrupted install, not an API change: `fastmcp` 3.4.2 is a split package (`fastmcp` + `fastmcp-slim`), and `fastmcp-slim`'s `__init__.py` — which the package's own install record claims should exist — was physically missing from `site-packages/fastmcp/` (only a stale `__pycache__` entry remained), so Python fell back to treating it as an empty PEP 420 namespace package. Fixed with `pip install --force-reinstall --no-deps fastmcp==3.4.2 fastmcp-slim==3.4.2` (restored the missing file; no code changes needed — `mcp_server.py`'s import was always correct). Restarted via `run_mcp.sh restart`: listening on :8080, clean startup log, `GET /mcp` → 406 (the established correct/healthy response for that endpoint).
- **CP2 dry-run itself was not run this session** — got to the point of having both vLLM and MCP healthy, but the actual `run_phase3.py --stage 2 --dry-run --limit 8` invocation was interrupted by the user twice and the session moved to wrap-up before it was retried.
**State left in:**
- vLLM running on Somatic (systemd unit `ubik-vllm`, healthy, model loaded).
- MCP server running on Hippocampal (PID from this session, healthy, port 8080).
- No repo code changes this session — both fixes were environment-level (`.wslconfig` on the Windows host, `pip install --force-reinstall` in the shared DeepSeek venv).
- `.wslconfig` backup left at `/mnt/c/Users/gasu.Adrian/.wslconfig.bak.<timestamp>` on Somatic.
- CP2 dry-run still not actually executed — config and infra are both ready, just needs the command run.
**Files changed:**
- None in the repo. Environment only: Somatic `.wslconfig` (`vmIdleTimeout=-1`), Hippocampal DeepSeek venv (`fastmcp`/`fastmcp-slim` reinstalled).
- SESSIONS.md: this entry
**Next session should:**
- Run `run_phase3.py --stage 2 --dry-run --limit 8` (both blockers now cleared) and hand-score per `qa/rubric.md` (gate: confidence ≥0.7 on ≥6/8).
- Keep an eye out for the intermittent WSL `p9io`/`CheckConnection` poweroff-on-boot bug recurring — if it becomes frequent, needs Windows-side investigation (Event Viewer, WSL version, network adapter behavior around boot), not just a retry.
- Consider whether `run_mcp.sh`/health-check tooling should proactively detect the "process running but import-crashed" pattern (stale PID file, no port listener) rather than relying on manual log inspection.
- Everything else carried over from the 2026-07-06(i)/07-09/07-11(a) pending lists (vLLM 0.24.0 upgrade, live vLLM verification under torch 2.9.1, chromadb version alignment, single-venv doc correction, no-fix CVEs, older backlog).
---

## Session: 2026-07-13 — Node: Hippocampal
**Goal:** Run the actual CP2 dry-run now that vLLM/MCP were confirmed healthy in 2026-07-11(b). Discovered CP2 was never really reachable — found and started fixing a deeper Somatic networking problem instead.
**Completed:**
- **Ran CP2 for real — all 8 files quarantined**: `run_phase3.py --stage 2 --dry-run --limit 8` failed every file with `All enrichment endpoints failed: transport error: All connection attempts failed` against `http://100.75.228.46:8002`, despite vLLM's own `/health` (checked from inside WSL) returning 200 moments earlier.
- **Root-caused: WSL2 NAT networking never exposed vLLM externally.** `netstat` on the Windows host showed vLLM's port only bound to `127.0.0.1`/`[::1]:8002` via `wslrelay.exe` (WSL's `localhostForwarding` mechanism) — which only forwards Windows' own loopback, never external interfaces like the Tailscale IP. No `netsh interface portproxy` rule existed to bridge that gap. (A prior successful external test from 2026-07-04(b) most likely relied on a portproxy rule that has since gone stale — WSL's internal IP changes on every restart, and we forced several restarts this week.)
- **Fixed by switching to mirrored networking mode**: added `networkingMode=mirrored` to `/mnt/c/Users/gasu.Adrian/.wslconfig` (backed up first) and `wsl --shutdown` to apply. Confirmed working: WSL's `eth2` interface now shows `100.75.228.46` directly — the same Tailscale IP as the Windows host, no relay needed.
- **Hit a follow-on blocker from the switch**: restarting vLLM under the new mode got silently skipped by maestro's own "already running" guard (`curl ... && echo ALREADY_RUNNING_UNMANAGED`) — something was already answering (with 404, not connection-refused) on port 8002. Traced to an **orphaned `wslrelay.exe` (PID 4204) still running on the Windows host from before the mode switch**, still squatting on `127.0.0.1:8002`. Under mirrored mode, WSL shares Windows' network namespace entirely, so this Windows-side leftover blocks the port for the Linux side too.
- **Could not kill it via SSH**: `taskkill /PID 4204 /F` → `Access denied` (wslrelay.exe runs under the WSL platform service, not the SSH user's privilege level). Needs a manual kill on the physical machine (Task Manager or admin PowerShell) — handed off to the user, not resolved this session.
- **Side effect discovered**: a manual `systemd-run` test (run to see raw output past maestro's early-exit guard) got far enough to load ~20 GB into GPU memory before crashing on the port-bind conflict, **without running vllm_server.py's graceful CUDA cleanup** — `nvidia-smi` shows 20.3 GB used with **zero** owning processes in `--query-compute-apps` (a real VRAM leak, not just a stopped service). Plan: one more `wsl --shutdown` after the port is freed should clear this, since it fully reinitializes WSL's GPU passthrough.
**State left in:**
- vLLM is NOT running. Port 8002 on Windows is stuck held by orphaned `wslrelay.exe` (PID 4204) pending manual user kill.
- GPU has ~20 GB leaked/orphaned VRAM, expected to clear on the next `wsl --shutdown` (not yet done — waiting on the port fix first).
- `.wslconfig` now has both `vmIdleTimeout=-1` (from 07-11) and `networkingMode=mirrored` (new today), with a fresh timestamped backup taken before this edit.
- CP2 dry-run has still never successfully run — now blocked on infrastructure (port + VRAM), not config; the enrichment config itself (from 2026-07-11(a)) is confirmed correct.
- MCP server (fixed 07-11(b)) was not re-checked this session; assume unaffected since no work touched Hippocampal.
**Files changed:**
- None in the repo — all changes were Windows-host (`.wslconfig`) and runtime state on Somatic.
- SESSIONS.md: this entry
**Next session should:**
1. Confirm the user killed the orphaned `wslrelay.exe`, then run one more `wsl --shutdown` + restart to both keep the port clear and reclaim the leaked ~20 GB VRAM.
2. Start vLLM, verify `curl http://100.75.228.46:8002/health` succeeds **directly from Hippocampal** (not just from inside WSL) before declaring it fixed — that's the actual gap that's bitten us twice now.
3. Once confirmed reachable, run `run_phase3.py --stage 2 --dry-run --limit 8` again and hand-score per `qa/rubric.md` (gate: confidence ≥0.7 on ≥6/8).
4. Watch for whether mirrored networking mode reintroduces the intermittent `p9io`/`CheckConnection` poweroff bug from 07-11(b), or resolves it (mirrored mode changes the network stack significantly, could go either way).
5. Everything else carried over from the 2026-07-06(i)/07-09/07-11(a) pending lists (vLLM 0.24.0 upgrade, live vLLM verification under torch 2.9.1, chromadb version alignment, single-venv doc correction, no-fix CVEs, older backlog).
---

## Session: 2026-07-15 — Node: Somatic (Adrian / WSL2 Ubuntu)
**Goal:** Fix vLLM failing to start via `maestro start` ("vllm did not become healthy within 300s"). Working from Hippocampal over SSH into the Somatic WSL box (`ssh windows-server` → `wsl`, driven non-interactively via `ssh -o RemoteCommand=none ... 'wsl -e bash -l'` and base64-encoded PowerShell for Windows-side calls).

> **CORRECTION TO THE 2026-07-13 ENTRY ABOVE:** that entry's diagnosis ("orphaned `wslrelay.exe` squatting on 8002 + ~20 GB VRAM leak blocking vLLM") was **wrong**. The real root cause is a **port collision on 8002**, found and fixed this session (see below). The "VRAM leak" framing is refuted — vLLM died in ~1s and never touched the GPU. This entry supersedes the 2026-07-13 root-cause claim; the 2026-07-13 entry is retained as the honest record of what was believed at the time.

**ROOT CAUSE FOUND (this was a port collision, NOT VRAM/model/GPU):**
- vLLM's real crash, captured by launching `python -m vllm.entrypoints.openai.api_server` manually with the config args: **`OSError: [Errno 98] Address already in use` on port 8002.** It died in ~1s, never touched the GPU — which is why every `maestro start` hit the 300s health timeout.
- Port 8002 was held by a stray **nginx/1.29.5 serving a "Rotating Cube" demo page** — this is the **"unrelated nginx site"** already documented in SESSIONS.md 2026-06-12 (*"100.79.166.114:8002 is an unrelated nginx site"*). It is NOT part of UBIK.
- **The trigger:** `~/.wslconfig` (on Windows, `C:\Users\gasu.Adrian\.wslconfig`) had `networkingMode=mirrored` (added ~2026-07-11, same edit as `vmIdleTimeout=-1`). Mirrored mode collapses WSL + Windows onto one shared localhost, so vLLM-in-WSL:8002 and the pre-existing Windows/stray nginx:8002 — which coexisted fine under NAT (different IPs) back in June — now collided on a single shared socket.
- Confirmed the 8002 holder is a **leaked/zombie hvsocket binding**: `netstat` attributed it to "wslrelay" PID 9512, but that PID had NO path, NO start time, and NO CIM record, and it **survived `wsl --shutdown`** (twice) — so it is owned by the Windows HNS layer, not the WSL VM. Non-admin `gasu` cannot restart HNS; only admin/reboot clears the stuck socket itself.

**FIX APPLIED:**
- Reverted `~/.wslconfig`: `networkingMode=mirrored` → `networkingMode=NAT` (backup: `C:\Users\gasu.Adrian\.wslconfig.bak-nat-fix-20260715-190055`). This restores June's topology: WSL gets its own IP (`172.27.177.90/20` confirmed), so vLLM binds 8002 in WSL's OWN namespace, independent of the stuck Windows-side :8002 zombie.
- **VERIFIED vLLM now starts cleanly under NAT:** manual launch got all the way through model load, GPU alloc, KV cache, CUDA-graph capture, to `Starting vLLM API server 0 on http://0.0.0.0:8002` → `Application startup complete` → `Started server process`. No more Errno 98. (It only stopped because MY 90s test `timeout` killed it — not a vLLM fault. GPU was clean ~31GB free; model files present; vllm 0.13.0 / torch 2.9.1+cu128 / RTX 5090 sm_120 all fine.)

**STILL BROKEN — the remaining problem for next session:**
- **`maestro start` STILL fails vLLM at 300s even though manual launch succeeds.** After a maestro attempt, NO vllm process survives and 8002 is empty — so maestro's launch differs from a plain manual launch, OR maestro health-checks vLLM at an address that NAT mode broke.
- Strong lead: `maestro status` reports **`Somatic: 100.92.95.39`** and checks vllm health over that **Tailscale IP**, not localhost. Under NAT, WSL no longer shares the host Tailscale IP, so `100.92.95.39:8002` likely no longer routes into WSL → health check times out even if vLLM is up. i.e. the NAT fix may have traded a bind-collision for a reachability gap that maestro's health check depends on.
- Was about to read maestro's vLLM service source to see its exact launch command + health-check URL when we wrapped. maestro = bash wrapper `~/.local/bin/maestro` → `python -m maestro` with `PYTHONPATH=/home/gasu/ubik`, using venv `~/pytorch_env`. Package dir under `/home/gasu/ubik` (exact path not yet confirmed; `~/ubik/maestro/services/` did NOT exist — need to locate the real `maestro` package dir and its `*vllm*` / health files).

**Other findings (context):**
- No `~/ubik/.env` or `~/ubik/ingestion/.env` currently exist (the 2026-06-20 "set UBIK_ENRICHMENT_MODEL in ingestion/.env" item — that file is absent).
- vllm_config.yaml: model `~/ubik/models/deepseek-awq/DeepSeek-R1-Distill-Qwen-14B-AWQ`, quant `awq_marlin`, dtype float16, gpu_memory_utilization 0.93, max_model_len 98304, port 8002. Model files present and valid.
- The external reachability check `curl http://100.75.228.46:8002/health` (the never-passed item) was NOT reached — blocked upstream by the maestro-vs-manual issue. Note 100.75.228.46 is the Hippocampal-facing IP; SESSIONS.md June entries used Somatic 100.92.95.39 — clarify which IP is canonical for the somatic vLLM endpoint next session.
- SSH/zellij nesting fix from earlier today is unrelated to this and already applied on the WSL side (SSH_CONNECTION forwarding via WSLENV + .bashrc guard; VNC autostart disabled).

**State left in:**
- `.wslconfig` = NAT (mirrored reverted); WSL rebooted into NAT (IP 172.27.177.90). vLLM proven startable manually on 8002. NOT currently running (test instances were killed). maestro still cannot bring vLLM to healthy.
- Zombie :8002 hvsocket binding (PID 9512) still stuck on the Windows host — harmless to WSL-namespace vLLM under NAT, but will need admin HNS-restart or a Windows reboot to fully clear if it ever interferes.
- neo4j/chromadb/mcp/whisperx all unhealthy in `maestro status` — those are Hippocampal-side services (Mac), expected down from here; not this session's target.

**Files changed:**
- `C:\Users\gasu.Adrian\.wslconfig`: networkingMode mirrored → NAT (backup `.bak-nat-fix-20260715-190055`)
- SESSIONS.md: this entry

**Next session should:**
1. Read maestro's vLLM service source (locate the real `maestro` package dir under `/home/gasu/ubik`; find its `*vllm*`/health-check file). Determine (a) the exact launch command maestro uses vs. the manual one that works, and (b) the health-check URL/host — confirm whether it targets `100.92.95.39:8002` (Tailscale) vs `localhost:8002`.
2. If maestro health-checks over the Tailscale IP: under NAT, either (a) re-add WSL port-forwarding so `<host-tailscale-IP>:8002` reaches WSL `172.27.177.90:8002` (netsh portproxy or WSL localhostForwarding), or (b) point maestro's vllm health check at `localhost:8002`. Decide with Gines — this is the mirrored-vs-NAT tradeoff (mirrored gave free external reach but caused the collision; NAT fixes the collision but needs explicit forwarding).
3. Then re-run `maestro start`, confirm vLLM healthy, and finally run the external reachability check (`curl http://<canonical-somatic-IP>:8002/health` from Hippocampal) — the item that has never passed.
4. Clarify canonical somatic vLLM IP (`100.92.95.39` vs `100.75.228.46`) and recreate `ingestion/.env` with VLLM host/port + `UBIK_ENRICHMENT_MODEL`.
---

## Session: 2026-07-15 (b) — Node: Hippocampal (driving Somatic over SSH)
**Goal:** Resolve the last open item from 2026-07-15 — why `maestro start --service vllm` still failed at 300s after the NAT revert, while manual launch worked. User instinct: "it's most likely a simple thing (venv)." It was — in the sense that there was nothing left to fix; vLLM only needed starting.

**Investigated and RULED OUT (all red herrings):**
- **venv**: not the problem. `vllm_server.py`'s shebang is `#!/home/gasu/pytorch_env/bin/python`, so it runs under the venv regardless of how maestro/systemd-run invokes it (system `/usr/bin/python3` lacks vllm but is never reached). venv has `vllm 0.13.0` + a valid `vllm` binary; import works.
- **maestro's probe IP `100.92.95.39` "stale"**: not stale — CORRECT. Discovered **Tailscale runs INSIDE WSL** as a separate tailnet node (`tailscale0` interface holds `100.92.95.39/32`, device `adrian-1`, identity `gasu04@`). So vLLM binding `0.0.0.0:8002` in WSL is directly reachable at `http://100.92.95.39:8002` via Tailscale — no portproxy/localhostForwarding/Windows-host hop needed. This **corrects the 2026-07-06 entry** which called `100.92.95.39` stale/offline; it was reachable all along. Distinct from the Windows host's Tailscale IP `100.75.228.46` (device `adrian`, `acefesan@`, what `ssh windows-server`/`windows-plain` target) — two tailnet identities on one physical box.
- **health-check reachability under NAT**: structurally fine. Pre-flight `curl http://100.92.95.39:8002/health` returned instant (0 ms) "connection refused" — proves the route is intact; only failed because nothing was listening yet.

**Actual root cause (confirmed): the 2026-07-15 diagnosis was right and is now fully resolved.** The only real fault was the `networkingMode=mirrored` port collision (Windows stray nginx:8002 vs vLLM:8002), fixed by the 07-15 NAT revert. vLLM had simply not been started again since, because the `wsl --shutdown` (to apply NAT) wiped the transient `ubik-vllm` systemd user unit, and nothing re-ran `maestro start` until now. The "maestro fails at 300s while manual works" gap closed on its own once the port was clear under NAT — the 07-15 hypothesis that NAT broke the health-check IP was wrong.

**RESOLVED — vLLM is up:**
- Pre-flight all green: port 8002 free in WSL, no orphan procs, GPU 72 MiB/32 GB free, `loginctl show-user gasu` → `Linger=yes`, reachability path proven (instant conn-refused).
- `python -m maestro start --service vllm` → **`✓ vllm started`** (no 300s timeout). vLLM confirmed healthy and serving by the user.
- This means the **never-passed external reachability check is structurally closed**: `curl http://100.92.95.39:8002/health` from Hippocampal now routes correctly. (Did not capture a final 200 curl in this session — user confirmed healthy before the verification curl ran.)

**Cleanup done:**
- `maestro/.env.example`: `SOMATIC_TAILSCALE_IP` default `100.79.166.114` (the genuinely-stale `adrian-wsl` IP) → `100.92.95.39`, with an explanatory comment that this is the WSL-internal Tailscale IP, not the Windows host's. `maestro/.env` was already correct at `100.92.95.39`.
- Added `windows-plain` SSH alias to `~/.ssh/config` (same host/user/key as `windows-server`, but `RequestTTY no` + `RemoteCommand none` → lands in the Windows shell instead of auto-entering WSL). For Windows-side admin (`taskkill`, `Restart-Service hns/LxssManager`, `netsh`, `Get-NetTCPConnection`). Verified: `ver` → Windows 10.0.26200.8875, `whoami` → `adrian\gasu`. Runs non-elevated; admin-only ops still need interactive admin PowerShell at the box.
- Wrote a Claude memory (`somatic-vllm-networking-tailscale.md`) capturing the Tailscale-runs-inside-WSL fact + the correct "vLLM won't start" diagnostic flow (check port + NAT first, don't chase venv/IP theories). Indexed in `MEMORY.md`.

**State left in:**
- vLLM running and healthy on Somatic (`ubik-vllm` systemd user unit, recreated by maestro this session).
- `.wslconfig` = NAT (mirrored reverted 07-15). WSL IP `172.27.177.90` internal / `100.92.95.39` Tailscale.
- Zombie :8002 hvsocket binding may still exist on the Windows host (harmless under NAT); not re-checked this session.
- Maestro web panel / Docker / Neo4j / ChromaDB / MCP on Hippocampal: not touched this session (Docker was fixed earlier in the day by restarting Docker Desktop after its VM died).

**Files changed:**
- `maestro/.env.example`: `SOMATIC_TAILSCALE_IP` stale default corrected + explanatory comment
- `~/.ssh/config` (local, not in repo): added `windows-plain` alias
- SESSIONS.md: this entry
- (local Claude memory, not in repo): `somatic-vllm-networking-tailscale.md` + MEMORY.md index

**Next session should:**
1. Capture the final 200 from `curl http://100.92.95.39:8002/health` from Hippocampal (the never-passed check) as explicit confirmation — a 5-second sanity check now that vLLM is up.
2. Run **CP2**: `run_phase3.py --stage 2 --dry-run --limit 8`, hand-score per `qa/rubric.md` (gate: confidence ≥0.7 on ≥6/8) — the original gate from 2026-06-20, now genuinely unblocked (vLLM reachable at the IP maestro/ingestion both use).
3. Watch for the `ubik-vllm` unit disappearing again after any `wsl --shutdown` (transient units don't survive VM teardown) — re-run `maestro start --service vllm` to recreate. Consider persisting it as an installed `.service` file (long-standing backlog item).
4. Everything else carried over from the 2026-07-06(i)/07-09/07-11(a) pending lists (vLLM 0.24.0 upgrade, live vLLM verification under torch 2.9.1, chromadb version alignment, single-venv doc correction, no-fix CVEs, older backlog).
---

## Session: 2026-07-18 23:18 — Node: Hippocampal
**Goal:** Fix the Claude Code startup warning "Settings (.claude/settings.json): Invalid or malformed JSON".
**Completed:**
- Diagnosed the fault: `.claude/settings.json` parsed cleanly through the closing `}` (line 21), then failed with `json.JSONDecodeError: Extra data: line 22`. Root cause was a leaked heredoc delimiter — a stray `EOF ` line (with a trailing space, so the shell never matched it as the terminator) written into the file, most likely from a `cat > settings.json <<EOF … EOF` creation.
- The permissions block itself (defaultMode acceptEdits; allow git add/commit, pytest, pip install, docker compose/ps; ask git push, rm -rf; deny tccutil, sudo) was valid and left untouched.
- Removed the stray `EOF ` line and its trailing blank line via a surgical edit (no shell command; permissions config unchanged).
- Re-validated with `python3 -c "json.load(...)"` → **VALID JSON**.
**State left in:**
- `.claude/settings.json` is valid on disk and confirmed parseable. The startup "malformed JSON" warning should be gone; a definitive check requires a session relaunch (the warning only fires when the harness reads the file at boot) — not yet done from within this session.
- No other settings issues found or touched. Global `~/.claude/` config not modified.
**Files changed:**
- `.claude/settings.json`: removed leaked heredoc `EOF ` delimiter + trailing blank line (JSON body unchanged).
- SESSIONS.md: this entry.
**Next session should:**
- On next `claude` launch from the UBIK root, confirm no "Invalid or malformed JSON" warning appears for `.claude/settings.json`.
---

## Session: 2026-07-18 23:22 — Node: Hippocampal
**Goal:** Confirm the settings fix was the only source of the malformed-JSON warning by auditing every settings file Claude Code reads.
**Completed:**
- Committed the settings fix + prior log entry as `1657a3b` and pushed to `origin/master` (`dc4157f..1657a3b`). Post-commit hook mirrored SESSIONS.md to Drive.
- Read-only validated all remaining settings files with `json.load`:
  - `.claude/settings.local.json` (project) → VALID
  - `~/.claude/settings.json` (global) → VALID
  - `~/.claude/settings.local.json` (global) → VALID
  - `/Library/Application Support/ClaudeCode/managed-settings.json` (enterprise policy) → absent (no managed policy)
- Conclusion: the project `.claude/settings.json` (fixed in the prior entry) was the sole source of the warning. No other config file is malformed.
**State left in:**
- All settings files Claude Code reads are valid JSON. Warning expected to be gone on next launch; still not re-verified from within a live restart (session relaunch is a user action — cannot self-restart).
**Files changed:**
- No config changed this entry (audit was read-only).
- SESSIONS.md: this entry.
**Next session should:**
- Nothing outstanding on the settings issue; resume the carried-over vLLM/CP2 backlog from the earlier entries when ready.
---

## Session: 2026-07-19 21:04 — Node: Hippocampal
**Goal:** Remove TradingAgents and FinRobot from UBIK and ensure they don't share UBIK's venv.
**Key finding — they were never in the UBIK repo.** TradingAgents and FinRobot are independent projects with their own GitHub remotes (`gasu04/TradingAgents`, `gasu04/FinRobot`) — not git-tracked by UBIK, no submodule, no code/config reference in the tree. The only tie to UBIK was a **misattributed backlog line** in this log ("TradingAgents 3.10→3.12", "FinRobot requirements refresh"), which is being dropped here as out of scope.
**Completed:**
- Located the actual copies: two physical checkouts each on the 990PRO volume — `FinanceAI/{TradingAgents,FinRobot}` and `Claude/{TradingAgents,FinRobot}`. (`~/Claude` is a symlink → `/Volumes/990PRO 4T/Claude`, so the apparent "third copy" in the home dir was the same inodes, not a separate copy.) All at identical git HEADs (TradingAgents `a438acd`, FinRobot `e86ba85`).
- Verified before deleting: no uncommitted tracked changes except benign `.DS_Store`/`config_api_keys` on FinRobot; **secrets files byte-identical across copies** (`config_api_keys`, `OAI_CONFIG_LIST` same sha256 — nothing unique lost); untracked `results/` outputs identical and present in the keeper; no `.env`/key files in TradingAgents.
- **De-duplicated to one copy each:** kept `FinanceAI/{TradingAgents,FinRobot}` (dedicated finance folder); moved the redundant `Claude/` copies to `~/.Trash` (dated, reversible): `TradingAgents_volClaude_dedup_20260719` (1.5G), `FinRobot_volClaude_dedup_20260719` (26M). Both projects remain on GitHub regardless.
- **Venv isolation confirmed:** the remaining copies use their own venvs (TradingAgents: uv Py3.10 `.venv` + Homebrew Py3.13 `venv`; FinRobot: none). UBIK uses `DeepSeek/venv` + `~/ubik-chromadb-venv`. No symlinks or overlap — they never shared UBIK's venv and still don't.
**State left in:**
- One copy each of TradingAgents/FinRobot remains, at `/Volumes/990PRO 4T/FinanceAI/`. Redundant copies in `~/.Trash` (empty Trash to reclaim ~1.5G once satisfied).
- No change to the UBIK repo tree (these were never in it). `~/Claude` symlink untouched.
- **Backlog correction:** TradingAgents/FinRobot maintenance items are dropped from UBIK's carried-over backlog — they are separate projects, not UBIK work.
**Files changed:**
- No UBIK repo files changed on disk (TradingAgents/FinRobot were external).
- SESSIONS.md: this entry.
**Next session should:**
- Resume the genuine UBIK backlog (vLLM/CP2 and the housekeeping items), now free of the misattributed TradingAgents/FinRobot lines.
---

## Session: 2026-07-19 21:08 — Node: Hippocampal
**Goal:** Empty the Trash to reclaim disk space after the TradingAgents/FinRobot de-dup.
**Completed:**
- Emptied `~/.Trash` entirely — **~4.0G reclaimed** (verified 0B / empty afterward). This is a **permanent, irreversible** delete, confirmed by the user at full scope.
- Permanently removed: today's de-dup copies (TradingAgents 1.5G, FinRobot 26M) + the 2026-07-06 evictions (rag_env, AutoGPT, nifi, automatisch, Cross-Platform-Workflow-Orchestrator, claude-code, evernote-sdk-python, gastos-promed, pythonProject, helloworld, test-setup, UBIKParallax-source-v5, install.sh) + a stray `Google Drive.app`.
- Note: the GitHub-backed projects among those (AutoGPT, nifi, automatisch, claude-code, Cross-Platform-Workflow-Orchestrator, gastos-promed; plus TradingAgents/FinRobot which still live under `FinanceAI/`) remain re-clonable from `gasu04`. Nothing UBIK depends on was in Trash.
**State left in:**
- `~/.Trash` empty (0B). The reversibility safety net from the 2026-07-06/07-19 evictions is now gone by design — those items are only recoverable via their GitHub remotes (where they have one).
- No UBIK repo files changed on disk (Trash cleanup is outside the repo).
**Files changed:**
- No repo files changed (disk-only cleanup).
- SESSIONS.md: this entry.
**Next session should:**
- Resume the genuine UBIK backlog (vLLM/CP2 and the housekeeping items).
---

## Session: 2026-07-19 21:18 — Node: Hippocampal
**Goal:** Run CP2 (Phase 3 enrichment quality gate), the gate blocked since 2026-06-20, now that vLLM is reachable. Predecessor: 2026-07-15(b).

**Completed:**

**Step 0 — IP hypothesis CONFIRMED (empirically, from Hippocampal):**
- `curl http://100.92.95.39:8002/health` → **HTTP/1.1 200 OK** (server: uvicorn, content-length: 0, time_total=0.28–0.40s; reached from Hippocampal `100.103.242.91` over Tailscale). vLLM is up and serving.
- `curl http://100.75.228.46:8002/health` → **timeout (5s, http_code 000)**. The Windows-host Tailscale IP does NOT route into the WSL namespace under networkingMode=NAT. No undocumented portproxy exists — the two-tailnet-identity model in 07-15(b) is correct as written. This reproduces the 07-13 CP2 failure cause live, so it is the confirmed root cause, not a coincidence.

**Step 1 — never-passed health check now captured (verbatim):**
```
HTTP/1.1 200 OK
date: Mon, 20 Jul 2026 01:55:08 GMT
server: uvicorn
content-length: 0
```
`/v1/models` also probed: served id `/home/gasu/ubik/models/deepseek-awq/DeepSeek-R1-Distill-Qwen-14B-AWQ` — matches `UBIK_ENRICHMENT_MODEL` exactly.

**Step 2 — `ingestion/.env` IP reconciliation — live config ALREADY correct; template was stale:**
- **Correction to the brief's premise:** the brief's table predicted *both* `ingestion/.env` and `.env.example` held `100.75.228.46`. In fact the gitignored **live `ingestion/.env` was already at `100.92.95.39`** (both `SOMATIC_HOST` and `SOMIC_TAILSCALE_IP`) — already corrected at some prior point. Only the **committed `ingestion/.env.example`** still showed `100.75.228.46` in two places.
- Edited `ingestion/.env.example` only: `SOMATIC_HOST` and `SOMATIC_TAILSCALE_IP` → `100.92.95.39`, with comments mirroring `maestro/.env.example` (this is the WSL-internal tailnet node `adrian-1`/`gasu04@`, not the Windows host `adrian`/`acefesan@`; the Windows IP does not route into WSL under NAT). No active var now holds `100.75.228.46`; only explanatory comments mention it.
- Verified through the **real code path** (`load_config()`, not the bare `EndpointConfig.from_env()` the brief's snippet used — which bypasses `.env` loading and misleads by falling through to the unresolved `ubik-somatic` hostname): `sensitive_endpoint=standard_endpoint=for_tier("therapy")=for_tier("business")=http://100.92.95.39:8002/v1`; `model=/home/gasu/ubik/models/deepseek-awq/DeepSeek-R1-Distill-Qwen-14B-AWQ`. Matches the brief's expectation. (Hostname `ubik-somatic` does not resolve on this Mac — NXDOMAIN — so the explicit-IP config is load-bearing; do not remove `SOMATIC_TAILSCALE_IP`.)

**Step 3 — CP2 run and hand-scoring:**
- Prereq: **no Hippocampal Python env existed** for ingestion. `venv` symlink at repo root → `/home/gasu/pytorch_env` (Somatic Linux path, unreachable from Mac; this is the "single-venv doc correction" backlog item made concrete). `ENVIRONMENT.md` also documents the Somatic env only. Created a dedicated Hippocampal venv `~/.virtualenvs/ubik-ingestion` (uv, Python 3.12.12) and installed the stage-2 subset: python-docx, httpx, openai, pydantic, python-dotenv, PyYAML, jsonschema. Skipped openai-whisper / pdfplumber / pytesseract / pdf2image / google-* (not exercised by stage-2 on .docx; need system binaries). **Flagging as a new backlog item:** ingestion needs a documented/installed Hippocampal env (this venv is ad-hoc, not yet committed as a Make/README step).
- `python run_phase3.py --stage 2 --dry-run --limit 8` → **Stage 2: 3 processed, 0 skipped, 0 quarantined** (70s wall; all three `/v1/chat/completions` returned 200). Note: `--limit` is applied **per source directory** inside `enrich_directory`, but only `sources/tactiq/` has files (3), so exactly 3 ran. The other 5 buckets (gemini/fireflies/letters/memory_notes/constitution) are empty.
- No reasoning-chain leak: 0 `<think>`/`<reasoning>` tags in any enriched body (Tier 1 stripping held).

**CP2 scoring table (hand-judged against qa/rubric.md — model conf ≠ rubric verdict):**

| # | File | meeting_type | diarization_status | voice_corpus_eligible | conf | band | my verdict |
|---|------|-------------|--------------------|-----------------------|------|------|------------|
| 1 | 12 23-12-2025 Khaled…actinium 225 RFCA | business ✓ | mono ✓ | false ✓ (mono hard-rule) | 0.95 | auto-approve | **PASS** — real multi-party dialogue (clinical isotopes); participants inferred from filename not diarization (112 generic "Speaker:" labels), names plausible but unverified |
| 2 | 3 Protocolo de Entregas…03-12-2025 | business ✓ | multi ⚠️ | false ✓ | 0.85 | auto-approve | **PASS (caveat)** — ⚠️ body is NOT a transcript: Tactiq metadata stub + one AI summary paragraph, no dialogue. `diarization:multi` reflects 3 colon markers but is misleading; participants `[Ginés Sanchez Urrutia]` = creator field, not attendees. Metadata extraction correct; conf 0.85 overstates a record with no verbatim content |
| 3 | Meeting Transcription 16-12-2025 | business ✓ | multi ✓ | false ✓ | 0.95 | auto-approve | **PASS** — real long dialogue, Fabio Robledo named; clean |

- All 3 confidences ≥ 0.7 (actually all ≥ 0.85, auto-approve band). Hard rules all hold: no therapy, mono file 1 voice-blocked, Ginés/Ginés Alberto not collapsed (bare "Ginés" = father only).
- **GATE JUDGEMENT — pass with a denominator caveat that needs a decision, not a silent pass.** Gate is "conf ≥0.7 on ≥6 of 8." I have **3 of 3 ≥0.7**, but only **3 source files exist** (not 8). Per Step 4's "don't loosen a gate to pass" rule, I am NOT declaring this a clean 6/8 pass; I'm reporting 3/3 quality passes against a denominator of 3, and flagging that the corpus is too small to exercise the 6/8 threshold as written. Decision needed (Gines): is CP2 "passed" on the strength of 3/3 clean enrichments, or does the gate require populating ≥6 source files first?

**State left in:**
- vLLM healthy on Somatic (`100.92.95.39:8002`). `ubik-vllm` transient unit still up (watch for disappearance after any `wsl --shutdown`).
- New Hippocampal venv `~/.virtualenvs/ubik-ingestion` (Python 3.12, stage-2 deps only) — ad-hoc, not yet a committed setup step.
- 3 enriched outputs in `ingestion/enriched/*.transcript`; `quarantine/enrichment/` empty. Nothing written to ChromaDB/Neo4j (dry-run held).
- `ingestion/.env.example` IP corrected (uncommitted, in the Step-2 commit below).

**Files changed:**
- `ingestion/.env.example`: `SOMATIC_HOST` + `SOMATIC_TAILSCALE_IP` `100.75.228.46` → `100.92.95.39` + explanatory comments (mirrors `maestro/.env.example`).
- SESSIONS.md: this entry.
- (local, not in repo): `~/.virtualenvs/ubik-ingestion` created.

**Next session should:**
1. Get Gines's decision on the CP2 denominator (3/3 quality-pass vs the "≥6 of 8" threshold). If "pass on 3/3": proceed to plan CP3. If "needs ≥6 files": populate `sources/` from the Ingested_data / Drive corpus first, then re-run.
2. Investigate File 2's content-classification oddity (a no-dialogue Tactiq summary stub scored as `diarization:multi` with 0.85 conf) — likely warrants a content-type rule or a confidence penalty for near-empty transcripts. Diagnosis only; do not silently tune.
3. Persist/document the Hippocampal ingestion env (`~/.virtualenvs/ubik-ingestion` → a committed setup step; fold into the "single-venv doc correction" backlog item — `ENVIRONMENT.md` currently describes Somatic only).
4. Carried backlog, unchanged: vLLM 0.24.0 upgrade (18 CVEs, plan ready); live vLLM inference verification under torch 2.9.1 (partially done this session — chat completions + models endpoints confirmed); chromadb version alignment across three envs (1.5.1/1.4.1/1.3.7); single-venv doc correction; persist `ubik-vllm`/`ubik-whisperx` as installed `.service` files; per-service `maestro shutdown --service NAME`; WhisperX health tests; retire `ubik-memory-sweep` skill; update ingestion loader (new fields + `EPISODIC` token); TradingAgents Python 3.10→3.12 before October EOL.

---

## Session: 2026-07-19 22:19 - Node: Hippocampal
**Goal:** Troubleshoot WhisperX on Somatic (unhealthy); make maestro manage the new install; surface WhisperX + UBIKParallax on the :8090 web panel.
**Root cause (WhisperX):** unit `ubik-whisperx` had run 1+ day but `/health` reported `model_loaded:false` - neither `whisperx` nor `whisper` was installed in the shared venv (`~/ubik/venv` -> `~/pytorch_env`). Every startup since Jul 15 failed at import; the logged "No module named 'whisper'" was the fallback masking the missing primary. GPU was also 98% full (vLLM gpu_memory_utilization 0.93), so CUDA was not viable regardless.
**Completed (decisions: isolated venv + CPU):**
- Built `~/ubik-whisperx-venv` on Somatic: whisperx 3.8.6 (+ faster-whisper/ctranslate2), fastapi/uvicorn/python-multipart, and pip `static-ffmpeg` (symlinked ffmpeg/ffprobe into the venv bin). Runs CPU/int8.
- Made `somatic/whisperx_server.py` device/compute env-driven (`WHISPERX_DEVICE`/`WHISPERX_COMPUTE_TYPE`; defaults still cuda/float16). Applied on the Mac (committed) and patched in-place on the Somatic copy (backup `.bak.20260719`).
- Recreated the `ubik-whisperx` systemd user unit on the new venv with `WHISPERX_DEVICE=cpu`, `WHISPERX_COMPUTE_TYPE=int8`, PATH incl. venv bin. large-v2 downloaded to HF cache; `/health` -> `model_loaded:true, device:cpu`.
- maestro now targets the isolated venv: added `SomaticConfig.whisperx_venv/whisperx_device/whisperx_compute_type` (env-overridable), threaded into `WhisperXService`, and `_remote_start` launches the new venv python with the CPU env + PATH. status/stop were already venv-independent.
- Fixed a real display bug: `maestro status` counted whisperx (7/7) but never rendered its row - `_SERVICE_ORDER`/`_SERVICE_NODES` in `display.py` omitted it. Added; CLI + web now show whisperx healthy.
- Web panel (:8090): added config-driven UBIKParallax link (`MaestroConfig.parallax_url`, env `UBIK_PARALLAX_URL`, default live GitHub Pages `https://gasu04.github.io/UBIKParallax/`) in `/api/config` + app.js (red accent), plus a `Cache-Control: no-cache` middleware so the panel stops serving stale JS/CSS. Restarted the launchd web agent.
**State left in:**
- WhisperX healthy on Somatic (CPU, isolated venv); shows healthy in `maestro status` and the web panel. vLLM untouched/healthy.
- The Mac `maestro` command is the alias `cd UBIK && DeepSeek/venv/bin/python -m maestro` (runs repo source); web served by launchd `com.ubik.maestro-web` via `~/ubik-bin/ubik-maestro-python`.
- Web panel http://100.103.242.91:8090/ up to date + UBIKParallax button (browser needs ONE hard refresh to clear old cached app.js).
- ffmpeg exists only in the whisperx venv (pip static-ffmpeg); system ffmpeg still absent (torchcodec warning is harmless for the ctranslate2 path).
- Somatic `/home/gasu/ubik` has an in-place edit to `somatic/whisperx_server.py` (canonical change committed on the Mac).
**Files changed:**
- `somatic/whisperx_server.py`: device/compute env-driven
- `maestro/config.py`: whisperx venv/device/compute + parallax_url
- `maestro/services/__init__.py`: pass whisperx config through
- `maestro/services/whisperx_service.py`: remote_venv/device/compute params + `_remote_start` venv+env
- `maestro/display.py`: whisperx added to status table order + node map
- `maestro/web/server.py`: parallax link in /api/config + no-cache middleware
- `maestro/web/static/app.js`, `maestro/web/static/style.css`: UBIKParallax link + accent
- SESSIONS.md: this entry
**Next session should:**
- Optional: finish per-service `maestro shutdown --service NAME` (started, deferred) so whisperx can be stopped in isolation; add WhisperX health/lifecycle tests.
- Persist `ubik-whisperx`/`ubik-vllm` as installed `.service` files (transient units vanish on `wsl --shutdown`).
- First whisperx model load pulls large-v2 (~3GB) to the HF cache; later loads are fast.
---

## Session: 2026-07-20 20:26 - Node: Hippocampal
**Goal:** Finish the per-service `maestro shutdown --service NAME` feature (carried over, deferred last session).
**Completed:**
- `shutdown.py`: added an optional `service_filter: set[str]` to `_services_in_shutdown_order`, `orderly_shutdown`, and `_verify_all_down` so a single service can be stopped without touching the rest of the cluster. In `orderly_shutdown` the parameter is named `service_filter` (not `services`) to avoid shadowing the local `services` list var - thread carefully if touching this.
- `cli.py`: added `shutdown --service NAME` (click.Choice-validated against ALL_SERVICE_NAMES), symmetric with the existing `start --service`. `--service` + `--emergency` are mutually exclusive (exits 2); dry-run output now names the scope (service name / local / cluster).
- `web/server.py`: `ShutdownRequest` gains `service: Optional[str]`; `/api/shutdown` honors it (stops one service) and rejects service+emergency with HTTP 400.
- Tests: +8 in `test_shutdown.py` (4 service-filter unit tests incl. unknown-name-empty and filter+local_only composition; 4 CLI tests incl. mutual-exclusion and unknown-name rejection).
- **Live-verified:** `shutdown --service whisperx` stopped only whisperx (curl /health -> unreachable) while vLLM stayed HTTP 200; `start --service whisperx` brought it back healthy (`model_loaded:true, device:cpu`). Full maestro suite: 225 passed, 1 pre-existing error (`test_logger.py::TestConfigureLogging::test_creates_log_directory` teardown `lost sys.stderr`, already documented - unrelated).
**State left in:**
- Feature shipped: committed `9d337f1`, pushed to origin/master.
- WhisperX currently healthy on Somatic (restarted during verification).
- **`maestro/services/vllm_service.py` is intentionally UNSTAGED** - it carries pre-existing Blackwell/FlashInfer env changes (`VLLM_USE_FLASHINFER_SAMPLER=0` workaround, `_BLACKWELL_ENV` applied to remote start, VLLM_VENV_PATH resolution in _remote_start) from other work, NOT part of this feature. Held per user instruction - commit only when told.
**Files changed:**
- `maestro/shutdown.py`: service_filter across order/verify paths
- `maestro/cli.py`: shutdown --service option + emergency guard
- `maestro/web/server.py`: ShutdownRequest.service + handler
- `maestro/tests/test_shutdown.py`: +8 tests
- SESSIONS.md: this entry
**Next session should:**
- Await user go-ahead to commit `maestro/services/vllm_service.py` (Blackwell/FlashInfer vLLM env) as its own commit.
- Remaining carried backlog: persist ubik-vllm/ubik-whisperx as installed .service files; WhisperX health/lifecycle tests; retire ubik-memory-sweep; update ingestion loader; vLLM 0.24.0 upgrade; chromadb version alignment; single-venv doc correction.
---

## Session: 2026-07-20 20:53 — Node: Hippocampal (driving Somatic over SSH)
**Goal:** Upgrade Somatic vLLM 0.13.0 → 0.24.0 (Option B: parallel venv, zero-risk rollback) and wire maestro to manage 0.24. Clears the 38 vLLM CVEs (9 require 0.24.0). Predecessor: 2026-07-19 (CP2) + the concurrent 2026-07-20 per-service-shutdown session.

**Investigation findings (correcting the stale 07-05e plan):**
- Live Somatic (verified): vllm 0.13.0, **torch already 2.9.1+cu128** (the "torch 2.9.0 drift" backlog item is stale — already fixed), cuda 12.8, ray 2.56.0 (clean). Only vLLM itself left to upgrade.
- Fresh `pip-audit`: **52 vulns, 38 in vllm 0.13.0** (~24 distinct; grew from the logged "18"). 9 CVEs require **0.24.0** as the floor — the old plan's "0.22.x" target is stale. Latest upstream is now 0.25.1.
- **0.24.0 publishes NO cu128 wheel** (only +cpu, +cu129, and a default/cu13 build). Our working 0.13 stack is cu128 → **0.24.0 forces a move to CUDA 12.9** (user chose cu129 wheel + torch cu129). Driver 591.86 / max CUDA 13.1 supports it.
- The Tier-1 VRAM-cleanup import (`destroy_model_parallel`/`cleanup_dist_env_and_memory` from `vllm.distributed.parallel_state`) is **stable through current main** — the 07-05e worry was unfounded; no `vllm_server.py` patching needed. `awq_marlin` still registered; model loads fine.

**Completed — upgrade (Option B):**
- `~/pytorch_env_vllm_0.13_freeze.txt` created (rollback snapshot; was absent). Installed `uv` on Somatic (was Hippocampal-only).
- Built `~/pytorch_env_vllm024` (Python 3.12.3) + `vllm==0.24.0+cu129` wheel + torch 2.11.0+cu129 + cuda 12.9 + triton 3.6.0 (194 pkgs). First attempt with default wheel failed (`libcudart.so.13` — cu13 build vs cu128 torch); the cu129 wheel from GitHub releases fixed it.
- **Crash #1 (root cause pinned):** FlashInfer 0.6.12 misdetects Blackwell sm_120 → `RuntimeError: FlashInfer requires GPUs with sm75 or higher` in the sampler profile run. vLLM 0.24 routes sampling through FlashInfer by default.
- **Crash #2 (my bug):** retry with `VLLM_USE_FLASHINFER_SAMPLER=false` failed on `int('false')` ValueError — the flag wants `0`/`1`, not `false`.
- **Fix:** `VLLM_USE_FLASHINFER_SAMPLER=0` → `/health` 200 in ~90s. CP2 acceptance vs the 0.13 baseline: identical on all scored rubric fields (meeting_type/diarization/voice_eligible/conf all match; conf 0.95/0.95), ~2.4× faster (29s vs 70s), 0 reasoning-tag leaks. One file quarantined on a `25-12-2025` date-format slip — schema correctly trapping a model error, not a defect.

**Completed — maestro wiring (proper fix, mirrors the WhisperX pattern from the 07-19 session):**
- **First attempt was wrong:** I read `os.environ["VLLM_VENV_PATH"]` directly in `_remote_start`. Maestro's CLI only loads `maestro/.env` into `os.environ` when `--config` is passed — so `VLLM_VENV_PATH` was absent at runtime → fell back to `vllm_server.py`'s 0.13 shebang → unit launched `vllm_server.py vllm_server.py …` (`unrecognized arguments`), 300s health timeout.
- **Corrected to the WhisperX pattern:** (a) `config.py: SomaticConfig.vllm_venv` typed Field (alias `VLLM_VENV_PATH`, default the 0.24 venv) — populated via `get_config()` which auto-loads `.env`; (b) `vllm_service.py: VllmService.__init__` gains `remote_venv` → `self._remote_venv`, and `_remote_start` uses it with the `"$PYTHON" "$SERVER"` form + `--setenv` from `_BLACKWELL_ENV` (incl. the FlashInfer flag); (c) `__init__.py` passes `remote_venv=somatic.vllm_venv`. No raw `os.environ`.
- `_BLACKWELL_ENV` now carries `VLLM_USE_FLASHINFER_SAMPLER: "0"` and is applied to **both** local + remote paths (previously local-only). Updated the stale `VLLM_FLASH_ATTN_VERSION` comment (no-op on ≥0.24, kept for 0.13).

**Live-verified (end-to-end through maestro):**
- `maestro start --service vllm` → ✓ healthy; process is `pytorch_env_vllm024/bin/python` (not the 0.13 shebang); unit `Environment` includes `VLLM_USE_FLASHINFER_SAMPLER=0`; `/health` 200.
- `maestro shutdown --service vllm` → VRAM 131 MiB (graceful Tier-1 cleanup held), no survivors.
- `maestro start --service vllm` (2nd cycle) → ✓ healthy (repeatable).
- CP2 through the maestro-managed instance: 3 processed / 0 quarantined, conf 0.95/0.90/0.95 (File 2 passed this run — confirms the earlier quarantine was non-deterministic, not systematic).
- maestro test suite: 171 passed in the relevant files (vllm/shutdown/orchestrator); full suite 225 passed + 1 pre-existing `test_logger.py` stderr-teardown flake (documented, unrelated).

**State left in:**
- **vLLM 0.24.0 is the active, maestro-managed version** on Somatic (`ubik-vllm` unit, `~/pytorch_env_vllm024`). 0.13 rollback intact at `~/pytorch_env` + `~/pytorch_env_vllm_0.13_freeze.txt`; flip `VLLM_VENV_PATH` to revert.
- WhisperX unaffected (its own venv).
- Maestro web panel / other services untouched.

**Files changed (uncommitted):**
- `maestro/config.py`: +`SomaticConfig.vllm_venv` Field.
- `maestro/services/vllm_service.py`: `remote_venv` param + `_remote_venv`; `_remote_start` mirrors WhisperX (explicit venv python + `--setenv`); `_BLACKWELL_ENV` gains FlashInfer flag + applies remotely.
- `maestro/services/__init__.py`: pass `remote_venv=somatic.vllm_venv`.
- `maestro/.env` (gitignored): `VLLM_VENV_PATH=/home/gasu/pytorch_env_vllm024`.
- `maestro/.env.example`: default → 0.24 venv + two-venv comment.
- `ingestion/enriched_baseline_013/` (gitignored): 0.13 CP2 outputs preserved for comparison.
- SESSIONS.md: this entry.
- (Somatic, not in repo): `~/pytorch_env_vllm024`, `~/pytorch_env_vllm_0.13_freeze.txt`, `uv` installed at `~/.local/bin`.

**Next session should:**
1. **Commit** the 4 maestro files + `.env.example` as one focused "vLLM 0.24 + maestro venv-aware wiring" commit (the concurrent shutdown session deliberately left `vllm_service.py` unstaged for this). Suggested message references the FlashInfer sm_120 workaround + the WhisperX-pattern mirror.
2. Optional: chase a FlashInfer build with sm_120 cubins to re-enable the FlashInfer sampler (faster) and drop the workaround — separate task.
3. Optional: FA2/FA3/FA-default benchmarking on 0.24 (`VLLM_FLASH_ATTN_VERSION` is a no-op on 0.24; engine auto-selects).
4. Carried backlog, unchanged: persist `ubik-vllm`/`ubik-whisperx` as installed `.service` files (transient units vanish on `wsl --shutdown`); WhisperX health/lifecycle tests; retire `ubik-memory-sweep`; update ingestion loader (new fields + `EPISODIC` token); chromadb version alignment across three envs (1.5.1/1.4.1/1.3.7); single-venv doc correction (`ENVIRONMENT.md` still describes Somatic only); the `test_logger.py` stderr-teardown flake.
5. The 0.24 upgrade **closes the 38 vLLM CVEs** — re-run `pip-audit` next session to confirm the new count (expect the vllm/torch rows gone; chromadb/diskcache/nltk no-fix remain).

---

**Addendum 2026-07-20 21:00 — CVE verification done (same session).**

`pip-audit` on the 0.24 venv hit a limitation: vllm/torch/torchvision/torchaudio all carry `+cu129` local-version tags, which pip-audit **skips** ("Dependency not found on PyPI") — so it reported only 2 vulns (diskcache, setuptools) and couldn't speak to the vLLM/torch CVEs at all. The `-r` resolver path then choked on a `cuda-toolkit==12.9.1` dependency conflict. **Worked around it by querying OSV.dev directly** (the same DB pip-audit uses), which has no local-version problem:

| Package | Before (0.13 env) | After (0.24 venv) | Note |
|---|---|---|---|
| **vllm** | 0.13.0 → **42 vulns** | 0.24.0 → **0** | ✅ all cleared. Sanity: OSV returns 42 for 0.13.0, confirming it indexes vLLM advisories — "0.24.0 → 0" is real, not a DB gap. |
| **torch** | 2.9.1 → **4 vulns** | 2.11.0 → **1** | ✅ 3 of 4 cleared. Remaining: **CVE-2025-3000** (`torch.jit.script` memory corruption, CVSS ~3.4 low) — was present on 2.9.1 too (not a regression); fixed only in torch 2.13.0, not yet released. |
| setuptools | 80.10.2 → 1 (PYSEC-2026-3447) | reverted to 80.10.2 | see below |
| diskcache | 5.6.3 → 1 (PYSEC-2026-2447) | unchanged | no upstream fix |
| ray / triton / flashinfer / torchaudio / torchvision | — | **0** each | clean |

**Net: 52 → 3 vulns, all 3 non-actionable** (1 torch low-sev pre-existing pending 2.13.0; 2 no-fix/not-in-scope). **Headline confirmed: the 38–42 vLLM CVEs are gone.**

**setuptools bump attempted then reverted:** tried `setuptools>=83.0.0` to clear PYSEC-2026-3447 — it cleared the CVE but **violated vLLM's declared `<81.0.0` and torch's `<82` constraints** (no version satisfies both the fix ≥83 and the caps <81). Reverted to 80.10.2 (dependency-clean, vLLM import + `/health` re-verified). The CVE is a `MANIFEST.in` Unicode-glob exclude-evasion bug exploitable only when building an sdist from attacker-controlled filenames — not in scope for a serving inference server that never builds packages. Revisit when vLLM/torch relax their caps or a constraint-valid fix exists. **No repo change from this** (setuptools is in the Somatic venv, not git-tracked); the 0.24 venv on Somatic now has setuptools 80.10.2.


---

## Session: 2026-07-23 00:30 — Node: Hippocampal (driving Somatic over SSH)
**Goal:** Diagnose `maestro start` failing (docker + vllm both "did not become healthy"), then per user request: verify UBIK/vLLM/WhisperX venv isolation, audit project structure for cross-repo interference, root-cause the *endemic* vLLM-start failure, and produce a deep CLAUDE.md-compliance hardening list.

**Immediate incident — two INDEPENDENT failures (not one):**
- **Docker (Hippocampal):** `docker info` → `500 Internal Server Error`; backend log → `connect tcp 192.168.65.7:2376: no route to host` + `guest-services/stats.sock: connection refused`. Docker Desktop's app/backend processes are up but its **Linux VM is wedged**. Cascades: `neo4j`/`chromadb`/`mcp` all skip on the failed `docker` dependency. Same mode as 07-22 (fixed then by restarting Docker Desktop). **Restarted Docker Desktop this session (quit + pkill + relaunch); as of wrap-up the daemon still returns HTTP 000 — needs manual attention at the machine (likely the macOS admin dialog per 06-xx line 77, or VM re-wedged). NOT yet recovered.**
- **vLLM (Somatic):** unit launches cleanly (`STARTED_UNIT=ubik-vllm rc=0`, 3×) then systemd issues a clean **Stop** ~15s later, mid model-load (journal: `Started` → 15s → `Stopping`, no crash/OOM). WSL Ubuntu found `Stopped`; `last` showed rapid reboot churn 18:25-18:29. `.wslconfig` verified intact (`vmIdleTimeout=-1`, `networkingMode=NAT`) so NOT the idle-timeout regression — matches the intermittent **WSL2 `p9io`/`CheckConnection` VM-poweroff** bug (07-11b, line 677). Issued `wsl --shutdown` to force a clean VM reinit; VM confirmed Stopped. **Did NOT re-run `maestro start --service vllm` yet** (user redirected to the audit).

**vLLM endemic root-cause (the real deliverable — synthesized across 07-11/13/15/19/23):**
The failure is NOT one bug — it's a **fragile launch architecture with a diagnostic blind spot**. The proximate cause differed every time (idle-timeout teardown; p9io poweroff; mirrored-mode nginx:8002 collision; orphaned wslrelay hvsocket; direct-os.environ bug; tonight clean VM teardown). Three common denominators:
1. **Transient systemd unit** (`systemd-run --unit=ubik-vllm`) — vanishes on any `wsl --shutdown`/VM teardown. Backlog since line 368; still not persisted as an installed `.service`.
2. **WSL2 platform fragility** (p9io transport, HNS socket leaks) tears the VM down ~15s in — intermittent, host-side, unfixable from the VM.
3. **DIAGNOSTIC BLIND SPOT (the actual code defect):** `maestro/services/base.py::_wait_for_healthy` polls `/health` for the full 300s and reports a generic *"did not become healthy within 300s"* — it **never checks `systemctl --user is-active ubik-vllm` or reads the journal**. When the unit dies mid-load, maestro is blind — which is *why every recurrence needed a fresh manual journal dig to find a different cause*.
Durable fixes (priority): **(A)** make the health-wait detect unit death and surface the last ~20 journal lines (HIGH value, LOW risk, pure maestro code); **(B)** persist `ubik-vllm`/`ubik-whisperx` as installed+enabled `.service` files; **(C)** add a 1-2× start retry with a `wsl --shutdown` between (matches the documented "3rd attempt works"); **(D)** Windows-side p9io investigation (out of repo scope).

**Venv isolation audit — VERDICT: NO dedicated UBIK venv exists.**
- `UBIK/venv` is a **broken symlink → `/home/gasu/pytorch_env`** (Linux path, dead on macOS; violates §3.5).
- `maestro`/`ingestion`/`hippocampal` **all run under the shared DeepSeek venv** (`/Volumes/990PRO 4T/DeepSeek/venv`, py3.13.7, 381 pkgs incl. langchain/langgraph/transformers/whisper). The `maestro` shell alias hardcodes that interpreter.
- **Concrete interference:** hippocampal pins `fastmcp==2.14.3` but the shared venv carries **3.4.2** (major override — pin can never hold while shared; last `pip install` wins). Also `sentence-transformers` 5.1.2→5.6.0, `python-dotenv` drift.
- **Somatic isolation is CORRECT:** `pytorch_env_vllm024` (vLLM) + `ubik-whisperx-venv` (WhisperX) are properly separate. vLLM/WhisperX isolation = GOOD, per past decisions.
- Root `requirements.txt`/`requirements-frozen.txt` actually describe the **Somatic GPU node** (torch+cu128, CUDA) — mislocated at repo root, a trap for provisioning macOS. No `.python-version`/`tox.ini`/working setup script. Only `deepseek/` is a proper PEP-621 package.

**Project-structure audit (vs §3.2) — CRITICAL privacy finding + FIXED this session:**
- **`chromadb_data/` (37M, personal episodic/semantic memory) was NOT gitignored** — `.gitignore` covered only `data/`/`models/`. A single `git add -A` would have committed personal memory content permanently (§2.4). Same exposure: `backups/`, `Ingested_data/`, `Artifacts/`, `chromadb_data_test/`, `.tmp.drive*`, `enriched_baseline_013/` (meeting transcripts), plus vendored `open-notebook/`/`UBIKParallax-source-v6/`. Nothing tracked yet — latent trap.
- **FIXED:** extended `.gitignore` to cover all of the above (+ `*.gguf`). Verified all 8 dirs now `git check-ignore`-clean. **This is the one code change this session.**
- Other gaps: no subproject has `requirements.lock` (§2.7); `e2e/` and `docs/` exist nowhere; `pyproject.toml` only in deepseek; `somatic/` least §3.2-compliant; `ingestion/` has no top-level `__init__.py`.

**CLAUDE.md code-compliance audit (maestro, Part 2):**
- PASS: §2.1 Config (IP/port literals are settings-defaults/docstrings — allowed; ports DI'd from config), §2.2 Async-first (100% AsyncClient), §2.5 Docs (rich — only the Tier-classification line missing).
- **FAIL §2.3 Resilience (headline):** NO circuit breaker, NO retry+jitter anywhere; the canonical `mcp_client/circuit_breaker.py`+`resilience.py` are **never imported**. `base.py:290-318` is a constant-3s poll — the exact probe-storm pattern §2.3 forbids. (Time source `perf_counter` OK; health checks OK.)
- **FAIL §2.8 Code Org:** no `UbikError` hierarchy; **94 bare `except Exception`** in deep code (only top-level allowed). DI + type hints GOOD.
- PARTIAL §2.4 (no `SafeJSONFormatter`; low risk, infra-only), §2.6 (broad per-module tests but flat dir, no unit/integration/e2e/fixtures).

**Files changed:**
- `.gitignore`: +chromadb_data/ (+test), Ingested_data/, Artifacts/, backups/, *.gguf, .tmp.drive*/, open-notebook/, UBIKParallax-source-v6/, ingestion/enriched_baseline_*/ — closes the personal-memory commit trap.
- `SESSIONS.md`: this entry.

**State left in:**
- **Docker Desktop: NOT recovered** (daemon HTTP 000 after restart) — blocks neo4j/chromadb/mcp. Needs manual look at the machine.
- **vLLM: NOT running.** WSL VM cleanly shut down (ready for a fresh `maestro start --service vllm`, which per history should succeed on a clean VM).
- No maestro/ingestion/hippocampal *code* touched — audits were read-only; only `.gitignore` hardened.

**Next session should (prioritized hardening backlog from tonight's audits):**
1. **Recover Docker** (check macOS admin dialog / Docker Desktop → Troubleshoot → Restart or full quit-relaunch), confirm neo4j/chromadb/mcp come healthy. Then `maestro start --service vllm` on the clean WSL VM and verify `/health` from Hippocampal.
2. **vLLM durable fix (A):** teach `maestro/services/base.py::_wait_for_healthy` (or vllm `_remote_start`) to detect `ubik-vllm` unit death during the wait and surface the journal tail — turns the 300s silent timeout into an instant actionable error. Highest value / lowest risk.
3. **Give UBIK its own venv:** create `ubik-venv` (py3.13) = union of maestro+ingestion+hippocampal reqs (NO torch/langchain), resolve fastmcp 2 vs 3, fix the dead `venv` symlink, repoint the `maestro` alias, correct `ENVIRONMENT.md` (the documented single-venv-doc backlog).
4. **§2.3 resilience:** import the canonical circuit-breaker+retry into maestro's probes/SSH (do NOT reimplement — §3.4).
5. **Persist `ubik-vllm`/`ubik-whisperx` as installed `.service` files** (vLLM durable fix B; kills the transient-unit failure mode).
6. Lower: §2.8 UbikError hierarchy + narrow the 94 bare excepts; §2.6 test-dir restructure; §2.5 Tier lines; §2.7 per-subproject requirements.lock; namespace/relocate the Somatic-flavored root requirements files.
7. Carried backlog (unchanged): chromadb version alignment (client 1.4.1 vs native server 1.3.7 vs Somatic 1.5.1); retire `ubik-memory-sweep`; update ingestion loader; `test_logger.py` stderr-teardown flake.

## Session: 2026-07-23 02:10 — Node: Hippocampal (continuation)
**Goal:** Complete the "fix both now" recovery left open in the 00:30 entry (Docker was NOT recovered / vLLM not running), and re-confirm the CLAUDE.md audit on request.

**Docker — RECOVERED.**
- Prior restart had **hung**: backend log stuck on `waiting for electron to quit`; both `Docker Desktop` and `com.docker.backend` were actually DOWN (the 00:30 "HTTP 000" was a half-dead app, not a wedged VM).
- Fix: `pkill -9 -f "Docker Desktop"; pkill -9 -f "com.docker.backend"; open -a Docker`. Daemon came up clean.
- Verified serving (not just container-up):
  - **neo4j** 7474 → HTTP 200, container `Up (healthy)` ✓
  - **chromadb** 8001 `/api/v2/heartbeat` → 200 ✓ — **container shows `unhealthy` but that is a stale healthcheck still probing the deprecated `/api/v1` endpoint; the service is up.** (Cosmetic; worth fixing the compose healthcheck to hit v2.)
  - **open-notebook** container Up ✓
  - **mcp** 8080 → 000 — not a container; maestro-managed process, was skipped while docker was down. Not yet restarted (user redirected).

**CLAUDE.md audit — re-confirmed on request** (no change from 00:30 findings): two hard FAILs — **§2.3 Resilience** (no circuit breaker / no retry+jitter; canonical `mcp_client/` never imported; `base.py:290-318` constant-3s poll) and **§2.8 Code Org** (no `UbikError`; 94 bare `except Exception`; `base.py:353` is one). PARTIAL: §2.4 (no `SafeJSONFormatter`), §2.6 (no maestro tests/). PASS: §2.1, §2.2, §2.5. The `_wait_for_healthy` diagnostic blind spot remains the #1 endemic-vLLM fix.

**State left in:**
- **Docker: healthy** — neo4j + chromadb serving, open-notebook up. chromadb container flag cosmetically `unhealthy` (v1 healthcheck vs v2 service).
- **mcp (8080): not running** — safe to start now that docker dep is healthy.
- **vLLM: not running.** WSL VM still cleanly shut down, ready for `maestro start --service vllm`.
- No code changed this continuation (`.gitignore` from 00:30 remains the only code change of the day).

**Next session should:**
1. Start mcp, then `maestro start --service vllm` on the clean WSL VM; verify `/health` from Hippocampal.
2. Fix chromadb compose healthcheck to probe `/api/v2/heartbeat` (kills the false `unhealthy`).
3. Begin the hardening backlog — highest value first: `_wait_for_healthy` unit-death detection (endemic-vLLM fix A), then UBIK-own-venv, then §2.3 import of canonical resilience.
---

## Session: 2026-07-23 03:05 — Node: Hippocampal
**Goal:** Implement **Layer C** of HARDENING_PLAN_2026-07-23.md — give maestro out-of-band eyes on the vLLM unit so a death mid-load is reported instantly with the journal tail, instead of a silent 300s health-wait timeout. (Highest-value / lowest-risk layer; pure diagnostic, no launch-behavior change.)

**Completed:**
- Added `UbikService._liveness_diagnostic()` hook to `maestro/services/base.py` — default returns `None` (no behavior change for any other service).
- Wired it into `_wait_for_healthy`: after each unhealthy probe, consults the hook at most every `liveness_interval` (default 9.0s, new kwarg) so an SSH round-trip isn't paid every 3s poll; if the hook reports death, logs `"<name> died during startup wait (Ns): <journal>"` and returns `False` immediately.
- Overrode `_liveness_diagnostic` in `VllmService` (remote path only): one SSH call runs `systemctl --user is-active ubik-vllm`; returns `None` when active/activating (still loading) or when SSH is unreachable (never false-alarm), else the `journalctl --user -u ubik-vllm -n 30` tail (truncated 2000 chars). Privacy §2.4: journal is engine output, not memory content — documented in the docstring.
- New `maestro/tests/test_liveness.py` — 6 tests: abort-on-death (<5s wall, not 300s; diag logged), keep-waiting-while-activating, SSH-down→None, unit-active→None, dead→journal, not-remote→None (no SSH call). All green.

**Verification:**
- `pytest maestro/tests/test_liveness.py` → 6 passed.
- Regression: `test_shutdown + test_service_probes + test_orchestrator + test_remote` → 190 passed.
- `ast.parse` + import check on both changed modules OK; hook present on base class.
- ruff/mypy NOT run — neither installed in the DeepSeek venv (missing lint/type CI tooling is already on the hardening backlog; noted, not fixed here).

**Files changed:**
- `maestro/services/base.py`: +`_liveness_diagnostic` no-op hook; `_wait_for_healthy` gains `liveness_interval` kwarg + throttled early-abort-on-death.
- `maestro/services/vllm_service.py`: +`_liveness_diagnostic` remote override (systemctl is-active + conditional journal tail).
- `maestro/tests/test_liveness.py`: new (6 tests).
- `HARDENING_PLAN_2026-07-23.md`: (created earlier this session) the full 5-layer plan; Layer C now implemented.

**State left in:**
- Layer C DONE and green. No launch behavior changed — vLLM still starts via the transient `systemd-run` unit (Layer A not yet done); the new hook only *observes* it. Safe to deploy.
- Docker healthy (neo4j+chromadb serving; chromadb container flag cosmetically `unhealthy`). mcp + vLLM still not started.

**Next session should:**
- Layer A (persistent, enabled `ubik-vllm.service` with `Restart=on-failure`, rendered from config; retire transient `systemd-run`), then Layer B (Somatic Windows: systemd+linger, boot Scheduled Task, keepalive heartbeat — needs Windows admin). See HARDENING_PLAN_2026-07-23.md §3 for order.
---

## Session: [2026-07-23 21:42] — [Node: Hippocampal]
**Goal:** Implement Layer A of the hardening plan — replace the transient `systemd-run` vLLM unit with a persistent, enabled, self-healing systemd *user* unit rendered from config; verify with tests.
**Completed:**
- `maestro/services/vllm_service.py` (v0.7.0 → 0.8.0): added pure renderer `_render_vllm_unit(*, python, server, config, model, port, stop_grace_s)` producing the full unit file. All node-specific values are injected (no `/home/gasu`, no `pytorch_env` literals — §2.1). Unit carries the 4 Blackwell/SM120 `Environment=` lines (incl. `VLLM_USE_FLASHINFER_SAMPLER=0`), `Restart=on-failure` / `RestartSec=10` / `StartLimitBurst=4` (self-healing with storm cap), and `KillSignal=SIGTERM` + `KillMode=mixed` + `TimeoutStopSec=90` (graceful VRAM release).
- Rewrote `_remote_start` to **install-then-start**: renders the unit, writes it under `$HOME/.config/systemd/user/ubik-vllm.service` via a quoted heredoc (`<<'__UBIK_VLLM_UNIT__'` — no bash expansion of the unit body), sha256-compares to skip a needless `daemon-reload`, then `reset-failed` / `enable` / `restart`. The transient `systemd-run --user` path is retired. `MISSING_` prerequisite and SSH-down results still short-circuit (return False) before `_wait_for_healthy`.
- Confirmed `load_config()` in `somatic/inference/vllm_server.py` tolerates a missing config file (merges defaults) → `--config` is passed unconditionally, keeping the renderer a pure function.
**Verification:**
- `bash -n` on the generated install script → BASH SYNTAX OK; rendered unit inspected line-by-line.
- New `maestro/tests/test_vllm_unit.py` — 9 tests (5 renderer: config values used / no hardcoded literals / all 4 Blackwell env / self-healing+VRAM directives / trailing newline; 4 `_remote_start`: installs+enables+restarts & no `systemd-run`, MISSING_ short-circuit, SSH-down short-circuit, healthy path awaits `_wait_for_healthy`). All green.
- Updated the now-stale `test_remote.py::test_remote_start_drives_vllm_server_wrapper` assertion to the persistent-unit mechanism (`systemctl --user enable/restart ubik-vllm`, `Restart=on-failure`, `systemd-run` absent). WhisperX test left untouched (WhisperX still uses transient `systemd-run` — not part of Layer A).
- Full maestro suite (excluding known `test_logger.py` stderr-teardown flake): **608 passed, 0 failed**.
- ruff/mypy still NOT run — not installed in the DeepSeek venv (lint/type CI tooling remains on the backlog).
**Files changed:**
- `maestro/services/vllm_service.py`: +`_render_vllm_unit` renderer; `_remote_start` rewritten to install+enable+restart a persistent unit; version 0.8.0.
- `maestro/tests/test_vllm_unit.py`: new (9 tests).
- `maestro/tests/test_remote.py`: updated one vLLM assertion for the persistent-unit mechanism.
**State left in:**
- Layer A DONE and green. vLLM now launches as a persistent, enabled, self-healing unit — survives service crash (Restart=on-failure) and, once installed, survives a maestro restart. NOT yet field-tested against a live Somatic node (Somatic/WSL not started this session); tests are unit-level only.
- Not deployed to Somatic yet: next real vLLM start via maestro will install the unit.
- Docker: neo4j+chromadb serving; mcp + vLLM not started.
**Next session should:**
- Field-test Layer A: start vLLM via maestro against the live Somatic node, confirm `systemctl --user status ubik-vllm` shows enabled+active and that a `kill` of the process triggers Restart. Then Layer B (Somatic Windows: `[boot] systemd=true`, `loginctl enable-linger`, boot Scheduled Task, keepalive heartbeat — needs Windows admin). See HARDENING_PLAN_2026-07-23.md §3 for order.
---

## Session: [2026-07-23 21:52] — [Node: Hippocampal]
**Goal:** Commit and push the Layers A+C hardening work.
**Completed:**
- Staged the coherent Layer A+C set only (code + both new test files + the updated `test_remote.py` assertion + SESSIONS.md + HARDENING_PLAN_2026-07-23.md + the privacy `.gitignore` hardening). Left unrelated untracked docs out (`DECISION_*.md`, `MAESTRO-0.12.0.md`, `START_HERE.md`, `UBIK_Claude_Prompts/`, `.claude/plans/`).
- Committed to `master` as `d0af25e` — "maestro: Layers A+C — persistent self-healing vLLM unit + startup liveness eyes". `sync_sessions` hook synced SESSIONS.md → Google Drive on commit.
- Pushed to `origin/master` (github.com/gasu04/UBIK): `d2d0751..d0af25e`.
**State left in:**
- Layers A+C committed and pushed. Working tree still holds unrelated untracked docs (intentionally uncommitted) and unrelated untracked data dirs (gitignored).
- Layer A remains unit-tested only — NOT field-tested against a live Somatic node.
- Docker: neo4j+chromadb serving; mcp + vLLM not started.
**Files changed:**
- (none new — this entry documents the commit/push of the prior entry's work.)
**Next session should:**
- Field-test Layer A against the live Somatic node (start vLLM via maestro; confirm `systemctl --user status ubik-vllm` enabled+active; verify a process kill triggers Restart). Then Layer B (Somatic Windows: systemd=true, enable-linger, boot Scheduled Task, keepalive — needs Windows admin). See HARDENING_PLAN_2026-07-23.md §3.
---

## Session: 2026-07-23 19:51 — [Node: Somatic]
**Goal:** Implement Layer B of `HARDENING_PLAN_2026-07-23.md` — keep the WSL VM itself alive and auto-starting at the Windows-host level, independent of whatever is running inside it (Layers A/C, done separately on Hippocampal same day, cover in-VM vLLM recovery).
**Completed:**
- Audited current state before changing anything: confirmed B.1 (systemd + linger) and most of B.4 (`.wslconfig`) were **already correctly configured** on this box — `/etc/wsl.conf` already has `systemd=true` + `default=gasu`, `loginctl show-user gasu` already reports `Linger=yes`, and `C:\Users\gasu.Adrian\.wslconfig` already has `vmIdleTimeout=-1` and `networkingMode=NAT`. No edits needed for those.
- Built `somatic/windows/` (B.2 + B.3 deliverables): `ubik-wsl-boot.ps1` (boots the VM on Windows startup via a Task Scheduler boot trigger), `ubik-wsl-keepalive.ps1` (2-minute heartbeat that logs and re-boots the VM on failure — targets the `p9io`/`CheckConnection` teardown bug), matching Task Scheduler XML definitions (`LogonType=S4U`, `RunLevel=HighestAvailable`, no stored password), and a README with exact `schtasks`/XML-import install commands and the plan's 4-point verification checklist. Did **not** create the actual Windows Scheduled Tasks, edit `.wslconfig`, or run `wsl --shutdown` myself — those need Windows admin at the machine, and `wsl --shutdown` would have killed the running session. Committed as `b5942b0` (later rebased to `fdb7cc2`, see below).
- Initially misread this node's uncommitted `maestro/` deletion as a gap/regression, until re-reading `SESSIONS.md` from GitHub surfaced the 2026-07-04(b) Hippocampal entry: the deletion was **intentional** (maestro moved to Hippocampal-only, SSH-driven remote control of both nodes; Somatic never needed its own copy) — not a bug to fix.
- **Discovered local was 45 commits behind `origin/master`** (this checkout hadn't pulled since before the 2026-07-04(b) remote-control switch and all the way through today's Layer A+C work and `HARDENING_PLAN_2026-07-23.md`), while also having diverged with 3 local-only, never-pushed commits. Reconciled via rebase: stashed all uncommitted WIP (the intentional `maestro/` deletion, `vllm_config.yaml`/`.bashrc`/`whisperx_server.py` edits, assorted untracked cruft), restored `maestro/` from HEAD to unblock the rebase, `git rebase origin/master`, skipped local commit `5766602` (2026-07-15 session entry — confirmed byte-identical to one origin already folded in via commit `edabee7`), then popped the stash back. One real conflict, in `config/models/vllm_config.yaml`: both sides had already independently landed `gpu_memory_utilization: 0.93`; kept origin's version (it also carries an explanatory comment). `somatic/whisperx_server.py`'s stashed edit turned out byte-identical to origin's already-committed version (env-driven device/compute-type) — applied as a no-op.
- Local `master` is now `fdb7cc2`, 2 commits ahead of `origin/master`, 0 behind (the `.bashrc` dotfiles-symlink commit and the Layer B commit, replayed cleanly on top of all 45 origin commits including Layer A/C and the hardening plan).
**State left in:**
- `maestro/` is back on disk with the full Layer A implementation (persistent `ubik-vllm` unit, `Restart=on-failure`) — this Somatic checkout can now see it for reference, though per the 2026-07-04(b) decision it doesn't need to run it locally.
- `somatic/windows/` Scheduled Tasks are **not yet installed** on the Windows host — install is a manual next step per the new README.
- Not yet pushed to `origin/master` — local is 2 commits ahead, clean fast-forward available.
- Still uncommitted, left untouched (predate this session, not part of this task): `dotfiles/.bashrc` (added an `opencode` PATH export), and untracked cruft (`,`, `0q,`, `c,`, `\`, `scratchpad-trinity/`, `scripts/sync_from_github.sh`, `somatic/whisperx_server.py.bak.20260719`).
**Files changed:**
- `somatic/windows/ubik-wsl-boot.ps1`, `ubik-wsl-keepalive.ps1`, `ubik-wsl-boot-task.xml`, `ubik-wsl-keepalive-task.xml`, `README.md`: new — Layer B deliverables
- `SESSIONS.md`: this entry
**Next session should:**
1. Push `fdb7cc2` (local is ahead of `origin/master` by 2 commits, clean fast-forward, not yet done pending user confirmation).
2. Install the two Scheduled Tasks on the Somatic Windows host per `somatic/windows/README.md`, then run Layer B's 4-point verification checklist — now unblocked since Layer A's `ubik-vllm` unit exists again (via `maestro/`, driven from Hippocampal).
3. Decide on the small pending `dotfiles/.bashrc` edit (opencode PATH) and the long-standing untracked cruft at repo root — neither is part of Layer B and both were left as-is.
---
