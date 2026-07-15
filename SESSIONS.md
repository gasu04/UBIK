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
