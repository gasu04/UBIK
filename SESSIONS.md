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
