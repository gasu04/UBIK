# CLAUDE.md — UBIK Coding Standards

**Version:** 3.1.0 **Supersedes:** 3.0.0, 2.0.0 (January 2025) **Scope:** Coding rules for all Claude Code sessions in the UBIK repository **Re-read from disk at every session start**

---

## Core Philosophy

**"Build systems that survive their creators."**

Four qualities matter above all:

- **Resilience** — graceful degradation over catastrophic failure
- **Clarity** — code that explains itself
- **Portability** — run anywhere, depend on nothing specific
- **Maintainability** — future readers (including you in six months) are users too

---

## How to Use This Document

This file has three layers, and they have an explicit order of precedence:

1. **Mandatory Baseline (Part 2)** — non-negotiable rules, each marked \[MANDATORY\]. These exist because UBIK has already been hurt by violating them. If a rule here conflicts with an apparent simplification, the rule wins.
2. **Operating Principles (Part 1)** — govern *how* Claude approaches work within that baseline. When the baseline doesn't explicitly require something, these principles apply.
3. **Reference Material (Parts 3–4)** — concrete values (ports, paths, file layouts) and session-management tips.

**Precedence rule:** When Part 1 and Part 2 appear to conflict, Part 2 wins. Example: Simplicity First's "no error handling for impossible scenarios" does not override the Mandatory Baseline requirement to wrap every external service call with a circuit breaker. Likewise, Surgical Changes' "don't touch adjacent code" does not excuse leaving a module without a docstring when §2.5 requires one.

**Honest tradeoff:** §2.5 (documentation) and §2.6 (testing) are strict. They will sometimes feel heavier than §1.2 Simplicity First would call for. Simplicity First governs the *implementation* — the code you write to solve the problem. Documentation and test thoroughness are part of the baseline and override simplicity when they conflict. This is deliberate: thorough docstrings and tests are the lowest-risk way for a less-experienced coder to catch mistakes before they ship.

---

## Table of Contents

- [Part 1 — The Four Operating Principles](#part-1--the-four-operating-principles)
  - [1.1 Think Before Coding](#11-think-before-coding)
  - [1.2 Simplicity First](#12-simplicity-first)
  - [1.3 Surgical Changes](#13-surgical-changes)
  - [1.4 Goal-Driven Execution](#14-goal-driven-execution)
- [Part 2 — Mandatory Baseline](#part-2--mandatory-baseline)
  - [2.1 Configuration](#21-configuration-mandatory)
  - [2.2 Async-First](#22-async-first-mandatory)
  - [2.3 Resilience](#23-resilience-mandatory)
  - [2.4 Privacy Logging](#24-privacy-logging-mandatory)
  - [2.5 Documentation](#25-documentation-mandatory)
  - [2.6 Testing](#26-testing-mandatory)
  - [2.7 Security](#27-security-mandatory)
  - [2.8 Code Organization](#28-code-organization-mandatory)
- [Part 3 — UBIK Reference](#part-3--ubik-reference)
- [Part 4 — Session Management](#part-4--session-management)
- [Appendix A — Quick Reference Checklists](#appendix-a--quick-reference-checklists)
- [Appendix B — What Changed from v2.0](#appendix-b--what-changed-from-v20)

---

## Part 1 — The Four Operating Principles

These govern Claude's process for any task not otherwise specified by Part 2.

### 1.1 Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before writing implementation:

- State assumptions explicitly. If uncertain, ask.
- If multiple reasonable interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop and name what's unclear. Ask.

**Why this matters for UBIK:** architecture decisions here have 20-year consequences. A wrong interpretation compounds.

### 1.2 Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked — unless required by Part 2.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

Ask: *"Would a senior engineer say this is overcomplicated?"* If yes, simplify.

### 1.3 Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting — unless Part 2 explicitly requires it (e.g., §2.5 module docstrings must be added if missing).
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

- Remove imports, variables, or functions that *your* changes made unused.
- Don't remove pre-existing dead code unless asked.

**The test:** every changed line should trace directly to the user's request, or to a Part 2 mandate.

### 1.4 Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform imperative tasks into verifiable goals:

| Instead of... | Transform to... |
|:---:|:---:|
| "Add validation" | "Write tests for invalid inputs, then make them pass" |
| "Fix the bug" | "Write a test that reproduces it, then make it pass" |
| "Refactor X" | "Ensure tests pass before and after" |

For multi-step tasks, state a brief plan before acting:

1. \[Step\] → verify: \[check\]
2. \[Step\] → verify: \[check\]
3. \[Step\] → verify: \[check\]

Strong success criteria let Claude loop independently. Weak criteria ("make it work") require constant clarification.

---

## Part 2 — Mandatory Baseline

Every rule in this part is \[MANDATORY\]. They hold regardless of task size or apparent simplicity.

### 2.1 Configuration \[MANDATORY\]

**No hardcoded values in application code. Ever.**

- All network addresses (IPs, hostnames) come from environment variables or a typed settings object.
- All ports come from environment variables.
- All credentials come from environment variables or a secrets manager.
- All timeouts, retry counts, and thresholds come from configuration.

Required artifacts in every subproject:

- `.env.example` — template listing every variable, with safe defaults and comments
- `.env` — actual values (gitignored)
- `settings.py` (or equivalent) — typed configuration loader (Pydantic BaseSettings or `@dataclass` with `field(default_factory=...)`)

**Self-test:** `grep -rE "\b100\.[0-9]+\.[0-9]+\.[0-9]+\b" <source_dir>/*.py` should return nothing from application code. Hardcoded Tailscale IPs are the specific bug this rule exists to prevent — Phase 3 v2.1 flagged this as mandatory after finding them in production code.

### 2.2 Async-First \[MANDATORY\]

When calling external services (vLLM, MCP, databases, HTTP APIs) from async code:

- Use **async clients**: `AsyncOpenAI` (not `OpenAI`), `httpx.AsyncClient` (not `httpx.Client`), `asyncpg` (not `psycopg2`). Never the sync variant inside an async function — it blocks the event loop and freezes the entire service.
- On connection error: **explicitly `aclose()`** the client and recreate it. Never reuse a client that may have corrupted connection state. A boolean flag (`_connected = False`) is not sufficient.
- Use `@asynccontextmanager` for any resource with a lifecycle (connections, sessions, transactions).

**Self-test for non-blocking:** if an external call takes 2 seconds, an unrelated background coroutine must continue making progress during that window. If a counter incremented every 100ms doesn't advance, you're blocking. See `~/ubik/somatic/tests/test_async_not_blocking.py` for the canonical pattern.

### 2.3 Resilience \[MANDATORY\]

Every external service call must be wrapped with:

1. **Circuit Breaker with Probe Latch** — in HALF_OPEN state, exactly one probe request is allowed through. All others are rejected until the probe resolves. Naïve circuit breakers that let all HALF_OPEN traffic through cause thundering-herd failures.
2. **Retry with exponential backoff + jitter** — `delay = min(base * 2^attempt, max) + random_jitter`. The jitter is not optional; it prevents synchronized retry storms.
3. **Proper time source** — `time.monotonic()` for elapsed-time measurements, never `time.time()` (which can jump backward on NTP sync).
4. **Thread safety** — circuit breakers shared across coroutines need `asyncio.Lock`.
5. **Connection invalidation on error** — see §2.2.
6. **Health checks** — every service exposes a health endpoint reporting per-component status (healthy / degraded / unhealthy) with a timestamp.

**Reference implementation:** `~/ubik/somatic/mcp_client/` is the source of truth. The files below contain the canonical, tested versions. Do not reimplement these from memory — import them.

| Concern | Module |
|:---:|:---:|
| Circuit breaker (Probe Latch) | `mcp_client/circuit_breaker.py` |
| Retry + backoff + jitter | `mcp_client/resilience.py` |
| Connection lifecycle | `mcp_client/connection.py` |
| Composed client | `mcp_client/client.py` (HippocampalClientV2) |

If Claude is asked to "add a circuit breaker" or "implement retry" in a new module, the correct move is: import from the existing module. Do not write a new one.

### 2.4 Privacy Logging \[MANDATORY\]

UBIK handles personal memory content — letters, therapy sessions, family conversations, journal entries. This data must never appear in logs. Logs may be persisted indefinitely, shipped to monitoring systems, and viewed by debugging tools.

**Never log:**

- Memory content (episodic or semantic)
- Query text from the user
- Context strings passed to the model
- Full prompts or full model responses

**Do log:**

- Memory IDs (e.g., `ep_20240115_103000_123456`)
- Similarity / relevance scores
- Retrieval counts
- Timing metrics (milliseconds)
- Error types and classes — not error details that may embed content
- Token counts

Use a JSON formatter that actively redacts known-sensitive field names (`content`, `text`, `query`, `context`, `prompt`, `response`). See `~/ubik/somatic/logging_config.py` for the canonical `SafeJSONFormatter`.

**Self-test before merging:** run a sample query through the pipeline with logging at DEBUG, then grep the log file for any memory content or query text. Any hit is a bug.

### 2.5 Documentation \[MANDATORY\]

Every module must have a header docstring covering:

- What this module does
- How it fits into the larger system
- Key classes / functions it provides
- A usage example
- Dependencies (with version constraints where they matter)
- **Tier classification:** Tier 1 (silent-failure-critical, 100% coverage) or Tier 2 (standard, 80% coverage). For Tier 1, state *why* in one sentence. See §2.6 and §3.4.1.

Every public function, method, and class must have a full Google-style docstring:

- **Args:** each argument with type and meaning
- **Returns:** what comes back
- **Raises:** each exception that can propagate
- **Example:** at least one concrete invocation, for public API functions
- **Note:** any non-obvious behavior (caching, side effects, thread safety)

Private helpers (leading underscore) may use a one-line docstring if their behavior is self-evident from the signature and their caller is in the same module.

### 2.6 Testing \[MANDATORY\]

**Tier-based coverage — defined by failure mode, not module importance.**

- **Tier 1 (100% coverage):** code whose *silent* failure causes harm that can't be rolled back. Tests must specifically probe the quiet failure modes, not just the loud ones. See §3.4.1 for the canonical Tier 1 registry.
- **Tier 2 (80% minimum):** everything else. Loud failures (exceptions, None returns, failing health checks) are their own warning system; tests cover happy path plus obvious error branches.

**The operational test:** *"If this fails silently at 2am, can I tell? And can I undo the damage?"* If the answer is no-and-no, it's Tier 1. If failure is noisy or recoverable, it's Tier 2. Don't let Tier 1 sprawl — scope creep here is how test maintenance becomes a drag on shipping.

**Other requirements:**

- **Error handling:** explicitly tested — if a code path raises, there must be a test that exercises that raise
- **Async tests:** use pytest-asyncio with properly scoped fixtures
- **Naming:** `test_<function>_<scenario>_<expected>()` — descriptive names serve as documentation

Test structure:

```
tests/
├── unit/           # fast, isolated, no network
├── integration/    # cross-module, may touch local services
├── e2e/            # full pipeline tests
├── fixtures/       # sample data, mock responses
└── conftest.py
```

### 2.7 Security \[MANDATORY\]

- **Secrets** via environment variables or a secrets manager. Never in source control.
- **Input validation** via Pydantic models or equivalent. Never trust caller-supplied strings.
- **Dependency scanning** in CI (`pip-audit` or `safety`).
- **Pinned versions** in production (`requirements.lock` from `pip freeze`).
- **Non-root users** in containers.
- **No sensitive data** in URL parameters, query strings, or filenames.

### 2.8 Code Organization \[MANDATORY\]

- **Single responsibility:** one class / function, one reason to change.
- **Dependency injection:** construct dependencies outside the class that uses them. Never `self.db = PostgresDatabase()` inside `__init__`.
- **Type hints everywhere:** all function signatures, all class attributes, all return types. mypy must pass in CI.
- **Custom exception hierarchy:** rooted at a project-specific base (e.g., `UbikError`). Never raise bare `Exception`. Never catch bare `Exception` except at top-level request handlers.
- **Explicit exports:** every `__init__.py` defines `__all__` and exports the public API.

---

## Part 3 — UBIK Reference

### 3.1 Ports & Hosts — Canonical Assignments

These are the final values from Phase 3 v2.1. In code they are loaded from env vars; this table is for human reference only.

| Service | Node | Port | Env Var | Notes |
|:---:|:---:|:---:|:---:|:---:|
| ChromaDB | Hippocampal | 8001 | `HIPPOCAMPAL_CHROMA_PORT` | Vector store |
| MCP Server | Hippocampal | 8080 | `HIPPOCAMPAL_MCP_PORT` | Memory tool interface |
| Neo4j Bolt | Hippocampal | 7687 | — | Graph protocol |
| Neo4j HTTP | Hippocampal | 7474 | — | Browser UI |
| vLLM Server | Somatic | **8002** | `VLLM_PORT` | OpenAI-compatible; **moved from 8001 to resolve conflict with ChromaDB** |

Node addresses come from Tailscale. Use hostnames (`ubik-hippocampal`, `ubik-somatic`) in config, not IPs. If Tailscale hostnames are unavailable, use `TAILSCALE_HIPPOCAMPAL_IP` and `TAILSCALE_SOMATIC_IP` env vars — never literal `100.x.x.x` in code.

### 3.2 Project Structure

```
project_root/
├── config/              # .env, .env.example, settings.py
├── src/ or <module>/    # source code
├── tests/               # unit/, integration/, e2e/, fixtures/
├── scripts/             # setup.sh, health_check.py, run_service.sh
├── docs/                # architecture.md, api.md, deployment.md
├── logs/                # gitignored
├── data/                # gitignored
├── requirements.txt
├── pyproject.toml
├── README.md
└── CLAUDE.md            # this file
```

### 3.3 Naming Conventions

| Type | Convention | Example |
|:---:|:---:|:---:|
| Files | snake_case | `mcp_client.py` |
| Classes | PascalCase | `HippocampalClient` |
| Functions | snake_case | `get_rag_context()` |
| Constants | SCREAMING_SNAKE | `MAX_RETRIES = 3` |
| Private members | Leading underscore | `_internal_state` |
| Config vars (.env) | SCREAMING_SNAKE | `HIPPOCAMPAL_HOST` |

### 3.4 Where the Canonical Code Lives

Do not reimplement these. Import them.

| Concern | Location |
|:---:|:---:|
| Circuit Breaker (Probe Latch) | `~/ubik/somatic/mcp_client/circuit_breaker.py` |
| Retry + backoff + jitter | `~/ubik/somatic/mcp_client/resilience.py` |
| Connection lifecycle | `~/ubik/somatic/mcp_client/connection.py` |
| Unified MCP client | `~/ubik/somatic/mcp_client/client.py` |
| Safe JSON logger | `~/ubik/somatic/logging_config.py` |
| Settings loader (Pydantic) | `~/ubik/somatic/config/settings.py` |
| Non-blocking self-test | `~/ubik/somatic/tests/test_async_not_blocking.py` |

#### 3.4.1 Tier 1 Critical-Path Registry

These modules share one property: **failure is silent, and damage compounds before anyone notices.** They require 100% coverage per §2.6, and tests must deliberately probe the quiet failure modes — not just confirm the happy path works.

| # | Concern | Location | Why Tier 1 (silent-failure mode) |
|:---:|:---:|:---:|:---:|
| 1 | Memory writes (`store_episodic`, `store_semantic`, `update_identity_graph`) | `~/ubik/hippocampal/mcp_server.py` | Corrupted or mis-tagged writes can't be rolled back; bad data compounds in retrieval forever |
| 2 | Reasoning-chain stripping (`strip_reasoning`) | `~/ubik/somatic/prompts/parser.py` | A silent leak contaminates voice output without anyone noticing until a family member reads it |
| 3 | Voice Freeze state machine (`freeze_semantic`, `unfreeze_semantic`) | `~/ubik/hippocampal/mcp_server.py` | A silent unfreeze during post-training writes poisons the trained voice and is invisible in logs |
| 4 | Privacy redaction (`SafeJSONFormatter`) | `~/ubik/somatic/logging_config.py` | Silent failure = memory content lands in persisted logs; discovered only by manual audit |
| 5 | Ingestion deduplication (source-document-ID) | *(planned — tag location once built on Hippocampal Node)* | Silent duplicates skew retrieval probabilities; bias discovered years later, if at all |
| 6 | Circuit Breaker Probe Latch (`allow_request`) | `~/ubik/somatic/mcp_client/circuit_breaker.py` | Silent loss of the one-probe invariant = thundering herd when the Hippocampal Node recovers |

**Adding a new Tier 1 entry:** the proposing module's header docstring must declare the tier (§2.5), state the silent-failure reason in one sentence, and there must be at least one test that deliberately triggers the silent failure mode (not just the loud one). Changes to this registry require updating CLAUDE.md itself — it is load-bearing.

**Removing an entry:** only if the underlying failure mode has been made loud (e.g., the code now raises, fails a health check, or is monitored by an alert). Shrinking the registry through re-engineering the code is encouraged. Shrinking it by weakening the definition of "critical" is not.

### 3.5 Cross-Platform & Docker

- Use `pathlib.Path` for all filesystem paths. Never string concatenation, never hardcoded `/home/gasu/...` or `C:\Users\...`.
- Detect OS via `platform.system()` when platform-specific logic is unavoidable. UBIK runs on macOS (Hippocampal, Mac Mini M4) and Linux/WSL2 (Somatic).
- Docker: multi-stage builds, non-root user, explicit `HEALTHCHECK` directive, pinned base image digest.

---

## Part 4 — Session Management

### 4.1 Model Selection

CLAUDE.md is re-read from disk at the start of every session.

| Task type | Model |
|:---:|:---:|
| Bug fixes, unit tests, documentation | Sonnet 4.6 |
| Refactoring core architecture, complex race conditions | Opus 4.6 |

Override per-session: `/model <alias|name>` (e.g., `/model opus`).

### 4.2 Configuration Priority

Highest to lowest:

1. Session command — `/model <alias|name>`
2. CLI flag — `claude --model <alias|name>`
3. Environment variable — `ANTHROPIC_MODEL`
4. Settings file — `~/.claude/settings.json` → `"model": "..."`

### 4.3 Context Window

Run `/compact` every 20–30 turns to keep context lean.

`/compact` affects: the current session's conversation history (summarized and compressed).

`/compact` does **not** affect:

- `CLAUDE.md` — always re-read fresh from disk at session start
- `~/.claude/projects/*/memory/` — file-based, persist independently
- File edits already written to disk

---

## Appendix A — Quick Reference Checklists

### File Creation

- [ ] Module docstring (purpose, usage, dependencies)
- [ ] Type hints on every function signature
- [ ] Configuration loaded from env (no hardcoded values)
- [ ] Custom exceptions for error cases
- [ ] Context manager for any resource with a lifecycle
- [ ] Health check exposed (if a service)
- [ ] Test file created alongside

### Code Review

- [ ] No hardcoded values (grep confirms)
- [ ] Async client used inside async functions (`AsyncOpenAI`, `httpx.AsyncClient`)
- [ ] Circuit breaker + retry wrap every external call
- [ ] Logs contain no memory content, queries, or context strings
- [ ] Resources explicitly `aclose()`'d on error
- [ ] Type hints accurate, mypy passes
- [ ] Docstrings complete (Args / Returns / Raises)
- [ ] Tests cover error paths, not just happy path

### Performance

- [ ] Lazy loading for expensive resources (`@property` pattern)
- [ ] Connection pooling enabled (`httpx.Limits`, `asyncpg.create_pool`)
- [ ] Caching present where warranted (`@lru_cache`, TTL cache)
- [ ] Circuit breaker on every external call
- [ ] `AsyncOpenAI` (not `OpenAI`) in async paths
- [ ] No N+1 query patterns

### Before Declaring a Feature Done

- [ ] All tests pass (`pytest`)
- [ ] mypy passes
- [ ] ruff + black pass
- [ ] Logs reviewed at DEBUG level — no sensitive fields leak
- [ ] Success criteria from the original request all verified (§1.4)
- [ ] CLAUDE.md consulted for anything that might apply

---

## Appendix B — What Changed from v2.0

| Area | v2.0 | v3.0 |
|:---:|:---:|:---:|
| Circuit breaker example code | Original (broken: no Probe Latch, time.time(), no lock) | Removed; points to `mcp_client/circuit_breaker.py` as source of truth |
| Retry code | No jitter | Jitter is now an explicit mandatory element (§2.3) |
| Async vs sync clients | Not addressed | §2.2 mandates AsyncOpenAI, aclose() on error |
| Logging privacy | Generic structured logging | §2.4 explicitly prohibits logging memory content |
| Hardcoded IPs | Implicit | §2.1 explicitly forbids; grep-testable self-test |
| Karpathy Principles | §11 (appended) | Part 1 (structural); precedence rule kept and tightened |
| Code examples | ~800 lines of inline Python | Removed; canonical implementations referenced |
| Testing strictness | 80% coverage / 100% critical | Retained — explicit conflict note with Simplicity First |
| Docstring strictness | Full Google style everywhere | Retained — with a carve-out for trivial private helpers |
| Length | ~1,200 lines | ~380 lines |

---

## Version History

| Version | Date | Changes |
|:---:|:---:|:---:|
| 3.1.0 | 2026-04 | §2.6 rewritten around tier-based coverage defined by failure mode, not module importance; §3.4.1 Tier 1 Critical-Path Registry added (6 initial entries); §2.5 requires module docstrings to declare tier classification |
| 3.0.0 | 2026-04 | Full restructure: four principles woven structurally (not appended); resilience mandates aligned with Phase 3 v2.1 (Probe Latch circuit breaker, jitter, async-first, connection hygiene); privacy logging rule added; verbose code examples removed in favor of canonical module references; ~70% shorter |
| 2.0.0 | 2025-01 | Derived from UBIK project learnings; Karpathy Principles appended as §11 |
| 1.0.0 | 2024-01 | Initial release |

---

*"The best code is no code at all. The second best is code so clear it documents itself."*

---

## Session Journaling (MANDATORY)

**SESSIONS.md** is the canonical session log for all UBIK and DeepSeek work.
It syncs to Google Drive automatically via `scripts/sync_sessions.sh` on every commit that touches it.
GdriveMirror path: `/Volumes/GdriveMirror/Ubik/SESSIONS.md`

At the START of every session:
- Read the last 20 lines of `SESSIONS.md` (at the UBIK project root)
- State what you found before proceeding

At the END of every session (or when asked to wrap up):
- Append a structured entry to `SESSIONS.md` with this exact format:

## Session: [YYYY-MM-DD HH:MM] — [Node: Hippocampal|Somatic]
**Goal:** [what was attempted]
**Completed:**
- [bullet per thing actually done]
**State left in:**
- [what is running, what is broken, what is half-done]
**Files changed:**
- [path]: [one-line description of change]
**Next session should:**
- [the most important thing to do next]
---
