# CLAUDE.md - DeepSeek RAG Project Rules

This project follows the global coding standards defined in
`/Volumes/990PRO 4T/Claude/CLAUDE.md`.

---

## Project-Specific Rules

### Virtual Environment
- The `venv/` directory is **shared with other projects** — do NOT modify,
  recreate, or delete it.
- To install new dependencies: `venv/bin/pip install <package>` and then
  update `requirements_google_drive.txt` and `pyproject.toml`.

### Configuration
- All tuneable values live in `config/settings.py` as a dataclass.
- Never hardcode URLs, model names, paths, or API keys in source files.
- Copy `config/.env.example` → `config/.env` (or root `.env`) to override defaults.

### Secrets
- `credentials.json` and `token.json` are Google OAuth files — **never commit**.
- They are listed in `.gitignore`. Keep it that way.

### Paths
- Use `Path(__file__).parent` for paths relative to the current file.
- Never hardcode absolute paths (e.g., `/Volumes/990PRO 4T/...`).
- The embeddings directory is controlled by `EMBEDDINGS_BASE_DIR` env var.

### Tests
- Tests live in `tests/unit/` and `tests/integration/`.
- Integration tests require `token.json` (real credentials) and network access.
- Run only unit tests in CI: `pytest tests/unit/ -v`
- Run integration tests manually: `pytest tests/integration/ -v`

### Scripts
- Operational helper scripts (e.g., `free_memory.sh`) live in `scripts/`.

---

## Session Journaling (MANDATORY)

At the START of every session:
- Read the last 20 lines of SESSIONS.md to understand what was last done
- State what you found before proceeding

At the END of every session (or when asked to wrap up):
- Append a structured entry to SESSIONS.md with this exact format:

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
