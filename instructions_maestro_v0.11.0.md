# Maestro v0.11.0 — Instructions

UBIK Maestro is the infrastructure orchestrator for the UBIK cluster.
It monitors and manages Neo4j, ChromaDB, MCP, vLLM, Tailscale, and Docker
across two nodes:

- **Hippocampal** — Mac Mini M4 Pro (100.103.242.91): Neo4j, ChromaDB, MCP, Docker
- **Somatic** — PowerSpec RTX 5090 / WSL2 (100.92.95.39): vLLM inference

Configuration is loaded from `{UBIK_ROOT}/maestro/.env` (see `.env.example`).

---

## Syntax

```
maestro [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]
```

> Global options must come **before** the subcommand.

### Global Options

| Option | Description |
|--------|-------------|
| `--version` | Print maestro version and exit |
| `--config PATH` | Load a custom `.env` file instead of the default |
| `--log-level LEVEL` | Override `MAESTRO_LOG_LEVEL` for this invocation (DEBUG, INFO, WARNING, ERROR) |
| `-h, --help` | Show help |

---

## Commands

### `status` — One-shot health check

Probes all services concurrently and prints a table.
Exit code: `0` = healthy, `1` = degraded/unhealthy.

```bash
maestro status                          # full cluster check
maestro status --json                   # raw JSON output
maestro status --verbose                # include details dict per service
maestro status --service vllm           # probe one service only
maestro status --service neo4j --service chromadb   # probe multiple
maestro status --timeout 5              # custom per-check timeout (seconds)
```

---

### `start` — Start local services

Starts all unhealthy local services in dependency order, or a single named service.

```bash
maestro start                           # start all unhealthy local services
maestro start --service vllm            # start one specific service
maestro start --service neo4j
maestro start --timeout 15              # custom probe timeout
```

---

### `shutdown` — Stop local services

Graceful ordered stop (reverse dependency order), escalating to SIGKILL per
service if the graceful stop doesn't complete within 30 seconds.

```bash
maestro shutdown                        # orderly shutdown (default)
maestro shutdown --dry-run              # show what would stop, without stopping
maestro shutdown --emergency            # SIGKILL all local UBIK processes immediately
```

---

### `watch` — Continuous monitoring

Default (no flags): Rich TUI refreshing every interval seconds.

```bash
maestro watch                           # live TUI (default interval from config)
maestro watch --interval 60             # refresh every 60 seconds
maestro watch --once                    # one structured probe cycle, then exit
maestro watch --auto-restart            # daemon mode: auto-restart unhealthy services
maestro watch --auto-restart --interval 120
```

---

### `dashboard` — Interactive per-node dashboard

Split-panel TUI showing Hippocampal and Somatic services side by side
with colour-coded health indicators.

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Force immediate refresh |
| `s` | Start all unhealthy local services |
| `x` | Shutdown all services |

```bash
maestro dashboard                       # auto-refresh every 30 s (default)
maestro dashboard --refresh 60          # custom refresh interval
maestro dashboard --timeout 5           # custom per-check timeout
```

---

### `health` — Status + metrics combined

Runs a full service health check and usage metrics collection concurrently,
then prints both reports. Useful for a comprehensive snapshot.

Exit codes: `0` = all healthy, `1` = some services down, `2` = error.

```bash
maestro health                          # combined health + metrics report
maestro health --json                   # JSON output (status + metrics)
maestro health --timeout 5
```

---

### `metrics` — Usage statistics

Collects ChromaDB collection sizes, Neo4j graph counts, vLLM state,
GPU utilisation (Somatic only), and disk usage. All values are best-effort;
unavailable metrics are shown as N/A.

```bash
maestro metrics
```

---

### `logs` — Tail the maestro log

Reads from `{UBIK_ROOT}/logs/maestro/maestro.log`.

```bash
maestro logs                            # last 50 lines (default)
maestro logs --lines 100                # custom line count
maestro logs --follow                   # stream new entries in real time (Ctrl+C to stop)
maestro logs -n 20 -f                   # short flags
```

---

### `check` — Alias for `status`

```bash
maestro check
```

---

## Common Workflows

```bash
# Check cluster health
maestro status

# Start everything after a reboot
maestro start

# Full snapshot (health + GPU/DB metrics)
maestro health

# Restart vLLM only
maestro shutdown --service vllm   # (use start to bring it back)
maestro start --service vllm

# Monitor continuously with auto-recovery
maestro watch --auto-restart --interval 120

# Debug with verbose logging
maestro --log-level DEBUG status --verbose

# Safe shutdown preview
maestro shutdown --dry-run
```

---

## Notes

- **vLLM** runs only on the Somatic node. `start` and `shutdown` for vLLM
  must be run from the Somatic machine (or they will be skipped).
- The global wrapper `~/.local/bin/maestro` activates the maestro venv
  (`/home/gasu/ubik/venv`) automatically — no manual `source` needed.
- vLLM startup takes 30–120 s depending on torch compile cache warmth.
  `maestro start` waits up to 120 s before timing out.
