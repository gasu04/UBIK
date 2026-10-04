# UBIK Maestro v0.12.0 — Reference Guide

Infrastructure orchestrator for the UBIK two-node cluster.

---

## Cluster Topology

| Node | Hardware | Role | Tailscale IP |
|------|----------|------|--------------|
| **Hippocampal** | Mac Mini M4 Pro (macOS) | Neo4j · ChromaDB · MCP | 100.103.242.91 |
| **Somatic** | PowerSpec RTX 5090 (WSL2) | vLLM inference | 100.92.12.89 |

**Services:**

| Service | Node | Port |
|---------|------|------|
| neo4j | hippocampal | 7474 (HTTP) · 7687 (Bolt) |
| chromadb | hippocampal | 8000 |
| mcp | hippocampal | 8080 |
| vllm | somatic | 8002 |
| tailscale | mesh | — |
| docker | hippocampal | — |

**Startup dependency order:** docker → neo4j → chromadb → mcp

---

## Installation & Activation

```bash
# Activate the project venv (required before using maestro)
source /Volumes/990PRO\ 4T/DeepSeek/venv/bin/activate

# Verify
maestro --version
```

---

## Global Syntax

```
maestro [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]
```

> Global options **must come before** the subcommand.

| Global Option | Default | Description |
|---------------|---------|-------------|
| `--config PATH` | `{UBIK_ROOT}/maestro/.env` | Override the .env file |
| `--log-level LEVEL` | from config | DEBUG · INFO · WARNING · ERROR · CRITICAL |
| `--version` | — | Print version and exit |
| `-h, --help` | — | Show help |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All probed services healthy / action succeeded |
| `1` | At least one service degraded or unhealthy |
| `2` | Fatal error (config failure, unhandled exception) |

---

## Commands

### `status` — One-shot health check

Probes all services concurrently and prints a Rich table.

```
maestro status [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | off | Output raw JSON instead of the Rich table |
| `--verbose, -v` | off | Show the full details dict per service |
| `--timeout SECS` | 10.0 | Per-check network timeout |
| `--service NAME` | all | Probe only this service (repeatable) |

Service choices: `chromadb` · `docker` · `mcp` · `neo4j` · `tailscale` · `vllm`

```bash
# Common invocations
maestro status
maestro status --json
maestro status --verbose
maestro status --service neo4j
maestro status --service neo4j --service chromadb
maestro status --timeout 5
maestro --log-level DEBUG status --verbose
```

---

### `start` — Bring services up

Starts unhealthy local services in dependency order. Only services that belong to the current node can be started.

```
maestro start [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--service NAME` | all | Start exactly one named service |
| `--timeout SECS` | 10.0 | Per-probe timeout when checking current health |

```bash
maestro start                      # Start all unhealthy local services
maestro start --service mcp        # Start only MCP
maestro start --service neo4j
maestro start --service chromadb
```

---

### `dashboard` — Interactive TUI

Live colour-coded dashboard that auto-refreshes. Keyboard shortcuts work in POSIX terminals.

```
maestro dashboard [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--refresh SECS` | 30.0 | Seconds between auto-refresh cycles |
| `--timeout SECS` | 10.0 | Per-check network timeout |

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Force immediate refresh |
| `s` | Start all unhealthy local services |
| `x` | Shutdown all services |

```bash
maestro dashboard
maestro dashboard --refresh 60
maestro dashboard --timeout 5
```

---

### `watch` — Continuous monitoring

Runs as a TUI loop, a one-shot probe, or a background daemon with auto-restart.

```
maestro watch [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--interval SECS` | from config (300) | Seconds between check cycles |
| `--timeout SECS` | 10.0 | Per-check timeout |
| `--once` | off | Run one daemon cycle then exit (structured log, no TUI) |
| `--auto-restart` | off | Restart unhealthy local services each cycle (daemon mode) |

```bash
maestro watch                          # TUI, default interval
maestro watch --interval 60            # TUI, 60 s interval
maestro watch --once                   # Single probe cycle, exit (good for cron)
maestro watch --auto-restart           # Daemon: probe + restart loop
maestro watch --auto-restart --interval 120
```

**Cron example** (log once per minute):
```cron
* * * * * /path/to/venv/bin/maestro watch --once >> /var/log/ubik/cron.log 2>&1
```

---

### `shutdown` — Stop services

Stops local services in reverse dependency order (MCP → ChromaDB → Neo4j → Docker on Hippocampal; vLLM on Somatic). Escalates to SIGKILL after 30 s if graceful stop stalls.

```
maestro shutdown [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dry-run` | off | Report what would be stopped without stopping anything |
| `--emergency` | off | SIGKILL all local UBIK processes immediately |

```bash
maestro shutdown                  # Graceful ordered stop
maestro shutdown --dry-run        # Preview only
maestro shutdown --emergency      # Last-resort kill
```

---

### `logs` — Tail the operational log

Reads from `{UBIK_ROOT}/logs/maestro/maestro.log`.

```
maestro logs [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--lines N, -n N` | 50 | Number of lines to show |
| `--follow, -f` | off | Stream new entries in real time (Ctrl+C to stop) |

```bash
maestro logs
maestro logs --lines 100
maestro logs --follow
maestro logs -n 20 -f
```

---

### `metrics` — Usage statistics

Collects ChromaDB collection sizes, Neo4j graph counts, vLLM state, GPU utilisation (Somatic only), and disk usage. All values are best-effort — unavailable metrics show as N/A.

```
maestro metrics
```

---

### `health` — Combined status + metrics

Runs a full service health check and metrics collection concurrently.

```
maestro health [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--json` | off | Output status + metrics as combined JSON |
| `--timeout SECS` | 10.0 | Per-check timeout |

```bash
maestro health
maestro health --json
maestro health --timeout 5
```

---

### `check` — Backward-compatible alias for `status`

Same as `status` minus `--verbose`. Prefer `maestro status` in new scripts.

```bash
maestro check
maestro check --json
maestro check --service neo4j --service chromadb
```

---

## MCP Server — Direct Script Control

The MCP server can also be controlled directly via `hippocampal/run_mcp.sh` from the UBIK root.

```
./hippocampal/run_mcp.sh [COMMAND]
```

| Command | Description |
|---------|-------------|
| *(none)* | Run in **foreground** |
| `start` | Run in **background** (daemon mode) |
| `-d` / `--daemon` | Alias for `start` |
| `stop` | Stop the server (PID file + orphan sweep) |
| `restart` | Stop then start in background |
| `status` | Show PID, process list, and port state |
| `logs` | `tail -f` the MCP log file |

```bash
# From UBIK root
./hippocampal/run_mcp.sh start
./hippocampal/run_mcp.sh status
./hippocampal/run_mcp.sh logs
./hippocampal/run_mcp.sh stop
./hippocampal/run_mcp.sh restart
```

Log file: `hippocampal/logs/mcp_server.log`

---

## Common Operational Workflows

### Daily startup
```bash
maestro start        # bring up anything that's down
maestro status       # confirm everything is healthy
```

### Morning health report
```bash
maestro health       # status + metrics in one shot
```

### Investigate a failing service
```bash
maestro status --service mcp --verbose
maestro --log-level DEBUG status --service mcp
```

### Restart only MCP
```bash
maestro start --service mcp
# or directly:
./hippocampal/run_mcp.sh restart
```

### Graceful cluster shutdown
```bash
maestro shutdown --dry-run   # preview
maestro shutdown             # execute
```

### Continuous monitoring with auto-recovery
```bash
maestro watch --auto-restart --interval 120
```

### Export status for scripts / CI
```bash
maestro status --json | jq '.overall_status'
maestro health --json | jq '.status.services.vllm.status'
```

---

## Configuration

Config is loaded from `{UBIK_ROOT}/maestro/.env` (or `--config PATH`).
Environment variables always override the .env file.

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAESTRO_LOG_LEVEL` | INFO | Log verbosity |
| `MAESTRO_CHECK_INTERVAL_S` | 300 | Watch loop interval (seconds) |
| `HIPPOCAMPAL_TAILSCALE_IP` | 100.103.242.91 | Hippocampal node Tailscale IP |
| `SOMATIC_TAILSCALE_IP` | 100.92.12.89 | Somatic node Tailscale IP |

See `maestro/.env.example` for the full list.

---

*UBIK Maestro v0.12.0 · generated 2026-02-27*
