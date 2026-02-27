# UBIK Maestro

Orchestration daemon for the UBIK distributed AI memory system.
Maestro monitors, starts, and shuts down services across the two-node cluster — **Hippocampal** (Mac Mini M4 Pro, macOS) and **Somatic** (PowerSpec RTX 5090, WSL2 Linux) — over Tailscale.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Commands Reference](#commands-reference)
3. [Configuration Reference](#configuration-reference)
4. [Sync Strategy](#sync-strategy)
5. [Troubleshooting](#troubleshooting)
6. [How to Add a New Service](#how-to-add-a-new-service)

---

## Quick Start

### Prerequisites

| Node | OS | Python |
|------|----|--------|
| Hippocampal | macOS (arm64) | venv at `{UBIK_ROOT}/hippocampal/venv` |
| Somatic | Linux / WSL2 (x86_64) | conda env `pytorch_env` |

### Deploy (run on each node)

```bash
cd {UBIK_ROOT}
bash maestro/setup_maestro.sh
```

The script:
1. Auto-detects the local node (Hippocampal or Somatic)
2. Installs `maestro/requirements.txt` into the node's Python environment
3. Creates `maestro/.env` from `maestro/.env.example` (if not present)
4. Copies `NEO4J_PASSWORD` and `CHROMADB_TOKEN` from `hippocampal/.env`
5. Creates the log directory (`{UBIK_ROOT}/logs/maestro/`)
6. Runs `python -m maestro status` as a smoke test

### Add a shell alias (convenience)

**Hippocampal (macOS, zsh):**
```bash
echo "alias maestro='cd \"/Volumes/990PRO 4T/UBIK\" && \"/Volumes/990PRO 4T/UBIK/hippocampal/venv/bin/python\" -m maestro'" >> ~/.zshrc
source ~/.zshrc
```

**Somatic (WSL2, bash):**
```bash
echo "alias maestro='cd /home/gasu/ubik && conda run -n pytorch_env python -m maestro'" >> ~/.bashrc
source ~/.bashrc
```

After adding the alias:
```bash
maestro status
maestro health
maestro dashboard
```

### Typical first-run workflow

```bash
# 1. Check what's up
python -m maestro status

# 2. Start any services that are down
python -m maestro start

# 3. Full status + metrics
python -m maestro health

# 4. Open live dashboard
python -m maestro dashboard
```

---

## Commands Reference

All commands accept global options:

| Option | Default | Description |
|--------|---------|-------------|
| `--config PATH` | `maestro/.env` | Override the `.env` file |
| `--log-level LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `--version` | — | Print Maestro version and exit |
| `-h`, `--help` | — | Show help |

---

### `status` — One-shot health check

Probe all services and print a colour-coded table.

```bash
python -m maestro status
python -m maestro status --verbose
python -m maestro status --json
python -m maestro status --service neo4j --service chromadb
python -m maestro status --timeout 10
```

| Flag | Description |
|------|-------------|
| `--json` | Output raw JSON (suitable for scripting) |
| `--verbose` | Show extra probe details per service |
| `--service NAME` | Restrict check to named service(s); repeatable |
| `--timeout SECS` | Per-probe timeout (default: 5 s) |

**Exit codes:**

| Code | Meaning |
|------|---------|
| `0` | All services HEALTHY |
| `1` | At least one service DEGRADED |
| `2` | At least one service UNHEALTHY |

`check` is a backward-compatible alias for `status`.

---

### `start` — Bring local services up

Start any stopped local services in dependency order.

```bash
python -m maestro start
python -m maestro start --service neo4j
python -m maestro start --timeout 120
```

| Flag | Description |
|------|-------------|
| `--service NAME` | Start only this service; repeatable |
| `--timeout SECS` | Per-service health-wait timeout (default: 60 s) |

Services are started in topological order respecting `depends_on`.
Remote services (e.g. vLLM on Somatic when running on Hippocampal) are never started.

---

### `dashboard` — Live TUI dashboard

Interactive terminal dashboard with colour-coded service tiles, refreshed automatically.

```bash
python -m maestro dashboard
python -m maestro dashboard --refresh 30
python -m maestro dashboard --timeout 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--refresh SECS` | `60` | Seconds between full re-probes |
| `--timeout SECS` | `5` | Per-probe timeout |

Press **`q`** or **`Ctrl-C`** to exit.

---

### `watch` — Continuous background monitor

Run the full health-check loop in the foreground, optionally restarting unhealthy services.

```bash
python -m maestro watch
python -m maestro watch --interval 120 --auto-restart
python -m maestro watch --once
```

| Flag | Default | Description |
|------|---------|-------------|
| `--interval SECS` | `300` | Seconds between check cycles |
| `--timeout SECS` | `5` | Per-probe timeout |
| `--auto-restart` | off | Attempt `start` on unhealthy local services |
| `--once` | off | Run one cycle then exit |

The daemon writes structured JSON events to `{LOG_DIR}/maestro.log`.

---

### `shutdown` — Stop all local services

Graceful ordered stop, escalating to SIGKILL per service if needed.

```bash
python -m maestro shutdown
python -m maestro shutdown --dry-run
python -m maestro shutdown --emergency
```

| Flag | Description |
|------|-------------|
| `--dry-run` | List what would be stopped without stopping anything |
| `--emergency` | SIGKILL all local UBIK processes immediately (last resort) |

**Shutdown order** is the reverse of startup order:

- Hippocampal: MCP → ChromaDB → Neo4j → Docker
- Somatic: vLLM

---

### `logs` — Tail the operational log

```bash
python -m maestro logs
python -m maestro logs --lines 50
python -m maestro logs --follow
```

| Flag | Default | Description |
|------|---------|-------------|
| `-n`, `--lines N` | `20` | Lines to print from the end of the log |
| `-f`, `--follow` | off | Stream new log entries as they arrive (`tail -f` mode) |

---

### `metrics` — Usage statistics

Display a snapshot of UBIK resource usage.

```bash
python -m maestro metrics
```

Reports:
- ChromaDB: episodic and semantic vector counts
- Neo4j: node and relationship counts
- vLLM: running / stopped
- GPU: utilisation % and VRAM used (Somatic only)
- Disk: used space on the UBIK_ROOT partition

---

### `health` — Combined status + metrics

Run `status` and `metrics` concurrently and display both.

```bash
python -m maestro health
python -m maestro health --json
python -m maestro health --timeout 10
```

Exit codes match `status`.

---

## Configuration Reference

Configuration is loaded from `{UBIK_ROOT}/maestro/.env`.
Create it with `bash maestro/setup_maestro.sh` or copy `.env.example` manually.

### UBIK Root

| Variable | Default | Description |
|----------|---------|-------------|
| `UBIK_ROOT` | Auto-detected | Override project root path |

Auto-detection priority:
1. `$UBIK_ROOT` environment variable
2. macOS: `/Volumes/990PRO 4T/UBIK`
3. Linux: `/home/gasu/ubik`

---

### Maestro Orchestrator (`MAESTRO_` prefix)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAESTRO_LOG_DIR` | `{UBIK_ROOT}/logs/maestro/` | Log directory |
| `MAESTRO_CHECK_INTERVAL_S` | `300` | Health-check loop interval (seconds) |
| `MAESTRO_LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` \| `CRITICAL` |

---

### Hippocampal Node (`HIPPOCAMPAL_` prefix)

| Variable | Default | Description |
|----------|---------|-------------|
| `HIPPOCAMPAL_TAILSCALE_IP` | `100.103.242.91` | Tailscale IP of the Mac node |
| `HIPPOCAMPAL_TAILSCALE_HOSTNAME` | `ubik-hippocampal` | Tailscale hostname |

---

### Somatic Node (`SOMATIC_` prefix)

| Variable | Default | Description |
|----------|---------|-------------|
| `SOMATIC_TAILSCALE_IP` | `100.79.166.114` | Tailscale IP of the Linux node |
| `SOMATIC_TAILSCALE_HOSTNAME` | `ubik-somatic` | Tailscale hostname |

---

### Neo4j (no prefix)

| Variable | Default | Required |
|----------|---------|----------|
| `NEO4J_HTTP_PORT` | `7474` | no |
| `NEO4J_BOLT_PORT` | `7687` | no |
| `NEO4J_USER` | `neo4j` | no |
| `NEO4J_PASSWORD` | — | **yes** |

---

### ChromaDB (no prefix)

| Variable | Default | Required |
|----------|---------|----------|
| `CHROMADB_PORT` | `8001` | no |
| `CHROMADB_TOKEN` | — | **yes** |

---

### MCP Server (no prefix)

| Variable | Default | Required |
|----------|---------|----------|
| `MCP_PORT` | `8080` | no |

---

### vLLM (no prefix)

| Variable | Default | Required |
|----------|---------|----------|
| `VLLM_PORT` | `8002` | no |
| `VLLM_MODEL_PATH` | `/home/gasu/ubik/models/deepseek-awq/…` | no |

---

## Sync Strategy

Maestro lives inside the UBIK git repository.  The simplest way to keep both nodes in sync is:

### Option A — git (recommended)

```bash
# On whichever node you develop on:
git add maestro/
git commit -m "Update maestro"
git push

# On the other node:
git pull
bash maestro/setup_maestro.sh
```

### Option B — rsync over Tailscale

**From Hippocampal → Somatic:**
```bash
rsync -avz --delete \
  "/Volumes/990PRO 4T/UBIK/maestro/" \
  gasu@100.79.166.114:/home/gasu/ubik/maestro/

# Then on Somatic:
bash /home/gasu/ubik/maestro/setup_maestro.sh
```

**From Somatic → Hippocampal:**
```bash
rsync -avz --delete \
  /home/gasu/ubik/maestro/ \
  gasu@100.103.242.91:"/Volumes/990PRO 4T/UBIK/maestro/"

# Then on Hippocampal:
bash "/Volumes/990PRO 4T/UBIK/maestro/setup_maestro.sh"
```

> **Note:** `setup_maestro.sh` installs dependencies and runs a status check but never overwrites an existing `.env`.  New config keys can be added by merging `.env.example` manually.

---

## Troubleshooting

### `NEO4J_PASSWORD` / `CHROMADB_TOKEN` not set

```
pydantic_core.ValidationError: ... NEO4J_PASSWORD
```

Edit `maestro/.env` and set the required values:
```ini
NEO4J_PASSWORD=your_password_here
CHROMADB_TOKEN=your_token_here
```

Alternatively, re-run the setup script — it imports these from `hippocampal/.env` automatically.

---

### venv not found (Hippocampal)

```
ERROR: venv not found: /Volumes/990PRO 4T/UBIK/hippocampal/venv
```

Create it:
```bash
python3 -m venv "/Volumes/990PRO 4T/UBIK/hippocampal/venv"
bash maestro/setup_maestro.sh
```

Or override the path:
```bash
HIPPOCAMPAL_VENV_PATH=/path/to/other/venv bash maestro/setup_maestro.sh
```

---

### conda not found (Somatic)

```
ERROR: conda not found in PATH.
```

Source conda init or activate the base environment first:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
bash maestro/setup_maestro.sh
```

---

### Tailscale services all `UNHEALTHY`

```bash
python -m maestro status
# neo4j UNHEALTHY, chromadb UNHEALTHY …
```

Check Tailscale is up on both nodes:
```bash
tailscale status
tailscale ping 100.79.166.114   # reach Somatic from Hippocampal
```

If Tailscale IPs have changed, update `.env`:
```ini
HIPPOCAMPAL_TAILSCALE_IP=<new-ip>
SOMATIC_TAILSCALE_IP=<new-ip>
```

---

### Services down after reboot

Docker, Neo4j, ChromaDB, and MCP do not auto-start on boot.  Start them with:
```bash
python -m maestro start
```

To auto-start on login, add to your shell RC file:
```bash
# Add to ~/.zshrc (Hippocampal) or ~/.bashrc (Somatic)
cd "/Volumes/990PRO 4T/UBIK" && python -m maestro start --timeout 120
```

---

### `python -m maestro` not found / wrong Python

Make sure you're using the venv Python (Hippocampal) or conda Python (Somatic):

```bash
# Hippocampal
"/Volumes/990PRO 4T/UBIK/hippocampal/venv/bin/python" -m maestro status

# Somatic
conda run -n pytorch_env python -m maestro status
```

Or use the shell alias added by `setup_maestro.sh`.

---

### Log file location

```bash
# Default path
ls "{UBIK_ROOT}/logs/maestro/"

# Follow live
python -m maestro logs --follow

# Override log dir
MAESTRO_LOG_DIR=/tmp/maestro-logs python -m maestro status
```

---

### Enable debug logging

```bash
python -m maestro --log-level DEBUG status
```

Or permanently in `.env`:
```ini
MAESTRO_LOG_LEVEL=DEBUG
```

---

## How to Add a New Service

Maestro's service layer is designed for extension.  Adding a new service requires four steps.

### Step 1 — Create the service class

Create `maestro/services/{name}_service.py`:

```python
from pathlib import Path
from maestro.platform_detect import NodeType
from maestro.services.base import ProbeResult, UbikService, _run_proc


class MyNewService(UbikService):
    """Brief description of the service."""

    def __init__(
        self,
        *,
        ubik_root: Path,
        port: int = 9999,
        max_wait_s: float = 30.0,
    ) -> None:
        super().__init__(max_wait_s=max_wait_s)
        self._ubik_root = ubik_root
        self._port = port

    # ── Identity ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "mynewservice"

    @property
    def node(self) -> NodeType:
        # Which node runs this service?
        return NodeType.HIPPOCAMPAL  # or SOMATIC

    @property
    def ports(self) -> list[int]:
        return [self._port]

    @property
    def depends_on(self) -> list[str]:
        # Service names that must be started first.
        return ["docker"]

    # ── Health probe ─────────────────────────────────────────────────────────

    async def probe(self, host: str) -> ProbeResult:
        import time, httpx
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                url = f"http://{host}:{self._port}/health"
                resp = await client.get(url)
            latency_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                return ProbeResult(
                    name=self.name,
                    node=self.node,
                    healthy=True,
                    latency_ms=latency_ms,
                    details={"url": url, "http_status": resp.status_code},
                )
            return ProbeResult(
                name=self.name, node=self.node, healthy=False,
                latency_ms=latency_ms,
                error=f"Unexpected status {resp.status_code}",
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            return ProbeResult(
                name=self.name, node=self.node, healthy=False,
                latency_ms=latency_ms, error=str(exc),
            )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self, ubik_root: Path) -> bool:
        from maestro.platform_detect import detect_node
        identity = detect_node()
        if identity.node_type not in (NodeType.HIPPOCAMPAL, NodeType.UNKNOWN):
            return False  # wrong node

        rc, _, stderr = await _run_proc(
            "docker", "compose", "up", "-d", "mynewservice",
            cwd=str(ubik_root),
        )
        if rc != 0:
            return False

        result = await self.probe_with_timeout("localhost", timeout=self._max_wait_s)
        return result.healthy

    async def stop(self) -> bool:
        rc, _, _ = await _run_proc(
            "docker", "compose", "stop", "mynewservice",
            cwd=str(self._ubik_root),
        )
        return rc == 0
```

### Step 2 — Register the service

Edit `maestro/services/__init__.py`:

```python
from maestro.services.mynewservice_service import MyNewService

# Inside ServiceRegistry.__init__, after the existing services:
self.register(MyNewService(ubik_root=cfg.ubik_root))
```

And add it to `__all__` and `ALL_SERVICE_NAMES` in `health_runner.py`.

### Step 3 — Add display formatting (optional)

In `maestro/display.py`, add a branch to `format_details()`:

```python
elif result.service_name == "mynewservice":
    # Format relevant details from result.details dict
    return f"port {result.details.get('http_status', 'N/A')}"
```

### Step 4 — Write tests

Create `maestro/tests/test_mynewservice.py` following the pattern in
`test_service_probes.py`:

```python
class TestMyNewServiceProbe:
    @pytest.mark.asyncio
    async def test_probe_healthy_200(self):
        svc = MyNewService(ubik_root=Path("/fake/ubik"))
        with patch("maestro.services.mynewservice_service.httpx.AsyncClient",
                   _http_client_mock(200)):
            result = await svc.probe("127.0.0.1")
        assert result.healthy is True
```

Run the full suite to confirm nothing regressed:
```bash
python -m pytest maestro/tests/ -v --tb=short
```

---

*UBIK Maestro — "Build systems that survive their creators."*
