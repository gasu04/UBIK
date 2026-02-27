#!/usr/bin/env python3
"""
Maestro CLI — Command-Line Interface for UBIK Maestro

Entry point for all operator interactions with the orchestrator.  All
commands auto-detect the local node via :func:`~maestro.platform_detect.detect_node`
and behave accordingly (local vs remote probe hosts, start/stop eligibility).

Commands:
    status    One-shot health check — probe all services, print report, exit.
    start     Bring local services up in dependency order.
    dashboard Interactive TUI dashboard with colour-coded health indicators.
    watch     Continuous background daemon (probe + optional auto-restart).
    shutdown  Orderly stop of all local UBIK services.
    logs      Tail the Maestro operational log.
    metrics   Display current UBIK usage statistics.
    health    Combined status + metrics in one report.
    check     Alias for ``status`` (backward compatible).

Global options (accepted by every command):
    --config PATH       Override the .env file path.
    --log-level LEVEL   Override MAESTRO_LOG_LEVEL for this invocation.

Exit codes:
    0   All services healthy (or action succeeded).
    1   At least one service degraded or unhealthy.
    2   Fatal error (config failure, unhandled exception).

Usage::

    python -m maestro --help
    python -m maestro status
    python -m maestro status --json
    python -m maestro status --verbose
    python -m maestro start
    python -m maestro start --service neo4j
    python -m maestro dashboard --refresh 60
    python -m maestro watch --interval 120 --auto-restart
    python -m maestro watch --once
    python -m maestro shutdown --dry-run
    python -m maestro logs --lines 100 --follow
    python -m maestro metrics
    python -m maestro health

Author: UBIK Project
Version: 0.12.0
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.live import Live

from maestro import __version__
from maestro.config import AppConfig, get_config
from maestro.display import make_dashboard, print_check_results
from maestro.logger import configure_logging, get_logger, log_cluster_health
from maestro.services.health_runner import (
    ALL_SERVICE_NAMES,
    run_all_checks,
    run_selected_checks,
)
from maestro.services.models import ClusterHealth, ServiceStatus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


# ---------------------------------------------------------------------------
# Exit-code helpers
# ---------------------------------------------------------------------------

def _cluster_exit_code(cluster: ClusterHealth) -> int:
    """Map cluster overall status to a shell exit code.

    Task spec: 0 = all healthy, 1 = some services down, 2 = errors.
    This function covers the first two cases; callers use 2 for exceptions.

    Args:
        cluster: Aggregated health snapshot.

    Returns:
        0 when overall status is HEALTHY; 1 otherwise.
    """
    return 0 if cluster.overall_status == ServiceStatus.HEALTHY else 1


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_env_file(path: str) -> None:
    """Load key=value pairs from *path* into ``os.environ``.

    Uses ``os.environ.setdefault`` so explicit env vars take precedence.
    Lines starting with ``#`` and blank lines are silently skipped.

    Args:
        path: Filesystem path to the .env file.

    Raises:
        click.ClickException: If the file cannot be read.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                # Strip surrounding quotes from value if present
                key = key.strip()
                value = value.strip().strip("'\"")
                os.environ.setdefault(key, value)
    except OSError as exc:
        raise click.ClickException(f"Cannot read config file {path!r}: {exc}")


def _tail_lines(path: Path, n: int) -> list[str]:
    """Return the last *n* lines of *path* without reading the whole file.

    Uses binary backward-seek so it is efficient for large log files.

    Args:
        path: Absolute path to the file.
        n: Number of lines to return.

    Returns:
        List of decoded lines (no trailing newlines).  Empty list when the
        file does not exist or *n* is zero.
    """
    if not path.exists() or n <= 0:
        return []
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            remaining = fh.tell()
            lines_found = 0
            buf = b""
            while remaining > 0 and lines_found < n + 1:
                chunk = min(4096, remaining)
                remaining -= chunk
                fh.seek(remaining)
                data = fh.read(chunk)
                buf = data + buf
                lines_found = buf.count(b"\n")
        text = buf.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-n:] if len(lines) > n else lines
    except Exception as exc:
        return [f"[error reading log: {exc}]"]


def _follow_file(path: Path, console: Console) -> None:
    """Stream new lines from *path* to *console* until KeyboardInterrupt.

    Seeks to the end of the file before starting so only new output is
    shown.

    Args:
        path: Log file to follow.
        console: Rich console for output.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            fh.seek(0, 2)  # jump to end
            while True:
                line = fh.readline()
                if line:
                    console.print(line, end="")
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        console.print(f"[dim]Log file disappeared: {path}[/dim]")


async def _watch_loop(
    cfg: AppConfig,
    interval: float,
    timeout: float,
    console: Console,
) -> None:
    """Async implementation of the TUI watch dashboard loop.

    Runs indefinitely until cancelled.

    Args:
        cfg: Application configuration.
        interval: Seconds between health check cycles.
        timeout: Per-check timeout in seconds.
        console: Rich Console for output.
    """
    log = get_logger(__name__)
    cycle = 0
    cluster: ClusterHealth | None = None

    with Live(
        make_dashboard(cfg, None, 0, 0.0),
        console=console,
        refresh_per_second=4,
        transient=False,
    ) as live:
        while True:
            cycle += 1
            live.update(make_dashboard(cfg, cluster, cycle, 0.0, checking=True))

            cluster = await run_all_checks(cfg, timeout=timeout)
            log_cluster_health(log, cluster, cycle=cycle)

            loop = asyncio.get_running_loop()
            next_at = loop.time() + interval
            while True:
                remaining = next_at - loop.time()
                if remaining <= 0:
                    break
                live.update(make_dashboard(cfg, cluster, cycle, remaining))
                await asyncio.sleep(min(0.5, remaining))


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(version=__version__, prog_name="maestro")
@click.option(
    "--config",
    "config_path",
    default=None,
    metavar="PATH",
    help=(
        "Path to a custom .env file.  Overrides the default "
        "{UBIK_ROOT}/maestro/.env location."
    ),
)
@click.option(
    "--log-level",
    "log_level",
    default=None,
    type=click.Choice(_LOG_LEVELS, case_sensitive=False),
    metavar="LEVEL",
    help="Override MAESTRO_LOG_LEVEL for this invocation.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: Optional[str], log_level: Optional[str]) -> None:
    """UBIK Maestro — Infrastructure Orchestrator.

    Monitors Neo4j, ChromaDB, MCP, vLLM, Tailscale, and Docker across
    the Hippocampal (Mac Mini M4 Pro) and Somatic (PowerSpec WSL2) nodes.

    Configuration is read from environment variables or
    {UBIK_ROOT}/maestro/.env — see .env.example for all options.
    Global options (--config, --log-level) must come BEFORE the subcommand.

    \b
    Examples:
        maestro status
        maestro --log-level DEBUG status --verbose
        maestro --config /path/to/.env health
    """
    ctx.ensure_object(dict)

    # Apply .env override before any command runs so get_config() sees it.
    if config_path:
        _load_env_file(config_path)
        get_config.cache_clear()

    if log_level:
        os.environ["MAESTRO_LOG_LEVEL"] = log_level.upper()
        get_config.cache_clear()


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

@cli.command("status")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output raw JSON instead of the Rich table.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show service details dict for each probe result.",
)
@click.option(
    "--timeout",
    default=10.0,
    show_default=True,
    type=float,
    metavar="SECS",
    help="Per-check network timeout.",
)
@click.option(
    "--service",
    "services",
    multiple=True,
    type=click.Choice(sorted(ALL_SERVICE_NAMES), case_sensitive=False),
    metavar="NAME",
    help=(
        "Probe only this service.  Repeatable.  "
        f"Choices: {', '.join(sorted(ALL_SERVICE_NAMES))}"
    ),
)
def status_cmd(
    output_json: bool,
    verbose: bool,
    timeout: float,
    services: tuple[str, ...],
) -> None:
    """Run a one-shot health check and display results.

    Probes every UBIK service concurrently and exits with a code that
    reflects overall cluster health (0=healthy, 1=degraded/unhealthy).

    \b
    Examples:
        maestro status
        maestro status --json
        maestro status --verbose
        maestro status --service neo4j --service chromadb
    """
    console = Console(highlight=False)

    try:
        cfg = get_config()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(2)

    configure_logging(cfg)
    log = get_logger(__name__)
    service_set: Optional[set[str]] = set(services) if services else None

    try:
        cluster = asyncio.run(
            run_selected_checks(cfg, service_set, timeout=timeout)
        )
    except Exception as exc:
        console.print(f"[bold red]Health check failed:[/bold red] {exc}")
        sys.exit(2)

    log_cluster_health(log, cluster)

    if output_json:
        click.echo(cluster.to_json())
        sys.exit(_cluster_exit_code(cluster))

    print_check_results(cluster, cfg, console)

    if verbose:
        import json
        console.print()
        console.print("  [bold dim]Service Details[/bold dim]")
        console.print("  " + "-" * 50)
        for name, result in cluster.services.items():
            if result.details:
                details_str = json.dumps(result.details, indent=2, default=str)
                indented = "\n".join(
                    "    " + line for line in details_str.splitlines()
                )
                console.print(f"  [bold]{name}[/bold]")
                console.print(f"[dim]{indented}[/dim]")
        console.print()

    sys.exit(_cluster_exit_code(cluster))


# ---------------------------------------------------------------------------
# start command
# ---------------------------------------------------------------------------

@cli.command("start")
@click.option(
    "--service",
    "service_name",
    default=None,
    type=click.Choice(sorted(ALL_SERVICE_NAMES), case_sensitive=False),
    metavar="NAME",
    help=(
        "Start only this specific service.  When omitted, all unhealthy "
        "local services are started in dependency order."
    ),
)
@click.option(
    "--timeout",
    default=10.0,
    show_default=True,
    type=float,
    metavar="SECS",
    help="Per-probe timeout when checking current service health.",
)
def start_cmd(service_name: Optional[str], timeout: float) -> None:
    """Start local UBIK services in dependency order.

    Without --service, equivalent to running the full startup sequence:
    probes each local service and starts any that are unhealthy, respecting
    dependency order (Docker before Neo4j, etc.).

    With --service NAME, starts exactly one service (the named service must
    run on the local node).

    \b
    Examples:
        maestro start
        maestro start --service neo4j
        maestro start --service chromadb
    """
    from maestro.orchestrator import Orchestrator
    from maestro.platform_detect import detect_node
    from maestro.services import ServiceRegistry

    console = Console(highlight=False)

    try:
        cfg = get_config()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(2)

    configure_logging(cfg)
    identity = detect_node()
    registry = ServiceRegistry(cfg)

    if service_name:
        # ── Single service ────────────────────────────────────────────────
        svc = next((s for s in registry.get_all() if s.name == service_name), None)
        if svc is None:
            console.print(f"[bold red]Unknown service:[/bold red] {service_name}")
            sys.exit(2)

        if svc.node != identity.node_type:
            console.print(
                f"[yellow]Service [bold]{service_name}[/bold] runs on the "
                f"[bold]{svc.node.value}[/bold] node, not "
                f"[bold]{identity.node_type.value}[/bold] — cannot start locally.[/yellow]"
            )
            sys.exit(1)

        console.print(f"  Starting [bold]{service_name}[/bold]...")
        try:
            ok = asyncio.run(svc.start(cfg.ubik_root))
        except Exception as exc:
            console.print(f"[bold red]Start failed:[/bold red] {exc}")
            sys.exit(2)

        if ok:
            console.print(f"  [bold green]✓[/bold green] {service_name} started.")
        else:
            console.print(f"  [bold red]✗[/bold red] {service_name} failed to start.")
            sys.exit(1)

    else:
        # ── All local services ────────────────────────────────────────────
        orch = Orchestrator(registry, identity)
        console.print("  Starting all unhealthy local services...")
        try:
            failed = asyncio.run(orch.ensure_all_running())
        except Exception as exc:
            console.print(f"[bold red]Start error:[/bold red] {exc}")
            sys.exit(2)

        if failed:
            console.print(
                f"  [bold yellow]Partial start[/bold yellow] — "
                f"failed: {', '.join(failed)}"
            )
            sys.exit(1)
        else:
            console.print("  [bold green]All local services are running.[/bold green]")


# ---------------------------------------------------------------------------
# dashboard command
# ---------------------------------------------------------------------------

@cli.command("dashboard")
@click.option(
    "--refresh",
    "refresh",
    default=30.0,
    show_default=True,
    type=float,
    metavar="SECS",
    help="Seconds between auto-refresh cycles.",
)
@click.option(
    "--timeout",
    default=10.0,
    show_default=True,
    type=float,
    metavar="SECS",
    help="Per-check network timeout.",
)
def dashboard_cmd(refresh: float, timeout: float) -> None:
    """Interactive per-node status dashboard with keyboard shortcuts.

    Displays HIPPOCAMPAL and SOMATIC services with colour-coded health
    indicators, auto-refreshing every REFRESH seconds.

    \b
    Keyboard shortcuts (POSIX terminals):
        q   Quit
        r   Force immediate refresh
        s   Start all unhealthy local services
        x   Shutdown all services

    \b
    Examples:
        maestro dashboard
        maestro dashboard --refresh 60
        maestro dashboard --timeout 5
    """
    from maestro.dashboard import run_dashboard

    console = Console(highlight=False)

    try:
        cfg = get_config()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(2)

    try:
        run_dashboard(cfg, interval=refresh, timeout=timeout, console=console)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


# ---------------------------------------------------------------------------
# watch command
# ---------------------------------------------------------------------------

@cli.command("watch")
@click.option(
    "--interval",
    default=None,
    type=float,
    metavar="SECS",
    help=(
        "Seconds between check cycles.  "
        "Defaults to MAESTRO_CHECK_INTERVAL_S from config (300 s)."
    ),
)
@click.option(
    "--timeout",
    default=10.0,
    show_default=True,
    type=float,
    metavar="SECS",
    help="Per-check network timeout.",
)
@click.option(
    "--once",
    is_flag=True,
    default=False,
    help=(
        "Run one daemon check cycle (probe + log) then exit.  "
        "Uses the structured daemon backend instead of the TUI."
    ),
)
@click.option(
    "--auto-restart",
    "auto_restart",
    is_flag=True,
    default=False,
    help=(
        "Enable auto-restart of unhealthy local services "
        "(activates daemon mode, no TUI)."
    ),
)
def watch_cmd(
    interval: Optional[float],
    timeout: float,
    once: bool,
    auto_restart: bool,
) -> None:
    """Continuous health monitoring — TUI or background daemon.

    Default (no flags): Rich TUI updating every INTERVAL seconds.

    --once: runs one structured probe cycle then exits — useful for cron.

    --auto-restart: runs the daemon continuously, restarting any unhealthy
    local services after each cycle.

    \b
    Examples:
        maestro watch
        maestro watch --interval 60
        maestro watch --once
        maestro watch --auto-restart --interval 120
    """
    from maestro.daemon import MaestroDaemon

    console = Console(highlight=False)

    try:
        cfg = get_config()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(2)

    configure_logging(cfg)
    effective_interval = interval if interval is not None else float(
        cfg.maestro.check_interval_s
    )

    if once or auto_restart:
        # ── Daemon mode (structured logging, no TUI) ──────────────────────
        daemon = MaestroDaemon(
            check_interval_s=int(effective_interval),
            auto_restart=auto_restart,
            cfg=cfg,
        )
        try:
            if once:
                asyncio.run(daemon.run_once())
            else:
                asyncio.run(daemon.run())
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped.[/dim]")
        return

    # ── TUI mode ──────────────────────────────────────────────────────────
    console.print(
        f"  [bold blue]UBIK MAESTRO[/bold blue]  "
        f"[dim]interval: {effective_interval:.0f}s  "
        f"timeout: {timeout:.0f}s  "
        f"Ctrl+C to stop[/dim]\n"
    )
    try:
        asyncio.run(_watch_loop(cfg, effective_interval, timeout, console))
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


# ---------------------------------------------------------------------------
# shutdown command
# ---------------------------------------------------------------------------

@cli.command("shutdown")
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help=(
        "Report what would be stopped without actually stopping anything.  "
        "Safe to run at any time."
    ),
)
@click.option(
    "--force",
    "emergency",
    is_flag=True,
    default=False,
    help=(
        "Emergency SIGKILL of all local UBIK processes.  "
        "Last resort — prefer the default orderly shutdown."
    ),
)
def shutdown_cmd(dry_run: bool, emergency: bool) -> None:
    """Stop all local UBIK services in reverse dependency order.

    Default: graceful ordered stop (MCP → ChromaDB → Neo4j → Docker on
    Hippocampal; vLLM on Somatic), escalating to SIGKILL per service if
    the graceful stop does not complete within 30 seconds.

    \b
    Examples:
        maestro shutdown
        maestro shutdown --dry-run
        maestro shutdown --force
    """
    from maestro.platform_detect import detect_node
    from maestro.services import ServiceRegistry
    from maestro.shutdown import ShutdownController

    console = Console(highlight=False)

    try:
        cfg = get_config()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(2)

    configure_logging(cfg)
    identity = detect_node()
    registry = ServiceRegistry(cfg)
    ctrl = ShutdownController(registry, identity)

    try:
        if emergency:
            asyncio.run(ctrl.emergency_shutdown())
            console.print("[bold red]Emergency shutdown complete.[/bold red]")
        else:
            stopped = asyncio.run(ctrl.orderly_shutdown(dry_run=dry_run))
            if dry_run:
                console.print(
                    f"[bold yellow]DRY RUN[/bold yellow] — would stop "
                    f"{len(stopped)} service(s): {', '.join(stopped) or 'none'}"
                )
            else:
                console.print(
                    f"[bold green]Shutdown complete[/bold green] — "
                    f"stopped {len(stopped)} service(s)"
                )
    except KeyboardInterrupt:
        console.print("\n[dim]Shutdown interrupted.[/dim]")


# ---------------------------------------------------------------------------
# logs command
# ---------------------------------------------------------------------------

@cli.command("logs")
@click.option(
    "--lines",
    "-n",
    default=50,
    show_default=True,
    type=int,
    metavar="N",
    help="Number of log lines to display.",
)
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    default=False,
    help="Keep reading new log entries as they are written (like tail -f).",
)
def logs_cmd(lines: int, follow: bool) -> None:
    """Tail the Maestro operational log.

    Reads from {UBIK_ROOT}/logs/maestro/maestro.log.  Use --follow to
    stream new entries in real time (Ctrl+C to stop).

    \b
    Examples:
        maestro logs
        maestro logs --lines 100
        maestro logs --follow
        maestro logs -n 20 -f
    """
    console = Console(highlight=False)

    try:
        cfg = get_config()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(2)

    log_path = cfg.log_dir / "maestro.log"

    if not log_path.exists():
        console.print(
            f"[yellow]Log file not found:[/yellow] [dim]{log_path}[/dim]\n"
            f"  Run [bold]maestro watch --once[/bold] to generate the first entry."
        )
        sys.exit(0)

    # Print last N lines
    recent = _tail_lines(log_path, lines)
    if recent:
        console.print(
            f"  [bold dim]Showing last {len(recent)} lines of "
            f"[/bold dim][dim]{log_path}[/dim]\n"
        )
        for line in recent:
            console.print(line)
    else:
        console.print(f"[dim]Log file is empty: {log_path}[/dim]")

    if follow:
        console.print("\n[dim]Following… (Ctrl+C to stop)[/dim]\n")
        _follow_file(log_path, console)


# ---------------------------------------------------------------------------
# metrics command
# ---------------------------------------------------------------------------

@cli.command("metrics")
def metrics_cmd() -> None:
    """Display current UBIK usage statistics.

    Collects ChromaDB collection sizes, Neo4j graph counts, vLLM state,
    GPU utilisation (Somatic node only), and disk usage.  All metrics are
    best-effort — unavailable values are shown as N/A.

    \b
    Examples:
        maestro metrics
    """
    from maestro.metrics import MetricsCollector
    from maestro.orchestrator import Orchestrator
    from maestro.platform_detect import detect_node
    from maestro.services import ServiceRegistry

    console = Console(highlight=False)

    try:
        cfg = get_config()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(2)

    identity = detect_node()
    registry = ServiceRegistry(cfg)
    orch = Orchestrator(registry, identity)
    collector = MetricsCollector()

    try:
        metrics = asyncio.run(collector.collect(orch))
    except Exception as exc:
        console.print(f"[bold red]Metrics collection failed:[/bold red] {exc}")
        sys.exit(2)

    console.print()
    for line in collector.format_report(metrics).splitlines():
        console.print(f"  {line}")
    console.print()


# ---------------------------------------------------------------------------
# health command
# ---------------------------------------------------------------------------

@cli.command("health")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output results as JSON (status + metrics combined).",
)
@click.option(
    "--timeout",
    default=10.0,
    show_default=True,
    type=float,
    metavar="SECS",
    help="Per-check network timeout.",
)
def health_cmd(output_json: bool, timeout: float) -> None:
    """Combined status check + usage metrics report.

    Runs a full service health check and a usage metrics collection
    concurrently, then prints both reports.

    Exit codes: 0 = all healthy, 1 = some services down, 2 = error.

    \b
    Examples:
        maestro health
        maestro health --json
        maestro health --timeout 5
    """
    from maestro.metrics import MetricsCollector
    from maestro.orchestrator import Orchestrator
    from maestro.platform_detect import detect_node
    from maestro.services import ServiceRegistry

    console = Console(highlight=False)

    try:
        cfg = get_config()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(2)

    configure_logging(cfg)
    log = get_logger(__name__)
    identity = detect_node()
    registry = ServiceRegistry(cfg)
    orch = Orchestrator(registry, identity)
    collector = MetricsCollector()

    async def _gather_both():
        return await asyncio.gather(
            run_all_checks(cfg, timeout=timeout),
            collector.collect(orch),
            return_exceptions=True,
        )

    try:
        raw_cluster, raw_metrics = asyncio.run(_gather_both())
    except Exception as exc:
        console.print(f"[bold red]Health check failed:[/bold red] {exc}")
        sys.exit(2)

    # Handle partial gather failures
    if isinstance(raw_cluster, BaseException):
        console.print(f"[bold red]Status error:[/bold red] {raw_cluster}")
        sys.exit(2)

    cluster: ClusterHealth = raw_cluster
    log_cluster_health(log, cluster)

    if output_json:
        import json
        import dataclasses

        metrics_dict: dict = {}
        if not isinstance(raw_metrics, BaseException):
            metrics_dict = dataclasses.asdict(raw_metrics)
            # Convert datetime to ISO string
            if "timestamp" in metrics_dict:
                metrics_dict["timestamp"] = str(metrics_dict["timestamp"])

        combined = {
            "status": json.loads(cluster.to_json()),
            "metrics": metrics_dict,
        }
        click.echo(json.dumps(combined, indent=2, default=str))
        sys.exit(_cluster_exit_code(cluster))

    # Rich output
    print_check_results(cluster, cfg, console)

    if not isinstance(raw_metrics, BaseException):
        console.print()
        for line in collector.format_report(raw_metrics).splitlines():
            console.print(f"  {line}")
        console.print()
    else:
        console.print(
            f"[dim]Metrics unavailable: {raw_metrics}[/dim]"
        )

    sys.exit(_cluster_exit_code(cluster))


# ---------------------------------------------------------------------------
# check command  (backward-compatible alias for status)
# ---------------------------------------------------------------------------

@cli.command("check", hidden=False)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output raw JSON instead of the Rich table.",
)
@click.option(
    "--timeout",
    default=10.0,
    show_default=True,
    type=float,
    metavar="SECS",
    help="Per-check network timeout.",
)
@click.option(
    "--service",
    "services",
    multiple=True,
    type=click.Choice(sorted(ALL_SERVICE_NAMES), case_sensitive=False),
    metavar="NAME",
    help=(
        "Probe only this service.  Repeatable.  "
        f"Choices: {', '.join(sorted(ALL_SERVICE_NAMES))}"
    ),
)
def check_cmd(
    output_json: bool,
    timeout: float,
    services: tuple[str, ...],
) -> None:
    """Run a one-shot health check (alias for ``status``).

    Retained for backward compatibility.  Prefer ``maestro status``.

    \b
    Examples:
        maestro check
        maestro check --json
        maestro check --service neo4j --service chromadb
    """
    console = Console(highlight=False)

    try:
        cfg = get_config()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        sys.exit(2)

    configure_logging(cfg)
    log = get_logger(__name__)
    service_set: Optional[set[str]] = set(services) if services else None

    try:
        cluster = asyncio.run(
            run_selected_checks(cfg, service_set, timeout=timeout)
        )
    except Exception as exc:
        console.print(f"[bold red]Health check failed:[/bold red] {exc}")
        sys.exit(2)

    log_cluster_health(log, cluster)

    if output_json:
        click.echo(cluster.to_json())
    else:
        print_check_results(cluster, cfg, console)

    sys.exit(_cluster_exit_code(cluster))
