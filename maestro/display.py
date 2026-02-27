#!/usr/bin/env python3
"""
Maestro Display — Rich Terminal Rendering

Provides all Rich-based UI primitives for the Maestro CLI.  Two
rendering modes are supported:

One-shot (``check`` command):
    print_check_results(cluster, cfg, console)
    Prints a formatted table + summary and returns.

Live dashboard (``watch`` command):
    make_dashboard(cfg, cluster, cycle, next_in) → RenderableType
    Returns a renderable that a ``rich.live.Live`` context can display
    and update each refresh cycle.

Design decisions:
    • All colours are driven by ServiceStatus — no magic strings.
    • format_details() uses Rich markup strings; Table cells parse
      markup automatically.
    • Latency colour thresholds: green <50 ms, yellow <200 ms, red ≥200 ms.
    • Service → Node mapping is centralised in _SERVICE_NODES so the
      display layer never imports or inspects config objects for routing.

Author: UBIK Project
Version: 0.1.0
"""

from datetime import timezone
from typing import Any

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from maestro.config import AppConfig
from maestro.services.models import ClusterHealth, ServiceResult, ServiceStatus

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_STATUS_STYLES: dict[ServiceStatus, str] = {
    ServiceStatus.HEALTHY: "bold green",
    ServiceStatus.DEGRADED: "bold yellow",
    ServiceStatus.UNHEALTHY: "bold red",
}

_STATUS_ICONS: dict[ServiceStatus, str] = {
    ServiceStatus.HEALTHY: "●",
    ServiceStatus.DEGRADED: "◑",
    ServiceStatus.UNHEALTHY: "✗",
}

# Maps service name → which node/layer hosts it (display only).
_SERVICE_NODES: dict[str, str] = {
    "neo4j": "hippocampal",
    "chromadb": "hippocampal",
    "mcp": "hippocampal",
    "vllm": "somatic",
    "tailscale": "mesh",
    "docker": "local",
}

# Canonical display order for services in the table.
_SERVICE_ORDER: tuple[str, ...] = (
    "neo4j",
    "chromadb",
    "mcp",
    "vllm",
    "tailscale",
    "docker",
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_latency(ms: float | None) -> Text:
    """Format a latency value as a coloured Rich Text.

    Args:
        ms: Latency in milliseconds, or ``None`` if measurement failed.

    Returns:
        Coloured :class:`rich.text.Text` — green for fast, yellow for
        moderate, red for slow, dim dash for absent.
    """
    if ms is None:
        return Text("—", style="dim")
    if ms < 50:
        return Text(f"{ms:.0f} ms", style="green")
    if ms < 200:
        return Text(f"{ms:.0f} ms", style="yellow")
    return Text(f"{ms:.0f} ms", style="red")


def format_details(result: ServiceResult) -> str:
    """Extract a compact, markup-annotated details string from a result.

    Returns a Rich markup string tailored to each service's ``details``
    dict.  Falls back to the error message when no details are present.

    Args:
        result: Service health check result.

    Returns:
        Rich markup string suitable for a Table cell.

    Example:
        >>> r = ServiceResult("neo4j", ServiceStatus.HEALTHY,
        ...     details={"node_count": 47, "has_core_identity": True})
        >>> format_details(r)
        '47 nodes · CoreIdentity: [green]✓[/green]'
    """
    d: dict[str, Any] = result.details
    name = result.service_name

    # No details at all → fall back to error
    if not d:
        if result.error:
            return f"[red]{result.error[:70]}[/red]"
        return "[dim]—[/dim]"

    if name == "neo4j":
        count = d.get("node_count", "?")
        ci = d.get("has_core_identity")
        ci_str = "[green]✓[/green]" if ci else "[red]✗[/red]"
        return f"{count} nodes · CoreIdentity: {ci_str}"

    if name == "chromadb":
        found: list[str] = d.get("collections_found", [])
        missing: list[str] = d.get("missing_collections", [])
        total = 2
        if missing:
            return (
                f"[yellow]{len(found)}/{total} collections[/yellow]"
                f" · missing: {', '.join(missing)}"
            )
        return (
            f"[green]{len(found)}/{total} collections[/green]"
            " · ubik_episodic, ubik_semantic"
        )

    if name == "mcp":
        code = d.get("http_status", "?")
        return f"HTTP [dim]{code}[/dim]"

    if name == "vllm":
        n = d.get("models_loaded", 0)
        ids: list[str] = d.get("model_ids", [])
        if ids:
            short = ids[0].split("/")[-1][:32]
            return f"[green]{n}[/green] model · [dim]{short}[/dim]"
        return "[yellow]server up · no models loaded[/yellow]"

    if name == "tailscale":
        self_on: bool = d.get("self_online", False)
        so_on: bool = d.get("somatic_online", False)
        so_host: str = d.get("somatic_hostname", d.get("somatic_ip", "somatic"))
        peers: int | str = d.get("peer_count", "?")
        self_str = "[green]online[/green]" if self_on else "[red]offline[/red]"
        so_str = "[green]online[/green]" if so_on else "[red]offline[/red]"
        return f"self: {self_str} · {so_host}: {so_str} · {peers} peers"

    if name == "docker":
        containers: dict[str, str] = d.get("containers", {})
        running = sum(1 for st in containers.values() if st == "running")
        total_c = len(containers)
        color = "green" if running == total_c else "yellow"
        names_str = ", ".join(containers.keys())
        return f"[{color}]{running}/{total_c}[/{color}] running · {names_str}"

    # Generic fallback
    if result.error:
        return f"[red]{result.error[:70]}[/red]"
    items = list(d.items())[:3]
    return "  ".join(f"[dim]{k}:[/dim] {v}" for k, v in items)


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def make_service_table(cluster: ClusterHealth) -> Table:
    """Build a Rich Table showing one row per service health result.

    Rows appear in canonical order (:data:`_SERVICE_ORDER`); services
    absent from *cluster* are silently skipped so partial checks (via
    ``--service``) render correctly.

    Args:
        cluster: Aggregated health check snapshot.

    Returns:
        :class:`rich.table.Table` ready to print or embed in a layout.
    """
    table = Table(
        show_header=True,
        header_style="bold dim",
        box=box.SIMPLE,
        padding=(0, 1),
        expand=True,
    )

    table.add_column("Service", style="bold", min_width=10)
    table.add_column("Node", style="dim", min_width=12)
    table.add_column("Status", min_width=14)
    table.add_column("Latency", justify="right", min_width=9)
    table.add_column("Details")

    for svc_name in _SERVICE_ORDER:
        if svc_name not in cluster.services:
            continue
        result = cluster.services[svc_name]

        icon = _STATUS_ICONS[result.status]
        style = _STATUS_STYLES[result.status]
        status_cell = Text(f"{icon} {result.status.value.upper()}", style=style)
        node = _SERVICE_NODES.get(svc_name, "—")
        latency = _format_latency(result.latency_ms)
        details = format_details(result)

        table.add_row(svc_name, node, status_cell, latency, details)

    return table


# ---------------------------------------------------------------------------
# Dashboard renderable (watch mode)
# ---------------------------------------------------------------------------

def make_dashboard(
    cfg: AppConfig,
    cluster: ClusterHealth | None,
    cycle: int,
    next_in: float,
    *,
    checking: bool = False,
) -> RenderableType:
    """Build a full-screen dashboard renderable for the watch loop.

    Returns a :class:`rich.console.Group` containing a header Panel,
    the service table, and a footer summary line.  All content is
    rebuilt from scratch each call; the Live context handles diffing.

    Args:
        cfg: Application config — provides node IPs for the header.
        cluster: Latest health snapshot, or ``None`` on the very first
            cycle before any check has completed.
        cycle: Current watch cycle number (1-based).
        next_in: Seconds until the next check fires.  Used in the
            countdown display.
        checking: When ``True``, replaces the countdown with
            ``[checking...]`` to signal an active probe.

    Returns:
        A Rich renderable that ``Live.update()`` accepts directly.
    """
    h_ip = cfg.hippocampal.tailscale_ip
    s_ip = cfg.somatic.tailscale_ip

    # ── No results yet ────────────────────────────────────────────────────
    if cluster is None:
        return Panel(
            Text(
                f"  Running initial health check (cycle {cycle})...",
                style="dim",
            ),
            title="[bold blue]UBIK MAESTRO[/bold blue]",
            border_style="blue",
        )

    # ── Header ────────────────────────────────────────────────────────────
    overall = cluster.overall_status
    overall_style = _STATUS_STYLES[overall]
    overall_icon = _STATUS_ICONS[overall]

    checked_str = (
        cluster.checked_at
        .astimezone(timezone.utc)
        .strftime("%H:%M:%S UTC")
    )

    if checking:
        timing_str = "[dim]checking...[/dim]"
    elif next_in > 0:
        timing_str = f"next in [bold]{next_in:.0f}s[/bold]"
    else:
        timing_str = "[dim]—[/dim]"

    header_body = Text.from_markup(
        f"  [dim]Hippocampal:[/dim] {h_ip}  "
        f"[dim]Somatic:[/dim] {s_ip}\n"
        f"  Cycle [bold]{cycle}[/bold]  ·  "
        f"checked [dim]{checked_str}[/dim]  ·  "
        f"{timing_str}"
    )

    panel_title = (
        f"[bold blue]UBIK MAESTRO[/bold blue]  "
        f"[{overall_style}]{overall_icon} {overall.value.upper()}[/{overall_style}]"
    )

    header_panel = Panel(
        header_body,
        title=panel_title,
        border_style="blue",
        padding=(0, 1),
    )

    # ── Service table ─────────────────────────────────────────────────────
    table = make_service_table(cluster)

    # ── Footer ────────────────────────────────────────────────────────────
    h_count = len(cluster.healthy_services)
    t_count = len(cluster.services)
    unhealthy = cluster.unhealthy_services

    footer_parts = [
        f"  [{overall_style}]{overall_icon} "
        f"{overall.value.upper()}[/{overall_style}]",
        f"[bold]{h_count}/{t_count}[/bold] healthy",
    ]
    if unhealthy:
        footer_parts.append(f"[dim]issues:[/dim] {', '.join(unhealthy)}")

    footer = Text.from_markup("  ·  ".join(footer_parts))

    return Group(header_panel, table, footer, Text(""))


# ---------------------------------------------------------------------------
# One-shot print (check mode)
# ---------------------------------------------------------------------------

def print_check_results(
    cluster: ClusterHealth,
    cfg: AppConfig,
    console: Console,
) -> None:
    """Print a formatted one-shot health check report.

    Writes a header line, the service table, and a summary footer to
    *console*.  Does not call ``sys.exit`` — the caller decides the
    exit code.

    Args:
        cluster: Aggregated health check snapshot.
        cfg: Application config (used for context in the header).
        console: Rich Console instance to write output to.
    """
    overall = cluster.overall_status
    style = _STATUS_STYLES[overall]
    icon = _STATUS_ICONS[overall]
    ts = (
        cluster.checked_at
        .astimezone(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    console.print()
    console.print(
        f"  [bold blue]UBIK MAESTRO[/bold blue]  "
        f"[{style}]{icon} {overall.value.upper()}[/{style}]  "
        f"[dim]{ts}[/dim]"
    )
    console.print(
        f"  [dim]Hippocampal:[/dim] {cfg.hippocampal.tailscale_ip}  "
        f"[dim]Somatic:[/dim] {cfg.somatic.tailscale_ip}"
    )
    console.print()
    console.print(make_service_table(cluster))

    h = len(cluster.healthy_services)
    t = len(cluster.services)
    unhealthy = cluster.unhealthy_services

    footer = (
        f"  [{style}]{icon} {overall.value.upper()}[/{style}]"
        f"  ·  [bold]{h}/{t}[/bold] healthy"
    )
    if unhealthy:
        footer += f"  ·  [dim]issues:[/dim] {', '.join(unhealthy)}"

    console.print(Text.from_markup(footer))
    console.print()
