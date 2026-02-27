#!/usr/bin/env python3
"""
Maestro — Status Dashboard (TUI)

Real-time terminal UI displaying UBIK service health across both nodes.
Renders per-node service tables with color-coded status, auto-refresh
via ``rich.live.Live``, and optional keyboard shortcuts on POSIX terminals.

Layout::

    ┌─ UBIK MAESTRO — System Dashboard ────────────────────────────┐
    │  Last check: 2026-02-25 10:30:00 UTC                         │
    │  Running on: HIPPOCAMPAL (mac.lan)  next refresh in 30s      │
    └──────────────────────────────────────────────────────────────┘

    HIPPOCAMPAL NODE (Local)
     Service        Status      Latency    Since
     docker         ✓ UP         45ms      3h 12m
     neo4j          ✓ UP        120ms      3h 12m
     chromadb       ✓ UP         85ms      3h 12m
     mcp            ✓ UP         62ms      3h 12m

    SOMATIC NODE (Remote via Tailscale)
     Service        Status      Latency    Since
     vllm           ✗ DOWN        —          —

    NETWORK
     Connection     Status      Latency
     tailscale      ✓ UP         12ms

    [q] Quit  [r] Refresh  [s] Start All  [x] Shutdown

Keyboard shortcuts (POSIX terminals only — ``termios`` required):
    q   Quit the dashboard
    r   Force an immediate refresh
    s   Trigger :meth:`Orchestrator.ensure_all_running` for local services
    x   Stop all registered services (orderly shutdown)

Fallback behaviour:
    When ``rich`` is unavailable, :meth:`Dashboard.run` prints a single
    plain-text health summary and returns.

Public API:
    Dashboard          — configurable TUI dashboard class
    run_dashboard(...) — convenience wrapper; creates and runs a Dashboard

Author: UBIK Project
Version: 0.1.0
"""

import asyncio
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Optional

from maestro.config import AppConfig, get_config
from maestro.platform_detect import NodeType, detect_node
from maestro.services.health_runner import run_all_checks
from maestro.services.models import ClusterHealth, ServiceStatus

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from rich import box
    from rich.console import Console, Group, RenderableType
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False

try:
    import termios
    import tty
    import select as _select

    _TERMIOS_AVAILABLE = True
except ImportError:
    _TERMIOS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_INTERVAL: float = 30.0
_DEFAULT_TIMEOUT: float = 10.0
_KEY_POLL_INTERVAL: float = 0.1
_RENDER_REFRESH_PER_S: int = 4

# Canonical service grouping by node for display ordering.
_HIPPOCAMPAL_SERVICES: tuple[str, ...] = ("docker", "neo4j", "chromadb", "mcp")
_SOMATIC_SERVICES: tuple[str, ...] = ("vllm",)
_NETWORK_SERVICES: tuple[str, ...] = ("tailscale",)

_STATUS_STYLES: dict = {
    ServiceStatus.HEALTHY: "bold green",
    ServiceStatus.DEGRADED: "bold yellow",
    ServiceStatus.UNHEALTHY: "bold red",
}

_STATUS_LABELS: dict = {
    ServiceStatus.HEALTHY: ("✓", "UP"),
    ServiceStatus.DEGRADED: ("◑", "DEGRADED"),
    ServiceStatus.UNHEALTHY: ("✗", "DOWN"),
}


# ---------------------------------------------------------------------------
# Keyboard reader thread (POSIX only)
# ---------------------------------------------------------------------------


def _read_keys_thread(
    key_queue: "asyncio.Queue[str]",
    loop: "asyncio.AbstractEventLoop",
    stop_flag: threading.Event,
    fd: int,
) -> None:
    """Read single keystrokes from *fd* in cbreak terminal mode.

    Runs in a daemon thread.  Each character is pushed into *key_queue*
    via :meth:`~asyncio.AbstractEventLoop.call_soon_threadsafe` so it is
    safe to consume from the async event loop.

    Args:
        key_queue: Asyncio queue owned by *loop*.
        loop: Running event loop that owns *key_queue*.
        stop_flag: Set this event to request a clean exit.
        fd: File descriptor to read from (normally ``sys.stdin.fileno()``).

    Note:
        Uses ``tty.setcbreak`` (not ``setraw``) so ``Ctrl+C`` still raises
        ``SIGINT`` and terminal signals remain active.
    """
    if not _TERMIOS_AVAILABLE:  # pragma: no cover
        return

    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop_flag.is_set():
            ready, _, _ = _select.select([fd], [], [], _KEY_POLL_INTERVAL)
            if ready:
                try:
                    ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                    if ch:
                        loop.call_soon_threadsafe(key_queue.put_nowait, ch)
                except OSError:
                    break
    except Exception:
        pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_latency(ms: float | None) -> "Text":
    """Format a latency value as a coloured Rich Text.

    Args:
        ms: Latency in milliseconds, or ``None`` if unavailable.

    Returns:
        Green text for fast (<50 ms), yellow for moderate (<200 ms),
        red for slow (≥200 ms), dim dash when *ms* is ``None``.
    """
    if ms is None:
        return Text("—", style="dim")
    if ms < 50:
        return Text(f"{ms:.0f}ms", style="green")
    if ms < 200:
        return Text(f"{ms:.0f}ms", style="yellow")
    return Text(f"{ms:.0f}ms", style="red")


def _fmt_since(dt: datetime | None) -> "Text":
    """Format a state-onset datetime as elapsed time from now.

    Args:
        dt: UTC datetime when the current service state began, or ``None``.

    Returns:
        Text like ``"3h 12m"``, ``"45m"``, ``"30s"``, or a dim dash
        when *dt* is ``None``.
    """
    if dt is None:
        return Text("—", style="dim")
    total = int((datetime.now(timezone.utc) - dt).total_seconds())
    if total < 60:
        return Text(f"{total}s", style="dim")
    if total < 3600:
        return Text(f"{total // 60}m", style="dim")
    hours = total // 3600
    minutes = (total % 3600) // 60
    return Text(f"{hours}h {minutes}m", style="dim")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class Dashboard:
    """Rich TUI dashboard for UBIK Maestro.

    Renders per-node service tables with color-coded health indicators
    inside a ``rich.live.Live`` auto-refresh loop.  Keyboard shortcuts
    allow quitting, forcing a refresh, starting services, and shutdown.

    Args:
        cfg: Application configuration.  Loaded via :func:`get_config`
            on first use when ``None``.
        interval: Seconds between auto-refresh cycles.  Default 30.
        timeout: Per-check network timeout in seconds.  Default 10.
        console: Rich Console to render to.  Created automatically
            when ``None``.

    Example::

        from maestro.dashboard import Dashboard
        import asyncio

        async def main():
            await Dashboard(interval=30).run_async()

        asyncio.run(main())
    """

    def __init__(
        self,
        cfg: AppConfig | None = None,
        *,
        interval: float = _DEFAULT_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        console: Optional["Console"] = None,
    ) -> None:
        self._cfg = cfg
        self._interval = interval
        self._timeout = timeout
        self._console = console

        self._cluster: ClusterHealth | None = None
        self._checking: bool = False
        self._last_check: datetime | None = None

        # "Since" column: maps service name → datetime when its current state began.
        self._state_start: dict[str, datetime] = {}
        self._last_healthy: dict[str, bool] = {}

        # Async queue for keyboard input (populated in run_async).
        self._key_queue: Optional["asyncio.Queue[str]"] = None
        self._action_message: str | None = None

        try:
            self._identity = detect_node()
        except Exception:
            self._identity = None

    # ── State tracking ──────────────────────────────────────────────────────

    def _update_state_tracking(self, cluster: ClusterHealth) -> None:
        """Record onset timestamps whenever a service changes health state.

        Compares each service's current healthy/unhealthy state against the
        last known state.  Records :attr:`ServiceResult.checked_at` as the
        onset time whenever the state differs from the previous check.

        Args:
            cluster: Freshly retrieved cluster snapshot.
        """
        for name, result in cluster.services.items():
            healthy = result.status == ServiceStatus.HEALTHY
            if self._last_healthy.get(name) != healthy:
                self._state_start[name] = result.checked_at
            self._last_healthy[name] = healthy

    def _since(self, name: str) -> datetime | None:
        """Return when *name* entered its current health state, or ``None``."""
        return self._state_start.get(name)

    # ── Table builders ───────────────────────────────────────────────────────

    def _service_table(
        self,
        names: tuple[str, ...],
        *,
        dim: bool = False,
    ) -> "Table":
        """Build a service status table for a set of service names.

        Args:
            names: Service names to include, in display order.
            dim: When ``True``, each data row is dimmed — used to indicate
                services on the remote node.

        Returns:
            Rich Table with Service / Status / Latency / Since columns.
        """
        table = Table(
            show_header=True,
            header_style="bold dim",
            box=box.SIMPLE,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Service", style="bold", min_width=14)
        table.add_column("Status", min_width=12)
        table.add_column("Latency", justify="right", min_width=9)
        table.add_column("Since", min_width=10)

        for name in names:
            if self._cluster is None or name not in self._cluster.services:
                table.add_row(
                    name,
                    Text("? UNKNOWN", style="dim"),
                    Text("—", style="dim"),
                    Text("—", style="dim"),
                )
                continue

            result = self._cluster.services[name]
            icon, label = _STATUS_LABELS.get(result.status, ("?", "UNKNOWN"))
            style = _STATUS_STYLES.get(result.status, "dim")
            row_style = "dim" if dim else ""
            table.add_row(
                name,
                Text(f"{icon} {label}", style=style),
                _fmt_latency(result.latency_ms),
                _fmt_since(self._since(name)),
                style=row_style,
            )

        return table

    def _network_table(self) -> "Table":
        """Build the network status table (Tailscale row).

        Returns:
            Rich Table with Connection / Status / Latency columns.
        """
        table = Table(
            show_header=True,
            header_style="bold dim",
            box=box.SIMPLE,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Connection", style="bold", min_width=14)
        table.add_column("Status", min_width=12)
        table.add_column("Latency", justify="right", min_width=9)

        for name in _NETWORK_SERVICES:
            if self._cluster is None or name not in self._cluster.services:
                table.add_row(
                    name,
                    Text("? UNKNOWN", style="dim"),
                    Text("—", style="dim"),
                )
                continue

            result = self._cluster.services[name]
            icon, label = _STATUS_LABELS.get(result.status, ("?", "UNKNOWN"))
            style = _STATUS_STYLES.get(result.status, "dim")
            table.add_row(
                name,
                Text(f"{icon} {label}", style=style),
                _fmt_latency(result.latency_ms),
            )

        return table

    # ── Full renderable ──────────────────────────────────────────────────────

    def render(self) -> "RenderableType":
        """Build the complete dashboard renderable from current state.

        Pure function: reads only from instance state, performs no I/O.
        Called on every Live refresh cycle; rebuilding from scratch each
        time is intentional (Rich handles efficient diffing).

        Returns:
            Rich ``Group`` containing header, two node sections, network
            section, and footer keyboard-shortcut hint.
        """
        last_check_str = (
            self._last_check.strftime("%Y-%m-%d %H:%M:%S UTC")
            if self._last_check
            else "—"
        )
        node_str = (
            self._identity.node_type.value.upper() if self._identity else "UNKNOWN"
        )
        hostname = self._identity.hostname if self._identity else "unknown"

        if self._action_message:
            timing_str = f"[yellow]{self._action_message}[/yellow]"
        elif self._checking:
            timing_str = "[dim]checking...[/dim]"
        else:
            timing_str = f"[dim]next refresh in {self._interval:.0f}s[/dim]"

        header = Panel(
            Text.from_markup(
                f"  Last check: [dim]{last_check_str}[/dim]\n"
                f"  Running on: [bold]{node_str}[/bold] "
                f"[dim]({hostname})[/dim]  {timing_str}"
            ),
            title="[bold blue]UBIK MAESTRO — System Dashboard[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )

        # Determine local vs remote orientation
        local_node = self._identity.node_type if self._identity else NodeType.UNKNOWN
        hippo_is_local = local_node in (NodeType.HIPPOCAMPAL, NodeType.UNKNOWN)
        somatic_is_local = local_node == NodeType.SOMATIC

        hippo_label = (
            "HIPPOCAMPAL NODE [dim](Local)[/dim]"
            if hippo_is_local
            else "HIPPOCAMPAL NODE [dim](Remote via Tailscale)[/dim]"
        )
        somatic_label = (
            "SOMATIC NODE [dim](Local)[/dim]"
            if somatic_is_local
            else "SOMATIC NODE [dim](Remote via Tailscale)[/dim]"
        )

        hippo_section = Group(
            Text.from_markup(f"\n  [bold]{hippo_label}[/bold]"),
            self._service_table(_HIPPOCAMPAL_SERVICES, dim=not hippo_is_local),
        )
        somatic_section = Group(
            Text.from_markup(f"\n  [bold]{somatic_label}[/bold]"),
            self._service_table(_SOMATIC_SERVICES, dim=not somatic_is_local),
        )
        network_section = Group(
            Text("\n  NETWORK", style="bold"),
            self._network_table(),
        )
        footer = Text.from_markup(
            "\n  [dim][q] Quit  [r] Refresh  [s] Start All  [x] Shutdown[/dim]"
        )

        return Group(header, hippo_section, somatic_section, network_section, footer)

    # ── Async actions ────────────────────────────────────────────────────────

    async def _do_check(self) -> None:
        """Run a full health check and update cluster state.

        Sets :attr:`_checking` to ``True`` while the check is in progress
        so ``render()`` can display a "checking..." indicator.  Any
        exception is captured into :attr:`_action_message`.
        """
        cfg = self._cfg or get_config()
        self._checking = True
        self._action_message = None
        try:
            cluster = await run_all_checks(cfg, timeout=self._timeout)
            self._cluster = cluster
            self._last_check = cluster.checked_at
            self._update_state_tracking(cluster)
        except Exception as exc:
            self._action_message = f"Check failed: {exc}"
        finally:
            self._checking = False

    async def _handle_key(self, key: str) -> str | None:
        """Process a single keypress and return an action string or ``None``.

        Args:
            key: Single character received from the keyboard thread.

        Returns:
            ``"quit"`` to exit the loop, ``"refresh"`` to skip the countdown
            and run an immediate check, ``None`` for actions handled
            internally (start-all, shutdown).
        """
        key_lower = key.lower()
        if key_lower == "q":
            return "quit"
        if key_lower == "r":
            self._action_message = None
            return "refresh"
        if key_lower == "s":
            self._action_message = "Starting unhealthy services..."
            asyncio.create_task(self._start_all())
            return None
        if key_lower == "x":
            self._action_message = "Shutting down all services..."
            asyncio.create_task(self._shutdown_all())
            return None
        return None

    async def _start_all(self) -> None:
        """Trigger :meth:`~Orchestrator.ensure_all_running` for local services."""
        try:
            from maestro.orchestrator import Orchestrator
            from maestro.services import ServiceRegistry

            cfg = self._cfg or get_config()
            identity = self._identity or detect_node()
            orch = Orchestrator(ServiceRegistry(cfg), identity)
            failed = await orch.ensure_all_running()
            self._action_message = (
                f"Failed: {', '.join(failed)}" if failed else "All services started."
            )
        except Exception as exc:
            self._action_message = f"Start error: {exc}"

    async def _shutdown_all(self) -> None:
        """Stop all registered services in reverse dependency order."""
        try:
            from maestro.services import ServiceRegistry

            cfg = self._cfg or get_config()
            registry = ServiceRegistry(cfg)
            failed: list[str] = []
            for svc in reversed(registry.get_all()):
                if not await svc.stop():
                    failed.append(svc.name)
            self._action_message = (
                f"Shutdown partial — failed: {', '.join(failed)}"
                if failed
                else "All services stopped."
            )
        except Exception as exc:
            self._action_message = f"Shutdown error: {exc}"

    # ── Main run loop ────────────────────────────────────────────────────────

    async def run_async(self) -> None:
        """Async dashboard loop — runs until the user presses 'q'.

        Performs an initial health check, then alternates between a
        countdown period (during which keypresses are polled) and a
        scheduled health check.  A keyboard daemon thread is started on
        POSIX terminals (``termios`` available + stdin is a tty).

        Raises:
            KeyboardInterrupt: Propagated unchanged so the calling
                :func:`asyncio.run` and CLI layer can print "Stopped.".
        """
        console = self._console or Console()
        self._key_queue: asyncio.Queue[str] = asyncio.Queue()

        stop_flag = threading.Event()
        kbd_thread: threading.Thread | None = None

        if _TERMIOS_AVAILABLE and sys.stdin.isatty():
            loop = asyncio.get_running_loop()
            kbd_thread = threading.Thread(
                target=_read_keys_thread,
                args=(self._key_queue, loop, stop_flag, sys.stdin.fileno()),
                daemon=True,
                name="maestro-kbd",
            )
            kbd_thread.start()

        try:
            with Live(
                self.render(),
                console=console,
                refresh_per_second=_RENDER_REFRESH_PER_S,
                transient=False,
            ) as live:
                # Initial health check
                await self._do_check()
                live.update(self.render())

                while True:
                    # ── Countdown to next scheduled check ─────────────────
                    event_loop = asyncio.get_running_loop()
                    next_at = event_loop.time() + self._interval
                    skip_countdown = False

                    while not skip_countdown:
                        # Drain keyboard queue
                        while not self._key_queue.empty():
                            key = await self._key_queue.get()
                            action = await self._handle_key(key)
                            live.update(self.render())
                            if action == "quit":
                                return
                            if action == "refresh":
                                skip_countdown = True
                                break

                        if skip_countdown:
                            break

                        remaining = next_at - event_loop.time()
                        if remaining <= 0:
                            break

                        live.update(self.render())
                        await asyncio.sleep(min(0.3, remaining))

                    # ── Scheduled or forced check ──────────────────────────
                    await self._do_check()
                    live.update(self.render())

        finally:
            stop_flag.set()
            if kbd_thread is not None:
                kbd_thread.join(timeout=1.0)

    def run(self) -> None:
        """Run the dashboard synchronously (blocks until 'q' or Ctrl+C).

        Falls back to a one-shot plain-text health summary when ``rich``
        is not installed.
        """
        if not _RICH_AVAILABLE:  # pragma: no cover
            self._run_plain()
            return
        asyncio.run(self.run_async())

    def _run_plain(self) -> None:
        """Print a one-shot plain-text health summary (fallback without rich).

        Runs a single health check, prints a formatted table to stdout,
        and returns.  Never raises.
        """
        cfg = self._cfg or get_config()
        try:
            cluster = asyncio.run(run_all_checks(cfg, timeout=self._timeout))
        except Exception as exc:
            print(f"Health check failed: {exc}", file=sys.stderr)
            return

        print()
        print("UBIK MAESTRO — System Status")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"Time: {now}")
        print("-" * 60)
        for name, result in cluster.services.items():
            status = "UP" if result.status == ServiceStatus.HEALTHY else "DOWN"
            latency = (
                f"{result.latency_ms:.0f}ms"
                if result.latency_ms is not None
                else "—"
            )
            print(f"  {name:<14} {status:<8} {latency}")
        print("-" * 60)
        h = len(cluster.healthy_services)
        t = len(cluster.services)
        print(f"  {h}/{t} services healthy")
        print()


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def run_dashboard(
    cfg: AppConfig | None = None,
    *,
    interval: float = _DEFAULT_INTERVAL,
    timeout: float = _DEFAULT_TIMEOUT,
    console: Optional["Console"] = None,
) -> None:
    """Create and run a :class:`Dashboard`.

    Args:
        cfg: Application configuration.  Resolved via :func:`get_config`
            when ``None``.
        interval: Seconds between auto-refresh cycles.
        timeout: Per-check network timeout in seconds.
        console: Rich Console for output.

    Note:
        Blocks until the user presses 'q' or sends ``KeyboardInterrupt``.
    """
    Dashboard(cfg, interval=interval, timeout=timeout, console=console).run()
