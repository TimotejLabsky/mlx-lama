"""Live TUI display for server monitoring."""

import threading

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .stats import (
    HardwareStats,
    InferenceStats,
    LogEntry,
    StatsCollector,
    get_stats_collector,
)


def format_tokens(n: int) -> str:
    """Format token count with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_time(ms: float) -> str:
    """Format milliseconds to human readable."""
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f}ms"


def create_progress_bar(
    percent: float,
    width: int = 20,
    filled_color: str = "green",
    empty_color: str = "grey37",
) -> Text:
    """Create a colored progress bar."""
    filled = int(percent / 100 * width)
    empty = width - filled

    bar = Text()
    bar.append("█" * filled, style=filled_color)
    bar.append("░" * empty, style=empty_color)
    return bar


def build_header(model: str, backend: str, host: str, port: int) -> Panel:
    """Build the header panel."""
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left")
    table.add_column(justify="left")
    table.add_column(justify="right")
    table.add_column(justify="left")

    table.add_row(
        Text("Model:", style="dim"),
        Text(model, style="cyan bold"),
        Text("Backend:", style="dim"),
        Text(backend, style="magenta"),
    )
    table.add_row(
        Text("Endpoint:", style="dim"),
        Text(f"http://{host}:{port}", style="blue underline"),
        Text("Status:", style="dim"),
        Text("● Running", style="green bold"),
    )

    return Panel(table, title="[bold]mlx-lama server[/bold]", border_style="blue")


def build_inference_panel(stats: InferenceStats) -> Panel:
    """Build the inference stats panel."""
    table = Table.grid(padding=(0, 3))
    table.add_column(justify="left")
    table.add_column(justify="right", style="cyan")
    table.add_column(justify="left")
    table.add_column(justify="right", style="cyan")
    table.add_column(justify="left")
    table.add_column(justify="right", style="cyan")

    # Row 1: Request counts
    active_style = "yellow bold" if stats.active_requests > 0 else "white"
    queue_style = "red bold" if stats.queued_requests > 0 else "white"
    table.add_row(
        Text("Requests:", style="dim"),
        Text(str(stats.total_requests), style="white bold"),
        Text("Active:", style="dim"),
        Text(str(stats.active_requests), style=active_style),
        Text("Queue:", style="dim"),
        Text(str(stats.queued_requests), style=queue_style),
    )

    # Row 2: Performance metrics
    tps = stats.tokens_per_second
    tps_style = "green bold" if tps > 30 else "yellow bold" if tps > 10 else "white"
    table.add_row(
        Text("Tokens/s:", style="dim"),
        Text(f"{stats.tokens_per_second:.1f}", style=tps_style),
        Text("Avg latency:", style="dim"),
        Text(format_time(stats.avg_latency_ms), style="white"),
        "",
        "",
    )

    # Row 3: Token counts
    total_tokens = stats.total_prompt_tokens + stats.total_completion_tokens
    table.add_row(
        Text("Total tokens:", style="dim"),
        Text(format_tokens(total_tokens), style="white"),
        Text("Prompt:", style="dim"),
        Text(format_tokens(stats.total_prompt_tokens), style="white"),
        Text("Completion:", style="dim"),
        Text(format_tokens(stats.total_completion_tokens), style="white"),
    )

    return Panel(table, title="[bold green]Inference[/bold green]", border_style="green")


def build_hardware_panel(stats: HardwareStats) -> Panel:
    """Build the hardware stats panel."""
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left", width=12)
    table.add_column(justify="right", width=22)
    table.add_column(justify="left", width=20)
    table.add_column(justify="right", width=16)

    # Memory bar color based on usage
    mp = stats.memory_percent
    mem_color = "green" if mp < 70 else "yellow" if mp < 90 else "red"
    mem_bar = create_progress_bar(mp, width=16, filled_color=mem_color)

    # Memory min/max display
    mem_minmax = ""
    if stats.memory_min_gb > 0 and stats.memory_max_gb > 0:
        mem_minmax = f"↓{stats.memory_min_gb:.1f} ↑{stats.memory_max_gb:.1f}"

    table.add_row(
        Text("Memory:", style="dim"),
        Text(f"{stats.memory_used_gb:.1f} / {stats.memory_total_gb:.0f} GB", style="white"),
        mem_bar,
        Text(mem_minmax, style="dim") if mem_minmax else Text(f"{mp:.0f}%", style=mem_color),
    )

    # CPU bar
    cpu_color = "green" if stats.cpu_percent < 70 else "yellow" if stats.cpu_percent < 90 else "red"
    cpu_bar = create_progress_bar(stats.cpu_percent, width=16, filled_color=cpu_color)

    table.add_row(
        Text("CPU:", style="dim"),
        Text(f"{stats.cpu_percent:.1f}%", style="white"),
        cpu_bar,
        Text(f"{stats.cpu_percent:.0f}%", style=cpu_color),
    )

    # GPU
    gpu_percent = stats.gpu_percent if stats.gpu_percent > 0 else 0
    gpu_color = "green" if gpu_percent < 70 else "yellow" if gpu_percent < 90 else "red"
    gpu_bar = create_progress_bar(gpu_percent, width=16, filled_color=gpu_color)

    gpu_label = f"{gpu_percent:.0f}%" if gpu_percent > 0 else "N/A"
    gpu_minmax = ""
    if stats.gpu_min > 0 and stats.gpu_max > 0:
        gpu_minmax = f"↓{stats.gpu_min:.0f}% ↑{stats.gpu_max:.0f}%"

    table.add_row(
        Text("GPU:", style="dim"),
        Text(gpu_label, style="white" if gpu_percent > 0 else "dim"),
        gpu_bar if gpu_percent > 0 else Text("░" * 16, style="grey37"),
        Text(gpu_minmax, style="dim") if gpu_minmax else Text(gpu_label, style=gpu_color),
    )

    # Temperature and power (if available)
    extras = []
    if stats.temperature_c:
        tc = stats.temperature_c
        temp_style = "green" if tc < 70 else "yellow" if tc < 85 else "red"
        extras.append(Text(f"Temp: {stats.temperature_c:.0f}°C", style=temp_style))
    if stats.power_w:
        extras.append(Text(f"Power: {stats.power_w:.0f}W", style="white"))

    if extras:
        table.add_row(
            Text("Thermal:", style="dim"),
            extras[0] if extras else "",
            extras[1] if len(extras) > 1 else "",
            "",
        )

    return Panel(table, title="[bold yellow]Hardware[/bold yellow]", border_style="yellow")


def build_requests_panel(stats: InferenceStats) -> Panel:
    """Build the recent requests panel."""
    table = Table(
        show_header=True,
        header_style="bold dim",
        box=None,
        padding=(0, 1),
    )
    table.add_column("Time", style="dim", width=8)
    table.add_column("Endpoint", width=20)
    table.add_column("Tokens", justify="right", width=8)
    table.add_column("Latency", justify="right", width=8)
    table.add_column("Status", justify="center", width=6)

    for req in stats.recent_requests[:5]:
        time_str = req.started_at.strftime("%H:%M:%S")
        tokens = req.prompt_tokens + req.completion_tokens
        latency = format_time(req.latency_ms)

        if req.status == "completed":
            status = Text("✓", style="green")
        elif req.status == "error":
            status = Text("✗", style="red")
        elif req.status == "streaming":
            status = Text("◌", style="yellow")
        else:
            status = Text("○", style="dim")

        table.add_row(
            time_str,
            req.endpoint[:20],
            str(tokens),
            latency,
            status,
        )

    # Fill empty rows
    for _ in range(5 - len(stats.recent_requests)):
        table.add_row("", "", "", "", "")

    return Panel(table, title="[bold cyan]Recent Requests[/bold cyan]", border_style="cyan")


def build_logs_panel(logs: list[LogEntry], max_lines: int = 8) -> Panel:
    """Build the backend logs panel."""
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Time", style="dim", width=8, no_wrap=True)
    table.add_column("Message", overflow="ellipsis", no_wrap=True)

    # Get recent logs
    recent = logs[-max_lines:] if len(logs) > max_lines else logs

    for entry in recent:
        time_str = entry.timestamp.strftime("%H:%M:%S")

        # Color based on level
        if entry.level == "error":
            style = "red"
        elif entry.level == "warning":
            style = "yellow"
        elif entry.level == "debug":
            style = "dim"
        else:
            style = "white"

        # Truncate long messages
        msg = entry.message
        if len(msg) > 80:
            msg = msg[:77] + "..."

        table.add_row(time_str, Text(msg, style=style))

    # Fill empty rows
    for _ in range(max_lines - len(recent)):
        table.add_row("", "")

    return Panel(table, title="[bold magenta]Backend Logs[/bold magenta]", border_style="magenta")


def build_display(
    model: str,
    backend: str,
    host: str,
    port: int,
    collector: StatsCollector,
) -> Group:
    """Build the complete display."""
    inference_stats = collector.get_inference_stats()
    hardware_stats = collector.get_hardware_stats()
    log_entries = collector.logs.get_entries()

    return Group(
        build_header(model, backend, host, port),
        build_inference_panel(inference_stats),
        build_hardware_panel(hardware_stats),
        build_requests_panel(inference_stats),
        build_logs_panel(log_entries),
        Text("Press Ctrl+C to stop", style="dim", justify="center"),
    )


class LiveTop:
    """Live monitoring display for the server."""

    def __init__(
        self,
        model: str,
        backend: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        refresh_rate: float = 2.0,
    ):
        self.model = model
        self.backend = backend
        self.host = host
        self.port = port
        self.refresh_rate = refresh_rate
        self.collector = get_stats_collector()
        self._stop_event = threading.Event()
        self._live: Live | None = None

    def start(self, console: Console | None = None) -> Live:
        """Start the live display and return the Live context."""
        if console is None:
            console = Console()

        self._live = Live(
            build_display(
                self.model,
                self.backend,
                self.host,
                self.port,
                self.collector,
            ),
            console=console,
            refresh_per_second=self.refresh_rate,
            screen=True,  # Use alternate screen (like htop)
        )
        return self._live

    def update(self) -> None:
        """Update the display."""
        if self._live:
            self._live.update(
                build_display(
                    self.model,
                    self.backend,
                    self.host,
                    self.port,
                    self.collector,
                )
            )

    def stop(self) -> None:
        """Stop the live display."""
        self._stop_event.set()
        if self._live:
            self._live.stop()
