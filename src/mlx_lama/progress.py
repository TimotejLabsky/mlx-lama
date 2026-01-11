"""Progress indicators and formatting utilities."""

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from contextlib import contextmanager
from typing import Generator


console = Console()


def create_download_progress() -> Progress:
    """Create a progress bar for downloads."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def create_spinner_progress() -> Progress:
    """Create a simple spinner progress."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
    )


@contextmanager
def status_spinner(message: str) -> Generator[None, None, None]:
    """Context manager for a simple status spinner."""
    with console.status(f"[bold blue]{message}"):
        yield


def print_model_info(
    name: str,
    repo: str,
    size: str = "unknown",
    quantization: str = "unknown",
    backend: str = "mlx-lm",
    description: str = "",
) -> None:
    """Print formatted model information."""
    text = Text()
    text.append(f"Model: ", style="dim")
    text.append(f"{name}\n", style="bold cyan")
    text.append(f"Repo: ", style="dim")
    text.append(f"{repo}\n", style="blue")
    text.append(f"Size: ", style="dim")
    text.append(f"{size}\n", style="green")
    text.append(f"Quantization: ", style="dim")
    text.append(f"{quantization}\n", style="yellow")
    text.append(f"Backend: ", style="dim")
    text.append(f"{backend}\n", style="magenta")

    if description:
        text.append(f"Description: ", style="dim")
        text.append(f"{description}", style="white")

    console.print(Panel(text, title="Model Info", border_style="blue"))


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]![/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_tokens_per_sec(tokens: float) -> str:
    """Format tokens per second."""
    return f"{tokens:.1f} tok/s"
