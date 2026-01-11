"""Ollama-compatible CLI for mlx-lama."""

import typer
from rich.console import Console
from rich.table import Table
from typing import Optional

from . import __version__
from .config import get_config

app = typer.Typer(
    name="mlx-lama",
    help="Ollama-compatible CLI for MLX models on Apple Silicon",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"mlx-lama version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """mlx-lama: Run LLMs on Apple Silicon with MLX."""
    pass


@app.command()
def pull(
    model: str = typer.Argument(..., help="Model name (e.g., qwen-coder:32b)"),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Preferred backend for this model"
    ),
) -> None:
    """Download a model from HuggingFace."""
    from .models import pull_model

    pull_model(model, backend=backend)


@app.command()
def run(
    model: str = typer.Argument(..., help="Model name to run"),
    prompt: Optional[str] = typer.Argument(None, help="Prompt (omit for interactive mode)"),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Backend to use"
    ),
) -> None:
    """Run a model - interactive chat or one-shot generation."""
    from .models import run_model

    run_model(model, prompt=prompt, backend=backend)


@app.command()
def serve(
    model: str = typer.Argument(..., help="Model name to serve"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Host to bind to"),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Backend to use"
    ),
) -> None:
    """Start an OpenAI-compatible API server."""
    from .models import serve_model

    serve_model(model, host=host, port=port, backend=backend)


@app.command("list")
def list_models() -> None:
    """List downloaded models."""
    from .models import get_downloaded_models

    models = get_downloaded_models()

    if not models:
        console.print("[dim]No models downloaded yet.[/dim]")
        console.print("\nRun [bold]mlx-lama pull <model>[/bold] to download a model.")
        return

    table = Table(title="Downloaded Models")
    table.add_column("Name", style="cyan")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Modified", style="dim")
    table.add_column("Backend", style="magenta")

    for model in models:
        table.add_row(
            model["name"],
            model["size"],
            model["modified"],
            model.get("backend", "-"),
        )

    console.print(table)


@app.command()
def rm(
    model: str = typer.Argument(..., help="Model name to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Remove a downloaded model."""
    from .models import remove_model

    remove_model(model, force=force)


@app.command()
def show(
    model: str = typer.Argument(..., help="Model name to show info for"),
) -> None:
    """Show model information."""
    from .models import show_model

    show_model(model)


@app.command()
def ps() -> None:
    """Show running models."""
    from .process import get_running_servers

    servers = get_running_servers()

    if not servers:
        console.print("[dim]No models currently running.[/dim]")
        return

    table = Table(title="Running Models")
    table.add_column("Model", style="cyan")
    table.add_column("PID", style="dim")
    table.add_column("Port", style="green")
    table.add_column("Backend", style="magenta")
    table.add_column("Uptime", style="dim")

    for server in servers:
        table.add_row(
            server["model"],
            str(server["pid"]),
            str(server["port"]),
            server["backend"],
            server["uptime"],
        )

    console.print(table)


@app.command()
def stop(
    model: Optional[str] = typer.Argument(None, help="Model to stop (stops all if omitted)"),
) -> None:
    """Stop running model server(s)."""
    from .process import stop_server

    stop_server(model)


@app.command()
def backends() -> None:
    """List available backends."""
    from .backends import get_available_backends

    available = get_available_backends()

    table = Table(title="Available Backends")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Batching", justify="center")
    table.add_column("Vision", justify="center")
    table.add_column("Description")

    for backend in available:
        status = "[green]available[/green]" if backend["available"] else "[red]not installed[/red]"
        batching = "[green]✓[/green]" if backend["batching"] else "[dim]-[/dim]"
        vision = "[green]✓[/green]" if backend["vision"] else "[dim]-[/dim]"

        table.add_row(
            backend["name"],
            status,
            batching,
            vision,
            backend["description"],
        )

    console.print(table)

    config = get_config()
    console.print(f"\n[dim]Default backend:[/dim] [bold]{config.default_backend}[/bold]")


if __name__ == "__main__":
    app()
