"""Ollama-compatible CLI for mlx-lama."""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
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
def backends(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info"),
) -> None:
    """List available backends and their status."""
    from .backends import get_available_backends

    available = get_available_backends()

    table = Table(title="Backends")
    table.add_column("Name", style="cyan")
    table.add_column("Status")
    table.add_column("Detection", style="dim")
    table.add_column("Batching", justify="center")
    table.add_column("Vision", justify="center")
    table.add_column("Description")

    for backend in available:
        if backend["available"]:
            status = "[green]✓ available[/green]"
            detection = backend.get("details", backend.get("method", ""))
            if backend.get("version"):
                detection = f"{detection} ({backend['version']})"
        else:
            status = "[red]✗ not installed[/red]"
            detection = backend.get("details", "")

        batching = "[green]✓[/green]" if backend["batching"] else "[dim]-[/dim]"
        vision = "[green]✓[/green]" if backend["vision"] else "[dim]-[/dim]"

        table.add_row(
            backend["name"],
            status,
            detection[:40] if detection else "",
            batching,
            vision,
            backend["description"],
        )

    console.print(table)

    # Show install hints for unavailable backends
    unavailable = [b for b in available if not b["available"]]
    if unavailable and verbose:
        console.print("\n[bold]Installation options:[/bold]")
        for backend in unavailable:
            if backend.get("install_options"):
                console.print(f"\n[cyan]{backend['name']}[/cyan]:")
                for opt in backend["install_options"]:
                    console.print(f"  [dim]{opt.description}:[/dim]")
                    console.print(f"    [green]{opt.command}[/green]")

    config = get_config()
    console.print(f"\n[dim]Default backend:[/dim] [bold]{config.default_backend}[/bold]")

    if unavailable:
        console.print(
            "\n[dim]Tip: Run [/dim][bold]mlx-lama install <backend>[/bold][dim] "
            "to install a backend[/dim]"
        )


@app.command()
def install(
    backend: str = typer.Argument(..., help="Backend to install (mlx-lm, vllm, ollama)"),
    method: Optional[str] = typer.Option(
        None,
        "--method",
        "-m",
        help="Installation method (uv, pip, brew, manual)",
    ),
) -> None:
    """Install a backend."""
    from .backends import get_available_backends, install_backend
    from .progress import print_success, print_error, print_info

    # Check if already installed
    backends_list = get_available_backends()
    for b in backends_list:
        if b["name"] == backend:
            if b["available"]:
                console.print(
                    f"[yellow]Backend '{backend}' is already installed[/yellow]"
                )
                console.print(f"[dim]Detected via: {b.get('details', 'unknown')}[/dim]")
                return
            break
    else:
        print_error(f"Unknown backend: {backend}")
        console.print("\nAvailable backends: mlx-lm, vllm, ollama")
        raise typer.Exit(1)

    # Show available install methods
    for b in backends_list:
        if b["name"] == backend and b.get("install_options"):
            if not method:
                console.print(f"\n[bold]Installing {backend}...[/bold]\n")
                console.print("[dim]Available methods:[/dim]")
                for i, opt in enumerate(b["install_options"]):
                    marker = "[green]→[/green]" if i == 0 else " "
                    console.print(
                        f"  {marker} {opt.method.value}: {opt.description}"
                    )
                console.print()

            try:
                success = install_backend(backend, method=method)
                if success:
                    print_success(f"Backend '{backend}' installed successfully!")
                    print_info("Run 'mlx-lama backends' to verify.")
                else:
                    print_error(f"Failed to install '{backend}'")
                    raise typer.Exit(1)
            except ValueError as e:
                print_error(str(e))
                raise typer.Exit(1)
            break


if __name__ == "__main__":
    app()
