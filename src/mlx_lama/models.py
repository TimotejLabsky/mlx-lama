"""Model management - pull, list, remove, show, run, serve."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, HfApi
from rich.console import Console
from rich.prompt import Confirm

from .config import get_config
from .registry import get_registry
from .progress import (
    create_download_progress,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_model_info,
    format_size,
    status_spinner,
)

console = Console()


def get_model_path(model: str) -> Path:
    """Get the local path for a model."""
    config = get_config()
    registry = get_registry()

    # Resolve alias to repo
    repo = registry.resolve(model)

    # Create safe directory name from repo
    safe_name = repo.replace("/", "--")
    return config.models_dir / safe_name


def is_model_downloaded(model: str) -> bool:
    """Check if a model is already downloaded."""
    path = get_model_path(model)
    return path.exists() and (path / "config.json").exists()


def pull_model(model: str, backend: Optional[str] = None) -> Path:
    """Download a model from HuggingFace.

    Args:
        model: Model alias or HuggingFace repo
        backend: Preferred backend (saved in model config)

    Returns:
        Path to the downloaded model
    """
    registry = get_registry()
    config = get_config()

    # Resolve alias to repo
    model_info = registry.get(model)
    if model_info:
        repo = model_info.repo
        print_info(f"Resolving [bold]{model}[/bold] → [blue]{repo}[/blue]")
    else:
        repo = model
        print_info(f"Using repo: [blue]{repo}[/blue]")

    # Check if already downloaded
    model_path = get_model_path(model)
    if model_path.exists() and (model_path / "config.json").exists():
        print_warning(f"Model already exists at {model_path}")
        if not Confirm.ask("Re-download?", default=False):
            return model_path

    # Download with progress
    console.print(f"\n[bold]Downloading {repo}...[/bold]\n")

    try:
        with create_download_progress() as progress:
            task = progress.add_task(f"Downloading {repo}", total=None)

            def progress_callback(current: int, total: int) -> None:
                progress.update(task, completed=current, total=total)

            # Download using huggingface_hub
            local_path = snapshot_download(
                repo_id=repo,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )

        print_success(f"Downloaded to {local_path}")

        # Get model size
        size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        print_info(f"Total size: {format_size(size)}")

        return Path(local_path)

    except Exception as e:
        print_error(f"Failed to download: {e}")
        raise


def get_downloaded_models() -> list[dict]:
    """Get list of downloaded models with metadata."""
    config = get_config()
    models = []

    if not config.models_dir.exists():
        return models

    for model_dir in config.models_dir.iterdir():
        if not model_dir.is_dir():
            continue

        config_file = model_dir / "config.json"
        if not config_file.exists():
            continue

        # Calculate size
        size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())

        # Get modification time
        mtime = datetime.fromtimestamp(model_dir.stat().st_mtime)

        # Convert directory name back to repo format
        repo_name = model_dir.name.replace("--", "/")

        # Try to find alias
        registry = get_registry()
        alias = None
        for info in registry.list_all():
            if info.repo == repo_name:
                alias = info.alias
                break

        models.append({
            "name": alias or repo_name,
            "path": str(model_dir),
            "size": format_size(size),
            "modified": mtime.strftime("%Y-%m-%d %H:%M"),
            "backend": "mlx-lm",  # TODO: Read from model config
        })

    return sorted(models, key=lambda x: x["name"])


def remove_model(model: str, force: bool = False) -> None:
    """Remove a downloaded model."""
    model_path = get_model_path(model)

    if not model_path.exists():
        print_error(f"Model not found: {model}")
        return

    if not force:
        size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        if not Confirm.ask(
            f"Remove {model} ({format_size(size)})?",
            default=False,
        ):
            return

    with status_spinner(f"Removing {model}..."):
        shutil.rmtree(model_path)

    print_success(f"Removed {model}")


def show_model(model: str) -> None:
    """Show information about a model."""
    registry = get_registry()
    model_info = registry.get(model)

    if model_info:
        # Show registry info
        model_path = get_model_path(model)
        size = "not downloaded"
        if model_path.exists():
            size_bytes = sum(
                f.stat().st_size for f in model_path.rglob("*") if f.is_file()
            )
            size = format_size(size_bytes)

        print_model_info(
            name=model_info.alias,
            repo=model_info.repo,
            size=size,
            quantization=model_info.quantization or "unknown",
            backend=model_info.default_backend,
            description=model_info.description or "",
        )
    else:
        # Try to show info for direct repo
        model_path = get_model_path(model)
        if model_path.exists():
            size_bytes = sum(
                f.stat().st_size for f in model_path.rglob("*") if f.is_file()
            )
            print_model_info(
                name=model,
                repo=model,
                size=format_size(size_bytes),
                quantization="unknown",
                backend="mlx-lm",
            )
        else:
            print_error(f"Model not found: {model}")


def run_model(
    model: str,
    prompt: Optional[str] = None,
    backend: Optional[str] = None,
) -> None:
    """Run a model - interactive chat or one-shot generation."""
    from .backends import get_backend, get_default_backend

    # Ensure model is downloaded
    if not is_model_downloaded(model):
        print_info(f"Model not downloaded. Pulling {model}...")
        pull_model(model)

    model_path = get_model_path(model)

    # Get backend
    backend_instance = None
    if backend:
        backend_instance = get_backend(backend)
        if not backend_instance:
            print_error(f"Backend not found: {backend}")
            return
    else:
        # Use model's default or system default
        registry = get_registry()
        model_info = registry.get(model)
        if model_info:
            backend_instance = get_backend(model_info.default_backend)

        if not backend_instance:
            backend_instance = get_default_backend()

    if not backend_instance:
        print_error("No backend available. Install mlx-lm: uv add mlx-lm")
        return

    if not backend_instance.is_available():
        print_error(
            f"Backend {backend_instance.name} is not available. "
            f"Install dependencies first."
        )
        return

    if prompt:
        # One-shot generation
        console.print(f"\n[dim]Using backend: {backend_instance.name}[/dim]\n")

        for token in backend_instance.generate(str(model_path), prompt):
            console.print(token, end="")

        console.print()  # Final newline
    else:
        # Interactive chat
        console.print(f"\n[bold]Interactive chat with {model}[/bold]")
        console.print(f"[dim]Backend: {backend_instance.name}[/dim]")
        console.print("[dim]Type 'exit' or Ctrl+C to quit[/dim]\n")

        messages: list[dict] = []

        try:
            while True:
                user_input = console.input("[bold cyan]>>> [/bold cyan]")

                if user_input.lower() in ("exit", "quit", "/bye"):
                    break

                if not user_input.strip():
                    continue

                messages.append({"role": "user", "content": user_input})

                console.print("[bold green]", end="")
                response_text = ""

                for token in backend_instance.chat(str(model_path), messages):
                    console.print(token, end="")
                    response_text += token

                console.print("[/bold green]\n")
                messages.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")


def serve_model(
    model: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    backend: Optional[str] = None,
) -> None:
    """Start an OpenAI-compatible server for a model."""
    from .backends import get_backend, get_default_backend
    from .process import save_server_info, wait_for_server

    # Ensure model is downloaded
    if not is_model_downloaded(model):
        print_info(f"Model not downloaded. Pulling {model}...")
        pull_model(model)

    model_path = get_model_path(model)

    # Get backend
    backend_instance = None
    if backend:
        backend_instance = get_backend(backend)
        if not backend_instance:
            print_error(f"Backend not found: {backend}")
            return
    else:
        registry = get_registry()
        model_info = registry.get(model)
        if model_info:
            backend_instance = get_backend(model_info.default_backend)

        if not backend_instance:
            backend_instance = get_default_backend()

    if not backend_instance:
        print_error("No backend available. Install mlx-lm: uv add mlx-lm")
        return

    if not backend_instance.is_available():
        print_error(
            f"Backend {backend_instance.name} is not available. "
            f"Install dependencies first."
        )
        return

    console.print(f"\n[bold]Starting server for {model}[/bold]")
    console.print(f"[dim]Backend: {backend_instance.name}[/dim]")
    console.print(f"[dim]Endpoint: http://{host}:{port}[/dim]\n")

    try:
        process = backend_instance.serve(str(model_path), host=host, port=port)

        # Save server info
        save_server_info(
            model=model,
            pid=process.pid,
            port=port,
            backend=backend_instance.name,
        )

        # Wait for server to be ready
        if wait_for_server(host, port):
            print_success(f"Server running at http://{host}:{port}")
            console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")

            # Wait for process or keyboard interrupt
            try:
                process.wait()
            except KeyboardInterrupt:
                console.print("\n[dim]Stopping server...[/dim]")
                backend_instance.stop_server(process)
                print_success("Server stopped")
        else:
            print_error("Server failed to start")
            process.kill()

    except Exception as e:
        print_error(f"Failed to start server: {e}")
        raise
