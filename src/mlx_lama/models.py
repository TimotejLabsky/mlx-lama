"""Model management - pull, list, remove, show, run, serve."""

import os
import random
import shutil
import socket
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TransferSpeedColumn,
)
from rich.prompt import Confirm

from .progress import (
    format_size,
    print_error,
    print_info,
    print_model_info,
    print_success,
    print_warning,
    status_spinner,
)
from .registry import get_registry

console = Console()


def get_free_port() -> int:
    """Get a random free port in the ephemeral range (49152-65535)."""
    for _ in range(10):
        port = random.randint(49152, 65535)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError("Could not find free port")


def get_hf_cache_dir() -> Path:
    """Get the HuggingFace cache directory."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def get_model_repo(model: str) -> str:
    """Resolve model alias to HuggingFace repo."""
    registry = get_registry()
    model_info = registry.get(model)
    if model_info:
        return model_info.repo
    return model


def get_model_cache_path(repo: str) -> Path | None:
    """Get the cached path for a model repo, if it exists."""
    cache_dir = get_hf_cache_dir()
    # HuggingFace cache uses models--org--name format
    safe_name = f"models--{repo.replace('/', '--')}"
    model_dir = cache_dir / safe_name

    if model_dir.exists():
        # Find the snapshot directory
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            # Get the most recent snapshot
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                return snapshots[-1]
    return None


def is_model_downloaded(model: str) -> bool:
    """Check if a model is already downloaded in HuggingFace cache."""
    repo = get_model_repo(model)
    cache_path = get_model_cache_path(repo)

    if cache_path is None:
        return False

    # Check for essential files
    return (cache_path / "config.json").exists()


def get_model_path(model: str) -> Path:
    """Get the local path for a model (from HF cache)."""
    repo = get_model_repo(model)
    cache_path = get_model_cache_path(repo)

    if cache_path:
        return cache_path

    # Return expected path even if not downloaded
    cache_dir = get_hf_cache_dir()
    safe_name = f"models--{repo.replace('/', '--')}"
    return cache_dir / safe_name / "snapshots" / "main"


def pull_model(model: str, backend: str | None = None) -> Path:  # noqa: ARG001
    """Download a model from HuggingFace.

    Downloads to the shared HuggingFace cache (~/.cache/huggingface/hub/)
    so models are shared with mlx-lm, vllm-mlx, and other HF tools.

    Args:
        model: Model alias or HuggingFace repo (e.g., "mlx-community/Qwen2.5-7B-Instruct-4bit")
        backend: Preferred backend (for future use)

    Returns:
        Path to the downloaded model
    """
    registry = get_registry()

    # Resolve alias to repo
    model_info = registry.get(model)
    if model_info:
        repo = model_info.repo
        print_info(f"Resolving [bold]{model}[/bold] → [blue]{repo}[/blue]")
    else:
        repo = model
        print_info(f"Model: [blue]{repo}[/blue]")

    # Check if already downloaded
    if is_model_downloaded(model):
        cache_path = get_model_cache_path(repo)
        if cache_path:
            print_warning("Model already downloaded")
            print_info(f"Location: {cache_path}")
            if not Confirm.ask("Re-download?", default=False):
                return cache_path

    # Get model info from HuggingFace
    console.print(f"\n[bold]Downloading {repo}...[/bold]")

    try:
        api = HfApi()
        model_info_hf = api.model_info(repo)

        # Calculate total size
        total_size = sum(s.size for s in model_info_hf.siblings if s.size)
        if total_size > 0:
            console.print(f"[dim]Total size: {format_size(total_size)}[/dim]\n")
    except Exception:
        total_size = 0
        console.print()

    try:
        # Download with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=console,
        ) as progress:
            progress.add_task("Downloading", total=total_size if total_size > 0 else None)

            # Download using huggingface_hub (uses shared cache)
            local_path = snapshot_download(repo_id=repo)

        print_success("Downloaded successfully!")
        print_info(f"Location: {local_path}")

        # Calculate actual size
        path = Path(local_path)
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print_info(f"Size: {format_size(size)}")

        return path

    except Exception as e:
        print_error(f"Failed to download: {e}")
        raise


def get_downloaded_models() -> list[dict]:
    """Get list of downloaded models with metadata from HuggingFace cache."""
    models = []
    cache_dir = get_hf_cache_dir()

    if not cache_dir.exists():
        return models

    # Scan HuggingFace cache for MLX models
    for model_dir in cache_dir.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue

        # Find snapshot directory
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            continue

        # Use most recent snapshot
        snapshot = snapshots[-1]
        config_file = snapshot / "config.json"
        if not config_file.exists():
            continue

        # Check if it's an MLX model (has .safetensors files, typical for MLX)
        # This is a heuristic - MLX models typically have model.safetensors
        has_safetensors = any(snapshot.glob("*.safetensors"))
        if not has_safetensors:
            continue

        # Calculate size from blobs (actual storage)
        blobs_dir = model_dir / "blobs"
        if blobs_dir.exists():
            size = sum(f.stat().st_size for f in blobs_dir.iterdir() if f.is_file())
        else:
            size = sum(f.stat().st_size for f in snapshot.rglob("*") if f.is_file())

        # Get modification time
        mtime = datetime.fromtimestamp(model_dir.stat().st_mtime)

        # Convert directory name back to repo format (models--org--name -> org/name)
        repo_name = model_dir.name.replace("models--", "").replace("--", "/")

        # Try to find alias
        registry = get_registry()
        alias = None
        for info in registry.list_all():
            if info.repo == repo_name:
                alias = info.alias
                break

        models.append({
            "name": alias or repo_name,
            "path": str(snapshot),
            "size": format_size(size),
            "modified": mtime.strftime("%Y-%m-%d %H:%M"),
            "backend": "mlx-lm",
        })

    return sorted(models, key=lambda x: x["name"])


def remove_model(model: str, force: bool = False) -> None:
    """Remove a downloaded model from HuggingFace cache."""
    repo = get_model_repo(model)
    cache_dir = get_hf_cache_dir()

    # Find the model directory in cache
    safe_name = f"models--{repo.replace('/', '--')}"
    model_dir = cache_dir / safe_name

    if not model_dir.exists():
        print_error(f"Model not found: {model}")
        return

    # Calculate size from blobs (actual disk usage)
    blobs_dir = model_dir / "blobs"
    if blobs_dir.exists():
        size = sum(f.stat().st_size for f in blobs_dir.iterdir() if f.is_file())
    else:
        size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())

    if not force:
        if not Confirm.ask(
            f"Remove {model} ({format_size(size)})?",
            default=False,
        ):
            return

    with status_spinner(f"Removing {model}..."):
        shutil.rmtree(model_dir)

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
    prompt: str | None = None,
    backend: str | None = None,
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
    backend: str | None = None,
    top: bool = False,
) -> None:
    """Start an OpenAI-compatible server for a model."""
    from .backends import get_backend, get_default_backend
    from .process import get_running_servers, save_server_info, wait_for_server

    # Check if a server is already running on the requested port
    running = get_running_servers()
    for server in running:
        if server["port"] == port:
            print_error(
                f"Server already running on port {port} "
                f"(model: {server['model']}, PID: {server['pid']})"
            )
            console.print(f"[dim]Stop it with: mlx-lama stop {server['model']}[/dim]")
            return

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

    # When using TUI, proxy runs on user port and backend on random high port
    if top:
        backend_port = get_free_port()
        proxy_port = port
    else:
        backend_port = port
        proxy_port = None

    console.print(f"\n[bold]Starting server for {model}[/bold]")
    console.print(f"[dim]Backend: {backend_instance.name}[/dim]")
    console.print(f"[dim]Endpoint: http://{host}:{port}[/dim]\n")

    try:
        process = backend_instance.serve(str(model_path), host=host, port=backend_port)

        # Save server info (use user-facing port)
        save_server_info(
            model=model,
            pid=process.pid,
            port=port,
            backend=backend_instance.name,
        )

        # Wait for backend server to be ready (can take 60s+ for mlx-lm)
        with status_spinner("Loading model (this may take a minute)..."):
            server_ready = wait_for_server(host, backend_port)

        if server_ready:
            if top:
                # Start live monitoring TUI with stats-capturing proxy
                import asyncio
                import threading
                import time

                from .stats import LogParser, get_stats_collector, reset_stats_collector
                from .top import LiveTop

                # Reset stats for fresh start
                reset_stats_collector()
                collector = get_stats_collector()

                # Start log parser to capture backend stdout/stderr
                log_parser = LogParser(process, collector)
                log_parser.start()

                def on_request_complete(
                    endpoint: str, prompt_tokens: int, completion_tokens: int, latency_ms: float
                ):
                    """Callback from proxy when request completes."""
                    import uuid
                    request_id = str(uuid.uuid4())[:8]
                    collector.start_request(request_id, endpoint)
                    collector.finish_request(
                        request_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                    )

                # Start proxy in background thread on user-specified port
                import uvicorn

                from .proxy import create_proxy_app

                backend_url = f"http://{host}:{backend_port}"

                app = create_proxy_app(backend_url, on_request_complete)

                def run_proxy():
                    config = uvicorn.Config(app, host=host, port=proxy_port, log_level="error")
                    server = uvicorn.Server(config)
                    asyncio.run(server.serve())

                proxy_thread = threading.Thread(target=run_proxy, daemon=True)
                proxy_thread.start()

                # Give proxy a moment to start
                time.sleep(0.5)

                print_success(f"Server running at http://{host}:{port}")

                live_top = LiveTop(
                    model=model,
                    backend=backend_instance.name,
                    host=host,
                    port=port,  # Show user-specified port in UI
                )

                try:
                    with live_top.start(console):
                        while process.poll() is None:
                            live_top.update()
                            time.sleep(0.5)
                except KeyboardInterrupt:
                    pass
                finally:
                    live_top.stop()
                    log_parser.stop()
                    console.print("\n[dim]Stopping server...[/dim]")
                    backend_instance.stop_server(process)
                    print_success("Server stopped")
            else:
                # Normal mode (no --top)
                print_success(f"Server running at http://{host}:{port}")
                console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")

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
