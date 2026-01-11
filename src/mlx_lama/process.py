"""Process management for running servers."""

import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console

from .config import get_config
from .progress import print_success, print_error, print_warning

console = Console()


def get_servers_file() -> Path:
    """Get the path to the servers state file."""
    config = get_config()
    return config.home / "servers.json"


def load_servers() -> dict:
    """Load the servers state."""
    servers_file = get_servers_file()
    if not servers_file.exists():
        return {}

    try:
        with open(servers_file) as f:
            return json.load(f)
    except Exception:
        return {}


def save_servers(servers: dict) -> None:
    """Save the servers state."""
    servers_file = get_servers_file()
    servers_file.parent.mkdir(parents=True, exist_ok=True)

    with open(servers_file, "w") as f:
        json.dump(servers, f, indent=2)


def save_server_info(
    model: str,
    pid: int,
    port: int,
    backend: str,
) -> None:
    """Save info about a running server."""
    servers = load_servers()

    servers[str(pid)] = {
        "model": model,
        "pid": pid,
        "port": port,
        "backend": backend,
        "started": datetime.now().isoformat(),
    }

    save_servers(servers)


def remove_server_info(pid: int) -> None:
    """Remove info about a stopped server."""
    servers = load_servers()

    if str(pid) in servers:
        del servers[str(pid)]
        save_servers(servers)


def is_process_running(pid: int) -> bool:
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def get_running_servers() -> list[dict]:
    """Get list of currently running servers."""
    servers = load_servers()
    running = []

    for pid_str, info in list(servers.items()):
        pid = int(pid_str)

        if is_process_running(pid):
            # Calculate uptime
            started = datetime.fromisoformat(info["started"])
            uptime = datetime.now() - started
            hours, remainder = divmod(int(uptime.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)

            if hours > 0:
                uptime_str = f"{hours}h {minutes}m"
            elif minutes > 0:
                uptime_str = f"{minutes}m {seconds}s"
            else:
                uptime_str = f"{seconds}s"

            running.append({
                **info,
                "uptime": uptime_str,
            })
        else:
            # Process is dead, clean up
            del servers[pid_str]

    # Save cleaned up state
    save_servers(servers)

    return running


def stop_server(model: Optional[str] = None) -> None:
    """Stop running server(s).

    Args:
        model: Stop only this model's server, or all if None
    """
    servers = load_servers()
    stopped = 0

    for pid_str, info in list(servers.items()):
        pid = int(pid_str)

        # Filter by model if specified
        if model and info["model"] != model:
            continue

        if is_process_running(pid):
            try:
                console.print(f"[dim]Stopping {info['model']} (PID {pid})...[/dim]")
                os.kill(pid, signal.SIGTERM)

                # Wait for graceful shutdown
                for _ in range(50):  # 5 seconds
                    if not is_process_running(pid):
                        break
                    time.sleep(0.1)
                else:
                    # Force kill if still running
                    os.kill(pid, signal.SIGKILL)

                print_success(f"Stopped {info['model']}")
                stopped += 1

            except Exception as e:
                print_error(f"Failed to stop {info['model']}: {e}")

        # Remove from state
        del servers[pid_str]

    save_servers(servers)

    if stopped == 0:
        if model:
            print_warning(f"No running server found for {model}")
        else:
            print_warning("No running servers found")


def wait_for_server(
    host: str,
    port: int,
    timeout: float = 30.0,
    interval: float = 0.5,
) -> bool:
    """Wait for a server to be ready.

    Args:
        host: Server host
        port: Server port
        timeout: Maximum time to wait in seconds
        interval: Check interval in seconds

    Returns:
        True if server is ready, False if timeout
    """
    url = f"http://{host}:{port}/v1/models"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                return True
        except httpx.RequestError:
            pass

        time.sleep(interval)

    return False
