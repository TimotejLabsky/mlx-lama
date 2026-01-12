"""Server manager for hot-reload capability."""

import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass

from .backends import get_backend, get_default_backend
from .backends.base import Backend
from .process import wait_for_server
from .registry import get_registry
from .stats import LogParser, get_stats_collector


@dataclass
class ServerState:
    """Current state of the backend server."""

    model: str
    model_path: str
    backend_name: str
    host: str
    port: int
    status: str = "stopped"  # stopped, starting, running, stopping, reloading


class ServerManager:
    """Manages backend server lifecycle with hot-reload support.

    The proxy runs on the user-specified port and stays up during reloads.
    The backend runs on a dynamic port and can be stopped/started.
    """

    def __init__(
        self,
        model: str,
        model_path: str,
        backend: Backend,
        host: str = "127.0.0.1",
        port: int = 8000,
        on_status_change: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.model_path = model_path
        self.backend = backend
        self.host = host
        self.port = port
        self.on_status_change = on_status_change

        self._process: subprocess.Popen | None = None
        self._log_parser: LogParser | None = None
        self._status = "stopped"
        self._lock = threading.Lock()
        self._error: str | None = None

    @property
    def status(self) -> str:
        """Get current server status."""
        return self._status

    @property
    def error(self) -> str | None:
        """Get last error message."""
        return self._error

    @property
    def is_running(self) -> bool:
        """Check if server is currently running."""
        return self._process is not None and self._process.poll() is None

    def _set_status(self, status: str) -> None:
        """Update status and notify callback."""
        self._status = status
        if self.on_status_change:
            self.on_status_change(status)

    def start(self) -> bool:
        """Start the backend server.

        Returns:
            True if server started successfully, False otherwise.
        """
        with self._lock:
            if self.is_running:
                return True

            self._set_status("starting")
            self._error = None

            try:
                # Start backend process
                self._process = self.backend.serve(
                    self.model_path,
                    host=self.host,
                    port=self.port,
                )

                # Start log parser
                collector = get_stats_collector()
                self._log_parser = LogParser(self._process, collector)
                self._log_parser.start()

                # Wait for server to be ready
                if wait_for_server(self.host, self.port, timeout=120.0):
                    self._set_status("running")
                    return True
                else:
                    self._error = "Server startup timeout"
                    self._set_status("stopped")
                    self._cleanup()
                    return False

            except Exception as e:
                self._error = str(e)
                self._set_status("stopped")
                self._cleanup()
                return False

    def stop(self) -> None:
        """Stop the backend server."""
        with self._lock:
            if not self.is_running:
                return

            self._set_status("stopping")
            self._cleanup()
            self._set_status("stopped")

    def _cleanup(self) -> None:
        """Clean up process and threads."""
        if self._log_parser:
            self._log_parser.stop()
            self._log_parser = None

        if self._process:
            self.backend.stop_server(self._process)
            self._process = None

    def reload(
        self,
        model: str | None = None,
        model_path: str | None = None,
        backend: Backend | None = None,
    ) -> bool:
        """Hot-reload with optional new model/backend.

        Args:
            model: New model name (or None to keep current)
            model_path: New model path (or None to keep current)
            backend: New backend (or None to keep current)

        Returns:
            True if reload successful, False otherwise.
        """
        with self._lock:
            self._set_status("reloading")

            # Stop current server
            self._cleanup()

            # Update configuration
            if model:
                self.model = model
            if model_path:
                self.model_path = model_path
            if backend:
                self.backend = backend

            # Clear stats for fresh start
            from .stats import reset_stats_collector

            reset_stats_collector()

        # Start with new configuration (releases lock)
        return self.start()

    def change_model(self, model: str, model_path: str) -> bool:
        """Change to a different model.

        Args:
            model: New model name
            model_path: Path to model files

        Returns:
            True if change successful, False otherwise.
        """
        return self.reload(model=model, model_path=model_path)

    def change_backend(self, backend_name: str) -> bool:
        """Change to a different backend.

        Args:
            backend_name: Name of the backend to use

        Returns:
            True if change successful, False otherwise.
        """
        new_backend = get_backend(backend_name)
        if not new_backend:
            self._error = f"Backend not found: {backend_name}"
            return False

        if not new_backend.is_available():
            self._error = f"Backend not available: {backend_name}"
            return False

        return self.reload(backend=new_backend)

    def get_state(self) -> ServerState:
        """Get current server state."""
        return ServerState(
            model=self.model,
            model_path=self.model_path,
            backend_name=self.backend.name,
            host=self.host,
            port=self.port,
            status=self._status,
        )


def get_model_backend(model: str, backend_name: str | None = None) -> Backend | None:
    """Get the backend for a model.

    Args:
        model: Model name or alias
        backend_name: Explicit backend name, or None to auto-detect

    Returns:
        Backend instance or None if not found
    """
    if backend_name:
        return get_backend(backend_name)

    # Check model's default backend
    registry = get_registry()
    model_info = registry.get(model)
    if model_info:
        backend = get_backend(model_info.default_backend)
        if backend:
            return backend

    # Fall back to system default
    return get_default_backend()
