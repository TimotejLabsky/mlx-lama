"""Backend registry and management."""

import subprocess
from typing import Optional

from .base import Backend, BackendCapabilities, InstallOption, InstallMethod, DetectionResult


# Backend registry
_backends: dict[str, type[Backend]] = {}


def register_backend(name: str, backend_class: type[Backend]) -> None:
    """Register a backend."""
    _backends[name] = backend_class


def get_backend(name: str) -> Optional[Backend]:
    """Get a backend instance by name."""
    _load_backends()
    if name not in _backends:
        return None
    return _backends[name]()


def get_available_backends() -> list[dict]:
    """Get list of all backends with their status."""
    _load_backends()

    result = []
    for name, backend_class in _backends.items():
        try:
            backend = backend_class()
            caps = backend.capabilities()
            detection = backend.detect()

            result.append({
                "name": name,
                "available": detection.available,
                "method": detection.method,
                "version": detection.version,
                "details": detection.details,
                "batching": caps.continuous_batching,
                "vision": caps.vision_support,
                "description": backend.description,
                "install_options": backend.get_install_commands(),
            })
        except Exception as e:
            result.append({
                "name": name,
                "available": False,
                "method": "error",
                "version": None,
                "details": str(e),
                "batching": False,
                "vision": False,
                "description": "Error loading backend",
                "install_options": [],
            })

    return result


def get_default_backend() -> Optional[Backend]:
    """Get the default backend (first available)."""
    _load_backends()

    # Priority order
    for name in ["mlx-lm", "vllm-mlx", "ollama"]:
        if name in _backends:
            backend = _backends[name]()
            if backend.is_available():
                return backend

    return None


def install_backend(name: str, method: Optional[str] = None) -> bool:
    """Install a backend.

    Args:
        name: Backend name to install
        method: Installation method (uv, pip, brew, manual). If None, uses first available.

    Returns:
        True if installation succeeded
    """
    _load_backends()

    if name not in _backends:
        raise ValueError(f"Unknown backend: {name}")

    backend = _backends[name]()
    options = backend.get_install_commands()

    if not options:
        raise ValueError(f"No installation options for {name}")

    # Find the right option
    option = None
    if method:
        for opt in options:
            if opt.method.value == method:
                option = opt
                break
        if not option:
            raise ValueError(f"No {method} installation option for {name}")
    else:
        option = options[0]  # Use first (recommended) option

    # Execute installation
    try:
        print(f"Installing {name} via {option.method.value}...")
        print(f"Running: {option.command}")

        result = subprocess.run(
            option.command,
            shell=True,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"Successfully installed {name}")
            return True
        else:
            print(f"Installation failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"Installation error: {e}")
        return False


def _load_backends() -> None:
    """Load all backend modules to trigger registration."""
    if _backends:
        return  # Already loaded

    # Import backends to trigger registration
    try:
        from . import mlx_lm_backend  # noqa: F401
    except ImportError:
        pass

    try:
        from . import vllm_mlx_backend  # noqa: F401
    except ImportError:
        pass

    try:
        from . import ollama_backend  # noqa: F401
    except ImportError:
        pass


__all__ = [
    "Backend",
    "BackendCapabilities",
    "InstallOption",
    "InstallMethod",
    "DetectionResult",
    "register_backend",
    "get_backend",
    "get_available_backends",
    "get_default_backend",
    "install_backend",
]
