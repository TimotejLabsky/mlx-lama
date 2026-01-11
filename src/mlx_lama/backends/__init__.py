"""Backend registry and management."""

from typing import Optional
from .base import Backend, BackendCapabilities


# Backend registry
_backends: dict[str, type[Backend]] = {}


def register_backend(name: str, backend_class: type[Backend]) -> None:
    """Register a backend."""
    _backends[name] = backend_class


def get_backend(name: str) -> Optional[Backend]:
    """Get a backend instance by name."""
    if name not in _backends:
        return None
    return _backends[name]()


def get_available_backends() -> list[dict]:
    """Get list of all backends with their status."""
    # Import backends to trigger registration
    from . import mlx_lm_backend
    from . import vllm_mlx_backend

    result = []
    for name, backend_class in _backends.items():
        try:
            backend = backend_class()
            caps = backend.capabilities()
            result.append({
                "name": name,
                "available": backend.is_available(),
                "batching": caps.continuous_batching,
                "vision": caps.vision_support,
                "description": backend.description,
            })
        except Exception:
            result.append({
                "name": name,
                "available": False,
                "batching": False,
                "vision": False,
                "description": "Error loading backend",
            })

    return result


def get_default_backend() -> Optional[Backend]:
    """Get the default backend (mlx-lm if available)."""
    for name in ["mlx-lm", "vllm"]:
        backend = get_backend(name)
        if backend and backend.is_available():
            return backend
    return None


__all__ = [
    "Backend",
    "BackendCapabilities",
    "register_backend",
    "get_backend",
    "get_available_backends",
    "get_default_backend",
]
