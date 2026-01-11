"""Model registry mapping aliases to HuggingFace repos."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

from .config import get_config


@dataclass
class ModelInfo:
    """Information about a model."""

    alias: str
    repo: str
    default_backend: str = "mlx-lm"
    quantization: Optional[str] = None
    description: Optional[str] = None

    @property
    def name(self) -> str:
        """Short name without tag."""
        return self.alias.split(":")[0]

    @property
    def tag(self) -> str:
        """Tag portion of alias (e.g., '32b' from 'qwen:32b')."""
        parts = self.alias.split(":")
        return parts[1] if len(parts) > 1 else "latest"


# Default model registry - built-in aliases
DEFAULT_MODELS: dict[str, dict] = {
    # Qwen Coder models
    "qwen-coder:32b": {
        "repo": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "Qwen 2.5 Coder 32B - Best open-source coding model",
    },
    "qwen-coder:14b": {
        "repo": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "Qwen 2.5 Coder 14B - Fast coding model",
    },
    "qwen-coder:7b": {
        "repo": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "Qwen 2.5 Coder 7B - Lightweight coding model",
    },
    # Qwen general models
    "qwen:32b": {
        "repo": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "Qwen 2.5 32B - Excellent all-around model",
    },
    "qwen:72b": {
        "repo": "mlx-community/Qwen2.5-72B-Instruct-4bit",
        "default_backend": "vllm",
        "quantization": "4bit",
        "description": "Qwen 2.5 72B - Large reasoning model",
    },
    "qwen:7b": {
        "repo": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "Qwen 2.5 7B - Fast general model",
    },
    # Llama models
    "llama:70b": {
        "repo": "mlx-community/Llama-3.3-70B-Instruct-4bit",
        "default_backend": "vllm",
        "quantization": "4bit",
        "description": "Llama 3.3 70B - GPT-4 class model",
    },
    "llama:8b": {
        "repo": "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "Llama 3.1 8B - Fast general model",
    },
    "llama:3b": {
        "repo": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "Llama 3.2 3B - Very fast, lightweight",
    },
    # DeepSeek models
    "deepseek-r1:32b": {
        "repo": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "DeepSeek R1 32B - Best reasoning model",
    },
    "deepseek-coder:16b": {
        "repo": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "DeepSeek Coder V2 16B - Fast coding",
    },
    # Mistral models
    "mistral:7b": {
        "repo": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "Mistral 7B - Efficient general model",
    },
    "codestral:22b": {
        "repo": "mlx-community/Codestral-22B-v0.1-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "Codestral 22B - Mistral's coding model",
    },
    # Gemma models
    "gemma:9b": {
        "repo": "mlx-community/gemma-2-9b-it-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "Gemma 2 9B - Google's efficient model",
    },
    "gemma:2b": {
        "repo": "mlx-community/gemma-2-2b-it-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "Gemma 2 2B - Very lightweight",
    },
    # Vision models
    "llava:34b": {
        "repo": "mlx-community/llava-v1.6-34b-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "LLaVA 34B - Vision + Language model",
    },
    "qwen-vl:7b": {
        "repo": "mlx-community/Qwen2-VL-7B-Instruct-4bit",
        "default_backend": "mlx-lm",
        "quantization": "4bit",
        "description": "Qwen2 VL 7B - Vision + Language",
    },
}


class ModelRegistry:
    """Registry for model aliases and their HuggingFace repos."""

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._load_defaults()
        self._load_user_registry()

    def _load_defaults(self) -> None:
        """Load default built-in models."""
        for alias, info in DEFAULT_MODELS.items():
            self._models[alias] = ModelInfo(alias=alias, **info)

    def _load_user_registry(self) -> None:
        """Load user's custom registry if it exists."""
        config = get_config()
        if not config.registry_file.exists():
            return

        try:
            with open(config.registry_file) as f:
                data = yaml.safe_load(f) or {}

            for alias, info in data.get("models", {}).items():
                if isinstance(info, dict) and "repo" in info:
                    self._models[alias] = ModelInfo(alias=alias, **info)
        except Exception:
            pass  # Ignore invalid registry files

    def get(self, alias: str) -> Optional[ModelInfo]:
        """Get model info by alias."""
        # Direct match
        if alias in self._models:
            return self._models[alias]

        # Try with :latest tag
        if ":" not in alias:
            for tag in ["latest", "7b", "8b"]:
                full_alias = f"{alias}:{tag}"
                if full_alias in self._models:
                    return self._models[full_alias]

        return None

    def resolve(self, alias: str) -> str:
        """Resolve alias to HuggingFace repo, or return as-is if already a repo."""
        # If it looks like a HF repo (contains /), return as-is
        if "/" in alias:
            return alias

        model = self.get(alias)
        if model:
            return model.repo

        # Not found - might be a direct HF repo reference
        return alias

    def list_all(self) -> list[ModelInfo]:
        """List all registered models."""
        return list(self._models.values())

    def add(self, alias: str, repo: str, **kwargs) -> None:
        """Add a model to user registry."""
        self._models[alias] = ModelInfo(alias=alias, repo=repo, **kwargs)
        self._save_user_registry()

    def _save_user_registry(self) -> None:
        """Save user-added models to registry file."""
        config = get_config()
        config.ensure_dirs()

        # Only save non-default models
        user_models = {
            alias: {
                "repo": info.repo,
                "default_backend": info.default_backend,
                "quantization": info.quantization,
                "description": info.description,
            }
            for alias, info in self._models.items()
            if alias not in DEFAULT_MODELS
        }

        if user_models:
            with open(config.registry_file, "w") as f:
                yaml.dump({"models": user_models}, f, default_flow_style=False)


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
