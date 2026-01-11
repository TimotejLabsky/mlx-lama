"""Configuration and paths for mlx-lama."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import yaml


def get_default_home() -> Path:
    """Get the default mlx-lama home directory."""
    return Path.home() / ".mlx-lama"


@dataclass
class Config:
    """mlx-lama configuration."""

    home: Path = field(default_factory=get_default_home)
    default_backend: str = "mlx-lm"
    default_port: int = 8000

    @property
    def models_dir(self) -> Path:
        """Directory where models are stored."""
        return self.home / "models"

    @property
    def registry_file(self) -> Path:
        """User's custom model registry."""
        return self.home / "registry.yaml"

    @property
    def config_file(self) -> Path:
        """User's config file."""
        return self.home / "config.yaml"

    @property
    def pid_file(self) -> Path:
        """PID file for running server."""
        return self.home / "server.pid"

    def ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.home.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, home: Optional[Path] = None) -> "Config":
        """Load config from file or create default."""
        config = cls(home=home) if home else cls()

        if config.config_file.exists():
            with open(config.config_file) as f:
                data = yaml.safe_load(f) or {}
                if "default_backend" in data:
                    config.default_backend = data["default_backend"]
                if "default_port" in data:
                    config.default_port = data["default_port"]

        return config

    def save(self) -> None:
        """Save config to file."""
        self.ensure_dirs()
        data = {
            "default_backend": self.default_backend,
            "default_port": self.default_port,
        }
        with open(self.config_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = Config.load()
        _config.ensure_dirs()
    return _config
