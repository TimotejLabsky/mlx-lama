# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mlx-lama is an Ollama-compatible CLI for running LLMs on Apple Silicon using MLX. It provides familiar commands (`pull`, `run`, `serve`, `list`, `rm`) while leveraging native Apple Silicon performance through pluggable backends.

## Development Commands

```bash
# Install dependencies
uv sync

# Install with optional backends
uv sync --extra mlx    # mlx-lm backend
uv sync --extra vllm   # vllm-mlx backend
uv sync --extra dev    # development tools (pytest, ruff)

# Run the CLI
uv run mlx-lama --help

# Run tests
uv run pytest

# Run single test
uv run pytest tests/test_file.py::test_name

# Lint and format
uv run ruff check src/
uv run ruff format src/
```

## Architecture

### Core Components

- **`cli.py`** - Typer-based CLI entry point. Commands are defined here but delegate to modules in `models.py` and `backends/`

- **`backends/`** - Pluggable inference backend system:
  - `base.py` - Abstract `Backend` class defining the interface (`generate()`, `serve()`, `detect()`, `capabilities()`)
  - Backends self-register via `register_backend()` when imported
  - Detection uses a priority chain: service → Python package → system binary
  - Each backend declares `python_packages`, `binary_names`, `service_ports` for auto-detection

- **`registry.py`** - Model alias system mapping short names (e.g., `qwen:32b`) to HuggingFace repos. Built-in aliases in `DEFAULT_MODELS` dict, user customization via `~/.mlx-lama/registry.yaml`

- **`models.py`** - Model operations (`pull_model`, `run_model`, `serve_model`). Uses HuggingFace's shared cache (`~/.cache/huggingface/hub/`) so models are shared with mlx-lm and other HF tools

- **`config.py`** - Configuration dataclass. User config stored at `~/.mlx-lama/config.yaml`

### Key Patterns

- Backends are lazy-loaded via `_load_backends()` to avoid import errors when optional dependencies are missing
- Models can be referenced by alias (`qwen:32b`) or direct HuggingFace repo (`mlx-community/Qwen2.5-32B-Instruct-4bit`)
- The `ModelRegistry` tries tag fallbacks (`:latest`, `:7b`, `:8b`) when no exact match is found

## Configuration Files

- `~/.mlx-lama/config.yaml` - User settings (default_backend, default_port)
- `~/.mlx-lama/registry.yaml` - Custom model aliases
- Models stored in HuggingFace cache: `~/.cache/huggingface/hub/`
