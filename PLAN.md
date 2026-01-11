# mlx-lama Development Plan

Ollama-compatible CLI for MLX models on Apple Silicon with pluggable backends.

## Goals

1. **Ollama CLI compatibility** - Same commands, same UX
2. **Native MLX performance** - Faster than GGUF-based solutions
3. **Pluggable backends** - mlx-lm, vllm-mlx, future backends
4. **Rich feedback** - Progress bars, stats, model info

## Repository Structure

```
mlx-lama/
├── pyproject.toml          # uv-managed dependencies
├── README.md
├── LICENSE                  # MIT
├── PLAN.md                  # This file
├── src/
│   └── mlx_lama/
│       ├── __init__.py
│       ├── cli.py          # Typer CLI (Ollama-compatible)
│       ├── config.py       # Settings, paths
│       ├── models.py       # Model manager (pull, list, rm)
│       ├── registry.py     # Model aliases → HF paths
│       ├── process.py      # Server process management
│       ├── progress.py     # Rich progress bars
│       ├── stats.py        # Performance stats
│       │
│       ├── backends/
│       │   ├── __init__.py # Backend registry
│       │   ├── base.py     # Protocol definition
│       │   ├── mlx_lm.py   # mlx-lm backend
│       │   └── vllm_mlx.py # vllm-mlx backend
│       │
│       └── data/
│           └── models.yaml # Default model registry
└── tests/
```

## CLI Commands (Ollama-compatible)

| Command | Description |
|---------|-------------|
| `mlx-lama pull <model>` | Download model from HuggingFace |
| `mlx-lama run <model> [prompt]` | Run model (interactive or one-shot) |
| `mlx-lama serve <model>` | Start OpenAI-compatible server |
| `mlx-lama list` | List downloaded models |
| `mlx-lama rm <model>` | Remove model |
| `mlx-lama show <model>` | Show model info |
| `mlx-lama ps` | Show running models |
| `mlx-lama stop` | Stop running server |
| `mlx-lama backends` | List available backends |

## Implementation Phases

### Phase 1: Project Setup ✅
- [x] Initialize with uv
- [x] pyproject.toml with dependencies
- [x] Basic CLI structure with Typer
- [x] Config and paths

### Phase 2: Model Registry
- [ ] Model aliases (qwen:32b → HF path)
- [ ] YAML registry with defaults
- [ ] Registry loading and merging

### Phase 3: Model Management
- [ ] `pull` - Download from HuggingFace with progress
- [ ] `list` - Show downloaded models with sizes
- [ ] `rm` - Remove models
- [ ] `show` - Model info (size, quantization, etc.)

### Phase 4: Backend Architecture
- [ ] Backend protocol/interface
- [ ] mlx-lm backend implementation
- [ ] Backend registry and selection

### Phase 5: Run & Serve
- [ ] `run` - Interactive chat / one-shot generation
- [ ] `serve` - OpenAI API server
- [ ] `ps` / `stop` - Process management

### Phase 6: vllm-mlx Backend
- [ ] vllm-mlx backend implementation
- [ ] Auto backend selection based on use case

## Dependencies

```toml
[project.dependencies]
typer = {extras = ["all"], version = ">=0.9.0"}
rich = ">=13.0.0"
pyyaml = ">=6.0"
huggingface-hub = ">=0.20.0"
httpx = ">=0.25.0"

[project.optional-dependencies]
mlx = ["mlx-lm>=0.19.0"]
vllm = ["vllm-mlx>=0.1.0"]
```

## Model Registry Format

```yaml
# ~/.mlx-lama/registry.yaml
models:
  qwen-coder:32b:
    repo: mlx-community/Qwen2.5-Coder-32B-Instruct-4bit
    default_backend: mlx-lm

  llama:70b:
    repo: mlx-community/Llama-3.3-70B-Instruct-4bit
    default_backend: vllm

  llama:8b:
    repo: mlx-community/Llama-3.1-8B-Instruct-4bit
    default_backend: mlx-lm
```

## Backend Interface

```python
from typing import Protocol, Generator
from dataclasses import dataclass

@dataclass
class BackendCapabilities:
    continuous_batching: bool
    vision_support: bool
    tool_calling: bool
    max_concurrent: int

class Backend(Protocol):
    name: str

    def is_available(self) -> bool:
        """Check if backend dependencies are installed"""
        ...

    def serve(self, model_path: str, port: int = 8000) -> subprocess.Popen:
        """Start OpenAI-compatible server"""
        ...

    def generate(self, model_path: str, prompt: str) -> Generator[str, None, None]:
        """Stream generation"""
        ...

    def capabilities(self) -> BackendCapabilities:
        """What this backend supports"""
        ...
```

## Config Locations

- `~/.mlx-lama/` - Main config directory
- `~/.mlx-lama/models/` - Downloaded models
- `~/.mlx-lama/registry.yaml` - Custom model aliases
- `~/.mlx-lama/config.yaml` - User settings
