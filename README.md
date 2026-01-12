# mlx-lama

> **Disclaimer:** This project was generated and tested on a single machine (Apple Silicon Mac). It is pending full review and broader testing. Use at your own discretion.

Ollama-compatible CLI for MLX models on Apple Silicon with multiple backend support.

Run LLMs locally with native Apple Silicon performance using MLX, with the familiar Ollama command-line interface. Supports multiple inference backends (`mlx-lm`, `vllm-mlx`, `ollama`) that can be hot-swapped at runtime.

## Features

- **Ollama-compatible CLI** - Same commands you know: `pull`, `run`, `serve`, `list`, `rm`
- **Native MLX performance** - 15-20% faster than GGUF-based solutions on Apple Silicon
- **Pluggable backends** - Use `mlx-lm` for simplicity, `vllm-mlx` for high throughput, or `ollama` for compatibility
- **Live monitoring TUI** - Real-time stats dashboard with hardware metrics, request tracking, and backend logs
- **Hot-reload backends** - Switch backends on the fly without restarting the server
- **Model registry** - Short aliases like `qwen:32b` instead of full HuggingFace paths
- **Rich feedback** - Progress bars, stats, and beautiful terminal output

## Installation

```bash
# Install with uv (recommended)
uv add mlx-lama

# Or with pip
pip install mlx-lama

# Install with MLX backend
uv add mlx-lama[mlx]

# Install with vLLM backend (high throughput)
uv add mlx-lama[vllm]

# Install monitoring extras (GPU stats)
uv add mlx-lama[monitor]
```

## Quick Start

```bash
# Pull a model
mlx-lama pull qwen-coder:32b

# Run interactively
mlx-lama run qwen-coder:32b

# One-shot generation
mlx-lama run llama:8b "Explain quantum computing"

# Start OpenAI-compatible server with live TUI
mlx-lama serve qwen:32b --port 8000

# Start server without TUI
mlx-lama serve qwen:32b --no-top

# List downloaded models
mlx-lama list

# Show model info
mlx-lama show qwen-coder:32b
```

## Available Models

| Model | Size | Description |
|-------|------|-------------|
| `qwen-coder:32b` | ~18GB | Qwen 2.5 Coder - Best open-source coding model |
| `qwen-coder:14b` | ~8GB | Qwen 2.5 Coder - Fast coding model |
| `qwen-coder:7b` | ~4GB | Qwen 2.5 Coder - Lightweight coding |
| `qwen:32b` | ~18GB | Qwen 2.5 - Excellent all-around |
| `qwen:72b` | ~35GB | Qwen 2.5 - Large reasoning model |
| `llama:70b` | ~35GB | Llama 3.3 - GPT-4 class |
| `llama:8b` | ~4GB | Llama 3.1 - Fast general model |
| `llama:3b` | ~2GB | Llama 3.2 - Very fast, lightweight |
| `deepseek-r1:32b` | ~18GB | DeepSeek R1 - Best reasoning |
| `deepseek-coder:16b` | ~8GB | DeepSeek Coder V2 |
| `codestral:22b` | ~12GB | Mistral's coding model |
| `mistral:7b` | ~4GB | Mistral 7B - Efficient |
| `gemma:9b` | ~5GB | Google's efficient model |
| `gemma:2b` | ~1GB | Very lightweight |
| `llava:34b` | ~18GB | Vision + Language model |
| `qwen-vl:7b` | ~4GB | Qwen2 Vision + Language |

You can also use direct HuggingFace repo paths: `mlx-lama run mlx-community/Qwen2.5-32B-Instruct-4bit`

## Commands

```bash
mlx-lama pull <model>      # Download a model
mlx-lama run <model>       # Interactive chat
mlx-lama run <model> "p"   # One-shot generation
mlx-lama serve <model>     # Start OpenAI API server with live TUI
mlx-lama list              # List downloaded models
mlx-lama rm <model>        # Remove a model
mlx-lama show <model>      # Show model info
mlx-lama ps                # Show running servers
mlx-lama stop              # Stop running servers
mlx-lama backends          # List available backends
mlx-lama install <backend> # Install a backend
```

## Live Monitoring TUI

When running `mlx-lama serve`, a live monitoring dashboard is displayed showing:

- **Server info** - Model, backend, endpoint URL
- **Inference stats** - Requests, tokens/s, latency
- **Hardware stats** - Memory, CPU, GPU usage with progress bars
- **Recent requests** - Last 5 requests with status
- **Backend logs** - Live log output from the backend

Keyboard shortcuts in the TUI:
- `r` - Switch to a different backend (hot-reload)
- `Ctrl+C` - Quit

Use `--no-top` to disable the TUI and run in simple mode.

## Backends

mlx-lama supports multiple backends:

| Backend | Best For | Features |
|---------|----------|----------|
| `mlx-lm` | Single user, simple | Fast startup, low memory |
| `vllm-mlx` | Multi-user, API | Continuous batching, high throughput |
| `ollama` | Compatibility | Uses existing Ollama installation |

```bash
# Use a specific backend
mlx-lama serve qwen:32b --backend vllm-mlx

# Check available backends
mlx-lama backends

# Install a backend
mlx-lama install mlx-lm
mlx-lama install vllm
```

## Configuration

Configuration is stored in `~/.mlx-lama/`:

```
~/.mlx-lama/
├── config.yaml      # User settings
└── registry.yaml    # Custom model aliases
```

Models are stored in the HuggingFace cache (`~/.cache/huggingface/hub/`) and shared with other HF tools.

### Custom Model Aliases

Add custom models to `~/.mlx-lama/registry.yaml`:

```yaml
models:
  my-model:latest:
    repo: username/my-custom-model
    default_backend: mlx-lm
    description: My custom fine-tuned model
```

## API Server

The `serve` command starts an OpenAI-compatible API:

```bash
mlx-lama serve qwen:32b --port 8000
```

Then use with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen:32b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Requirements

- macOS 13+ (Ventura or later)
- Apple Silicon (M1, M2, M3, M4)
- Python 3.10+

## Development

```bash
# Clone and install
git clone https://github.com/timotejlabsky/mlx-lama
cd mlx-lama
uv sync --extra dev

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/
uv run ruff format src/
```

## License

MIT
