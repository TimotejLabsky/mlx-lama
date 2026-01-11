# mlx-lama

Ollama-compatible CLI for MLX models on Apple Silicon.

Run LLMs locally with native Apple Silicon performance using MLX, with the familiar Ollama command-line interface.

## Features

- **Ollama-compatible CLI** - Same commands you know: `pull`, `run`, `serve`, `list`, `rm`
- **Native MLX performance** - 15-20% faster than GGUF-based solutions on Apple Silicon
- **Pluggable backends** - Use `mlx-lm` for simplicity or `vllm-mlx` for high throughput
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
```

## Quick Start

```bash
# Pull a model
mlx-lama pull qwen-coder:32b

# Run interactively
mlx-lama run qwen-coder:32b

# One-shot generation
mlx-lama run llama:8b "Explain quantum computing"

# Start OpenAI-compatible server
mlx-lama serve qwen:32b --port 8000

# List downloaded models
mlx-lama list

# Show model info
mlx-lama show qwen-coder:32b
```

## Available Models

| Model | Size | Description |
|-------|------|-------------|
| `qwen-coder:32b` | ~18GB | Best open-source coding model |
| `qwen-coder:14b` | ~8GB | Fast coding model |
| `qwen:32b` | ~18GB | Excellent all-around |
| `llama:70b` | ~35GB | GPT-4 class |
| `llama:8b` | ~4GB | Fast general model |
| `deepseek-r1:32b` | ~18GB | Best reasoning |
| `mistral:7b` | ~4GB | Efficient |
| `gemma:9b` | ~5GB | Google's efficient model |

Use `mlx-lama list --all` to see all available models.

## Commands

```bash
mlx-lama pull <model>     # Download a model
mlx-lama run <model>      # Interactive chat
mlx-lama run <model> "p"  # One-shot generation
mlx-lama serve <model>    # Start OpenAI API server
mlx-lama list             # List downloaded models
mlx-lama rm <model>       # Remove a model
mlx-lama show <model>     # Show model info
mlx-lama ps               # Show running servers
mlx-lama stop             # Stop running servers
mlx-lama backends         # List available backends
```

## Backends

mlx-lama supports multiple backends:

| Backend | Best For | Features |
|---------|----------|----------|
| `mlx-lm` | Single user, simple | Fast startup, low memory |
| `vllm` | Multi-user, API | Continuous batching, high throughput |

```bash
# Use a specific backend
mlx-lama serve qwen:32b --backend vllm

# Check available backends
mlx-lama backends
```

## Configuration

Configuration is stored in `~/.mlx-lama/`:

```
~/.mlx-lama/
├── config.yaml      # User settings
├── registry.yaml    # Custom model aliases
└── models/          # Downloaded models
```

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

## License

MIT
