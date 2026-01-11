"""MLX-LM backend implementation."""

import subprocess
import sys
from typing import Generator

from .base import Backend, BackendCapabilities
from . import register_backend


class MlxLmBackend(Backend):
    """Backend using mlx-lm for inference."""

    name = "mlx-lm"
    description = "Native MLX backend - fast, simple, single-user"

    def is_available(self) -> bool:
        """Check if mlx-lm is installed."""
        try:
            import mlx_lm  # noqa: F401

            return True
        except ImportError:
            return False

    def capabilities(self) -> BackendCapabilities:
        """MLX-LM capabilities."""
        return BackendCapabilities(
            continuous_batching=False,
            vision_support=False,  # mlx-vlm is separate
            tool_calling=False,
            max_concurrent=1,
        )

    def generate(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """Generate text using mlx-lm."""
        try:
            from mlx_lm import load, generate
        except ImportError:
            raise RuntimeError(
                "mlx-lm is not installed. Install with: uv add mlx-lm"
            )

        # Load model
        model, tokenizer = load(model_path)

        if stream:
            # Streaming generation
            from mlx_lm.utils import generate_step

            prompt_tokens = tokenizer.encode(prompt)

            for token, _ in generate_step(
                prompt_tokens,
                model,
                max_tokens=max_tokens,
                temp=temperature,
            ):
                text = tokenizer.decode([token])
                yield text
        else:
            # Non-streaming
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
            )
            yield response

    def serve(
        self,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> subprocess.Popen:
        """Start mlx-lm server."""
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.server",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return process


# Register this backend
register_backend("mlx-lm", MlxLmBackend)
