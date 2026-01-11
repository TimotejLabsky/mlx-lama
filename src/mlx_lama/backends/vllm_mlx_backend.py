"""vLLM-MLX backend implementation."""

import subprocess
import sys
from typing import Generator

from .base import Backend, BackendCapabilities
from . import register_backend


class VllmMlxBackend(Backend):
    """Backend using vllm-mlx for high-throughput inference."""

    name = "vllm"
    description = "vLLM backend - continuous batching, multi-user"

    def is_available(self) -> bool:
        """Check if vllm-mlx is installed."""
        try:
            # Try importing vllm-mlx specific module
            # Note: The actual import may vary based on vllm-mlx package structure
            import vllm  # noqa: F401

            return True
        except ImportError:
            return False

    def capabilities(self) -> BackendCapabilities:
        """vLLM-MLX capabilities."""
        return BackendCapabilities(
            continuous_batching=True,
            vision_support=True,
            tool_calling=True,
            max_concurrent=100,
        )

    def generate(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """Generate text using vllm-mlx.

        Note: vLLM is primarily designed for server mode.
        This is a simplified implementation for direct generation.
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise RuntimeError(
                "vllm-mlx is not installed. Install with: uv add vllm-mlx"
            )

        llm = LLM(model=model_path)
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )

        outputs = llm.generate([prompt], sampling_params)

        for output in outputs:
            yield output.outputs[0].text

    def serve(
        self,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> subprocess.Popen:
        """Start vllm-mlx OpenAI-compatible server."""
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
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
register_backend("vllm", VllmMlxBackend)
