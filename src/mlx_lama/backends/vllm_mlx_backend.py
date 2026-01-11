"""vLLM-MLX backend implementation."""

import subprocess
import sys
from typing import Generator

from .base import Backend, BackendCapabilities, InstallOption, InstallMethod
from . import register_backend


class VllmMlxBackend(Backend):
    """Backend using vllm-mlx for high-throughput inference."""

    name = "vllm"
    description = "vLLM backend - continuous batching, multi-user"

    # Detection configuration
    python_packages = ["vllm", "vllm_mlx", "vllm-mlx"]
    binary_names = ["vllm"]

    # Installation options
    install_options = [
        InstallOption(
            method=InstallMethod.UV,
            command="uv add vllm-mlx",
            description="Install vllm-mlx with uv",
            packages=["vllm-mlx"],
        ),
        InstallOption(
            method=InstallMethod.PIP,
            command="pip install vllm-mlx",
            description="Install vllm-mlx with pip",
            packages=["vllm-mlx"],
        ),
        InstallOption(
            method=InstallMethod.MANUAL,
            command="git clone https://github.com/vllm-project/vllm-metal && cd vllm-metal && pip install -e .",
            description="Install from source (vllm-metal)",
            packages=[],
        ),
    ]

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
        """Generate text using vllm-mlx."""
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
