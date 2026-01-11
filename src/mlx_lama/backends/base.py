"""Base backend interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Optional
import subprocess


@dataclass
class BackendCapabilities:
    """Capabilities of a backend."""

    continuous_batching: bool = False
    vision_support: bool = False
    tool_calling: bool = False
    max_concurrent: int = 1


class Backend(ABC):
    """Abstract base class for inference backends."""

    name: str = "base"
    description: str = "Base backend"

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend's dependencies are installed."""
        ...

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Get the capabilities of this backend."""
        ...

    @abstractmethod
    def generate(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """Generate text from a prompt.

        Args:
            model_path: Path to the model directory
            prompt: The prompt to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream tokens

        Yields:
            Generated text tokens
        """
        ...

    @abstractmethod
    def serve(
        self,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> subprocess.Popen:
        """Start an OpenAI-compatible server.

        Args:
            model_path: Path to the model directory
            host: Host to bind to
            port: Port to listen on

        Returns:
            The server process
        """
        ...

    def chat(
        self,
        model_path: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """Chat with a model using message format.

        Default implementation converts to prompt format.
        Subclasses can override for native chat support.
        """
        # Simple conversion to prompt format
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)

        yield from self.generate(
            model_path,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

    def stop_server(self, process: subprocess.Popen) -> None:
        """Stop a running server process."""
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
