"""Ollama backend implementation - uses system Ollama installation."""

import subprocess
from typing import Generator

import httpx

from .base import Backend, BackendCapabilities, InstallOption, InstallMethod, DetectionResult
from . import register_backend


class OllamaBackend(Backend):
    """Backend using system Ollama installation.

    This backend connects to an already-running Ollama server,
    or can start one if Ollama is installed via brew/binary.
    """

    name = "ollama"
    description = "System Ollama - GGUF models, easy setup"

    # Detection configuration
    python_packages = []  # Ollama is typically not a Python package
    binary_names = ["ollama"]
    service_ports = [11434]
    service_endpoints = ["http://127.0.0.1:11434/api/tags"]

    # Installation options
    install_options = [
        InstallOption(
            method=InstallMethod.BREW,
            command="brew install ollama",
            description="Install with Homebrew (recommended)",
            packages=["ollama"],
        ),
        InstallOption(
            method=InstallMethod.MANUAL,
            command="curl -fsSL https://ollama.com/install.sh | sh",
            description="Install via official script",
            packages=[],
        ),
    ]

    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self._process: subprocess.Popen | None = None

    def detect(self) -> DetectionResult:
        """Detect Ollama - check service first, then binary."""
        # Check if service is already running
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            if response.status_code == 200:
                return DetectionResult(
                    available=True,
                    method="service",
                    details=f"Ollama running at {self.base_url}"
                )
        except Exception:
            pass

        # Check for binary
        result = self._detect_binary()
        if result.available:
            return result

        return DetectionResult(
            available=False,
            method="none",
            details="Ollama not found. Install with: brew install ollama"
        )

    def capabilities(self) -> BackendCapabilities:
        """Ollama capabilities."""
        return BackendCapabilities(
            continuous_batching=True,
            vision_support=True,
            tool_calling=True,
            max_concurrent=10,
        )

    def _ensure_service(self) -> bool:
        """Ensure Ollama service is running."""
        # Check if already running
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            if response.status_code == 200:
                return True
        except Exception:
            pass

        # Try to start service
        detection = self._detect_binary()
        if not detection.available or not detection.path:
            return False

        try:
            self._process = subprocess.Popen(
                [detection.path, "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Wait for service to be ready
            import time
            for _ in range(30):
                try:
                    response = httpx.get(f"{self.base_url}/api/tags", timeout=1.0)
                    if response.status_code == 200:
                        return True
                except Exception:
                    pass
                time.sleep(0.5)

            return False

        except Exception:
            return False

    def generate(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """Generate text using Ollama API."""
        if not self._ensure_service():
            raise RuntimeError("Ollama service not available")

        # For Ollama, model_path might be a local path or model name
        # If it's a path, we need to use the model name from registry
        model_name = model_path.split("/")[-1] if "/" in model_path else model_path

        with httpx.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
                "stream": stream,
            },
            timeout=None,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]

    def serve(
        self,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> subprocess.Popen:
        """Ollama already serves on its port.

        For compatibility, we'll start a proxy or just return the existing process.
        Note: Ollama uses port 11434 by default, not the requested port.
        """
        if not self._ensure_service():
            raise RuntimeError("Failed to start Ollama service")

        # Return a simple process that keeps running
        # In a real implementation, you might proxy to the requested port
        process = subprocess.Popen(
            ["sleep", "infinity"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return process

    def chat(
        self,
        model_path: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """Chat using Ollama's native chat API."""
        if not self._ensure_service():
            raise RuntimeError("Ollama service not available")

        model_name = model_path.split("/")[-1] if "/" in model_path else model_path

        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": model_name,
                "messages": messages,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
                "stream": stream,
            },
            timeout=None,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]


# Register this backend
register_backend("ollama", OllamaBackend)
