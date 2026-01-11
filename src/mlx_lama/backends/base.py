"""Base backend interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator
import shutil
import subprocess


class InstallMethod(Enum):
    """How a backend can be installed."""
    PIP = "pip"
    UV = "uv"
    BREW = "brew"
    MANUAL = "manual"


@dataclass
class DetectionResult:
    """Result of backend detection."""
    available: bool
    method: str  # How it was detected (python, binary, service, etc.)
    version: str | None = None
    path: str | None = None
    details: str | None = None


@dataclass
class InstallOption:
    """An installation option for a backend."""
    method: InstallMethod
    command: str
    description: str
    packages: list[str] = field(default_factory=list)


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

    # Detection configuration - subclasses should override
    python_packages: list[str] = []  # Python packages to check
    binary_names: list[str] = []  # System binaries to check
    service_ports: list[int] = []  # Ports to check for running services
    service_endpoints: list[str] = []  # HTTP endpoints to check

    # Installation options - subclasses should override
    install_options: list[InstallOption] = []

    def detect(self) -> DetectionResult:
        """Detect if this backend is available.

        Checks in order:
        1. Running service (fastest if already running)
        2. Python package
        3. System binary

        Returns:
            DetectionResult with availability info
        """
        # Check for running service first
        if self.service_ports or self.service_endpoints:
            result = self._detect_service()
            if result.available:
                return result

        # Check Python packages
        if self.python_packages:
            result = self._detect_python()
            if result.available:
                return result

        # Check system binaries
        if self.binary_names:
            result = self._detect_binary()
            if result.available:
                return result

        return DetectionResult(
            available=False,
            method="none",
            details="No installation found"
        )

    def _detect_python(self) -> DetectionResult:
        """Detect Python package installation."""
        for package in self.python_packages:
            try:
                # Try to import the package
                module = __import__(package.replace("-", "_"))
                version = getattr(module, "__version__", "unknown")
                return DetectionResult(
                    available=True,
                    method="python",
                    version=version,
                    details=f"Python package: {package}"
                )
            except ImportError:
                continue

        return DetectionResult(available=False, method="python")

    def _detect_binary(self) -> DetectionResult:
        """Detect system binary installation."""
        for binary in self.binary_names:
            path = shutil.which(binary)
            if path:
                # Try to get version
                version = self._get_binary_version(path)
                return DetectionResult(
                    available=True,
                    method="binary",
                    version=version,
                    path=path,
                    details=f"System binary: {path}"
                )

        return DetectionResult(available=False, method="binary")

    def _detect_service(self) -> DetectionResult:
        """Detect running service."""
        import httpx

        # Check endpoints
        for endpoint in self.service_endpoints:
            try:
                response = httpx.get(endpoint, timeout=1.0)
                if response.status_code == 200:
                    return DetectionResult(
                        available=True,
                        method="service",
                        details=f"Running service: {endpoint}"
                    )
            except Exception:
                continue

        # Check ports
        for port in self.service_ports:
            if self._is_port_open(port):
                return DetectionResult(
                    available=True,
                    method="service",
                    details=f"Service on port {port}"
                )

        return DetectionResult(available=False, method="service")

    def _get_binary_version(self, path: str) -> str | None:
        """Try to get version from a binary."""
        for flag in ["--version", "-version", "version", "-V"]:
            try:
                result = subprocess.run(
                    [path, flag],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout:
                    # Extract first line, often contains version
                    return result.stdout.strip().split("\n")[0][:50]
            except Exception:
                continue
        return None

    def _is_port_open(self, port: int, host: str = "127.0.0.1") -> bool:
        """Check if a port is open."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex((host, port))
            return result == 0
        finally:
            sock.close()

    def is_available(self) -> bool:
        """Check if this backend is available (any method)."""
        return self.detect().available

    def get_install_commands(self) -> list[InstallOption]:
        """Get available installation options."""
        return self.install_options

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
        """Generate text from a prompt."""
        ...

    @abstractmethod
    def serve(
        self,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> subprocess.Popen:
        """Start an OpenAI-compatible server."""
        ...

    def chat(
        self,
        model_path: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """Chat with a model using message format."""
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
