"""Stats collection for performance and hardware monitoring."""

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import httpx


@dataclass
class RequestStats:
    """Stats for a single request."""
    request_id: str
    endpoint: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    status: str = "pending"  # pending, streaming, completed, error
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None


@dataclass
class InferenceStats:
    """Aggregated inference statistics."""
    total_requests: int = 0
    active_requests: int = 0
    queued_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    tokens_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    recent_requests: list[RequestStats] = field(default_factory=list)


@dataclass
class HardwareStats:
    """Hardware monitoring statistics."""
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    memory_percent: float = 0.0
    gpu_percent: float = 0.0  # Apple Silicon GPU utilization
    cpu_percent: float = 0.0
    temperature_c: float | None = None
    power_w: float | None = None


class StatsCollector:
    """Collects and aggregates performance and hardware stats."""

    def __init__(self, max_recent: int = 10):
        self.max_recent = max_recent
        self._requests: dict[str, RequestStats] = {}
        self._completed_requests: list[RequestStats] = []
        self._total_tokens = 0
        self._total_time = 0.0
        self._start_time = time.time()

    def start_request(self, request_id: str, endpoint: str) -> None:
        """Record start of a request."""
        self._requests[request_id] = RequestStats(
            request_id=request_id,
            endpoint=endpoint,
            status="streaming",
        )

    def update_request(
        self,
        request_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Update token counts for a request."""
        if request_id in self._requests:
            req = self._requests[request_id]
            req.prompt_tokens = prompt_tokens
            req.completion_tokens = completion_tokens

    def finish_request(
        self,
        request_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: bool = False,
    ) -> None:
        """Record completion of a request."""
        if request_id in self._requests:
            req = self._requests[request_id]
            req.finished_at = datetime.now()
            req.prompt_tokens = prompt_tokens
            req.completion_tokens = completion_tokens
            req.latency_ms = (req.finished_at - req.started_at).total_seconds() * 1000
            req.status = "error" if error else "completed"

            # Track totals
            self._total_tokens += completion_tokens
            self._total_time += req.latency_ms / 1000

            # Move to completed
            self._completed_requests.append(req)
            if len(self._completed_requests) > self.max_recent:
                self._completed_requests.pop(0)

            del self._requests[request_id]

    def get_inference_stats(self) -> InferenceStats:
        """Get current inference statistics."""
        active = list(self._requests.values())
        completed = self._completed_requests[-self.max_recent:]

        # Calculate tokens per second
        elapsed = time.time() - self._start_time
        tps = self._total_tokens / elapsed if elapsed > 0 else 0.0

        # Calculate average latency
        latencies = [r.latency_ms for r in completed if r.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Total tokens
        total_prompt = sum(r.prompt_tokens for r in completed)
        total_completion = sum(r.completion_tokens for r in completed)

        return InferenceStats(
            total_requests=len(completed),
            active_requests=len(active),
            queued_requests=0,  # TODO: get from engine
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            tokens_per_second=tps,
            avg_latency_ms=avg_latency,
            recent_requests=list(reversed(completed[-5:])),
        )

    def get_hardware_stats(self) -> HardwareStats:
        """Get current hardware statistics."""
        stats = HardwareStats()

        # Try apple-gpu first for GPU utilization on Apple Silicon
        try:
            from apple_gpu import accelerator_performance_statistics
            gpu_info = accelerator_performance_statistics()

            # apple-gpu returns dict with 'Device Utilization %' (0-100)
            if gpu_info:
                stats.gpu_percent = float(gpu_info.get('Device Utilization %', 0))
        except ImportError:
            # apple-gpu not installed, will show N/A for GPU
            pass
        except Exception:
            # apple-gpu failed, will show N/A for GPU
            pass

        # Fallback: Memory stats using vm_stat (always available on macOS)
        try:
            # Get total memory
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                stats.memory_total_gb = int(result.stdout.strip()) / (1024**3)

            # Get memory pressure/usage
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                page_size = 16384  # Default for Apple Silicon

                pages_active = 0
                pages_wired = 0
                pages_compressed = 0

                for line in lines:
                    if "page size of" in line:
                        page_size = int(line.split()[-2])
                    elif "Pages free:" in line:
                        pass  # Not used in memory calculation
                    elif "Pages active:" in line:
                        pages_active = int(line.split()[-1].rstrip("."))
                    elif "Pages inactive:" in line:
                        pass  # Not used in memory calculation
                    elif "Pages wired down:" in line:
                        pages_wired = int(line.split()[-1].rstrip("."))
                    elif "Pages occupied by compressor:" in line:
                        pages_compressed = int(line.split()[-1].rstrip("."))

                used_pages = pages_active + pages_wired + pages_compressed
                stats.memory_used_gb = (used_pages * page_size) / (1024**3)
                stats.memory_percent = (stats.memory_used_gb / stats.memory_total_gb) * 100 if stats.memory_total_gb > 0 else 0

        except Exception:
            pass

        # Fallback: CPU usage
        try:
            result = subprocess.run(
                ["ps", "-A", "-o", "%cpu"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                total_cpu = sum(float(line.strip()) for line in lines if line.strip())
                # Normalize to 100% (total across all cores)
                cpu_count = os.cpu_count() or 1
                stats.cpu_percent = min(total_cpu / cpu_count, 100.0)
        except Exception:
            pass

        # Fallback: Temperature from thermal sensors (if available)
        try:
            result = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0 and "CPU_Scheduler_Limit" in result.stdout:
                # Thermal throttling indicates high temp
                if "Speed_Limit" in result.stdout:
                    for line in result.stdout.split("\n"):
                        if "CPU_Speed_Limit" in line:
                            limit = int(line.split("=")[-1].strip())
                            if limit < 100:
                                stats.temperature_c = 80 + (100 - limit) * 0.5
        except Exception:
            pass

        # Note: GPU stats not available without macmon
        # stats.gpu_percent remains 0.0 (will show N/A in UI)

        return stats


    def poll_backend_stats(self, host: str, port: int) -> bool:
        """Poll backend stats endpoint for request metrics.

        Tries common stats endpoints used by OpenAI-compatible servers.
        Returns True if stats were successfully fetched.
        """
        endpoints = [
            f"http://{host}:{port}/metrics",  # Prometheus format (vLLM)
            f"http://{host}:{port}/v1/stats",  # Custom stats endpoint
            f"http://{host}:{port}/stats",  # Simple stats
        ]

        for endpoint in endpoints:
            try:
                response = httpx.get(endpoint, timeout=1.0)
                if response.status_code == 200:
                    return self._parse_stats_response(endpoint, response.text)
            except Exception:
                continue

        return False

    def _parse_stats_response(self, endpoint: str, text: str) -> bool:
        """Parse stats response based on endpoint type."""
        if "/metrics" in endpoint:
            return self._parse_prometheus_metrics(text)
        else:
            return self._parse_json_stats(text)

    def _parse_prometheus_metrics(self, text: str) -> bool:
        """Parse Prometheus-format metrics from vLLM."""
        try:
            for line in text.split("\n"):
                if line.startswith("#") or not line.strip():
                    continue

                # Parse vLLM metrics
                # vllm_request_success_total, vllm_num_requests_running, etc.
                if "vllm_num_requests_running" in line:
                    match = re.search(r"(\d+(?:\.\d+)?)\s*$", line)
                    if match:
                        # Update active requests count indirectly
                        pass

                if "vllm_avg_generation_throughput_toks_per_s" in line:
                    match = re.search(r"(\d+(?:\.\d+)?)\s*$", line)
                    if match:
                        # Could update tokens/s metric
                        pass

            return True
        except Exception:
            return False

    def _parse_json_stats(self, text: str) -> bool:
        """Parse JSON stats response."""
        import json
        try:
            data = json.loads(text)
            # Handle common stats formats
            if "requests" in data:
                pass  # Could update request counts
            return True
        except Exception:
            return False

    def parse_log_line(self, line: str) -> None:
        """Parse a backend log line to extract request info.

        Handles common log formats from vLLM, mlx-lm server, etc.
        """
        line = line.strip()
        if not line:
            return

        # Uvicorn format: INFO:     127.0.0.1:xxx - "POST /v1/chat/completions HTTP/1.1" 200 OK
        # vLLM format: Processed request req-xxx: prompt=N tokens, generation=M tokens
        # Or: prompt_tokens=N, completion_tokens=M, total_tokens=T

        # Uvicorn access log pattern
        uvicorn_match = re.search(
            r'"(POST|GET)\s+(/v1/(?:chat/)?completions)[^"]*"\s+(\d{3})',
            line
        )
        if uvicorn_match:
            self._handle_uvicorn_log(uvicorn_match)
            return

        # vLLM token stats pattern (various formats)
        token_patterns = [
            # prompt_tokens=N, completion_tokens=M
            r"prompt_tokens[=:]\s*(\d+).*?completion_tokens[=:]\s*(\d+)",
            # prompt=N.*generation=M or prompt=N.*completion=M
            r"prompt[=:]\s*(\d+).*?(?:generation|completion)[=:]\s*(\d+)",
            # Generated N tokens in Xs
            r"[Gg]enerated\s+(\d+)\s+tokens?\s+in\s+([\d.]+)",
            # throughput: X tokens/s
            r"throughput[=:]\s*([\d.]+)\s*tokens?/s",
        ]

        for pattern in token_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                self._handle_token_log(pattern, match)
                return

    def _handle_uvicorn_log(self, match: re.Match) -> None:
        """Handle uvicorn access log match."""
        import uuid

        groups = match.groups()
        endpoint = groups[1] if len(groups) > 1 else "/v1/completions"
        status = int(groups[2]) if len(groups) > 2 else 200

        request_id = str(uuid.uuid4())[:8]
        self.start_request(request_id, endpoint)
        self.finish_request(request_id, error=(status >= 400))

    def _handle_token_log(self, pattern: str, match: re.Match) -> None:
        """Handle token-related log match."""
        import uuid

        groups = match.groups()

        if "prompt_tokens" in pattern or "prompt[=:]" in pattern:
            # Token count pattern: prompt and completion tokens
            prompt_tokens = int(groups[0]) if groups[0] else 0
            completion_tokens = int(groups[1]) if len(groups) > 1 else 0

            # Update the most recent request with token counts
            if self._completed_requests:
                last_req = self._completed_requests[-1]
                last_req.prompt_tokens = prompt_tokens
                last_req.completion_tokens = completion_tokens
            else:
                # No recent request, create one
                request_id = str(uuid.uuid4())[:8]
                self.start_request(request_id, "/v1/completions")
                self.finish_request(
                    request_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

        elif "enerated" in pattern:
            # Generated N tokens in Xs
            tokens = int(groups[0]) if groups[0] else 0

            # Update the most recent request
            if self._completed_requests:
                self._completed_requests[-1].completion_tokens = tokens
            else:
                request_id = str(uuid.uuid4())[:8]
                self.start_request(request_id, "/v1/completions")
                self.finish_request(request_id, completion_tokens=tokens)

        elif "throughput" in pattern:
            # Throughput log - just informational, could update tps
            pass


class LogParser(threading.Thread):
    """Background thread to parse backend logs and update stats."""

    def __init__(self, process: subprocess.Popen, collector: "StatsCollector"):
        super().__init__(daemon=True)
        self.process = process
        self.collector = collector
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Read and parse stdout/stderr from the backend process."""
        import io
        import selectors

        sel = selectors.DefaultSelector()

        if self.process.stdout:
            sel.register(self.process.stdout, selectors.EVENT_READ, "stdout")
        if self.process.stderr:
            sel.register(self.process.stderr, selectors.EVENT_READ, "stderr")

        while not self._stop_event.is_set() and self.process.poll() is None:
            events = sel.select(timeout=0.1)
            for key, _ in events:
                stream = key.fileobj
                if isinstance(stream, io.IOBase):
                    raw_line = stream.readline()
                    if raw_line:
                        if isinstance(raw_line, bytes):
                            line_str = raw_line.decode("utf-8", errors="ignore")
                        else:
                            line_str = str(raw_line)
                        self.collector.parse_log_line(line_str)

        sel.close()

    def stop(self) -> None:
        """Signal the parser to stop."""
        self._stop_event.set()


# Global stats collector instance
_collector: StatsCollector | None = None


def get_stats_collector() -> StatsCollector:
    """Get the global stats collector."""
    global _collector
    if _collector is None:
        _collector = StatsCollector()
    return _collector


def reset_stats_collector() -> None:
    """Reset the global stats collector."""
    global _collector
    _collector = StatsCollector()
