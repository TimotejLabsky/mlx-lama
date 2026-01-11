"""Lightweight stats-capturing proxy for backend servers."""

import time
from collections.abc import Callable

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


def create_proxy_app(
    backend_url: str,
    on_request_complete: Callable[[str, int, int, float], None] | None = None,
) -> Starlette:
    """Create a lightweight proxy that captures request stats.

    Args:
        backend_url: URL of the backend server (e.g., "http://127.0.0.1:8001")
        on_request_complete: Callback(endpoint, prompt_tokens, completion_tokens, latency_ms)
    """

    async def proxy_request(request: Request):
        """Forward request to backend and capture stats."""
        start_time = time.time()
        path = request.url.path
        query = str(request.url.query)
        url = f"{backend_url}{path}"
        if query:
            url = f"{url}?{query}"

        # Read request body
        body = await request.body()

        async with httpx.AsyncClient(timeout=300.0) as client:
            # Forward request
            response = await client.request(
                method=request.method,
                url=url,
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                content=body,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract usage from JSON responses
            prompt_tokens = 0
            completion_tokens = 0

            if response.headers.get("content-type", "").startswith("application/json"):
                try:
                    data = response.json()
                    if "usage" in data:
                        prompt_tokens = data["usage"].get("prompt_tokens", 0)
                        completion_tokens = data["usage"].get("completion_tokens", 0)
                except Exception:
                    pass

            # Notify callback
            if on_request_complete and "/v1/" in path:
                on_request_complete(path, prompt_tokens, completion_tokens, latency_ms)

            return JSONResponse(
                content=response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                status_code=response.status_code,
                headers={k: v for k, v in response.headers.items() if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")},
            )

    # Routes - catch all
    routes = [
        Route("/{path:path}", proxy_request, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]),
    ]

    return Starlette(routes=routes)


async def run_proxy(
    listen_host: str,
    listen_port: int,
    backend_host: str,
    backend_port: int,
    on_request_complete: Callable[[str, int, int, float], None] | None = None,
):
    """Run the proxy server.

    Args:
        listen_host: Host to listen on
        listen_port: Port to listen on
        backend_host: Backend host
        backend_port: Backend port
        on_request_complete: Stats callback
    """
    import uvicorn

    backend_url = f"http://{backend_host}:{backend_port}"
    app = create_proxy_app(backend_url, on_request_complete)

    config = uvicorn.Config(
        app,
        host=listen_host,
        port=listen_port,
        log_level="warning",  # Minimize logging
    )
    server = uvicorn.Server(config)
    await server.serve()
