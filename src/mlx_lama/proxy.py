"""Lightweight stats-capturing proxy for backend servers."""

import json
import time
from collections.abc import Callable

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
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

        # Check if this is a streaming request
        is_streaming = False
        try:
            body_json = json.loads(body)
            is_streaming = body_json.get("stream", False)
        except Exception:
            pass

        if is_streaming:
            # Handle streaming response - keep client alive during streaming
            client = httpx.AsyncClient(timeout=300.0)
            req = client.build_request(
                method=request.method,
                url=url,
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                content=body,
            )
            response = await client.send(req, stream=True)

            prompt_tokens = 0
            completion_tokens = 0

            async def stream_generator():
                nonlocal prompt_tokens, completion_tokens
                try:
                    async for chunk in response.aiter_bytes():
                        # Try to parse SSE data for usage info
                        try:
                            chunk_str = chunk.decode("utf-8")
                            for line in chunk_str.split("\n"):
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    data = json.loads(line[6:])
                                    if "usage" in data:
                                        prompt_tokens = data["usage"].get("prompt_tokens", 0)
                                        completion_tokens = data["usage"].get(
                                            "completion_tokens", 0
                                        )
                        except Exception:
                            pass
                        yield chunk
                finally:
                    await response.aclose()
                    await client.aclose()

                    # Report stats after stream completes
                    latency_ms = (time.time() - start_time) * 1000
                    if on_request_complete and "/v1/" in path:
                        on_request_complete(path, prompt_tokens, completion_tokens, latency_ms)

            return StreamingResponse(
                stream_generator(),
                status_code=response.status_code,
                headers={
                    k: v
                    for k, v in response.headers.items()
                    if k.lower()
                    not in ("content-length", "content-encoding", "transfer-encoding")
                },
                media_type=response.headers.get("content-type", "text/event-stream"),
            )
        else:
            # Handle non-streaming response
            async with httpx.AsyncClient(timeout=300.0) as client:
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

                content_type = response.headers.get("content-type", "")
                if content_type.startswith("application/json"):
                    return JSONResponse(
                        content=response.json(),
                        status_code=response.status_code,
                        headers={
                            k: v
                            for k, v in response.headers.items()
                            if k.lower()
                            not in ("content-length", "content-encoding", "transfer-encoding")
                        },
                    )
                else:
                    from starlette.responses import Response

                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        headers={
                            k: v
                            for k, v in response.headers.items()
                            if k.lower()
                            not in ("content-length", "content-encoding", "transfer-encoding")
                        },
                        media_type=content_type,
                    )

    # Routes - catch all
    routes = [
        Route(
            "/{path:path}",
            proxy_request,
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        ),
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
