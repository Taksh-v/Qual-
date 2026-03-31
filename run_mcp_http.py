from __future__ import annotations

import contextlib

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mcp_integration.config import load_settings
from mcp_integration.server import mcp

_settings = load_settings()


class OriginGuardMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not _settings.require_origin_check:
            return await call_next(request)

        if request.url.path.startswith("/mcp"):
            origin = (request.headers.get("origin") or "").strip()
            if origin and origin not in _settings.allowed_origins:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Origin not allowed"},
                )
        return await call_next(request)


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "qual_mcp"})


@contextlib.asynccontextmanager
async def lifespan(_: Starlette):
    async with mcp.session_manager.run():
        yield


app = Starlette(
    routes=[
        Route("/health", health),
        Mount("/", app=mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)
app.add_middleware(OriginGuardMiddleware)


def main() -> None:
    uvicorn.run(app, host=_settings.host, port=_settings.port, log_level="info")


if __name__ == "__main__":
    main()
