from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import ClassVar

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logging import configure_logging, log
from app.routes import api_router


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    log.info(
        "promptlabs.api.startup",
        api_key_required=bool(settings.api_key),
        default_model=settings.default_model,
    )
    yield
    log.info("promptlabs.api.shutdown")


app = FastAPI(
    title="PromptLabs API",
    version="0.1.0",
    description="The closed prompt-engineering loop.",
    lifespan=lifespan,
)


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Require `Authorization: Bearer <api_key>` on every non-public route when
    `PROMPTLABS_API_KEY` is set. No-op otherwise (local dev).
    """

    PUBLIC_PATHS: ClassVar[set[str]] = {"/healthz", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if not settings.api_key or request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth[7:].strip() != settings.api_key:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                detail="missing_or_invalid_api_key",
            )
        return await call_next(request)


app.add_middleware(BearerTokenMiddleware)

# CORS: if explicit origins provided via env, allow them; otherwise allow localhost:* for dev
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router)
