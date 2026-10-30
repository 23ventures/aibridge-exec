"""HTTP API over the provider adapters. Stateless: keys arrive per request and are never stored.

Endpoints
    GET  /health
    GET  /v1/providers
    POST /v1/models     {"provider"}                              header X-Provider-Key
    POST /v1/chat       {"provider", "model", "messages", ...}    header X-Provider-Key
"""
from __future__ import annotations

import hmac
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..providers import AsyncChatClient, ProviderError, get_provider, list_providers
from ..providers.errors import HTTP_STATUS_FOR_CODE
from .settings import ServerSettings

log = logging.getLogger("aibridge_exec.server")

KEY_HEADER = "X-Provider-Key"


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatBody(BaseModel):
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=256)
    messages: list[Message] = Field(min_length=1)
    system: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    stream: bool = True
    base_url: str | None = Field(default=None, max_length=2048)


class ModelsBody(BaseModel):
    provider: str = Field(min_length=1, max_length=64)
    base_url: str | None = Field(default=None, max_length=2048)


def _error(code: str, message: str, status: int | None = None, **extra: object) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, **extra}},
                        status_code=status or HTTP_STATUS_FOR_CODE.get(code, 400))


def create_app(settings: ServerSettings | None = None, *, client: AsyncChatClient | None = None) -> FastAPI:
    settings = settings or ServerSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.client = client or AsyncChatClient(base_urls=settings.base_urls)
        if not settings.api_tokens:
            log.warning("AIBRIDGE_EXEC_API_TOKENS is not set: the server accepts unauthenticated requests")
        yield
        if client is None:
            await app.state.client.aclose()

    app = FastAPI(title="aibridge-exec", version=__version__, lifespan=lifespan,
                  docs_url="/docs", redoc_url=None)
    app.state.settings = settings
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type", KEY_HEADER])

    @app.exception_handler(ProviderError)
    async def provider_error(_: Request, exc: ProviderError) -> JSONResponse:
        return _error(exc.code, exc.message, provider=exc.provider, upstream_status=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e.get("loc", ())), "msg": e.get("msg")} for e in exc.errors()]
        return _error("bad_request", "invalid request body", 422, details=details)

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if not settings.api_tokens:
            return
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not any(hmac.compare_digest(token.encode(), t.encode()) for t in settings.api_tokens):
            raise ProviderError("auth", "missing or invalid service token")

    def enabled(provider_id: str) -> bool:
        if not settings.provider_enabled(provider_id):
            return False
        try:
            needs_url = get_provider(provider_id).needs_custom_base_url
        except ProviderError:
            return True  # unknown ids are reported by resolve()
        # openai_compatible and ollama have no public endpoint: only with a URL to call.
        return not needs_url or settings.allow_custom_base_url or provider_id in settings.base_urls

    def resolve(provider_id: str, base_url: str | None):
        if not enabled(provider_id):
            raise ProviderError("unknown_provider", f"provider {provider_id!r} is not enabled on this server",
                                provider=provider_id)
        adapter = get_provider(provider_id)
        if base_url and not settings.allow_custom_base_url:
            raise ProviderError("config", "custom base_url is disabled on this server", provider=provider_id)
        if base_url and not base_url.startswith(("https://", "http://")):
            raise ProviderError("config", "base_url must be an http(s) URL", provider=provider_id)
        return adapter

    def check_size(body: ChatBody) -> None:
        if len(body.messages) > settings.max_messages:
            raise ProviderError("bad_request", f"too many messages (max {settings.max_messages})")
        total = len(body.system or "")
        for message in body.messages:
            if len(message.content) > settings.max_message_chars:
                raise ProviderError("bad_request", f"message too long (max {settings.max_message_chars} chars)")
            total += len(message.content)
        if total > settings.max_total_chars:
            raise ProviderError("bad_request", f"conversation too long (max {settings.max_total_chars} chars)")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/providers", dependencies=[Depends(authorize)])
    async def providers() -> dict[str, list[dict[str, str]]]:
        return {"providers": [p.info.to_dict() for p in list_providers() if enabled(p.id)]}

    @app.post("/v1/models", dependencies=[Depends(authorize)])
    async def models(body: ModelsBody, request: Request,
                     x_provider_key: Annotated[str | None, Header()] = None) -> dict[str, list[dict]]:
        resolve(body.provider, body.base_url)
        found = await request.app.state.client.list_models(body.provider, api_key=x_provider_key or "",
                                                           base_url=body.base_url)
        return {"models": [m.to_dict() for m in found]}

    @app.post("/v1/chat", dependencies=[Depends(authorize)])
    async def chat(body: ChatBody, request: Request,
                   x_provider_key: Annotated[str | None, Header()] = None):
        resolve(body.provider, body.base_url)
        check_size(body)
        client: AsyncChatClient = request.app.state.client
        kwargs = dict(api_key=x_provider_key or "", model=body.model,
                      messages=[m.model_dump() for m in body.messages], system=body.system,
                      temperature=body.temperature, max_tokens=body.max_tokens, base_url=body.base_url)
        if not body.stream:
            return (await client.chat(body.provider, **kwargs)).to_dict()

        # Fail fast (auth, unknown model) with a normal HTTP error before the stream starts.
        stream = client.stream(body.provider, **kwargs)
        first = await anext(stream)

        async def events() -> AsyncIterator[str]:
            try:
                yield _sse(first.to_dict())
                async for event in stream:
                    yield _sse(event.to_dict())
            except ProviderError as exc:
                yield _sse({"type": "error", "error": exc.to_dict()})
            finally:
                await stream.aclose()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def _sse(payload: dict) -> str:
    return f"event: {payload['type']}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
