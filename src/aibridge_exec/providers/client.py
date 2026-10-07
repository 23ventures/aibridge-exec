"""Run adapter calls over HTTP (``httpx``), sync and async.

The API key is passed per call and only ever placed in the request headers.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from typing import Any

import httpx

from .base import HttpCall, ProviderAdapter
from .errors import ProviderError, redact
from .registry import get_provider
from .sse import SSEDecoder
from .types import ChatMessage, ChatRequest, ChatResponse, ModelInfo, StreamEvent

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
MAX_MODEL_PAGES = 20

Messages = Iterable["ChatMessage | Mapping[str, Any]"]


def _request(model: str, messages: Messages, system: str | None, temperature: float | None,
             max_tokens: int | None) -> ChatRequest:
    try:
        return ChatRequest.build(model, messages, system=system, temperature=temperature, max_tokens=max_tokens)
    except (ValueError, TypeError, KeyError) as exc:
        raise ProviderError("bad_request", f"invalid request: {exc}") from None


def _require_key(adapter: ProviderAdapter, api_key: str | None) -> str:
    if not api_key or not api_key.strip():
        raise ProviderError("auth", "an API key is required", provider=adapter.id)
    return api_key.strip()


def _transport_error(adapter: ProviderAdapter, exc: Exception, api_key: str | None) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError("timeout", "the provider did not respond in time", provider=adapter.id)
    return ProviderError("network", redact(f"could not reach the provider: {type(exc).__name__}", api_key),
                         provider=adapter.id)


def _json(adapter: ProviderAdapter, response: httpx.Response, api_key: str) -> Any:
    try:
        return response.json()
    except ValueError:
        raise ProviderError("upstream", "the provider returned an invalid response", provider=adapter.id,
                            status=response.status_code) from None


class _Base:
    def __init__(self, *, base_urls: Mapping[str, str] | None = None) -> None:
        self.base_urls = dict(base_urls or {})

    def _base_url(self, adapter: ProviderAdapter, base_url: str | None) -> str | None:
        return base_url or self.base_urls.get(adapter.id)


class ChatClient(_Base):
    """Synchronous client.

    >>> client = ChatClient()
    >>> reply = client.chat("openai", api_key="sk-...", model="gpt-4o-mini",
    ...                     messages=[{"role": "user", "content": "Hello"}])
    >>> reply.text
    """

    def __init__(self, *, http: httpx.Client | None = None, base_urls: Mapping[str, str] | None = None,
                 timeout: httpx.Timeout = DEFAULT_TIMEOUT) -> None:
        super().__init__(base_urls=base_urls)
        self._http = http or httpx.Client(timeout=timeout)
        self._owns_http = http is None

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "ChatClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _send(self, adapter: ProviderAdapter, call: HttpCall, api_key: str) -> httpx.Response:
        try:
            response = self._http.request(call.method, call.url, headers=call.headers, params=call.params,
                                          json=call.json)
        except httpx.HTTPError as exc:
            raise _transport_error(adapter, exc, api_key) from None
        if response.status_code >= 400:
            raise adapter.error_from_response(response.status_code, response.text, api_key)
        return response

    def chat(self, provider: str, *, api_key: str, model: str, messages: Messages, system: str | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             base_url: str | None = None) -> ChatResponse:
        adapter = get_provider(provider)
        key = _require_key(adapter, api_key)
        request = _request(model, messages, system, temperature, max_tokens)
        call = adapter.chat_call(request, key, base_url=self._base_url(adapter, base_url), stream=False)
        return adapter.parse_chat(_json(adapter, self._send(adapter, call, key), key), request)

    def stream(self, provider: str, *, api_key: str, model: str, messages: Messages, system: str | None = None,
               temperature: float | None = None, max_tokens: int | None = None,
               base_url: str | None = None) -> Iterator[StreamEvent]:
        adapter = get_provider(provider)
        key = _require_key(adapter, api_key)
        request = _request(model, messages, system, temperature, max_tokens)
        call = adapter.chat_call(request, key, base_url=self._base_url(adapter, base_url), stream=True)
        parser, decoder = adapter.stream_parser(request), SSEDecoder()
        try:
            with self._http.stream(call.method, call.url, headers=call.headers, params=call.params,
                                   json=call.json) as response:
                if response.status_code >= 400:
                    raise adapter.error_from_response(response.status_code, response.read().decode(errors="replace"),
                                                      key)
                for chunk in response.iter_text():
                    for event, data in decoder.feed(chunk):
                        yield from parser.feed(event, data)
                for event, data in decoder.flush():
                    yield from parser.feed(event, data)
        except httpx.HTTPError as exc:
            raise _transport_error(adapter, exc, key) from None
        yield parser.finish()

    def list_models(self, provider: str, *, api_key: str, base_url: str | None = None) -> list[ModelInfo]:
        adapter = get_provider(provider)
        key = _require_key(adapter, api_key)
        call: HttpCall | None = adapter.models_call(key, base_url=self._base_url(adapter, base_url))
        models: list[ModelInfo] = []
        for _ in range(MAX_MODEL_PAGES):
            if call is None:
                break
            payload = _json(adapter, self._send(adapter, call, key), key)
            models.extend(adapter.parse_models(payload))
            call = adapter.next_models_page(payload, call)
        return models


class AsyncChatClient(_Base):
    """Asynchronous client with the same methods as ``ChatClient``; ``stream`` is an async iterator."""

    def __init__(self, *, http: httpx.AsyncClient | None = None, base_urls: Mapping[str, str] | None = None,
                 timeout: httpx.Timeout = DEFAULT_TIMEOUT) -> None:
        super().__init__(base_urls=base_urls)
        self._http = http or httpx.AsyncClient(timeout=timeout)
        self._owns_http = http is None

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> "AsyncChatClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _send(self, adapter: ProviderAdapter, call: HttpCall, api_key: str) -> httpx.Response:
        try:
            response = await self._http.request(call.method, call.url, headers=call.headers, params=call.params,
                                                json=call.json)
        except httpx.HTTPError as exc:
            raise _transport_error(adapter, exc, api_key) from None
        if response.status_code >= 400:
            raise adapter.error_from_response(response.status_code, response.text, api_key)
        return response

    async def chat(self, provider: str, *, api_key: str, model: str, messages: Messages, system: str | None = None,
                   temperature: float | None = None, max_tokens: int | None = None,
                   base_url: str | None = None) -> ChatResponse:
        adapter = get_provider(provider)
        key = _require_key(adapter, api_key)
        request = _request(model, messages, system, temperature, max_tokens)
        call = adapter.chat_call(request, key, base_url=self._base_url(adapter, base_url), stream=False)
        return adapter.parse_chat(_json(adapter, await self._send(adapter, call, key), key), request)

    async def stream(self, provider: str, *, api_key: str, model: str, messages: Messages,
                     system: str | None = None, temperature: float | None = None, max_tokens: int | None = None,
                     base_url: str | None = None) -> AsyncIterator[StreamEvent]:
        adapter = get_provider(provider)
        key = _require_key(adapter, api_key)
        request = _request(model, messages, system, temperature, max_tokens)
        call = adapter.chat_call(request, key, base_url=self._base_url(adapter, base_url), stream=True)
        parser, decoder = adapter.stream_parser(request), SSEDecoder()
        try:
            async with self._http.stream(call.method, call.url, headers=call.headers, params=call.params,
                                         json=call.json) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise adapter.error_from_response(response.status_code, body, key)
                async for chunk in response.aiter_text():
                    for event, data in decoder.feed(chunk):
                        for item in parser.feed(event, data):
                            yield item
                for event, data in decoder.flush():
                    for item in parser.feed(event, data):
                        yield item
        except httpx.HTTPError as exc:
            raise _transport_error(adapter, exc, key) from None
        yield parser.finish()

    async def list_models(self, provider: str, *, api_key: str, base_url: str | None = None) -> list[ModelInfo]:
        adapter = get_provider(provider)
        key = _require_key(adapter, api_key)
        call: HttpCall | None = adapter.models_call(key, base_url=self._base_url(adapter, base_url))
        models: list[ModelInfo] = []
        for _ in range(MAX_MODEL_PAGES):
            if call is None:
                break
            payload = _json(adapter, await self._send(adapter, call, key), key)
            models.extend(adapter.parse_models(payload))
            call = adapter.next_models_page(payload, call)
        return models


def stream_event_json(event: StreamEvent) -> str:
    return json.dumps(event.to_dict(), separators=(",", ":"))
