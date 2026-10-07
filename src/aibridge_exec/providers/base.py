"""Adapter interface: each provider describes HTTP calls and parses responses.

Adapters do no I/O. ``ChatClient``/``AsyncChatClient`` run the calls, so the same adapter
works sync or async and is tested without a network.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .errors import ProviderError, code_for_status, redact
from .types import ChatRequest, ChatResponse, ModelInfo, ProviderInfo, StreamEvent


@dataclass(frozen=True)
class HttpCall:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    json: dict[str, Any] | None = None


class StreamParser(ABC):
    """Stateful parser for one streamed response."""

    @abstractmethod
    def feed(self, event: str | None, data: str) -> list[StreamEvent]:
        """Handle one SSE event; return zero or more stream events."""

    @abstractmethod
    def finish(self) -> StreamEvent:
        """The final ``done`` event (usage, finish reason) once the stream ends."""


class ProviderAdapter(ABC):
    info: ProviderInfo

    @property
    def id(self) -> str:
        return self.info.id

    def base_url(self, override: str | None) -> str:
        return (override or self.info.default_base_url).rstrip("/")

    @abstractmethod
    def chat_call(self, request: ChatRequest, api_key: str, *, base_url: str | None, stream: bool) -> HttpCall:
        ...

    @abstractmethod
    def parse_chat(self, payload: dict[str, Any], request: ChatRequest) -> ChatResponse:
        ...

    @abstractmethod
    def stream_parser(self, request: ChatRequest) -> StreamParser:
        ...

    @abstractmethod
    def models_call(self, api_key: str, *, base_url: str | None) -> HttpCall:
        ...

    @abstractmethod
    def parse_models(self, payload: Any) -> list[ModelInfo]:
        ...

    def next_models_page(self, payload: Any, call: HttpCall) -> HttpCall | None:
        """Follow-up call for paginated model lists, if any."""
        return None

    def error_from_response(self, status: int, body: str, api_key: str | None) -> ProviderError:
        message = _error_message(body) or f"HTTP {status}"
        return ProviderError(code_for_status(status), redact(message, api_key)[:500], provider=self.id, status=status)


def _error_message(body: str) -> str | None:
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return body.strip()[:300] or None
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    error = payload.get("error", payload)
    if isinstance(error, dict):
        message = error.get("message") or error.get("detail")
        if isinstance(message, str):
            return message
    if isinstance(error, str):
        return error
    message = payload.get("message") or payload.get("detail")
    return message if isinstance(message, str) else None


def stream_error(adapter_id: str, payload: dict[str, Any], api_key: str | None) -> ProviderError:
    error = payload.get("error") if isinstance(payload.get("error"), dict) else payload
    message = error.get("message") if isinstance(error, dict) else None
    kind = str(error.get("type") or error.get("code") or "") if isinstance(error, dict) else ""
    code = "rate_limited" if "rate" in kind else "upstream" if "overload" in kind or "server" in kind else "upstream"
    return ProviderError(code, redact(str(message or "stream error"), api_key)[:500], provider=adapter_id)
