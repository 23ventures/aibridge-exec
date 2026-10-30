"""Cohere v2 Chat API (native)."""
from __future__ import annotations

import json
from typing import Any

from .base import HttpCall, ProviderAdapter, StreamParser, stream_error
from .types import ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent, Usage

_FINISH: dict[str, FinishReason] = {
    "COMPLETE": "stop",
    "STOP_SEQUENCE": "stop",
    "MAX_TOKENS": "length",
    "TOOL_CALL": "tool_calls",
    "ERROR": "error",
    "TIMEOUT": "error",
}


class CohereAdapter(ProviderAdapter):
    info = ProviderInfo(
        id="cohere",
        label="Cohere",
        default_base_url="https://api.cohere.com",
        key_url="https://dashboard.cohere.com/api-keys",
    )

    def _headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}

    def chat_call(self, request: ChatRequest, api_key: str, *, base_url: str | None, stream: bool) -> HttpCall:
        system, turns = request.split_system()
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": m.role, "content": m.content} for m in turns
        ]
        body: dict[str, Any] = {"model": request.model, "messages": messages, "stream": stream}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            body["response_format"] = {"type": "json_object"}
            if request.response_format.schema:
                body["response_format"]["json_schema"] = request.response_format.json_schema()
        if request.thinking is not None:
            body["thinking"] = {"type": "enabled" if request.thinking else "disabled"}
        return HttpCall("POST", f"{self.base_url(base_url)}/v2/chat", self._headers(api_key), json=body)

    def parse_chat(self, payload: dict[str, Any], request: ChatRequest) -> ChatResponse:
        if payload.get("type") == "error" or (payload.get("message") and isinstance(payload["message"], str)):
            raise stream_error(self.id, payload, None)
        message = payload.get("message") or {}
        text = "".join(part.get("text", "") for part in message.get("content") or []
                       if isinstance(part, dict) and part.get("type") == "text")
        return ChatResponse(text=text, model=request.model, finish_reason=_finish(payload.get("finish_reason")),
                            usage=_usage(payload.get("usage")))

    def stream_parser(self, request: ChatRequest) -> StreamParser:
        return _CohereStreamParser(self.id, request.model)

    def models_call(self, api_key: str, *, base_url: str | None) -> HttpCall:
        return HttpCall("GET", f"{self.base_url(base_url)}/v1/models", self._headers(api_key),
                        params={"endpoint": "chat", "page_size": "1000"})

    def parse_models(self, payload: Any) -> list[ModelInfo]:
        models = []
        for item in (payload or {}).get("models", []):
            if isinstance(item, dict) and item.get("name"):
                context = item.get("context_length")
                models.append(ModelInfo(str(item["name"]), None, int(context) if isinstance(context, (int, float))
                                        else None))
        return sorted(models, key=lambda m: m.id)

    def next_models_page(self, payload: Any, call: HttpCall) -> HttpCall | None:
        token = payload.get("next_page_token") if isinstance(payload, dict) else None
        return HttpCall(call.method, call.url, call.headers, {**call.params, "page_token": token}) if token else None


class _CohereStreamParser(StreamParser):
    def __init__(self, provider_id: str, model: str) -> None:
        self.provider_id = provider_id
        self.model = model
        self.finish_reason: FinishReason | None = None
        self.usage = Usage()

    def feed(self, event: str | None, data: str) -> list[StreamEvent]:
        try:
            payload = json.loads(data)
        except ValueError:
            return []
        if not isinstance(payload, dict):
            return []
        kind = payload.get("type") or event
        if kind == "error" or isinstance(payload.get("error"), dict):
            raise stream_error(self.provider_id, payload, None)
        delta = payload.get("delta") or {}
        if kind == "content-delta":
            content = ((delta.get("message") or {}).get("content") or {})
            text = content.get("text") if isinstance(content, dict) else None
            return [StreamEvent("delta", text=text)] if text else []
        if kind == "message-end":
            self.finish_reason = _finish(delta.get("finish_reason"))
            self.usage = _usage(delta.get("usage"))
        return []

    def finish(self) -> StreamEvent:
        return StreamEvent("done", finish_reason=self.finish_reason, usage=self.usage, model=self.model)


def _usage(usage: Any) -> Usage:
    if not isinstance(usage, dict):
        return Usage()
    tokens = usage.get("tokens") or usage.get("billed_units") or {}
    return Usage(tokens.get("input_tokens"), tokens.get("output_tokens"))


def _finish(value: Any) -> FinishReason | None:
    return _FINISH.get(str(value), "other") if value else None
