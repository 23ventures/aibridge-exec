"""Anthropic Messages API."""
from __future__ import annotations

import json
from typing import Any

from .base import HttpCall, ProviderAdapter, StreamParser, stream_error
from .types import (ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent, Usage,
                    merge_consecutive)

API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096

_FINISH: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


class AnthropicAdapter(ProviderAdapter):
    info = ProviderInfo(
        id="anthropic",
        label="Anthropic",
        default_base_url="https://api.anthropic.com/v1",
        key_url="https://console.anthropic.com/settings/keys",
        key_hint="sk-ant-...",
    )

    def _headers(self, api_key: str) -> dict[str, str]:
        return {"x-api-key": api_key, "anthropic-version": API_VERSION, "content-type": "application/json"}

    def chat_call(self, request: ChatRequest, api_key: str, *, base_url: str | None, stream: bool) -> HttpCall:
        system, turns = request.split_system()
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": [{"role": m.role, "content": m.content} for m in merge_consecutive(turns)],
        }
        if system:
            body["system"] = system
        if request.temperature is not None:
            body["temperature"] = min(request.temperature, 1.0)  # Anthropic accepts 0..1
        if stream:
            body["stream"] = True
        return HttpCall("POST", f"{self.base_url(base_url)}/messages", self._headers(api_key), json=body)

    def parse_chat(self, payload: dict[str, Any], request: ChatRequest) -> ChatResponse:
        if payload.get("type") == "error":
            raise stream_error(self.id, payload, None)
        text = "".join(block.get("text", "") for block in payload.get("content") or [] if block.get("type") == "text")
        usage = payload.get("usage") or {}
        return ChatResponse(
            text=text,
            model=payload.get("model") or request.model,
            finish_reason=_finish(payload.get("stop_reason")),
            usage=Usage(usage.get("input_tokens"), usage.get("output_tokens")),
        )

    def stream_parser(self, request: ChatRequest) -> StreamParser:
        return _AnthropicStreamParser(self.id, request.model)

    def models_call(self, api_key: str, *, base_url: str | None) -> HttpCall:
        return HttpCall("GET", f"{self.base_url(base_url)}/models", self._headers(api_key), params={"limit": "1000"})

    def parse_models(self, payload: Any) -> list[ModelInfo]:
        return [ModelInfo(item["id"], item.get("display_name"))
                for item in (payload or {}).get("data", []) if isinstance(item, dict) and item.get("id")]

    def next_models_page(self, payload: Any, call: HttpCall) -> HttpCall | None:
        if isinstance(payload, dict) and payload.get("has_more") and payload.get("last_id"):
            return HttpCall(call.method, call.url, call.headers, {**call.params, "after_id": payload["last_id"]})
        return None


class _AnthropicStreamParser(StreamParser):
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
        kind = payload.get("type") or event
        if kind == "error":
            raise stream_error(self.provider_id, payload, None)
        if kind == "message_start":
            message = payload.get("message") or {}
            self.model = message.get("model") or self.model
            usage = message.get("usage") or {}
            self.usage = Usage(usage.get("input_tokens"), usage.get("output_tokens"))
        elif kind == "content_block_delta":
            delta = payload.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                return [StreamEvent("delta", text=delta["text"])]
        elif kind == "message_delta":
            reason = (payload.get("delta") or {}).get("stop_reason")
            if reason:
                self.finish_reason = _finish(reason)
            usage = payload.get("usage") or {}
            if usage.get("output_tokens") is not None:
                self.usage.output_tokens = usage["output_tokens"]
            if usage.get("input_tokens") is not None:
                self.usage.input_tokens = usage["input_tokens"]
        return []

    def finish(self) -> StreamEvent:
        return StreamEvent("done", finish_reason=self.finish_reason, usage=self.usage, model=self.model)


def _finish(value: Any) -> FinishReason | None:
    return _FINISH.get(str(value), "other") if value else None
