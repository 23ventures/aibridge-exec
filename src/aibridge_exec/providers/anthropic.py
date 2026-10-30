"""Anthropic Messages API."""
from __future__ import annotations

import json
import re
from typing import Any

from .base import HttpCall, ProviderAdapter, StreamParser, stream_error
from .types import (ChatMessage, ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent,
                    Usage, merge_consecutive)

API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
#: Anthropic accepts at most four cache breakpoints per request.
MAX_CACHE_BREAKPOINTS = 4
#: Sent as the first turn when the conversation would otherwise start with the assistant.
LEADING_USER_TURN = "Continue."

_FINISH: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}

_FAMILY_VERSION = re.compile(r"(opus|sonnet)-(\d{1,2})(?:-(\d{1,2}))?(?!\d)")


def skips_temperature(model: str) -> bool:
    """Opus 4.6+ and Sonnet 4.5+ reject a custom temperature."""
    match = _FAMILY_VERSION.search(model)
    if not match:
        return False
    family, major, minor = match.group(1), int(match.group(2)), match.group(3)
    # "-20250514"-style date suffixes are not minor versions.
    minor_number = int(minor) if minor is not None and len(minor) <= 2 else 0
    version = (major, minor_number)
    return version >= ((4, 6) if family == "opus" else (4, 5))


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
        systems, turns = request.system_parts()
        turns = merge_consecutive(turns)
        if turns and turns[0].role == "assistant":
            turns.insert(0, ChatMessage("user", LEADING_USER_TURN))
        budget = [MAX_CACHE_BREAKPOINTS]

        def block(message: ChatMessage) -> dict[str, Any]:
            out: dict[str, Any] = {"type": "text", "text": message.content}
            if message.cache and budget[0] > 0:
                out["cache_control"] = {"type": "ephemeral"}
                budget[0] -= 1
            return out

        system_blocks = [block(m) for m in systems]
        messages = []
        for turn in turns:
            if turn.cache:
                messages.append({"role": turn.role, "content": [block(turn)]})
            else:
                messages.append({"role": turn.role, "content": turn.content})
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": messages,
        }
        if system_blocks:
            if any("cache_control" in b for b in system_blocks):
                body["system"] = system_blocks
            else:
                body["system"] = "\n\n".join(b["text"] for b in system_blocks)
        if request.temperature is not None and not skips_temperature(request.model):
            body["temperature"] = min(request.temperature, 1.0)  # Anthropic accepts 0..1
        fmt = request.response_format
        if fmt is not None:
            body["tools"] = [{"name": fmt.name, "description": fmt.description, "input_schema": fmt.json_schema()}]
            body["tool_choice"] = {"type": "tool", "name": fmt.name}
        if stream:
            body["stream"] = True
        return HttpCall("POST", f"{self.base_url(base_url)}/messages", self._headers(api_key), json=body)

    def parse_chat(self, payload: dict[str, Any], request: ChatRequest) -> ChatResponse:
        if payload.get("type") == "error":
            raise stream_error(self.id, payload, None)
        blocks = payload.get("content") or []
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        if not text:  # structured output: the forced tool's input is the reply
            tool = next((b for b in blocks if b.get("type") == "tool_use"), None)
            if tool is not None:
                text = json.dumps(tool.get("input") or {})
        return ChatResponse(
            text=text,
            model=payload.get("model") or request.model,
            finish_reason=_finish(payload.get("stop_reason")),
            usage=_usage(payload.get("usage") or {}),
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


def _usage(usage: dict[str, Any], previous: Usage | None = None) -> Usage:
    """Anthropic reports uncached input separately from cache reads and writes; ``input_tokens``
    here is the total, like every other provider's prompt count."""
    previous = previous or Usage()
    uncached = usage.get("input_tokens")
    write = usage.get("cache_creation_input_tokens")
    read = usage.get("cache_read_input_tokens")
    if uncached is None and write is None and read is None:
        input_tokens = previous.input_tokens
        write, read = previous.cache_write_tokens, previous.cache_read_tokens
    else:
        input_tokens = (uncached or 0) + (write or 0) + (read or 0)
    output = usage.get("output_tokens")
    return Usage(input_tokens, output if output is not None else previous.output_tokens,
                 cache_read_tokens=read, cache_write_tokens=write)


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
        if not isinstance(payload, dict):
            return []
        kind = payload.get("type") or event
        if kind == "error":
            raise stream_error(self.provider_id, payload, None)
        if kind == "message_start":
            message = payload.get("message") or {}
            self.model = message.get("model") or self.model
            self.usage = _usage(message.get("usage") or {})
        elif kind == "content_block_delta":
            delta = payload.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                return [StreamEvent("delta", text=delta["text"])]
            if delta.get("type") == "input_json_delta" and delta.get("partial_json"):
                return [StreamEvent("delta", text=delta["partial_json"])]
        elif kind == "message_delta":
            reason = (payload.get("delta") or {}).get("stop_reason")
            if reason:
                self.finish_reason = _finish(reason)
            self.usage = _usage(payload.get("usage") or {}, self.usage)
        return []

    def finish(self) -> StreamEvent:
        return StreamEvent("done", finish_reason=self.finish_reason, usage=self.usage, model=self.model)


def _finish(value: Any) -> FinishReason | None:
    return _FINISH.get(str(value), "other") if value else None
