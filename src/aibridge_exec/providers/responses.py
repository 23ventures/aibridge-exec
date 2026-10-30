"""The Responses protocol (``POST /responses``), used natively by Meta's Model API.

Request: ``model``, ``input`` (role/content items; system included), ``temperature``,
``max_output_tokens``, flat function ``tools``. Streaming events: ``response.output_text.delta``,
``response.function_call_arguments.delta``, then ``response.completed`` with usage.
"""
from __future__ import annotations

import json
from typing import Any

from .base import HttpCall, ProviderAdapter, StreamParser, stream_error
from .types import ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent, Usage

_STATUS_FINISH: dict[str, FinishReason] = {"completed": "stop", "incomplete": "length", "failed": "error"}


class ResponsesAdapter(ProviderAdapter):
    def __init__(self, info: ProviderInfo) -> None:
        self.info = info

    def _headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def chat_call(self, request: ChatRequest, api_key: str, *, base_url: str | None, stream: bool) -> HttpCall:
        system, turns = request.split_system()
        items = ([{"role": "system", "content": system}] if system else []) + [
            {"role": m.role, "content": m.content} for m in turns
        ]
        body: dict[str, Any] = {"model": request.model, "input": items, "stream": stream}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["max_output_tokens"] = request.max_tokens
        fmt = request.response_format
        if fmt is not None:
            body["tools"] = [{"type": "function", "name": fmt.name, "description": fmt.description,
                              "parameters": fmt.json_schema()}]
            body["tool_choice"] = {"type": "function", "name": fmt.name}
        if request.reasoning_effort and request.reasoning_effort not in ("lowest", "none"):
            body["reasoning"] = {"effort": request.reasoning_effort}
        return HttpCall("POST", f"{self.base_url(base_url)}/responses", self._headers(api_key), json=body)

    def parse_chat(self, payload: dict[str, Any], request: ChatRequest) -> ChatResponse:
        if isinstance(payload.get("error"), dict):
            raise stream_error(self.id, payload, None)
        return ChatResponse(text=_output_text(payload), model=payload.get("model") or request.model,
                            finish_reason=_finish(payload), usage=_usage(payload.get("usage")))

    def stream_parser(self, request: ChatRequest) -> StreamParser:
        return _ResponsesStreamParser(self.id, request.model)

    def models_call(self, api_key: str, *, base_url: str | None) -> HttpCall:
        return HttpCall("GET", f"{self.base_url(base_url)}/models", self._headers(api_key))

    def parse_models(self, payload: Any) -> list[ModelInfo]:
        items = payload.get("data", []) if isinstance(payload, dict) else []
        return sorted((ModelInfo(str(i["id"])) for i in items if isinstance(i, dict) and i.get("id")),
                      key=lambda m: m.id)


def _output_text(response: dict[str, Any]) -> str:
    text, arguments = [], None
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    text.append(part.get("text") or "")
        elif item.get("type") == "function_call" and arguments is None:
            arguments = item.get("arguments")
    if text and "".join(text):
        return "".join(text)
    if arguments:
        return arguments if isinstance(arguments, str) else json.dumps(arguments)
    return response.get("output_text") or ""


def _finish(response: dict[str, Any]) -> FinishReason | None:
    if any(isinstance(i, dict) and i.get("type") == "function_call" for i in response.get("output") or []):
        return "tool_calls"
    status = response.get("status")
    if status == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        return "content_filter" if reason == "content_filter" else "length"
    return _STATUS_FINISH.get(str(status)) if status else None


def _usage(usage: Any) -> Usage:
    if not isinstance(usage, dict):
        return Usage()
    details = usage.get("input_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return Usage(usage.get("input_tokens"), usage.get("output_tokens"),
                 cache_read_tokens=cached if isinstance(cached, int) and cached > 0 else None)


class _ResponsesStreamParser(StreamParser):
    def __init__(self, provider_id: str, model: str) -> None:
        self.provider_id = provider_id
        self.model = model
        self.finish_reason: FinishReason | None = None
        self.usage = Usage()
        self.saw_text = False

    def feed(self, event: str | None, data: str) -> list[StreamEvent]:
        try:
            payload = json.loads(data)
        except ValueError:
            return []
        if not isinstance(payload, dict):
            return []
        kind = payload.get("type") or event
        if kind in ("error", "response.failed"):
            body = payload.get("response") if kind == "response.failed" else payload
            raise stream_error(self.provider_id, body if isinstance(body, dict) else payload, None)
        if kind == "response.output_text.delta" and payload.get("delta"):
            self.saw_text = True
            return [StreamEvent("delta", text=payload["delta"])]
        if kind == "response.function_call_arguments.delta" and payload.get("delta") and not self.saw_text:
            return [StreamEvent("delta", text=payload["delta"])]
        if kind in ("response.completed", "response.incomplete"):
            response = payload.get("response") or {}
            self.model = response.get("model") or self.model
            self.usage = _usage(response.get("usage") or payload.get("usage"))
            self.finish_reason = _finish(response) or ("length" if kind == "response.incomplete" else "stop")
        return []

    def finish(self) -> StreamEvent:
        return StreamEvent("done", finish_reason=self.finish_reason, usage=self.usage, model=self.model)


def meta() -> ResponsesAdapter:
    return ResponsesAdapter(ProviderInfo(
        id="meta", label="Meta (Muse)", default_base_url="https://api.meta.ai/v1",
        key_url="https://dev.meta.ai", key_hint="",
    ))
