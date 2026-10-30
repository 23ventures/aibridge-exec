"""Alibaba Cloud Model Studio (Qwen) through the native DashScope API.

``POST {base}/services/aigc/text-generation/generation`` with ``input.messages`` and
``parameters``; streaming is Server-Sent Events enabled by the ``X-DashScope-SSE`` header, with
``incremental_output`` so each event carries only new text. The default base is the US (Virginia)
endpoint; set a server or client base URL for another region, e.g.
``https://dashscope-intl.aliyuncs.com/api/v1`` (Singapore) or a workspace endpoint.
"""
from __future__ import annotations

import json
from typing import Any

from .base import HttpCall, ProviderAdapter, StreamParser, stream_error
from .types import ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent, Usage

_FINISH: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
}


class DashScopeAdapter(ProviderAdapter):
    info = ProviderInfo(
        id="alibaba",
        label="Alibaba Cloud (Qwen)",
        default_base_url="https://dashscope-us.aliyuncs.com/api/v1",
        key_url="https://modelstudio.console.alibabacloud.com/?tab=playground#/api-key",
        key_hint="sk-...",
    )

    def _headers(self, api_key: str, stream: bool = False) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if stream:
            headers["X-DashScope-SSE"] = "enable"
        return headers

    def chat_call(self, request: ChatRequest, api_key: str, *, base_url: str | None, stream: bool) -> HttpCall:
        system, turns = request.split_system()
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": m.role, "content": m.content} for m in turns
        ]
        parameters: dict[str, Any] = {"result_format": "message"}
        if request.temperature is not None:
            parameters["temperature"] = request.temperature
        if request.max_tokens is not None:
            parameters["max_tokens"] = request.max_tokens
        if stream:
            parameters["incremental_output"] = True
        if request.response_format is not None:
            parameters["response_format"] = {"type": "json_object"}
        if request.thinking is not None:
            parameters["enable_thinking"] = request.thinking
        body = {"model": request.model, "input": {"messages": messages}, "parameters": parameters}
        url = f"{self.base_url(base_url)}/services/aigc/text-generation/generation"
        return HttpCall("POST", url, self._headers(api_key, stream), json=body)

    def parse_chat(self, payload: dict[str, Any], request: ChatRequest) -> ChatResponse:
        if payload.get("code") and not payload.get("output"):
            raise stream_error(self.id, payload, None)
        text, finish = _choice(payload)
        return ChatResponse(text=text, model=request.model, finish_reason=finish, usage=_usage(payload))

    def stream_parser(self, request: ChatRequest) -> StreamParser:
        return _DashScopeStreamParser(self.id, request.model)

    def models_call(self, api_key: str, *, base_url: str | None) -> HttpCall:
        # The native API has no model list; the same host serves one under compatible-mode.
        base = self.base_url(base_url)
        root = base[: -len("/api/v1")] if base.endswith("/api/v1") else base
        return HttpCall("GET", f"{root}/compatible-mode/v1/models", self._headers(api_key))

    def parse_models(self, payload: Any) -> list[ModelInfo]:
        items = payload.get("data", []) if isinstance(payload, dict) else []
        return sorted((ModelInfo(str(i["id"])) for i in items if isinstance(i, dict) and i.get("id")),
                      key=lambda m: m.id)


class _DashScopeStreamParser(StreamParser):
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
        if event == "error" or (payload.get("code") and not payload.get("output")):
            raise stream_error(self.provider_id, payload, None)
        usage = _usage(payload)
        if usage.input_tokens is not None or usage.output_tokens is not None:
            self.usage = usage
        text, finish = _choice(payload)
        if finish:
            self.finish_reason = finish
        return [StreamEvent("delta", text=text)] if text else []

    def finish(self) -> StreamEvent:
        return StreamEvent("done", finish_reason=self.finish_reason, usage=self.usage, model=self.model)


def _choice(payload: dict[str, Any]) -> tuple[str, FinishReason | None]:
    output = payload.get("output") or {}
    choices = output.get("choices") or []
    if choices:
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        text = content or ""
        if not text:
            for call in message.get("tool_calls") or []:
                arguments = (call.get("function") or {}).get("arguments") if isinstance(call, dict) else None
                if arguments:
                    text = arguments
                    break
        reason = choice.get("finish_reason")
    else:  # result_format "text"
        text, reason = output.get("text") or "", output.get("finish_reason")
    finish = _FINISH.get(str(reason), "other") if reason and reason != "null" else None
    return text, finish


def _usage(payload: dict[str, Any]) -> Usage:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return Usage()
    return Usage(usage.get("input_tokens"), usage.get("output_tokens"))
