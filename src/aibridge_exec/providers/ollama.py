"""Ollama's native API (``/api/chat``, ``/api/tags``).

Local servers take no key; Ollama's cloud (``https://ollama.com``) takes a Bearer key, sent
only when one is given. Streams are newline-delimited JSON. Token counts are the server's own
(``prompt_eval_count`` / ``eval_count``). There is no public default server, so callers pass
``base_url`` (the server's root, e.g. ``http://gpu-box:11434``).
"""
from __future__ import annotations

import json
from typing import Any

from .base import HttpCall, ProviderAdapter, StreamParser, stream_error
from .types import ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent, Usage

_FINISH: dict[str, FinishReason] = {"stop": "stop", "length": "length"}


class OllamaAdapter(ProviderAdapter):
    info = ProviderInfo(
        id="ollama",
        label="Ollama",
        default_base_url="http://localhost:11434",
        key_url="https://ollama.com/settings/keys",
    )
    requires_key = False
    stream_format = "ndjson"
    needs_custom_base_url = True

    def base_url(self, override: str | None) -> str:
        url = (override or self.info.default_base_url).rstrip("/")
        for suffix in ("/api", "/v1"):  # accept the root, /api or an OpenAI-style /v1 URL
            if url.endswith(suffix):
                url = url[: -len(suffix)]
        return url

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def chat_call(self, request: ChatRequest, api_key: str, *, base_url: str | None, stream: bool) -> HttpCall:
        system, turns = request.split_system()
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": m.role, "content": m.content} for m in turns
        ]
        body: dict[str, Any] = {"model": request.model, "messages": messages, "stream": stream}
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            body["options"] = options
        if request.response_format is not None:
            body["format"] = request.response_format.json_schema() if request.response_format.schema else "json"
        if request.thinking is not None:
            body["think"] = request.thinking
        return HttpCall("POST", f"{self.base_url(base_url)}/api/chat", self._headers(api_key), json=body)

    def parse_chat(self, payload: dict[str, Any], request: ChatRequest) -> ChatResponse:
        if payload.get("error"):
            raise stream_error(self.id, {"message": str(payload["error"])}, None)
        message = payload.get("message") or {}
        return ChatResponse(text=message.get("content") or "", model=payload.get("model") or request.model,
                            finish_reason=_finish(payload), usage=_usage(payload))

    def stream_parser(self, request: ChatRequest) -> StreamParser:
        return _OllamaStreamParser(self.id, request.model)

    def models_call(self, api_key: str, *, base_url: str | None) -> HttpCall:
        return HttpCall("GET", f"{self.base_url(base_url)}/api/tags", self._headers(api_key))

    def parse_models(self, payload: Any) -> list[ModelInfo]:
        items = payload.get("models", []) if isinstance(payload, dict) else []
        return sorted((ModelInfo(str(i.get("model") or i["name"]), i.get("name"))
                       for i in items if isinstance(i, dict) and (i.get("model") or i.get("name"))),
                      key=lambda m: m.id)


def _finish(payload: dict[str, Any]) -> FinishReason | None:
    if not payload.get("done"):
        return None
    return _FINISH.get(str(payload.get("done_reason") or "stop"), "other")


def _usage(payload: dict[str, Any]) -> Usage:
    return Usage(payload.get("prompt_eval_count"), payload.get("eval_count"))


class _OllamaStreamParser(StreamParser):
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
        if payload.get("error"):
            raise stream_error(self.provider_id, {"message": payload["error"]}, None)
        self.model = payload.get("model") or self.model
        if payload.get("done"):
            self.finish_reason = _finish(payload)
            self.usage = _usage(payload)
        text = (payload.get("message") or {}).get("content") or ""
        return [StreamEvent("delta", text=text)] if text else []

    def finish(self) -> StreamEvent:
        return StreamEvent("done", finish_reason=self.finish_reason, usage=self.usage, model=self.model)
