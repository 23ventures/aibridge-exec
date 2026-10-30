"""Google Gemini API (Generative Language API, API-key auth)."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from .base import HttpCall, ProviderAdapter, StreamParser, stream_error
from .types import (ChatMessage, ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent, Usage,
                    merge_consecutive)

_FINISH: dict[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "MALFORMED_FUNCTION_CALL": "error",
}


class GeminiAdapter(ProviderAdapter):
    info = ProviderInfo(
        id="gemini",
        label="Google Gemini",
        default_base_url="https://generativelanguage.googleapis.com/v1beta",
        key_url="https://aistudio.google.com/apikey",
        key_hint="AIza...",
    )

    def _headers(self, api_key: str) -> dict[str, str]:
        return {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    def chat_call(self, request: ChatRequest, api_key: str, *, base_url: str | None, stream: bool) -> HttpCall:
        system, turns = request.split_system()
        body: dict[str, Any] = {
            "contents": [
                {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
                for m in _leading_user(merge_consecutive(turns))
            ]
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        config: dict[str, Any] = {}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_tokens is not None:
            config["maxOutputTokens"] = request.max_tokens
        if request.response_format is not None:
            config["responseMimeType"] = "application/json"
            if request.response_format.schema:
                config["responseSchema"] = to_gemini_schema(request.response_format.json_schema())
        if config:
            body["generationConfig"] = config
        model = quote(request.model.removeprefix("models/"), safe="-._")
        method = "streamGenerateContent" if stream else "generateContent"
        params = {"alt": "sse"} if stream else {}
        return HttpCall("POST", f"{self.base_url(base_url)}/models/{model}:{method}", self._headers(api_key),
                        params=params, json=body)

    def parse_chat(self, payload: dict[str, Any], request: ChatRequest) -> ChatResponse:
        if isinstance(payload.get("error"), dict):
            raise stream_error(self.id, payload, None)
        text, finish = _candidate(payload)
        return ChatResponse(text=text, model=payload.get("modelVersion") or request.model,
                            finish_reason=finish, usage=_usage(payload) or Usage())

    def stream_parser(self, request: ChatRequest) -> StreamParser:
        return _GeminiStreamParser(self.id, request.model)

    def models_call(self, api_key: str, *, base_url: str | None) -> HttpCall:
        return HttpCall("GET", f"{self.base_url(base_url)}/models", self._headers(api_key), params={"pageSize": "1000"})

    def parse_models(self, payload: Any) -> list[ModelInfo]:
        models = []
        for item in (payload or {}).get("models", []):
            if "generateContent" not in (item.get("supportedGenerationMethods") or []):
                continue
            model_id = str(item.get("name", "")).removeprefix("models/")
            if model_id:
                models.append(ModelInfo(model_id, item.get("displayName"), item.get("inputTokenLimit")))
        return sorted(models, key=lambda m: m.id)

    def next_models_page(self, payload: Any, call: HttpCall) -> HttpCall | None:
        token = payload.get("nextPageToken") if isinstance(payload, dict) else None
        return HttpCall(call.method, call.url, call.headers, {**call.params, "pageToken": token}) if token else None


class _GeminiStreamParser(StreamParser):
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
        if isinstance(payload.get("error"), dict):
            raise stream_error(self.provider_id, payload, None)
        self.model = payload.get("modelVersion") or self.model
        usage = _usage(payload)
        if usage:
            self.usage = usage
        text, finish = _candidate(payload)
        if finish:
            self.finish_reason = finish
        return [StreamEvent("delta", text=text)] if text else []

    def finish(self) -> StreamEvent:
        return StreamEvent("done", finish_reason=self.finish_reason, usage=self.usage, model=self.model)


def _candidate(payload: dict[str, Any]) -> tuple[str, FinishReason | None]:
    candidates = payload.get("candidates") or []
    if not candidates:
        blocked = (payload.get("promptFeedback") or {}).get("blockReason")
        return "", ("content_filter" if blocked else None)
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    reason = candidate.get("finishReason")
    finish = _FINISH.get(reason, "other") if reason and reason != "FINISH_REASON_UNSPECIFIED" else None
    return text, finish


def _usage(payload: dict[str, Any]) -> Usage | None:
    meta = payload.get("usageMetadata")
    if not isinstance(meta, dict):
        return None
    output = meta.get("candidatesTokenCount")
    if output is not None and meta.get("thoughtsTokenCount"):
        output += meta["thoughtsTokenCount"]
    return Usage(meta.get("promptTokenCount"), output)


def _leading_user(turns: list[ChatMessage]) -> list[ChatMessage]:
    """Gemini conversations must open with a user turn."""
    if turns and turns[0].role == "assistant":
        return [ChatMessage("user", "Continue."), *turns]
    return turns


_GEMINI_TYPES = {"object": "OBJECT", "array": "ARRAY", "string": "STRING", "boolean": "BOOLEAN",
                 "number": "NUMBER", "integer": "INTEGER"}


def to_gemini_schema(schema: Any) -> dict[str, Any]:
    """JSON Schema to Gemini's ``responseSchema`` (an OpenAPI subset with upper-case types).

    Keeps type, description, enum, required, nullable, properties and items; drops the rest.
    """
    if not isinstance(schema, dict):
        return {"type": "STRING"}
    kind = schema.get("type", "object")
    nullable = False
    if isinstance(kind, list):  # ["string", "null"]
        nullable = "null" in kind
        kind = next((k for k in kind if k != "null"), "string")
    out: dict[str, Any] = {"type": _GEMINI_TYPES.get(str(kind).lower(), "STRING")}
    if nullable or schema.get("nullable"):
        out["nullable"] = True
    if isinstance(schema.get("description"), str):
        out["description"] = schema["description"]
    if isinstance(schema.get("enum"), list):
        out["enum"] = [str(v) for v in schema["enum"]]
    if out["type"] == "OBJECT":
        properties = schema.get("properties") or {}
        out["properties"] = {key: to_gemini_schema(value) for key, value in properties.items()}
        required = [key for key in schema.get("required") or [] if key in properties]
        if required:
            out["required"] = required
    elif out["type"] == "ARRAY":
        out["items"] = to_gemini_schema(schema.get("items") or {"type": "string"})
    return out
