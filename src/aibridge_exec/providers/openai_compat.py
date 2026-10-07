"""OpenAI Chat Completions and the providers that speak the same protocol."""
from __future__ import annotations

import json
from typing import Any

from .base import HttpCall, ProviderAdapter, StreamParser, stream_error
from .errors import ProviderError
from .types import ChatRequest, ChatResponse, FinishReason, ModelInfo, ProviderInfo, StreamEvent, Usage

_FINISH: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "content_filter": "content_filter",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "eos": "stop",
}

# Non-chat model families in OpenAI's /models list.
_OPENAI_NON_CHAT = ("embedding", "whisper", "tts", "dall-e", "moderation", "davinci", "babbage",
                    "transcribe", "realtime", "audio", "image", "search", "computer-use", "sora")


class OpenAICompatibleAdapter(ProviderAdapter):
    def __init__(
        self,
        info: ProviderInfo,
        *,
        max_tokens_field: str = "max_tokens",
        stream_usage_option: bool = False,
        extra_headers: dict[str, str] | None = None,
        non_chat_markers: tuple[str, ...] = (),
    ) -> None:
        self.info = info
        self.max_tokens_field = max_tokens_field
        self.stream_usage_option = stream_usage_option
        self.extra_headers = dict(extra_headers or {})
        self.non_chat_markers = non_chat_markers

    def _headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", **self.extra_headers}

    def chat_call(self, request: ChatRequest, api_key: str, *, base_url: str | None, stream: bool) -> HttpCall:
        system, turns = request.split_system()
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": m.role, "content": m.content} for m in turns
        ]
        body: dict[str, Any] = {"model": request.model, "messages": messages, "stream": stream}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body[self.max_tokens_field] = request.max_tokens
        if stream and self.stream_usage_option:
            body["stream_options"] = {"include_usage": True}
        return HttpCall("POST", f"{self.base_url(base_url)}/chat/completions", self._headers(api_key), json=body)

    def parse_chat(self, payload: dict[str, Any], request: ChatRequest) -> ChatResponse:
        if isinstance(payload.get("error"), dict):
            raise stream_error(self.id, payload, None)
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return ChatResponse(
            text=_text(message.get("content")),
            model=payload.get("model") or request.model,
            finish_reason=_finish(choice.get("finish_reason")),
            usage=_usage(payload),
        )

    def stream_parser(self, request: ChatRequest) -> StreamParser:
        return _OpenAIStreamParser(self.id, request.model)

    def models_call(self, api_key: str, *, base_url: str | None) -> HttpCall:
        return HttpCall("GET", f"{self.base_url(base_url)}/models", self._headers(api_key))

    def parse_models(self, payload: Any) -> list[ModelInfo]:
        items = payload.get("data", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
        models = []
        for item in items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            model_id = str(item["id"])
            if self.non_chat_markers and any(marker in model_id.lower() for marker in self.non_chat_markers):
                continue
            if item.get("type") and item["type"] not in ("chat", "model", "language", "code"):
                continue  # Together: image, embedding, rerank, ...
            capabilities = item.get("capabilities")
            if isinstance(capabilities, dict) and capabilities.get("completion_chat") is False:
                continue  # Mistral
            context = item.get("context_length") or item.get("context_window") or item.get("max_context_length")
            models.append(ModelInfo(model_id, item.get("name") or item.get("display_name"),
                                    int(context) if isinstance(context, (int, float)) else None))
        return sorted(models, key=lambda m: m.id)


class _OpenAIStreamParser(StreamParser):
    def __init__(self, provider_id: str, model: str) -> None:
        self.provider_id = provider_id
        self.model = model
        self.finish_reason: FinishReason | None = None
        self.usage = Usage()

    def feed(self, event: str | None, data: str) -> list[StreamEvent]:
        if data.strip() == "[DONE]":
            return []
        try:
            payload = json.loads(data)
        except ValueError:
            return []
        if isinstance(payload.get("error"), dict):
            raise stream_error(self.provider_id, payload, None)
        self.model = payload.get("model") or self.model
        usage = _usage(payload)
        if usage.input_tokens is not None or usage.output_tokens is not None:
            self.usage = usage
        events = []
        for choice in payload.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            text = _text((choice.get("delta") or {}).get("content"))
            if text:
                events.append(StreamEvent("delta", text=text))
            if choice.get("finish_reason"):
                self.finish_reason = _finish(choice["finish_reason"])
        return events

    def finish(self) -> StreamEvent:
        return StreamEvent("done", finish_reason=self.finish_reason, usage=self.usage, model=self.model)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _finish(value: Any) -> FinishReason | None:
    if not value:
        return None
    return _FINISH.get(str(value), "other")


def _usage(payload: dict[str, Any]) -> Usage:
    usage = payload.get("usage") or (payload.get("x_groq") or {}).get("usage") or {}
    if not isinstance(usage, dict):
        return Usage()
    return Usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))


def _info(id_: str, label: str, base_url: str, key_url: str, hint: str = "") -> ProviderInfo:
    return ProviderInfo(id=id_, label=label, default_base_url=base_url, key_url=key_url, key_hint=hint)


def builtin_openai_compatible() -> list[OpenAICompatibleAdapter]:
    return [
        OpenAICompatibleAdapter(
            _info("openai", "OpenAI", "https://api.openai.com/v1", "https://platform.openai.com/api-keys", "sk-..."),
            max_tokens_field="max_completion_tokens", stream_usage_option=True, non_chat_markers=_OPENAI_NON_CHAT,
        ),
        OpenAICompatibleAdapter(
            _info("xai", "xAI (Grok)", "https://api.x.ai/v1", "https://console.x.ai", "xai-..."),
            stream_usage_option=True, non_chat_markers=("image", "imagine"),
        ),
        OpenAICompatibleAdapter(
            _info("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "https://openrouter.ai/keys", "sk-or-..."),
            stream_usage_option=True,
        ),
        OpenAICompatibleAdapter(
            _info("groq", "Groq", "https://api.groq.com/openai/v1", "https://console.groq.com/keys", "gsk_..."),
            non_chat_markers=("whisper", "tts", "guard"),
        ),
        OpenAICompatibleAdapter(
            _info("deepseek", "DeepSeek", "https://api.deepseek.com", "https://platform.deepseek.com/api_keys", "sk-..."),
            stream_usage_option=True,
        ),
        OpenAICompatibleAdapter(
            _info("mistral", "Mistral", "https://api.mistral.ai/v1", "https://console.mistral.ai/api-keys"),
            non_chat_markers=("embed", "moderation", "ocr"),
        ),
        OpenAICompatibleAdapter(
            _info("together", "Together AI", "https://api.together.xyz/v1", "https://api.together.ai/settings/api-keys"),
        ),
    ]


def custom_openai_compatible() -> OpenAICompatibleAdapter:
    """Any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, gateways). Needs ``base_url``."""
    return _CustomAdapter(_info("openai_compatible", "OpenAI-compatible (custom URL)", "", ""))


class _CustomAdapter(OpenAICompatibleAdapter):
    def base_url(self, override: str | None) -> str:
        if not override:
            raise ProviderError("config", "base_url is required for openai_compatible", provider=self.id)
        return override.rstrip("/")
