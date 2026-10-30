"""OpenAI Chat Completions, and the providers whose own API uses the same protocol.

Each provider is a first-class adapter with its own id, URL and model rules (Kimi, NVIDIA, Z.ai,
MiniMax, Arcee and AI21 speak this protocol natively); ``openai_compatible`` is only for
arbitrary endpoints the caller points at.
"""
from __future__ import annotations

import json
import re
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

# How each adapter asks for JSON when ``ChatRequest.response_format`` is set:
#   "json_schema"  response_format={"type": "json_schema", ...}
#   "json_object"  response_format={"type": "json_object"} (schema stays in the prompt)
#   "tool"         one function tool whose arguments are the JSON reply
#   None           nothing extra: the prompt carries the instructions
STRUCTURED_MODES = ("json_schema", "json_object", "tool", None)


class OpenAICompatibleAdapter(ProviderAdapter):
    def __init__(
        self,
        info: ProviderInfo,
        *,
        max_tokens_field: str = "max_tokens",
        stream_usage_option: bool = False,
        extra_headers: dict[str, str] | None = None,
        non_chat_markers: tuple[str, ...] = (),
        structured: str | None = "json_object",
        force_tool: bool = True,
        strip_think: bool = False,
    ) -> None:
        if structured not in STRUCTURED_MODES:
            raise ValueError(f"structured must be one of {STRUCTURED_MODES}")
        self.info = info
        self.max_tokens_field = max_tokens_field
        self.stream_usage_option = stream_usage_option
        self.extra_headers = dict(extra_headers or {})
        self.non_chat_markers = non_chat_markers
        self.structured = structured
        self.force_tool = force_tool
        self.strip_think = strip_think

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    # -- hooks for model rules --------------------------------------------------------------
    def tune(self, body: dict[str, Any], request: ChatRequest, stream: bool) -> None:
        """Adjust the request body for model-specific rules (override in provider adapters)."""

    def _structured(self, body: dict[str, Any], request: ChatRequest) -> None:
        fmt = request.response_format
        if fmt is None or self.structured is None:
            return
        if self.structured == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif self.structured == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": fmt.name, "description": fmt.description, "schema": fmt.json_schema(), "strict": False}}
        else:
            body["tools"] = [{"type": "function", "function": {
                "name": fmt.name, "description": fmt.description, "parameters": fmt.json_schema()}}]
            body["tool_choice"] = ({"type": "function", "function": {"name": fmt.name}} if self.force_tool
                                   else "auto")

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
        self._structured(body, request)
        self.tune(body, request, stream)
        return HttpCall("POST", f"{self.base_url(base_url)}/chat/completions", self._headers(api_key), json=body)

    def parse_chat(self, payload: dict[str, Any], request: ChatRequest) -> ChatResponse:
        if isinstance(payload.get("error"), dict):
            raise stream_error(self.id, payload, None)
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = _text(message.get("content"))
        if self.strip_think:
            text = strip_think(text)
        if not text:
            text = _tool_arguments(message)
        return ChatResponse(
            text=text,
            model=payload.get("model") or request.model,
            finish_reason=_finish(choice.get("finish_reason")),
            usage=_usage(payload),
        )

    def stream_parser(self, request: ChatRequest) -> StreamParser:
        return _OpenAIStreamParser(self.id, request.model, think_filter=self.strip_think)

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


# ------------------------------------------------------------------ model rules
# Reasoning models: the lowest effort each family accepts, for ``reasoning_effort="lowest"``.
_OPENAI_EFFORTS: dict[str, tuple[str, ...]] = {
    "gpt-5.6": ("none", "low", "medium", "high"),
    "gpt-5.4": ("none", "low", "medium", "high"),
    "gpt-5.2": ("none", "low", "medium", "high"),
    "gpt-5.1": ("none", "low", "medium", "high"),
    "gpt-5": ("minimal", "low", "medium", "high"),
    "o4-mini": ("low", "medium", "high"),
    "o3-mini": ("low", "medium", "high"),
    "o3": ("low", "medium", "high"),
    "o1-mini": (),
    "o1": ("low", "medium", "high"),
}
_LOWEST = ("none", "minimal", "low")


def _family(model: str, table: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    for prefix in sorted(table, key=len, reverse=True):
        if model.startswith(prefix):
            return table[prefix]
    return None


def is_openai_reasoning_model(model: str) -> bool:
    """gpt-5 and the o-series take no temperature and think before answering."""
    return model.startswith("gpt-5") or bool(re.match(r"o\d", model))


class OpenAIAdapter(OpenAICompatibleAdapter):
    def tune(self, body: dict[str, Any], request: ChatRequest, stream: bool) -> None:
        model = request.model
        if is_openai_reasoning_model(model):
            body.pop("temperature", None)  # only the default (1) is accepted
        effort = request.reasoning_effort
        allowed = _family(model, _OPENAI_EFFORTS)
        if effort and allowed:
            if effort == "lowest":
                effort = next((e for e in _LOWEST if e in allowed), None)
            if effort in allowed:
                body["reasoning_effort"] = effort


class KimiAdapter(OpenAICompatibleAdapter):
    """Moonshot: kimi-k* reasoning models take ``max_completion_tokens`` and no temperature."""

    def tune(self, body: dict[str, Any], request: ChatRequest, stream: bool) -> None:
        if request.model.startswith("kimi-k"):
            body.pop("temperature", None)
            if "max_tokens" in body:
                body["max_completion_tokens"] = body.pop("max_tokens")
        elif "temperature" in body:
            body["temperature"] = min(body["temperature"], 1.0)  # Moonshot accepts 0..1
        if request.thinking is not None:
            body["thinking"] = {"type": "enabled" if request.thinking else "disabled"}
        if request.reasoning_effort and request.reasoning_effort != "lowest":
            body["reasoning_effort"] = request.reasoning_effort


class ThinkingSwitchAdapter(OpenAICompatibleAdapter):
    """Providers with a ``thinking: {"type": ...}`` switch (Z.ai GLM, MiniMax)."""

    def __init__(self, info: ProviderInfo, *, on: str, off: str = "disabled", **kwargs: Any) -> None:
        super().__init__(info, **kwargs)
        self.on, self.off = on, off

    def tune(self, body: dict[str, Any], request: ChatRequest, stream: bool) -> None:
        if request.thinking is not None:
            body["thinking"] = {"type": self.on if request.thinking else self.off}
        elif request.response_format is not None:
            body["thinking"] = {"type": self.off}  # tool calls are not allowed while thinking


class NvidiaAdapter(OpenAICompatibleAdapter):
    """NVIDIA NIM. Nemotron models switch thinking per request; ``<think>`` text is removed."""

    _PROMPT_SWITCHED = ("nvidia-nemotron-nano-9b-v2", "llama-3.3-nemotron-super-49b-v1.5")

    def tune(self, body: dict[str, Any], request: ChatRequest, stream: bool) -> None:
        thinking = request.thinking
        if thinking is None and request.response_format is not None:
            thinking = False
        if thinking is None:
            return
        model = request.model
        if "nemotron-3-" in model:
            body["chat_template_kwargs"] = {"enable_thinking": thinking}
        elif any(name in model for name in self._PROMPT_SWITCHED):
            switch = "/think" if thinking else "/no_think"
            messages = body["messages"]
            if messages and messages[0]["role"] == "system":
                messages[0] = {"role": "system", "content": f"{switch}\n{messages[0]['content']}"}
            else:
                messages.insert(0, {"role": "system", "content": switch})


# ------------------------------------------------------------------ parsing helpers
class _ThinkFilter:
    """Hides ``<think>...</think>`` spans in streamed text, even when tags are split across chunks."""

    def __init__(self) -> None:
        self.buffer = ""
        self.inside = False

    def feed(self, text: str) -> str:
        self.buffer += text
        out = []
        while self.buffer:
            if self.inside:
                end = self.buffer.find("</think>")
                if end < 0:
                    self.buffer = self.buffer[-len("</think>"):]  # keep a possible partial tag
                    break
                self.buffer, self.inside = self.buffer[end + len("</think>"):], False
            else:
                start = self.buffer.find("<think>")
                if start < 0:
                    keep = _partial_tag_suffix(self.buffer, "<think>")
                    out.append(self.buffer[:len(self.buffer) - keep])
                    self.buffer = self.buffer[len(self.buffer) - keep:]
                    break
                out.append(self.buffer[:start])
                self.buffer, self.inside = self.buffer[start + len("<think>"):], True
        return "".join(out)

    def flush(self) -> str:
        tail, self.buffer = ("" if self.inside else self.buffer), ""
        return tail


def _partial_tag_suffix(text: str, tag: str) -> int:
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


def strip_think(text: str) -> str:
    """Text after the last ``</think>``; before an unclosed ``<think>``; else unchanged."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].lstrip()
    if "<think>" in text:
        return text.split("<think>", 1)[0].rstrip()
    return text


def _tool_arguments(message: dict[str, Any]) -> str:
    for call in message.get("tool_calls") or []:
        arguments = (call.get("function") or {}).get("arguments") if isinstance(call, dict) else None
        if arguments:
            return arguments if isinstance(arguments, str) else json.dumps(arguments)
    function_call = message.get("function_call") or {}
    arguments = function_call.get("arguments") if isinstance(function_call, dict) else None
    if arguments:
        return arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ""


class _OpenAIStreamParser(StreamParser):
    def __init__(self, provider_id: str, model: str, *, think_filter: bool = False) -> None:
        self.provider_id = provider_id
        self.model = model
        self.finish_reason: FinishReason | None = None
        self.usage = Usage()
        self.think = _ThinkFilter() if think_filter else None

    def feed(self, event: str | None, data: str) -> list[StreamEvent]:
        if data.strip() == "[DONE]":
            return []
        try:
            payload = json.loads(data)
        except ValueError:
            return []
        if not isinstance(payload, dict):
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
            delta = choice.get("delta") or {}
            text = _text(delta.get("content"))
            if self.think is not None and text:
                text = self.think.feed(text)
            if not text:  # structured output arrives as tool-call argument fragments
                for call in delta.get("tool_calls") or []:
                    fragment = (call.get("function") or {}).get("arguments") if isinstance(call, dict) else None
                    if isinstance(fragment, str):
                        text += fragment
            if text:
                events.append(StreamEvent("delta", text=text))
            if choice.get("finish_reason"):
                self.finish_reason = _finish(choice["finish_reason"])
        return events

    def finish(self) -> StreamEvent:
        return StreamEvent("done", finish_reason=self.finish_reason, usage=self.usage, model=self.model)

    def flush_events(self) -> list[StreamEvent]:
        tail = self.think.flush() if self.think is not None else ""
        return [StreamEvent("delta", text=tail)] if tail else []


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
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return Usage(usage.get("prompt_tokens"), usage.get("completion_tokens"),
                 cache_read_tokens=cached if isinstance(cached, int) and cached > 0 else None)


def _info(id_: str, label: str, base_url: str, key_url: str, hint: str = "") -> ProviderInfo:
    return ProviderInfo(id=id_, label=label, default_base_url=base_url, key_url=key_url, key_hint=hint)


def builtin_openai_compatible() -> list[OpenAICompatibleAdapter]:
    return [
        OpenAIAdapter(
            _info("openai", "OpenAI", "https://api.openai.com/v1", "https://platform.openai.com/api-keys", "sk-..."),
            max_tokens_field="max_completion_tokens", stream_usage_option=True, non_chat_markers=_OPENAI_NON_CHAT,
            structured="tool", force_tool=False,
        ),
        OpenAICompatibleAdapter(
            _info("xai", "xAI (Grok)", "https://api.x.ai/v1", "https://console.x.ai", "xai-..."),
            stream_usage_option=True, non_chat_markers=("image", "imagine"), structured="tool", force_tool=False,
        ),
        OpenAICompatibleAdapter(
            _info("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "https://openrouter.ai/keys", "sk-or-..."),
            stream_usage_option=True, structured=None,
        ),
        OpenAICompatibleAdapter(
            _info("groq", "Groq", "https://api.groq.com/openai/v1", "https://console.groq.com/keys", "gsk_..."),
            non_chat_markers=("whisper", "tts", "guard"),
        ),
        OpenAICompatibleAdapter(
            _info("deepseek", "DeepSeek", "https://api.deepseek.com", "https://platform.deepseek.com/api_keys", "sk-..."),
            stream_usage_option=True, structured="tool", force_tool=False,
        ),
        OpenAICompatibleAdapter(
            _info("mistral", "Mistral", "https://api.mistral.ai/v1", "https://console.mistral.ai/api-keys"),
            non_chat_markers=("embed", "moderation", "ocr"), structured="tool", force_tool=False,
        ),
        OpenAICompatibleAdapter(
            _info("together", "Together AI", "https://api.together.xyz/v1", "https://api.together.ai/settings/api-keys"),
        ),
        KimiAdapter(
            _info("kimi", "Kimi (Moonshot)", "https://api.moonshot.ai/v1", "https://platform.moonshot.ai/console/api-keys",
                  "sk-..."),
            stream_usage_option=True, structured="json_schema",
        ),
        NvidiaAdapter(
            _info("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1",
                  "https://build.nvidia.com/settings/api-keys", "nvapi-..."),
            stream_usage_option=True, structured="tool", strip_think=True,
        ),
        ThinkingSwitchAdapter(
            _info("zai", "Z.ai (GLM)", "https://api.z.ai/api/paas/v4", "https://z.ai/manage-apikey/apikey-list"),
            on="enabled", stream_usage_option=True, structured="tool",
        ),
        ThinkingSwitchAdapter(
            _info("minimax", "MiniMax", "https://api.minimax.io/v1",
                  "https://platform.minimax.io/user-center/basic-information/interface-key"),
            on="adaptive", stream_usage_option=True, structured="tool",
        ),
        OpenAICompatibleAdapter(
            _info("arcee", "Arcee AI", "https://api.arcee.ai/api/v1", "https://chat.arcee.ai"),
            stream_usage_option=True, structured="tool",
        ),
        OpenAICompatibleAdapter(
            _info("ai21", "AI21 (Jamba)", "https://api.ai21.com/studio/v1", "https://studio.ai21.com/account/api-key"),
            structured="json_object",
        ),
    ]


def custom_openai_compatible() -> OpenAICompatibleAdapter:
    """Any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, gateways). Needs ``base_url``."""
    return _CustomAdapter(_info("openai_compatible", "OpenAI-compatible (custom URL)", "", ""), structured=None)


class _CustomAdapter(OpenAICompatibleAdapter):
    needs_custom_base_url = True

    def base_url(self, override: str | None) -> str:
        if not override:
            raise ProviderError("config", "base_url is required for openai_compatible", provider=self.id)
        return override.rstrip("/")
