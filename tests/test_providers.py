import asyncio
import json

import httpx
import pytest

from aibridge_exec.providers import AsyncChatClient, ChatClient, ProviderError, get_provider, list_providers
from aibridge_exec.providers.sse import SSEDecoder

KEY = "test-key-1234567890"
MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]


def sse(*events):
    out = []
    for event in events:
        if isinstance(event, tuple):
            name, data = event
            out.append(f"event: {name}\ndata: {json.dumps(data)}\n\n")
        elif isinstance(event, str):
            out.append(f"data: {event}\n\n")
        else:
            out.append(f"data: {json.dumps(event)}\n\n")
    return "".join(out)


class Recorder:
    """httpx MockTransport that records requests and replays canned responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        status, body, content_type = self.responses.pop(0)
        if not isinstance(body, str):
            body = json.dumps(body)
        return httpx.Response(status, text=body, headers={"content-type": content_type})

    def client(self, **kwargs):
        return ChatClient(http=httpx.Client(transport=httpx.MockTransport(self)), **kwargs)

    @property
    def body(self):
        return json.loads(self.requests[-1].content)


def ok_json(body):
    return 200, body, "application/json"


def ok_sse(text):
    return 200, text, "text/event-stream"


# ---------------------------------------------------------------- SSE decoder
def test_sse_decoder_handles_split_chunks_crlf_comments_and_multiline():
    decoder = SSEDecoder()
    events = []
    for chunk in [": keep-alive\r\n\r\nevent: a\r", "\ndata: {\"x\"", ":1}\r\n\r\ndata: l1\ndata: l2\n", "\n", "data: tail"]:
        events += decoder.feed(chunk)
    events += decoder.flush()
    assert events == [("a", '{"x":1}'), (None, "l1\nl2"), (None, "tail")]


def test_registry_lists_builtin_providers():
    ids = {p.id for p in list_providers()}
    assert {"openai", "anthropic", "gemini", "xai", "openrouter", "groq", "deepseek", "mistral", "together",
            "openai_compatible"} <= ids
    with pytest.raises(ProviderError) as err:
        get_provider("nope")
    assert err.value.code == "unknown_provider"


# ---------------------------------------------------------------- OpenAI-compatible
def test_openai_stream_request_shape_and_events():
    rec = Recorder(ok_sse(sse(
        {"model": "gpt-x", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"index": 0, "delta": {"content": "Hel"}}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}},
        "[DONE]",
    )))
    events = list(rec.client().stream("openai", api_key=KEY, model="gpt-x", messages=MESSAGES, max_tokens=50,
                                      temperature=0.2))
    request = rec.requests[0]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert rec.body == {
        "model": "gpt-x", "stream": True, "temperature": 0.2, "max_completion_tokens": 50,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
    }
    assert "".join(e.text for e in events if e.type == "delta") == "Hello"
    done = events[-1]
    assert done.type == "done" and done.finish_reason == "stop" and done.model == "gpt-x"
    assert (done.usage.input_tokens, done.usage.output_tokens) == (7, 2)


def test_compatible_providers_use_their_base_url_and_max_tokens():
    for provider, url in [("xai", "https://api.x.ai/v1"), ("groq", "https://api.groq.com/openai/v1"),
                          ("openrouter", "https://openrouter.ai/api/v1"), ("mistral", "https://api.mistral.ai/v1"),
                          ("deepseek", "https://api.deepseek.com"), ("together", "https://api.together.xyz/v1")]:
        rec = Recorder(ok_json({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}))
        reply = rec.client().chat(provider, api_key=KEY, model="m", messages=MESSAGES, max_tokens=5)
        assert str(rec.requests[0].url) == f"{url}/chat/completions", provider
        assert rec.body["max_tokens"] == 5 and "max_completion_tokens" not in rec.body
        assert reply.text == "ok"


def test_groq_usage_in_x_groq():
    rec = Recorder(ok_sse(sse({"choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": "stop"}],
                               "x_groq": {"usage": {"prompt_tokens": 3, "completion_tokens": 1}}}, "[DONE]")))
    done = list(rec.client().stream("groq", api_key=KEY, model="m", messages=MESSAGES))[-1]
    assert (done.usage.input_tokens, done.usage.output_tokens) == (3, 1)


def test_openai_non_stream_and_models_filter():
    rec = Recorder(
        ok_json({"model": "gpt-x", "choices": [{"message": {"content": "Hi!"}, "finish_reason": "length"}],
                 "usage": {"prompt_tokens": 4, "completion_tokens": 9}}),
        ok_json({"data": [{"id": "gpt-x"}, {"id": "text-embedding-3-small"}, {"id": "whisper-1"},
                          {"id": "gpt-a"}]}),
    )
    client = rec.client()
    reply = client.chat("openai", api_key=KEY, model="gpt-x", messages=MESSAGES)
    assert (reply.text, reply.finish_reason, reply.usage.output_tokens) == ("Hi!", "length", 9)
    assert "stream_options" not in rec.body and rec.body["stream"] is False
    assert [m.id for m in client.list_models("openai", api_key=KEY)] == ["gpt-a", "gpt-x"]


def test_together_and_mistral_model_lists():
    rec = Recorder(
        ok_json([{"id": "chat-1", "type": "chat", "display_name": "Chat 1", "context_length": 8192},
                 {"id": "img-1", "type": "image"}]),
        ok_json({"data": [{"id": "mistral-large", "capabilities": {"completion_chat": True}},
                          {"id": "mistral-embed", "capabilities": {"completion_chat": False}}]}),
    )
    client = rec.client()
    together = client.list_models("together", api_key=KEY)
    assert [(m.id, m.name, m.context_length) for m in together] == [("chat-1", "Chat 1", 8192)]
    assert [m.id for m in client.list_models("mistral", api_key=KEY)] == ["mistral-large"]


def test_stream_error_chunk_raises():
    rec = Recorder(ok_sse(sse({"error": {"message": "boom", "type": "server_error"}})))
    with pytest.raises(ProviderError) as err:
        list(rec.client().stream("openrouter", api_key=KEY, model="m", messages=MESSAGES))
    assert err.value.code == "upstream" and "boom" in err.value.message


def test_custom_openai_compatible_requires_base_url():
    rec = Recorder(ok_json({"choices": [{"message": {"content": "local"}}]}))
    client = rec.client()
    with pytest.raises(ProviderError) as err:
        client.chat("openai_compatible", api_key=KEY, model="llama", messages=MESSAGES)
    assert err.value.code == "config"
    reply = client.chat("openai_compatible", api_key=KEY, model="llama", messages=MESSAGES,
                        base_url="http://localhost:11434/v1/")
    assert str(rec.requests[0].url) == "http://localhost:11434/v1/chat/completions"
    assert reply.text == "local"


def test_base_url_overrides_per_provider():
    rec = Recorder(ok_json({"choices": [{"message": {"content": "x"}}]}))
    rec.client(base_urls={"openai": "http://proxy.local/v1"}).chat("openai", api_key=KEY, model="m",
                                                                  messages=MESSAGES)
    assert str(rec.requests[0].url) == "http://proxy.local/v1/chat/completions"


# ---------------------------------------------------------------- Anthropic
def test_anthropic_stream_request_and_events():
    rec = Recorder(ok_sse(sse(
        ("message_start", {"type": "message_start", "message": {"model": "claude-x",
                                                                "usage": {"input_tokens": 12, "output_tokens": 1}}}),
        ("ping", {"type": "ping"}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi "}}),
        ("content_block_delta", {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "there"}}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    )))
    messages = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"}, {"role": "user", "content": "d"}]
    events = list(rec.client().stream("anthropic", api_key=KEY, model="claude-x", messages=messages,
                                      system="Sys", temperature=1.5))
    request = rec.requests[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == KEY and request.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in request.headers
    assert rec.body == {
        "model": "claude-x", "max_tokens": 4096, "system": "Sys", "temperature": 1.0, "stream": True,
        "messages": [{"role": "user", "content": "a\n\nb"}, {"role": "assistant", "content": "c"},
                     {"role": "user", "content": "d"}],
    }
    assert "".join(e.text for e in events if e.type == "delta") == "Hi there"
    done = events[-1]
    assert (done.finish_reason, done.usage.input_tokens, done.usage.output_tokens, done.model) == \
        ("stop", 12, 5, "claude-x")


def test_anthropic_error_event_and_non_stream():
    rec = Recorder(
        ok_sse(sse(("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))),
        ok_json({"model": "claude-x", "content": [{"type": "thinking", "thinking": "..."},
                                                  {"type": "text", "text": "Answer"}],
                 "stop_reason": "max_tokens", "usage": {"input_tokens": 3, "output_tokens": 4}}),
    )
    client = rec.client()
    with pytest.raises(ProviderError) as err:
        list(client.stream("anthropic", api_key=KEY, model="claude-x", messages=MESSAGES))
    assert err.value.code == "upstream" and err.value.message == "Overloaded"
    reply = client.chat("anthropic", api_key=KEY, model="claude-x", messages=MESSAGES)
    assert (reply.text, reply.finish_reason, reply.usage.output_tokens) == ("Answer", "length", 4)
    assert rec.body["system"] == "Be brief." and "stream" not in rec.body


def test_anthropic_models_pagination():
    rec = Recorder(
        ok_json({"data": [{"id": "claude-a", "display_name": "Claude A"}], "has_more": True, "last_id": "claude-a"}),
        ok_json({"data": [{"id": "claude-b", "display_name": "Claude B"}], "has_more": False}),
    )
    models = rec.client().list_models("anthropic", api_key=KEY)
    assert [(m.id, m.name) for m in models] == [("claude-a", "Claude A"), ("claude-b", "Claude B")]
    assert rec.requests[1].url.params["after_id"] == "claude-a"


# ---------------------------------------------------------------- Gemini
def test_gemini_stream_request_and_events():
    rec = Recorder(ok_sse(sse(
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "thinking", "thought": True}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "Bon"}]}}], "modelVersion": "gemini-x-001"},
        {"candidates": [{"content": {"parts": [{"text": "jour"}]}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 2, "thoughtsTokenCount": 10}},
    )))
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                {"role": "user", "content": "q2"}]
    events = list(rec.client().stream("gemini", api_key=KEY, model="gemini-x", messages=messages, system="S",
                                      max_tokens=100, temperature=0.5))
    request = rec.requests[0]
    assert request.url.path == "/v1beta/models/gemini-x:streamGenerateContent"
    assert request.url.params["alt"] == "sse"
    assert request.headers["x-goog-api-key"] == KEY and KEY not in str(request.url)
    assert rec.body == {
        "contents": [{"role": "user", "parts": [{"text": "q"}]}, {"role": "model", "parts": [{"text": "a"}]},
                     {"role": "user", "parts": [{"text": "q2"}]}],
        "systemInstruction": {"parts": [{"text": "S"}]},
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 100},
    }
    assert "".join(e.text for e in events if e.type == "delta") == "Bonjour"
    done = events[-1]
    assert (done.finish_reason, done.usage.input_tokens, done.usage.output_tokens, done.model) == \
        ("stop", 6, 12, "gemini-x-001")


def test_gemini_non_stream_blocked_and_models():
    rec = Recorder(
        ok_json({"promptFeedback": {"blockReason": "SAFETY"}}),
        ok_json({"models": [{"name": "models/gemini-a", "displayName": "A", "inputTokenLimit": 1000,
                             "supportedGenerationMethods": ["generateContent"]},
                            {"name": "models/embedding-1", "supportedGenerationMethods": ["embedContent"]}],
                 "nextPageToken": "p2"}),
        ok_json({"models": [{"name": "models/gemini-b", "supportedGenerationMethods": ["generateContent"]}]}),
    )
    client = rec.client()
    reply = client.chat("gemini", api_key=KEY, model="models/gemini-a", messages=MESSAGES)
    assert rec.requests[0].url.path == "/v1beta/models/gemini-a:generateContent"
    assert (reply.text, reply.finish_reason) == ("", "content_filter")
    models = client.list_models("gemini", api_key=KEY)
    assert [(m.id, m.context_length) for m in models] == [("gemini-a", 1000), ("gemini-b", None)]
    assert rec.requests[2].url.params["pageToken"] == "p2"


# ---------------------------------------------------------------- errors
@pytest.mark.parametrize("status,code", [(401, "auth"), (403, "auth"), (429, "rate_limited"), (404, "not_found"),
                                         (400, "bad_request"), (500, "upstream"), (529, "upstream")])
def test_http_errors_map_to_codes_and_redact_key(status, code):
    rec = Recorder((status, {"error": {"message": f"Incorrect API key provided: {KEY}"}}, "application/json"))
    with pytest.raises(ProviderError) as err:
        rec.client().chat("openai", api_key=KEY, model="m", messages=MESSAGES)
    assert err.value.code == code and err.value.status == status
    assert KEY not in err.value.message and "***" in err.value.message


def test_stream_http_error_before_body():
    rec = Recorder((401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}},
                    "application/json"))
    with pytest.raises(ProviderError) as err:
        list(rec.client().stream("anthropic", api_key=KEY, model="m", messages=MESSAGES))
    assert (err.value.code, err.value.message) == ("auth", "invalid x-api-key")


def test_network_and_timeout_errors():
    def fail(exc):
        def handler(request):
            raise exc
        return ChatClient(http=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(ProviderError) as err:
        fail(httpx.ConnectError("refused")).chat("openai", api_key=KEY, model="m", messages=MESSAGES)
    assert err.value.code == "network"
    with pytest.raises(ProviderError) as err:
        list(fail(httpx.ReadTimeout("slow")).stream("openai", api_key=KEY, model="m", messages=MESSAGES))
    assert err.value.code == "timeout"


def test_validation_errors():
    client = Recorder().client()
    with pytest.raises(ProviderError) as err:
        client.chat("openai", api_key="", model="m", messages=MESSAGES)
    assert err.value.code == "auth"
    for kwargs in [{"messages": []}, {"messages": [{"role": "tool", "content": "x"}]},
                   {"messages": MESSAGES, "temperature": 3}, {"messages": [{"role": "user"}]}]:
        with pytest.raises(ProviderError) as err:
            client.chat("openai", api_key=KEY, model="m", **kwargs)
        assert err.value.code == "bad_request"


# ---------------------------------------------------------------- async client
def test_async_client_stream_and_models():
    rec = Recorder(
        ok_sse(sse({"choices": [{"index": 0, "delta": {"content": "A"}}]},
                   {"choices": [{"index": 0, "delta": {"content": "B"}, "finish_reason": "stop"}]}, "[DONE]")),
        ok_json({"data": [{"id": "m1"}]}),
    )

    async def run():
        async with AsyncChatClient(http=httpx.AsyncClient(transport=httpx.MockTransport(rec))) as client:
            events = [e async for e in client.stream("xai", api_key=KEY, model="grok", messages=MESSAGES)]
            models = await client.list_models("xai", api_key=KEY)
        return events, models

    events, models = asyncio.run(run())
    assert "".join(e.text for e in events if e.type == "delta") == "AB"
    assert events[-1].finish_reason == "stop" and [m.id for m in models] == ["m1"]
