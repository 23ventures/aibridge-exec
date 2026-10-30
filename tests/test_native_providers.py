"""Native adapters added for the AIBridge platform's text providers, and the request options
(structured output, cache breakpoints, reasoning effort, thinking) they map natively."""
import asyncio
import json

import httpx
import pytest

from aibridge_exec.providers import AsyncChatClient, ProviderError, ResponseFormat
from aibridge_exec.providers.anthropic import skips_temperature
from aibridge_exec.providers.gemini import to_gemini_schema
from aibridge_exec.providers.openai_compat import _ThinkFilter, strip_think
from aibridge_exec.providers.sse import NDJSONDecoder

from .test_providers import KEY, MESSAGES, Recorder, ok_json, ok_sse, sse

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "tags": {"type": "array"}},
          "required": ["name"]}
FMT = ResponseFormat(schema=SCHEMA)


def tool_reply(arguments):
    return ok_json({"choices": [{"message": {"content": None, "tool_calls": [
        {"type": "function", "function": {"name": "Give_Information", "arguments": arguments}}]},
        "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 3, "completion_tokens": 4}})


# ---------------------------------------------------------------- decoders and helpers
def test_ndjson_decoder_handles_split_lines():
    decoder = NDJSONDecoder()
    events = decoder.feed('{"a":1}\n{"b"') + decoder.feed(':2}\n\n') + decoder.feed('{"c":3}') + decoder.flush()
    assert events == [(None, '{"a":1}'), (None, '{"b":2}'), (None, '{"c":3}')]


def test_think_filter_hides_reasoning_across_chunks():
    f = _ThinkFilter()
    out = "".join(f.feed(c) for c in ["Hi <thi", "nk>secret ", "stuff</th", "ink> there", " <", "b>"]) + f.flush()
    assert out == "Hi  there <b>"
    assert strip_think("<think>x</think>\nanswer") == "answer"
    assert strip_think("answer <think>unfinished") == "answer"


def test_response_format_validation():
    with pytest.raises(TypeError):
        ResponseFormat(schema="not a dict")
    with pytest.raises(ValueError):
        ResponseFormat(name="bad name")
    assert ResponseFormat().json_schema() == {"type": "object"}


def test_unknown_reasoning_effort_is_a_bad_request():
    with pytest.raises(ProviderError) as err:
        Recorder().client().chat("openai", api_key=KEY, model="gpt-5", messages=MESSAGES, reasoning_effort="max")
    assert err.value.code == "bad_request"


# ---------------------------------------------------------------- OpenAI and model rules
def test_openai_reasoning_models_drop_temperature_and_take_the_lowest_effort():
    cases = {"gpt-5.1-mini": "none", "gpt-5": "minimal", "o3-mini": "low", "o1-mini": None, "gpt-4o": None}
    for model, effort in cases.items():
        rec = Recorder(ok_json({"choices": [{"message": {"content": "ok"}}]}))
        rec.client().chat("openai", api_key=KEY, model=model, messages=MESSAGES, temperature=0.5,
                          reasoning_effort="lowest")
        assert rec.body.get("reasoning_effort") == effort, model
        assert ("temperature" in rec.body) == (model == "gpt-4o"), model


def test_openai_structured_output_uses_a_tool_and_returns_its_arguments():
    rec = Recorder(tool_reply('{"name": "x"}'))
    reply = rec.client().chat("openai", api_key=KEY, model="gpt-4o", messages=MESSAGES, response_format=FMT)
    tool = rec.body["tools"][0]
    assert tool["function"]["name"] == "Give_Information" and tool["function"]["parameters"] == SCHEMA
    assert rec.body["tool_choice"] == "auto"
    assert reply.text == '{"name": "x"}' and reply.finish_reason == "tool_calls"


def test_streamed_tool_arguments_arrive_as_text():
    rec = Recorder(ok_sse(sse(
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"na'}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'me":1}'}}]},
                      "finish_reason": "tool_calls"}]}, "[DONE]")))
    events = list(rec.client().stream("openai", api_key=KEY, model="gpt-4o", messages=MESSAGES, response_format=FMT))
    assert "".join(e.text for e in events if e.type == "delta") == '{"name":1}'


def test_openai_reports_cached_prompt_tokens():
    rec = Recorder(ok_json({"choices": [{"message": {"content": "ok"}}], "usage": {
        "prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 80}}}))
    usage = rec.client().chat("openai", api_key=KEY, model="gpt-4o", messages=MESSAGES).usage
    assert (usage.input_tokens, usage.cache_read_tokens) == (100, 80)
    assert usage.to_dict() == {"input_tokens": 100, "output_tokens": 5, "cache_read_tokens": 80}


@pytest.mark.parametrize("provider,url", [
    ("kimi", "https://api.moonshot.ai/v1"),
    ("nvidia", "https://integrate.api.nvidia.com/v1"),
    ("zai", "https://api.z.ai/api/paas/v4"),
    ("minimax", "https://api.minimax.io/v1"),
    ("arcee", "https://api.arcee.ai/api/v1"),
    ("ai21", "https://api.ai21.com/studio/v1"),
])
def test_native_openai_protocol_providers_use_their_own_endpoint(provider, url):
    rec = Recorder(ok_json({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}))
    reply = rec.client().chat(provider, api_key=KEY, model="m", messages=MESSAGES)
    assert str(rec.requests[0].url) == f"{url}/chat/completions"
    assert rec.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert reply.text == "ok"


def test_kimi_rules_and_json_schema_output():
    rec = Recorder(ok_json({"choices": [{"message": {"content": '{"name":"k"}'}}]}),
                   ok_json({"choices": [{"message": {"content": "ok"}}]}))
    client = rec.client()
    client.chat("kimi", api_key=KEY, model="kimi-k3", messages=MESSAGES, temperature=0.5, max_tokens=9,
                response_format=FMT)
    body = rec.body
    assert "temperature" not in body and body["max_completion_tokens"] == 9 and "max_tokens" not in body
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    client.chat("kimi", api_key=KEY, model="moonshot-v1-8k", messages=MESSAGES, temperature=1.7, thinking=False)
    assert rec.body["temperature"] == 1.0 and rec.body["thinking"] == {"type": "disabled"}


def test_nvidia_forces_the_tool_switches_thinking_and_strips_think_tags():
    rec = Recorder(tool_reply('{"name":"n"}'),
                   ok_json({"choices": [{"message": {"content": "<think>hmm</think>\nAnswer"}}]}),
                   ok_json({"choices": [{"message": {"content": "ok"}}]}))
    client = rec.client()
    reply = client.chat("nvidia", api_key=KEY, model="nvidia/nemotron-3-super", messages=MESSAGES,
                        response_format=FMT)
    assert rec.body["tool_choice"] == {"type": "function", "function": {"name": "Give_Information"}}
    assert rec.body["chat_template_kwargs"] == {"enable_thinking": False}
    assert reply.text == '{"name":"n"}'
    assert client.chat("nvidia", api_key=KEY, model="meta/llama", messages=MESSAGES).text == "Answer"
    client.chat("nvidia", api_key=KEY, model="nvidia/nvidia-nemotron-nano-9b-v2", messages=MESSAGES,
                thinking=False)
    assert rec.body["messages"][0] == {"role": "system", "content": "/no_think\nBe brief."}


def test_nvidia_stream_hides_think_spans():
    rec = Recorder(ok_sse(sse(
        {"choices": [{"index": 0, "delta": {"content": "<think>pla"}}]},
        {"choices": [{"index": 0, "delta": {"content": "n</think>Hel"}}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]}, "[DONE]")))
    events = list(rec.client().stream("nvidia", api_key=KEY, model="m", messages=MESSAGES))
    assert "".join(e.text for e in events if e.type == "delta") == "Hello"


@pytest.mark.parametrize("provider,on", [("zai", "enabled"), ("minimax", "adaptive")])
def test_thinking_switch_providers(provider, on):
    rec = Recorder(*[ok_json({"choices": [{"message": {"content": "ok"}}]})] * 3)
    client = rec.client()
    client.chat(provider, api_key=KEY, model="m", messages=MESSAGES, thinking=True)
    assert rec.body["thinking"] == {"type": on}
    client.chat(provider, api_key=KEY, model="m", messages=MESSAGES, thinking=False)
    assert rec.body["thinking"] == {"type": "disabled"}
    client.chat(provider, api_key=KEY, model="m", messages=MESSAGES, response_format=FMT)
    assert rec.body["thinking"] == {"type": "disabled"} and rec.body["tool_choice"]["type"] == "function"


def test_ai21_uses_json_mode():
    rec = Recorder(ok_json({"choices": [{"message": {"content": "{}"}}]}))
    rec.client().chat("ai21", api_key=KEY, model="jamba-mini", messages=MESSAGES, response_format=FMT)
    assert rec.body["response_format"] == {"type": "json_object"} and "tools" not in rec.body


# ---------------------------------------------------------------- Anthropic
def test_anthropic_cache_breakpoints_system_blocks_and_leading_user():
    rec = Recorder(ok_json({"content": [{"type": "text", "text": "ok"}], "usage": {
        "input_tokens": 20, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 3000,
        "output_tokens": 7}}))
    messages = [{"role": "system", "content": f"s{i}", "cache": True} for i in range(5)] + [
        {"role": "system", "content": "clock"}, {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "Hi"}]
    reply = rec.client().chat("anthropic", api_key=KEY, model="claude-sonnet-4-5", messages=messages,
                              temperature=0.3)
    system = rec.body["system"]
    assert [b["text"] for b in system] == ["s0", "s1", "s2", "s3", "s4", "clock"]
    assert [("cache_control" in b) for b in system] == [True, True, True, True, False, False]  # max 4
    assert rec.body["messages"][0] == {"role": "user", "content": "Continue."}
    assert "temperature" not in rec.body  # Sonnet 4.5+ takes no temperature
    assert (reply.usage.input_tokens, reply.usage.cache_write_tokens, reply.usage.cache_read_tokens) == (4020, 1000,
                                                                                                         3000)


def test_anthropic_without_cache_flags_sends_a_plain_system_string():
    rec = Recorder(ok_json({"content": [{"type": "text", "text": "ok"}], "usage": {}}))
    rec.client().chat("anthropic", api_key=KEY, model="claude-3-5-haiku", messages=MESSAGES, temperature=0.3)
    assert rec.body["system"] == "Be brief." and rec.body["temperature"] == 0.3


def test_anthropic_temperature_rule():
    assert skips_temperature("claude-opus-4-6") and skips_temperature("claude-sonnet-4-5-20250929")
    assert skips_temperature("claude-opus-5") and skips_temperature("claude-sonnet-5-1")
    assert not skips_temperature("claude-opus-4-1-20250805") and not skips_temperature("claude-sonnet-4-20250514")
    assert not skips_temperature("claude-3-7-sonnet") and not skips_temperature("claude-haiku-4-5")


def test_anthropic_structured_output_is_a_forced_tool():
    rec = Recorder(ok_json({"content": [{"type": "tool_use", "name": "Give_Information", "input": {"name": "a"}}],
                            "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 6}}))
    reply = rec.client().chat("anthropic", api_key=KEY, model="claude-x", messages=MESSAGES, response_format=FMT)
    assert rec.body["tool_choice"] == {"type": "tool", "name": "Give_Information"}
    assert rec.body["tools"][0]["input_schema"] == SCHEMA
    assert json.loads(reply.text) == {"name": "a"} and reply.finish_reason == "tool_calls"


def test_anthropic_stream_tool_json_and_cache_usage():
    rec = Recorder(ok_sse(sse(
        ("message_start", {"type": "message_start", "message": {"model": "claude-x", "usage": {
            "input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 1}}}),
        ("content_block_delta", {"type": "content_block_delta", "delta": {"type": "input_json_delta",
                                                                          "partial_json": '{"name":'}}),
        ("content_block_delta", {"type": "content_block_delta", "delta": {"type": "input_json_delta",
                                                                          "partial_json": '"z"}'}}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                           "usage": {"output_tokens": 12}}),
    )))
    events = list(rec.client().stream("anthropic", api_key=KEY, model="claude-x", messages=MESSAGES,
                                      response_format=FMT))
    assert "".join(e.text for e in events if e.type == "delta") == '{"name":"z"}'
    done = events[-1]
    assert (done.usage.input_tokens, done.usage.output_tokens, done.usage.cache_read_tokens) == (100, 12, 90)


# ---------------------------------------------------------------- Gemini
def test_gemini_structured_output_and_leading_user_turn():
    rec = Recorder(ok_json({"candidates": [{"content": {"parts": [{"text": '{"name":"g"}'}]},
                                            "finishReason": "STOP"}]}))
    messages = [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "go"}]
    rec.client().chat("gemini", api_key=KEY, model="gemini-x", messages=messages, response_format=FMT)
    config = rec.body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"] == {"type": "OBJECT", "properties": {
        "name": {"type": "STRING"}, "tags": {"type": "ARRAY", "items": {"type": "STRING"}}}, "required": ["name"]}
    assert rec.body["contents"][0] == {"role": "user", "parts": [{"text": "Continue."}]}


def test_gemini_schema_conversion_edge_cases():
    assert to_gemini_schema({"type": ["integer", "null"], "enum": [1, 2], "minimum": 0}) == {
        "type": "INTEGER", "nullable": True, "enum": ["1", "2"]}
    assert to_gemini_schema({"type": "object", "required": ["missing"]}) == {"type": "OBJECT", "properties": {}}


# ---------------------------------------------------------------- Cohere v2
def test_cohere_chat_stream_and_models():
    rec = Recorder(
        ok_json({"finish_reason": "COMPLETE", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Hi!"}]}, "usage": {"tokens": {"input_tokens": 4, "output_tokens": 2}}}),
        ok_sse(sse(("message-start", {"type": "message-start"}),
                   ("content-delta", {"type": "content-delta", "delta": {"message": {"content": {"text": "He"}}}}),
                   ("content-delta", {"type": "content-delta", "delta": {"message": {"content": {"text": "y"}}}}),
                   ("message-end", {"type": "message-end", "delta": {"finish_reason": "MAX_TOKENS", "usage": {
                       "billed_units": {"input_tokens": 3, "output_tokens": 2}}}}))),
        ok_json({"models": [{"name": "command-b", "context_length": 128000}], "next_page_token": "t2"}),
        ok_json({"models": [{"name": "command-a"}]}),
    )
    client = rec.client()
    reply = client.chat("cohere", api_key=KEY, model="command-a", messages=MESSAGES, max_tokens=5,
                        response_format=FMT)
    assert str(rec.requests[0].url) == "https://api.cohere.com/v2/chat"
    assert rec.body["messages"][0] == {"role": "system", "content": "Be brief."}
    assert rec.body["response_format"] == {"type": "json_object", "json_schema": SCHEMA}
    assert (reply.text, reply.finish_reason, reply.usage.input_tokens) == ("Hi!", "stop", 4)
    events = list(client.stream("cohere", api_key=KEY, model="command-a", messages=MESSAGES))
    assert "".join(e.text for e in events if e.type == "delta") == "Hey"
    assert events[-1].finish_reason == "length" and events[-1].usage.output_tokens == 2
    assert [m.id for m in client.list_models("cohere", api_key=KEY)] == ["command-b", "command-a"]
    assert rec.requests[-1].url.params["page_token"] == "t2"


# ---------------------------------------------------------------- Alibaba DashScope (native)
def test_dashscope_native_request_reply_and_stream():
    rec = Recorder(
        ok_json({"output": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant",
                                                                              "content": "ok"}}]},
                 "usage": {"input_tokens": 8, "output_tokens": 1}, "request_id": "r"}),
        ok_sse(sse(
            {"output": {"choices": [{"finish_reason": "null", "message": {"content": "Hel"}}]},
             "usage": {"input_tokens": 8, "output_tokens": 1}},
            {"output": {"choices": [{"finish_reason": "stop", "message": {"content": "lo"}}]},
             "usage": {"input_tokens": 8, "output_tokens": 2}})),
    )
    client = rec.client()
    reply = client.chat("alibaba", api_key=KEY, model="qwen-plus", messages=MESSAGES, temperature=0.4,
                        max_tokens=20, response_format=FMT, thinking=False)
    request = rec.requests[0]
    assert str(request.url) == "https://dashscope-us.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    assert rec.body == {"model": "qwen-plus", "input": {"messages": [
        {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]}, "parameters": {
        "result_format": "message", "temperature": 0.4, "max_tokens": 20, "response_format": {"type": "json_object"},
        "enable_thinking": False}}
    assert (reply.text, reply.finish_reason, reply.usage.output_tokens) == ("ok", "stop", 1)
    events = list(client.stream("alibaba", api_key=KEY, model="qwen-plus", messages=MESSAGES))
    assert rec.requests[1].headers["x-dashscope-sse"] == "enable"
    assert rec.body["parameters"]["incremental_output"] is True
    assert "".join(e.text for e in events if e.type == "delta") == "Hello"
    assert events[-1].finish_reason == "stop" and events[-1].usage.output_tokens == 2


def test_dashscope_errors():
    rec = Recorder((400, {"code": "InvalidParameter", "message": "bad model", "request_id": "r"}, "application/json"),
                   ok_sse('event:error\ndata: {"code":"Throttling.RateQuota","message":"slow down"}\n\n'))
    client = rec.client()
    with pytest.raises(ProviderError) as err:
        client.chat("alibaba", api_key=KEY, model="x", messages=MESSAGES)
    assert (err.value.code, err.value.message) == ("bad_request", "bad model")
    with pytest.raises(ProviderError) as err:
        list(client.stream("alibaba", api_key=KEY, model="x", messages=MESSAGES))
    assert err.value.message == "slow down"


# ---------------------------------------------------------------- Meta (Responses)
def test_meta_responses_request_output_and_stream():
    rec = Recorder(
        ok_json({"model": "muse-spark-1.1", "status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "Hi there"}]}],
            "usage": {"input_tokens": 9, "output_tokens": 3}}),
        ok_json({"status": "completed", "output": [{"type": "function_call", "name": "Give_Information",
                                                    "arguments": '{"name":"m"}'}], "usage": {}}),
        ok_sse(sse(("response.output_text.delta", {"type": "response.output_text.delta", "delta": "Yo"}),
                   ("response.completed", {"type": "response.completed", "response": {
                       "status": "completed", "usage": {"input_tokens": 4, "output_tokens": 1}}}))),
    )
    client = rec.client()
    reply = client.chat("meta", api_key=KEY, model="muse-spark-1.1", messages=MESSAGES, max_tokens=50,
                        temperature=0.2)
    assert str(rec.requests[0].url) == "https://api.meta.ai/v1/responses"
    assert rec.body == {"model": "muse-spark-1.1", "stream": False, "temperature": 0.2, "max_output_tokens": 50,
                        "input": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]}
    assert (reply.text, reply.usage.input_tokens, reply.finish_reason) == ("Hi there", 9, "stop")
    structured = client.chat("meta", api_key=KEY, model="m", messages=MESSAGES, response_format=FMT)
    assert rec.body["tool_choice"] == {"type": "function", "name": "Give_Information"}
    assert structured.text == '{"name":"m"}' and structured.finish_reason == "tool_calls"
    events = list(client.stream("meta", api_key=KEY, model="m", messages=MESSAGES))
    assert events[0].text == "Yo" and events[-1].usage.output_tokens == 1


# ---------------------------------------------------------------- Ollama (native)
def test_ollama_needs_no_key_and_streams_ndjson():
    lines = "\n".join(json.dumps(x) for x in [
        {"model": "llama3.2", "message": {"role": "assistant", "content": "Hel"}, "done": False},
        {"model": "llama3.2", "message": {"role": "assistant", "content": "lo"}, "done": False},
        {"model": "llama3.2", "message": {"content": ""}, "done": True, "done_reason": "stop",
         "prompt_eval_count": 12, "eval_count": 2},
    ]) + "\n"
    rec = Recorder(
        ok_json({"model": "llama3.2", "message": {"content": '{"name":"o"}'}, "done": True,
                 "prompt_eval_count": 7, "eval_count": 5}),
        (200, lines, "application/x-ndjson"),
    )
    client = rec.client()
    reply = client.chat("ollama", api_key="", model="llama3.2", messages=MESSAGES, max_tokens=99,
                        base_url="http://gpu-box:11434/v1/", response_format=FMT)
    assert str(rec.requests[0].url) == "http://gpu-box:11434/api/chat"
    assert "authorization" not in rec.requests[0].headers
    assert rec.body["options"] == {"num_predict": 99} and rec.body["format"] == SCHEMA
    assert (reply.text, reply.usage.input_tokens, reply.usage.output_tokens) == ('{"name":"o"}', 7, 5)
    events = list(client.stream("ollama", api_key="", model="llama3.2", messages=MESSAGES,
                                base_url="http://gpu-box:11434"))
    assert "".join(e.text for e in events if e.type == "delta") == "Hello"
    assert (events[-1].usage.input_tokens, events[-1].finish_reason) == (12, "stop")


def test_ollama_error_message():
    rec = Recorder((404, {"error": "model 'nope' not found"}, "application/json"))
    with pytest.raises(ProviderError) as err:
        rec.client().chat("ollama", api_key="", model="nope", messages=MESSAGES, base_url="http://h:11434")
    assert (err.value.code, err.value.message) == ("not_found", "model 'nope' not found")


def test_async_client_streams_ndjson():
    lines = json.dumps({"message": {"content": "a"}, "done": False}) + "\n" + json.dumps(
        {"message": {"content": "b"}, "done": True, "eval_count": 2}) + "\n"
    rec = Recorder((200, lines, "application/x-ndjson"))

    async def run():
        client = AsyncChatClient(http=httpx.AsyncClient(transport=httpx.MockTransport(rec)))
        return [e async for e in client.stream("ollama", api_key="", model="m", messages=MESSAGES,
                                               base_url="http://h:11434")]

    events = asyncio.run(run())
    assert "".join(e.text for e in events if e.type == "delta") == "ab" and events[-1].usage.output_tokens == 2
