import json

import httpx
import pytest
from fastapi.testclient import TestClient

from aibridge_exec.providers import AsyncChatClient
from aibridge_exec.server import ServerSettings, create_app

from .test_providers import KEY, Recorder, ok_json, ok_sse, sse

TOKEN = "service-token"
AUTH = {"Authorization": f"Bearer {TOKEN}", "X-Provider-Key": KEY}
BODY = {"provider": "openai", "model": "gpt-x", "messages": [{"role": "user", "content": "Hi"}]}


def make(rec, **settings):
    settings.setdefault("api_tokens", (TOKEN,))
    client = AsyncChatClient(http=httpx.AsyncClient(transport=httpx.MockTransport(rec)))
    return TestClient(create_app(ServerSettings(**settings), client=client))


def events(response):
    out = []
    for block in response.text.strip().split("\n\n"):
        data = [line[6:] for line in block.split("\n") if line.startswith("data: ")]
        out.append(json.loads("\n".join(data)))
    return out


def test_health_is_public_and_other_routes_need_the_service_token():
    with make(Recorder()) as http:
        assert http.get("/health").json()["status"] == "ok"
        response = http.get("/v1/providers")
        assert response.status_code == 401 and response.json()["error"]["code"] == "auth"
        assert http.get("/v1/providers", headers={"Authorization": "Bearer wrong"}).status_code == 401
        ids = [p["id"] for p in http.get("/v1/providers", headers=AUTH).json()["providers"]]
        assert "anthropic" in ids and "openai_compatible" not in ids


def test_open_mode_without_tokens():
    with make(Recorder(), api_tokens=()) as http:
        assert http.get("/v1/providers").status_code == 200


def test_chat_stream_relays_events():
    rec = Recorder(ok_sse(sse({"choices": [{"index": 0, "delta": {"content": "He"}}]},
                              {"choices": [{"index": 0, "delta": {"content": "y"}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 2, "completion_tokens": 1}}, "[DONE]")))
    with make(rec) as http:
        response = http.post("/v1/chat", json=BODY, headers=AUTH)
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    assert events(response) == [
        {"type": "delta", "text": "He"},
        {"type": "delta", "text": "y"},
        {"type": "done", "finish_reason": "stop", "model": "gpt-x", "usage": {"input_tokens": 2, "output_tokens": 1}},
    ]
    assert rec.requests[0].headers["authorization"] == f"Bearer {KEY}"  # provider key, not the service token


def test_chat_non_stream():
    rec = Recorder(ok_json({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}))
    with make(rec) as http:
        response = http.post("/v1/chat", json={**BODY, "stream": False}, headers=AUTH)
    assert response.json() == {"text": "ok", "model": "gpt-x", "finish_reason": "stop",
                               "usage": {"input_tokens": None, "output_tokens": None}}


def test_provider_errors_before_stream_are_http_errors_without_the_key():
    rec = Recorder((401, {"error": {"message": f"bad key {KEY}"}}, "application/json"))
    with make(rec) as http:
        response = http.post("/v1/chat", json=BODY, headers=AUTH)
    assert response.status_code == 401
    error = response.json()["error"]
    assert error["code"] == "auth" and error["provider"] == "openai" and KEY not in response.text


def test_errors_mid_stream_become_error_events():
    rec = Recorder(ok_sse(sse({"choices": [{"index": 0, "delta": {"content": "par"}}]},
                              {"error": {"message": "overloaded", "type": "server_error"}})))
    with make(rec) as http:
        response = http.post("/v1/chat", json=BODY, headers=AUTH)
    got = events(response)
    assert got[0] == {"type": "delta", "text": "par"}
    assert got[1]["type"] == "error" and got[1]["error"]["code"] == "upstream"


def test_missing_provider_key_and_validation():
    with make(Recorder()) as http:
        response = http.post("/v1/chat", json=BODY, headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 401 and response.json()["error"]["message"] == "an API key is required"
        response = http.post("/v1/chat", json={**BODY, "messages": []}, headers=AUTH)
        assert response.status_code == 422 and response.json()["error"]["code"] == "bad_request"
        response = http.post("/v1/chat", json={**BODY, "provider": "nope"}, headers=AUTH)
        assert response.status_code == 404


def test_size_limits():
    with make(Recorder(), max_messages=2, max_message_chars=10) as http:
        many = {**BODY, "messages": [{"role": "user", "content": "a"}] * 3}
        assert http.post("/v1/chat", json=many, headers=AUTH).json()["error"]["message"].startswith("too many")
        long = {**BODY, "messages": [{"role": "user", "content": "x" * 11}]}
        assert http.post("/v1/chat", json=long, headers=AUTH).json()["error"]["message"].startswith("message too")


def test_custom_base_url_is_off_by_default():
    rec = Recorder(ok_json({"choices": [{"message": {"content": "local"}}]}))
    with make(rec) as http:
        body = {**BODY, "base_url": "http://169.254.169.254/latest"}
        response = http.post("/v1/chat", json=body, headers=AUTH)
        assert response.status_code == 400 and response.json()["error"]["code"] == "config"
        response = http.post("/v1/chat", json={**BODY, "provider": "openai_compatible"}, headers=AUTH)
        assert response.status_code == 404
    assert rec.requests == []


def test_custom_base_url_when_enabled():
    rec = Recorder(ok_json({"choices": [{"message": {"content": "local"}}]}))
    with make(rec, allow_custom_base_url=True) as http:
        body = {**BODY, "provider": "openai_compatible", "base_url": "http://localhost:11434/v1", "stream": False}
        assert http.post("/v1/chat", json=body, headers=AUTH).json()["text"] == "local"
        bad = {**body, "base_url": "file:///etc/passwd"}
        assert http.post("/v1/chat", json=bad, headers=AUTH).json()["error"]["code"] == "config"


def test_models_endpoint():
    rec = Recorder(ok_json({"data": [{"id": "claude-a", "display_name": "Claude A"}], "has_more": False}))
    with make(rec) as http:
        response = http.post("/v1/models", json={"provider": "anthropic"}, headers=AUTH)
    assert response.json() == {"models": [{"id": "claude-a", "name": "Claude A", "context_length": None}]}


def test_settings_from_env():
    settings = ServerSettings.from_env({
        "AIBRIDGE_EXEC_API_TOKENS": "a, b",
        "AIBRIDGE_EXEC_CORS_ORIGINS": "https://app.example",
        "AIBRIDGE_EXEC_PROVIDERS": "openai,anthropic",
        "AIBRIDGE_EXEC_BASE_URL_OPENAI": "http://gw/v1",
        "AIBRIDGE_EXEC_ALLOW_CUSTOM_BASE_URL": "true",
    })
    assert settings.api_tokens == ("a", "b") and settings.cors_origins == ("https://app.example",)
    assert settings.base_urls == {"openai": "http://gw/v1"} and settings.allow_custom_base_url
    assert settings.provider_enabled("openai") and not settings.provider_enabled("gemini")


@pytest.mark.parametrize("argv", [["providers"]])
def test_cli_lists_providers(argv, capsys):
    from aibridge_exec.cli import main

    assert main(argv) == 0
    assert "anthropic" in capsys.readouterr().out


def test_url_only_providers_are_listed_only_with_a_url():
    with make(Recorder()) as http:
        ids = [p["id"] for p in http.get("/v1/providers", headers=AUTH).json()["providers"]]
    assert {"kimi", "cohere", "alibaba", "meta", "nvidia"} <= set(ids)
    assert "ollama" not in ids and "openai_compatible" not in ids
    with make(Recorder(), base_urls={"ollama": "http://gpu:11434"}) as http:
        ids = [p["id"] for p in http.get("/v1/providers", headers=AUTH).json()["providers"]]
        assert "ollama" in ids and "openai_compatible" not in ids
        response = http.post("/v1/models", json={"provider": "openai_compatible"}, headers=AUTH)
        assert response.status_code == 404
