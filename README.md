# aibridge-exec v1

One Python interface to many LLM providers, with your own API key on every call. Use it as a
library, or run it as a small stateless HTTP service that any language can call. It powers
the AIBridge platform and the aibridge-playground app.

- **Library:** `aibridge_exec.providers`: sync and async clients, streaming, model listing, typed errors
- **Media:** `aibridge_exec.media`: image and video generation (OpenAI, Google, Stability, Alibaba, Luma)
- **HTTP server** (optional extra): `aibridge-exec serve`, a FastAPI app with SSE streaming
- **Also included:** boundary contracts shared with the AIBridge platform (`aibridge_exec.contracts`)
  and a local Python code runtime (`aibridge_exec.code_runtime`)

It stores nothing, logs no keys and needs no database. `lint-imports` enforces that the package
never imports Flask, SQLAlchemy, Redis, Kafka, Celery or the platform, and that the library
modules never depend on the web server stack.

## Supported providers

| Provider id | Provider | Notes |
| --- | --- | --- |
| `openai` | OpenAI | Chat Completions API |
| `anthropic` | Anthropic | Messages API |
| `gemini` | Google Gemini | Gemini API (AI Studio keys) |
| `xai` | xAI (Grok) | OpenAI-compatible |
| `openrouter` | OpenRouter | OpenAI-compatible, hundreds of models with one key |
| `groq` | Groq | OpenAI-compatible |
| `deepseek` | DeepSeek | OpenAI-compatible |
| `mistral` | Mistral | OpenAI-compatible |
| `together` | Together AI | OpenAI-compatible |
| `openai_compatible` | Any OpenAI-compatible URL | vLLM, Ollama, LM Studio, gateways; you pass `base_url` |

Model ids change often, so the package doesn't hard-code them. Call `list_models` with your key
to get the ids your account can use right now.

## Install

Python 3.11 or newer.

```bash
# Library only (depends on httpx)
pip install "aibridge-exec @ git+https://github.com/your-org/aibridge-exec"

# Library plus the HTTP server (adds FastAPI and uvicorn)
pip install "aibridge-exec[server] @ git+https://github.com/your-org/aibridge-exec"
```

In a `requirements.txt`, pin a tag or commit:

```text
aibridge-exec @ git+https://github.com/your-org/aibridge-exec@v0.2.0
```

With Poetry or uv:

```bash
poetry add "git+https://github.com/your-org/aibridge-exec"
uv add "aibridge-exec @ git+https://github.com/your-org/aibridge-exec"
```

## Library quick start

```python
import os
from aibridge_exec.providers import ChatClient

with ChatClient() as client:
    models = client.list_models("anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
    print([m.id for m in models][:5])

    reply = client.chat(
        "anthropic",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=models[0].id,
        system="Answer in one sentence.",
        messages=[{"role": "user", "content": "What is a suspension bridge?"}],
    )
    print(reply.text)
    print(reply.usage.input_tokens, reply.usage.output_tokens, reply.finish_reason)
```

Every provider takes the same arguments: `model`, `messages` (roles `system`, `user`,
`assistant`), and optionally `system`, `temperature` (0 to 2) and `max_tokens`. Differences
between providers are handled for you, for example Anthropic's separate system prompt and
required `max_tokens`, or Gemini's `model` role name.

### Streaming

```python
with ChatClient() as client:
    for event in client.stream("openai", api_key=key, model="<model id>",
                               messages=[{"role": "user", "content": "Count to five"}]):
        if event.type == "delta":
            print(event.text, end="", flush=True)
        else:  # the last event is "done"
            print("\n", event.finish_reason, event.usage)
```

### Async

```python
from aibridge_exec.providers import AsyncChatClient

async with AsyncChatClient() as client:
    async for event in client.stream("gemini", api_key=key, model="<model id>", messages=messages):
        ...
    reply = await client.chat("xai", api_key=key, model="<model id>", messages=messages)
```

### Errors

All failures raise `ProviderError` with a stable `code`. Messages never contain your key.

| `code` | Meaning |
| --- | --- |
| `auth` | Missing, invalid or unauthorized key |
| `rate_limited` | The provider throttled the request or the quota ran out |
| `bad_request` | The provider rejected the request (bad parameters, unknown model) |
| `not_found` | Model or endpoint not found |
| `upstream` | Provider-side failure or overload |
| `timeout`, `network` | The provider couldn't be reached in time |

```python
from aibridge_exec.providers import ProviderError

try:
    client.chat(...)
except ProviderError as error:
    if error.code == "rate_limited":
        retry_later()
    else:
        raise
```

### Local models and gateways

```python
client.chat("openai_compatible", api_key="unused", model="llama3.1",
            base_url="http://localhost:11434/v1", messages=messages)

# Route a built-in provider through a proxy for every call
client = ChatClient(base_urls={"openai": "https://gateway.internal/openai/v1"})
```

### Adding a provider

Adapters only describe HTTP requests and parse responses, so a new provider is one class:
subclass `ProviderAdapter` (or reuse `OpenAICompatibleAdapter` with a new `ProviderInfo`) and call
`register_provider(MyAdapter())`. See `aibridge_exec/providers/anthropic.py` for a complete example.

## Image and video generation

`aibridge_exec.media` generates images and videos with the caller's key. It is library-only (the
HTTP server doesn't expose it yet).

| Provider id | Image | Video |
| --- | --- | --- |
| `openai` | GPT Image, DALL·E (generate and edit) | Sora |
| `gemini` | Gemini image models, Imagen (generate and edit) | Veo |
| `stability` | Stable Image (sd3, ultra, core, inpaint) | image-to-video |
| `alibaba` | Wan / Wanx | Wan |
| `luma` | | Dream Machine |

```python
from aibridge_exec.media import ImageGenerationRequest, VideoGenerationRequest
from aibridge_exec.media import get_image_provider, get_video_provider

images = get_image_provider("openai")(api_key=KEY).generate(
    ImageGenerationRequest(prompts=["a lighthouse at dusk"], model="gpt-image-1")
)
png = images.results[0].images[0].content

veo = get_video_provider("gemini")(api_key=KEY)
job = veo.start_generation(VideoGenerationRequest(model="veo-3.1-generate-preview", prompt="waves"))
status = veo.check_status(job.provider_job_id)   # poll until "completed" or "failed"
mp4 = veo.download_video(status.result_url)
```

Video is asynchronous: the caller schedules its own polling. Reference images for edits and
image-to-video are http(s) or data: URIs. Result URLs never contain the API key; `download_video`
adds it as a header. Sora image-to-video resizes the reference image, which needs Pillow:
`pip install "aibridge-exec[media]"`.

## Run the HTTP server

```bash
pip install "aibridge-exec[server] @ git+https://github.com/your-org/aibridge-exec"
export AIBRIDGE_EXEC_API_TOKENS="$(openssl rand -hex 32)"
aibridge-exec serve --host 0.0.0.0 --port 8000
```

Or with Docker, from a checkout of the repository:

```bash
docker build -t aibridge-exec .
docker run -p 8000:8000 -e AIBRIDGE_EXEC_API_TOKENS=change-me aibridge-exec
```

Interactive API docs are served at `/docs`.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `AIBRIDGE_EXEC_API_TOKENS` | empty | Comma-separated service tokens. Callers send `Authorization: Bearer <token>`. Empty means no authentication: only for local use. |
| `AIBRIDGE_EXEC_PROVIDERS` | all | Comma-separated provider ids to expose |
| `AIBRIDGE_EXEC_CORS_ORIGINS` | empty | Browser origins allowed to call the server directly |
| `AIBRIDGE_EXEC_ALLOW_CUSTOM_BASE_URL` | `false` | Let callers send `base_url` (enables `openai_compatible`). Leave off on public servers: it lets callers make the server request any URL. |
| `AIBRIDGE_EXEC_BASE_URL_<PROVIDER>` | provider default | Server-side URL override, for example `AIBRIDGE_EXEC_BASE_URL_OPENAI` |
| `AIBRIDGE_EXEC_MAX_MESSAGES` | `500` | Messages per request |
| `AIBRIDGE_EXEC_MAX_TOTAL_CHARS` | `1000000` | Characters per request |

### HTTP API

The provider key goes in the `X-Provider-Key` header, never in the URL or body. The service
token, if configured, goes in `Authorization`.

```bash
EXEC=http://localhost:8000
AUTH="Authorization: Bearer $AIBRIDGE_EXEC_API_TOKEN"

curl -s $EXEC/v1/providers -H "$AUTH"

curl -s $EXEC/v1/models -H "$AUTH" -H "X-Provider-Key: $OPENAI_API_KEY" \
  -H 'Content-Type: application/json' -d '{"provider": "openai"}'

curl -N $EXEC/v1/chat -H "$AUTH" -H "X-Provider-Key: $OPENAI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"provider": "openai", "model": "<model id>", "stream": true,
       "messages": [{"role": "user", "content": "Hello"}]}'
```

A streamed reply is Server-Sent Events, one JSON object per event:

```text
event: delta
data: {"type":"delta","text":"Hel"}

event: delta
data: {"type":"delta","text":"lo"}

event: done
data: {"type":"done","finish_reason":"stop","model":"...","usage":{"input_tokens":8,"output_tokens":2}}
```

If the provider fails mid-stream you get `event: error` with `{"type":"error","error":{"code":...,"message":...}}`.
Failures before the stream starts are normal HTTP errors with the same `error` object.
With `"stream": false` the response is `{"text", "model", "finish_reason", "usage"}`.

## Security notes

- Keys are used for one request and then discarded. The server doesn't log request bodies or headers.
- Put the server behind TLS and set `AIBRIDGE_EXEC_API_TOKENS` whenever it's reachable from a network.
- Keep `AIBRIDGE_EXEC_ALLOW_CUSTOM_BASE_URL` off unless every caller is trusted.
- The package also contains a local Python code runtime (`aibridge_exec.code_runtime`). It runs code
  as an ordinary process and is not a sandbox; don't expose it to untrusted input.

## Code runtime configuration

The code runtime reads settings from the environment. A host application can supply its own
configuration mapping instead:

```python
from aibridge_exec.code_runtime.settings import set_config_provider

set_config_provider(lambda: my_app_config)   # zero-argument callable returning a mapping
```

## Development

```bash
pip install -e ".[dev]"
lint-imports        # layer rules (see .importlinter)
pytest --cov        # provider adapters run against recorded HTTP shapes, no network needed
```

Layout:

```
src/aibridge_exec/providers/   adapters (openai_compat, anthropic, gemini), clients, SSE, errors
src/aibridge_exec/server/      FastAPI app, settings, ASGI entry point
src/aibridge_exec/cli.py       `aibridge-exec serve | providers`
src/aibridge_exec/contracts/   tool and agent contracts shared with the platform
src/aibridge_exec/code_runtime/ local Python execution runtime
```

## License

To be decided before the repository is made public (see the licensing checklist: add a
`LICENSE` file with the copyright holder and set `license` in `pyproject.toml`).

## Deploying to AWS

`.github/workflows/deploy.yml` tests, builds and deploys this repo to ECS Fargate on every push
to `main`; the task definitions are in `deploy/ecs/`. The one-time AWS and GitHub setup is in
the "AIBridge on AWS: setup and CI/CD" guide.
