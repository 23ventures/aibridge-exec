"""Stateless chat adapters for LLM providers. Bring your own key on every call.

Built in: ``openai``, ``anthropic``, ``gemini``, ``xai``, ``openrouter``, ``groq``, ``deepseek``,
``mistral``, ``together`` and ``openai_compatible`` (any OpenAI-compatible URL).

    from aibridge_exec.providers import ChatClient

    with ChatClient() as client:
        reply = client.chat("anthropic", api_key=KEY, model="claude-sonnet-4-5",
                            messages=[{"role": "user", "content": "Hello"}])
        print(reply.text, reply.usage)

        for event in client.stream("openai", api_key=KEY, model="gpt-4o-mini",
                                   messages=[{"role": "user", "content": "Hi"}]):
            if event.type == "delta":
                print(event.text, end="")
"""
from .base import HttpCall, ProviderAdapter, StreamParser
from .client import AsyncChatClient, ChatClient
from .errors import ProviderError
from .registry import get_provider, list_providers, register_provider
from .types import ChatMessage, ChatRequest, ChatResponse, ModelInfo, ProviderInfo, StreamEvent, Usage

__all__ = [
    "AsyncChatClient",
    "ChatClient",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "HttpCall",
    "ModelInfo",
    "ProviderAdapter",
    "ProviderError",
    "ProviderInfo",
    "StreamEvent",
    "StreamParser",
    "Usage",
    "get_provider",
    "list_providers",
    "register_provider",
]
