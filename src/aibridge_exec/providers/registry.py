"""Provider registry. ``register_provider`` adds your own adapter."""
from __future__ import annotations

from .anthropic import AnthropicAdapter
from .base import ProviderAdapter
from .errors import ProviderError
from .gemini import GeminiAdapter
from .openai_compat import builtin_openai_compatible, custom_openai_compatible

_PROVIDERS: dict[str, ProviderAdapter] = {}


def register_provider(adapter: ProviderAdapter, *, replace: bool = False) -> None:
    if adapter.id in _PROVIDERS and not replace:
        raise ValueError(f"provider {adapter.id!r} is already registered")
    _PROVIDERS[adapter.id] = adapter


def get_provider(provider_id: str) -> ProviderAdapter:
    try:
        return _PROVIDERS[provider_id]
    except KeyError:
        raise ProviderError("unknown_provider", f"unknown provider {provider_id!r}", provider=provider_id) from None


def list_providers() -> list[ProviderAdapter]:
    return list(_PROVIDERS.values())


for _adapter in [*builtin_openai_compatible(), AnthropicAdapter(), GeminiAdapter(), custom_openai_compatible()]:
    register_provider(_adapter)
