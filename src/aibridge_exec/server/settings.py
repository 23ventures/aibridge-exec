"""Server configuration from environment variables (prefix ``AIBRIDGE_EXEC_``)."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field


def _list(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (value or "").split(",") if item.strip())


def _bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServerSettings:
    # Service tokens callers must send as "Authorization: Bearer <token>". Empty = open (local dev only).
    api_tokens: tuple[str, ...] = ()
    # Browser origins allowed to call the server directly (CORS). Empty = none.
    cors_origins: tuple[str, ...] = ()
    # Provider ids to expose. Empty = every registered provider except openai_compatible.
    providers: tuple[str, ...] = ()
    # Let callers send their own base_url (enables openai_compatible). Off by default: it lets
    # callers make the server send requests to any URL, including your internal network.
    allow_custom_base_url: bool = False
    # Server-side base URL overrides, e.g. {"openai": "https://gateway.internal/openai/v1"}.
    base_urls: Mapping[str, str] = field(default_factory=dict)
    max_messages: int = 500
    max_message_chars: int = 200_000
    max_total_chars: int = 1_000_000

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ServerSettings":
        env = os.environ if environ is None else environ
        prefix = "AIBRIDGE_EXEC_BASE_URL_"
        return cls(
            api_tokens=_list(env.get("AIBRIDGE_EXEC_API_TOKENS")),
            cors_origins=_list(env.get("AIBRIDGE_EXEC_CORS_ORIGINS")),
            providers=_list(env.get("AIBRIDGE_EXEC_PROVIDERS")),
            allow_custom_base_url=_bool(env.get("AIBRIDGE_EXEC_ALLOW_CUSTOM_BASE_URL")),
            base_urls={k[len(prefix):].lower(): v for k, v in env.items() if k.startswith(prefix) and v},
            max_messages=int(env.get("AIBRIDGE_EXEC_MAX_MESSAGES", cls.max_messages)),
            max_message_chars=int(env.get("AIBRIDGE_EXEC_MAX_MESSAGE_CHARS", cls.max_message_chars)),
            max_total_chars=int(env.get("AIBRIDGE_EXEC_MAX_TOTAL_CHARS", cls.max_total_chars)),
        )

    def provider_enabled(self, provider_id: str) -> bool:
        if self.providers:
            return provider_id in self.providers
        return provider_id != "openai_compatible" or self.allow_custom_base_url
