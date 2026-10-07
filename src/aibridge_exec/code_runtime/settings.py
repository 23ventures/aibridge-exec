from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


def _bool_value(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int_value(value, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


@dataclass(frozen=True)
class CodeRuntimeSettings:
    """Configuration owned by the code-execution module."""

    enabled: bool = True
    model: str = "gpt-4o-mini"
    timeout_seconds: int = 180
    memory_limit_mb: int = 2048
    max_tokens: int = 6000
    package_install_enabled: bool = True
    package_install_timeout_seconds: int = 120
    environment_profiles: str = ""
    sessions_dir: str = "/tmp/aibridge-code-sessions"
    pip_cache_dir: str = "/tmp/aibridge-code-runtime-cache/pip"
    session_ttl_seconds: int = 24 * 60 * 60
    session_quota_bytes: int = 250 * 1024 * 1024
    max_executions_per_session: int = 1000
    max_packages_per_install: int = 10
    package_blocklist: tuple[str, ...] = (
        "ansible",
        "docker",
        "paramiko",
        "pexpect",
        "pyautogui",
    )
    max_capture_bytes: int = 1_000_000
    max_input_bytes: int = 100 * 1024 * 1024
    max_artifact_bytes: int = 50 * 1024 * 1024
    max_output_bytes: int = 100 * 1024 * 1024
    max_output_files: int = 50
    max_concurrency: int = 4

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "CodeRuntimeSettings":
        return cls.from_sources(environ=environ)

    @classmethod
    def from_sources(
        cls,
        *,
        config: Mapping[str, object] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "CodeRuntimeSettings":
        environment = environ if environ is not None else os.environ
        values = dict(config or {})
        values.update(environment)
        defaults = cls()
        blocklist_raw = values.get(
            "ORCHESTRATOR_CODE_RUNTIME_PACKAGE_BLOCKLIST",
            ",".join(defaults.package_blocklist),
        )
        blocklist = tuple(
            item.strip()
            for item in str(blocklist_raw or "").split(",")
            if item.strip()
        )
        return cls(
            enabled=_bool_value(
                values.get("ORCHESTRATOR_CODE_RUNTIME_ENABLED"),
                defaults.enabled,
            ),
            model=str(
                values.get(
                    "ORCHESTRATOR_CODE_RUNTIME_MODEL",
                    defaults.model,
                )
            ),
            timeout_seconds=_int_value(
                values.get("ORCHESTRATOR_CODE_RUNTIME_TIMEOUT_SECONDS"),
                defaults.timeout_seconds,
            ),
            memory_limit_mb=_int_value(
                values.get("ORCHESTRATOR_CODE_RUNTIME_MEMORY_MB"),
                defaults.memory_limit_mb,
                minimum=64,
            ),
            max_tokens=_int_value(
                values.get("ORCHESTRATOR_CODE_RUNTIME_MAX_TOKENS"),
                defaults.max_tokens,
            ),
            package_install_enabled=_bool_value(
                values.get(
                    "ORCHESTRATOR_CODE_RUNTIME_PACKAGE_INSTALL_ENABLED"
                ),
                defaults.package_install_enabled,
            ),
            package_install_timeout_seconds=_int_value(
                values.get(
                    "ORCHESTRATOR_CODE_RUNTIME_PACKAGE_INSTALL_TIMEOUT_SECONDS"
                ),
                defaults.package_install_timeout_seconds,
            ),
            environment_profiles=str(
                values.get(
                    "ORCHESTRATOR_CODE_RUNTIME_ENVIRONMENT_PROFILES",
                    defaults.environment_profiles,
                )
            ),
            sessions_dir=str(
                values.get(
                    "ORCHESTRATOR_CODE_RUNTIME_SESSIONS_DIR",
                    defaults.sessions_dir,
                )
            ),
            pip_cache_dir=str(
                values.get(
                    "ORCHESTRATOR_CODE_RUNTIME_PIP_CACHE_DIR",
                    defaults.pip_cache_dir,
                )
            ),
            session_ttl_seconds=_int_value(
                values.get("ORCHESTRATOR_CODE_RUNTIME_SESSION_TTL_SECONDS"),
                defaults.session_ttl_seconds,
                minimum=60,
            ),
            session_quota_bytes=_int_value(
                values.get("ORCHESTRATOR_CODE_RUNTIME_SESSION_QUOTA_BYTES"),
                defaults.session_quota_bytes,
                minimum=1024 * 1024,
            ),
            max_executions_per_session=_int_value(
                values.get(
                    "ORCHESTRATOR_CODE_RUNTIME_MAX_EXECUTIONS_PER_SESSION"
                ),
                defaults.max_executions_per_session,
            ),
            max_packages_per_install=_int_value(
                values.get(
                    "ORCHESTRATOR_CODE_RUNTIME_MAX_PACKAGES_PER_INSTALL"
                ),
                defaults.max_packages_per_install,
            ),
            package_blocklist=blocklist,
            max_capture_bytes=_int_value(
                values.get("ORCHESTRATOR_CODE_RUNTIME_MAX_CAPTURE_BYTES"),
                defaults.max_capture_bytes,
                minimum=1024,
            ),
            max_input_bytes=_int_value(
                values.get("ORCHESTRATOR_CODE_RUNTIME_MAX_INPUT_BYTES"),
                defaults.max_input_bytes,
                minimum=1024,
            ),
            max_artifact_bytes=_int_value(
                values.get("ORCHESTRATOR_CODE_RUNTIME_MAX_ARTIFACT_BYTES"),
                defaults.max_artifact_bytes,
                minimum=1024,
            ),
            max_output_bytes=_int_value(
                values.get("ORCHESTRATOR_CODE_RUNTIME_MAX_OUTPUT_BYTES"),
                defaults.max_output_bytes,
                minimum=1024,
            ),
            max_output_files=_int_value(
                values.get("ORCHESTRATOR_CODE_RUNTIME_MAX_OUTPUT_FILES"),
                defaults.max_output_files,
            ),
            max_concurrency=_int_value(
                values.get("ORCHESTRATOR_CODE_RUNTIME_MAX_CONCURRENCY"),
                defaults.max_concurrency,
            ),
        )


def get_code_runtime_settings() -> CodeRuntimeSettings:
    return CodeRuntimeSettings.from_sources(
        config=_current_app_config(),
        environ=os.environ,
    )


_config_provider = None


def set_config_provider(provider) -> None:
    """Register a zero-argument callable returning the host app's config mapping.

    Without one, settings come from the environment only.
    """
    global _config_provider
    _config_provider = provider


def _current_app_config() -> Mapping[str, object]:
    if _config_provider is None:
        return {}
    return _config_provider() or {}
