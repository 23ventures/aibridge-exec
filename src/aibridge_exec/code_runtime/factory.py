from __future__ import annotations

import threading

from aibridge_exec.code_runtime.artifact_collector import ArtifactCollector
from aibridge_exec.code_runtime.artifact_materializer import InputMaterializer
from aibridge_exec.code_runtime.backends.local_process import LocalProcessBackend
from aibridge_exec.code_runtime.package_manager import PackageManager
from aibridge_exec.code_runtime.profiles import parse_environment_profile_overrides
from aibridge_exec.code_runtime.runtime import CodeExecutionRuntime
from aibridge_exec.code_runtime.session_manager import SessionManager
from aibridge_exec.code_runtime.settings import (
    CodeRuntimeSettings,
    get_code_runtime_settings,
)


_RUNTIME_CACHE: dict[CodeRuntimeSettings, CodeExecutionRuntime] = {}
_RUNTIME_CACHE_LOCK = threading.Lock()


def get_code_execution_runtime(
    settings: CodeRuntimeSettings | None = None,
) -> CodeExecutionRuntime:
    settings = settings or get_code_runtime_settings()
    with _RUNTIME_CACHE_LOCK:
        cached = _RUNTIME_CACHE.get(settings)
        if cached is not None:
            return cached
        runtime = _build_code_execution_runtime(settings)
        _RUNTIME_CACHE[settings] = runtime
        return runtime


def reset_code_execution_runtimes() -> None:
    """Drop cached runtime compositions, primarily for tests."""

    with _RUNTIME_CACHE_LOCK:
        _RUNTIME_CACHE.clear()


def _build_code_execution_runtime(
    settings: CodeRuntimeSettings,
) -> CodeExecutionRuntime:
    profile_overrides = parse_environment_profile_overrides(
        settings.environment_profiles
    )
    backend = LocalProcessBackend(
        base_dir=settings.sessions_dir,
        max_output_bytes=settings.max_capture_bytes,
        max_concurrency=settings.max_concurrency,
    )
    sessions = SessionManager(
        backend,
        ttl_seconds=settings.session_ttl_seconds,
        max_workspace_bytes=settings.session_quota_bytes,
        max_executions_per_session=settings.max_executions_per_session,
    )
    package_manager = PackageManager(
        sessions,
        cache_dir=settings.pip_cache_dir,
        max_packages_per_install=settings.max_packages_per_install,
        package_blocklist=set(settings.package_blocklist),
        environment_profile_overrides=profile_overrides,
    )
    return CodeExecutionRuntime(
        backend=backend,
        session_manager=sessions,
        package_manager=package_manager,
        input_materializer=InputMaterializer(
            max_input_bytes=settings.max_input_bytes
        ),
        artifact_collector=ArtifactCollector(
            max_file_bytes=settings.max_artifact_bytes,
            max_total_bytes=settings.max_output_bytes,
            max_files=settings.max_output_files,
        ),
        environment_profile_overrides=profile_overrides,
    )
