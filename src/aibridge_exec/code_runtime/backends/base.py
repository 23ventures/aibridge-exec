from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from aibridge_exec.code_runtime.backend_models import (
    BackendExecutionResult,
    BackendSession,
)


class ExecutionBackend(ABC):
    def environment_manifest_metadata(self) -> dict[str, str]:
        """Return stable backend identity without exposing implementation paths."""

        return {
            "runtime_image_digest": "unknown",
            "base_package_lock_digest": "",
            "package_index": "",
        }

    @abstractmethod
    def create_session(self, session_id: str, profile: str) -> BackendSession:
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        session_id: str,
        command: list[str],
        timeout_seconds: int,
        working_directory: str,
        environment: dict[str, str],
        *,
        memory_limit_mb: int | None = None,
        file_size_limit_bytes: int | None = None,
        network_policy: str = "none",
        operation_id: str | None = None,
        stdin_data: bytes | None = None,
        output_callback: Callable[[str, bytes], None] | None = None,
    ) -> BackendExecutionResult:
        raise NotImplementedError

    def cancel_operation(self, operation_id: str) -> bool:
        """Request cancellation of an active operation, if supported."""

        return False

    @abstractmethod
    def destroy_session(self, session_id: str) -> None:
        raise NotImplementedError
