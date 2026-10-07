from __future__ import annotations

import ast
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from aibridge_exec.code_runtime.artifact_collector import ArtifactCollector
from aibridge_exec.code_runtime.artifact_materializer import (
    InputMaterializer,
    safe_workspace_child,
)
from aibridge_exec.code_runtime.backends.base import ExecutionBackend
from aibridge_exec.code_runtime.contracts import PolicyViolation
from aibridge_exec.code_runtime.models import (
    ExecutionRequest,
    ExecutionResult,
    PackageInstallRequest,
    PackageInstallResult,
)
from aibridge_exec.code_runtime.package_manager import PackageManager
from aibridge_exec.code_runtime.result_normalizer import ResultNormalizer
from aibridge_exec.code_runtime.profiles import resolve_environment_profile
from aibridge_exec.code_runtime.python_support import (
    validate_python_code,
    write_runtime_helpers,
)
from aibridge_exec.code_runtime.session_manager import SessionManager


class CodeExecutionRuntime:
    def __init__(
        self,
        *,
        backend: ExecutionBackend,
        session_manager: SessionManager | None = None,
        package_manager: PackageManager | None = None,
        input_materializer: InputMaterializer | None = None,
        artifact_collector: ArtifactCollector | None = None,
        result_normalizer: ResultNormalizer | None = None,
        environment_profile_overrides: dict[str, Any] | None = None,
    ):
        self.backend = backend
        self.session_manager = session_manager or SessionManager(backend)
        self.environment_profile_overrides = dict(
            environment_profile_overrides
            or getattr(package_manager, "environment_profile_overrides", {})
            or {}
        )
        self.package_manager = package_manager or PackageManager(
            self.session_manager,
            environment_profile_overrides=self.environment_profile_overrides,
        )
        self.input_materializer = input_materializer or InputMaterializer()
        self.artifact_collector = artifact_collector or ArtifactCollector()
        self.result_normalizer = result_normalizer or ResultNormalizer()

    def execute(
        self,
        request: ExecutionRequest,
        output_callback: Callable[[str, bytes], None] | None = None,
    ) -> ExecutionResult:
        execution_id = f"exec_{uuid.uuid4().hex}"
        started = time.monotonic()
        try:
            resolve_environment_profile(
                request.profile,
                overrides=self.environment_profile_overrides,
            )
        except ValueError as exc:
            error = self.result_normalizer.normalize(
                exit_code=None,
                stdout="",
                stderr="",
                policy_code="unsupported_profile",
                policy_message=str(exc),
            )
            return self._result(
                request, execution_id, started, None, error=error
            )
        if str(request.language or "").strip().lower() != "python":
            error = self.result_normalizer.normalize(
                exit_code=None,
                stdout="",
                stderr="",
                policy_code="unsupported_language",
                policy_message="Only Python is supported by this execution profile",
            )
            return self._result(
                request, execution_id, started, None, error=error
            )

        self.session_manager.cleanup_expired_if_due()
        with self.session_manager.locked(request.session_id):
            backend_result = None
            try:
                session = self.session_manager.get_or_create(
                    request.session_id, request.profile
                )
                self.session_manager.assert_execution_allowed(session)
                self.input_materializer.materialize(
                    session, request.input_artifacts
                )
                safe_workspace_child(session.workspace, "sandbox_tools.py")
                write_runtime_helpers(session.workspace)
                before = self.artifact_collector.snapshot(session)

                # Preserve the existing import boundary while leaving syntax
                # errors to Python so they become normal structured results.
                try:
                    validate_python_code(
                        request.code,
                        allowed_imports=(),
                    )
                except PolicyViolation as exc:
                    if "invalid syntax" not in str(exc).lower():
                        raise
                _validate_local_process_policy(request.code)

                script_path = safe_workspace_child(
                    session.workspace, f"scripts/{execution_id}.py"
                )
                script_path.write_text(request.code or "", encoding="utf-8")
                working_directory = self._working_directory(
                    session.workspace, request.working_directory
                )
                backend_result = self.backend.execute(
                    request.session_id,
                    [str(session.interpreter), "-u", "-s", str(script_path)],
                    max(1, int(request.timeout_seconds)),
                    str(working_directory),
                    {
                        **_execution_environment(session.workspace),
                        "PYTHONUNBUFFERED": "1",
                    },
                    memory_limit_mb=request.memory_limit_mb,
                    file_size_limit_bytes=self.artifact_collector.max_file_bytes,
                    operation_id=request.operation_id or None,
                    stdin_data=(
                        request.stdin.encode("utf-8")
                        if request.stdin is not None
                        else None
                    ),
                    output_callback=output_callback,
                )
                self.session_manager.record_execution(session, execution_id)
                artifacts, missing = self.artifact_collector.collect(
                    session,
                    before=before,
                    expected=request.expected_outputs,
                )
                error = self.result_normalizer.normalize(
                    exit_code=backend_result.exit_code,
                    stdout=backend_result.stdout,
                    stderr=backend_result.stderr,
                    timed_out=backend_result.timed_out,
                    cancelled=backend_result.cancelled,
                    missing_outputs=missing
                    if backend_result.exit_code == 0
                    and not backend_result.timed_out
                    else (),
                )
                self.session_manager.assert_workspace_quota(session)
                return ExecutionResult(
                    status="success" if error is None else "failed",
                    execution_id=execution_id,
                    session_id=request.session_id,
                    exit_code=backend_result.exit_code,
                    stdout=backend_result.stdout,
                    stderr=backend_result.stderr,
                    error=error,
                    artifacts=artifacts,
                    duration_ms=backend_result.duration_ms,
                    stdout_truncated=backend_result.stdout_truncated,
                    stderr_truncated=backend_result.stderr_truncated,
                )
            except PolicyViolation as exc:
                error = self.result_normalizer.normalize(
                    exit_code=None,
                    stdout="",
                    stderr="",
                    policy_code=exc.code or "policy_violation",
                    policy_message=str(exc),
                )
                return ExecutionResult(
                    status="failed",
                    execution_id=execution_id,
                    session_id=request.session_id,
                    exit_code=(
                        backend_result.exit_code if backend_result else None
                    ),
                    stdout=backend_result.stdout if backend_result else "",
                    stderr=backend_result.stderr if backend_result else "",
                    error=error,
                    artifacts=(),
                    duration_ms=(
                        backend_result.duration_ms
                        if backend_result
                        else int((time.monotonic() - started) * 1000)
                    ),
                    stdout_truncated=(
                        backend_result.stdout_truncated
                        if backend_result
                        else False
                    ),
                    stderr_truncated=(
                        backend_result.stderr_truncated
                        if backend_result
                        else False
                    ),
                )

    def install_packages(
        self,
        request: PackageInstallRequest,
        output_callback: Callable[[str, bytes], None] | None = None,
    ) -> PackageInstallResult:
        return self.package_manager.install(request, output_callback=output_callback)

    def list_installed_packages(
        self, session_id: str, profile: str = "python-data"
    ):
        return self.package_manager.list_installed(session_id, profile)

    @staticmethod
    def _working_directory(workspace: Path, value: str | None) -> Path:
        if not value:
            return workspace
        target = safe_workspace_child(workspace, value)
        if not target.is_dir():
            raise PolicyViolation(
                f"Working directory does not exist: {value}",
                code="invalid_working_directory",
            )
        return target

    @staticmethod
    def _result(
        request: ExecutionRequest,
        execution_id: str,
        started: float,
        exit_code: int | None,
        *,
        error,
    ) -> ExecutionResult:
        return ExecutionResult(
            status="failed",
            execution_id=execution_id,
            session_id=request.session_id,
            exit_code=exit_code,
            stdout="",
            stderr="",
            error=error,
            artifacts=(),
            duration_ms=int((time.monotonic() - started) * 1000),
        )


def _execution_environment(workspace: Path) -> dict[str, str]:
    allowed = {"PATH", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(
        {
            "PYTHONIOENCODING": "UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(workspace),
            "TMPDIR": str(workspace / "state"),
        }
    )
    return env


def _validate_local_process_policy(code: str) -> None:
    """Block obvious shell/dynamic-import escapes in the development backend."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return
    denied_names = {"__import__"}
    denied_attributes = {
        ("os", "system"),
        ("os", "popen"),
        ("os", "fork"),
        ("os", "forkpty"),
        ("os", "posix_spawn"),
        ("os", "posix_spawnp"),
    }
    denied_prefixes = ("exec", "spawn")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in denied_names:
            raise PolicyViolation(
                f"Generated Python call is not allowed: {node.func.id}",
                code="denied_python_call",
            )
        if not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value.id if isinstance(node.func.value, ast.Name) else ""
        attribute = node.func.attr
        if (owner, attribute) in denied_attributes or (
            owner == "os"
            and any(attribute.startswith(prefix) for prefix in denied_prefixes)
        ):
            raise PolicyViolation(
                f"Generated Python call is not allowed: {owner}.{attribute}",
                code="denied_python_call",
            )
