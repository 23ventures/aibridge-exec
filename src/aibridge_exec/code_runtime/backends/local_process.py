from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Callable
import venv
from pathlib import Path

from aibridge_exec.code_runtime.backends.base import ExecutionBackend
from aibridge_exec.code_runtime.contracts import PolicyViolation
from aibridge_exec.code_runtime.backend_models import (
    BackendExecutionResult,
    BackendSession,
)


class LocalProcessBackend(ExecutionBackend):
    """Development backend. This is process isolation, not a security sandbox."""

    def __init__(
        self,
        *,
        base_dir: str | Path = "/tmp/aibridge-code-sessions",
        max_output_bytes: int = 1_000_000,
        inherit_system_packages: bool = False,
        max_concurrency: int = 4,
    ):
        self.base_dir = Path(base_dir).resolve()
        self.max_output_bytes = max(1024, int(max_output_bytes))
        self.inherit_system_packages = bool(inherit_system_packages)
        self._execution_slots = threading.BoundedSemaphore(
            max(1, int(max_concurrency))
        )
        self._active_guard = threading.Lock()
        self._active_operations: dict[str, subprocess.Popen] = {}
        self._cancel_requested: set[str] = set()
        self._cancelled_operations: set[str] = set()
        self.base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def environment_manifest_metadata(self) -> dict[str, str]:
        return {
            "runtime_image_digest": "local-process",
            "base_package_lock_digest": "",
            "package_index": "configured-registry",
        }

    def session_root(self, session_id: str) -> Path:
        raw = str(session_id or "").strip()
        if not raw:
            raise PolicyViolation("Code session_id is required", code="invalid_session")
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.base_dir / f"session-{digest}"

    def create_session(self, session_id: str, profile: str) -> BackendSession:
        root = self.session_root(session_id)
        if root.is_symlink():
            raise PolicyViolation(
                "Code session root was replaced by a symlink",
                code="unsafe_workspace_path",
            )
        workspace = root / "workspace"
        if workspace.is_symlink():
            raise PolicyViolation(
                "Code session workspace was replaced by a symlink",
                code="unsafe_workspace_path",
            )
        for name in ("inputs", "outputs", "scripts", "state"):
            directory = workspace / name
            if directory.is_symlink():
                raise PolicyViolation(
                    f"Code session directory was replaced by a symlink: {name}",
                    code="unsafe_workspace_path",
                )
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        for alias, target in (("input", "inputs"), ("output", "outputs")):
            alias_path = workspace / alias
            if not alias_path.exists():
                try:
                    alias_path.symlink_to(target, target_is_directory=True)
                except OSError:
                    alias_path.mkdir(exist_ok=True)

        (root / "execution-history").mkdir(parents=True, exist_ok=True)
        venv_dir = root / "venv"
        interpreter = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not interpreter.exists():
            venv.EnvBuilder(
                with_pip=True,
                clear=False,
                symlinks=os.name != "nt",
                system_site_packages=self.inherit_system_packages,
            ).create(str(venv_dir))
        else:
            _set_system_site_packages(
                venv_dir,
                enabled=self.inherit_system_packages,
            )
        requirements_path = root / "requirements.txt"
        if not requirements_path.exists():
            requirements_path.write_text("", encoding="utf-8")
        requirements_lock_path = root / "requirements.lock"
        if not requirements_lock_path.exists():
            requirements_lock_path.write_text("", encoding="utf-8")
        manifest_path = root / "package-manifest.json"
        if not manifest_path.exists():
            manifest = {}
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
        manifest.update(
            {
                "schema_version": 1,
                "runtime_image_digest": "local-process",
                "python_version": _interpreter_version(interpreter),
                "python_abi": _interpreter_abi(interpreter),
                "profile": profile,
                "profile_version": str(manifest.get("profile_version") or "1"),
                "base_package_lock_digest": str(
                    manifest.get("base_package_lock_digest") or ""
                ),
                "requested_packages": dict(
                    manifest.get("requested_packages") or {}
                ),
                "resolved_packages": dict(
                    manifest.get("resolved_packages") or {}
                ),
                "package_index": str(manifest.get("package_index") or ""),
                "packages": dict(manifest.get("packages") or {}),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return BackendSession(
            session_id=session_id,
            root=root,
            workspace=workspace,
            interpreter=interpreter,
            profile=profile,
        )

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
        session = self.create_session(
            session_id,
            profile=self._session_profile(session_id),
        )
        cwd = Path(working_directory).resolve()
        if cwd != session.workspace and session.workspace not in cwd.parents:
            raise PolicyViolation(
                "Working directory must remain inside the code session workspace",
                code="unsafe_working_directory",
            )

        history = session.root / "execution-history"
        stamp = f"{time.time_ns()}-{os.getpid()}"
        stdout_path = history / f"{stamp}.stdout"
        stderr_path = history / f"{stamp}.stderr"
        started = time.monotonic()
        timed_out = False
        terminated = False
        cancelled = False
        return_code: int | None = None
        acquired = self._execution_slots.acquire(
            timeout=max(1, int(timeout_seconds))
        )
        if not acquired:
            raise PolicyViolation(
                "Code runtime concurrency limit exceeded",
                code="concurrency_limit_exceeded",
            )
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                env=dict(environment or {}),
                stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=os.name != "nt",
                preexec_fn=_resource_limiter(
                    memory_limit_mb,
                    file_size_limit_bytes=file_size_limit_bytes,
                ),
            )
            if operation_id:
                with self._active_guard:
                    self._active_operations[operation_id] = process
                    cancel_now = operation_id in self._cancel_requested
                if cancel_now:
                    _terminate_process_group(process)
            stdout_buffer = bytearray()
            stderr_buffer = bytearray()
            stdout_state = {"truncated": False}
            stderr_state = {"truncated": False}
            readers = [
                threading.Thread(
                    target=_consume_bounded,
                    args=(
                        process.stdout,
                        stdout_buffer,
                        self.max_output_bytes,
                        stdout_state,
                        "stdout",
                        output_callback,
                    ),
                    daemon=True,
                ),
                threading.Thread(
                    target=_consume_bounded,
                    args=(
                        process.stderr,
                        stderr_buffer,
                        self.max_output_bytes,
                        stderr_state,
                        "stderr",
                        output_callback,
                    ),
                    daemon=True,
                ),
            ]
            for reader in readers:
                reader.start()
            if stdin_data is not None and process.stdin is not None:
                try:
                    process.stdin.write(stdin_data)
                    process.stdin.flush()
                finally:
                    process.stdin.close()
            try:
                return_code = process.wait(timeout=max(1, int(timeout_seconds)))
            except subprocess.TimeoutExpired:
                timed_out = True
                terminated = True
                _terminate_process_group(process)
                try:
                    return_code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    return_code = process.wait()
            for reader in readers:
                reader.join(timeout=5)
        finally:
            if operation_id:
                with self._active_guard:
                    self._active_operations.pop(operation_id, None)
                    self._cancel_requested.discard(operation_id)
                    cancelled = operation_id in self._cancelled_operations
                    self._cancelled_operations.discard(operation_id)
            self._execution_slots.release()

        stdout_truncated = stdout_state["truncated"]
        stderr_truncated = stderr_state["truncated"]
        stdout = _decode_captured(
            bytes(stdout_buffer), truncated=stdout_truncated
        )
        stderr = _decode_captured(
            bytes(stderr_buffer), truncated=stderr_truncated
        )
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        duration_ms = int((time.monotonic() - started) * 1000)
        audit = {
            "session_id_hash": hashlib.sha256(session_id.encode("utf-8")).hexdigest(),
            "command": [Path(command[0]).name, *command[1:]],
            "working_directory": str(cwd.relative_to(session.workspace) or "."),
            "exit_code": return_code,
            "timed_out": timed_out,
            "cancelled": cancelled,
            "duration_ms": duration_ms,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
        (history / f"{stamp}.json").write_text(
            json.dumps(audit, sort_keys=True), encoding="utf-8"
        )
        return BackendExecutionResult(
            exit_code=return_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            timed_out=timed_out,
            terminated=terminated,
            cancelled=cancelled,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    def cancel_operation(self, operation_id: str) -> bool:
        value = str(operation_id or "").strip()
        if not value:
            return False
        with self._active_guard:
            process = self._active_operations.get(value)
            self._cancel_requested.add(value)
            self._cancelled_operations.add(value)
        if process is not None:
            _terminate_process_group(process)
        return True

    def destroy_session(self, session_id: str) -> None:
        root = self.session_root(session_id)
        if root.exists():
            shutil.rmtree(root)

    def _session_profile(self, session_id: str) -> str:
        manifest_path = self.session_root(session_id) / "package-manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return str(manifest.get("profile") or "python-data")
        except (OSError, json.JSONDecodeError):
            return "python-data"


def _consume_bounded(
    stream,
    buffer: bytearray,
    limit: int,
    state: dict,
    stream_name: str = "stdout",
    output_callback: Callable[[str, bytes], None] | None = None,
) -> None:
    if stream is None:
        return
    try:
        while True:
            read = getattr(stream, "read1", stream.read)
            chunk = read(16 * 1024)
            if not chunk:
                break
            if output_callback is not None:
                output_callback(stream_name, chunk)
            if not state["truncated"] and len(buffer) + len(chunk) <= limit:
                buffer.extend(chunk)
                continue
            head_limit = limit // 2
            tail_limit = limit - head_limit
            if state["truncated"]:
                head = bytes(buffer[:head_limit])
                combined_tail = bytes(buffer[head_limit:]) + chunk
            else:
                combined = bytes(buffer) + chunk
                head = combined[:head_limit]
                combined_tail = combined[head_limit:]
            buffer.clear()
            buffer.extend(head)
            buffer.extend(combined_tail[-tail_limit:])
            state["truncated"] = True
    finally:
        stream.close()


def _decode_captured(content: bytes, *, truncated: bool) -> str:
    text = content.decode("utf-8", errors="replace")
    if truncated:
        text += "\n[output truncated by code runtime]"
    return text


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=2)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass


def _resource_limiter(
    memory_limit_mb: int | None,
    *,
    file_size_limit_bytes: int | None = None,
):
    if os.name == "nt" or (not memory_limit_mb and not file_size_limit_bytes):
        return None
    try:
        import resource
    except ImportError:
        return None

    def apply_limits():
        if memory_limit_mb:
            memory_bytes = max(64, int(memory_limit_mb)) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        if file_size_limit_bytes:
            file_bytes = max(1024, int(file_size_limit_bytes))
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply_limits


def _set_system_site_packages(venv_dir: Path, *, enabled: bool) -> None:
    """Keep existing session environments aligned with backend isolation policy."""

    config_path = venv_dir / "pyvenv.cfg"
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    desired = f"include-system-site-packages = {'true' if enabled else 'false'}"
    updated = []
    replaced = False
    for line in lines:
        if line.strip().lower().startswith("include-system-site-packages"):
            updated.append(desired)
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(desired)
    content = "\n".join(updated) + "\n"
    if content != "\n".join(lines) + "\n":
        temporary = config_path.with_suffix(".cfg.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(config_path)


def _interpreter_version(interpreter: Path) -> str:
    try:
        completed = subprocess.run(
            [
                str(interpreter),
                "-c",
                "import platform; print(platform.python_version())",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
            env={"PYTHONIOENCODING": "UTF-8"},
        )
        version = completed.stdout.decode(
            "utf-8", errors="replace"
        ).strip()
        if completed.returncode == 0 and version:
            return version
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.python_version()


def _interpreter_abi(interpreter: Path) -> str:
    try:
        completed = subprocess.run(
            [
                str(interpreter),
                "-c",
                "import sys; print(sys.implementation.cache_tag or '')",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
            env={"PYTHONIOENCODING": "UTF-8"},
        )
        abi = completed.stdout.decode("utf-8", errors="replace").strip()
        if completed.returncode == 0 and abi:
            return abi
    except (OSError, subprocess.SubprocessError):
        pass
    return getattr(sys.implementation, "cache_tag", "") or ""
