from __future__ import annotations

from pathlib import Path, PurePosixPath

from aibridge_exec.code_runtime.backend_models import BackendSession
from aibridge_exec.code_runtime.contracts import PolicyViolation
from aibridge_exec.code_runtime.models import ArtifactMount


class InputMaterializer:
    def __init__(self, *, max_input_bytes: int = 100 * 1024 * 1024):
        self.max_input_bytes = max(1024, int(max_input_bytes))

    def materialize(
        self,
        session: BackendSession,
        mounts: tuple[ArtifactMount, ...],
    ) -> list[str]:
        mounted = []
        total = 0
        for mount in mounts:
            content = mount.content or b""
            total += len(content)
            if total > self.max_input_bytes:
                raise PolicyViolation(
                    "Code execution input artifact limit exceeded",
                    code="input_quota_exceeded",
                )
            relative = normalized_workspace_path(
                mount.target_path or f"inputs/{Path(mount.artifact_ref).name}",
                default_directory="inputs",
            )
            if relative.parts[0] not in {"input", "inputs"}:
                raise PolicyViolation(
                    f"Input artifact target must be under inputs/: {mount.target_path}",
                    code="unsafe_input_path",
                )
            if relative.parts[0] == "input":
                relative = PurePosixPath("inputs", *relative.parts[1:])
            target = safe_workspace_child(session.workspace, relative.as_posix())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            mounted.append(f"inputs/{target.relative_to(session.workspace / 'inputs').as_posix()}")
        return mounted


def normalized_workspace_path(
    value: str,
    *,
    default_directory: str | None = None,
) -> PurePosixPath:
    raw = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise PolicyViolation(
            f"Unsafe sandbox-relative path: {value}",
            code="unsafe_workspace_path",
        )
    if default_directory and len(path.parts) == 1:
        path = PurePosixPath(default_directory, path)
    return path


def safe_workspace_child(workspace: Path, relative_path: str) -> Path:
    relative = normalized_workspace_path(relative_path)
    target = (workspace / relative.as_posix()).resolve()
    base = workspace.resolve()
    if target != base and base not in target.parents:
        raise PolicyViolation(
            f"Path escapes code session workspace: {relative_path}",
            code="unsafe_workspace_path",
        )
    return target
