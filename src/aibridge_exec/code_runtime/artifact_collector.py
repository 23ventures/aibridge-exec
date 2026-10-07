from __future__ import annotations

import mimetypes
from pathlib import Path

from aibridge_exec.code_runtime.artifact_materializer import (
    normalized_workspace_path,
    safe_workspace_child,
)
from aibridge_exec.code_runtime.backend_models import BackendSession
from aibridge_exec.code_runtime.contracts import PolicyViolation
from aibridge_exec.code_runtime.models import Artifact, ExpectedOutput


class ArtifactCollector:
    def __init__(
        self,
        *,
        max_file_bytes: int = 50 * 1024 * 1024,
        max_total_bytes: int = 100 * 1024 * 1024,
        max_files: int = 50,
    ):
        self.max_file_bytes = max(1024, int(max_file_bytes))
        self.max_total_bytes = max(1024, int(max_total_bytes))
        self.max_files = max(1, int(max_files))

    def snapshot(self, session: BackendSession) -> dict[str, tuple[int, int]]:
        return {
            path.relative_to(session.workspace / "outputs").as_posix(): (
                path.stat().st_mtime_ns,
                path.stat().st_size,
            )
            for path in (session.workspace / "outputs").rglob("*")
            if path.is_file() and not path.is_symlink()
        }

    def collect(
        self,
        session: BackendSession,
        *,
        before: dict[str, tuple[int, int]],
        expected: tuple[ExpectedOutput, ...],
    ) -> tuple[tuple[Artifact, ...], tuple[str, ...]]:
        expected_by_path = {
            self._output_relative(item.path): item for item in expected
        }
        missing = tuple(
            path
            for path in expected_by_path
            if not safe_workspace_child(
                session.workspace, f"outputs/{path}"
            ).is_file()
        )

        candidates: dict[str, Path] = {}
        output_dir = session.workspace / "outputs"
        for path in output_dir.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(output_dir).as_posix()
            state = (path.stat().st_mtime_ns, path.stat().st_size)
            if before.get(relative) != state or relative in expected_by_path:
                candidates[relative] = path

        if len(candidates) > self.max_files:
            raise PolicyViolation(
                "Code execution generated too many output files",
                code="output_file_limit_exceeded",
            )

        total = 0
        artifacts = []
        for relative, path in sorted(candidates.items()):
            size = path.stat().st_size
            if size > self.max_file_bytes:
                raise PolicyViolation(
                    f"Generated output exceeds file size limit: {relative}",
                    code="output_file_too_large",
                )
            total += size
            if total > self.max_total_bytes:
                raise PolicyViolation(
                    "Generated outputs exceed total size limit",
                    code="output_quota_exceeded",
                )
            expected_output = expected_by_path.get(relative)
            media_type = (
                expected_output.media_type
                if expected_output and expected_output.media_type
                else mimetypes.guess_type(path.name)[0]
                or "application/octet-stream"
            )
            artifacts.append(
                Artifact(
                    path=f"outputs/{relative}",
                    name=path.name,
                    media_type=media_type,
                    size_bytes=size,
                    content=path.read_bytes(),
                )
            )
        return tuple(artifacts), missing

    @staticmethod
    def _output_relative(value: str) -> str:
        path = normalized_workspace_path(
            value, default_directory="outputs"
        )
        if path.parts[0] not in {"output", "outputs"}:
            raise PolicyViolation(
                f"Expected output must be under outputs/: {value}",
                code="unsafe_output_path",
            )
        return PureOutputPath(path)


def PureOutputPath(path) -> str:
    return "/".join(path.parts[1:])
