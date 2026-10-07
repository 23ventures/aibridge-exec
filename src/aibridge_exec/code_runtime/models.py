from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArtifactMount:
    artifact_ref: str
    target_path: str
    content: bytes = field(default=b"", repr=False)
    media_type: str | None = None


@dataclass(frozen=True)
class ExpectedOutput:
    path: str
    media_type: str | None = None


@dataclass(frozen=True)
class Artifact:
    path: str
    name: str
    media_type: str
    size_bytes: int
    content: bytes = field(default=b"", repr=False)
    artifact_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "artifact_ref": self.artifact_ref,
        }


@dataclass(frozen=True)
class ExecutionError:
    category: str
    message: str
    type: str | None = None
    module: str | None = None
    suggested_package: str | None = None
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "category": self.category,
                "type": self.type,
                "module": self.module,
                "suggested_package": self.suggested_package,
                "message": self.message,
                "retryable": self.retryable,
                "details": dict(self.details or {}),
            }.items()
            if value not in (None, {}, "")
        }


@dataclass(frozen=True)
class ExecutionRequest:
    session_id: str
    language: str
    code: str
    input_artifacts: tuple[ArtifactMount, ...] = ()
    expected_outputs: tuple[ExpectedOutput, ...] = ()
    timeout_seconds: int = 60
    working_directory: str | None = None
    profile: str = "python-data"
    memory_limit_mb: int = 1024
    operation_id: str = ""
    stdin: str | None = None


@dataclass(frozen=True)
class SandboxProvenance:
    """Worker-attested identity for a sandbox operation."""

    worker_id: str
    backend: str
    runtime_image_digest: str
    operation_id: str
    execution_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "worker_id": self.worker_id,
                "backend": self.backend,
                "runtime_image_digest": self.runtime_image_digest,
                "operation_id": self.operation_id,
                "execution_id": self.execution_id,
            }.items()
            if value
        }


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    execution_id: str
    session_id: str
    exit_code: int | None
    stdout: str
    stderr: str
    error: ExecutionError | None
    artifacts: tuple[Artifact, ...]
    duration_ms: int
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    sandbox_provenance: SandboxProvenance | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    def to_dict(self, *, include_artifact_content: bool = False) -> dict[str, Any]:
        artifacts = []
        for artifact in self.artifacts:
            item = artifact.to_dict()
            if include_artifact_content:
                item["content"] = artifact.content
            artifacts.append(item)
        return {
            "status": self.status,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error.to_dict() if self.error else None,
            "artifacts": artifacts,
            "duration_ms": self.duration_ms,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "sandbox_provenance": (
                self.sandbox_provenance.to_dict()
                if self.sandbox_provenance
                else None
            ),
        }


@dataclass(frozen=True)
class InstalledPackage:
    name: str
    version: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class PackageInstallError:
    category: str
    message: str
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "message": self.message,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class PackageInstallRequest:
    session_id: str
    packages: tuple[str, ...]
    profile: str = "python-data"
    timeout_seconds: int = 120
    force_reinstall: bool = False
    operation_id: str = ""


@dataclass(frozen=True)
class PackageInstallResult:
    status: str
    session_id: str
    installed: tuple[InstalledPackage, ...]
    already_installed: tuple[InstalledPackage, ...]
    error: PackageInstallError | None
    duration_ms: int
    sandbox_provenance: SandboxProvenance | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "session_id": self.session_id,
            "installed": [item.to_dict() for item in self.installed],
            "already_installed": [
                item.to_dict() for item in self.already_installed
            ],
            "error": self.error.to_dict() if self.error else None,
            "duration_ms": self.duration_ms,
            "sandbox_provenance": (
                self.sandbox_provenance.to_dict()
                if self.sandbox_provenance
                else None
            ),
        }
