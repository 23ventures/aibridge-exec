from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from aibridge_exec.code_runtime.backend_models import BackendSession
from aibridge_exec.code_runtime.contracts import PolicyViolation
from aibridge_exec.code_runtime.models import (
    InstalledPackage,
    PackageInstallError,
    PackageInstallRequest,
    PackageInstallResult,
)
from aibridge_exec.code_runtime.profiles import resolve_environment_profile
from aibridge_exec.code_runtime.session_manager import SessionManager


DEFAULT_PACKAGE_BLOCKLIST = {
    "ansible",
    "docker",
    "paramiko",
    "pexpect",
    "pyautogui",
}
logger = logging.getLogger(__name__)


class PackageManager:
    def __init__(
        self,
        session_manager: SessionManager,
        *,
        cache_dir: str | Path = "/tmp/aibridge-code-runtime-cache/pip",
        max_packages_per_install: int = 10,
        package_blocklist: set[str] | None = None,
        environment_profile_overrides: dict[str, Any] | None = None,
    ):
        self.session_manager = session_manager
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_packages_per_install = max(1, int(max_packages_per_install))
        self.package_blocklist = {
            canonicalize_name(item)
            for item in (
                package_blocklist
                if package_blocklist is not None
                else DEFAULT_PACKAGE_BLOCKLIST
            )
        }
        self.environment_profile_overrides = dict(
            environment_profile_overrides or {}
        )

    def ensure_environment(
        self, session_id: str, profile: str = "python-data"
    ) -> BackendSession:
        return self.session_manager.get_or_create(session_id, profile)

    def install(
        self,
        request: PackageInstallRequest,
        output_callback: Callable[[str, bytes], None] | None = None,
    ) -> PackageInstallResult:
        started = time.monotonic()
        session_id = request.session_id
        try:
            resolve_environment_profile(
                request.profile,
                overrides=self.environment_profile_overrides,
            )
        except ValueError as exc:
            return self._failed(
                session_id,
                started,
                category="package_blocked",
                message=str(exc),
                retryable=False,
            )
        try:
            requirements = self._validate_packages(request.packages)
        except ValueError as exc:
            logger.info(
                "Code runtime rejected package request session=%s error=%s",
                session_id,
                exc,
            )
            return self._failed(
                session_id,
                started,
                category="package_blocked",
                message=str(exc),
                retryable=False,
            )

        with self.session_manager.locked(session_id):
            session = self.ensure_environment(session_id, request.profile)
            try:
                self.session_manager.assert_workspace_quota(session)
            except PolicyViolation as exc:
                return self._failed(
                    session_id,
                    started,
                    category="disk_quota_exceeded",
                    message=str(exc),
                    retryable=False,
                )
            current = self._installed_versions(session)
            already = []
            pending = []
            for requirement in requirements:
                normalized = canonicalize_name(requirement.name)
                version = current.get(normalized)
                if not request.force_reinstall and version and (
                    not requirement.specifier
                    or requirement.specifier.contains(version, prereleases=True)
                ):
                    already.append(
                        InstalledPackage(name=requirement.name, version=version)
                    )
                else:
                    pending.append(requirement)

            if not pending:
                self._update_manifest(
                    session,
                    requirements=requirements,
                    resolved_packages=current,
                    profile=request.profile,
                )
                self._write_audit(
                    session,
                    {
                        "event": "package_install",
                        "status": "success",
                        "packages": [str(item) for item in requirements],
                        "already_installed": True,
                        "duration_ms": _elapsed_ms(started),
                    },
                )
                return PackageInstallResult(
                    status="success",
                    session_id=session_id,
                    installed=(),
                    already_installed=tuple(already),
                    error=None,
                    duration_ms=_elapsed_ms(started),
                )

            command = [
                str(session.interpreter),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--cache-dir",
                str(self.cache_dir),
                *(["--upgrade", "--force-reinstall"] if request.force_reinstall else []),
                *[str(item) for item in pending],
            ]
            completed = self.session_manager.backend.execute(
                session_id,
                command,
                max(1, int(request.timeout_seconds)),
                str(session.workspace),
                _pip_environment(),
                network_policy="package_registry",
                operation_id=request.operation_id or None,
                output_callback=output_callback,
            )
            if completed.cancelled:
                result = self._failed(
                    session_id,
                    started,
                    category="cancelled",
                    message="Package installation was cancelled",
                    retryable=True,
                )
                self._write_audit(
                    session,
                    {
                        "event": "package_install",
                        "status": "cancelled",
                        "packages": [str(item) for item in requirements],
                        "duration_ms": _elapsed_ms(started),
                    },
                )
                return result
            if completed.timed_out:
                result = self._failed(
                    session_id,
                    started,
                    category="installation_timeout",
                    message="Package installation exceeded its timeout",
                    retryable=True,
                )
                self._write_audit(
                    session,
                    {
                        "event": "package_install",
                        "status": "failed",
                        "packages": [str(item) for item in requirements],
                        "error_category": "installation_timeout",
                        "duration_ms": _elapsed_ms(started),
                    },
                )
                return result

            if completed.exit_code != 0:
                stderr = completed.stderr
                category = _classify_pip_error(stderr)
                result = self._failed(
                    session_id,
                    started,
                    category=category,
                    message=stderr[-3000:] or "Package installation failed",
                    retryable=category
                    in {"network_failure", "installation_timeout"},
                )
                self._write_audit(
                    session,
                    {
                        "event": "package_install",
                        "status": "failed",
                        "packages": [str(item) for item in requirements],
                        "error_category": category,
                        "duration_ms": _elapsed_ms(started),
                    },
                )
                return result

            updated = self._installed_versions(session)
            installed = tuple(
                InstalledPackage(
                    name=requirement.name,
                    version=updated.get(canonicalize_name(requirement.name), "unknown"),
                )
                for requirement in pending
            )
            self._update_manifest(
                session,
                requirements=requirements,
                resolved_packages=updated,
                profile=request.profile,
            )
            self._write_audit(
                session,
                {
                    "event": "package_install",
                    "status": "success",
                    "packages": [str(item) for item in requirements],
                    "force_reinstall": request.force_reinstall,
                    "duration_ms": _elapsed_ms(started),
                },
            )
            try:
                self.session_manager.assert_workspace_quota(session)
            except PolicyViolation as exc:
                return self._failed(
                    session_id,
                    started,
                    category="disk_quota_exceeded",
                    message=str(exc),
                    retryable=False,
                )
            return PackageInstallResult(
                status="success",
                session_id=session_id,
                installed=installed,
                already_installed=tuple(already),
                error=None,
                duration_ms=_elapsed_ms(started),
            )

    def list_installed(
        self, session_id: str, profile: str = "python-data"
    ) -> list[InstalledPackage]:
        with self.session_manager.locked(session_id):
            session = self.ensure_environment(session_id, profile)
            return [
                InstalledPackage(name=name, version=version)
                for name, version in sorted(self._installed_versions(session).items())
            ]

    def _validate_packages(
        self, packages: tuple[str, ...]
    ) -> tuple[Requirement, ...]:
        if not packages:
            raise ValueError("At least one package is required")
        if len(packages) > self.max_packages_per_install:
            raise ValueError(
                f"At most {self.max_packages_per_install} packages may be installed at once"
            )
        parsed = []
        seen = set()
        for raw in packages:
            value = str(raw or "").strip()
            if (
                not value
                or "://" in value
                or value.startswith((".", "/", "-"))
                or " @ " in value
                or ";" in value
                or "[" in value
                or "]" in value
                or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._-]*(?:\s*(?:===|==|!=|~=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+(?:\s*,\s*(?:===|==|!=|~=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+)*)?",
                    value,
                )
            ):
                raise ValueError(f"Unsupported package specification: {value!r}")
            try:
                requirement = Requirement(value)
            except InvalidRequirement as exc:
                raise ValueError(f"Invalid package specification: {value!r}") from exc
            normalized = canonicalize_name(requirement.name)
            if normalized in self.package_blocklist:
                raise ValueError(f"Package is blocked by runtime policy: {requirement.name}")
            if normalized not in seen:
                parsed.append(requirement)
                seen.add(normalized)
        return tuple(parsed)

    def _installed_versions(self, session: BackendSession) -> dict[str, str]:
        completed = self.session_manager.backend.execute(
            session.session_id,
            [
                str(session.interpreter),
                "-m",
                "pip",
                "list",
                "--format=json",
                "--disable-pip-version-check",
            ],
            30,
            str(session.workspace),
            _pip_environment(),
            network_policy="none",
        )
        if completed.exit_code != 0 or completed.timed_out:
            return {}
        try:
            packages = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {}
        return {
            canonicalize_name(str(item.get("name") or "")): str(
                item.get("version") or ""
            )
            for item in packages
            if item.get("name") and item.get("version")
        }

    def _update_manifest(
        self,
        session: BackendSession,
        requirements: tuple[Requirement, ...],
        resolved_packages: dict[str, str],
        profile: str,
    ) -> None:
        manifest_path = session.root / "package-manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        packages = dict(manifest.get("packages") or {})
        requested_packages = dict(manifest.get("requested_packages") or {})
        for requirement in requirements:
            normalized = canonicalize_name(requirement.name)
            version = resolved_packages.get(normalized)
            if version:
                packages[requirement.name] = version
            requested_packages[requirement.name] = (
                str(requirement.specifier) or "*"
            )
        backend_metadata = (
            self.session_manager.backend.environment_manifest_metadata()
        )
        manifest = {
            "schema_version": 1,
            "runtime_image_digest": backend_metadata["runtime_image_digest"],
            "python_version": self._interpreter_value(
                session,
                "import platform; print(platform.python_version())",
            ),
            "python_abi": self._interpreter_value(
                session,
                "import sys; print(sys.implementation.cache_tag or '')",
            ),
            "profile": profile,
            "profile_version": "1",
            "base_package_lock_digest": backend_metadata[
                "base_package_lock_digest"
            ],
            "requested_packages": dict(sorted(requested_packages.items())),
            "resolved_packages": dict(sorted(resolved_packages.items())),
            "package_index": backend_metadata["package_index"],
            # Compatibility field for existing session inspection consumers.
            "packages": dict(sorted(packages.items())),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        (session.root / "requirements.txt").write_text(
            "".join(f"{name}=={version}\n" for name, version in sorted(packages.items())),
            encoding="utf-8",
        )
        (session.root / "requirements.lock").write_text(
            "".join(
                f"{name}=={version}\n"
                for name, version in sorted(resolved_packages.items())
            ),
            encoding="utf-8",
        )

    def _interpreter_value(self, session: BackendSession, code: str) -> str:
        completed = self.session_manager.backend.execute(
            session.session_id,
            [str(session.interpreter), "-c", code],
            10,
            str(session.workspace),
            {"PYTHONIOENCODING": "UTF-8"},
            network_policy="none",
        )
        if completed.exit_code != 0 or completed.timed_out:
            return ""
        return completed.stdout.strip()

    @staticmethod
    def _write_audit(session: BackendSession, record: dict) -> None:
        path = session.root / "execution-history" / "package-installs.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _failed(
        session_id: str,
        started: float,
        *,
        category: str,
        message: str,
        retryable: bool,
    ) -> PackageInstallResult:
        return PackageInstallResult(
            status="failed",
            session_id=session_id,
            installed=(),
            already_installed=(),
            error=PackageInstallError(
                category=category,
                message=message,
                retryable=retryable,
            ),
            duration_ms=_elapsed_ms(started),
        )


def _pip_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_TRUSTED_HOST",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update({"PYTHONIOENCODING": "UTF-8", "PIP_NO_INPUT": "1"})
    return env


def _classify_pip_error(stderr: str) -> str:
    lowered = stderr.lower()
    if any(
        marker in lowered
        for marker in (
            "temporary failure in name resolution",
            "name or service not known",
            "failed to establish a new connection",
            "connection error",
            "connection refused",
            "read timed out",
            "proxyerror",
        )
    ):
        return "network_failure"
    if "no matching distribution found" in lowered or "could not find a version" in lowered:
        return "package_not_found"
    if "resolutionimpossible" in lowered or "conflicting dependencies" in lowered:
        return "version_conflict"
    if "requires-python" in lowered or "python_requires" in lowered:
        return "unsupported_python_version"
    if "no space left on device" in lowered:
        return "disk_quota_exceeded"
    return "build_failure"


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
