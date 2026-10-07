from __future__ import annotations

import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aibridge_exec.code_runtime.backends.base import ExecutionBackend
from aibridge_exec.code_runtime.contracts import PolicyViolation
from aibridge_exec.code_runtime.backend_models import BackendSession


class SessionManager:
    def __init__(
        self,
        backend: ExecutionBackend,
        *,
        ttl_seconds: int = 24 * 60 * 60,
        max_workspace_bytes: int = 250 * 1024 * 1024,
        max_executions_per_session: int = 1000,
    ):
        self.backend = backend
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_workspace_bytes = max(1024 * 1024, int(max_workspace_bytes))
        self.max_executions_per_session = max(1, int(max_executions_per_session))
        self._locks_guard = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}
        self._last_cleanup_at = 0.0

    @contextmanager
    def locked(self, session_id: str):
        with self._locks_guard:
            lock = self._locks.setdefault(session_id, threading.RLock())
        with lock:
            with self._filesystem_lock(session_id):
                yield

    @contextmanager
    def _filesystem_lock(self, session_id: str):
        session_root = getattr(self.backend, "session_root", None)
        if not callable(session_root):
            yield
            return
        root = session_root(session_id)
        if root.is_symlink():
            raise PolicyViolation(
                "Code session root was replaced by a symlink",
                code="unsafe_workspace_path",
            )
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = root / ".session.lock"
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(str(lock_path), flags, 0o600)
        try:
            try:
                import fcntl
            except ImportError:
                yield
                return
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def get_or_create(
        self, session_id: str, profile: str = "python-data"
    ) -> BackendSession:
        session = self.backend.create_session(session_id, profile)
        metadata = self._read_metadata(session)
        now = datetime.now(timezone.utc)
        if not metadata:
            metadata = {
                "session_id": session_id,
                "profile": profile,
                "created_at": now.isoformat(),
                "execution_count": 0,
            }
        metadata["profile"] = profile
        metadata["last_accessed_at"] = now.isoformat()
        metadata["expires_at"] = (
            now + timedelta(seconds=self.ttl_seconds)
        ).isoformat()
        self._write_metadata(session, metadata)
        return session

    def record_execution(self, session: BackendSession, execution_id: str) -> None:
        metadata = self._read_metadata(session)
        count = int(metadata.get("execution_count") or 0) + 1
        metadata["execution_count"] = count
        metadata["last_execution_id"] = execution_id
        metadata["last_accessed_at"] = datetime.now(timezone.utc).isoformat()
        self._write_metadata(session, metadata)

    def assert_execution_allowed(self, session: BackendSession) -> None:
        metadata = self._read_metadata(session)
        if int(metadata.get("execution_count") or 0) >= self.max_executions_per_session:
            raise PolicyViolation(
                "Code session execution limit exceeded",
                code="execution_limit_exceeded",
            )
        self.assert_workspace_quota(session)

    def assert_workspace_quota(self, session: BackendSession) -> None:
        used = directory_size(session.root)
        if used > self.max_workspace_bytes:
            raise PolicyViolation(
                "Code session workspace quota exceeded",
                code="disk_quota_exceeded",
            )

    def cleanup_expired_if_due(self, *, interval_seconds: int = 300) -> list[str]:
        now = time.monotonic()
        with self._locks_guard:
            if now - self._last_cleanup_at < max(1, interval_seconds):
                return []
            self._last_cleanup_at = now
        return self.cleanup_expired()

    def cleanup_expired(self, *, now: datetime | None = None) -> list[str]:
        current = now or datetime.now(timezone.utc)
        removed = []
        base_dir = getattr(self.backend, "base_dir", None)
        if not base_dir:
            return removed
        for root in Path(base_dir).glob("session-*"):
            if root.is_symlink():
                continue
            metadata_path = root / "session.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                expires = datetime.fromisoformat(str(metadata["expires_at"]))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires <= current:
                shutil.rmtree(root)
                removed.append(root.name)
        return removed

    def destroy(self, session_id: str) -> None:
        self.backend.destroy_session(session_id)

    @staticmethod
    def _metadata_path(session: BackendSession) -> Path:
        return session.root / "session.json"

    def _read_metadata(self, session: BackendSession) -> dict:
        try:
            value = json.loads(
                self._metadata_path(session).read_text(encoding="utf-8")
            )
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_metadata(self, session: BackendSession, metadata: dict) -> None:
        target = self._metadata_path(session)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(target)


def directory_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total
