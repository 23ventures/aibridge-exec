"""Shared HTTP plumbing for the media providers.

Every provider takes the caller's key per instance and an optional ``httpx.Client`` (tests
inject a ``MockTransport``). Without one, each call opens and closes its own short-lived
client, so instances need no cleanup. Errors are raised as ``ProviderError`` with the
provider's status and message and the key redacted.
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Any, Callable, Optional, Tuple

import httpx

from ..providers.errors import ProviderError, code_for_status, redact

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = 60.0


def detect_mime(data: bytes, default: str = "image/jpeg") -> str:
    """Image MIME type from magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"GIF8":
        return "image/gif"
    return default


def error_message(response: httpx.Response) -> str:
    """Best-effort human message from a provider error body."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str) and err:
            return err
        for key in ("message", "detail", "failure_reason", "name"):
            if body.get(key):
                return str(body[key])
    return response.text[:500]


class MediaProvider:
    """Base for image and video providers."""

    provider_name: str = "unknown"
    #: Label used in error messages, e.g. "OpenAI".
    label: str = "Provider"
    timeout: float = 120.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        http: Optional[httpx.Client] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key
        self._http = http
        self._sleep = sleep

    # ------------------------------------------------------------ http
    def _require_key(self) -> str:
        if not self.api_key:
            raise ProviderError("auth", f"{self.label} API key is required", provider=self.provider_name)
        return self.api_key

    def _send(self, method: str, url: str, *, timeout: Optional[float] = None, **kwargs: Any) -> httpx.Response:
        """Send one request; transport failures become ``ProviderError``."""
        timeout = self.timeout if timeout is None else timeout
        # Provider download URLs (Veo, Luma) redirect to storage hosts.
        kwargs.setdefault("follow_redirects", True)
        try:
            if self._http is not None:
                return self._http.request(method, url, timeout=timeout, **kwargs)
            with httpx.Client(timeout=timeout) as client:
                return client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "timeout", f"{self.label} request timed out", provider=self.provider_name
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                "network",
                redact(f"{self.label} request failed: {exc}", self.api_key),
                provider=self.provider_name,
            ) from exc

    def _error(self, response: httpx.Response) -> ProviderError:
        status = response.status_code
        message = redact(f"{self.label} API error {status}: {error_message(response)}", self.api_key)
        return ProviderError(code_for_status(status), message, provider=self.provider_name, status=status)

    def _check(self, response: httpx.Response) -> httpx.Response:
        if response.status_code >= 400:
            raise self._error(response)
        return response

    def _json(self, response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(
                "upstream", f"{self.label} returned a non-JSON response", provider=self.provider_name,
                status=response.status_code,
            ) from exc

    # ------------------------------------------------------------ references
    def load_reference(self, source: str, retries: int = 2) -> Tuple[bytes, str]:
        """Bytes and MIME type of a reference image given as an http(s) or data: URI.

        URL downloads are retried with a short backoff (1s, 2s) on transport errors and 5xx.
        """
        if not isinstance(source, str) or not source:
            raise ProviderError("bad_request", "Reference image must be a non-empty URL", provider=self.provider_name)

        if source.startswith("data:"):
            try:
                header, encoded = source.split(",", 1)
                mime = header[5:].split(";")[0] or "application/octet-stream"
                data = base64.b64decode(encoded)
            except (ValueError, TypeError) as exc:
                raise ProviderError("bad_request", "Malformed data: URI reference", provider=self.provider_name) from exc
            return data, mime

        if not source.startswith(("http://", "https://")):
            raise ProviderError(
                "bad_request", "Reference images must be http(s) or data: URIs", provider=self.provider_name
            )

        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = self._send("GET", source, timeout=30.0)
                if response.status_code < 500:
                    if response.status_code >= 400:
                        raise ProviderError(
                            "bad_request",
                            f"Could not download reference image: HTTP {response.status_code}",
                            provider=self.provider_name,
                            status=response.status_code,
                        )
                    mime = response.headers.get("content-type", "").split(";")[0].strip()
                    return response.content, mime or detect_mime(response.content, "application/octet-stream")
                last = ProviderError(
                    "upstream", f"HTTP {response.status_code}", provider=self.provider_name, status=response.status_code
                )
            except ProviderError as exc:
                if exc.code not in ("network", "timeout", "upstream"):
                    raise
                last = exc
            if attempt < retries:
                wait = attempt + 1
                logger.warning("Reference download failed (attempt %d/%d), retrying in %ds", attempt + 1, retries + 1, wait)
                self._sleep(wait)
        raise ProviderError(
            "upstream",
            f"Failed to download reference image after {retries + 1} attempts: {last}",
            provider=self.provider_name,
        )

    def load_binary(self, source: str) -> bytes:
        return self.load_reference(source)[0]


class VideoProvider(MediaProvider):
    """Async video job: ``start_generation`` returns a job id, ``check_status`` polls it."""

    timeout = 30.0

    def start_generation(self, request):  # pragma: no cover - interface
        raise NotImplementedError

    def check_status(self, job_id: str):  # pragma: no cover - interface
        raise NotImplementedError

    def _download_headers(self) -> dict:
        return {}

    def download_video(self, url: str) -> bytes:
        """Bytes of a finished video at ``url`` (as returned in ``VideoStatusArtifact.result_url``)."""
        response = self._check(self._send("GET", url, timeout=DOWNLOAD_TIMEOUT, headers=self._download_headers()))
        return response.content
