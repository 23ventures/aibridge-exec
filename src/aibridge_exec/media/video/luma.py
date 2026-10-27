"""Luma Dream Machine (``/dream-machine/v1/generations``)."""
from __future__ import annotations

from typing import Any, Dict

from ...providers.errors import ProviderError
from .._base import VideoProvider
from ..types import VideoGenerationRequest, VideoGenerationResponse, VideoStatusArtifact

STATUS_MAP = {"queued": "pending", "dreaming": "processing", "completed": "completed", "failed": "failed"}


class LumaVideoProvider(VideoProvider):
    provider_name = "luma"
    label = "Luma"
    API_URL = "https://api.lumalabs.ai/dream-machine/v1/generations"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._require_key()}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }

    def start_generation(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        request.validate()
        headers = self._headers()
        payload: Dict[str, Any] = {"prompt": request.prompt, "aspect_ratio": request.aspect_ratio, "loop": False}
        # Provider options (model, resolution, keyframes, ...) pass straight through.
        payload.update(request.config or {})

        data = self._json(self._check(self._send("POST", self.API_URL, headers=headers, json=payload)))
        if not data.get("id"):
            raise ProviderError("upstream", "Luma returned no generation id", provider=self.provider_name)
        return VideoGenerationResponse.now(provider=self.provider_name, model=request.model, job_id=data["id"], status="queued")

    def check_status(self, job_id: str) -> VideoStatusArtifact:
        try:
            data = self._json(self._check(self._send("GET", f"{self.API_URL}/{job_id}", headers=self._headers())))
        except ProviderError as exc:
            return VideoStatusArtifact(provider_job_id=job_id, status="processing", error=str(exc))

        status = STATUS_MAP.get(data.get("state"), "processing")
        return VideoStatusArtifact(
            provider_job_id=job_id,
            status=status,
            result_url=(data.get("assets") or {}).get("video") if status == "completed" else None,
            error=data.get("failure_reason") if status == "failed" else None,
        )
