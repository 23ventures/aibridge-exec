"""Stability AI video (``/v2beta/image-to-video``).

Ported as-is from aibridgecore. Note that Stability's image-to-video endpoint expects a
multipart image upload, so this JSON text-only request is likely rejected by the API today.
"""
from __future__ import annotations

from typing import Any, Dict

from ...providers.errors import ProviderError
from .._base import VideoProvider
from ..types import VideoGenerationRequest, VideoGenerationResponse, VideoStatusArtifact

DIMENSIONS = {"16:9": (1024, 576), "1:1": (768, 768)}


class StabilityVideoProvider(VideoProvider):
    provider_name = "stability"
    label = "Stability"
    API_HOST = "https://api.stability.ai"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._require_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def start_generation(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        request.validate()
        headers = self._headers()
        payload: Dict[str, Any] = {
            "text_prompts": [{"text": request.prompt}],
            "cfg_scale": 1.8,
            "motion_bucket_id": 127,
        }
        if request.aspect_ratio in DIMENSIONS:
            payload["width"], payload["height"] = DIMENSIONS[request.aspect_ratio]

        url = f"{self.API_HOST}/v2beta/image-to-video"
        data = self._json(self._check(self._send("POST", url, headers=headers, json=payload)))
        if not data.get("id"):
            raise ProviderError("upstream", "Stability returned no generation id", provider=self.provider_name)
        return VideoGenerationResponse.now(provider=self.provider_name, model=request.model, job_id=data["id"], status="queued")

    def check_status(self, job_id: str) -> VideoStatusArtifact:
        url = f"{self.API_HOST}/v2beta/image-to-video/result/{job_id}"
        try:
            response = self._send("GET", url, headers=self._headers())
            if response.status_code == 202:
                return VideoStatusArtifact(provider_job_id=job_id, status="processing")
            data = self._json(self._check(response))
        except ProviderError as exc:
            return VideoStatusArtifact(provider_job_id=job_id, status="processing", error=str(exc))

        if data.get("finish_reason") == "SUCCESS":
            return VideoStatusArtifact(
                provider_job_id=job_id, status="completed", result_url=(data.get("video") or {}).get("url")
            )
        if data.get("status") == "error":
            return VideoStatusArtifact(provider_job_id=job_id, status="failed", error=data.get("message"))
        return VideoStatusArtifact(provider_job_id=job_id, status="processing")
