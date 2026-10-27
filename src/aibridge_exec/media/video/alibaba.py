"""Alibaba Wan video on DashScope (international endpoint)."""
from __future__ import annotations

from typing import Any, Dict

from ...providers.errors import ProviderError
from .._base import VideoProvider
from ..image.alibaba import DASHSCOPE_BASE
from ..types import VideoGenerationRequest, VideoGenerationResponse, VideoMode, VideoStatusArtifact

STATUS_MAP = {
    "PENDING": "pending",
    "RUNNING": "processing",
    "SUCCEEDED": "completed",
    "FAILED": "failed",
    "UNKNOWN": "failed",
}
SIZE_MAP = {"16:9": "1280*720", "9:16": "720*1280", "1:1": "1024*1024"}
SIZE_MAP_1080P = {"16:9": "1920*1080", "9:16": "1080*1920"}


class AlibabaWanProvider(VideoProvider):
    provider_name = "alibaba"
    label = "Alibaba"
    API_URL = f"{DASHSCOPE_BASE}/services/aigc/video-generation/video-synthesis"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._require_key()}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

    def start_generation(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        request.validate()
        headers = self._headers()
        model_name = request.model if "wan" in request.model else "wanx-v1"
        config = request.config or {}

        input_payload: Dict[str, Any] = {"prompt": request.prompt}
        if request.normalized_mode == VideoMode.IMG_TO_VIDEO.value:
            input_payload["img_url"] = request.images[0]

        size = SIZE_MAP.get(request.aspect_ratio, "1280*720")
        if config.get("resolution") == "1080p":
            size = SIZE_MAP_1080P.get(request.aspect_ratio, size)
        parameters: Dict[str, Any] = {"size": size}
        if request.negative_prompt:
            parameters["negative_prompt"] = request.negative_prompt
        for key in ("duration", "camera_motion", "seed"):
            if key in config:
                parameters[key] = config[key]
        if "use_prompt_extend" in config:
            value = config["use_prompt_extend"]
            parameters["use_prompt_extend"] = value if isinstance(value, bool) else str(value).lower() == "true"

        payload = {"model": model_name, "input": input_payload, "parameters": parameters}
        data = self._json(self._check(self._send("POST", self.API_URL, headers=headers, json=payload)))
        if data.get("code"):
            raise ProviderError(
                "bad_request",
                f"Alibaba API error: {data.get('message')} (code: {data.get('code')})",
                provider=self.provider_name,
            )
        task_id = (data.get("output") or {}).get("task_id")
        if not task_id:
            raise ProviderError("upstream", "Alibaba returned no task id", provider=self.provider_name)
        return VideoGenerationResponse.now(provider=self.provider_name, model=model_name, job_id=task_id, status="queued")

    def check_status(self, job_id: str) -> VideoStatusArtifact:
        try:
            data = self._json(self._check(self._send("GET", f"{DASHSCOPE_BASE}/tasks/{job_id}", headers=self._headers())))
        except ProviderError as exc:
            return VideoStatusArtifact(provider_job_id=job_id, status="processing", error=str(exc))

        output = data.get("output") or {}
        status = STATUS_MAP.get(output.get("task_status"), "processing")
        result_url = error = None
        if status == "completed":
            result_url = output.get("video_url")
            if not result_url and isinstance(output.get("results"), list) and output["results"]:
                result_url = output["results"][0].get("url")
        elif status == "failed":
            error = output.get("message") or data.get("message") or "Task Failed"
        return VideoStatusArtifact(provider_job_id=job_id, status=status, result_url=result_url, error=error)
