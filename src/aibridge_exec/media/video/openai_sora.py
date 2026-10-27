"""OpenAI Sora (``/v1/videos``).

Image-to-video needs the reference image at exactly the output size, so references are
center-cropped and resized first. That uses Pillow, from the ``media`` extra:
``pip install "aibridge-exec[media]"``.
"""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict

from ...providers.errors import ProviderError
from .._base import VideoProvider
from ..types import VideoGenerationRequest, VideoGenerationResponse, VideoMode, VideoStatusArtifact

STATUS_MAP = {
    "queued": "pending",
    "in_progress": "processing",
    "processing": "processing",
    "completed": "completed",
    "failed": "failed",
    "expired": "expired",
}


def output_size(aspect_ratio: str, config: Dict[str, Any]) -> str:
    is_1080p = bool(config) and config.get("resolution") == "1080p"
    if aspect_ratio == "9:16":
        return "1080x1920" if is_1080p else "720x1280"
    if aspect_ratio == "16:9":
        return "1920x1080" if is_1080p else "1280x720"
    return "1280x720"


class OpenAISoraProvider(VideoProvider):
    provider_name = "openai"
    label = "OpenAI Sora"
    BASE_URL = "https://api.openai.com/v1/videos"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._require_key()}", "Content-Type": "application/json"}

    def _download_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._require_key()}"}

    def start_generation(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        request.validate()
        headers = self._headers()
        config = request.config or {}
        size = output_size(request.aspect_ratio, config)

        payload: Dict[str, Any] = {"model": request.model, "prompt": request.prompt, "size": size}
        if "duration" in config:
            payload["seconds"] = str(config["duration"])
        elif request.duration_seconds:
            payload["seconds"] = str(request.duration_seconds)

        if request.normalized_mode == VideoMode.IMG_TO_VIDEO.value:
            payload["input_reference"] = self._input_reference(request.images[0], size)

        for key in ("quality", "n"):
            if key in config:
                payload[key] = config[key]

        data = self._json(self._check(self._send("POST", self.BASE_URL, headers=headers, json=payload)))
        job_id = data.get("id")
        if not job_id:
            raise ProviderError("upstream", "OpenAI Sora returned no job id", provider=self.provider_name)
        return VideoGenerationResponse.now(provider=self.provider_name, model=request.model, job_id=job_id, status="queued")

    def _input_reference(self, reference: Any, size: str) -> Dict[str, str]:
        """``{"file_id": ...}`` passes through; image URLs become a data: URI at ``size``."""
        if isinstance(reference, dict):
            if "file_id" in reference:
                return {"file_id": reference["file_id"]}
            source = reference.get("image_url") or reference.get("url")
            if not source:
                raise ProviderError("bad_request", "Unsupported image reference for OpenAI Sora", provider=self.provider_name)
        else:
            if not isinstance(reference, str) or not reference:
                raise ProviderError("bad_request", "img2video requires a valid image reference", provider=self.provider_name)
            if reference.startswith("file-"):
                return {"file_id": reference}
            source = reference

        data, _ = self.load_reference(source)
        mime, fitted = fit_image(data, size)
        return {"image_url": f"data:{mime};base64,{base64.b64encode(fitted).decode('ascii')}"}

    def check_status(self, job_id: str) -> VideoStatusArtifact:
        try:
            response = self._send("GET", f"{self.BASE_URL}/{job_id}", headers=self._headers())
            data = self._json(self._check(response))
        except ProviderError as exc:
            # Keep polling; the caller gives up after its own retry limit.
            return VideoStatusArtifact(provider_job_id=job_id, status="processing", error=str(exc))

        status = STATUS_MAP.get(data.get("status"), "processing")
        error = None
        if status == "failed":
            err = data.get("error") or {}
            error = err.get("message", "Unknown failure") if isinstance(err, dict) else str(err)
        return VideoStatusArtifact(
            provider_job_id=job_id,
            status=status,
            result_url=f"{self.BASE_URL}/{job_id}/content" if status == "completed" else None,
            error=error,
            progress=int(data.get("progress") or 0),
        )


def fit_image(data: bytes, size: str) -> tuple[str, bytes]:
    """Center-crop and resize image bytes to ``WxH``. PNG if it has alpha, else JPEG."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ProviderError(
            "config", 'Sora image-to-video needs Pillow: pip install "aibridge-exec[media]"', provider="openai"
        ) from exc

    width, height = (int(v) for v in size.split("x", 1))
    try:
        with Image.open(BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)
            has_alpha = "A" in image.getbands()
            fitted = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            out = BytesIO()
            if has_alpha:
                fitted.save(out, format="PNG")
                return "image/png", out.getvalue()
            fitted.convert("RGB").save(out, format="JPEG", quality=95)
            return "image/jpeg", out.getvalue()
    except OSError as exc:
        raise ProviderError("bad_request", "Reference is not a readable image", provider="openai") from exc
