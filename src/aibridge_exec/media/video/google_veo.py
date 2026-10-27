"""Google Veo on the Gemini API (AI Studio keys), over REST.

``start_generation`` calls ``models/{model}:predictLongRunning`` and returns the operation name
as the job id; ``check_status`` reads ``GET v1beta/{operation}``. These are the same calls the
google-genai SDK makes in Gemini Developer API mode.

The result URL is returned without the API key. ``download_video`` sends the key in a header,
so the key never ends up in stored URLs or logs.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Optional

from ...providers.errors import ProviderError
from .._base import DOWNLOAD_TIMEOUT, VideoProvider
from ..types import VideoGenerationRequest, VideoGenerationResponse, VideoMode, VideoStatusArtifact

logger = logging.getLogger(__name__)


class GoogleVeoProvider(VideoProvider):
    provider_name = "google"
    label = "Google Veo"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    DOWNLOAD_HOST = "generativelanguage.googleapis.com"

    def _headers(self) -> Dict[str, str]:
        return {"x-goog-api-key": self._require_key(), "Content-Type": "application/json"}

    def _download_headers(self) -> dict:
        return {"x-goog-api-key": self._require_key()}

    # ------------------------------------------------------------ start
    def start_generation(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        request.validate()
        headers = self._headers()
        model_name = self._normalize_model_name(request.model)

        instance: Dict[str, Any] = {"prompt": request.prompt}
        if request.normalized_mode == VideoMode.IMG_TO_VIDEO.value:
            instance["image"] = self._build_image(request.images[0])

        payload = {"instances": [instance], "parameters": self._build_parameters(request, model_name)}
        url = f"{self.BASE_URL}/models/{model_name}:predictLongRunning"
        data = self._json(self._check(self._send("POST", url, headers=headers, json=payload)))

        operation_name = data.get("name")
        if not operation_name:
            raise ProviderError("upstream", "Google Veo returned no operation id", provider=self.provider_name)
        return VideoGenerationResponse.now(
            provider=self.provider_name, model=f"models/{model_name}", job_id=operation_name, status="queued"
        )

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if not model_name:
            raise ProviderError("bad_request", "Google Veo model is required", provider="google")
        return model_name[7:] if model_name.startswith("models/") else model_name

    @staticmethod
    def _build_parameters(request: VideoGenerationRequest, model_name: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {"aspectRatio": request.aspect_ratio, "sampleCount": 1}

        if request.duration_seconds:
            duration = int(request.duration_seconds)
            if "veo-3" in model_name.lower():
                # Veo 3 accepts 4, 6 or 8 seconds.
                duration = 4 if duration <= 4 else 6 if duration <= 6 else 8
            else:
                duration = min(max(duration, 5), 8)
            params["durationSeconds"] = duration

        if request.negative_prompt:
            params["negativePrompt"] = request.negative_prompt

        config = request.config or {}
        if "resolution" in config:
            params["resolution"] = config["resolution"]
        if "person_generation" in config:
            params["personGeneration"] = config["person_generation"]
        if "seed" in config:
            # The Gemini Developer API rejects seed (only Vertex accepts it).
            logger.info("Ignoring seed for Google Veo: not supported on the Gemini API")
        return params

    def _build_image(self, reference: Any) -> Dict[str, str]:
        """Veo image instance ``{bytesBase64Encoded, mimeType}`` from a reference."""
        if isinstance(reference, dict):
            if reference.get("url"):
                reference = reference["url"]
            elif reference.get("inlineData"):
                inline = reference["inlineData"] or {}
                return {"bytesBase64Encoded": inline["data"], "mimeType": inline.get("mimeType", "image/png")}
            elif "imageBytes" in reference or ("mimeType" in reference and "data" in reference):
                raw = reference.get("imageBytes", reference.get("data"))
                encoded = raw if isinstance(raw, str) else base64.b64encode(raw).decode("ascii")
                return {"bytesBase64Encoded": encoded, "mimeType": reference.get("mimeType", "image/png")}
            elif reference.get("gcs_uri"):
                raise ProviderError(
                    "bad_request", "gs:// images need Vertex AI; the Gemini API needs a URL or bytes",
                    provider=self.provider_name,
                )
            else:
                raise ProviderError("bad_request", "Unsupported image reference for Google Veo", provider=self.provider_name)

        if isinstance(reference, str) and reference.startswith("gs://"):
            raise ProviderError(
                "bad_request", "gs:// images need Vertex AI; the Gemini API needs a URL or bytes",
                provider=self.provider_name,
            )

        data, mime = self.load_reference(reference)
        if not mime.startswith("image/"):
            raise ProviderError(
                "bad_request", f"Google Veo expected an image, received '{mime}'", provider=self.provider_name
            )
        return {"bytesBase64Encoded": base64.b64encode(data).decode("ascii"), "mimeType": mime}

    # ------------------------------------------------------------ status
    def check_status(self, job_id: str) -> VideoStatusArtifact:
        try:
            response = self._send("GET", f"{self.BASE_URL}/{job_id}", headers=self._headers())
            operation = self._json(self._check(response))
        except ProviderError as exc:
            logger.warning("Google Veo status check failed for %s: %s", job_id, exc)
            # Transient errors keep the job polling; auth / not-found / bad request end it.
            transient = exc.code in ("network", "timeout", "upstream", "rate_limited")
            return VideoStatusArtifact(
                provider_job_id=job_id, status="processing" if transient else "failed", error=str(exc)
            )

        if not operation.get("done"):
            return VideoStatusArtifact(provider_job_id=job_id, status="processing")

        error = operation.get("error")
        if error:
            message = error.get("message") if isinstance(error, dict) else None
            return VideoStatusArtifact(provider_job_id=job_id, status="failed", error=message or str(error))

        video_url = self._extract_result_url(operation)
        if not video_url:
            reasons = self._filtered_reasons(operation)
            return VideoStatusArtifact(
                provider_job_id=job_id,
                status="failed",
                error=f"Video blocked by safety filters: {reasons}" if reasons else "No video URL found in response",
            )
        return VideoStatusArtifact(provider_job_id=job_id, status="completed", result_url=video_url)

    @staticmethod
    def _generate_video_response(operation: dict) -> dict:
        return ((operation.get("response") or {}).get("generateVideoResponse")) or {}

    def _extract_result_url(self, operation: dict) -> Optional[str]:
        samples = self._generate_video_response(operation).get("generatedSamples") or []
        if not samples:
            return None
        return ((samples[0] or {}).get("video") or {}).get("uri")

    def _filtered_reasons(self, operation: dict) -> str:
        reasons = self._generate_video_response(operation).get("raiMediaFilteredReasons") or []
        return "; ".join(str(r) for r in reasons)

    # ------------------------------------------------------------ download
    def download_video(self, url: str) -> bytes:
        # Older stored URLs carry ?key=; only send our key header to Google's own host.
        headers = self._download_headers() if self.DOWNLOAD_HOST in url else {}
        return self._check(self._send("GET", url, timeout=DOWNLOAD_TIMEOUT, headers=headers)).content
