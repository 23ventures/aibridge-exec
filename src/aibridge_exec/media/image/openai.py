"""OpenAI image generation (GPT Image, DALL·E): ``/images/generations`` and ``/images/edits``."""
from __future__ import annotations

import base64
import logging
from typing import List, Optional

from ...providers.errors import ProviderError
from .._base import MediaProvider
from ..types import (
    ImageArtifact,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageMode,
    PromptImageResult,
)

logger = logging.getLogger(__name__)


class OpenAIImageProvider(MediaProvider):
    provider_name = "openai"
    label = "OpenAI"
    BASE_URL = "https://api.openai.com/v1"
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 1.5

    def generate(self, request: ImageGenerationRequest, storage=None) -> ImageGenerationResponse:
        request.validate()
        headers = {"Authorization": f"Bearer {self._require_key()}"}
        results: List[PromptImageResult] = []
        usage = {"input_tokens": 0, "output_tokens": 0}

        for idx, prompt in enumerate(request.prompts):
            image_bytes: Optional[bytes] = None
            mask_bytes: Optional[bytes] = None
            if request.mode in (ImageMode.EDIT, ImageMode.IMG_TO_IMG) and request.images:
                image_bytes = self.load_binary(request.images[min(idx, len(request.images) - 1)])
                if request.masks and len(request.masks) > idx:
                    mask_bytes = self.load_binary(request.masks[idx])

            if image_bytes is None:
                body = self._post(
                    f"{self.BASE_URL}/images/generations",
                    headers=headers,
                    json={"model": request.model, "prompt": prompt, "n": request.n, "size": request.size},
                )
            else:
                files = {"image": ("image.png", image_bytes, "image/png")}
                if mask_bytes:
                    files["mask"] = ("mask.png", mask_bytes, "image/png")
                body = self._post(
                    f"{self.BASE_URL}/images/edits",
                    headers=headers,
                    data={"model": request.model, "prompt": prompt, "n": str(request.n), "size": request.size},
                    files=files,
                )

            reported = body.get("usage") or body.get("usage_metadata")
            if reported:
                usage["input_tokens"] += reported.get("input_tokens") or 0
                usage["output_tokens"] += reported.get("output_tokens") or 0

            artifacts = [
                ImageArtifact(
                    content=base64.b64decode(item["b64_json"]),
                    content_type="image/png",
                    provider=self.provider_name,
                    model=request.model,
                    size=request.size,
                )
                for item in body.get("data", [])
                if item.get("b64_json")
            ]
            results.append(PromptImageResult(prompt_index=idx, prompt=prompt, images=artifacts))

        return ImageGenerationResponse.now(
            provider=self.provider_name, model=request.model, results=results, context={"usage": usage}
        )

    def _post(self, url: str, **kwargs) -> dict:
        """POST with retries on 429, 5xx and transport errors (1s, then x1.5)."""
        delay = 1.0
        for attempt in range(self.MAX_RETRIES):
            last_attempt = attempt == self.MAX_RETRIES - 1
            try:
                response = self._send("POST", url, **kwargs)
            except ProviderError:
                if last_attempt:
                    raise
            else:
                status = response.status_code
                if status < 400:
                    return self._json(response)
                if (status != 429 and status < 500) or last_attempt:
                    raise self._error(response)
                logger.warning("OpenAI images returned %s, retrying in %.1fs", status, delay)
            self._sleep(delay)
            delay *= self.BACKOFF_FACTOR
        raise ProviderError("upstream", "OpenAI request failed after retries", provider=self.provider_name)
