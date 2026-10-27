"""Google image generation on the Gemini API (AI Studio keys).

- ``gemini-*`` image models (e.g. gemini-2.5-flash-image): ``:generateContent`` with inline images.
- ``imagen-*`` models: ``:predict`` for text-to-image and for edits.
"""
from __future__ import annotations

import base64
import logging
from typing import Dict, List

from ...providers.errors import ProviderError
from .._base import MediaProvider, detect_mime
from ..types import (
    ImageArtifact,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageMode,
    PromptImageResult,
)

logger = logging.getLogger(__name__)

IMAGEN_ASPECT_RATIOS = {"16:9", "9:16", "3:4", "4:3"}


class GoogleImagenProvider(MediaProvider):
    # Kept as "google" to match the value stored with earlier generations.
    provider_name = "google"
    label = "Google"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    timeout = 90.0

    def generate(self, request: ImageGenerationRequest, storage=None) -> ImageGenerationResponse:
        request.validate()
        headers = {"x-goog-api-key": self._require_key(), "Content-Type": "application/json"}

        model_name = request.model or "imagen-3.0-generate-001"
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        if "gemini" in model_name.lower():
            return self._generate_content(request, model_name, headers)
        return self._predict(request, model_name, headers)

    # ------------------------------------------------------------ Imagen :predict
    def _predict(self, request: ImageGenerationRequest, model_name: str, headers: Dict[str, str]) -> ImageGenerationResponse:
        url = f"{self.BASE_URL}/{model_name}:predict"
        is_edit = bool(request.images) and request.mode in (ImageMode.EDIT, ImageMode.IMG_TO_IMG)
        results: List[PromptImageResult] = []

        for idx, prompt in enumerate(request.prompts):
            if is_edit:
                image_idx = min(idx, len(request.images) - 1)
                instance: dict = {
                    "prompt": prompt,
                    "image": {"bytesBase64Encoded": self._b64(self.load_binary(request.images[image_idx]))},
                }
                if request.masks and len(request.masks) > image_idx:
                    instance["mask"] = {
                        "image": {"bytesBase64Encoded": self._b64(self.load_binary(request.masks[image_idx]))}
                    }
                parameters: dict = {"sampleCount": request.n}
            else:
                instance = {"prompt": prompt}
                aspect = request.aspect_ratio if request.aspect_ratio in IMAGEN_ASPECT_RATIOS else "1:1"
                parameters = {"sampleCount": request.n, "aspectRatio": aspect}

            response = self._check(
                self._send("POST", url, headers=headers, json={"instances": [instance], "parameters": parameters}, timeout=60.0)
            )
            predictions = self._json(response).get("predictions", [])
            if not predictions and not is_edit:
                raise ProviderError("upstream", "No image predictions returned from API.", provider=self.provider_name)

            artifacts = []
            for pred in predictions:
                data = None
                if isinstance(pred, dict):
                    data = pred.get("bytesBase64Encoded") or (pred.get("image") or {}).get("bytesBase64Encoded")
                elif isinstance(pred, str):
                    data = pred
                if not data:
                    logger.warning("Could not extract image data from Imagen prediction")
                    continue
                artifacts.append(
                    ImageArtifact(
                        content=base64.b64decode(data),
                        content_type="image/png",
                        size=request.size,
                        provider=self.provider_name,
                        model=model_name,
                    )
                )
            results.append(PromptImageResult(prompt_index=idx, prompt=prompt, images=artifacts))

        return ImageGenerationResponse.now(provider=self.provider_name, model=model_name, results=results)

    # ------------------------------------------------------------ Gemini :generateContent
    def _generate_content(
        self, request: ImageGenerationRequest, model_name: str, headers: Dict[str, str]
    ) -> ImageGenerationResponse:
        url = f"{self.BASE_URL}/{model_name}:generateContent"
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        thought_sig = request.config.get("thoughtSignature") or request.config.get("thought_signature")
        results: List[PromptImageResult] = []

        for idx, prompt in enumerate(request.prompts):
            parts: List[dict] = [{"text": prompt}]
            if request.images and len(request.images) > idx:
                image_bytes = self.load_binary(request.images[idx])
                parts.append({"inlineData": {"mimeType": detect_mime(image_bytes), "data": self._b64(image_bytes)}})

            contents: List[dict] = []
            if thought_sig:
                # Carries the previous turn's signature so multi-turn edits stay consistent.
                contents.append(
                    {
                        "role": "model",
                        "parts": [{"text": "Previous image generated successfully.", "thoughtSignature": thought_sig}],
                    }
                )
            contents.append({"role": "user", "parts": parts})

            body = self._json(self._check(self._send("POST", url, headers=headers, json={"contents": contents})))

            reported = body.get("usageMetadata")
            if reported:
                usage["input_tokens"] += reported.get("promptTokenCount", 0) or 0
                usage["output_tokens"] += reported.get("candidatesTokenCount", 0) or 0
                usage["total_tokens"] += reported.get("totalTokenCount", 0) or 0

            candidates = body.get("candidates", [])
            if not candidates:
                raise ProviderError(
                    "upstream", "No candidates returned from Gemini Image API.", provider=self.provider_name
                )
            parts_out = (candidates[0].get("content") or {}).get("parts", [])

            new_sig = None
            for part in parts_out:
                new_sig = part.get("thoughtSignature") or part.get("thought_signature") or new_sig

            artifacts = [
                ImageArtifact(
                    content=base64.b64decode(part["inlineData"]["data"]),
                    content_type=part["inlineData"].get("mimeType", "image/png"),
                    size=request.size,
                    provider=self.provider_name,
                    model=model_name,
                    metadata={"thoughtSignature": new_sig} if new_sig else {},
                )
                for part in parts_out
                if "inlineData" in part
            ]
            results.append(PromptImageResult(prompt_index=idx, prompt=prompt, images=artifacts))

        return ImageGenerationResponse.now(
            provider=self.provider_name, model=model_name, results=results, context={"usage": usage}
        )

    @staticmethod
    def _b64(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")
