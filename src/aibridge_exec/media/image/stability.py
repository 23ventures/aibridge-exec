"""Stability AI image generation (Stable Image v2beta: sd3, ultra, core, inpaint)."""
from __future__ import annotations

from typing import List

from .._base import MediaProvider
from ..types import (
    ImageArtifact,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageMode,
    PromptImageResult,
)

STABILITY_ENDPOINT_MAP = {
    "sd3": "/generate/sd3",
    "sd3-turbo": "/generate/sd3",
    "sd3-large": "/generate/sd3",
    "sd3-large-turbo": "/generate/sd3",
    "sd3-medium": "/generate/sd3",
    "stable-image-ultra": "/generate/ultra",
    "stable-image-core": "/generate/core",
}

VALID_STYLES = {
    "3d-model", "analog-film", "anime", "cinematic", "comic-book", "digital-art", "enhance",
    "fantasy-art", "isometric", "line-art", "low-poly", "modeling-compound", "neon-punk",
    "origami", "photographic", "pixel-art", "tile-texture",
}


class StabilityImageProvider(MediaProvider):
    provider_name = "stability"
    label = "Stability"
    BASE_URL = "https://api.stability.ai/v2beta/stable-image"

    def generate(self, request: ImageGenerationRequest, storage=None) -> ImageGenerationResponse:
        request.validate()
        headers = {"Authorization": f"Bearer {self._require_key()}", "Accept": "image/*"}
        results: List[PromptImageResult] = []

        for idx, prompt in enumerate(request.prompts):
            has_image = request.images is not None and len(request.images) > idx
            has_mask = request.masks is not None and len(request.masks) > idx
            endpoint = self.resolve_endpoint(request.model, request.mode, has_image, has_mask)
            # Fetch references once per prompt, not once per sample.
            image_bytes = self.load_binary(request.images[idx]) if has_image else None
            mask_bytes = self.load_binary(request.masks[idx]) if has_mask else None

            images = []
            for _ in range(request.n):
                # Stability requires multipart even when no file is sent.
                files: dict = {"none": (None, "")}
                data: dict = {
                    "prompt": prompt,
                    "output_format": request.config.get("output_format", "png"),
                    "model": request.model,
                }
                for key in ("negative_prompt", "seed", "cfg_scale"):
                    if request.config.get(key):
                        data[key] = str(request.config[key])

                style = request.style or request.config.get("style_preset")
                if isinstance(style, str) and style.lower() in VALID_STYLES:
                    data["style_preset"] = style.lower()

                if image_bytes is not None:
                    files["image"] = ("image.png", image_bytes, "image/png")
                    if "/generate/" in endpoint:
                        data["mode"] = "image-to-image"
                        data["strength"] = str(request.config.get("strength", 0.5))
                if mask_bytes is not None:
                    files["mask"] = ("mask.png", mask_bytes, "image/png")

                response = self._check(self._send("POST", endpoint, headers=headers, files=files, data=data))
                images.append(
                    ImageArtifact(
                        content=response.content,
                        content_type=response.headers.get("content-type", "image/png"),
                        provider=self.provider_name,
                        model=request.model,
                        size=request.size,
                    )
                )
            results.append(PromptImageResult(prompt_index=idx, prompt=prompt, images=images))

        return ImageGenerationResponse.now(provider=self.provider_name, model=request.model, results=results)

    def resolve_endpoint(self, model: str, mode: ImageMode, has_image: bool, has_mask: bool) -> str:
        """Endpoint for a model and mode: masks and edits go to inpaint, else by model family."""
        if has_mask and mode in (ImageMode.EDIT, ImageMode.IMG_TO_IMG):
            return f"{self.BASE_URL}/edit/inpaint"
        if has_image and mode == ImageMode.EDIT:
            return f"{self.BASE_URL}/edit/inpaint"
        model_lower = (model or "").lower()
        for key, path in STABILITY_ENDPOINT_MAP.items():
            if key in model_lower:
                return f"{self.BASE_URL}{path}"
        return f"{self.BASE_URL}/generate/sd3"
