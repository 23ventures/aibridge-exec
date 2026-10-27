"""Request and result types for image and video generation.

Field names match the aibridgecore media contracts these replace, so callers that built
``ImageGenerationRequest`` / ``VideoGenerationRequest`` objects keep working unchanged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..providers.errors import ProviderError


def _invalid(message: str) -> ProviderError:
    return ProviderError("bad_request", message)


# ---------------------------------------------------------------- images
class ImageMode(str, Enum):
    """The kind of image operation requested."""

    TEXT_TO_IMAGE = "text2img"
    EDIT = "edit"
    IMG_TO_IMG = "img2img"


@dataclass
class ImageArtifact:
    content: bytes
    content_type: str
    provider: str
    model: str
    size: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ImageGenerationRequest:
    prompts: List[str]
    model: str
    n: int = 1
    size: str = "1024x1024"
    mode: ImageMode = ImageMode.TEXT_TO_IMAGE
    # For providers that take an aspect ratio instead of a pixel size.
    aspect_ratio: Optional[str] = None
    quality: Optional[str] = None
    style: Optional[str] = None
    # Reference images (http(s) or data: URIs) for edit / img2img.
    images: Optional[List[str]] = None
    masks: Optional[List[str]] = None
    # Provider-specific options (thoughtSignature, seed, cfg_scale, ...).
    config: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompts:
            raise _invalid("At least one prompt is required")
        if not self.model:
            raise _invalid("model is required")
        if self.n < 1:
            raise _invalid("n must be >= 1")
        if self.mode in (ImageMode.EDIT, ImageMode.IMG_TO_IMG) and not self.images:
            raise _invalid(f"Mode '{self.mode.value}' requires at least one reference image")
        if self.masks and not self.images:
            raise _invalid("Masks require base images")
        if self.images and len(self.images) != len(self.prompts):
            raise _invalid(
                f"image count must match prompts count: expected {len(self.prompts)}, got {len(self.images)}"
            )
        if self.masks and len(self.masks) != len(self.images or []):
            raise _invalid(
                f"masks count must match images count: expected {len(self.images or [])}, got {len(self.masks)}"
            )


@dataclass
class PromptImageResult:
    prompt_index: int
    prompt: str
    images: List[ImageArtifact]


@dataclass
class ImageGenerationResponse:
    provider: str
    model: str
    created_at: int
    results: List[PromptImageResult]
    # Set by the caller when it routed the request to a fallback model.
    fallback_notice: Optional[str] = None
    # Holds ``{"usage": {...}}`` for providers that report token usage.
    context: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def now(
        cls,
        provider: str,
        model: str,
        results: List[PromptImageResult],
        fallback_notice: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> "ImageGenerationResponse":
        return cls(
            provider=provider,
            model=model,
            created_at=int(time.time()),
            results=results,
            fallback_notice=fallback_notice,
            context=context or {},
        )


# ---------------------------------------------------------------- video
class VideoMode(str, Enum):
    TEXT_TO_VIDEO = "text2video"
    IMG_TO_VIDEO = "img2video"
    VIDEO_TO_VIDEO = "video2video"


_VIDEO_MODES = {m.value for m in VideoMode}


@dataclass
class VideoGenerationRequest:
    model: str
    prompt: str
    duration_seconds: int = 5
    aspect_ratio: str = "16:9"
    mode: str = VideoMode.TEXT_TO_VIDEO.value
    images: List[Any] = field(default_factory=list)
    videos: List[Any] = field(default_factory=list)
    negative_prompt: Optional[str] = None
    seed: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_mode(self) -> str:
        mode = self.mode.value if isinstance(self.mode, VideoMode) else (self.mode or "")
        return mode.lower() or VideoMode.TEXT_TO_VIDEO.value

    def validate(self) -> None:
        if not self.prompt:
            raise _invalid("Prompt cannot be empty")
        if not self.model:
            raise _invalid("model is required")
        mode = self.normalized_mode
        if mode not in _VIDEO_MODES:
            raise _invalid(f"Unsupported video mode '{self.mode}': expected text2video | img2video")
        if mode == VideoMode.IMG_TO_VIDEO.value and not self.images:
            raise _invalid("img2video mode requires at least one image reference")
        if mode == VideoMode.VIDEO_TO_VIDEO.value:
            raise _invalid("video2video is planned but not enabled yet")


@dataclass
class VideoGenerationResponse:
    provider_job_id: str
    status: str
    provider: str
    model: str

    @staticmethod
    def now(provider: str, model: str, job_id: str, status: str) -> "VideoGenerationResponse":
        return VideoGenerationResponse(provider=provider, model=model, provider_job_id=job_id, status=status)


@dataclass
class VideoStatusArtifact:
    """Normalized job status: pending | processing | completed | failed | expired."""

    provider_job_id: str
    status: str
    result_url: Optional[str] = None
    error: Optional[str] = None
    progress: int = 0
