"""Image and video generation with the caller's own API key.

    from aibridge_exec.media import ImageGenerationRequest, get_image_provider

    provider = get_image_provider("openai")(api_key=KEY)
    response = provider.generate(ImageGenerationRequest(prompts=["a lighthouse"], model="gpt-image-1"))
    png = response.results[0].images[0].content

Video is asynchronous: ``start_generation`` returns a job id, ``check_status`` polls it, and
``download_video`` fetches the finished file from ``result_url``.

    provider = get_video_provider("gemini")(api_key=KEY)
    job = provider.start_generation(VideoGenerationRequest(model="veo-3.1-generate-preview", prompt="..."))
    status = provider.check_status(job.provider_job_id)   # repeat until completed / failed
    mp4 = provider.download_video(status.result_url)

Reference images are http(s) or data: URIs. Errors are ``ProviderError`` (see
``aibridge_exec.providers.errors``) with the provider's status and message and the key redacted.
"""
from __future__ import annotations

from typing import Dict, Type

from ..providers.errors import ProviderError
from ._base import MediaProvider, VideoProvider
from .image.alibaba import AlibabaWanImageProvider
from .image.gemini import GoogleImagenProvider
from .image.openai import OpenAIImageProvider
from .image.stability import StabilityImageProvider
from .types import (
    ImageArtifact,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageMode,
    PromptImageResult,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoMode,
    VideoStatusArtifact,
)
from .video.alibaba import AlibabaWanProvider
from .video.google_veo import GoogleVeoProvider
from .video.luma import LumaVideoProvider
from .video.openai_sora import OpenAISoraProvider
from .video.stability import StabilityVideoProvider

IMAGE_PROVIDERS: Dict[str, Type[MediaProvider]] = {
    "openai": OpenAIImageProvider,
    "gemini": GoogleImagenProvider,
    "stability": StabilityImageProvider,
    "alibaba": AlibabaWanImageProvider,
}

VIDEO_PROVIDERS: Dict[str, Type[VideoProvider]] = {
    "openai": OpenAISoraProvider,
    "gemini": GoogleVeoProvider,
    "luma": LumaVideoProvider,
    "stability": StabilityVideoProvider,
    "alibaba": AlibabaWanProvider,
}


def get_image_provider(provider_id: str) -> Type[MediaProvider]:
    try:
        return IMAGE_PROVIDERS[provider_id]
    except KeyError:
        raise ProviderError("unknown_provider", f"No image provider '{provider_id}'") from None


def get_video_provider(provider_id: str) -> Type[VideoProvider]:
    try:
        return VIDEO_PROVIDERS[provider_id]
    except KeyError:
        raise ProviderError("unknown_provider", f"No video provider '{provider_id}'") from None


__all__ = [
    "AlibabaWanImageProvider",
    "AlibabaWanProvider",
    "GoogleImagenProvider",
    "GoogleVeoProvider",
    "IMAGE_PROVIDERS",
    "ImageArtifact",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "ImageMode",
    "LumaVideoProvider",
    "MediaProvider",
    "OpenAIImageProvider",
    "OpenAISoraProvider",
    "PromptImageResult",
    "ProviderError",
    "StabilityImageProvider",
    "StabilityVideoProvider",
    "VIDEO_PROVIDERS",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
    "VideoMode",
    "VideoProvider",
    "VideoStatusArtifact",
    "get_image_provider",
    "get_video_provider",
]
