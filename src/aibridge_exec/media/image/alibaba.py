"""Alibaba Wan / Wanx image generation on DashScope (international endpoint).

DashScope image synthesis is asynchronous; ``generate`` submits a task and polls it
(every 2s, up to 90s) before downloading the results.
"""
from __future__ import annotations

import logging
from typing import List

from ...providers.errors import ProviderError
from .._base import DOWNLOAD_TIMEOUT, MediaProvider
from ..types import ImageArtifact, ImageGenerationRequest, ImageGenerationResponse, PromptImageResult

logger = logging.getLogger(__name__)

DASHSCOPE_BASE = "https://dashscope-us.aliyuncs.com/api/v1"
SIZE_MAP = {"16:9": "1280*720", "9:16": "720*1280", "1:1": "1024*1024"}


class AlibabaWanImageProvider(MediaProvider):
    provider_name = "alibaba"
    label = "Alibaba"
    API_URL = f"{DASHSCOPE_BASE}/services/aigc/text2image/image-synthesis"
    POLL_INTERVAL = 2.0
    MAX_POLLS = 45
    timeout = 30.0

    def generate(self, request: ImageGenerationRequest, storage=None) -> ImageGenerationResponse:
        request.validate()
        headers = {
            "Authorization": f"Bearer {self._require_key()}",
            "Content-Type": "application/json",
            "X-DashScope-WorkSpace": "default",
            "X-DashScope-Async": "enable",
        }
        model_name = request.model or "wanx-v1"
        results: List[PromptImageResult] = []

        for idx, prompt in enumerate(request.prompts):
            payload = {
                "model": model_name,
                "input": {"prompt": prompt},
                "parameters": {"style": "<auto>", "n": request.n, "size": SIZE_MAP.get(request.size, "1024*1024")},
            }
            data = self._json(self._check(self._send("POST", self.API_URL, headers=headers, json=payload)))
            if data.get("code"):
                raise ProviderError(
                    "bad_request", f"Alibaba API error: {data.get('message')}", provider=self.provider_name
                )
            task_id = (data.get("output") or {}).get("task_id")
            if not task_id:
                raise ProviderError("upstream", "Alibaba returned no task id", provider=self.provider_name)

            urls = self._wait_for_task(task_id, headers)
            artifacts = []
            for url in urls:
                try:
                    content = self._check(self._send("GET", url, timeout=DOWNLOAD_TIMEOUT)).content
                except ProviderError as exc:
                    logger.warning("Failed to download Alibaba image: %s", exc)
                    continue
                artifacts.append(
                    ImageArtifact(
                        content=content,
                        content_type="image/png",
                        size=request.size,
                        provider=self.provider_name,
                        model=model_name,
                    )
                )
            results.append(PromptImageResult(prompt_index=idx, prompt=prompt, images=artifacts))

        return ImageGenerationResponse.now(provider=self.provider_name, model=model_name, results=results)

    def _wait_for_task(self, task_id: str, headers: dict) -> List[str]:
        poll_url = f"{DASHSCOPE_BASE}/tasks/{task_id}"
        for _ in range(self.MAX_POLLS):
            self._sleep(self.POLL_INTERVAL)
            output = (self._json(self._check(self._send("GET", poll_url, headers=headers))).get("output") or {})
            status = output.get("task_status")
            if status == "SUCCEEDED":
                return [item["url"] for item in output.get("results", []) if item.get("url")]
            if status in ("FAILED", "UNKNOWN"):
                raise ProviderError(
                    "upstream",
                    f"Alibaba task failed: {output.get('message', 'Unknown error')}",
                    provider=self.provider_name,
                )
        raise ProviderError("timeout", "Alibaba image generation timed out", provider=self.provider_name)
