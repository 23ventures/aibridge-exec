import base64
import io
import json

import httpx
import pytest

from aibridge_exec.media import (
    AlibabaWanImageProvider,
    AlibabaWanProvider,
    GoogleImagenProvider,
    GoogleVeoProvider,
    ImageGenerationRequest,
    ImageMode,
    LumaVideoProvider,
    OpenAIImageProvider,
    OpenAISoraProvider,
    ProviderError,
    StabilityImageProvider,
    StabilityVideoProvider,
    VideoGenerationRequest,
    get_image_provider,
    get_video_provider,
)

KEY = "test-key-1234567890"
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 16
PNG_B64 = base64.b64encode(PNG).decode()
REF_URL = "https://files.example.com/ref.png"


class Router:
    """MockTransport that answers by (method, url-prefix) and records requests."""

    def __init__(self, routes):
        self.routes = routes  # list of (method, prefix, response-or-list)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        url = str(request.url)
        for method, prefix, reply in self.routes:
            if request.method == method and url.startswith(prefix):
                if isinstance(reply, list):
                    reply = reply.pop(0) if len(reply) > 1 else reply[0]
                return reply() if callable(reply) else reply
        raise AssertionError(f"unexpected request {request.method} {url}")

    def provider(self, cls, **kwargs):
        sleeps = []
        p = cls(api_key=KEY, http=httpx.Client(transport=httpx.MockTransport(self)), sleep=sleeps.append, **kwargs)
        p.sleeps = sleeps
        return p

    def find(self, method, prefix):
        return [r for r in self.requests if r.method == method and str(r.url).startswith(prefix)]


def ok(body, status=200):
    return httpx.Response(status, json=body)


def png_response():
    return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})


def img_req(**kw):
    kw.setdefault("prompts", ["a lighthouse"])
    kw.setdefault("model", "gpt-image-1")
    return ImageGenerationRequest(**kw)


# ---------------------------------------------------------------- registry & validation
def test_registry_lookups():
    assert get_image_provider("openai") is OpenAIImageProvider
    assert get_video_provider("gemini") is GoogleVeoProvider
    with pytest.raises(ProviderError) as exc:
        get_video_provider("nope")
    assert exc.value.code == "unknown_provider"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompts": []},
        {"n": 0},
        {"mode": ImageMode.EDIT},
        {"masks": ["m"]},
        {"images": ["a", "b"]},
    ],
)
def test_image_request_validation(kwargs):
    with pytest.raises(ProviderError) as exc:
        img_req(**kwargs).validate()
    assert exc.value.code == "bad_request"


def test_video_request_validation():
    with pytest.raises(ProviderError):
        VideoGenerationRequest(model="m", prompt="p", mode="img2video").validate()
    with pytest.raises(ProviderError):
        VideoGenerationRequest(model="m", prompt="p", mode="video2video").validate()
    VideoGenerationRequest(model="m", prompt="p", mode="TEXT2VIDEO").validate()


def test_missing_key_is_auth_error():
    with pytest.raises(ProviderError) as exc:
        OpenAIImageProvider(api_key=None).generate(img_req())
    assert exc.value.code == "auth"


# ---------------------------------------------------------------- OpenAI images
def test_openai_text_to_image_and_usage():
    router = Router([("POST", "https://api.openai.com/v1/images/generations",
                      ok({"data": [{"b64_json": PNG_B64}], "usage": {"input_tokens": 7, "output_tokens": 100}}))])
    resp = router.provider(OpenAIImageProvider).generate(img_req(size="512x512"))
    sent = json.loads(router.requests[0].content)
    assert sent == {"model": "gpt-image-1", "prompt": "a lighthouse", "n": 1, "size": "512x512"}
    assert router.requests[0].headers["authorization"] == f"Bearer {KEY}"
    art = resp.results[0].images[0]
    assert art.content == PNG and art.content_type == "image/png" and art.provider == "openai"
    assert resp.context == {"usage": {"input_tokens": 7, "output_tokens": 100}}


def test_openai_edit_uploads_reference_as_multipart():
    router = Router([
        ("GET", REF_URL, png_response()),
        ("POST", "https://api.openai.com/v1/images/edits", ok({"data": [{"b64_json": PNG_B64}]})),
    ])
    router.provider(OpenAIImageProvider).generate(img_req(mode=ImageMode.EDIT, images=[REF_URL]))
    edit = router.find("POST", "https://api.openai.com/v1/images/edits")[0]
    assert edit.headers["content-type"].startswith("multipart/form-data")
    assert b'name="image"; filename="image.png"' in edit.content
    assert b'name="model"\r\n\r\ngpt-image-1' in edit.content


def test_openai_retries_5xx_then_succeeds():
    router = Router([("POST", "https://api.openai.com/v1/images/generations",
                      [ok({"error": {"message": "busy"}}, 503), ok({"data": [{"b64_json": PNG_B64}]})])])
    provider = router.provider(OpenAIImageProvider)
    provider.generate(img_req())
    assert len(router.requests) == 2 and provider.sleeps == [1.0]


def test_openai_quota_error_keeps_signal_text_and_redacts_key():
    body = {"error": {"message": f"You exceeded your current quota (key {KEY})", "code": "insufficient_quota"}}
    router = Router([("POST", "https://api.openai.com/v1/images/generations", ok(body, 429))])
    with pytest.raises(ProviderError) as exc:
        router.provider(OpenAIImageProvider).generate(img_req())
    err = exc.value
    assert err.code == "rate_limited" and err.status == 429
    assert "429" in str(err) and "quota" in str(err)  # platform's rate-limit detector reads these
    assert KEY not in str(err)
    assert len(router.requests) == 3  # retried up to MAX_RETRIES


def test_openai_400_is_not_retried():
    router = Router([("POST", "https://api.openai.com", ok({"error": {"message": "bad size"}}, 400))])
    with pytest.raises(ProviderError) as exc:
        router.provider(OpenAIImageProvider).generate(img_req())
    assert exc.value.code == "bad_request" and len(router.requests) == 1


# ---------------------------------------------------------------- Google images
def test_imagen_predict_uses_header_key_and_aspect_ratio():
    router = Router([("POST", "https://generativelanguage.googleapis.com/v1beta/models/imagen-4.0-generate-001:predict",
                      ok({"predictions": [{"bytesBase64Encoded": PNG_B64}]}))])
    resp = router.provider(GoogleImagenProvider).generate(
        img_req(model="imagen-4.0-generate-001", n=2, aspect_ratio="16:9"))
    req = router.requests[0]
    assert "key=" not in str(req.url) and req.headers["x-goog-api-key"] == KEY
    assert json.loads(req.content)["parameters"] == {"sampleCount": 2, "aspectRatio": "16:9"}
    assert resp.model == "models/imagen-4.0-generate-001" and resp.provider == "google"


def test_gemini_image_inline_reference_signature_and_usage():
    reply = {
        "candidates": [{"content": {"parts": [
            {"text": "here", "thoughtSignature": "sig-2"},
            {"inlineData": {"mimeType": "image/png", "data": PNG_B64}},
        ]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1290, "totalTokenCount": 1300},
    }
    router = Router([
        ("GET", REF_URL, png_response()),
        ("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent", ok(reply)),
    ])
    resp = router.provider(GoogleImagenProvider).generate(img_req(
        model="gemini-2.5-flash-image", mode=ImageMode.EDIT, images=[REF_URL], config={"thoughtSignature": "sig-1"}))
    sent = json.loads(router.find("POST", "https://generativelanguage")[0].content)
    assert sent["contents"][0]["parts"][0]["thoughtSignature"] == "sig-1"
    assert sent["contents"][1]["parts"][1]["inlineData"] == {"mimeType": "image/png", "data": PNG_B64}
    art = resp.results[0].images[0]
    assert art.metadata == {"thoughtSignature": "sig-2"} and art.content == PNG
    assert resp.context["usage"] == {"input_tokens": 10, "output_tokens": 1290, "total_tokens": 1300}


def test_gemini_image_no_candidates_raises():
    router = Router([("POST", "https://generativelanguage", ok({"candidates": []}))])
    with pytest.raises(ProviderError):
        router.provider(GoogleImagenProvider).generate(img_req(model="gemini-2.5-flash-image"))


# ---------------------------------------------------------------- Stability images
def test_stability_endpoint_style_and_one_reference_download_per_prompt():
    router = Router([
        ("GET", REF_URL, png_response()),
        ("POST", "https://api.stability.ai/v2beta/stable-image/generate/ultra", png_response()),
    ])
    resp = router.provider(StabilityImageProvider).generate(img_req(
        model="stable-image-ultra", n=2, style="Anime", mode=ImageMode.IMG_TO_IMG, images=[REF_URL],
        config={"seed": 42}))
    posts = router.find("POST", "https://api.stability.ai")
    assert len(posts) == 2 and len(router.find("GET", REF_URL)) == 1
    body = posts[0].content
    assert b'name="style_preset"\r\n\r\nanime' in body and b'name="mode"\r\n\r\nimage-to-image' in body
    assert b'name="seed"\r\n\r\n42' in body
    assert len(resp.results[0].images) == 2


def test_stability_routes_masks_to_inpaint():
    p = StabilityImageProvider(api_key=KEY)
    assert p.resolve_endpoint("sd3-large", ImageMode.EDIT, True, True).endswith("/edit/inpaint")
    assert p.resolve_endpoint("stable-image-core", ImageMode.TEXT_TO_IMAGE, False, False).endswith("/generate/core")
    assert p.resolve_endpoint("whatever", ImageMode.TEXT_TO_IMAGE, False, False).endswith("/generate/sd3")


# ---------------------------------------------------------------- Alibaba images
def test_alibaba_image_submits_async_polls_and_downloads():
    router = Router([
        ("POST", "https://dashscope-us.aliyuncs.com/api/v1/services/aigc/text2image", ok({"output": {"task_id": "t1"}})),
        ("GET", "https://dashscope-us.aliyuncs.com/api/v1/tasks/t1", [
            ok({"output": {"task_status": "RUNNING"}}),
            ok({"output": {"task_status": "SUCCEEDED", "results": [{"url": "https://oss.example.com/1.png"}]}}),
        ]),
        ("GET", "https://oss.example.com/1.png", png_response()),
    ])
    provider = router.provider(AlibabaWanImageProvider)
    resp = provider.generate(img_req(model="wan2.2-t2i-flash", size="16:9"))
    submit = router.requests[0]
    assert submit.headers["x-dashscope-async"] == "enable"
    assert json.loads(submit.content)["parameters"]["size"] == "1280*720"
    assert resp.results[0].images[0].content == PNG and provider.sleeps == [2.0, 2.0]


def test_alibaba_image_failed_task():
    router = Router([
        ("POST", "https://dashscope-us", ok({"output": {"task_id": "t1"}})),
        ("GET", "https://dashscope-us", ok({"output": {"task_status": "FAILED", "message": "blocked"}})),
    ])
    with pytest.raises(ProviderError, match="blocked"):
        router.provider(AlibabaWanImageProvider).generate(img_req(model="wanx-v1"))


# ---------------------------------------------------------------- Veo
VEO = "https://generativelanguage.googleapis.com/v1beta"


def test_veo_start_text2video_parameters():
    router = Router([("POST", f"{VEO}/models/veo-3.1-generate-preview:predictLongRunning",
                      ok({"name": "models/veo-3.1-generate-preview/operations/op1"}))])
    resp = router.provider(GoogleVeoProvider).start_generation(VideoGenerationRequest(
        model="models/veo-3.1-generate-preview", prompt="waves", duration_seconds=5, negative_prompt="rain",
        config={"resolution": "1080p", "seed": 3}))
    req = router.requests[0]
    assert req.headers["x-goog-api-key"] == KEY and "key=" not in str(req.url)
    body = json.loads(req.content)
    assert body["instances"] == [{"prompt": "waves"}]
    # Veo 3 snaps 5s up to 6s; seed is dropped (Gemini API rejects it)
    assert body["parameters"] == {"aspectRatio": "16:9", "sampleCount": 1, "durationSeconds": 6,
                                  "negativePrompt": "rain", "resolution": "1080p"}
    assert resp.provider_job_id == "models/veo-3.1-generate-preview/operations/op1"
    assert resp.model == "models/veo-3.1-generate-preview" and resp.status == "queued"


def test_veo2_duration_clamped_and_img2video_reference():
    router = Router([
        ("GET", REF_URL, png_response()),
        ("POST", f"{VEO}/models/veo-2.0-generate-001:predictLongRunning", ok({"name": "op"})),
    ])
    router.provider(GoogleVeoProvider).start_generation(VideoGenerationRequest(
        model="veo-2.0-generate-001", prompt="p", duration_seconds=12, mode="img2video", images=[REF_URL]))
    body = json.loads(router.find("POST", VEO)[0].content)
    assert body["parameters"]["durationSeconds"] == 8
    assert body["instances"][0]["image"] == {"bytesBase64Encoded": PNG_B64, "mimeType": "image/png"}


def test_veo_rejects_non_image_reference():
    router = Router([("GET", REF_URL, httpx.Response(200, content=b"x", headers={"content-type": "text/html"}))])
    with pytest.raises(ProviderError, match="expected an image"):
        router.provider(GoogleVeoProvider).start_generation(VideoGenerationRequest(
            model="veo-3.0-generate-001", prompt="p", mode="img2video", images=[REF_URL]))


def test_veo_status_lifecycle_and_url_has_no_key():
    done = {"done": True, "response": {"generateVideoResponse": {"generatedSamples": [
        {"video": {"uri": f"{VEO}/files/abc:download?alt=media"}}]}}}
    router = Router([("GET", f"{VEO}/models/veo/operations/op1", [ok({"name": "op1"}), ok(done)])])
    provider = router.provider(GoogleVeoProvider)
    assert provider.check_status("models/veo/operations/op1").status == "processing"
    status = provider.check_status("models/veo/operations/op1")
    assert status.status == "completed"
    assert status.result_url == f"{VEO}/files/abc:download?alt=media" and KEY not in status.result_url


def test_veo_status_safety_filter_and_error():
    filtered = {"done": True, "response": {"generateVideoResponse": {"raiMediaFilteredReasons": ["celebrity"]}}}
    errored = {"done": True, "error": {"code": 3, "message": "bad prompt"}}
    router = Router([("GET", f"{VEO}/op", [ok(filtered), ok(errored)])])
    provider = router.provider(GoogleVeoProvider)
    s1 = provider.check_status("op")
    assert s1.status == "failed" and "celebrity" in s1.error
    s2 = provider.check_status("op")
    assert s2.status == "failed" and s2.error == "bad prompt"


def test_veo_status_transient_error_keeps_polling_but_404_fails():
    router = Router([("GET", f"{VEO}/op", [ok({}, 503), ok({"error": {"message": "gone"}}, 404)])])
    provider = router.provider(GoogleVeoProvider)
    assert provider.check_status("op").status == "processing"
    assert provider.check_status("op").status == "failed"


def test_veo_download_sends_key_header_only_to_google():
    router = Router([
        ("GET", f"{VEO}/files/abc", httpx.Response(302, headers={"location": "https://storage.example.com/v.mp4"})),
        ("GET", "https://storage.example.com/v.mp4", httpx.Response(200, content=b"MP4")),
        ("GET", "https://cdn.example.com/v.mp4", httpx.Response(200, content=b"MP4")),
    ])
    provider = router.provider(GoogleVeoProvider)
    assert provider.download_video(f"{VEO}/files/abc:download?alt=media") == b"MP4"
    assert router.requests[0].headers["x-goog-api-key"] == KEY
    provider.download_video("https://cdn.example.com/v.mp4")
    assert "x-goog-api-key" not in router.requests[-1].headers


# ---------------------------------------------------------------- Sora
SORA = "https://api.openai.com/v1/videos"


def test_sora_text2video_payload():
    router = Router([("POST", SORA, ok({"id": "video_1", "status": "queued"}))])
    resp = router.provider(OpenAISoraProvider).start_generation(VideoGenerationRequest(
        model="sora-2", prompt="p", aspect_ratio="9:16", duration_seconds=8,
        config={"resolution": "1080p", "quality": "high"}))
    assert json.loads(router.requests[0].content) == {
        "model": "sora-2", "prompt": "p", "size": "1080x1920", "seconds": "8", "quality": "high"}
    assert resp.provider_job_id == "video_1"


def test_sora_img2video_fits_reference_to_output_size():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (300, 300), "red").save(buf, format="JPEG")
    router = Router([
        ("GET", REF_URL, httpx.Response(200, content=buf.getvalue(), headers={"content-type": "image/jpeg"})),
        ("POST", SORA, ok({"id": "video_2"})),
    ])
    router.provider(OpenAISoraProvider).start_generation(VideoGenerationRequest(
        model="sora-2", prompt="p", mode="img2video", images=[REF_URL]))
    ref = json.loads(router.find("POST", SORA)[0].content)["input_reference"]["image_url"]
    assert ref.startswith("data:image/jpeg;base64,")
    with Image.open(io.BytesIO(base64.b64decode(ref.split(",", 1)[1]))) as fitted:
        assert fitted.size == (1280, 720)


def test_sora_file_id_reference_passes_through():
    router = Router([("POST", SORA, ok({"id": "v"}))])
    router.provider(OpenAISoraProvider).start_generation(VideoGenerationRequest(
        model="sora-2", prompt="p", mode="img2video", images=["file-abc"]))
    assert json.loads(router.requests[0].content)["input_reference"] == {"file_id": "file-abc"}


def test_sora_status_mapping_and_download():
    router = Router([
        ("GET", f"{SORA}/video_1/content", httpx.Response(200, content=b"MP4")),
        ("GET", f"{SORA}/video_1", [
            ok({"status": "in_progress", "progress": 40}),
            ok({"status": "completed"}),
            ok({"status": "failed", "error": {"message": "moderation"}}),
            ok({}, 500),
        ]),
    ])
    provider = router.provider(OpenAISoraProvider)
    s = provider.check_status("video_1")
    assert (s.status, s.progress) == ("processing", 40)
    s = provider.check_status("video_1")
    assert s.status == "completed" and s.result_url == f"{SORA}/video_1/content"
    assert provider.check_status("video_1").error == "moderation"
    assert provider.check_status("video_1").status == "processing"  # transient
    assert provider.download_video(f"{SORA}/video_1/content") == b"MP4"
    assert router.requests[-1].headers["authorization"] == f"Bearer {KEY}"


def test_sora_start_error_message():
    router = Router([("POST", SORA, ok({"error": {"message": "billing hard limit reached"}}, 400))])
    with pytest.raises(ProviderError, match="billing hard limit"):
        router.provider(OpenAISoraProvider).start_generation(VideoGenerationRequest(model="sora-2", prompt="p"))


# ---------------------------------------------------------------- Luma, Stability, Alibaba video
LUMA = "https://api.lumalabs.ai/dream-machine/v1/generations"


def test_luma_start_and_status():
    router = Router([
        ("POST", LUMA, ok({"id": "g1"})),
        ("GET", f"{LUMA}/g1", [ok({"state": "dreaming"}),
                               ok({"state": "completed", "assets": {"video": "https://cdn.luma/v.mp4"}})]),
    ])
    provider = router.provider(LumaVideoProvider)
    resp = provider.start_generation(VideoGenerationRequest(model="ray-2", prompt="p", config={"model": "ray-2"}))
    assert json.loads(router.requests[0].content) == {"prompt": "p", "aspect_ratio": "16:9", "loop": False, "model": "ray-2"}
    assert resp.provider_job_id == "g1"
    assert provider.check_status("g1").status == "processing"
    assert provider.check_status("g1").result_url == "https://cdn.luma/v.mp4"


def test_stability_video_status_202_is_processing():
    url = "https://api.stability.ai/v2beta/image-to-video/result/j1"
    router = Router([("GET", url, [httpx.Response(202, json={}),
                                   ok({"finish_reason": "SUCCESS", "video": {"url": "https://s/v.mp4"}})])])
    provider = router.provider(StabilityVideoProvider)
    assert provider.check_status("j1").status == "processing"
    assert provider.check_status("j1").result_url == "https://s/v.mp4"


def test_alibaba_video_start_and_status():
    router = Router([
        ("POST", "https://dashscope-us.aliyuncs.com/api/v1/services/aigc/video-generation", ok({"output": {"task_id": "t9"}})),
        ("GET", "https://dashscope-us.aliyuncs.com/api/v1/tasks/t9", [
            ok({"output": {"task_status": "SUCCEEDED", "video_url": "https://oss/v.mp4"}}),
            ok({"output": {"task_status": "FAILED", "message": "nsfw"}}),
        ]),
    ])
    provider = router.provider(AlibabaWanProvider)
    resp = provider.start_generation(VideoGenerationRequest(
        model="wan2.2-i2v-plus", prompt="p", mode="img2video", images=[REF_URL], aspect_ratio="9:16",
        config={"resolution": "1080p", "use_prompt_extend": "True", "duration": 5}))
    body = json.loads(router.requests[0].content)
    assert body["input"] == {"prompt": "p", "img_url": REF_URL}
    assert body["parameters"] == {"size": "1080*1920", "duration": 5, "use_prompt_extend": True}
    assert resp.provider_job_id == "t9" and resp.model == "wan2.2-i2v-plus"
    assert provider.check_status("t9").result_url == "https://oss/v.mp4"
    assert provider.check_status("t9").error == "nsfw"


# ---------------------------------------------------------------- references
def test_reference_data_uri_and_rejects_local_paths():
    p = OpenAIImageProvider(api_key=KEY)
    assert p.load_reference(f"data:image/png;base64,{PNG_B64}") == (PNG, "image/png")
    with pytest.raises(ProviderError, match="http"):
        p.load_reference("/etc/passwd")


def test_reference_download_retries_then_fails():
    router = Router([("GET", REF_URL, ok({}, 502))])
    provider = router.provider(OpenAIImageProvider)
    with pytest.raises(ProviderError, match="after 3 attempts"):
        provider.load_reference(REF_URL)
    assert provider.sleeps == [1, 2]


def test_reference_404_is_not_retried():
    router = Router([("GET", REF_URL, ok({}, 404))])
    with pytest.raises(ProviderError) as exc:
        router.provider(OpenAIImageProvider).load_reference(REF_URL)
    assert exc.value.code == "bad_request" and len(router.requests) == 1
