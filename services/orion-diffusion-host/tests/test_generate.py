"""POST /generate, with a fake pipe injected -- no GPU, no torch, no network.

This is deliberately NOT a mocked-response test that just checks status
codes. It injects a fake diffusers-shaped pipe (`__call__` -> object with
`.images`) into `app.main._pipe` and asserts the *real* PNG bytes that come
back out of `/generate` are sniffable by
`orion.reverie.visual_storage.sniff_image` -- the actual downstream
consumer's contract (a caller passes this endpoint's response body straight
into `store_visual_artifact`). Proving the wire format matches the real
consumer is the point; proving diffusion model quality is not (that needs a
real GPU and is out of scope for a fast unit test).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_mod
from orion.reverie.visual_storage import sniff_image


class FakePipe:
    """Records every call's kwargs and returns a real (tiny) PIL image --
    real PNG bytes out, so the response is genuinely sniffable, not just a
    literal we assert against."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        width = kwargs.get("width") or 8
        height = kwargs.get("height") or 8
        image = Image.new("RGB", (width, height), color=(100, 150, 200))
        return SimpleNamespace(images=[image])


def test_generate_returns_real_sniffable_png(monkeypatch):
    fake = FakePipe()
    monkeypatch.setattr(main_mod, "_pipe", fake)
    client = TestClient(main_mod.app)

    resp = client.post("/generate", json={"prompt": "orion dreaming in color"})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    sniffed = sniff_image(resp.content)
    assert sniffed is not None
    mime, width, height = sniffed
    assert mime == "image/png"
    assert (width, height) == (
        main_mod.settings.DIFFUSION_DEFAULT_WIDTH,
        main_mod.settings.DIFFUSION_DEFAULT_HEIGHT,
    )


def test_generate_uses_settings_defaults_when_request_omits_them(monkeypatch):
    fake = FakePipe()
    monkeypatch.setattr(main_mod, "_pipe", fake)
    client = TestClient(main_mod.app)

    resp = client.post("/generate", json={"prompt": "orion dreaming"})

    assert resp.status_code == 200
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["width"] == main_mod.settings.DIFFUSION_DEFAULT_WIDTH
    assert call["height"] == main_mod.settings.DIFFUSION_DEFAULT_HEIGHT
    assert call["num_inference_steps"] == main_mod.settings.DIFFUSION_NUM_INFERENCE_STEPS
    assert call["guidance_scale"] == main_mod.settings.DIFFUSION_GUIDANCE_SCALE
    assert call["generator"] is None


def test_generate_request_overrides_settings_defaults(monkeypatch):
    fake = FakePipe()
    monkeypatch.setattr(main_mod, "_pipe", fake)
    client = TestClient(main_mod.app)

    resp = client.post(
        "/generate",
        json={
            "prompt": "orion dreaming",
            "width": 256,
            "height": 256,
            "num_inference_steps": 4,
            "guidance_scale": 1.5,
        },
    )

    assert resp.status_code == 200
    call = fake.calls[0]
    assert call["width"] == 256
    assert call["height"] == 256
    assert call["num_inference_steps"] == 4
    assert call["guidance_scale"] == 1.5


def test_generate_prompt_too_long_rejected(monkeypatch):
    fake = FakePipe()
    monkeypatch.setattr(main_mod, "_pipe", fake)
    client = TestClient(main_mod.app)

    too_long = "x" * (main_mod.settings.DIFFUSION_MAX_PROMPT_CHARS + 1)
    resp = client.post("/generate", json={"prompt": too_long})

    assert resp.status_code == 422
    assert not fake.calls  # rejected before ever touching the pipe


def test_generate_pipe_exception_returns_500_not_crash(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("simulated CUDA OOM")

    monkeypatch.setattr(main_mod, "_pipe", _boom)
    client = TestClient(main_mod.app)

    resp = client.post("/generate", json={"prompt": "orion dreaming"})

    assert resp.status_code == 500
    assert "simulated CUDA OOM" in resp.json()["detail"]


def test_ready_and_health_report_loaded_when_pipe_present(monkeypatch):
    fake = FakePipe()
    monkeypatch.setattr(main_mod, "_pipe", fake)
    client = TestClient(main_mod.app)

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True

    health = client.get("/health")
    assert health.json()["model_loaded"] is True
    assert health.json()["model_id"] == main_mod.settings.DIFFUSION_MODEL_ID
