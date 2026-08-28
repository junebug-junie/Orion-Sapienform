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


class FakeTokenizer:
    """Minimal stand-in for a HF tokenizer -- `model_max_length` +
    `__call__(text) -> {"input_ids": [...]}`, the only two surfaces
    `_log_prompt_token_budget` touches."""

    def __init__(self, model_max_length: int, tokens_per_char: float = 1.0):
        self.model_max_length = model_max_length
        self._tokens_per_char = tokens_per_char

    def __call__(self, text: str):
        return {"input_ids": [0] * max(1, int(len(text) * self._tokens_per_char))}


def _capture_warnings(monkeypatch):
    """loguru does not propagate to stdlib logging (so pytest's `caplog`
    fixture cannot see it) without a bridge handler this repo's test suite
    does not set up elsewhere -- monkeypatching `logger.warning` directly is
    the simpler, dependency-free way to assert on a loguru call."""
    calls: list[str] = []
    monkeypatch.setattr(
        main_mod.logger, "warning", lambda fmt, *args, **kw: calls.append(fmt.format(*args))
    )
    return calls


def test_log_prompt_token_budget_warns_when_over_real_model_max(monkeypatch):
    """The actual regression this exists for (2026-08-28): a prompt over the
    tokenizer's real model_max_length must produce a visible warning --
    diffusers itself never surfaces this, silently truncating instead."""
    fake = FakePipe()
    fake.tokenizer = FakeTokenizer(model_max_length=77, tokens_per_char=1.0)
    monkeypatch.setattr(main_mod, "_pipe", fake)
    calls = _capture_warnings(monkeypatch)

    main_mod._log_prompt_token_budget("x" * 200)

    assert any("exceeds" in c and "budget" in c for c in calls)


def test_log_prompt_token_budget_silent_when_within_budget(monkeypatch):
    fake = FakePipe()
    fake.tokenizer = FakeTokenizer(model_max_length=77, tokens_per_char=1.0)
    monkeypatch.setattr(main_mod, "_pipe", fake)
    calls = _capture_warnings(monkeypatch)

    main_mod._log_prompt_token_budget("x" * 10)

    assert calls == []


def test_log_prompt_token_budget_checks_both_encoders(monkeypatch):
    """SDXL-family pipelines carry a second encoder (tokenizer_2,
    OpenCLIP-bigG) -- must be checked too, not just the first."""
    fake = FakePipe()
    fake.tokenizer = FakeTokenizer(model_max_length=77, tokens_per_char=0.1)  # never trips
    fake.tokenizer_2 = FakeTokenizer(model_max_length=77, tokens_per_char=1.0)  # trips
    monkeypatch.setattr(main_mod, "_pipe", fake)
    calls = _capture_warnings(monkeypatch)

    main_mod._log_prompt_token_budget("x" * 200)

    assert any("tokenizer_2" in c for c in calls)


def test_log_prompt_token_budget_never_raises_when_pipe_has_no_tokenizer(monkeypatch):
    """FakePipe (this file's default) has no .tokenizer at all -- the
    real-world case for every OTHER test in this file, and the case a
    not-yet-loaded/differently-shaped pipe hits too. Must no-op, not raise."""
    fake = FakePipe()
    monkeypatch.setattr(main_mod, "_pipe", fake)

    main_mod._log_prompt_token_budget("anything")  # must not raise


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
        raise RuntimeError("simulated CUDA OOM, path=/mnt/storage-warm/models/diffusion")

    monkeypatch.setattr(main_mod, "_pipe", _boom)
    client = TestClient(main_mod.app)

    resp = client.post("/generate", json={"prompt": "orion dreaming"})

    assert resp.status_code == 500
    detail = resp.json()["detail"]
    # Review finding: the raw exception text can embed local paths / driver
    # internals -- the client-facing detail must be generic, never the raw
    # str(exc). Only the exception *class name* is safe to surface.
    assert "RuntimeError" in detail
    assert "simulated CUDA OOM" not in detail
    assert "/mnt/storage-warm" not in detail


def test_generate_429_when_already_in_flight(monkeypatch):
    class AlwaysLocked:
        def locked(self):
            return True

    fake = FakePipe()
    monkeypatch.setattr(main_mod, "_pipe", fake)
    monkeypatch.setattr(main_mod, "_generation_lock", AlwaysLocked())
    client = TestClient(main_mod.app)

    resp = client.post("/generate", json={"prompt": "orion dreaming"})

    assert resp.status_code == 429
    assert not fake.calls  # rejected before ever touching the pipe


def test_generate_rejects_out_of_bounds_dimensions(monkeypatch):
    fake = FakePipe()
    monkeypatch.setattr(main_mod, "_pipe", fake)
    client = TestClient(main_mod.app)

    zero_width = client.post("/generate", json={"prompt": "x", "width": 0})
    assert zero_width.status_code == 422

    negative_height = client.post("/generate", json={"prompt": "x", "height": -8})
    assert negative_height.status_code == 422

    too_large = client.post("/generate", json={"prompt": "x", "width": 4096})
    assert too_large.status_code == 422

    assert not fake.calls  # all three rejected before ever touching the pipe


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
