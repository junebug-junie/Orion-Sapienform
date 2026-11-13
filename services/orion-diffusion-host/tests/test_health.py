"""/health and /ready honesty pins.

`TestClient(app)` used without `with` never runs the FastAPI lifespan (no
startup, no shutdown -- starlette only triggers lifespan as a context
manager), so `_pipe` stays `None` for the whole test module regardless of
what a real deployment would load. That is also exactly the state this test
wants to pin: /health and /ready must report "not loaded" honestly rather
than assume success (CLAUDE.md §0A "no empty-shell cognition") -- a real
load attempt (network weight download, real GPU) has no place in a fast
unit test anyway; see test_generate.py for the mocked end-to-end path that
exercises real behavior without a real model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.main import app, settings


def test_health_ok():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["ok"] is True
    assert body["service"] == settings.SERVICE_NAME
    assert body["version"] == settings.SERVICE_VERSION
    # Honesty pin: lifespan never ran in this test, so /health must not
    # claim a model is loaded.
    assert body["model_loaded"] is False
    assert body["model_id"] is None


def test_ready_503_when_not_loaded():
    client = TestClient(app)
    resp = client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ready"] is False
    assert body["model_loaded"] is False


def test_generate_503_when_not_loaded():
    client = TestClient(app)
    resp = client.post("/generate", json={"prompt": "a calm orion"})
    assert resp.status_code == 503


# --- /ready needs room on the card (incident 2026-10-16 -> 10-20) -------------------------------
# With CPU offload a "loaded" FLUX pipeline sits in CPU RAM, so loaded said nothing about the card.
# The agent-gpu2 seat held ~24GB of gpu2 and left ~7.6GB; diffusion answered ready for 4 days.
import pytest

import app.main as diffusion_main


@pytest.fixture
def loaded(monkeypatch):
    monkeypatch.setattr(diffusion_main, "_pipe", object())
    monkeypatch.setattr(diffusion_main, "_draining", False)
    monkeypatch.setattr(settings, "DIFFUSION_READY_MIN_GPU_GB", 20.0)
    return monkeypatch


def test_ready_503_when_loaded_but_the_card_is_taken(loaded):
    loaded.setattr(diffusion_main, "_gpu_available_gb", lambda: 7.6)
    resp = TestClient(app).get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ready"] is False and body["model_loaded"] is True
    assert body["gpu_available_gb"] == 7.6 and body["gpu_min_gb"] == 20.0


def test_ready_200_when_loaded_and_the_card_has_room(loaded):
    loaded.setattr(diffusion_main, "_gpu_available_gb", lambda: 30.5)
    resp = TestClient(app).get("/ready")
    assert resp.status_code == 200 and resp.json()["ready"] is True


def test_ready_fails_closed_when_the_card_cannot_be_read(loaded):
    def boom():
        raise RuntimeError("CUDA error: no device")
    loaded.setattr(diffusion_main, "_gpu_available_gb", boom)
    resp = TestClient(app).get("/ready")
    assert resp.status_code == 503
    assert "CUDA error" in resp.json()["gpu_check_error"]


def test_ready_room_check_off_at_zero(loaded):
    loaded.setattr(settings, "DIFFUSION_READY_MIN_GPU_GB", 0.0)
    loaded.setattr(diffusion_main, "_gpu_available_gb", lambda: 0.0)
    assert TestClient(app).get("/ready").status_code == 200


def test_draining_is_still_not_ready_with_room(loaded):
    loaded.setattr(diffusion_main, "_draining", True)
    loaded.setattr(diffusion_main, "_gpu_available_gb", lambda: 30.5)
    assert TestClient(app).get("/ready").status_code == 503


def test_gpu_available_is_free_plus_own_reserved_in_gib(monkeypatch):
    import types

    gib = 1024 ** 3
    seen = {}

    def mem_get_info(device):
        seen["device"] = str(device)
        return (int(7.6 * gib), 32 * gib)

    fake_torch = types.SimpleNamespace(
        device=lambda d: d,
        cuda=types.SimpleNamespace(mem_get_info=mem_get_info, memory_reserved=lambda device: int(0.4 * gib)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert round(diffusion_main._gpu_available_gb(), 2) == 8.0
    assert seen["device"] == settings.DIFFUSION_DEVICE


def test_ready_times_out_closed_on_a_hung_gpu_read(loaded):
    import time as _time

    loaded.setattr(diffusion_main, "_GPU_CHECK_TIMEOUT_SEC", 0.05)
    loaded.setattr(diffusion_main, "_gpu_available_gb", lambda: _time.sleep(0.5) or 30.0)
    resp = TestClient(app).get("/ready")
    assert resp.status_code == 503 and resp.json()["gpu_check_error"]
