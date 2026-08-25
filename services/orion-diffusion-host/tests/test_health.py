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
