"""Skeleton smoke test: /health responds and honestly reports no model loaded.

Pins the "no empty-shell cognition" boundary the other direction -- this
service must not claim `model_loaded: true` until a later patch actually
wires one in (CLAUDE.md §0A "no empty-shell cognition").
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
    # Honesty pin: this patch never loads a model, so /health must not claim
    # it did.
    assert body["model_loaded"] is False
