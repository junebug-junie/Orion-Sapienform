"""FastAPI HTTP surface tests -- review gap: only the underlying
svc.stats()/svc.latest_h1_dict() methods were tested directly, never
main.py's actual HTTP wiring (/health, /h1)."""
from __future__ import annotations

import os

os.environ.setdefault("ORION_BUS_ENABLED", "false")

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint_returns_ok_and_stats() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "orion-heartbeat"
    assert "events_absorbed" in body
    assert "allowlisted_organs" in body


def test_h1_endpoint_before_first_computation() -> None:
    with TestClient(app) as client:
        resp = client.get("/h1")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"ok": False, "reason": "no_h1_computed_yet"}
