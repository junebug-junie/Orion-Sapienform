from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orion.schemas.attention_frame import OpenLoopV1
import scripts.attention_loops_routes as routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ORION_ATTENTION_PENDING_CARDS_ENABLED", "true")

    def _fake_loops():
        return [
            (
                OpenLoopV1(id="open-loop-x", target_type="anomaly",
                           description="reactor mismatch", why_it_matters="unresolved",
                           salience=0.7, salience_features={"evidence_strength": 0.8}),
                datetime.now(timezone.utc), 2, "", "chat",
            )
        ]

    published = {}

    def _fake_publish(outcome):
        published["outcome"] = outcome

    monkeypatch.setattr(routes, "load_pending_loops", _fake_loops)
    monkeypatch.setattr(routes, "latest_salience_for_theme", lambda k: (0.6, {"evidence_strength": 0.8}))
    monkeypatch.setattr(routes, "latest_scope_for_theme", lambda k: "chat")
    monkeypatch.setattr(routes, "persist_loop_outcome", lambda o: True)
    monkeypatch.setattr(routes, "suppress_loop", lambda k: True)
    monkeypatch.setattr(routes, "publish_loop_outcome", _fake_publish)

    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), published


def test_list_loops_returns_cards(client):
    c, _ = client
    resp = c.get("/api/attention/loops")
    assert resp.status_code == 200
    data = resp.json()
    assert data and data[0]["title"] == "reactor mismatch"
    assert not data[0]["title"].startswith("open-loop-")


def test_resolve_writes_outcome_and_suppresses(client):
    c, published = client
    resp = c.post("/api/attention/loops/open-loop-x/resolve", json={"note": "done"})
    assert resp.status_code == 200
    assert published["outcome"].verdict == "resolved"


def test_dismiss_writes_outcome(client):
    c, published = client
    resp = c.post("/api/attention/loops/open-loop-x/dismiss", json={"note": "noise"})
    assert resp.status_code == 200
    assert published["outcome"].verdict == "dismissed"


def test_cards_disabled_returns_empty(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("ORION_ATTENTION_PENDING_CARDS_ENABLED", raising=False)
    resp = c.get("/api/attention/loops")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_loops_marks_reverie_scope_as_chronic_pressure(client, monkeypatch):
    c, _ = client

    def _fake_loops():
        return [
            (
                OpenLoopV1(id="open-loop-node", target_type="other",
                           description="substrate:node:substrate.perception",
                           salience=0.9, salience_features={}),
                datetime.now(timezone.utc), 5000, "", "reverie",
            )
        ]

    monkeypatch.setattr(routes, "load_pending_loops", _fake_loops)
    resp = c.get("/api/attention/loops")
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["card_kind"] == "chronic_pressure"


def test_resolve_rejects_chronic_pressure_loop(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(routes, "latest_scope_for_theme", lambda k: "reverie")
    resp = c.post("/api/attention/loops/open-loop-node/resolve", json={"note": ""})
    assert resp.status_code == 409
