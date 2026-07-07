from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts import self_brain_routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    app = FastAPI()
    app.include_router(self_brain_routes.router)
    return TestClient(app)


def _frame(seq):
    return {
        "frame_id": f"f{seq}",
        "generated_at": "2026-07-07T12:00:00+00:00",
        "tick_seq": seq,
        "phase": "live",
        "regions": [],
        "nodes": [],
        "edges": [],
    }


def _fake_engine(tail_rows):
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute(stmt, params=None):
        m = MagicMock()
        m.mappings.return_value.all.return_value = tail_rows
        m.mappings.return_value.first.return_value = tail_rows[0] if tail_rows else None
        return m

    conn.execute.side_effect = execute
    return engine


def test_tail_returns_ascending_and_200(client):
    rows = [{"frame_json": _frame(3)}, {"frame_json": _frame(2)}]  # DESC from DB
    with patch.object(self_brain_routes, "_engine", return_value=_fake_engine(rows)):
        r = client.get("/api/self-brain/frames/tail?limit=2")
    assert r.status_code == 200
    body = r.json()
    assert [f["tick_seq"] for f in body["frames"]] == [2, 3]


def test_tail_degrades_to_empty_when_no_engine(client):
    with patch.object(self_brain_routes, "_engine", return_value=None):
        r = client.get("/api/self-brain/frames/tail")
    assert r.status_code == 200
    assert r.json()["frames"] == []


def test_router_is_read_only(client):
    routes = [r for r in client.app.routes if hasattr(r, "methods")]
    for route in routes:
        if str(route.path).startswith("/api/self-brain"):
            assert route.methods <= {"GET", "HEAD"}, route.path


def test_tail_degrades_to_200_when_create_engine_raises(client, monkeypatch):
    import sqlalchemy

    def boom(*a, **k):
        raise RuntimeError("bad dsn")

    monkeypatch.setattr(sqlalchemy, "create_engine", boom)
    r = client.get("/api/self-brain/frames/tail")
    assert r.status_code == 200
    assert r.json()["frames"] == []
