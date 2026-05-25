from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import substrate_self_state_routes  # noqa: E402


def _sample_self_state() -> dict:
    return {
        "schema_version": "self.state.v1",
        "self_state_id": "self.state:tick_exec:frame_exec:self_state_policy.v1",
        "generated_at": "2026-05-24T12:00:00+00:00",
        "source_field_tick_id": "tick_exec",
        "source_field_generated_at": "2026-05-24T12:00:00+00:00",
        "source_attention_frame_id": "attention.frame:tick_exec:field_attention_policy.v1",
        "source_attention_generated_at": "2026-05-24T12:00:00+00:00",
        "self_state_policy_id": "self_state_policy.v1",
        "overall_condition": "loaded",
        "overall_intensity": 0.55,
        "overall_confidence": 0.7,
        "dimensions": {},
        "dominant_attention_targets": ["node:athena"],
        "dominant_field_channels": {"execution_load": 1.0},
        "unresolved_pressures": [],
        "stabilizing_factors": [],
        "warnings": [],
        "summary_labels": ["execution_loaded"],
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    app = FastAPI()
    app.include_router(substrate_self_state_routes.router)
    return TestClient(app)


def _fake_engine_with_state(state_json: dict | None):
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute(stmt, params=None):
        m = MagicMock()
        if state_json is None:
            m.mappings.return_value.first.return_value = None
        else:
            m.mappings.return_value.first.return_value = {"self_state_json": state_json}
        return m

    conn.execute.side_effect = execute
    return fake_engine


def test_self_state_latest_returns_state(client):
    state = _sample_self_state()
    fake_engine = _fake_engine_with_state(state)

    with patch.object(substrate_self_state_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/self-state/latest")

    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "self.state.v1"
    assert body["overall_condition"] == "loaded"


def test_self_state_latest_not_found(client):
    fake_engine = _fake_engine_with_state(None)

    with patch.object(substrate_self_state_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/self-state/latest")

    assert r.status_code == 404
