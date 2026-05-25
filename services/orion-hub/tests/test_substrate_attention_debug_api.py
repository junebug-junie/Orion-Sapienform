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

from scripts import substrate_attention_routes  # noqa: E402


def _sample_attention_frame() -> dict:
    return {
        "schema_version": "field.attention.frame.v1",
        "frame_id": "attention.frame:tick_exec:field_attention_policy.v1",
        "generated_at": "2026-05-24T12:00:00+00:00",
        "source_field_tick_id": "tick_exec",
        "source_field_generated_at": "2026-05-24T12:00:00+00:00",
        "attention_policy_id": "field_attention_policy.v1",
        "overall_salience": 0.72,
        "dominant_targets": [
            {
                "target_id": "node:athena",
                "target_kind": "node",
                "salience_score": 0.8,
                "pressure_score": 0.7,
                "novelty_score": 0.0,
                "urgency_score": 0.0,
                "confidence_score": 0.1,
                "dominant_channels": {"execution_load": 0.7},
                "reasons": ["node execution_load is elevated"],
                "evidence_refs": ["field:tick_exec"],
                "suggested_observation_mode": "inspect",
            }
        ],
        "node_targets": [],
        "capability_targets": [],
        "system_targets": [],
        "suppressed_targets": [],
        "recent_perturbations": [],
        "warnings": [],
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    app = FastAPI()
    app.include_router(substrate_attention_routes.router)
    return TestClient(app)


def _fake_engine_with_frame(frame_json: dict | None):
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute(stmt, params=None):
        m = MagicMock()
        if frame_json is None:
            m.mappings.return_value.first.return_value = None
        else:
            m.mappings.return_value.first.return_value = {"frame_json": frame_json}
        return m

    conn.execute.side_effect = execute
    return fake_engine


def test_attention_latest_returns_frame(client):
    frame = _sample_attention_frame()
    fake_engine = _fake_engine_with_frame(frame)

    with patch.object(substrate_attention_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/attention/latest")

    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "field.attention.frame.v1"
    assert body["dominant_targets"][0]["target_id"] == "node:athena"


def test_attention_latest_not_found(client):
    fake_engine = _fake_engine_with_frame(None)

    with patch.object(substrate_attention_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/attention/latest")

    assert r.status_code == 404
