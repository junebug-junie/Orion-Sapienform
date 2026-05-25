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

from scripts import substrate_proposal_routes  # noqa: E402


def _sample_proposal_frame() -> dict:
    return {
        "schema_version": "proposal.frame.v1",
        "frame_id": "proposal.frame:self.state:tick:policy:self_state_policy.v1:proposal_policy.v1",
        "generated_at": "2026-05-24T12:00:00+00:00",
        "source_self_state_id": "self.state:tick:frame:policy",
        "source_self_state_generated_at": "2026-05-24T12:00:00+00:00",
        "source_attention_frame_id": "attention.frame:tick:policy",
        "source_field_tick_id": "tick",
        "proposal_policy_id": "proposal_policy.v1",
        "overall_action_pressure": 0.5,
        "overall_risk": 0.1,
        "policy_required": True,
        "candidates": [],
        "suppressed_candidates": [],
        "dominant_motivations": [],
        "warnings": [],
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    app = FastAPI()
    app.include_router(substrate_proposal_routes.router)
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
            m.mappings.return_value.first.return_value = {"proposal_frame_json": frame_json}
        return m

    conn.execute.side_effect = execute
    return fake_engine


def test_proposals_latest_returns_frame(client):
    frame = _sample_proposal_frame()
    fake_engine = _fake_engine_with_frame(frame)

    with patch.object(substrate_proposal_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/proposals/latest")

    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "proposal.frame.v1"


def test_proposals_latest_not_found(client):
    fake_engine = _fake_engine_with_frame(None)

    with patch.object(substrate_proposal_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/proposals/latest")

    assert r.status_code == 404
