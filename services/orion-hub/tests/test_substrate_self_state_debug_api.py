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


def _sample_self_state(**overrides: object) -> dict:
    state = {
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
        "dimensions": {
            "resource_pressure": {
                "dimension_id": "resource_pressure",
                "score": 0.6,
                "confidence": 0.8,
                "dominant_evidence": ["execution_load=0.60"],
                "reasons": ["driven by pressure=0.60 (node: circe)"],
            }
        },
        "dominant_attention_targets": ["node:athena"],
        "dominant_field_channels": {"execution_load": 1.0},
        "unresolved_pressures": [],
        "stabilizing_factors": [],
        "warnings": [],
        "summary_labels": ["execution_loaded"],
    }
    state.update(overrides)
    return state


def _legacy_self_state_payload_with_policy_pressure() -> dict:
    # Mirrors the exact real production row from the 2026-07-12 schema-drift
    # incident: a valid SelfStateV1 shape except for one dimension_id
    # ("policy_pressure") no longer accepted by the current schema.
    return {
        "schema_version": "self.state.v1",
        "self_state_id": "self.state:legacy",
        "generated_at": "2026-07-12T12:00:00+00:00",
        "source_field_tick_id": "tick",
        "source_field_generated_at": "2026-07-12T12:00:00+00:00",
        "source_attention_frame_id": "frame",
        "source_attention_generated_at": "2026-07-12T12:00:00+00:00",
        "self_state_policy_id": "self_state_policy.v1",
        "overall_condition": "steady",
        "overall_intensity": 0.4,
        "overall_confidence": 0.8,
        "dimensions": {
            "policy_pressure": {
                "dimension_id": "policy_pressure",
                "score": 0.0,
                "confidence": 0.5,
                "dominant_evidence": [],
                "reasons": ["policy_pressure from field+attention channel synthesis"],
            }
        },
    }


def _sample_field_state(**overrides: object) -> dict:
    state = {
        "schema_version": "field.state.v1",
        "generated_at": "2026-05-24T12:00:00+00:00",
        "tick_id": "tick_exec",
        "node_vectors": {"node:circe": {"pressure": 0.6}},
        "capability_vectors": {"capability:execution": {"pressure": 0.6}},
        "capability_provenance": {"capability:execution": {"pressure": "node:circe"}},
        "edges": [],
        "recent_perturbations": [],
    }
    state.update(overrides)
    return state


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


def _fake_engine_with_joined_rows(
    *, self_state_json: dict | None, field_json: dict | None
):
    """A fake engine that dispatches on which table the query text targets,
    so both the self-state loader and the field-state loader can be exercised
    against the same patched `_engine()` within one request (mirrors how the
    real evidence-trail endpoint issues two separate queries)."""
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute(stmt, params=None):
        m = MagicMock()
        sql = str(stmt)
        if "substrate_self_state" in sql:
            if self_state_json is None:
                m.mappings.return_value.first.return_value = None
            else:
                m.mappings.return_value.first.return_value = {
                    "self_state_json": self_state_json
                }
        elif "substrate_field_state" in sql:
            if field_json is None:
                m.mappings.return_value.first.return_value = None
            else:
                m.mappings.return_value.first.return_value = {"field_json": field_json}
        else:
            raise AssertionError(f"unexpected query: {sql}")
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


def test_self_state_latest_degrades_gracefully_on_legacy_incompatible_row(client):
    """Regression test for the same bug class as the 2026-07-12 schema-drift
    incident: a row saved before `policy_pressure` was removed from
    SelfStateV1.dimensions' valid dimension_id values must not 500 this
    debug endpoint -- it should degrade to a 404 like "no snapshot found"."""
    fake_engine = _fake_engine_with_state(_legacy_self_state_payload_with_policy_pressure())

    with patch.object(substrate_self_state_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/self-state/latest")

    assert r.status_code == 404


def test_evidence_trail_latest_joins_self_state_and_field_state(client):
    state = _sample_self_state()
    field = _sample_field_state()
    fake_engine = _fake_engine_with_joined_rows(self_state_json=state, field_json=field)

    with patch.object(substrate_self_state_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/self-state/latest/evidence-trail")

    assert r.status_code == 200
    body = r.json()
    assert body["self_state_id"] == state["self_state_id"]
    assert body["source_field_tick_id"] == "tick_exec"
    assert body["field_state_available"] is True
    # Self-state's already-summarized per-dimension evidence is preserved.
    assert (
        body["self_state"]["dimensions"]["resource_pressure"]["dominant_evidence"]
        == ["execution_load=0.60"]
    )
    assert (
        body["self_state"]["dimensions"]["resource_pressure"]["reasons"]
        == ["driven by pressure=0.60 (node: circe)"]
    )
    # Raw field data joined in alongside the summary.
    assert body["field_state"]["node_vectors"] == {"node:circe": {"pressure": 0.6}}
    assert body["field_state"]["capability_vectors"] == {
        "capability:execution": {"pressure": 0.6}
    }
    assert body["field_state"]["capability_provenance"] == {
        "capability:execution": {"pressure": "node:circe"}
    }


def test_evidence_trail_by_id_joins_self_state_and_field_state(client):
    state = _sample_self_state()
    field = _sample_field_state()
    fake_engine = _fake_engine_with_joined_rows(self_state_json=state, field_json=field)

    with patch.object(substrate_self_state_routes, "_engine", return_value=fake_engine):
        r = client.get(
            f"/api/substrate/self-state/{state['self_state_id']}/evidence-trail"
        )

    assert r.status_code == 200
    body = r.json()
    assert body["self_state_id"] == state["self_state_id"]
    assert body["field_state"]["node_vectors"] == {"node:circe": {"pressure": 0.6}}


def test_evidence_trail_not_found_when_self_state_missing(client):
    fake_engine = _fake_engine_with_joined_rows(self_state_json=None, field_json=None)

    with patch.object(substrate_self_state_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/self-state/latest/evidence-trail")

    assert r.status_code == 404


def test_evidence_trail_degrades_sensibly_when_field_state_missing(client):
    """Self-state exists but its source field-state row is gone (e.g. pruned) --
    the endpoint should still return the self-state's own evidence, with
    field_state explicitly marked unavailable rather than 404ing/500ing."""
    state = _sample_self_state()
    fake_engine = _fake_engine_with_joined_rows(self_state_json=state, field_json=None)

    with patch.object(substrate_self_state_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/self-state/latest/evidence-trail")

    assert r.status_code == 200
    body = r.json()
    assert body["field_state_available"] is False
    assert body["field_state"] is None
    assert body["self_state"]["self_state_id"] == state["self_state_id"]


def test_evidence_trail_degrades_gracefully_on_legacy_incompatible_self_state(client):
    fake_engine = _fake_engine_with_joined_rows(
        self_state_json=_legacy_self_state_payload_with_policy_pressure(), field_json=None
    )

    with patch.object(substrate_self_state_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/self-state/latest/evidence-trail")

    assert r.status_code == 404
