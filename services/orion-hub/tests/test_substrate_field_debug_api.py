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

from scripts import substrate_field_routes  # noqa: E402


def _atlas_field_state() -> dict:
    return {
        "schema_version": "field.state.v1",
        "generated_at": "2026-05-24T12:00:00+00:00",
        "tick_id": "tick_atlas_test",
        "node_vectors": {
            "node:atlas": {
                "availability": 1.0,
                "gpu_pressure": 0.72,
                "memory_pressure": 0.31,
                "staleness": 0.0,
                "cpu_pressure": 0.0,
                "thermal_pressure": 0.0,
                "disk_pressure": 0.0,
                "expected_offline_suppression": 0.0,
            }
        },
        "capability_vectors": {
            "capability:llm_inference": {
                "pressure": 0.61,
                "confidence": 0.78,
                "available_capacity": 0.39,
            }
        },
        "edges": [
            {
                "source_id": "node:atlas",
                "target_id": "capability:llm_inference",
                "edge_type": "node_capability",
                "weight": 0.85,
                "channel_map": {"gpu_pressure": "pressure", "memory_pressure": "pressure"},
            }
        ],
        "recent_perturbations": ["state_delta:delta_atlas_gpu"],
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    app = FastAPI()
    app.include_router(substrate_field_routes.router)
    return TestClient(app)


def _fake_engine_with_field(field_json: dict | None):
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute(stmt, params=None):
        m = MagicMock()
        if field_json is None:
            m.mappings.return_value.first.return_value = None
        else:
            m.mappings.return_value.first.return_value = {"field_json": field_json}
        return m

    conn.execute.side_effect = execute
    return fake_engine


def test_field_latest_returns_parsed_state(client):
    field = _atlas_field_state()
    fake_engine = _fake_engine_with_field(field)

    with patch.object(substrate_field_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/field/latest")

    assert r.status_code == 200
    body = r.json()
    assert body["tick_id"] == "tick_atlas_test"
    assert body["node_vectors"]["node:atlas"]["gpu_pressure"] == 0.72


def test_field_latest_not_found(client):
    fake_engine = _fake_engine_with_field(None)

    with patch.object(substrate_field_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/field/latest")

    assert r.status_code == 404


def test_field_node_atlas_returns_vector_and_capabilities(client):
    field = _atlas_field_state()
    fake_engine = _fake_engine_with_field(field)

    with patch.object(substrate_field_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/field/node/atlas")

    assert r.status_code == 200
    body = r.json()
    assert body["node_id"] == "atlas"
    assert "gpu_pressure" in body["field_vector"]
    assert body["field_vector"]["gpu_pressure"] == 0.72
    assert isinstance(body["connected_capabilities"], list)
    assert body["connected_capabilities"][0]["capability_id"] == "llm_inference"
    assert body["connected_capabilities"][0]["pressure"] == 0.61
    assert body["connected_capabilities"][0]["edge_weight"] == 0.85
    assert body["recent_perturbations"] == ["state_delta:delta_atlas_gpu"]


def test_field_capability_llm_inference(client):
    field = _atlas_field_state()
    fake_engine = _fake_engine_with_field(field)

    with patch.object(substrate_field_routes, "_engine", return_value=fake_engine):
        r = client.get("/api/substrate/field/capability/llm_inference")

    assert r.status_code == 200
    body = r.json()
    assert body["capability_id"] == "llm_inference"
    assert body["field_vector"]["pressure"] == 0.61
    assert isinstance(body["connected_nodes"], list)
    assert body["connected_nodes"][0]["node_id"] == "atlas"
    assert body["connected_nodes"][0]["pressure"] == 0.72
    assert body["connected_nodes"][0]["edge_weight"] == 0.85
