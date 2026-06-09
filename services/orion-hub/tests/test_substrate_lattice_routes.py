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

from scripts import substrate_lattice_routes  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    app = FastAPI()
    app.include_router(substrate_lattice_routes.router)
    return TestClient(app)


# ── /lanes ──────────────────────────────────────────────────────


def test_lanes_returns_known_lanes(client) -> None:
    resp = client.get("/api/substrate-lattice/lanes")
    assert resp.status_code == 200
    lanes = resp.json()
    assert isinstance(lanes, list)
    lane_ids = [lane["lane_id"] for lane in lanes]
    assert "transport" in lane_ids


def test_lanes_transport_lane_is_live(client) -> None:
    resp = client.get("/api/substrate-lattice/lanes")
    lanes = {lane["lane_id"]: lane for lane in resp.json()}
    transport = lanes["transport"]
    assert transport["producer_id"] == "orion-bus"
    assert transport["source_service"] == "orion-bus"
    assert transport["status"] == "live"


def test_lanes_has_no_post_routes() -> None:
    post_routes = [
        r for r in substrate_lattice_routes.router.routes
        if "POST" in getattr(r, "methods", set())
        and getattr(r, "path", "").endswith("/lanes")
    ]
    assert post_routes == []


# ── /transport/latest ───────────────────────────────────────────


def _sample_projection() -> dict:
    return {
        "schema_version": "transport_bus.projection.v1",
        "updated_at": "2026-06-08T01:00:00+00:00",
        "projection_id": "active_transport_bus_projection",
        "buses": {
            "bus:athena": {
                "schema_version": "transport_bus.state.v1",
                "target_id": "bus:athena",
                "node_id": "node:athena",
                "sample_window_id": "window:abc",
                "source_trace_id": "bus.transport:abc",
                "redis_ping_ok": True,
                "streams_observed": 10,
                "total_stream_depth": 100,
                "max_stream_depth": 20,
                "uncataloged_stream_count": 0,
                "backpressure_count": 0,
                "observer_failure_count": 0,
                "bus_health": 1.0,
                "delivery_confidence": 1.0,
                "stream_depth_pressure": 0.0,
                "backpressure": 0.0,
                "catalog_drift_pressure": 0.5,
                "observer_failure_pressure": 0.0,
                "transport_pressure": 0.0,
                "contract_pressure": 1.0,
                "reliability_pressure": 0.0,
                "evidence_event_ids": ["evt:abc"],
                "observed_at": "2026-06-08T01:00:00+00:00",
            }
        },
    }


def test_transport_latest_returns_proof_chain(client) -> None:
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value={
            "projection": _sample_projection(),
            "bus_summary": {
                "bus_id": "bus:athena",
                "bus_health": 1.0,
                "transport_pressure": 0.0,
                "contract_pressure": 1.0,
                "catalog_drift_pressure": 0.5,
                "observer_failure_pressure": 0.0,
                "delivery_confidence": 1.0,
                "observed_at": "2026-06-08T01:00:00+00:00",
            },
            "receipts": [],
            "field_vector": {"pressure": 0.5},
            "attention": {"dominant_targets": ["capability:transport"]},
            "self_state": {"transport_integrity": {"score": 0.8, "confidence": 0.9}},
            "proposals": {"count": 2, "transport_count": 1, "candidates": []},
            "policy": {"approved_count": 2},
            "dispatch": {"dispatch_mode": "dry_run", "dispatch_count": 0},
            "feedback": {"outcome_status": "dry_run_only"},
            "motifs": [],
        }
    ):
        resp = client.get("/api/substrate-lattice/transport/latest")
    assert resp.status_code == 200
    body = resp.json()
    assert "projection" in body
    assert "bus_summary" in body
    assert "attention" in body
    assert "self_state" in body
    assert "proposals" in body
    assert "dispatch" in body
    assert "feedback" in body
    assert "motifs" in body
    assert "receipts" in body
    assert "field_vector" in body
    assert "policy" in body


def test_transport_latest_404_when_no_projection(client) -> None:
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=None
    ):
        resp = client.get("/api/substrate-lattice/transport/latest")
    assert resp.status_code == 404


# ── _load_transport_proof_chain internals ────────────────────────


def _make_engine_sequence(*results):
    """Returns a fake engine where each successive execute() call returns the next result dict or None."""
    fake = MagicMock()
    conn = MagicMock()
    fake.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake.connect.return_value.__exit__ = MagicMock(return_value=False)

    call_results = list(results)
    call_idx = [0]

    def execute(stmt, params=None):
        m = MagicMock()
        idx = call_idx[0]
        call_idx[0] += 1
        if idx < len(call_results):
            row = call_results[idx]
            if row is None:
                m.mappings.return_value.first.return_value = None
                m.mappings.return_value.all.return_value = []
            else:
                m.mappings.return_value.first.return_value = row
                m.mappings.return_value.all.return_value = [row]
        else:
            m.mappings.return_value.first.return_value = None
            m.mappings.return_value.all.return_value = []
        return m
    conn.execute.side_effect = execute
    return fake


def test_load_transport_proof_chain_returns_none_when_no_projection(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import json as _json
    # First query is the projection query — returns None
    fake = _make_engine_sequence(None)
    with patch.object(substrate_lattice_routes, "_engine", return_value=fake):
        result = substrate_lattice_routes._load_transport_proof_chain()
    assert result is None


def test_load_transport_proof_chain_empty_buses_yields_empty_bus_summary(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import json as _json
    proj = {"schema_version": "transport_bus.projection.v1", "updated_at": "2026-06-08T00:00:00+00:00", "projection_id": "active_transport_bus_projection", "buses": {}}
    # Provide proj for first query, then None for all subsequent
    fake = _make_engine_sequence(
        {"projection_json": _json.dumps(proj)},  # M3 projection
        None,  # M3 receipts
        None,  # M4 field
        None,  # M5 attention
        None,  # L6 self-state
        None,  # L7 proposals
        None,  # L8 policy
        None,  # L9 dispatch
        None,  # L10 feedback
        None,  # L11 consolidation
    )
    with patch.object(substrate_lattice_routes, "_engine", return_value=fake):
        result = substrate_lattice_routes._load_transport_proof_chain()
    assert result is not None
    assert result["bus_summary"]["bus_id"] is None
    assert result["bus_summary"]["bus_health"] is None
    assert result["motifs"] == []


def test_load_transport_proof_chain_with_projection_returns_full_structure(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import json as _json
    proj = {
        "schema_version": "transport_bus.projection.v1",
        "updated_at": "2026-06-08T01:00:00+00:00",
        "projection_id": "active_transport_bus_projection",
        "buses": {
            "bus:athena": {
                "schema_version": "transport_bus.state.v1",
                "target_id": "bus:athena", "node_id": "node:athena",
                "sample_window_id": "w", "source_trace_id": "bus.transport:abc",
                "bus_health": 1.0, "delivery_confidence": 1.0,
                "transport_pressure": 0.0, "contract_pressure": 1.0,
                "catalog_drift_pressure": 0.5, "observer_failure_pressure": 0.0,
                "stream_depth_pressure": 0.0, "backpressure": 0.0,
                "reliability_pressure": 0.0, "streams_observed": 10,
                "total_stream_depth": 100, "max_stream_depth": 20,
                "uncataloged_stream_count": 0, "backpressure_count": 0,
                "observer_failure_count": 0,
                "evidence_event_ids": ["evt:1"],
                "observed_at": "2026-06-08T01:00:00+00:00",
                "redis_ping_ok": True,
            }
        },
    }
    fake = _make_engine_sequence(
        {"projection_json": _json.dumps(proj)},  # M3 projection
        None,  # M3 receipts (all())
        None,  # M4 field
        None,  # M5 attention
        None,  # L6 self-state
        None,  # L7 proposals
        None,  # L8 policy
        None,  # L9 dispatch
        None,  # L10 feedback
        None,  # L11 consolidation
    )
    with patch.object(substrate_lattice_routes, "_engine", return_value=fake):
        result = substrate_lattice_routes._load_transport_proof_chain()
    assert result is not None
    assert result["bus_summary"]["bus_health"] == 1.0
    assert result["bus_summary"]["contract_pressure"] == 1.0
    assert result["projection"]["buses"]["bus:athena"]["source_trace_id"] == "bus.transport:abc"
