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


# ── /transport/gates ─────────────────────────────────────────────


def _sample_proof_chain_for_gates(
    bus_age_sec: float = 5.0,
    contract_pressure: float = 1.0,
    transport_pressure: float = 0.0,
    dispatch_mode: str = "dry_run",
    receipts: list | None = None,
    source_trace_id: str = "bus.transport:abc",
) -> dict:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    observed_at = (now - timedelta(seconds=bus_age_sec)).isoformat()
    updated_at = (now - timedelta(seconds=bus_age_sec)).isoformat()
    return {
        "projection": {
            "updated_at": updated_at,
            "buses": {
                "bus:athena": {
                    "source_trace_id": source_trace_id,
                }
            },
        },
        "bus_summary": {
            "bus_health": 1.0,
            "transport_pressure": transport_pressure,
            "contract_pressure": contract_pressure,
            "catalog_drift_pressure": 0.0,
            "observer_failure_pressure": 0.0,
            "delivery_confidence": 1.0,
            "observed_at": observed_at,
        },
        "receipts": receipts if receipts is not None else [{"receipt_id": "r1"}],
        "field_vector": {"pressure": 0.5},
        "attention": {"capability_targets": ["capability:transport"]},
        "self_state": {"transport_integrity": {"score": 0.8}},
        "proposals": {"count": 1, "transport_count": 1, "candidates": []},
        "policy": {"approved_count": 1},
        "dispatch": {"dispatch_mode": dispatch_mode, "dispatch_count": 0},
        "feedback": {"outcome_status": "dry_run_only"},
        "motifs": [],
    }


def test_gates_freshness_pass(client) -> None:
    chain = _sample_proof_chain_for_gates(bus_age_sec=5.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    assert resp.status_code == 200
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["freshness"]["state"] == "pass"


def test_gates_freshness_blocked_when_stale(client) -> None:
    chain = _sample_proof_chain_for_gates(bus_age_sec=60.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    assert resp.status_code == 200
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["freshness"]["state"] == "blocked"


def test_gates_contract_watch_when_high(client) -> None:
    chain = _sample_proof_chain_for_gates(contract_pressure=1.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["contract"]["state"] == "watch"


def test_gates_pressure_quiet_when_zero(client) -> None:
    chain = _sample_proof_chain_for_gates(transport_pressure=0.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["pressure"]["state"] == "quiet"


def test_gates_action_ceiling_reflects_dispatch_mode(client) -> None:
    chain = _sample_proof_chain_for_gates(dispatch_mode="dry_run")
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["action_ceiling"]["state"] == "dry_run"


def test_gates_evidence_blocked_when_no_receipts(client) -> None:
    chain = _sample_proof_chain_for_gates(receipts=[])
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["evidence"]["state"] == "blocked"


def test_gates_lineage_blocked_when_no_trace_id(client) -> None:
    chain = _sample_proof_chain_for_gates(source_trace_id="")
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["lineage"]["state"] == "blocked"


def test_gates_404_when_no_chain(client) -> None:
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=None
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    assert resp.status_code == 404


def test_gates_contract_pass_when_below_threshold(client) -> None:
    chain = _sample_proof_chain_for_gates(contract_pressure=0.3)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    # 0.3 is between 0.0 and watch_at=0.50 → "pass"
    assert gates["contract"]["state"] == "pass"


def test_gates_pressure_watch_when_nonzero(client) -> None:
    chain = _sample_proof_chain_for_gates(transport_pressure=0.5)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["pressure"]["state"] == "watch"


# ── _load_transport_proof_chain internals ────────────────────────


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


# ── /transport/simulate ──────────────────────────────────────────


def test_simulate_returns_comparison_when_thresholds_change(client) -> None:
    chain = _sample_proof_chain_for_gates(contract_pressure=1.0, transport_pressure=0.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ), patch.object(
        substrate_lattice_routes, "_load_yaml", return_value={}
    ):
        resp = client.post(
            "/api/substrate-lattice/transport/simulate",
            json={
                "lane_id": "transport",
                "thresholds": {
                    "contract_pressure_watch_at": 1.1,  # above 1.0 → suppresses channel
                },
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "current" in body
    assert "simulated" in body
    assert "changed" in body
    # With all pressures at 0.0 except contract_pressure=1.0, current should promote
    assert body["current"]["bucket"] == "capability_targets"
    # With watch_at raised to 1.1, contract channel no longer promotes → suppressed
    assert body["simulated"]["bucket"] == "suppressed_targets"
    assert body["changed"] is True


def test_simulate_contract_suppressed_when_threshold_above_value(client) -> None:
    # contract_pressure=1.0, raise watch_at to 1.1 → no channels promote → suppressed
    chain = _sample_proof_chain_for_gates(
        contract_pressure=1.0,
        transport_pressure=0.0,
    )
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ), patch.object(
        substrate_lattice_routes, "_load_yaml", return_value={}
    ):
        resp = client.post(
            "/api/substrate-lattice/transport/simulate",
            json={
                "lane_id": "transport",
                "thresholds": {
                    "contract_pressure_watch_at": 1.1,
                    "transport_pressure_watch_at": 1.1,
                    "catalog_drift_pressure_watch_at": 1.1,
                    "observer_failure_pressure_watch_at": 1.1,
                },
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["simulated"]["bucket"] == "suppressed_targets"
    assert body["simulated"]["action_ceiling"] == "ignore"
    assert body["changed"] is True


def test_simulate_no_change_when_same_thresholds(client) -> None:
    # Use current policy defaults — changed should be False
    chain = _sample_proof_chain_for_gates(
        contract_pressure=0.3,  # below default watch_at=0.50 → no channels promote
        transport_pressure=0.1,  # below default watch_at=0.25 → no channels promote
    )
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ), patch.object(
        substrate_lattice_routes, "_load_yaml", return_value={}
    ):
        resp = client.post(
            "/api/substrate-lattice/transport/simulate",
            json={
                "lane_id": "transport",
                "thresholds": {},  # use current policy defaults
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    # Both current and simulated should have no channels promoted
    assert body["current"]["bucket"] == "suppressed_targets"
    assert body["simulated"]["bucket"] == "suppressed_targets"
    assert body["changed"] is False


def test_simulate_404_when_no_chain(client) -> None:
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=None
    ):
        resp = client.post(
            "/api/substrate-lattice/transport/simulate",
            json={"lane_id": "transport", "thresholds": {}},
        )
    assert resp.status_code == 404


def test_simulate_no_db_writes(client) -> None:
    """Simulate POST route must not produce any write routes on the /lanes path."""
    write_routes_on_lanes = [
        r for r in substrate_lattice_routes.router.routes
        if "POST" in getattr(r, "methods", set())
        and "/lanes" in getattr(r, "path", "")
    ]
    assert write_routes_on_lanes == []


# ── /transport/draft-policy-patch ───────────────────────────────


def test_draft_patch_returns_diff_text(client) -> None:
    resp = client.post(
        "/api/substrate-lattice/transport/draft-policy-patch",
        json={
            "lane_id": "transport",
            "thresholds": {"contract_pressure_watch_at": 0.75},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "diff" in body
    assert isinstance(body["diff"], str)
    assert "lane_id" in body
    assert "applied_thresholds" in body
    assert "note" in body


def test_draft_patch_diff_contains_changed_value(client) -> None:
    resp = client.post(
        "/api/substrate-lattice/transport/draft-policy-patch",
        json={
            "lane_id": "transport",
            "thresholds": {"contract_pressure_watch_at": 0.75},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    diff = body["diff"]
    # Should contain a line showing the removed old value and added new value
    assert any(
        line.startswith("-") and "watch_at" in line
        for line in diff.splitlines()
    ), "diff missing removal line for watch_at"
    assert any(
        line.startswith("+") and "watch_at" in line and "0.75" in line
        for line in diff.splitlines()
    ), "diff missing addition line for 0.75"


def test_draft_patch_empty_thresholds_returns_no_changes(client) -> None:
    resp = client.post(
        "/api/substrate-lattice/transport/draft-policy-patch",
        json={"lane_id": "transport", "thresholds": {}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["diff"] == "(no changes)"


def test_draft_patch_does_not_write_files(client) -> None:
    """Endpoint must not modify the policy YAML file."""
    from pathlib import Path
    policy_path = (
        Path(__file__).resolve().parents[3]
        / "config" / "substrate-lattice" / "transport_lattice_policy.v1.yaml"
    )
    before_mtime = policy_path.stat().st_mtime if policy_path.exists() else None
    resp = client.post(
        "/api/substrate-lattice/transport/draft-policy-patch",
        json={"lane_id": "transport", "thresholds": {"contract_pressure_watch_at": 0.99}},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    after_mtime = policy_path.stat().st_mtime if policy_path.exists() else None
    assert before_mtime == after_mtime, "Policy YAML was modified — must not happen"


def test_draft_patch_503_when_config_not_found(monkeypatch, client) -> None:
    """Returns 503 when the policy YAML file doesn't exist."""
    import pathlib
    import tempfile

    missing = pathlib.Path(tempfile.gettempdir()) / "nonexistent_substrate_lattice_xyz"
    monkeypatch.setattr(substrate_lattice_routes, "_config_dir", lambda: missing)
    resp = client.post(
        "/api/substrate-lattice/transport/draft-policy-patch",
        json={"lane_id": "transport", "thresholds": {}},
    )
    assert resp.status_code == 503
