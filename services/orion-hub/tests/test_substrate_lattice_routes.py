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


def test_lanes_grammar_producers_are_not_marked_planned(client) -> None:
    """All four lanes' grammar producers are live on orion:grammar:event and
    listed in orion/bus/channels.yaml; the rail said "planned" for two of them."""
    import yaml as _yaml
    doc = _yaml.safe_load((REPO_ROOT / "orion" / "bus" / "channels.yaml").read_text(encoding="utf-8"))
    entry = next(c for c in doc["channels"] if c["name"] == "orion:grammar:event")
    producers = set(entry["producer_services"])
    for lane in client.get("/api/substrate-lattice/lanes").json():
        assert lane["status"] == "live"
        assert lane["source_service"] in producers


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


def test_transport_latest_returns_proof_chain(client) -> None:
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain",
        return_value=_sample_proof_chain_for_gates()
    ):
        resp = client.get("/api/substrate-lattice/transport/latest")
    assert resp.status_code == 200
    body = resp.json()
    assert "transport" in body
    assert "freshness_threshold_sec" in body
    assert "verdict" in body
    transport = body["transport"]
    for key in ("m3", "m3_receipts", "m4", "m5", "l7", "l8", "l9", "l10", "l11"):
        assert key in transport, f"Missing key: {key}"
        layer = transport[key]
        assert "status" in layer
        assert "source_table" in layer
        assert "values" in layer


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
    # First query is the projection query — returns None
    fake = _make_engine_sequence(None)
    with patch.object(substrate_lattice_routes, "_engine", return_value=fake):
        result = substrate_lattice_routes._load_transport_proof_chain()
    assert result is None


def test_load_transport_proof_chain_empty_buses_yields_empty_bus_summary(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import json as _json
    proj = {
        "schema_version": "transport_bus.projection.v1",
        "updated_at": "2026-06-08T00:00:00+00:00",
        "projection_id": "active_transport_bus_projection",
        "buses": {},
    }
    fake = _make_engine_sequence(
        {"projection_json": _json.dumps(proj), "updated_at": "2026-06-08T00:00:00+00:00"},  # M3
        None,  # M3 receipts
        None,  # M4 field
        None,  # M5 attention
        None,  # L7 proposals
        None,  # L8 policy
        None,  # L9 dispatch
        None,  # L10 feedback
        None,  # L11 consolidation
    )
    with patch.object(substrate_lattice_routes, "_engine", return_value=fake):
        result = substrate_lattice_routes._load_transport_proof_chain()
    assert result is not None
    assert "transport" in result
    m3 = result["transport"]["m3"]
    assert m3["status"] in ("fresh", "stale", "missing")
    assert m3["values"]["buses"] == {}
    assert result["transport"]["l11"]["values"].get("motifs", []) == []


# ── /transport/gates ─────────────────────────────────────────────


def _sample_proof_chain_for_gates(
    bus_age_sec: float = 5.0,
    contract_pressure: float = 1.0,
    stream_backlog_pressure: float = 0.0,
    dispatch_mode: str = "dry_run",
    receipts: list | None = None,
    source_trace_id: str = "bus.transport:abc",
    freshness_threshold_sec: int = 60,
    capability_transport_bucket: str | None = "dominant_targets",
) -> dict:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    ts = (now - timedelta(seconds=bus_age_sec)).isoformat()
    is_fresh = bus_age_sec <= freshness_threshold_sec
    m3_status = "fresh" if is_fresh else "stale"

    receipt_list = receipts if receipts is not None else [{"receipt_id": "r1"}]

    return {
        "transport": {
            "m3": {
                "status": m3_status,
                "source_table": "substrate_transport_bus_projection",
                "timestamp": ts,
                "age_sec": bus_age_sec,
                "values": {
                    "projection_id": "active_transport_bus_projection",
                    "observed_at": ts,
                    # Per-bus readings, the real TransportBusProjectionV1 shape:
                    # the projection's top level carries no pressure fields.
                    # The simulator/lattice-values panel read these via
                    # _channel_value() (max across buses).
                    "buses": {
                        "bus:athena": {
                            "source_trace_id": source_trace_id,
                            "stream_backlog_health": 1.0,
                            "stream_backlog_pressure": stream_backlog_pressure,
                            "contract_pressure": contract_pressure,
                            "catalog_drift_pressure": 0.0,
                            "observer_failure_pressure": 0.0,
                            "delivery_confidence": 1.0,
                        }
                    } if source_trace_id else {},
                },
            },
            "m3_receipts": {
                "status": "fresh",
                "source_table": "substrate_reduction_receipts",
                "timestamp": ts,
                "age_sec": bus_age_sec,
                "values": {"count": len(receipt_list), "receipts": receipt_list},
            },
            "m4": {
                # m3_status (fresh/stale) reused here rather than an independent
                # calculation -- both lanes share the same bus_age_sec test knob,
                # and _compute_gates' contract gate now genuinely depends on M4's
                # freshness (it's where the value is read from), not M3's, so this
                # must actually vary with bus_age_sec, not be hardcoded "fresh".
                "status": m3_status,
                "source_table": "substrate_field_state",
                "timestamp": ts,
                "age_sec": bus_age_sec,
                # capability:transport's real field vector -- what _compute_gates
                # actually reads post 2026-07-27. pressure/contract_pressure mirror
                # the same test knobs used for M3 above so both code paths can be
                # exercised from one fixture without a parameter per code path.
                "values": {
                    "field_vector": {
                        "pressure": stream_backlog_pressure,
                        "contract_pressure": contract_pressure,
                    },
                    "has_transport_vector": True,
                },
            },
            "m5": {
                "status": "fresh",
                "source_table": "substrate_attention_frames",
                "timestamp": ts,
                "age_sec": bus_age_sec,
                "values": {
                    "frame_id": "frame:1",
                    "dominant_targets": [
                        {"target_id": "capability:transport", "bucket": "dominant_targets",
                         "salience_score": None, "suggested_observation_mode": None,
                         "dominant_channels": [], "reasons": []}
                    ] if capability_transport_bucket == "dominant_targets" else [],
                    "capability_targets": [
                        {"target_id": "capability:transport", "bucket": "capability_targets",
                         "salience_score": None, "suggested_observation_mode": None,
                         "dominant_channels": [], "reasons": []}
                    ] if capability_transport_bucket == "capability_targets" else [],
                    "suppressed_targets": [
                        {"target_id": "capability:transport", "bucket": "suppressed_targets",
                         "salience_score": None, "suggested_observation_mode": None,
                         "dominant_channels": [], "reasons": []}
                    ] if capability_transport_bucket == "suppressed_targets" else [],
                    "capability_transport_bucket": capability_transport_bucket,
                },
            },
            "l7": {
                "status": "fresh",
                "source_table": "substrate_proposal_frames",
                "timestamp": ts,
                "age_sec": bus_age_sec,
                "values": {"count": 1, "transport_count": 1, "candidates": []},
            },
            "l8": {
                "status": "fresh",
                "source_table": "substrate_policy_decision_frames",
                "timestamp": ts,
                "age_sec": bus_age_sec,
                "values": {"approved_count": 1, "rejected_count": 0, "policy_mode": "dry_run"},
            },
            "l9": {
                "status": "fresh",
                "source_table": "substrate_execution_dispatch_frames",
                "timestamp": ts,
                "age_sec": bus_age_sec,
                "values": {"dispatch_mode": dispatch_mode, "dispatch_count": 0, "blocked_count": 0},
            },
            "l10": {
                "status": "fresh",
                "source_table": "substrate_feedback_frames",
                "timestamp": ts,
                "age_sec": bus_age_sec,
                "values": {"outcome_status": "dry_run_only", "feedback_kind": None},
            },
            "l11": {
                "status": "fresh",
                "source_table": "substrate_consolidation_frames",
                "timestamp": ts,
                "age_sec": bus_age_sec,
                "values": {"frame_id": "cons:1", "motifs": []},
            },
        },
        "freshness_threshold_sec": freshness_threshold_sec,
        "verdict": "Transport lane live and fresh through all layers.",
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
    chain = _sample_proof_chain_for_gates(bus_age_sec=61.0)
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
    chain = _sample_proof_chain_for_gates(stream_backlog_pressure=0.0)
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
    chain = _sample_proof_chain_for_gates(stream_backlog_pressure=0.5)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["pressure"]["state"] == "watch"


def test_gates_pressure_reads_m4_field_vector_not_m3_top_level(client) -> None:
    """Regression test, 2026-07-27: the pressure/contract gates used to read
    m3.values.stream_backlog_pressure/contract_pressure, keys that don't exist
    on the real TransportBusProjectionV1 schema (only nested under
    buses[bus_id][...]) -- always silently defaulting to 0.0/"quiet"/"pass"
    regardless of real bus state. Proves the fix by setting M3's top-level
    keys to values that would trip the OLD (buggy) thresholds if read, while
    M4's field_vector (what the code now actually reads) says otherwise."""
    chain = _sample_proof_chain_for_gates(
        stream_backlog_pressure=0.9, contract_pressure=0.9,
    )
    # Poison M3's top-level values -- if the code regresses to reading these,
    # both gates would report "watch"/high; with the fix, M4 alone decides.
    chain["transport"]["m3"]["values"]["stream_backlog_pressure"] = 0.9
    chain["transport"]["m3"]["values"]["contract_pressure"] = 0.9
    chain["transport"]["m4"]["values"]["field_vector"]["pressure"] = 0.0
    chain["transport"]["m4"]["values"]["field_vector"]["contract_pressure"] = 0.0
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["pressure"]["state"] == "quiet"
    assert gates["contract"]["state"] == "quiet"
    assert "bus_synaptic_pressure=0.00" in gates["pressure"]["reason"]


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
                "stream_backlog_health": 1.0, "delivery_confidence": 1.0,
                "stream_backlog_pressure": 0.0, "contract_pressure": 1.0,
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
        {"projection_json": _json.dumps(proj), "updated_at": "2026-06-08T01:00:00+00:00"},  # M3
        None,  # M3 receipts (all())
        None,  # M4 field
        None,  # M5 attention
        None,  # L7 proposals
        None,  # L8 policy
        None,  # L9 dispatch
        None,  # L10 feedback
        None,  # L11 consolidation
    )
    with patch.object(substrate_lattice_routes, "_engine", return_value=fake):
        result = substrate_lattice_routes._load_transport_proof_chain()
    assert result is not None
    assert "transport" in result
    assert "freshness_threshold_sec" in result
    assert "verdict" in result
    m3 = result["transport"]["m3"]
    assert m3["status"] in ("fresh", "stale", "missing")
    m3_vals = m3["values"]
    assert "buses" in m3_vals
    assert m3_vals["buses"]["bus:athena"]["source_trace_id"] == "bus.transport:abc"


# ── /transport/simulate ──────────────────────────────────────────


def test_simulate_returns_comparison_when_thresholds_change(client) -> None:
    chain = _sample_proof_chain_for_gates(contract_pressure=1.0, stream_backlog_pressure=0.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
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
        stream_backlog_pressure=0.0,
    )
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.post(
            "/api/substrate-lattice/transport/simulate",
            json={
                "lane_id": "transport",
                "thresholds": {
                    "contract_pressure_watch_at": 1.1,
                    "bus_synaptic_pressure_watch_at": 1.1,
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
        stream_backlog_pressure=0.1,  # below default watch_at=0.25 → no channels promote
    )
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
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
    """Simulate endpoint must not access the DB engine beyond loading the proof chain."""
    chain = _sample_proof_chain_for_gates(contract_pressure=1.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ), patch.object(substrate_lattice_routes, "_engine") as mock_engine:
        resp = client.post(
            "/api/substrate-lattice/transport/simulate",
            json={"lane_id": "transport", "thresholds": {}},
        )
    assert resp.status_code == 200
    mock_engine.assert_not_called()  # no direct engine access — load was fully mocked


def _policy_doc() -> dict:
    import yaml as _yaml
    path = REPO_ROOT / "config" / "substrate-lattice" / "transport_lattice_policy.v1.yaml"
    return _yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_policy_has_no_unread_weights_and_no_hub_mirror() -> None:
    """Kill-means-kill regression, 2026-09-25: dimension_weights was read by no
    process, and the hub's hand-copied _TRANSPORT_CHANNELS mirror of it was
    still keyed on the retired stream_backlog_pressure channel."""
    doc = _policy_doc()
    assert "dimension_weights" not in doc
    assert "healthy_idle" not in doc
    assert not hasattr(substrate_lattice_routes, "_TRANSPORT_CHANNELS")
    assert "stream_backlog_pressure" not in (doc.get("channels") or {})


def test_policy_ceilings_are_all_ranked() -> None:
    ceilings = {
        str(ch.get("action_ceiling"))
        for ch in (_policy_doc().get("channels") or {}).values()
    }
    unranked = ceilings - set(substrate_lattice_routes._CEILING_RANK)
    assert not unranked, f"policy action_ceiling values missing from _CEILING_RANK: {unranked}"


def test_latest_lattice_channels_come_from_policy_yaml(client) -> None:
    chain = _sample_proof_chain_for_gates(contract_pressure=0.6, stream_backlog_pressure=0.3)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/latest")
    assert resp.status_code == 200
    rows = {r["channel_id"]: r for r in resp.json()["lattice_channels"]}
    policy = _policy_doc()["channels"]
    assert list(rows) == list(policy)
    for ch_id, ch_def in policy.items():
        assert rows[ch_id]["watch_at"] == ch_def["watch_at"]
        assert rows[ch_id]["action_ceiling"] == ch_def["action_ceiling"]
    # bus_synaptic_pressure is read from M4 (the fixture's `pressure` knob),
    # the other channels from M3's per-bus rows.
    assert rows["bus_synaptic_pressure"]["value"] == 0.3
    assert rows["bus_synaptic_pressure"]["state"] == "watch"
    assert rows["contract_pressure"]["value"] == 0.6
    assert rows["contract_pressure"]["state"] == "watch"
    assert rows["catalog_drift_pressure"]["state"] == "quiet"


def test_lattice_channel_unmeasured_when_m4_stale_not_calm() -> None:
    chain = _sample_proof_chain_for_gates(stream_backlog_pressure=0.9)
    chain["transport"]["m4"]["status"] = "stale"
    rows = {
        r["channel_id"]: r
        for r in substrate_lattice_routes._lattice_channel_rows(
            chain, substrate_lattice_routes._policy_channels()
        )
    }
    assert rows["bus_synaptic_pressure"]["value"] is None
    assert rows["bus_synaptic_pressure"]["state"] == "unmeasured"


def test_channel_value_takes_max_across_buses() -> None:
    chain = _sample_proof_chain_for_gates(contract_pressure=0.2)
    chain["transport"]["m3"]["values"]["buses"]["bus:circe"] = {"contract_pressure": 0.7}
    value, source = substrate_lattice_routes._channel_value(chain, "contract_pressure")
    assert value == 0.7
    assert source == "M3 max(buses[*].contract_pressure)"


def test_simulate_salience_is_strongest_promoted_reading(client) -> None:
    # contract 0.6 (>= 0.50) and bus_synaptic 0.3 (>= 0.25) both promote.
    chain = _sample_proof_chain_for_gates(contract_pressure=0.6, stream_backlog_pressure=0.3)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.post(
            "/api/substrate-lattice/transport/simulate",
            json={"lane_id": "transport", "thresholds": {}},
        )
    body = resp.json()
    assert body["current"]["salience"] == 0.6
    assert set(body["current"]["promoted_channels"]) == {"contract_pressure", "bus_synaptic_pressure"}
    # read_only (bus_synaptic) outranks summarize (contract)
    assert body["current"]["action_ceiling"] == "read_only"


def test_simulate_unmeasured_channel_never_promotes(client) -> None:
    chain = _sample_proof_chain_for_gates(contract_pressure=0.0, stream_backlog_pressure=0.9)
    chain["transport"]["m4"]["status"] = "missing"
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.post(
            "/api/substrate-lattice/transport/simulate",
            json={"lane_id": "transport", "thresholds": {}},
        )
    body = resp.json()
    assert body["current"]["bucket"] == "suppressed_targets"
    assert "bus_synaptic_pressure" in body["current"]["unmeasured_channels"]
    assert body["channel_values"]["bus_synaptic_pressure"] is None


def test_simulate_reports_retired_channel_threshold_as_ignored(client) -> None:
    chain = _sample_proof_chain_for_gates(contract_pressure=1.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.post(
            "/api/substrate-lattice/transport/simulate",
            json={"lane_id": "transport", "thresholds": {"stream_backlog_pressure_watch_at": 0.0}},
        )
    body = resp.json()
    assert body["ignored_thresholds"] == ["stream_backlog_pressure_watch_at"]
    assert "stream_backlog_pressure_watch_at" not in body["applied_thresholds"]
    assert body["changed"] is False


def test_simulate_503_when_policy_has_no_channels(client) -> None:
    chain = _sample_proof_chain_for_gates()
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ), patch.object(substrate_lattice_routes, "_load_yaml", return_value={}):
        resp = client.post(
            "/api/substrate-lattice/transport/simulate",
            json={"lane_id": "transport", "thresholds": {}},
        )
    assert resp.status_code == 503


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


# ── New V1.1 tests: load chain status classification ─────────────


def test_load_chain_fresh_status_when_age_below_threshold(monkeypatch) -> None:
    """M3 status is 'fresh' when projection updated_at is recent."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import json as _json
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    fresh_ts = (now - timedelta(seconds=5)).isoformat()
    proj = {"buses": {"bus:athena": {"source_trace_id": "trace:1", "stream_backlog_health": 1.0, "contract_pressure": 1.0}}}
    fake = _make_engine_sequence(
        {"projection_json": _json.dumps(proj), "updated_at": fresh_ts},
        None, None, None, None, None, None, None, None,
    )
    with patch.object(substrate_lattice_routes, "_engine", return_value=fake):
        result = substrate_lattice_routes._load_transport_proof_chain(freshness_threshold_sec=60)
    assert result is not None
    assert result["transport"]["m3"]["status"] == "fresh"
    assert result["transport"]["m3"]["age_sec"] < 60


def test_load_chain_stale_status_when_age_above_threshold(monkeypatch) -> None:
    """M3 status is 'stale' when projection updated_at is old."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import json as _json
    from datetime import datetime, timedelta, timezone

    old_ts = (datetime.now(timezone.utc) - timedelta(hours=75)).isoformat()
    proj = {"buses": {}}
    fake = _make_engine_sequence(
        {"projection_json": _json.dumps(proj), "updated_at": old_ts},
        None, None, None, None, None, None, None, None,
    )
    with patch.object(substrate_lattice_routes, "_engine", return_value=fake):
        result = substrate_lattice_routes._load_transport_proof_chain(freshness_threshold_sec=60)
    assert result is not None
    assert result["transport"]["m3"]["status"] == "stale"
    assert result["transport"]["m3"]["age_sec"] > 60


def test_load_chain_missing_layer_when_no_row(monkeypatch) -> None:
    """L11 status is 'missing' when consolidation frames table has no rows."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    import json as _json
    from datetime import datetime, timedelta, timezone

    fresh_ts = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    proj = {"buses": {}}
    fake = _make_engine_sequence(
        {"projection_json": _json.dumps(proj), "updated_at": fresh_ts},
        None, None, None, None, None, None, None,
        None,  # L11 returns None
    )
    with patch.object(substrate_lattice_routes, "_engine", return_value=fake):
        result = substrate_lattice_routes._load_transport_proof_chain(freshness_threshold_sec=60)
    assert result is not None
    assert result["transport"]["l11"]["status"] == "missing"
    assert result["transport"]["l11"]["values"] == {}


# ── New V1.1 tests: M5 capability:transport detection ────────────


def test_m5_capability_transport_detected_in_dominant_targets(monkeypatch) -> None:
    """capability_transport_bucket is set when target is in dominant_targets."""
    import json as _json
    from datetime import datetime, timedelta, timezone

    ts = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    proj = {"buses": {}}
    attn_frame = {
        "frame_id": "f1",
        "dominant_targets": [{"target_id": "capability:transport"}],
        "capability_targets": [],
        "suppressed_targets": [],
    }
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    fake = _make_engine_sequence(
        {"projection_json": _json.dumps(proj), "updated_at": ts},
        None,  # receipts
        None,  # M4 field
        {"frame_json": _json.dumps(attn_frame), "generated_at": ts},  # M5 attention
        None, None, None, None, None,  # L7-L11
    )
    with patch.object(substrate_lattice_routes, "_engine", return_value=fake):
        result = substrate_lattice_routes._load_transport_proof_chain()
    assert result is not None
    m5_vals = result["transport"]["m5"]["values"]
    assert m5_vals["capability_transport_bucket"] == "dominant_targets"
    assert m5_vals["dominant_targets"][0]["target_id"] == "capability:transport"
    assert m5_vals["dominant_targets"][0]["bucket"] == "dominant_targets"


def test_m5_capability_transport_detected_in_capability_targets(monkeypatch) -> None:
    """capability_transport_bucket is 'capability_targets' when only in that bucket."""
    import json as _json
    from datetime import datetime, timedelta, timezone

    ts = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    proj = {"buses": {}}
    attn_frame = {
        "frame_id": "f1",
        "dominant_targets": ["some:other:target"],
        "capability_targets": [{"target_id": "capability:transport", "salience_score": 0.6}],
        "suppressed_targets": [],
    }
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    fake = _make_engine_sequence(
        {"projection_json": _json.dumps(proj), "updated_at": ts},
        None,
        None,
        {"frame_json": _json.dumps(attn_frame), "generated_at": ts},
        None, None, None, None, None,
    )
    with patch.object(substrate_lattice_routes, "_engine", return_value=fake):
        result = substrate_lattice_routes._load_transport_proof_chain()
    m5_vals = result["transport"]["m5"]["values"]
    assert m5_vals["capability_transport_bucket"] == "capability_targets"
    assert m5_vals["capability_targets"][0]["salience_score"] == 0.6


def test_m5_capability_transport_detected_in_suppressed_targets(monkeypatch) -> None:
    """capability_transport_bucket is 'suppressed_targets' when only suppressed."""
    import json as _json
    from datetime import datetime, timedelta, timezone

    ts = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    proj = {"buses": {}}
    attn_frame = {
        "frame_id": "f1",
        "dominant_targets": [],
        "capability_targets": [],
        "suppressed_targets": ["capability:transport"],  # string form
    }
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    fake = _make_engine_sequence(
        {"projection_json": _json.dumps(proj), "updated_at": ts},
        None,
        None,
        {"frame_json": _json.dumps(attn_frame), "generated_at": ts},
        None, None, None, None, None,
    )
    with patch.object(substrate_lattice_routes, "_engine", return_value=fake):
        result = substrate_lattice_routes._load_transport_proof_chain()
    m5_vals = result["transport"]["m5"]["values"]
    assert m5_vals["capability_transport_bucket"] == "suppressed_targets"
    # String target normalized to object
    assert m5_vals["suppressed_targets"][0]["target_id"] == "capability:transport"
    assert m5_vals["suppressed_targets"][0]["bucket"] == "suppressed_targets"


def test_m5_capability_transport_not_found(monkeypatch) -> None:
    """capability_transport_bucket is None when not in any bucket."""
    import json as _json
    from datetime import datetime, timedelta, timezone

    ts = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    proj = {"buses": {}}
    attn_frame = {
        "frame_id": "f1",
        "dominant_targets": ["capability:biometrics"],
        "capability_targets": [],
        "suppressed_targets": [],
    }
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    fake = _make_engine_sequence(
        {"projection_json": _json.dumps(proj), "updated_at": ts},
        None,
        None,
        {"frame_json": _json.dumps(attn_frame), "generated_at": ts},
        None, None, None, None, None,
    )
    with patch.object(substrate_lattice_routes, "_engine", return_value=fake):
        result = substrate_lattice_routes._load_transport_proof_chain()
    m5_vals = result["transport"]["m5"]["values"]
    assert m5_vals["capability_transport_bucket"] is None


# ── New V1.1 tests: additional gate assertions ────────────────────


def test_gates_contract_not_quiet_when_contract_pressure_1(client) -> None:
    """contract gate is 'watch' (not 'quiet') when contract_pressure=1.0."""
    chain = _sample_proof_chain_for_gates(contract_pressure=1.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    assert resp.status_code == 200
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["contract"]["state"] != "quiet", "contract gate must not be quiet when pressure=1.0"
    assert gates["contract"]["state"] == "watch"


def test_gates_contract_unknown_when_m3_stale(client) -> None:
    """contract gate is 'unknown' when M3 projection is stale."""
    chain = _sample_proof_chain_for_gates(bus_age_sec=3600.0, freshness_threshold_sec=60)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["contract"]["state"] == "unknown"


def test_gates_pressure_unknown_when_m4_stale(client) -> None:
    """Regression test, 2026-07-27 (review-caught gap): the pressure gate must
    go 'unknown' when M4 is stale/missing, exactly like the contract gate --
    reading from a stale/missing field vector as if it were a real 'quiet' or
    'watch' reading is the same failure class this whole patch exists to fix.
    Poisons the pressure value high (0.9) to prove it's the staleness check
    that suppresses it, not the value itself happening to read low."""
    chain = _sample_proof_chain_for_gates(bus_age_sec=3600.0, freshness_threshold_sec=60)
    chain["transport"]["m4"]["values"]["field_vector"]["pressure"] = 0.9
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["pressure"]["state"] == "unknown"


def test_gates_pressure_missing_when_m4_absent(client) -> None:
    """pressure gate is 'unknown' when M4 has no data at all (status=missing),
    not silently defaulting to 'quiet' via the 0.0 fallback."""
    chain = _sample_proof_chain_for_gates()
    chain["transport"]["m4"] = {
        "status": "missing", "source_table": "substrate_field_state",
        "timestamp": None, "age_sec": None, "values": None,
    }
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["pressure"]["state"] == "unknown"


def test_gates_attention_pass_when_capability_transport_present(client) -> None:
    """attention gate is 'pass' when capability:transport is in any bucket."""
    chain = _sample_proof_chain_for_gates(capability_transport_bucket="dominant_targets")
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert "attention" in gates
    assert gates["attention"]["state"] == "pass"


def test_gates_attention_blocked_when_capability_transport_absent(client) -> None:
    """attention gate is 'blocked' when capability:transport is not in any bucket."""
    chain = _sample_proof_chain_for_gates(capability_transport_bucket=None)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["attention"]["state"] == "blocked"


# ── New V1.1 tests: _normalize_targets ───────────────────────────


def test_normalize_targets_strings_become_objects() -> None:
    """String targets are normalized to object shape."""
    result = substrate_lattice_routes._normalize_targets(
        ["capability:transport", "capability:biometrics"],
        "dominant_targets"
    )
    assert len(result) == 2
    assert result[0]["target_id"] == "capability:transport"
    assert result[0]["bucket"] == "dominant_targets"
    assert result[0]["salience_score"] is None
    assert result[0]["dominant_channels"] == []


def test_normalize_targets_dicts_preserve_fields() -> None:
    """Dict targets with known fields are preserved."""
    raw = [{"target_id": "capability:transport", "salience_score": 0.75,
            "dominant_channels": ["contract_pressure"], "reasons": ["drift"]}]
    result = substrate_lattice_routes._normalize_targets(raw, "capability_targets")
    assert result[0]["salience_score"] == 0.75
    assert result[0]["dominant_channels"] == ["contract_pressure"]
    assert result[0]["bucket"] == "capability_targets"


def test_coerce_str_list_handles_dict_channel_shapes() -> None:
    """Attention frames may store dominant_channels as a dict, not a list."""
    assert substrate_lattice_routes._coerce_str_list({"cortex_exec_step_load": 0.7}) == ["cortex_exec_step_load"]
    result = substrate_lattice_routes._normalize_targets(
        [{"target_id": "capability:transport", "dominant_channels": {"cortex_exec_step_load": 0.7}}],
        "dominant_targets",
    )
    assert result[0]["dominant_channels"] == ["cortex_exec_step_load"]


# ── New V1.1 tests: response shape completeness ───────────────────


def test_transport_latest_404_when_no_projection_row(client) -> None:
    """Returns 404 — never falls back to schema defaults when projection missing."""
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=None
    ):
        resp = client.get("/api/substrate-lattice/transport/latest")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "transport_projection_not_found"


def test_transport_latest_includes_verdict(client) -> None:
    chain = _sample_proof_chain_for_gates()
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/latest")
    assert resp.status_code == 200
    body = resp.json()
    assert "verdict" in body
    assert isinstance(body["verdict"], str)
    assert len(body["verdict"]) > 0


def test_transport_latest_layer_has_source_metadata(client) -> None:
    chain = _sample_proof_chain_for_gates()
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/latest")
    transport = resp.json()["transport"]
    m3 = transport["m3"]
    assert "source_table" in m3
    assert "timestamp" in m3
    assert "age_sec" in m3
    assert "status" in m3
