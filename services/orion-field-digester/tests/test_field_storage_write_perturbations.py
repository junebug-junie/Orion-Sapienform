"""storage_write deltas -> node:substrate.storage_write write_failure_pressure ->
capability:storage reliability_pressure. Deltas come from the real reducer
(orion/substrate/storage_write_loop/) so this reader stays tied to what the
lane actually writes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from app.digestion.decay import EXPIRING_NODE_CHANNELS, NODE_DECAY_CHANNELS
from app.digestion.diffusion import apply_diffusion
from app.digestion.perturbation import apply_perturbations
from app.graph.lattice import load_lattice
from app.ingest.state_deltas import delta_to_perturbations
from app.tensor.channels import NODE_CHANNELS, SINGLE_OBSERVER_NODE_CHANNELS
from app.tensor.field_state import empty_field_state
from app.tensor.reconcile import reconcile_field_state_with_lattice
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.storage_write_loop.pipeline import empty_storage_write_projection
from orion.substrate.storage_write_loop.reducer import reduce_storage_write_trace_events

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[3]
NODE = "node:substrate.storage_write"
CAP = "capability:storage"


def _delta(committed: int, failed: int, *, cls: str = "serialization", family: str = "home_cooling_sample"):
    trace = "sql_writer.storage:athena:20261002T120000Z"
    events = []
    if committed or failed:
        eid = f"{trace}:00:storage_write_window_observed"
        classes = f"{cls}:{failed}" if failed else "none"
        events.append(GrammarEventV1(
            event_id=eid, event_kind="atom_emitted", trace_id=trace, emitted_at=NOW,
            atom=GrammarAtomV1(atom_id=eid, trace_id=trace, atom_type="observation",
                               semantic_role="storage_write_window_observed", layer="storage",
                               summary=f"family={family} attempted={committed + failed} committed={committed} "
                                       f"duplicate=0 failed={failed} classes={classes} p50_ms=3 p95_ms=9"),
            provenance=GrammarProvenanceV1(source_service="orion-sql-writer"),
        ))
    eid = f"{trace}:99:storage_writer_window_completed"
    events.append(GrammarEventV1(
        event_id=eid, event_kind="atom_emitted", trace_id=trace, emitted_at=NOW,
        atom=GrammarAtomV1(atom_id=eid, trace_id=trace, atom_type="observation",
                           semantic_role="storage_writer_window_completed", layer="storage",
                           summary="writer=athena window_sec=60.0"),
        provenance=GrammarProvenanceV1(source_service="orion-sql-writer"),
    ))
    _proj, receipt = reduce_storage_write_trace_events(
        events=events, projection=empty_storage_write_projection(now=NOW), now=NOW
    )
    (delta,) = receipt.state_deltas
    return delta


def _lattice():
    return load_lattice(REPO / "config" / "field" / "orion_field_topology.v1.yaml")


def test_receipt_becomes_a_replace_perturbation_on_the_storage_node():
    (p,) = delta_to_perturbations(_delta(0, 7))
    assert (p.node_id, p.channel, p.intensity, p.mode) == (NODE, "write_failure_pressure", 0.7, "replace")


def test_not_measured_window_writes_nothing():
    assert delta_to_perturbations(_delta(0, 0)) == []


def test_channel_is_declared_single_observer_expiring_and_never_decays():
    assert "write_failure_pressure" in NODE_CHANNELS
    assert "write_failure_pressure" not in NODE_DECAY_CHANNELS
    assert SINGLE_OBSERVER_NODE_CHANNELS["write_failure_pressure"] == NODE
    assert EXPIRING_NODE_CHANNELS["write_failure_pressure"] == 180.0


def test_failing_writes_reach_storage_reliability_not_load():
    lattice = _lattice()
    state = empty_field_state(lattice=lattice, now=NOW, tick_id="t")
    apply_perturbations(state, delta_to_perturbations(_delta(0, 10)), now=NOW)
    state = reconcile_field_state_with_lattice(state, lattice=lattice)
    assert state.node_vectors[NODE]["write_failure_pressure"] == 1.0
    apply_diffusion(state, diffusion_rate=1.0)
    cap = state.capability_vectors[CAP]
    assert cap["reliability_pressure"] == 0.85  # edge weight
    assert state.capability_provenance[CAP]["reliability_pressure"] == NODE


def test_calm_reading_is_a_measured_zero_attributed_to_the_writer():
    lattice = _lattice()
    state = empty_field_state(lattice=lattice, now=NOW, tick_id="t")
    apply_perturbations(state, delta_to_perturbations(_delta(300, 0)), now=NOW)
    apply_diffusion(state, diffusion_rate=1.0)
    assert state.capability_vectors[CAP]["reliability_pressure"] == 0.0
    assert state.capability_provenance[CAP]["reliability_pressure"] == NODE


def test_channel_is_never_seeded_on_other_nodes():
    lattice = _lattice()
    state = reconcile_field_state_with_lattice(
        empty_field_state(lattice=lattice, now=NOW, tick_id="t"), lattice=lattice
    )
    for node_id, vec in state.node_vectors.items():
        assert "write_failure_pressure" not in vec, node_id


def test_silent_writer_expires_to_unmeasured_not_held_calm():
    """A dead writer (or a Postgres outage long enough that its own report cannot
    be stored) must not leave its last calm 0.0 standing as a live reading."""
    from app.tensor.update_rules import run_digestion_tick

    lattice = _lattice()
    state = empty_field_state(lattice=lattice, now=NOW, tick_id="t")
    state.generated_at = NOW
    apply_perturbations(state, delta_to_perturbations(_delta(300, 0)), now=NOW)

    def tick(at):
        state.generated_at = at
        run_digestion_tick(
            state,
            perturbations=[],
            decay_rate=0.92,
            diffusion_rate=1.0,
            staleness_threshold_sec=90.0,
            store=None,
            significance_window_seconds=60.0,
            significance_check_interval_sec=1e9,
        )

    tick(NOW + timedelta(seconds=170))
    assert state.node_vectors[NODE]["write_failure_pressure"] == 0.0
    assert state.capability_provenance[CAP]["reliability_pressure"] == NODE
    tick(NOW + timedelta(seconds=181))
    assert "write_failure_pressure" not in state.node_vectors[NODE]
    assert "reliability_pressure" not in state.capability_provenance[CAP]


def test_field_gate_is_per_lane_and_default_off(monkeypatch):
    import app.settings as settings_mod
    from app.worker import delta_digestion_enabled

    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.delenv("ENABLE_STORAGE_WRITE_FIELD_DIGESTION", raising=False)
    settings_mod._settings = None
    try:
        assert settings_mod.get_settings().enable_storage_write_field_digestion is False
    finally:
        settings_mod._settings = None
    off = SimpleNamespace(enable_storage_write_field_digestion=False, enable_rpc_delivery_field_digestion=True)
    on = SimpleNamespace(enable_storage_write_field_digestion=True, enable_rpc_delivery_field_digestion=False)
    assert delta_digestion_enabled("storage_write", off) is False
    assert delta_digestion_enabled("storage_write", on) is True
    assert delta_digestion_enabled("rpc_delivery", on) is False
