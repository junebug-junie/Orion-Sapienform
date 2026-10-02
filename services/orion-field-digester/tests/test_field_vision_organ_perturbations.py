"""vision_organ deltas -> node:substrate.vision_organ -> capability:vision."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from orion.schemas.field_state import FieldStateV1
from orion.schemas.state_delta import StateDeltaV1

from app.digestion.decay import EXPIRING_NODE_CHANNELS, NODE_DECAY_CHANNELS, expire_unrefreshed_channels
from app.digestion.diffusion import apply_diffusion
from app.digestion.perturbation import apply_perturbations
from app.graph.lattice import load_lattice
from app.ingest.state_deltas import delta_to_perturbations
from app.tensor.channels import NODE_CHANNELS, RETIRED_PSEUDO_NODES, SINGLE_OBSERVER_NODE_CHANNELS
from app.tensor.field_state import empty_field_state
from app.tensor.reconcile import reconcile_field_state_with_lattice

NOW = datetime(2026, 10, 2, 4, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[3]
NODE = "node:substrate.vision_organ"
CHANNELS = ("vision_frame_staleness", "vision_processing_failure_pressure")


def _delta(hints: dict, *, delta_id: str = "delta_vision_1", node_id: str = NODE) -> StateDeltaV1:
    return StateDeltaV1(
        delta_id=delta_id,
        target_projection="active_vision_organ_projection",
        target_kind="vision_organ",
        target_id=node_id,
        operation="update",
        after={"node_id": node_id, "pressure_hints": hints, "status": "reporting"},
        caused_by_event_ids=["gev_1"],
        reducer_id="vision_organ_reducer",
    )


def _lattice():
    return load_lattice(REPO / "config" / "field" / "orion_field_topology.v1.yaml")


def test_hints_become_replace_perturbations_on_the_organ_node() -> None:
    out = delta_to_perturbations(_delta({"vision_frame_staleness": 0.5, "vision_processing_failure_pressure": 0.1}))
    assert sorted((p.node_id, p.channel, p.intensity, p.mode) for p in out) == [
        (NODE, "vision_frame_staleness", 0.5, "replace"),
        (NODE, "vision_processing_failure_pressure", 0.1, "replace"),
    ]


def test_omitted_failure_hint_writes_nothing_for_it() -> None:
    out = delta_to_perturbations(_delta({"vision_frame_staleness": 1.0}))
    assert [p.channel for p in out] == ["vision_frame_staleness"]


def test_intensity_is_clamped() -> None:
    assert delta_to_perturbations(_delta({"vision_frame_staleness": 9.0}))[0].intensity == 1.0


def test_channels_are_declared_single_observer_and_expire_instead_of_decaying() -> None:
    for ch in CHANNELS:
        assert ch in NODE_CHANNELS
        assert ch not in NODE_DECAY_CHANNELS, "decay would fade an outage into a fake calm 0.0"
        assert SINGLE_OBSERVER_NODE_CHANNELS[ch] == NODE
        assert EXPIRING_NODE_CHANNELS[ch] == 300.0


def test_a_stopped_lane_reads_unmeasured_not_last_value() -> None:
    state = empty_field_state(lattice=_lattice(), now=NOW, tick_id="t")
    apply_perturbations(state, delta_to_perturbations(_delta({"vision_frame_staleness": 0.0})), now=NOW)
    expire_unrefreshed_channels(state, now=NOW + timedelta(seconds=200))
    assert state.node_vectors[NODE]["vision_frame_staleness"] == 0.0
    expire_unrefreshed_channels(state, now=NOW + timedelta(seconds=301))
    assert "vision_frame_staleness" not in state.node_vectors[NODE]


def test_live_topology_carries_both_readings_into_capability_vision() -> None:
    state = empty_field_state(lattice=_lattice(), now=NOW, tick_id="t")
    apply_perturbations(
        state,
        delta_to_perturbations(_delta({"vision_frame_staleness": 1.0, "vision_processing_failure_pressure": 1.0})),
        now=NOW,
    )
    apply_diffusion(state, diffusion_rate=1.0)
    cap = state.capability_vectors["capability:vision"]
    assert cap["pressure"] == 0.85
    assert cap["reliability_pressure"] == 0.85
    assert state.capability_provenance["capability:vision"]["pressure"] == NODE


def test_calm_reading_is_attributed_not_anonymous() -> None:
    state = empty_field_state(lattice=_lattice(), now=NOW, tick_id="t")
    apply_perturbations(
        state,
        delta_to_perturbations(_delta({"vision_frame_staleness": 0.0, "vision_processing_failure_pressure": 0.0})),
        now=NOW,
    )
    apply_diffusion(state, diffusion_rate=1.0)
    assert state.capability_vectors["capability:vision"]["pressure"] == 0.0
    assert state.capability_provenance["capability:vision"]["pressure"] == NODE


def test_retired_vision_node_is_pruned_and_cannot_be_resurrected() -> None:
    assert "node:substrate.vision" in RETIRED_PSEUDO_NODES
    state = FieldStateV1(
        generated_at=NOW,
        tick_id="t",
        node_vectors={
            "node:substrate.vision": {"prediction_error": 0.0},
            NODE: {"vision_frame_staleness": 0.0},
        },
        node_vector_updated_at={"node:substrate.vision": {"prediction_error": NOW}},
    )
    out = reconcile_field_state_with_lattice(state, lattice=_lattice())
    assert "node:substrate.vision" not in out.node_vectors
    assert out.node_vectors[NODE]["vision_frame_staleness"] == 0.0
    assert delta_to_perturbations(_delta({"vision_frame_staleness": 1.0}, node_id="node:substrate.vision")) == []


def test_single_observer_channels_never_seed_other_nodes() -> None:
    state = empty_field_state(lattice=_lattice(), now=NOW, tick_id="t")
    out = reconcile_field_state_with_lattice(state, lattice=_lattice())
    for node_id, vec in out.node_vectors.items():
        if node_id != NODE:
            for ch in CHANNELS:
                assert ch not in vec, (node_id, ch)


def test_field_gate_is_per_lane_and_default_off(monkeypatch) -> None:
    import app.settings as settings_mod
    from app.worker import delta_digestion_enabled

    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.delenv("ENABLE_VISION_ORGAN_FIELD_DIGESTION", raising=False)
    settings_mod._settings = None
    try:
        assert settings_mod.get_settings().enable_vision_organ_field_digestion is False
    finally:
        settings_mod._settings = None
    assert delta_digestion_enabled("vision_organ", SimpleNamespace(enable_vision_organ_field_digestion=False)) is False
    assert delta_digestion_enabled("vision_organ", SimpleNamespace(enable_vision_organ_field_digestion=True)) is True
