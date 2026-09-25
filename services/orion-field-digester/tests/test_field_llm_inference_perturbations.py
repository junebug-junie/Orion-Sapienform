"""llm_inference_node deltas -> node inference_failure_pressure -> capability:llm_inference."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.schemas.state_delta import StateDeltaV1

from app.digestion.decay import NODE_DECAY_CHANNELS
from app.digestion.diffusion import apply_diffusion
from app.digestion.perturbation import apply_perturbations
from app.graph.lattice import load_lattice
from app.ingest.state_deltas import delta_to_perturbations
from app.tensor.channels import NODE_CHANNELS
from app.tensor.field_state import empty_field_state

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[3]


def _delta(hints: dict, *, node_id: str = "circe", operation: str = "update") -> StateDeltaV1:
    return StateDeltaV1(
        delta_id="delta_llm_1",
        target_projection="active_llm_inference_projection",
        target_kind="llm_inference_node",
        target_id=f"llm_node:{node_id}",
        operation=operation,
        after={"node_id": node_id, "pressure_hints": hints, "calls": 4},
        caused_by_event_ids=["gev_1"],
        reducer_id="llm_inference_reducer",
    )


def test_hint_becomes_a_replace_perturbation_on_the_serving_node():
    out = delta_to_perturbations(_delta({"inference_failure_pressure": 0.25}))
    assert len(out) == 1
    p = out[0]
    assert (p.node_id, p.channel, p.intensity, p.mode) == ("node:circe", "inference_failure_pressure", 0.25, "replace")


def test_no_hint_writes_nothing():
    """No upstream traffic that window: the reducer omits the hint, and nothing may
    be written -- a fabricated 0.0 would read as 'backend confirmed healthy'."""
    assert delta_to_perturbations(_delta({})) == []


def test_intensity_is_clamped():
    assert delta_to_perturbations(_delta({"inference_failure_pressure": 7.0}))[0].intensity == 1.0


def test_channel_is_declared_and_holds_instead_of_decaying():
    """Decay would fade a real failure reading into a fake calm 0.0 whenever callers
    stop calling the node. replace-mode writes still move it back down on the next
    measured window, so holding is not a ratchet."""
    assert "inference_failure_pressure" in NODE_CHANNELS
    assert "inference_failure_pressure" not in NODE_DECAY_CHANNELS


def test_failure_reading_survives_idle_minutes_and_clears_on_next_window():
    from datetime import timedelta

    from app.digestion.decay import apply_decay

    lattice = load_lattice(REPO / "config" / "field" / "orion_field_topology.v1.yaml")
    state = empty_field_state(lattice=lattice, now=NOW, tick_id="t")
    apply_perturbations(state, delta_to_perturbations(_delta({"inference_failure_pressure": 1.0})), now=NOW)
    for i in range(1, 300):  # ten idle minutes of 2s ticks
        apply_decay(state, decay_rate=0.92, now=NOW + timedelta(seconds=2 * i), staleness_threshold_sec=90.0)
    assert state.node_vectors["node:circe"]["inference_failure_pressure"] == 1.0
    apply_perturbations(state, delta_to_perturbations(_delta({"inference_failure_pressure": 0.0})), now=NOW)
    assert state.node_vectors["node:circe"]["inference_failure_pressure"] == 0.0


def test_live_topology_carries_failures_to_llm_inference_reliability_only():
    # The old compatibility alias topology file was deleted
    # 2026-09-25; the canonical file is the only topology.
    for name in ("orion_field_topology.v1.yaml",):
        lattice = load_lattice(REPO / "config" / "field" / name)
        state = empty_field_state(lattice=lattice, now=NOW, tick_id="t")
        apply_perturbations(state, delta_to_perturbations(_delta({"inference_failure_pressure": 1.0})), now=NOW)
        assert state.node_vectors["node:circe"]["inference_failure_pressure"] == 1.0
        apply_diffusion(state, diffusion_rate=1.0)
        cap = state.capability_vectors["capability:llm_inference"]
        assert cap["reliability_pressure"] == 0.85, name  # circe edge weight
        assert state.capability_provenance["capability:llm_inference"]["reliability_pressure"] == "node:circe"
        # failures do not masquerade as load
        assert cap["pressure"] == 0.0
        assert cap["reasoning_pressure"] == 0.0


def test_field_gate_is_per_lane_and_default_off(monkeypatch):
    from types import SimpleNamespace

    import app.settings as settings_mod
    from app.worker import delta_digestion_enabled

    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.delenv("ENABLE_LLM_INFERENCE_FIELD_DIGESTION", raising=False)
    settings_mod._settings = None
    try:
        defaults = settings_mod.get_settings()
        assert defaults.enable_llm_inference_field_digestion is False
    finally:
        settings_mod._settings = None

    off = SimpleNamespace(enable_transport_field_digestion=True, enable_llm_inference_field_digestion=False)
    on = SimpleNamespace(enable_transport_field_digestion=False, enable_llm_inference_field_digestion=True)
    assert delta_digestion_enabled("llm_inference_node", off) is False
    assert delta_digestion_enabled("llm_inference_node", on) is True
    assert delta_digestion_enabled("transport_bus", off) is True
    assert delta_digestion_enabled("transport_bus", on) is False
    assert delta_digestion_enabled("execution_run", off) is True
