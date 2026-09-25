"""rpc_delivery deltas -> node:substrate.rpc_delivery rpc_timeout_pressure ->
capability:transport reliability_pressure. Fed by the real producer-side
receipt builder so the reader stays tied to what the bridge actually writes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from app.digestion.decay import NODE_DECAY_CHANNELS, apply_decay
from app.digestion.diffusion import apply_diffusion
from app.digestion.perturbation import apply_perturbations
from app.graph.lattice import load_lattice
from app.ingest.state_deltas import delta_to_perturbations
from app.tensor.channels import NODE_CHANNELS
from app.tensor.field_state import empty_field_state
from app.tensor.reconcile import reconcile_field_state_with_lattice
from orion.substrate.rpc_delivery import RpcDeliveryWindow, rpc_delivery_receipt

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[3]
NODE = "node:substrate.rpc_delivery"


def _delta(successes: int, timeouts: int, hop: str = "orion:exec:request:LLMGatewayService"):
    win = RpcDeliveryWindow()
    win.fold(
        {
            "service": "cortex-exec",
            "instance": "background",
            "window_end": NOW.isoformat(),
            "channel_latency": {hop: {"success_count": successes, "timeout_count": timeouts}},
        }
    )
    reading = win.reading(NOW.timestamp())
    (delta,) = rpc_delivery_receipt(reading, now=NOW).state_deltas
    return delta


def _lattice():
    return load_lattice(REPO / "config" / "field" / "orion_field_topology.v1.yaml")


def test_receipt_becomes_a_replace_perturbation_on_the_bridge_node():
    (p,) = delta_to_perturbations(_delta(18, 2))
    assert (p.node_id, p.channel, p.intensity, p.mode) == (NODE, "rpc_timeout_pressure", 0.1, "replace")


def test_delta_without_the_hint_writes_nothing():
    d = _delta(1, 0).model_copy(update={"after": {"node_id": NODE, "pressure_hints": {}}})
    assert delta_to_perturbations(d) == []


def test_channel_is_declared_and_holds_instead_of_decaying():
    assert "rpc_timeout_pressure" in NODE_CHANNELS
    assert "rpc_timeout_pressure" not in NODE_DECAY_CHANNELS


def test_pseudo_node_survives_reconcile_and_reaches_transport_reliability():
    lattice = _lattice()
    state = empty_field_state(lattice=lattice, now=NOW, tick_id="t")
    apply_perturbations(state, delta_to_perturbations(_delta(0, 10)), now=NOW)
    state = reconcile_field_state_with_lattice(state, lattice=lattice)
    assert state.node_vectors[NODE]["rpc_timeout_pressure"] == 1.0
    apply_diffusion(state, diffusion_rate=1.0)
    cap = state.capability_vectors["capability:transport"]
    assert cap["reliability_pressure"] == 0.85  # edge weight
    assert state.capability_provenance["capability:transport"]["reliability_pressure"] == NODE
    # delivery failures are not load
    assert cap["pressure"] == 0.0


def test_calm_reading_is_a_measured_zero_with_provenance():
    lattice = _lattice()
    state = empty_field_state(lattice=lattice, now=NOW, tick_id="t")
    apply_perturbations(state, delta_to_perturbations(_delta(40, 0)), now=NOW)
    apply_diffusion(state, diffusion_rate=1.0)
    assert state.capability_vectors["capability:transport"]["reliability_pressure"] == 0.0
    # observer_failure_pressure (node:athena) and this edge both feed it; a
    # measured zero is still attributed, not left anonymous.
    assert state.capability_provenance["capability:transport"]["reliability_pressure"] in {
        NODE,
        "node:athena",
    }


def test_failure_holds_through_idle_ticks_and_clears_on_next_reading():
    lattice = _lattice()
    state = empty_field_state(lattice=lattice, now=NOW, tick_id="t")
    apply_perturbations(state, delta_to_perturbations(_delta(4, 6)), now=NOW)
    for i in range(1, 300):
        apply_decay(state, decay_rate=0.92, now=NOW + timedelta(seconds=2 * i), staleness_threshold_sec=90.0)
    assert state.node_vectors[NODE]["rpc_timeout_pressure"] == 0.6
    apply_perturbations(state, delta_to_perturbations(_delta(30, 0)), now=NOW)
    assert state.node_vectors[NODE]["rpc_timeout_pressure"] == 0.0


def test_field_gate_is_per_lane_and_default_off(monkeypatch):
    import app.settings as settings_mod
    from app.worker import delta_digestion_enabled

    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.delenv("ENABLE_RPC_DELIVERY_FIELD_DIGESTION", raising=False)
    settings_mod._settings = None
    try:
        assert settings_mod.get_settings().enable_rpc_delivery_field_digestion is False
    finally:
        settings_mod._settings = None

    off = SimpleNamespace(
        enable_transport_field_digestion=True,
        enable_llm_inference_field_digestion=True,
        enable_rpc_delivery_field_digestion=False,
    )
    on = SimpleNamespace(
        enable_transport_field_digestion=False,
        enable_llm_inference_field_digestion=False,
        enable_rpc_delivery_field_digestion=True,
    )
    assert delta_digestion_enabled("rpc_delivery", off) is False
    assert delta_digestion_enabled("rpc_delivery", on) is True
    assert delta_digestion_enabled("llm_inference_node", on) is False
