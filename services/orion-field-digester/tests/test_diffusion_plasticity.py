"""Causal Geometry v1, Rung 3A: apply_diffusion's read-only learned-overlay
adapter for capability_capability edges.

See `app/digestion/diffusion.py`'s module docstring,
`orion/substrate/field_topology_learned_store.py`'s
`FieldTopologyLearnedWeightsStore.current_overlay()`, and the design spec
(docs/superpowers/specs/2026-07-16-causal-geometry-v1-design.md, Phase B) for
the full picture. This file enforces the hard constraints from that rung's
task spec:

1. `FIELD_PLASTICITY_ENABLED` unset/false -> `apply_diffusion` is byte-
   identical to the pre-plasticity implementation, including never
   constructing `FieldTopologyLearnedWeightsStore` (or even calling this
   module's own `_load_learned_overlay` helper) at all.
2. `FIELD_PLASTICITY_ENABLED` true -> a `capability_capability` edge with an
   adopted overlay delta present uses `clamp01(edge.weight + delta)` as its
   diffusion contribution weight instead of the raw designed `edge.weight`.
   `edge.weight` itself is never mutated in place.
3. A non-cap-cap edge (e.g. `node_capability`) is unaffected by an overlay
   entry even if one happens to exist for its (source_id, target_id) pair --
   the read-path gates on `edge_type` defensively, not just on upstream
   proposal generation only ever producing cap->cap deltas.
4. The overlay read path degrades gracefully: if
   `FieldTopologyLearnedWeightsStore()` fails to construct, or
   `current_overlay()` raises, `apply_diffusion` must not raise -- it falls
   back to raw `edge.weight` for that tick.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import app.digestion.diffusion as diffusion_module
from app.digestion.diffusion import apply_diffusion

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _reset_learned_store_cache():
    """`_get_learned_store()` caches its store at module level (so production
    doesn't reconnect sqlite every diffusion tick) -- reset it before and after
    every test so a store constructed/monkeypatched in one test can never leak
    into the next."""
    diffusion_module._LEARNED_STORE = None
    yield
    diffusion_module._LEARNED_STORE = None

CAP_CAP_EDGE_REF = "capability:transport->capability:orchestration"


def _state(
    edges: list[FieldEdgeV1],
    node_vectors: dict[str, dict[str, float]] | None = None,
    capability_vectors: dict[str, dict[str, float]] | None = None,
) -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_plasticity",
        node_vectors=node_vectors or {},
        capability_vectors=capability_vectors or {},
        edges=edges,
    )


def _cap_cap_edge(weight: float = 0.70) -> FieldEdgeV1:
    return FieldEdgeV1(
        source_id="capability:transport",
        target_id="capability:orchestration",
        edge_type="capability_capability",
        weight=weight,
        channel_map={"transport_pressure": "transport_pressure"},
    )


def _refuse_to_load_overlay() -> dict[str, float]:
    raise AssertionError(
        "the learned-overlay adapter must never be invoked when "
        "FIELD_PLASTICITY_ENABLED is off"
    )


# -- (1) Flag off: byte-identical to pre-plasticity behavior --------------


def test_flag_unset_never_loads_overlay_and_matches_golden_output(monkeypatch) -> None:
    monkeypatch.delenv("FIELD_PLASTICITY_ENABLED", raising=False)
    monkeypatch.setattr(diffusion_module, "_load_learned_overlay", _refuse_to_load_overlay)

    edge = _cap_cap_edge(weight=0.70)
    state = _state([edge], capability_vectors={"capability:transport": {"transport_pressure": 0.8}})

    apply_diffusion(state, diffusion_rate=1.0)

    # Golden output computed from the pre-plasticity formula:
    # contribution = clamp01(0.8 * 0.70 * 1.0) = 0.56.
    assert state.capability_vectors["capability:orchestration"]["transport_pressure"] == pytest.approx(0.56)
    assert (
        state.capability_provenance["capability:orchestration"]["transport_pressure"]
        == "capability:transport"
    )
    assert edge.weight == 0.70
    assert edge.weight_source == "designed"
    assert edge.learned_at is None


def test_flag_explicitly_false_also_never_loads_overlay(monkeypatch) -> None:
    monkeypatch.setenv("FIELD_PLASTICITY_ENABLED", "false")
    monkeypatch.setattr(diffusion_module, "_load_learned_overlay", _refuse_to_load_overlay)

    edge = _cap_cap_edge(weight=0.70)
    state = _state([edge], capability_vectors={"capability:transport": {"transport_pressure": 0.8}})

    apply_diffusion(state, diffusion_rate=1.0)

    assert state.capability_vectors["capability:orchestration"]["transport_pressure"] == pytest.approx(0.56)


def test_flag_off_existing_provenance_suite_scenario_is_unaffected(monkeypatch) -> None:
    # Reuses the exact fixture shape from
    # test_diffusion_provenance.py::test_confidence_and_available_capacity_derived_from_final_pressure
    # as an independent golden check that ordinary node_capability diffusion
    # (unrelated to plasticity entirely) is untouched by this patch.
    monkeypatch.delenv("FIELD_PLASTICITY_ENABLED", raising=False)

    edge = FieldEdgeV1(
        source_id="node:atlas",
        target_id="capability:llm_inference",
        edge_type="node_capability",
        weight=0.80,
        channel_map={"gpu_pressure": "pressure"},
    )
    state = _state([edge], node_vectors={"node:atlas": {"gpu_pressure": 0.5}})

    apply_diffusion(state, diffusion_rate=1.0)

    tgt = state.capability_vectors["capability:llm_inference"]
    assert tgt["pressure"] == 0.40
    assert tgt["available_capacity"] == 0.60
    assert tgt["confidence"] == 0.80


# -- (2) Flag on: overlay applied to cap->cap edges only -------------------


def test_flag_on_applies_overlay_delta_to_cap_cap_edge(monkeypatch) -> None:
    monkeypatch.setenv("FIELD_PLASTICITY_ENABLED", "true")
    monkeypatch.setattr(
        diffusion_module, "_load_learned_overlay", lambda: {CAP_CAP_EDGE_REF: 0.20}
    )

    edge = _cap_cap_edge(weight=0.70)
    state = _state([edge], capability_vectors={"capability:transport": {"transport_pressure": 0.8}})

    apply_diffusion(state, diffusion_rate=1.0)

    # effective_weight = clamp01(0.70 + 0.20) = 0.90;
    # contribution = clamp01(0.8 * 0.90 * 1.0) = 0.72 -- not the raw-weight 0.56.
    assert state.capability_vectors["capability:orchestration"]["transport_pressure"] == pytest.approx(0.72)
    assert edge.weight == 0.70  # never mutated in place
    assert edge.weight_source == "learned"
    assert edge.learned_at is not None


def test_flag_on_clamps_effective_weight_to_valid_range(monkeypatch) -> None:
    monkeypatch.setenv("FIELD_PLASTICITY_ENABLED", "true")
    monkeypatch.setattr(
        diffusion_module, "_load_learned_overlay", lambda: {CAP_CAP_EDGE_REF: 0.50}
    )

    edge = _cap_cap_edge(weight=0.90)
    state = _state([edge], capability_vectors={"capability:transport": {"transport_pressure": 1.0}})

    apply_diffusion(state, diffusion_rate=1.0)

    # 0.90 + 0.50 = 1.40 -> clamped to 1.0 before use in the contribution formula.
    assert state.capability_vectors["capability:orchestration"]["transport_pressure"] == 1.0


def test_flag_on_does_not_affect_non_cap_cap_edge_with_matching_overlay_entry(monkeypatch) -> None:
    monkeypatch.setenv("FIELD_PLASTICITY_ENABLED", "true")
    # An overlay entry keyed for this exact (source_id, target_id) pair
    # exists, but the edge itself is node_capability, not
    # capability_capability -- proves the read-path gates on edge_type
    # defensively rather than trusting that only cap->cap deltas ever exist.
    overlay_ref = "node:athena->capability:orchestration"
    monkeypatch.setattr(diffusion_module, "_load_learned_overlay", lambda: {overlay_ref: 0.50})

    edge = FieldEdgeV1(
        source_id="node:athena",
        target_id="capability:orchestration",
        edge_type="node_capability",
        weight=0.40,
        channel_map={"cpu_pressure": "pressure"},
    )
    state = _state([edge], node_vectors={"node:athena": {"cpu_pressure": 1.0}})

    apply_diffusion(state, diffusion_rate=1.0)

    # contribution = clamp01(1.0 * 0.40 * 1.0) = 0.40 -- raw weight, overlay ignored.
    assert state.capability_vectors["capability:orchestration"]["pressure"] == 0.40
    assert edge.weight_source == "designed"
    assert edge.learned_at is None


def test_flag_on_no_adopted_delta_for_this_edge_falls_back_to_raw_weight(monkeypatch) -> None:
    monkeypatch.setenv("FIELD_PLASTICITY_ENABLED", "true")
    # Overlay has entries, but none for this edge's ref -- must not
    # spuriously apply some other edge's delta.
    monkeypatch.setattr(
        diffusion_module,
        "_load_learned_overlay",
        lambda: {"capability:other->capability:unrelated": 0.30},
    )

    edge = _cap_cap_edge(weight=0.70)
    state = _state([edge], capability_vectors={"capability:transport": {"transport_pressure": 0.8}})

    apply_diffusion(state, diffusion_rate=1.0)

    assert state.capability_vectors["capability:orchestration"]["transport_pressure"] == pytest.approx(0.56)
    assert edge.weight_source == "designed"


# -- (4) Graceful degradation: never raise ----------------------------------


def test_flag_on_store_construction_failure_degrades_to_raw_weight_without_raising(
    monkeypatch,
) -> None:
    class _BoomStore:
        def __init__(self) -> None:
            raise RuntimeError("sqlite path misconfigured")

    monkeypatch.setenv("FIELD_PLASTICITY_ENABLED", "true")
    monkeypatch.setattr(
        "orion.substrate.field_topology_learned_store.FieldTopologyLearnedWeightsStore",
        _BoomStore,
    )

    edge = _cap_cap_edge(weight=0.70)
    state = _state([edge], capability_vectors={"capability:transport": {"transport_pressure": 0.8}})

    # Must not raise -- degrades to raw edge.weight for this tick.
    apply_diffusion(state, diffusion_rate=1.0)

    assert state.capability_vectors["capability:orchestration"]["transport_pressure"] == pytest.approx(0.56)
    assert edge.weight_source == "designed"


def test_flag_on_current_overlay_call_failure_degrades_to_raw_weight_without_raising(
    monkeypatch,
) -> None:
    class _BoomOverlayStore:
        def __init__(self) -> None:
            pass

        def current_overlay(self, *, now=None):
            raise RuntimeError("overlay read failed")

    monkeypatch.setenv("FIELD_PLASTICITY_ENABLED", "true")
    monkeypatch.setattr(
        "orion.substrate.field_topology_learned_store.FieldTopologyLearnedWeightsStore",
        _BoomOverlayStore,
    )

    edge = _cap_cap_edge(weight=0.70)
    state = _state([edge], capability_vectors={"capability:transport": {"transport_pressure": 0.8}})

    apply_diffusion(state, diffusion_rate=1.0)

    assert state.capability_vectors["capability:orchestration"]["transport_pressure"] == pytest.approx(0.56)
    assert edge.weight_source == "designed"


def test_learned_store_is_constructed_once_and_reused_across_ticks(monkeypatch) -> None:
    """Regression: `_load_learned_overlay` previously built a brand-new
    `FieldTopologyLearnedWeightsStore()` (a sqlite reconnect once persistence lands)
    on every single `apply_diffusion` call/tick. `_get_learned_store()` must cache it
    instead -- construct once, reuse across ticks."""
    construction_count = 0

    class _CountingStore:
        def __init__(self, *, sql_db_path=None) -> None:
            nonlocal construction_count
            construction_count += 1

        def current_overlay(self, *, now=None):
            return {}

    monkeypatch.setenv("FIELD_PLASTICITY_ENABLED", "true")
    monkeypatch.setattr(
        "orion.substrate.field_topology_learned_store.FieldTopologyLearnedWeightsStore",
        _CountingStore,
    )

    edge = _cap_cap_edge(weight=0.70)
    state = _state([edge], capability_vectors={"capability:transport": {"transport_pressure": 0.8}})

    apply_diffusion(state, diffusion_rate=1.0)
    apply_diffusion(state, diffusion_rate=1.0)
    apply_diffusion(state, diffusion_rate=1.0)

    assert construction_count == 1


def test_load_learned_overlay_helper_itself_never_raises_on_store_failure(monkeypatch) -> None:
    # Direct unit check of the adapter function in isolation (not just via
    # apply_diffusion), matching "every new adapter must degrade gracefully
    # ... never raise."
    class _BoomStore:
        def __init__(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "orion.substrate.field_topology_learned_store.FieldTopologyLearnedWeightsStore",
        _BoomStore,
    )

    result = diffusion_module._load_learned_overlay()

    assert result == {}


@pytest.mark.parametrize("raw_value", ["1", "true", "TRUE", "yes", "on", "On"])
def test_plasticity_enabled_truthy_values(monkeypatch, raw_value: str) -> None:
    monkeypatch.setenv("FIELD_PLASTICITY_ENABLED", raw_value)
    assert diffusion_module._plasticity_enabled() is True


@pytest.mark.parametrize("raw_value", ["0", "false", "no", "off", "", "garbage"])
def test_plasticity_enabled_falsy_values(monkeypatch, raw_value: str) -> None:
    monkeypatch.setenv("FIELD_PLASTICITY_ENABLED", raw_value)
    assert diffusion_module._plasticity_enabled() is False
