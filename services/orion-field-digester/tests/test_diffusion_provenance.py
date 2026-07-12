"""Unit tests for apply_diffusion's node-provenance tracking (Phase 3,
2026-07-12) and its memoryless-recompute fix (2026-07-12, later same day):
state.capability_provenance records which edge source contributed the
largest weighted amount to each diffused channel THIS TICK, and the
diffused value itself is now recomputed fresh from current node/capability
state every tick rather than accumulating across ticks -- see
orion/self_state/deviation.py sibling module and app/digestion/diffusion.py's
module docstring for the full incident writeup (resource_pressure was
dead-flat 1.0 for the entire observed history due to additive cross-tick
accumulation overwhelming an 8%/tick decay).
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.digestion.diffusion import apply_diffusion

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1

NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


def _state(edges: list[FieldEdgeV1], node_vectors: dict[str, dict[str, float]]) -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_diffusion_provenance",
        node_vectors=node_vectors,
        edges=edges,
    )


def test_single_edge_records_source_as_provenance() -> None:
    edge = FieldEdgeV1(
        source_id="node:circe",
        target_id="capability:llm_inference",
        edge_type="node_capability",
        weight=0.50,
        channel_map={"gpu_pressure": "pressure"},
    )
    state = _state([edge], {"node:circe": {"gpu_pressure": 1.0}})

    apply_diffusion(state, diffusion_rate=1.0)

    assert state.capability_vectors["capability:llm_inference"]["pressure"] == 0.50
    assert state.capability_provenance["capability:llm_inference"]["pressure"] == "node:circe"


def test_larger_contribution_wins_provenance_over_smaller() -> None:
    # atlas (weight 0.85) and circe (weight 0.50) both feed
    # capability:llm_inference's "pressure" channel via gpu_pressure -- the
    # larger weighted contribution should "win" provenance for this tick.
    edges = [
        FieldEdgeV1(
            source_id="node:atlas",
            target_id="capability:llm_inference",
            edge_type="node_capability",
            weight=0.85,
            channel_map={"gpu_pressure": "pressure"},
        ),
        FieldEdgeV1(
            source_id="node:circe",
            target_id="capability:llm_inference",
            edge_type="node_capability",
            weight=0.50,
            channel_map={"gpu_pressure": "pressure"},
        ),
    ]
    state = _state(
        edges,
        {
            "node:atlas": {"gpu_pressure": 1.0},
            "node:circe": {"gpu_pressure": 0.2},
        },
    )

    apply_diffusion(state, diffusion_rate=1.0)

    # atlas contributes 1.0*0.85=0.85, circe contributes 0.2*0.50=0.10 -- atlas wins.
    assert state.capability_provenance["capability:llm_inference"]["pressure"] == "node:atlas"


def test_zero_contribution_edge_does_not_beat_a_real_contribution_same_tick() -> None:
    # Both atlas's and circe's edges are present (real topology edges are
    # static -- always in state.edges, every tick) and both target the same
    # channel. atlas has a real contribution this tick; circe's node vector
    # has no gpu_pressure at all (contribution 0). The zero-contribution edge
    # must not win/clobber the real one just by being evaluated in some
    # iteration order.
    edges = [
        FieldEdgeV1(
            source_id="node:atlas",
            target_id="capability:llm_inference",
            edge_type="node_capability",
            weight=0.85,
            channel_map={"gpu_pressure": "pressure"},
        ),
        FieldEdgeV1(
            source_id="node:circe",
            target_id="capability:llm_inference",
            edge_type="node_capability",
            weight=0.50,
            channel_map={"gpu_pressure": "pressure"},
        ),
    ]
    state = _state(
        edges,
        {
            "node:atlas": {"gpu_pressure": 1.0},
            "node:circe": {},  # no gpu_pressure this tick -> contribution 0.0
        },
    )

    apply_diffusion(state, diffusion_rate=1.0)

    assert state.capability_provenance["capability:llm_inference"]["pressure"] == "node:atlas"
    assert state.capability_vectors["capability:llm_inference"]["pressure"] == 0.85


def test_no_current_contributor_resets_value_and_clears_provenance() -> None:
    # The actual memoryless-recompute fix: a channel that WAS diffused in a
    # prior tick (simulated here via pre-seeded capability_vectors/
    # capability_provenance) but has zero real contributors THIS tick must
    # reset to 0.0 and drop its provenance, not keep displaying a stale value
    # attributed to a source that isn't contributing anymore.
    edge = FieldEdgeV1(
        source_id="node:circe",
        target_id="capability:llm_inference",
        edge_type="node_capability",
        weight=0.50,
        channel_map={"gpu_pressure": "pressure"},
    )
    state = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_reset",
        node_vectors={"node:circe": {}},  # circe reporting, but no gpu_pressure this tick
        capability_vectors={"capability:llm_inference": {"pressure": 0.5}},
        capability_provenance={"capability:llm_inference": {"pressure": "node:atlas"}},
        edges=[edge],
    )

    apply_diffusion(state, diffusion_rate=1.0)

    assert state.capability_vectors["capability:llm_inference"]["pressure"] == 0.0
    assert "pressure" not in state.capability_provenance.get("capability:llm_inference", {})


def test_sustained_real_contribution_does_not_ratchet_up_across_ticks() -> None:
    # The actual live incident this fix addresses: resource_pressure was
    # dead-flat 1.0 for the entire observed history because the OLD
    # implementation added each tick's contribution onto whatever was already
    # there (min(1.0, tgt.get(ch, 0.0) + contribution)), which an 8%/tick
    # decay could never keep up with. Calling apply_diffusion repeatedly with
    # the SAME steady, moderate input must produce the SAME steady value each
    # time, not a value that climbs toward (and gets stuck at) 1.0.
    edge = FieldEdgeV1(
        source_id="node:atlas",
        target_id="capability:llm_inference",
        edge_type="node_capability",
        weight=0.50,
        channel_map={"gpu_pressure": "pressure"},
    )
    state = _state([edge], {"node:atlas": {"gpu_pressure": 0.6}})

    values = []
    for _ in range(10):
        apply_diffusion(state, diffusion_rate=1.0)
        values.append(state.capability_vectors["capability:llm_inference"]["pressure"])

    # 0.6 * 0.50 = 0.30, every single tick -- never climbs.
    assert all(v == 0.30 for v in values), values


def test_multiple_channels_feeding_same_target_use_max_not_sum() -> None:
    # node:athena -> capability:orchestration maps BOTH cpu_pressure AND
    # transport_pressure onto "pressure" (real topology edge,
    # config/field/orion_field_topology.v1.yaml). Two real, legitimate
    # stressors must combine via max() -- whichever is worse dominates --
    # not sum, which would let two only-moderately-stressed channels alone
    # push the target near/at ceiling in a single tick.
    edge = FieldEdgeV1(
        source_id="node:athena",
        target_id="capability:orchestration",
        edge_type="node_capability",
        weight=0.90,
        channel_map={"cpu_pressure": "pressure", "transport_pressure": "pressure"},
    )
    state = _state(
        [edge],
        {"node:athena": {"cpu_pressure": 0.5, "transport_pressure": 0.5}},
    )

    apply_diffusion(state, diffusion_rate=1.0)

    # Each contributes 0.5*0.90=0.45. Summed, that's 0.90 (near ceiling from
    # one tick alone); maxed, it's 0.45 -- whichever channel is worse.
    assert state.capability_vectors["capability:orchestration"]["pressure"] == 0.45


def test_capability_capability_edge_records_capability_id_as_provenance() -> None:
    # capability_capability edges (e.g. capability:transport -> capability:
    # orchestration) have a capability_id as source_id, not a node -- the
    # provenance mechanism is source-agnostic and records whatever
    # edge.source_id is, node or capability.
    edge = FieldEdgeV1(
        source_id="capability:transport",
        target_id="capability:orchestration",
        edge_type="capability_capability",
        weight=0.70,
        channel_map={"transport_pressure": "transport_pressure"},
    )
    state = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_cap_cap",
        capability_vectors={"capability:transport": {"transport_pressure": 0.8}},
        edges=[edge],
    )

    apply_diffusion(state, diffusion_rate=1.0)

    assert (
        state.capability_provenance["capability:orchestration"]["transport_pressure"]
        == "capability:transport"
    )


def test_confidence_and_available_capacity_derived_from_final_pressure() -> None:
    edge = FieldEdgeV1(
        source_id="node:atlas",
        target_id="capability:llm_inference",
        edge_type="node_capability",
        weight=0.80,
        channel_map={"gpu_pressure": "pressure"},
    )
    state = _state([edge], {"node:atlas": {"gpu_pressure": 0.5}})

    apply_diffusion(state, diffusion_rate=1.0)

    tgt = state.capability_vectors["capability:llm_inference"]
    assert tgt["pressure"] == 0.40
    assert tgt["available_capacity"] == 0.60
    assert tgt["confidence"] == 0.80


def test_capability_with_no_pressure_target_keeps_reconciled_baseline() -> None:
    # A capability whose only diffusion target is NOT "pressure" (e.g. only
    # receives reliability_pressure) must not have its pre-existing
    # confidence/available_capacity baseline (set by the caller's lattice
    # reconciliation before apply_diffusion ever runs) clobbered or dropped --
    # only channels apply_diffusion actually recomputes should change.
    edge = FieldEdgeV1(
        source_id="node:athena",
        target_id="capability:orchestration",
        edge_type="node_capability",
        weight=0.90,
        channel_map={"execution_friction": "reliability_pressure"},
    )
    state = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_no_pressure_target",
        node_vectors={"node:athena": {"execution_friction": 0.5}},
        capability_vectors={"capability:orchestration": {"confidence": 1.0, "available_capacity": 1.0}},
        edges=[edge],
    )

    apply_diffusion(state, diffusion_rate=1.0)

    tgt = state.capability_vectors["capability:orchestration"]
    assert tgt["reliability_pressure"] == 0.45
    # Untouched -- "pressure" was never a target for this capability in this
    # test, so the derived-formula block never ran.
    assert tgt["confidence"] == 1.0
    assert tgt["available_capacity"] == 1.0


def test_direct_confidence_and_capacity_diffusion_beats_pressure_derived_formula() -> None:
    # capability:transport (real topology edge, node:athena -> capability:
    # transport) has direct edges into "confidence" (from delivery_confidence)
    # and "available_capacity" (from bus_health) AND a "pressure" target in
    # the same edge -- the pressure-derived formula must not clobber the
    # direct diffusion values just because "pressure" is also present.
    edge = FieldEdgeV1(
        source_id="node:athena",
        target_id="capability:transport",
        edge_type="node_capability",
        weight=0.85,
        channel_map={
            "transport_pressure": "pressure",
            "delivery_confidence": "confidence",
            "bus_health": "available_capacity",
        },
    )
    state = _state(
        [edge],
        {
            "node:athena": {
                "transport_pressure": 0.9,
                "delivery_confidence": 0.95,
                "bus_health": 0.7,
            }
        },
    )

    apply_diffusion(state, diffusion_rate=1.0)

    tgt = state.capability_vectors["capability:transport"]
    # pressure-derived formula would give confidence=1-0.5*0.765=0.6175 and
    # available_capacity=1-0.765=0.235 -- direct diffusion must win instead.
    assert tgt["pressure"] == 0.765
    assert tgt["confidence"] == round(0.95 * 0.85, 10)
    assert tgt["available_capacity"] == round(0.7 * 0.85, 10)


def test_derived_formula_still_falls_back_when_direct_edge_contributes_nothing_this_tick() -> None:
    # Regression found by code review (2026-07-12): the precedence guard
    # must key off "did a real (>0) contribution land THIS tick", not "is
    # this channel ever configured as a diffusion target for this
    # capability" -- otherwise a capability:transport tick where
    # delivery_confidence/bus_health are temporarily absent from the source
    # (0.0 contribution, no entry in best_source) would hard-floor
    # confidence/available_capacity at 0.0 instead of falling back to the
    # pressure-derived formula.
    edge = FieldEdgeV1(
        source_id="node:athena",
        target_id="capability:transport",
        edge_type="node_capability",
        weight=0.85,
        channel_map={
            "transport_pressure": "pressure",
            "delivery_confidence": "confidence",
            "bus_health": "available_capacity",
        },
    )
    state = _state(
        [edge],
        {"node:athena": {"transport_pressure": 0.9}},  # no delivery_confidence/bus_health this tick
    )

    apply_diffusion(state, diffusion_rate=1.0)

    tgt = state.capability_vectors["capability:transport"]
    assert tgt["pressure"] == 0.765
    # Fallback to the derived formula, not hard-floored at 0.0.
    assert tgt["confidence"] == max(0.0, 1.0 - 0.5 * 0.765)
    assert tgt["available_capacity"] == max(0.0, 1.0 - 0.765)
