"""Unit tests for apply_diffusion's Phase 3 (2026-07-12) node-provenance
tracking -- state.capability_provenance recording which edge source
contributed the largest weighted amount to each diffused channel this tick.
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


def test_zero_contribution_does_not_overwrite_existing_provenance() -> None:
    # A second edge whose source has no value for its mapped channel this
    # tick (contribution 0.0) must not clobber a real contributor's
    # provenance recorded moments before in the same pass.
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


def test_zero_contribution_edge_does_not_clobber_prior_tick_provenance() -> None:
    # Regression found by code review (2026-07-12, empirically reproduced):
    # 0.0 was both the "no contribution recorded this call" sentinel AND a
    # legitimate zero-contribution value, so a single edge contributing
    # nothing this tick (its source has no value for the mapped channel)
    # would still satisfy the old `contribution >= _max_contribution.get(key,
    # 0.0)` check on the first edge processed for a (target, channel) key --
    # silently overwriting real provenance recorded on a prior tick, even
    # though the accumulated value itself is unchanged (adding 0.0 is a
    # no-op). This state carries capability_provenance forward from a prior
    # tick (as the persistent FieldStateV1 object genuinely does across real
    # ticks) where atlas was the real contributor; this tick's only edge
    # (circe) contributes nothing and must not claim credit.
    state = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_stale_zero_contribution",
        node_vectors={"node:circe": {}},  # no gpu_pressure this tick
        capability_vectors={"capability:llm_inference": {"pressure": 0.5}},
        capability_provenance={"capability:llm_inference": {"pressure": "node:atlas"}},
        edges=[
            FieldEdgeV1(
                source_id="node:circe",
                target_id="capability:llm_inference",
                edge_type="node_capability",
                weight=0.50,
                channel_map={"gpu_pressure": "pressure"},
            ),
        ],
    )

    apply_diffusion(state, diffusion_rate=1.0)

    # Accumulated value unchanged (0.0 contribution added).
    assert state.capability_vectors["capability:llm_inference"]["pressure"] == 0.5
    # Provenance must still credit atlas (the real prior contributor), not
    # circe (which contributed nothing this tick).
    assert state.capability_provenance["capability:llm_inference"]["pressure"] == "node:atlas"


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
