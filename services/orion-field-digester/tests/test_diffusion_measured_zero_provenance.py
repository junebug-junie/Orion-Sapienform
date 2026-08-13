"""A capability measured as healthy must be distinguishable from one never measured.

Before this, `apply_diffusion` attributed provenance only for contributions
> 0.0, so a source reporting a real 0.0 produced `pressure=0.0`,
`capability_provenance={}`, and confidence/available_capacity fabricated back to
1.0 by the derived fallback -- byte-for-byte identical to a capability with no
inbound edge at all. That is what made `capability:vision`'s real edge (PR #1616)
indistinguishable at rest from the fabricated vector it was built to replace.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.digestion.diffusion import apply_diffusion

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)


def _vision_edge():
    return FieldEdgeV1(
        source_id="node:substrate.vision",
        target_id="capability:vision",
        edge_type="node_capability",
        weight=0.85,
        channel_map={"prediction_error": "pressure"},
    )


def _edge(source: str, target: str, channel_map: dict[str, str], weight: float = 0.85):
    return FieldEdgeV1(
        source_id=source,
        target_id=target,
        edge_type="node_capability",
        weight=weight,
        channel_map=channel_map,
    )


VISION_EDGE = _vision_edge()


def _state(node_vectors: dict, edges: list[FieldEdgeV1] | None = None) -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_measured_zero_provenance",
        node_vectors=node_vectors,
        edges=edges or [],
    )


def test_measured_zero_claims_provenance() -> None:
    """The headline case: a healthy eye is attributable, not anonymous."""
    state = _state({"node:substrate.vision": {"prediction_error": 0.0}}, [VISION_EDGE])
    apply_diffusion(state, diffusion_rate=1.0)
    assert state.capability_vectors["capability:vision"]["pressure"] == 0.0
    assert state.capability_provenance["capability:vision"]["pressure"] == "node:substrate.vision"


def test_unmeasured_channel_stays_anonymous() -> None:
    """
    The case the old guard was protecting, and which must still hold: the source
    carries no value for the mapped channel at all, so nothing is known and
    nothing may be claimed.
    """
    state = _state({"node:substrate.vision": {"some_other_channel": 0.4}}, [VISION_EDGE])
    apply_diffusion(state, diffusion_rate=1.0)
    assert state.capability_vectors["capability:vision"]["pressure"] == 0.0
    assert "pressure" not in state.capability_provenance.get("capability:vision", {})


def test_measured_zero_never_displaces_a_real_contribution() -> None:
    """Ordering must not decide the winner: the real signal owns the channel."""
    state = _state(
        {
            "node:quiet": {"prediction_error": 0.0},
            "node:loud": {"prediction_error": 0.8},
        }
    )
    edges = [
        _edge("node:quiet", "capability:x", {"prediction_error": "pressure"}),
        _edge("node:loud", "capability:x", {"prediction_error": "pressure"}),
    ]
    state.edges = edges
    apply_diffusion(state, diffusion_rate=1.0)
    assert state.capability_provenance["capability:x"]["pressure"] == "node:loud"
    assert state.capability_vectors["capability:x"]["pressure"] > 0.0

    # ...and the same holds with the edges evaluated in the other order.
    state2 = _state(
        {
            "node:quiet": {"prediction_error": 0.0},
            "node:loud": {"prediction_error": 0.8},
        }
    )
    state2.edges = list(reversed(edges))
    apply_diffusion(state2, diffusion_rate=1.0)
    assert state2.capability_provenance["capability:x"]["pressure"] == "node:loud"


def test_derived_fallback_still_runs_for_a_measured_zero() -> None:
    """
    Regression guard for the trap in this fix: `best_source` doubles as the
    gate for the derived confidence/available_capacity fallback. Routing
    measured zeros through it would skip that fallback and hard-floor both
    channels to 0.0 instead of deriving them from pressure.
    """
    state = _state({"node:substrate.vision": {"prediction_error": 0.0}}, [VISION_EDGE])
    apply_diffusion(state, diffusion_rate=1.0)
    vec = state.capability_vectors["capability:vision"]
    assert vec["available_capacity"] == 1.0, "derived fallback must still run"
    assert vec["confidence"] == 1.0


def test_stale_provenance_is_cleared_when_the_source_stops_reporting() -> None:
    state = _state({"node:substrate.vision": {"prediction_error": 0.0}}, [VISION_EDGE])
    apply_diffusion(state, diffusion_rate=1.0)
    assert "pressure" in state.capability_provenance["capability:vision"]

    # Same state object, but the source no longer carries the channel.
    state.node_vectors["node:substrate.vision"] = {}
    apply_diffusion(state, diffusion_rate=1.0)
    assert "pressure" not in state.capability_provenance["capability:vision"]


def test_a_real_reading_still_attributes_normally() -> None:
    state = _state({"node:substrate.vision": {"prediction_error": 1.0}}, [VISION_EDGE])
    apply_diffusion(state, diffusion_rate=1.0)
    assert state.capability_vectors["capability:vision"]["pressure"] > 0.0
    assert state.capability_provenance["capability:vision"]["pressure"] == "node:substrate.vision"
