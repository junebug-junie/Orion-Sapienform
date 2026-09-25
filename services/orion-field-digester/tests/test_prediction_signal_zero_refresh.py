"""A calm (0.0) prediction_signal receipt must overwrite a prior non-zero field
value and re-stamp node_vector_updated_at.

Digester half of the 2026-09-25 fix: orion-substrate-runtime now sends a
prediction-error receipt on every tick, including exact 0.0. This pins that
the digester actually lands that zero instead of dropping it -- otherwise the
producer fix would be a no-op and node:substrate.route's field value would
stay frozen at its last non-zero reading (live: 0.0003, 12h+ stale).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.digestion.perturbation import apply_perturbations
from app.ingest.state_deltas import delta_to_perturbations
from orion.schemas.field_state import FieldStateV1
from orion.schemas.state_delta import StateDeltaV1

T0 = datetime(2026, 9, 24, 12, 11, 48, tzinfo=timezone.utc)
T1 = T0 + timedelta(hours=12)


def _delta(value: float, at: datetime) -> StateDeltaV1:
    return StateDeltaV1(
        delta_id=f"prediction_error:route_arbitration:{at.isoformat()}",
        target_projection="substrate.route_arbitration.projection",
        target_kind="prediction_signal",
        target_id="node:substrate.route",
        operation="update",
        after={
            "node_id": "node:substrate.route",
            "pressure_hints": {"prediction_error": value},
        },
        caused_by_event_ids=[],
        reducer_id="substrate.route_arbitration",
    )


def test_zero_prediction_error_replaces_stale_nonzero_and_restamps() -> None:
    state = FieldStateV1(generated_at=T0, tick_id="t0", node_vectors={}, edges=[])
    state = apply_perturbations(state, delta_to_perturbations(_delta(0.0003, T0)), now=T0)
    assert state.node_vectors["node:substrate.route"]["prediction_error"] == 0.0003

    perturbations = delta_to_perturbations(_delta(0.0, T1))
    assert [(p.channel, p.intensity, p.mode) for p in perturbations] == [
        ("prediction_error", 0.0, "replace")
    ]
    state = apply_perturbations(state, perturbations, now=T1)

    assert state.node_vectors["node:substrate.route"]["prediction_error"] == 0.0
    assert state.node_vector_updated_at["node:substrate.route"]["prediction_error"] == T1
