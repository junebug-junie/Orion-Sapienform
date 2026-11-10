"""A prediction_signal delta that says "no reading" must leave the field channel
ABSENT, never write 0.0 into it.

PE suite rebuild spec (docs/superpowers/specs/2026-11-08-prediction-error-suite-
rebuild-design.md), retire step 8: "missing ticks must not write 0.0 into the
field". Live before the fix, 2026-11-08: the camera embeddings stopped from
08:59:58 to 09:01:33 UTC. The perception tick kept writing every 30 s, and the
field held prediction_error = 0.0 for node:substrate.perception at 09:00:33,
09:01:03 and 09:01:34 -- a dead camera read exactly like a calm one.

The producer (orion-substrate-runtime ``_prediction_error_receipt``) now sends
``prediction_error_missing_reason`` and no ``pressure_hints.prediction_error``
on such ticks. This file pins the digester half: the channel and its
``node_vector_updated_at`` stamp are dropped, a later real reading restores
them, and a real 0.0 is still a reading.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.digestion.perturbation import apply_perturbations
from app.ingest.state_deltas import delta_to_perturbations
from orion.schemas.field_state import FieldStateV1
from orion.schemas.state_delta import StateDeltaV1
from orion.substrate.prediction_error_freshness import PREDICTION_ERROR_MISSING_REASON_KEY

ROOM = "node:substrate.perception"
T0 = datetime(2026, 11, 8, 8, 59, 32, tzinfo=timezone.utc)


def _delta(at: datetime, *, value: float | None = None, missing: str | None = None) -> StateDeltaV1:
    after: dict = {
        "node_id": ROOM,
        "pressure_hints": {} if value is None else {"prediction_error": value},
        "definition_version": 1,
    }
    if missing is not None:
        after[PREDICTION_ERROR_MISSING_REASON_KEY] = missing
    return StateDeltaV1(
        delta_id=f"prediction_error:perception:{at.isoformat()}",
        target_projection="substrate.perception.projection",
        target_kind="prediction_signal",
        target_id=ROOM,
        operation="update",
        after=after,
        caused_by_event_ids=[],
        reducer_id="substrate.perception",
    )


def _state() -> FieldStateV1:
    return FieldStateV1(generated_at=T0, tick_id="t0", node_vectors={}, edges=[])


def _apply(state: FieldStateV1, delta: StateDeltaV1, at: datetime) -> FieldStateV1:
    return apply_perturbations(state, delta_to_perturbations(delta), now=at)


def test_missing_tick_drops_the_channel_instead_of_writing_zero() -> None:
    state = _apply(_state(), _delta(T0, value=0.2637), T0)
    assert state.node_vectors[ROOM]["prediction_error"] == 0.2637

    t1 = T0 + timedelta(seconds=61)
    state = _apply(state, _delta(t1, missing="no_scored_frame_since_last_tick"), t1)

    assert "prediction_error" not in state.node_vectors[ROOM]
    assert "prediction_error" not in state.node_vector_updated_at.get(ROOM, {})


def test_a_real_reading_after_a_missing_stretch_restores_the_channel() -> None:
    state = _apply(_state(), _delta(T0, missing="stale"), T0)
    assert "prediction_error" not in state.node_vectors.get(ROOM, {})

    t1 = T0 + timedelta(seconds=30)
    state = _apply(state, _delta(t1, value=0.0087), t1)
    assert state.node_vectors[ROOM]["prediction_error"] == 0.0087
    assert state.node_vector_updated_at[ROOM]["prediction_error"] == t1


def test_a_real_zero_is_still_a_reading() -> None:
    # The rest point must stay reachable: a scored calm frame is 0.0, not absent.
    state = _apply(_state(), _delta(T0, value=0.0), T0)
    assert state.node_vectors[ROOM]["prediction_error"] == 0.0
    assert state.node_vector_updated_at[ROOM]["prediction_error"] == T0


def test_missing_reason_wins_over_a_value_sent_alongside_it() -> None:
    perturbations = delta_to_perturbations(_delta(T0, value=0.0, missing="warming"))
    assert [(p.channel, p.mode) for p in perturbations] == [("prediction_error", "clear")]


def test_clear_on_an_unknown_node_creates_nothing() -> None:
    state = _apply(_state(), _delta(T0, missing="stale"), T0)
    assert ROOM not in state.node_vectors
    assert ROOM not in state.node_vector_updated_at


def test_clear_leaves_other_channels_on_the_node_alone() -> None:
    state = _state()
    state.node_vectors[ROOM] = {"prediction_error": 0.4, "embedding_staleness": 1.0}
    state.node_vector_updated_at[ROOM] = {"prediction_error": T0, "embedding_staleness": T0}
    t1 = T0 + timedelta(seconds=30)
    state = _apply(state, _delta(t1, missing="stale"), t1)
    assert state.node_vectors[ROOM] == {"embedding_staleness": 1.0}
    assert state.node_vector_updated_at[ROOM] == {"embedding_staleness": T0}
