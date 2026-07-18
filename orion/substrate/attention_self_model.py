"""AST/HOT attention self-model reducer — rung 4 (read-only, Phase 1).

Pure function, no I/O. Takes the three already-parsed real inputs and
returns one `AttentionSelfModelV1` answering: what's currently salient, what
was last dispatched, *why* (top-down goal bias vs. pure bottom-up salience —
the "aboutness" claim AST/HOT instrumentation needs to make honestly), how
confident, and what's predicted to shift next.

Mirrors `scripts/analysis/measure_origination_gate.py`'s separation of a pure
replay/summary layer from an I/O layer, for the same testability reason that
script documents: this function is exercised directly with synthetic
fixtures, no DB or bus involved, by `orion/substrate/tests/
test_attention_self_model.py`.

Read-only. This module has no bus consumer/producer wiring — see
`docs/superpowers/specs/2026-07-18-objective-3-consciousness-scaffolded-roadmap-design.md`
Phase 1: the reducer must be measured against real historical data
(`scripts/analysis/measure_ast_hot_reducer.py`) before anything downstream
consumes it, matching the program charter's own "measure before minting"
process rule (sec 7).
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.attention_self_model import AttentionSelfModelV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.self_state import SelfStateV1

# 2x the live ORION_ATTENTION_BROADCAST_INTERVAL_SEC default (30s, confirmed
# live 2026-07-18 via `docker exec orion-athena-substrate-runtime env`) —
# same 2x-tick-interval convention `orion/substrate/felt_state_reader.py`
# already uses for its own staleness gates (curiosity_signals, reverie).
DEFAULT_BROADCAST_STALE_THRESHOLD_SEC = 60.0


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def reduce_attention_self_model(
    broadcast: AttentionBroadcastProjectionV1 | None,
    field_frame: FieldAttentionFrameV1 | None,
    self_state: SelfStateV1 | None,
    *,
    now: datetime | None = None,
    broadcast_stale_threshold_sec: float = DEFAULT_BROADCAST_STALE_THRESHOLD_SEC,
) -> AttentionSelfModelV1:
    """Unify the GWT-dispatch lane and the general field lane into one
    inspectable AST/HOT self-model. Never raises: any of the three inputs may
    be `None` (partial data honestly represented, not a crash).

    `now` is the reference tick this self-model is being generated *for* —
    normally the field-lane tick's own `generated_at` (the highest-frequency
    real signal, so it drives the replay/live cadence per the Phase 1 design).
    Falls back to `self_state.generated_at`, then `broadcast.generated_at`,
    then wall-clock `datetime.now(timezone.utc)` if none of the three inputs
    are present.
    """

    reference_ts = (
        now
        or (field_frame.generated_at if field_frame is not None else None)
        or (self_state.generated_at if self_state is not None else None)
        or (broadcast.generated_at if broadcast is not None else None)
        or datetime.now(timezone.utc)
    )
    if reference_ts.tzinfo is None:
        reference_ts = reference_ts.replace(tzinfo=timezone.utc)

    model = AttentionSelfModelV1(generated_at=reference_ts)

    # --- Field lane: what's currently salient -------------------------------
    if field_frame is not None:
        model.field_lane_present = True
        model.field_overall_salience = _round_or_none(field_frame.overall_salience)
        model.field_salient_target_ids = [t.target_id for t in field_frame.dominant_targets]

    # --- Self-state: predicted shift + a confidence fallback ---------------
    if self_state is not None:
        model.self_state_present = True
        top_drift_dim: str | None = None
        top_drift_val = 0.0
        for dim_id, delta in (self_state.dimension_trajectory or {}).items():
            if abs(delta) > abs(top_drift_val):
                top_drift_dim, top_drift_val = dim_id, delta
        if self_state.trajectory_condition != "unknown":
            if top_drift_dim is not None:
                model.predicted_shift = (
                    f"trajectory={self_state.trajectory_condition}; "
                    f"largest-moving dimension={top_drift_dim} (delta={top_drift_val:+.3f})"
                )
                model.predicted_shift_basis = (
                    "self_state.trajectory_condition + self_state.dimension_trajectory"
                )
            else:
                model.predicted_shift = f"trajectory={self_state.trajectory_condition}"
                model.predicted_shift_basis = "self_state.trajectory_condition"

    # --- Broadcast lane: what was last dispatched + cadence-mismatch state -
    broadcast_age_sec: float | None = None
    broadcast_stale = True
    if broadcast is not None:
        model.broadcast_lane_present = True
        model.broadcast_selected_action_type = broadcast.selected_action_type
        model.broadcast_selected_open_loop_id = broadcast.selected_open_loop_id
        model.broadcast_selected_description = broadcast.selected_description
        model.broadcast_attended_node_ids = list(broadcast.attended_node_ids)

        b_ts = broadcast.generated_at
        if b_ts.tzinfo is None:
            b_ts = b_ts.replace(tzinfo=timezone.utc)
        broadcast_age_sec = (reference_ts - b_ts).total_seconds()
        # A negative age means the broadcast snapshot is *newer* than the
        # reference tick (the only real-world case this occurs today: the
        # broadcast lane is a singleton upsert row — see module docstring —
        # so a historical replay tick that predates the single available
        # snapshot has, honestly, no broadcast data at all for that moment;
        # treat it as absent rather than guessing what an earlier snapshot
        # might have looked like).
        if broadcast_age_sec < 0:
            model.broadcast_lane_present = False
            model.broadcast_selected_action_type = None
            model.broadcast_selected_open_loop_id = None
            model.broadcast_selected_description = None
            model.broadcast_attended_node_ids = []
            broadcast_age_sec = None
            broadcast_stale = True
        else:
            broadcast_stale = broadcast_age_sec > broadcast_stale_threshold_sec

    model.broadcast_lane_age_sec = _round_or_none(broadcast_age_sec, 3)
    model.broadcast_lane_stale = broadcast_stale

    # --- Why: branch explicitly on voluntary_override -----------------------
    override = None
    if (
        broadcast is not None
        and model.broadcast_lane_present
        and not broadcast_stale
        and broadcast.frame is not None
    ):
        override = broadcast.frame.voluntary_override

    if override is not None:
        model.attention_reason = "top_down_override"
        model.voluntary_override = override
        model.confidence = _round_or_none(broadcast.coalition_stability_score)
        model.confidence_basis = "broadcast.coalition_stability_score (fresh, top-down)"
        model.reason_narrative = (
            f"Top-down goal bias flipped the winner: loop '{override.chosen_loop_id}' "
            f"(bottom_up={override.chosen_bottom_up:.2f}) beat '{override.beat_loop_id}' "
            f"(bottom_up={override.beat_bottom_up:.2f}) via applied_bias="
            f"{override.applied_bias:.2f}, effort_spent={override.effort_spent:.2f}, "
            f"goal_artifact_id={override.goal_artifact_id!r}, "
            f"goal_drive_origin={override.goal_drive_origin!r}."
        )
    elif model.broadcast_lane_present and not broadcast_stale:
        model.attention_reason = "bottom_up_salience"
        model.confidence = _round_or_none(broadcast.coalition_stability_score)
        model.confidence_basis = "broadcast.coalition_stability_score (fresh, bottom-up)"
        model.reason_narrative = (
            f"Pure bottom-up dispatch: '{model.broadcast_selected_open_loop_id}' "
            f"selected by salience alone; no active goal override at this tick."
        )
    elif model.field_lane_present or model.self_state_present:
        model.attention_reason = "field_salience_only"
        if self_state is not None:
            model.confidence = _round_or_none(self_state.overall_confidence)
            model.confidence_basis = "self_state.overall_confidence (broadcast lane stale/absent)"
        elif field_frame is not None and field_frame.dominant_targets:
            scores = [t.confidence_score for t in field_frame.dominant_targets]
            model.confidence = _round_or_none(sum(scores) / len(scores))
            model.confidence_basis = (
                "mean(field_attention_frame.dominant_targets[].confidence_score) "
                "(broadcast lane stale/absent)"
            )
        stale_note = (
            "no new GWT-dispatch-lane activity since last frame"
            if model.broadcast_lane_present
            else "no GWT-dispatch-lane data available"
        )
        n_targets = len(model.field_salient_target_ids)
        model.reason_narrative = (
            f"{stale_note}; reading field-lane salience only "
            f"({n_targets} dominant target(s), "
            f"overall_salience={model.field_overall_salience})."
        )
    else:
        model.attention_reason = "no_data"
        model.reason_narrative = "No attention data available from either lane."

    return model
