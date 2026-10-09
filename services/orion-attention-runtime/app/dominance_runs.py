"""Segment the existing selected winner sequence without selecting or scoring it.

An open run is a checkpoint, not a completed history row. A no-winner tick closes
it and leaves no active run. Timestamps are attention frame times; ended_at is
the transition time, while the last frame reference still belongs to the old run.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime

from orion.schemas.field_dominance_run import FieldDominanceRunV1


def advance_run(
    state: dict | None,
    *,
    target_id: str | None,
    target_kind: str | None,
    observed_at: datetime,
    field_tick_id: str,
    frame_id: str,
    min_streak: int,
    left_censored: bool = False,
) -> tuple[dict, FieldDominanceRunV1 | None]:
    """One real tick in, at most one completed run out. No debounce filtering.

    Replays of the last tick and out-of-order observations do not advance the
    checkpoint. No duration is inferred from the polling interval or old counter.
    """
    state = deepcopy(state) if state else {}
    if state and (
        state["last_field_tick_id"] == field_tick_id
        or observed_at <= datetime.fromisoformat(state["observed_at"])
    ):
        return state, None
    active = state.get("active")
    completed = None
    if active and (active["target_id"], active["target_kind"]) != (target_id, target_kind):
        completed = FieldDominanceRunV1(**active, ended_at=observed_at)
        active = None
    if target_id is not None:
        if not target_kind:
            raise ValueError("a focused target requires its kind")
        if active is None:
            active = dict(
                run_id=f"field-dominance-run:{frame_id}",
                target_id=target_id,
                target_kind=target_kind,
                started_at=observed_at.isoformat(),
                tick_count=1,
                min_streak_at_run=min_streak,
                first_source_attention_frame_id=frame_id,
                last_source_attention_frame_id=frame_id,
                left_censored=left_censored,
            )
        else:
            active["tick_count"] += 1
            active["last_source_attention_frame_id"] = frame_id
    return dict(last_field_tick_id=field_tick_id, observed_at=observed_at.isoformat(), active=active), completed
