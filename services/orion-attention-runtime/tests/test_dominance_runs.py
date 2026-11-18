from datetime import datetime, timedelta, timezone
import json

import pytest
from pydantic import ValidationError

from app.dominance_runs import advance_run
from orion.schemas.field_dominance_run import FieldDominanceRunV1

START = datetime(2026, 10, 9, tzinfo=timezone.utc)


def tick(state, target, i, **overrides):
    args = dict(target_id=target, target_kind="node" if target else None,
                observed_at=START + timedelta(seconds=i * 2), field_tick_id=f"tick-{i}",
                frame_id=f"frame-{i}", min_streak=3)
    args.update(overrides)
    return advance_run(state, **args)


def test_records_short_and_long_runs_only_on_transition():
    state, row = tick(None, "A", 0)
    assert row is None
    state, row = tick(state, "B", 1)
    assert row.target_id == "A"
    assert row.tick_count == 1  # not censored by the 3-tick goal debounce
    assert row.started_at == START
    assert row.ended_at == START + timedelta(seconds=2)
    assert row.first_source_attention_frame_id == row.last_source_attention_frame_id == "frame-0"
    for i in range(2, 101):
        state, row = tick(state, "B", i)
        assert row is None
    state, row = tick(state, "A", 101)
    assert row.target_id == "B"
    assert row.tick_count == 100
    assert row.first_source_attention_frame_id == "frame-1"
    assert row.last_source_attention_frame_id == "frame-100"
    assert row.ended_at == START + timedelta(seconds=202)


def test_no_target_closes_run_and_reentry_starts_fresh():
    state, _ = tick(None, "A", 0)
    state, row = tick(state, None, 1)
    assert row.tick_count == 1
    assert state["active"] is None
    state, row = tick(state, None, 2)
    assert row is None
    state, row = tick(state, "A", 3)
    assert row is None
    assert state["active"]["tick_count"] == 1
    assert state["active"]["first_source_attention_frame_id"] == "frame-3"


def test_restart_and_duplicate_do_not_reset_or_double_count():
    state, _ = tick(None, "A", 0)
    state, _ = tick(state, "A", 1)
    restarted = json.loads(json.dumps(state))
    replay, row = tick(restarted, "A", 1)
    assert replay == state and row is None
    state, row = tick(replay, "B", 2)
    assert row.tick_count == 2
    assert row.first_source_attention_frame_id == "frame-0"


def test_stale_tick_cannot_rewind_or_close_current_run():
    state, _ = tick(None, "A", 2)
    after, row = tick(state, "B", 1)
    assert after == state and row is None


def test_partial_first_run_and_threshold_at_start_are_preserved():
    state, _ = tick(None, "A", 0, left_censored=True)
    state, _ = tick(state, "A", 1, min_streak=7)
    state, row = tick(state, "B", 2, min_streak=7)
    assert row.left_censored
    assert row.min_streak_at_run == 3
    state, row = tick(state, None, 3)
    assert not row.left_censored
    assert row.min_streak_at_run == 7


def test_contract_rejects_impossible_durations_and_counts():
    state, _ = tick(None, "A", 0)
    _, row = tick(state, "B", 1)
    for changes in ({"tick_count": 0}, {"ended_at": START - timedelta(seconds=1)},
                    {"started_at": START.replace(tzinfo=None)}):
        with pytest.raises(ValidationError):
            FieldDominanceRunV1.model_validate(row.model_dump() | changes)
