from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.schemas.field_state import FieldStateV1
from orion.self_state.scoring import collect_field_channel_pressures

from app.digestion.perturbation import (
    RECENT_PERTURBATION_WINDOW_SECONDS,
    apply_perturbations,
)
from app.ingest.state_deltas import Perturbation

BASE = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


def _empty_state(tick_id: str) -> FieldStateV1:
    return FieldStateV1(generated_at=BASE, tick_id=tick_id)


def _perturbation(label: str) -> Perturbation:
    return Perturbation(node_id="node:athena", channel="execution_load", intensity=0.1, label=label)


def test_more_than_old_cap_within_one_instant_does_not_saturate_forever() -> None:
    # Regression guard for the original bug: the OLD mechanism hard-capped
    # recent_perturbations to the last 20 distinct labels EVER seen, with no
    # expiry -- once 20 distinct labels had ever arrived, recent_perturbation_count
    # was permanently pinned at 1.0, even if the burst that produced them was
    # a one-off and nothing followed for hours. This test applies 40 distinct
    # perturbations at a single instant (`now` fixed), then advances well past
    # the new window and applies nothing further -- the count must decay back
    # toward 0, which the old mechanism could never do.
    state = _empty_state("tick_burst")
    burst = [_perturbation(f"delta_{i}") for i in range(40)]
    apply_perturbations(state, burst, now=BASE)

    # Immediately after the burst, the window is still fresh -- saturated is
    # expected and correct (a genuine burst of real, current activity).
    channels, _ = collect_field_channel_pressures(state)
    assert channels["recent_perturbation_count"] == 1.0

    # Advance well past the window with no further activity. A tick still
    # runs (apply_perturbations is called every tick regardless of whether
    # perturbations is empty), so the window gets a chance to expire stale
    # entries.
    later = BASE + timedelta(seconds=RECENT_PERTURBATION_WINDOW_SECONDS + 30)
    apply_perturbations(state, [], now=later)

    channels_later, _ = collect_field_channel_pressures(state)
    assert channels_later.get("recent_perturbation_count", 0.0) < 1.0
    assert state.recent_perturbations == []
    assert state.recent_perturbation_at == []


def test_quiet_baseline_rate_does_not_saturate_within_a_window() -> None:
    # Real observed quiet-baseline rate (substrate_field_applied_deltas,
    # 2026-07-16 20:28-21:10 UTC): ~6-10 distinct applied deltas/minute.
    # Simulated here as one new distinct label every 6 seconds across a
    # single 60s window (10 total) -- must NOT saturate to 1.0, unlike the
    # old last-20-ever cap which would have hit 1.0 within ~2 minutes of any
    # sustained traffic and then stayed there permanently.
    state = _empty_state("tick_baseline")
    for i in range(10):
        ts = BASE + timedelta(seconds=6 * i)
        apply_perturbations(state, [_perturbation(f"baseline_{i}")], now=ts)

    channels, _ = collect_field_channel_pressures(state)
    assert channels["recent_perturbation_count"] < 1.0
    assert len(state.recent_perturbations) == 10


def test_labels_older_than_window_are_pruned_on_next_tick() -> None:
    state = _empty_state("tick_prune")
    apply_perturbations(state, [_perturbation("old_one")], now=BASE)
    assert state.recent_perturbations == ["old_one"]

    just_past_window = BASE + timedelta(seconds=RECENT_PERTURBATION_WINDOW_SECONDS + 1)
    apply_perturbations(state, [_perturbation("new_one")], now=just_past_window)

    assert state.recent_perturbations == ["new_one"]
    assert state.recent_perturbation_at == [just_past_window]


def test_recent_perturbations_and_timestamps_stay_index_aligned() -> None:
    state = _empty_state("tick_align")
    apply_perturbations(state, [_perturbation("a"), _perturbation("b")], now=BASE)
    assert len(state.recent_perturbations) == len(state.recent_perturbation_at) == 2
    assert state.recent_perturbations == ["a", "b"]
    assert state.recent_perturbation_at == [BASE, BASE]


def test_duplicate_label_within_window_not_double_counted() -> None:
    state = _empty_state("tick_dup")
    apply_perturbations(state, [_perturbation("dup"), _perturbation("dup")], now=BASE)
    assert state.recent_perturbations == ["dup"]
    assert len(state.recent_perturbation_at) == 1
