"""A stale contender must not beat a fresh one in the channel merge.

`prediction_error` is not in NODE_DECAY_CHANNELS (verified 2026-08-14: 28
entries, absent), so a value written by a slow producer persists byte-identical
until that producer fires again. Live, `node:substrate.codebase` changed value
6 times in 6,000 ticks while winning the merge on 68.7% of them.

Every expected value below is exact and hand-derived.
"""
from datetime import datetime, timedelta, timezone

import pytest

from orion.field.pressure import collect_field_channel_pressures
from orion.schemas.field_state import FieldStateV1

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
FRESH = NOW - timedelta(seconds=5)
STALE = NOW - timedelta(seconds=600)
THRESHOLD = 90.0


def _state(node_vectors, updated_at=None):
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_stale",
        node_vectors=node_vectors,
        node_vector_updated_at=updated_at or {},
    )


def test_default_is_byte_identical_to_the_historical_behaviour() -> None:
    """Opt-in only. With no threshold the stale high value still wins, exactly
    as it does on main -- so merging this patch changes nothing until a caller
    passes a threshold.

    Hand-computed: max(0.90, 0.20) = 0.90, won by node:slow.
    """
    state = _state(
        {"node:slow": {"prediction_error": 0.90}, "node:fast": {"prediction_error": 0.20}},
        {"node:slow": {"prediction_error": STALE}, "node:fast": {"prediction_error": FRESH}},
    )
    values, provenance = collect_field_channel_pressures(state)
    assert values["prediction_error"] == pytest.approx(0.90)
    assert provenance["prediction_error"] == "node:slow"


def test_a_stale_source_loses_to_a_fresh_one() -> None:
    """The fix. Same inputs as above, threshold supplied.

    Hand-computed: node:slow's write is 600s old (> 90s) and node:fast has a
    fresh write for the same channel, so node:slow is dropped and the merge
    becomes max(0.20) = 0.20.
    """
    state = _state(
        {"node:slow": {"prediction_error": 0.90}, "node:fast": {"prediction_error": 0.20}},
        {"node:slow": {"prediction_error": STALE}, "node:fast": {"prediction_error": FRESH}},
    )
    values, provenance = collect_field_channel_pressures(
        state, staleness_threshold_sec=THRESHOLD
    )
    assert values["prediction_error"] == pytest.approx(0.20)
    assert provenance["prediction_error"] == "node:fast"


def test_a_stale_source_still_wins_when_every_source_is_stale() -> None:
    """The guard against fabricating an absence. A channel quiet everywhere is
    quiet, not missing -- dropping every contender would invent a gap.

    Hand-computed: both writes 600s old, so neither is excluded and the merge
    is max(0.90, 0.20) = 0.90, unchanged.
    """
    state = _state(
        {"node:a": {"prediction_error": 0.90}, "node:b": {"prediction_error": 0.20}},
        {"node:a": {"prediction_error": STALE}, "node:b": {"prediction_error": STALE}},
    )
    values, provenance = collect_field_channel_pressures(
        state, staleness_threshold_sec=THRESHOLD
    )
    assert values["prediction_error"] == pytest.approx(0.90)
    assert provenance["prediction_error"] == "node:a"


def test_a_source_with_no_timestamp_always_participates() -> None:
    """A missing timestamp is not evidence of staleness. 75 of 149 live node
    channels carry none, and capability vectors carry none at all -- treating
    absent as stale would silently drop half the mesh.

    Hand-computed: node:untimed has no stamp, so it is never excluded;
    max(0.90, 0.20) = 0.90.
    """
    state = _state(
        {"node:untimed": {"prediction_error": 0.90}, "node:fast": {"prediction_error": 0.20}},
        {"node:fast": {"prediction_error": FRESH}},
    )
    values, provenance = collect_field_channel_pressures(
        state, staleness_threshold_sec=THRESHOLD
    )
    assert values["prediction_error"] == pytest.approx(0.90)
    assert provenance["prediction_error"] == "node:untimed"


def test_staleness_is_per_channel_not_per_node() -> None:
    """A node stale on one channel must keep contributing to the others.
    Hand-computed: node:a's prediction_error is stale (dropped, node:b is
    fresh) but its cpu_pressure is fresh and still wins max(0.80, 0.10).
    """
    state = _state(
        {
            "node:a": {"prediction_error": 0.90, "cpu_pressure": 0.80},
            "node:b": {"prediction_error": 0.20, "cpu_pressure": 0.10},
        },
        {
            "node:a": {"prediction_error": STALE, "cpu_pressure": FRESH},
            "node:b": {"prediction_error": FRESH, "cpu_pressure": FRESH},
        },
    )
    values, provenance = collect_field_channel_pressures(
        state, staleness_threshold_sec=THRESHOLD
    )
    assert values["prediction_error"] == pytest.approx(0.20)
    assert provenance["prediction_error"] == "node:b"
    assert values["cpu_pressure"] == pytest.approx(0.80)
    assert provenance["cpu_pressure"] == "node:a"


def test_the_threshold_boundary_is_strict() -> None:
    """Pins `>` rather than `>=`. A write exactly at the threshold is NOT
    stale; one a second past it is."""
    at = _state(
        {"node:a": {"prediction_error": 0.90}, "node:b": {"prediction_error": 0.20}},
        {
            "node:a": {"prediction_error": NOW - timedelta(seconds=THRESHOLD)},
            "node:b": {"prediction_error": FRESH},
        },
    )
    values, _ = collect_field_channel_pressures(at, staleness_threshold_sec=THRESHOLD)
    # Exactly THRESHOLD old: not strictly greater, so node:a survives. Only
    # deterministic because staleness is measured against generated_at (NOW),
    # not wall-clock -- against wall-clock this assertion is a race.
    assert values["prediction_error"] == pytest.approx(0.90)

    past = _state(
        {"node:a": {"prediction_error": 0.90}, "node:b": {"prediction_error": 0.20}},
        {
            "node:a": {"prediction_error": NOW - timedelta(seconds=THRESHOLD + 5)},
            "node:b": {"prediction_error": FRESH},
        },
    )
    values, _ = collect_field_channel_pressures(past, staleness_threshold_sec=THRESHOLD)
    assert values["prediction_error"] == pytest.approx(0.20)


def test_higher_is_better_channels_use_the_same_staleness_rule() -> None:
    """`availability` merges by min() (worst reading wins). A stale WORST
    reading must not pin the channel low forever either -- the failure is
    symmetric.

    Hand-computed: node:a's stale 0.10 is dropped, leaving min(0.90) = 0.90.
    """
    state = _state(
        {"node:a": {"availability": 0.10}, "node:b": {"availability": 0.90}},
        {"node:a": {"availability": STALE}, "node:b": {"availability": FRESH}},
    )
    values, provenance = collect_field_channel_pressures(
        state, staleness_threshold_sec=THRESHOLD
    )
    assert values["availability"] == pytest.approx(0.90)
    assert provenance["availability"] == "node:b"
