"""Implicit `decayed_unattended` derivation. Fixtures mirror the real live loops."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.substrate.attention.implicit_outcome import (
    DEFAULT_MIN_SILENCE,
    IMPLICIT_VERDICT,
    LoopObservation,
    derive_implicit_verdicts,
    median_gap,
)

NOW = datetime(2026, 8, 14, 18, 0, tzinfo=timezone.utc)


def _times(*hours_ago: float) -> list[datetime]:
    return sorted(NOW - timedelta(hours=h) for h in hours_ago)


def test_the_real_decayed_loop_is_labelled():
    """`open-loop-9d84d08cddf5` live: scored 4x on 2026-07-24, never again, 509h
    of silence, never labelled. The exact case this module exists for."""
    loop = LoopObservation(
        loop_id="open-loop-9d84d08cddf5",
        theme_key="theme-x",
        trace_times=_times(509, 509.2, 509.4, 509.6),
        last_salience=0.356,
    )
    out = derive_implicit_verdicts([loop], now=NOW)
    assert len(out) == 1
    assert out[0].verdict == IMPLICIT_VERDICT
    assert out[0].loop_id == "open-loop-9d84d08cddf5"
    assert out[0].salience_at_close == pytest.approx(0.356)


def test_a_human_verdict_is_never_overwritten():
    """`open-loop-e11d318a9036` live: 339h silent, but a human already resolved
    it. Relabelling a human judgement would corrupt the only clean labels there
    are."""
    loop = LoopObservation(
        loop_id="open-loop-e11d318a9036",
        theme_key="t",
        trace_times=_times(339, 340, 341, 342),
        existing_verdict="resolved",
    )
    assert derive_implicit_verdicts([loop], now=NOW) == []


@pytest.mark.parametrize("verdict", ["resolved", "dismissed"])
def test_both_terminal_verdicts_are_respected(verdict):
    loop = LoopObservation(
        loop_id="l", theme_key="t", trace_times=_times(900, 901, 902), existing_verdict=verdict
    )
    assert derive_implicit_verdicts([loop], now=NOW) == []


def test_an_actively_rescored_loop_is_not_an_outcome():
    """`open-loop-3be13c7644a0` live: 20h silent, still being re-scored. Not
    resolved, but not decayed either -- unresolved and decayed are different
    states and conflating them would poison the label set."""
    loop = LoopObservation(
        loop_id="open-loop-3be13c7644a0",
        theme_key="t",
        trace_times=_times(20, 30, 44, 60, 90),
    )
    assert derive_implicit_verdicts([loop], now=NOW) == []


def test_slow_cadence_loop_is_not_declared_dead_for_being_slow():
    """Silence past the 24h floor but well inside this loop's own rhythm: it is
    scored every ~5 days, so 30h of quiet is normal. An absolute-only rule would
    mislabel it -- the same failure mode as an absolute sigma_floor."""
    loop = LoopObservation(
        loop_id="slow", theme_key="t", trace_times=_times(30, 150, 270, 390)
    )
    assert derive_implicit_verdicts([loop], now=NOW) == []


def test_slow_cadence_loop_is_labelled_once_silence_clears_its_own_rhythm():
    loop = LoopObservation(
        loop_id="slow", theme_key="t", trace_times=_times(400, 520, 640, 760)
    )
    out = derive_implicit_verdicts([loop], now=NOW)
    assert len(out) == 1
    assert "own median gap" in out[0].reason


def test_inside_the_absolute_floor_nothing_is_labelled():
    """Even a burst-then-stop loop is not decayed after a few hours."""
    loop = LoopObservation(loop_id="l", theme_key="t", trace_times=_times(3, 3.1, 3.2))
    assert derive_implicit_verdicts([loop], now=NOW) == []


def test_loop_with_no_traces_is_not_labelled():
    assert derive_implicit_verdicts(
        [LoopObservation(loop_id="l", theme_key="t", trace_times=[])], now=NOW
    ) == []


def test_too_few_traces_falls_back_to_the_floor_and_says_so():
    loop = LoopObservation(loop_id="l", theme_key="t", trace_times=_times(200, 200.5))
    out = derive_implicit_verdicts([loop], now=NOW)
    assert len(out) == 1
    assert "too few traces for a cadence" in out[0].reason


def test_median_gap_refuses_a_sample_of_one():
    """Two points give one gap -- a sample of one, not a cadence. Returning it
    would repeat the short-window mistake that has already produced two wrong
    claims in this arc."""
    assert median_gap(_times(10, 20)) is None
    assert median_gap(_times(10)) is None
    assert median_gap(_times(10, 20, 30)) == timedelta(hours=10)


def test_default_floor_matches_the_existing_suppress_cooldown():
    """Reuse, not a second divergent constant: attention_loops_store.py's
    suppress_loop cooldown is 86400s."""
    assert DEFAULT_MIN_SILENCE.total_seconds() == 86400.0


def test_features_are_carried_onto_the_label():
    loop = LoopObservation(
        loop_id="l",
        theme_key="t",
        trace_times=_times(500, 501, 502),
        last_salience=0.42,
        last_features={"evidence_strength": 0.3},
    )
    out = derive_implicit_verdicts([loop], now=NOW)
    assert out[0].features_at_close == {"evidence_strength": 0.3}


def test_salience_is_clamped():
    loop = LoopObservation(
        loop_id="l", theme_key="t", trace_times=_times(500, 501, 502), last_salience=7.0
    )
    assert derive_implicit_verdicts([loop], now=NOW)[0].salience_at_close == 1.0
