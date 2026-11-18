"""Branch-level must-not tests for Hub's arousal readers (Temporal Self rev 4, R3, risk 10).

Readers and protected gates share one function: quiet hours / turn-in-flight / recent-chat /
the daily cap sit beside the outreach cooldown, and the curiosity hour window and cap sit
beside its cooldown. So for every protected reason, the decision must be identical under all
four arousal values and under no arousal at all -- and arousal may only ever ADD a hold,
never remove one. (The spec names a repo-root file; Hub's `scripts` package collides with the
root `scripts` package, so the Hub half lives here and the dream half in
services/orion-dream/tests/test_dream_reads_arousal.py.)
"""
from __future__ import annotations

import inspect
import itertools

import pytest

from scripts.curiosity_investigation import (
    CuriosityInvestigation,
    SchedulingGateInputs,
    scheduling_block_reason,
)
from scripts.endogenous_outreach import OutreachGateInputs, outreach_block_reason

AROUSAL = (None, "engaged", "idle", "strained", "unknown")
ADDED_BY_AROUSAL = {"arousal_strained", "arousal_engaged", "arousal_engaged_cooldown"}


def _sched_variants(level):
    """Every way a curiosity line can carry arousal: pacing (stretch), self-inquiry (idle-only)."""
    # "unknown" never reaches the gate (the reader returns None); passed anyway as a guard.
    yield dict(arousal_level=level)
    yield dict(arousal_level=level, arousal_engaged_cooldown_sec=99999.0)
    yield dict(arousal_level=level, arousal_idle_only=True)


SCHED_PROTECTED = {
    "disabled": dict(enabled=False),
    "daily_cap": dict(done_today=7, daily_cap=7),
    "outside_window": dict(local_hour=3, window_start_hour=8, window_end_hour=22),
    "cooldown": dict(seconds_since_last=60.0),
    "rest_drive_cooldown": dict(rest_drive_cooldown_sec=99999.0),
}


@pytest.mark.parametrize("reason", sorted(SCHED_PROTECTED))
def test_curiosity_protected_reasons_are_invariant_under_arousal(reason):
    base = dict(enabled=True, seconds_since_last=3000.0, min_cooldown_sec=1800.0, done_today=0, daily_cap=7)
    base.update(SCHED_PROTECTED[reason])
    for level in AROUSAL:
        for extra in _sched_variants(level):
            assert scheduling_block_reason(SchedulingGateInputs(**base, **extra)) == reason, (level, extra)


OUT_PROTECTED = {
    "disabled": dict(enabled=False),
    "turn_in_flight": dict(turn_in_flight=True),
    "recent_chat": dict(seconds_since_juniper_spoke=60.0, recent_chat_sec=900.0),
    "quiet_hours": dict(local_hour=2),
    "daily_cap": dict(sent_today=4),
    "cooldown": dict(seconds_since_last_outreach=60.0),
}


@pytest.mark.parametrize("reason", sorted(OUT_PROTECTED))
@pytest.mark.parametrize("skip", [False, True])
def test_outreach_protected_reasons_are_invariant_under_arousal(reason, skip):
    base = dict(enabled=True, turn_in_flight=False, local_hour=14, quiet_start_hour=23, quiet_end_hour=8,
                seconds_since_last_outreach=4000.0, min_cooldown_sec=2700.0, sent_today=0, daily_cap=4)
    base.update(OUT_PROTECTED[reason])
    expected = outreach_block_reason(OutreachGateInputs(**base), skip_schedule_gates=skip)
    for level in AROUSAL:
        got = outreach_block_reason(OutreachGateInputs(**base, arousal_level=level), skip_schedule_gates=skip)
        assert got == expected, (level, skip)


def test_curiosity_arousal_never_loosens_any_decision():
    """Exhaustive over a small grid: a hold without arousal is the same hold with it, and a
    pass with arousal was a pass without it."""
    grid = itertools.product(
        [True, False],                       # enabled
        [None, 60.0, 3000.0, 99999.0],       # seconds_since_last
        [0, 7],                              # done_today (cap 7)
        [None, 3],                           # local_hour (window 8-22)
        [None, 5000.0],                      # rest_drive_cooldown_sec
    )
    for enabled, since, done, hour, rest in grid:
        base = dict(enabled=enabled, seconds_since_last=since, min_cooldown_sec=1800.0, done_today=done,
                    daily_cap=7, local_hour=hour, window_start_hour=8, window_end_hour=22,
                    rest_drive_cooldown_sec=rest)
        without = scheduling_block_reason(SchedulingGateInputs(**base))
        for level in AROUSAL:
            for extra in _sched_variants(level):
                got = scheduling_block_reason(SchedulingGateInputs(**base, **extra))
                if without is not None:
                    assert got == without, (base, extra)
                else:
                    assert got is None or got in ADDED_BY_AROUSAL, (base, extra, got)


def test_outreach_arousal_never_loosens_any_decision():
    grid = itertools.product(
        [True, False], [False, True], [2, 14], [None, 60.0, 4000.0], [0, 4], [None, 60.0], [None, 99999.0],
        [False, True],
    )
    for enabled, busy, hour, since, sent, spoke, rest, skip in grid:
        base = dict(enabled=enabled, turn_in_flight=busy, local_hour=hour, quiet_start_hour=23, quiet_end_hour=8,
                    seconds_since_last_outreach=since, min_cooldown_sec=2700.0, sent_today=sent, daily_cap=4,
                    seconds_since_juniper_spoke=spoke, recent_chat_sec=900.0, rest_drive_cooldown_sec=rest)
        without = outreach_block_reason(OutreachGateInputs(**base), skip_schedule_gates=skip)
        for level in AROUSAL:
            got = outreach_block_reason(OutreachGateInputs(**base, arousal_level=level), skip_schedule_gates=skip)
            if without is not None:
                assert got == without
            else:
                assert got is None or got == "arousal_strained"


def test_urgent_runs_never_read_arousal():
    """Juniper- or watcher-requested urgent investigations bypass every pacing gate; arousal
    must not become the first one they read."""
    assert "arousal" not in inspect.getsource(CuriosityInvestigation.start_urgent)
