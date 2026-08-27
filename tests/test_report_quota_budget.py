"""The replay harness in `scripts/report_quota_budget.py`.

The script holds the decision-critical logic -- the clock grid, the warm-up
exclusion, and the deliberate absence of a verdict -- and had no tests until
code review 2026-08-27 said so. `replay()` is pure over a tick list, so none of
this needs a database.

Every expected count below is hand-computed and written as a literal.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from datetime import datetime, timedelta, timezone

import pytest

from orion.autonomy.quota_budget import LedgerTick

_SPEC = importlib.util.spec_from_file_location(
    "report_quota_budget",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "report_quota_budget.py",
)
assert _SPEC and _SPEC.loader
rqb = importlib.util.module_from_spec(_SPEC)
# Register before exec: @dataclass resolves annotations via
# sys.modules[cls.__module__], which is None for an unregistered module.
sys.modules["report_quota_budget"] = rqb
_SPEC.loader.exec_module(rqb)

T0 = datetime(2026, 8, 20, 0, 0, tzinfo=timezone.utc)


def tick(minutes: int, cost: float | None = 1.00, tokens: int = 100, unpriced: int = 0):
    return LedgerTick(
        observed_at=T0 + timedelta(minutes=minutes),
        total_tokens=tokens,
        cost_usd=cost,
        unpriced_session_count=unpriced,
    )


def test_empty_history_replays_nothing_and_concludes_nothing():
    r = rqb.replay([], allowance_usd=100.0, ask_usd=2.50, window_hours=5.0)
    assert r.considered == 0
    assert r.first is None and r.last is None


def test_too_little_history_for_one_window_yields_no_decision_points():
    """Ticks spanning 2h with a 5h window: the grid starts after the first full
    window, which is past the end of the data. Nothing to ask.
    """
    ticks = [tick(m) for m in range(0, 121, 15)]
    r = rqb.replay(ticks, allowance_usd=100.0, ask_usd=2.50, window_hours=5.0)
    assert r.considered == 0


def test_clock_grid_excludes_the_warm_up_window():
    """Ticks every 15 min from t=0 to t=600 (41 ticks). Window 5h = 300 min,
    step 60 min.

    Grid runs from first_tick + 300 = 300 through last_tick = 600, stepping 60:
    300, 360, 420, 480, 540, 600 -> 6 decision points, hand-counted.
    The 0-300 warm-up, whose windows are truncated at the query cutoff, is not
    sampled at all.
    """
    ticks = [tick(m) for m in range(0, 601, 15)]
    r = rqb.replay(
        ticks, allowance_usd=10_000.0, ask_usd=1.0, window_hours=5.0, step_minutes=60.0
    )

    assert r.considered == 6
    assert r.first == T0 + timedelta(minutes=300)
    assert r.last == T0 + timedelta(minutes=600)
    assert r.refused == 0  # allowance far above any window's spend


def test_a_producer_gap_is_sampled_and_fails_closed():
    """This is what the tick-driven grid structurally could not do.

    Ticks at 0..300 (every 15 min), then a gap, then one tick at 900.
    Window 300 min, step 60. Grid: 300, 360, ..., 900 -> 11 points.

    Windows [360,660], [420,720], [480,780], [540,840] contain no tick at all,
    so points 660, 720, 780 and 840 are UNKNOWN -> 4, hand-counted. Every other
    point has at least one tick behind it.
    """
    ticks = [tick(m) for m in range(0, 301, 15)] + [tick(900)]
    r = rqb.replay(
        ticks, allowance_usd=10_000.0, ask_usd=1.0, window_hours=5.0, step_minutes=60.0
    )

    assert r.considered == 11
    assert r.refused_unknown == 4
    # Refused despite a $10,000 allowance and a $1 ask: unknown fails closed.
    assert r.refused == 4
    assert r.refused_on_observed == 0


def test_refusals_on_observed_spend_are_counted_separately():
    """Ticks every 15 min from 0 to 600 at $10.00 each. Window 300 min holds at
    most 21 ticks (inclusive both ends) = $210.00. With a $50 allowance every
    decision point is over, so all 6 refuse, and none of them on unknown.
    """
    ticks = [tick(m, cost=10.00) for m in range(0, 601, 15)]
    r = rqb.replay(
        ticks, allowance_usd=50.0, ask_usd=1.0, window_hours=5.0, step_minutes=60.0
    )

    assert r.considered == 6
    assert r.refused == 6
    assert r.refused_unknown == 0
    assert r.refused_on_observed == 6
    assert r.exhausted_moments == 6
    assert r.peak_spend_usd == pytest.approx(210.00)


def test_floor_windows_are_counted():
    ticks = [tick(m, cost=1.00, unpriced=2) for m in range(0, 601, 15)]
    r = rqb.replay(
        ticks, allowance_usd=10_000.0, ask_usd=1.0, window_hours=5.0, step_minutes=60.0
    )
    assert r.considered == 6
    assert r.floor_moments == 6


def test_unconfigured_allowance_yields_no_decision_points():
    """Below MIN_MEANINGFUL_ALLOWANCE_USD every quota_state is None, so nothing
    is considered -- and the printer must not turn that into a verdict.
    """
    ticks = [tick(m) for m in range(0, 601, 15)]
    r = rqb.replay(ticks, allowance_usd=0.0, ask_usd=1.0, window_hours=5.0, step_minutes=60.0)
    assert r.considered == 0


def test_no_decision_points_renders_no_verdict(capsys):
    """The bug this pins: falling through to `real_refusals == 0` and printing
    a confident 'GATE: FAILED ... Do NOT wire it into the allocator' having
    replayed nothing at all.
    """
    r = rqb.replay([], allowance_usd=100.0, ask_usd=2.50, window_hours=5.0)
    rqb.print_replay(r, allowance_usd=100.0, ask_usd=2.50, window_hours=5.0)

    out = capsys.readouterr().out
    assert "NO DECISION POINTS" in out
    assert "GATE" not in out
    assert "FAILED" not in out
    assert "PASSED" not in out


def test_a_real_replay_renders_counts_but_still_no_verdict(capsys):
    """The dollar denominator is refuted, so a refusal count must not be
    dressed as a pass. Counts yes, verdict no.
    """
    ticks = [tick(m, cost=10.00) for m in range(0, 601, 15)]
    r = rqb.replay(ticks, allowance_usd=50.0, ask_usd=1.0, window_hours=5.0, step_minutes=60.0)
    rqb.print_replay(r, allowance_usd=50.0, ask_usd=1.0, window_hours=5.0)

    out = capsys.readouterr().out
    assert "would have refused:     6" in out
    assert "NO VERDICT IS RENDERED" in out
    assert "PASSED" not in out
    assert "until it bites" not in out


def test_trailing_window_elapsed_is_one():
    """A trailing N-hour window is fully elapsed by definition. Pinned because
    it is what makes `pace` and `projected_window_usd` degenerate here, which is
    why the script never prints them.
    """
    assert rqb.TRAILING_WINDOW_ELAPSED == 1.0
