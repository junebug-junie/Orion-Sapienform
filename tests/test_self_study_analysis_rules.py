"""The shared notability rules for `skills.self_study.analyze.v1`.

These are the whole point of the action: the rules decide whether Orion writes
anything at all, so the tests that matter are the ones proving each rule
REFUSES on the cases that would turn this into digest spam. Two of the refusals
below (`observation_gap` on a slow producer, `new_category` against an empty
baseline) are regression tests for defects the first live smoke against real
Postgres actually produced.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "services" / "orion-cortex-exec"))

from app.self_study_analysis import (  # noqa: E402
    ALL_RULES,
    GAP_MINUTES,
    MEAN_SHIFT_SIGMAS,
    MIN_BASELINE_ROWS,
    MIN_CATEGORY_ROWS,
    SourceWindow,
    _largest_gap_minutes,
    _stdev,
    evaluate_rules,
    finding_digest,
)

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
WINDOW = timedelta(hours=6)
SINCE = NOW - WINDOW


def _window(
    *,
    rows: int | None = None,
    stamps: list[datetime] | None = None,
    numeric: dict[str, list[float]] | None = None,
    categories: dict[str, dict[str, int]] | None = None,
) -> SourceWindow:
    stamps = stamps or []
    return SourceWindow(
        rows=rows if rows is not None else len(stamps),
        numeric=numeric or {},
        categories=categories or {},
        timestamps=stamps,
    )


def _evenly(n: int, *, since: datetime = SINCE, until: datetime = NOW) -> list[datetime]:
    """n stamps spread across the window, so the largest internal gap is small."""
    if n <= 0:
        return []
    step = (until - since) / (n + 1)
    return [since + step * (i + 1) for i in range(n)]


def _fired(recent: SourceWindow, baseline: SourceWindow) -> set[str]:
    findings, _ = evaluate_rules(
        recent=recent, baseline=baseline, recent_since=SINCE, recent_until=NOW
    )
    return {f.rule for f in findings}


# --- the quiet case, which must stay quiet ---------------------------------


def test_a_steady_producer_fires_nothing() -> None:
    recent = _window(
        stamps=_evenly(60),
        numeric={"score": [0.5] * 60},
        categories={"domain": {"git": 30, "graph": 30}},
    )
    baseline = _window(
        stamps=_evenly(58, since=SINCE - WINDOW, until=SINCE),
        numeric={"score": [0.5, 0.51, 0.49] * 19 + [0.5]},
        categories={"domain": {"git": 29, "graph": 29}},
    )
    findings, not_fired = evaluate_rules(
        recent=recent, baseline=baseline, recent_since=SINCE, recent_until=NOW
    )
    assert findings == []
    assert set(not_fired) == set(ALL_RULES)


def test_every_rule_is_reported_as_evaluated() -> None:
    """`rules_not_fired` must always account for the full rule set -- the
    negative space is rendered into the journal body, so a rule silently
    missing from both lists would read as "checked and clean"."""
    recent = _window(stamps=_evenly(30))
    baseline = _window(stamps=_evenly(30, since=SINCE - WINDOW, until=SINCE))
    findings, not_fired = evaluate_rules(
        recent=recent, baseline=baseline, recent_since=SINCE, recent_until=NOW
    )
    assert set(f.rule for f in findings) | set(not_fired) == set(ALL_RULES)


# --- producer_stalled ------------------------------------------------------


def test_producer_stalled_fires_on_an_empty_recent_window() -> None:
    assert "producer_stalled" in _fired(
        _window(rows=0), _window(stamps=_evenly(40, since=SINCE - WINDOW, until=SINCE))
    )


def test_producer_stalled_refuses_when_the_baseline_was_also_empty() -> None:
    """A producer that has never emitted has not stopped."""
    assert _fired(_window(rows=0), _window(rows=0)) == set()


# --- observation_gap -------------------------------------------------------


def test_observation_gap_fires_on_a_stall_inside_an_otherwise_busy_window() -> None:
    dense = [SINCE + timedelta(minutes=i) for i in range(60)]
    dense += [NOW - timedelta(minutes=i) for i in range(30)]
    recent = _window(stamps=dense)
    baseline = _window(stamps=_evenly(90, since=SINCE - WINDOW, until=SINCE))
    fired = _fired(recent, baseline)
    assert "observation_gap" in fired
    assert "producer_stalled" not in fired


def test_observation_gap_refuses_on_a_producer_that_is_always_slow() -> None:
    """REGRESSION (first live smoke, 2026-08-25). concept_induction really
    emits ~2 rows/day, so an absolute 120-min bar fired on it every window
    forever -- true and useless. The bar must be relative to the producer's own
    normal, so a slow-but-unchanged producer stays quiet."""
    slow_recent = [SINCE + timedelta(hours=1), SINCE + timedelta(hours=5)]
    slow_baseline = [
        SINCE - WINDOW + timedelta(hours=1),
        SINCE - WINDOW + timedelta(hours=5),
    ]
    fired = _fired(
        _window(stamps=slow_recent),
        _window(stamps=slow_baseline),
    )
    assert "observation_gap" not in fired


def test_observation_gap_refuses_without_enough_baseline_to_define_normal() -> None:
    thin_baseline = _window(stamps=_evenly(MIN_BASELINE_ROWS - 1, since=SINCE - WINDOW, until=SINCE))
    recent = _window(stamps=[SINCE + timedelta(minutes=1), NOW - timedelta(minutes=1)])
    assert "observation_gap" not in _fired(recent, thin_baseline)


def test_largest_gap_counts_the_window_edges() -> None:
    """A producer that died 3h before the window closed leaves its gap at the
    END, where a consecutive-pairs scan sees nothing."""
    stamps = [SINCE + timedelta(minutes=i) for i in range(10)]
    gap = _largest_gap_minutes(stamps, since=SINCE, until=NOW)
    assert gap is not None and gap > GAP_MINUTES


def test_largest_gap_of_an_empty_window_is_the_whole_window() -> None:
    assert _largest_gap_minutes([], since=SINCE, until=NOW) == 360.0


# --- volume_shift ----------------------------------------------------------


def test_volume_shift_fires_on_a_halving() -> None:
    assert "volume_shift" in _fired(
        _window(stamps=_evenly(20)),
        _window(stamps=_evenly(80, since=SINCE - WINDOW, until=SINCE)),
    )


def test_volume_shift_refuses_on_ordinary_drift() -> None:
    assert "volume_shift" not in _fired(
        _window(stamps=_evenly(55)),
        _window(stamps=_evenly(50, since=SINCE - WINDOW, until=SINCE)),
    )


def test_volume_shift_refuses_on_a_baseline_too_small_to_ratio() -> None:
    """4 rows -> 12 rows is a 3x ratio and complete noise."""
    assert "volume_shift" not in _fired(
        _window(stamps=_evenly(12)),
        _window(stamps=_evenly(MIN_BASELINE_ROWS - 1, since=SINCE - WINDOW, until=SINCE)),
    )


# --- new_category / lost_category ------------------------------------------


def test_new_category_fires_on_a_label_that_was_not_there_before() -> None:
    fired = _fired(
        _window(stamps=_evenly(40), categories={"event_type": {"seen": 30, "person_left": 10}}),
        _window(
            stamps=_evenly(40, since=SINCE - WINDOW, until=SINCE),
            categories={"event_type": {"seen": 40}},
        ),
    )
    assert "new_category" in fired


def test_new_category_refuses_against_an_empty_baseline() -> None:
    """REGRESSION (first live smoke, 2026-08-25). With no baseline rows every
    category is trivially "new" -- that fired two findings carrying no
    information at all."""
    assert "new_category" not in _fired(
        _window(stamps=_evenly(40), categories={"event_type": {"a": 20, "b": 20}}),
        _window(rows=0),
    )


def test_new_category_refuses_on_a_single_stray_row() -> None:
    assert "new_category" not in _fired(
        _window(
            stamps=_evenly(40),
            categories={"event_type": {"seen": 39, "rare": MIN_CATEGORY_ROWS - 1}},
        ),
        _window(
            stamps=_evenly(40, since=SINCE - WINDOW, until=SINCE),
            categories={"event_type": {"seen": 40}},
        ),
    )


def test_lost_category_fires_when_an_established_label_disappears() -> None:
    assert "lost_category" in _fired(
        _window(stamps=_evenly(40), categories={"domain": {"git": 40}}),
        _window(
            stamps=_evenly(40, since=SINCE - WINDOW, until=SINCE),
            categories={"domain": {"git": 30, "graph": 10}},
        ),
    )


def test_lost_category_defers_to_producer_stalled_on_an_empty_window() -> None:
    fired = _fired(
        _window(rows=0),
        _window(
            stamps=_evenly(40, since=SINCE - WINDOW, until=SINCE),
            categories={"domain": {"git": 40}},
        ),
    )
    assert fired == {"producer_stalled"}


# --- mean_shift ------------------------------------------------------------


def test_mean_shift_fires_past_one_baseline_sigma() -> None:
    baseline_values = [0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0]
    sigma = _stdev(baseline_values)
    assert sigma is not None
    shifted = [1.0 + MEAN_SHIFT_SIGMAS * sigma + 0.05] * 8
    assert "mean_shift" in _fired(
        _window(stamps=_evenly(8), numeric={"score": shifted}),
        _window(stamps=_evenly(8, since=SINCE - WINDOW, until=SINCE), numeric={"score": baseline_values}),
    )


def test_mean_shift_refuses_below_one_baseline_sigma() -> None:
    baseline_values = [0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0]
    sigma = _stdev(baseline_values)
    assert sigma is not None
    nudged = [1.0 + 0.5 * sigma] * 8
    assert "mean_shift" not in _fired(
        _window(stamps=_evenly(8), numeric={"score": nudged}),
        _window(stamps=_evenly(8, since=SINCE - WINDOW, until=SINCE), numeric={"score": baseline_values}),
    )


def test_mean_shift_refuses_when_the_baseline_has_no_spread() -> None:
    """sigma == 0 makes every difference infinitely many sigmas. A constant
    baseline cannot license a distributional claim."""
    assert "mean_shift" not in _fired(
        _window(stamps=_evenly(8), numeric={"score": [5.0] * 8}),
        _window(stamps=_evenly(8, since=SINCE - WINDOW, until=SINCE), numeric={"score": [1.0] * 8}),
    )


def test_stdev_of_one_value_is_none_not_zero() -> None:
    assert _stdev([3.0]) is None
    assert _stdev([]) is None


def test_mean_shift_refuses_on_a_short_baseline() -> None:
    short = [0.0, 2.0, 0.0, 2.0]
    assert len(short) < MIN_BASELINE_ROWS
    assert "mean_shift" not in _fired(
        _window(stamps=_evenly(8), numeric={"score": [50.0] * 8}),
        _window(stamps=_evenly(4, since=SINCE - WINDOW, until=SINCE), numeric={"score": short}),
    )


# --- digest ----------------------------------------------------------------


def test_digest_ignores_values_so_the_same_news_dedups() -> None:
    """Two consecutive windows reporting the same gap on the same producer are
    the same news even though the minute counts differ."""
    from app.self_study_analysis import AnalysisFindingV1

    a = AnalysisFindingV1(rule="observation_gap", detail="x", metric="gap_minutes", recent=140.0)
    b = AnalysisFindingV1(rule="observation_gap", detail="y", metric="gap_minutes", recent=190.0)
    assert finding_digest("vision_events", [a]) == finding_digest("vision_events", [b])


def test_digest_separates_sources_and_rule_sets() -> None:
    from app.self_study_analysis import AnalysisFindingV1

    gap = AnalysisFindingV1(rule="observation_gap", detail="x", metric="gap_minutes")
    vol = AnalysisFindingV1(rule="volume_shift", detail="x", metric="rows")
    assert finding_digest("vision_events", [gap]) != finding_digest("affective_state", [gap])
    assert finding_digest("vision_events", [gap]) != finding_digest("vision_events", [gap, vol])
    assert finding_digest("vision_events", [gap, vol]) == finding_digest("vision_events", [vol, gap])


# --- mutation-survivor closures -------------------------------------------
#
# Every test below exists because a specific semantic mutation to the REAL
# module survived the suite in adversarial review (2026-08-25). Named by what
# they pin, not by the mutation id, but the mutation is stated in each.


def test_producer_stalled_refuses_on_a_baseline_too_thin_to_call_it_a_stall() -> None:
    """REVIEW BLOCKER. Against live `memory_crystallizations` (~4.6 rows/day)
    the unguarded rule fired on 72 of 288 sampled windows with baselines of 1
    or 2 rows, publishing "The producer stopped." about a producer behaving
    exactly as it always does."""
    for baseline_rows in range(1, MIN_BASELINE_ROWS):
        fired = _fired(
            _window(rows=0),
            _window(stamps=_evenly(baseline_rows, since=SINCE - WINDOW, until=SINCE)),
        )
        assert "producer_stalled" not in fired, f"fired on baseline_rows={baseline_rows}"
    assert "producer_stalled" in _fired(
        _window(rows=0),
        _window(stamps=_evenly(MIN_BASELINE_ROWS, since=SINCE - WINDOW, until=SINCE)),
    )


def test_mean_shift_fires_in_the_downward_direction_too() -> None:
    """Mutation M23: dropping abs() survived, because every mean_shift fixture
    shifted UPWARD. A vision pipeline whose mean confidence COLLAPSES is the
    interesting direction and was untested."""
    baseline_values = [0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0]
    sigma = _stdev(baseline_values)
    assert sigma is not None
    dropped = [1.0 - MEAN_SHIFT_SIGMAS * sigma - 0.05] * 8
    findings, _ = evaluate_rules(
        recent=_window(stamps=_evenly(8), numeric={"score": dropped}),
        baseline=_window(
            stamps=_evenly(8, since=SINCE - WINDOW, until=SINCE),
            numeric={"score": baseline_values},
        ),
        recent_since=SINCE,
        recent_until=NOW,
    )
    shift = [f for f in findings if f.rule == "mean_shift"]
    assert shift, "a downward mean shift must fire"
    assert shift[0].recent is not None and shift[0].baseline is not None
    assert shift[0].recent < shift[0].baseline


def test_the_gap_bar_is_the_larger_of_the_floor_and_the_relative_multiple() -> None:
    """Mutations M08/M11: `max` -> `min` and the 2.0 multiple both survived, so
    the relative-bar arithmetic was entirely unpinned. `min` would reinstate
    the fires-every-window bug the live smoke found.

    Hand-computed. Baseline emits every 90 min across its window, edges
    included, so its largest gap is exactly 90 min and the bar is
    max(120, 2*90) = 180 min. The 120-min FLOOR is therefore not the binding
    constraint here, which is the whole point: a 150-min recent gap clears the
    floor and must still refuse."""
    baseline = _window(
        stamps=[SINCE - WINDOW + timedelta(minutes=90 * i) for i in range(1, 4)]
    )
    baseline_gap = _largest_gap_minutes(baseline.timestamps, since=SINCE - WINDOW, until=SINCE)
    assert baseline_gap == 90.0
    assert baseline.rows >= MIN_BASELINE_ROWS - 2  # 3 stamps; padded below
    baseline = _window(
        stamps=sorted(
            baseline.timestamps
            + [SINCE - WINDOW + timedelta(minutes=m) for m in (10, 20)]
        )
    )
    assert baseline.rows >= MIN_BASELINE_ROWS
    assert _largest_gap_minutes(baseline.timestamps, since=SINCE - WINDOW, until=SINCE) == 90.0

    over_floor_under_relative = [SINCE] + [
        SINCE + timedelta(minutes=150 + 30 * i) for i in range(9)
    ]
    assert _largest_gap_minutes(over_floor_under_relative, since=SINCE, until=NOW) == 150.0
    assert "observation_gap" not in _fired(_window(stamps=over_floor_under_relative), baseline)

    clears_both = [SINCE] + [SINCE + timedelta(minutes=200 + 20 * i) for i in range(9)]
    assert _largest_gap_minutes(clears_both, since=SINCE, until=NOW) == 200.0
    assert "observation_gap" in _fired(_window(stamps=clears_both), baseline)


def test_mean_shift_refuses_on_a_recent_window_too_thin_to_average() -> None:
    """Mutation M22: dropping the recent-side guard survived, so mean_shift
    could have fired off a single recent row."""
    baseline_values = [0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0]
    thin = [50.0] * (MIN_BASELINE_ROWS - 1)
    assert "mean_shift" not in _fired(
        _window(stamps=_evenly(len(thin)), numeric={"score": thin}),
        _window(
            stamps=_evenly(8, since=SINCE - WINDOW, until=SINCE),
            numeric={"score": baseline_values},
        ),
    )


def test_lost_category_honours_the_same_row_bar_as_new_category() -> None:
    """Mutations M06/M27: the bar was pinned on new_category but not on
    lost_category. Asymmetric coverage of one shared constant."""
    stray = MIN_CATEGORY_ROWS - 1
    assert "lost_category" not in _fired(
        _window(stamps=_evenly(40), categories={"domain": {"git": 40}}),
        _window(
            stamps=_evenly(40, since=SINCE - WINDOW, until=SINCE),
            categories={"domain": {"git": 40 - stray, "graph": stray}},
        ),
    )
    assert "lost_category" in _fired(
        _window(stamps=_evenly(40), categories={"domain": {"git": 40}}),
        _window(
            stamps=_evenly(40, since=SINCE - WINDOW, until=SINCE),
            categories={
                "domain": {"git": 40 - MIN_CATEGORY_ROWS, "graph": MIN_CATEGORY_ROWS}
            },
        ),
    )


def test_the_volume_ratio_band_is_inclusive_at_both_edges() -> None:
    """Mutations M02/M05/M10: every `>=`/`<=` could be flipped to `>`/`<`
    without a test noticing. Exact-boundary behaviour is a real decision and
    should be recorded, not left to whichever operator happened to type it."""
    baseline = _window(stamps=_evenly(40, since=SINCE - WINDOW, until=SINCE))
    assert "volume_shift" in _fired(_window(stamps=_evenly(80)), baseline)  # exactly 2.00x
    assert "volume_shift" in _fired(_window(stamps=_evenly(20)), baseline)  # exactly 0.50x
    assert "volume_shift" not in _fired(_window(stamps=_evenly(79)), baseline)
    assert "volume_shift" not in _fired(_window(stamps=_evenly(21)), baseline)


def test_mean_shift_is_inclusive_at_exactly_one_sigma() -> None:
    baseline_values = [0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0]
    sigma = _stdev(baseline_values)
    assert sigma is not None
    mean = sum(baseline_values) / len(baseline_values)
    at_bar = [mean + MEAN_SHIFT_SIGMAS * sigma] * 8
    assert "mean_shift" in _fired(
        _window(stamps=_evenly(8), numeric={"score": at_bar}),
        _window(
            stamps=_evenly(8, since=SINCE - WINDOW, until=SINCE),
            numeric={"score": baseline_values},
        ),
    )


def test_the_volume_baseline_bar_is_inclusive() -> None:
    at_bar = _window(stamps=_evenly(MIN_BASELINE_ROWS, since=SINCE - WINDOW, until=SINCE))
    assert "volume_shift" in _fired(_window(stamps=_evenly(MIN_BASELINE_ROWS * 3)), at_bar)


def test_a_degenerate_zero_width_window_has_no_measurable_gap() -> None:
    """Mutation M18: `until <= since` -> `until < since` survived."""
    assert _largest_gap_minutes([], since=SINCE, until=SINCE) is None
    assert _largest_gap_minutes([SINCE], since=SINCE, until=SINCE) is None
    assert _largest_gap_minutes([], since=NOW, until=SINCE) is None


def test_the_category_row_bar_is_pinned_to_its_actual_value() -> None:
    """Mutation M06 survived even the first round of closures, because those
    tests expressed the stray count as `MIN_CATEGORY_ROWS - 1` -- so lowering
    the constant moved the fixture with it. The bar is a real decision about
    how much evidence a one-sentence claim needs; pin the number itself."""
    assert MIN_CATEGORY_ROWS == 3
    baseline = _window(
        stamps=_evenly(40, since=SINCE - WINDOW, until=SINCE),
        categories={"event_type": {"seen": 40}},
    )
    for stray in (1, 2):
        assert "new_category" not in _fired(
            _window(stamps=_evenly(40), categories={"event_type": {"seen": 40, "rare": stray}}),
            baseline,
        ), f"fired on {stray} row(s)"
    assert "new_category" in _fired(
        _window(stamps=_evenly(40), categories={"event_type": {"seen": 40, "rare": 3}}),
        baseline,
    )
