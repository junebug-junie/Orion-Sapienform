"""Self-signal -> outcome association (self-calibration spec section E): the pure reducer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import math

import pytest

import orion.attention.self_signal_outcomes as sso
from orion.schemas.self_signal_outcome import (
    OutcomeHourV1,
    SelfSignalHourV1,
    SelfSignalOutcomeAssociationV1,
    SignalHourV1,
)

T0 = datetime(2026, 10, 1, 0, tzinfo=timezone.utc)
NOW = T0 + timedelta(days=30)
BIO = "node:substrate.biometrics"
EXE = "node:substrate.execution"
CAB = "node:substrate.cabinet"


def _hours(conds: list[str], events: list[int], *, signal: str = BIO, outcome: str = "turn_failed",
           exposure: int | list[int] = 3, history_hours: float = 168.0) -> list[SelfSignalHourV1]:
    """Hour i has the signal condition conds[i] and outcome events[i] (exposure fixed or per hour)."""
    out = []
    for i, (c, e) in enumerate(zip(conds, events)):
        h = T0 + timedelta(hours=i)
        exp = exposure[i] if isinstance(exposure, list) else exposure
        out.append(SelfSignalHourV1(
            hour_id=sso.hour_id(h), hour_start=h, computed_at=h + timedelta(hours=1), history_hours=history_hours,
            signals=[SignalHourV1(signal_id=signal, condition=c)],  # type: ignore[arg-type]
            outcomes=[OutcomeHourV1(outcome=outcome, exposure=exp, events=e)],  # type: ignore[arg-type]
        ))
    return out


def _pattern(n_high: int, n_normal: int, hit_after_high: float, hit_after_normal: float):
    """Alternate (high|normal, outcome-hour) pairs so hour H's condition maps to H+1's outcome."""
    conds, events = [], []
    for i in range(n_high):
        conds += ["high", "unknown"]
        events += [0, 1 if i < round(hit_after_high * n_high) else 0]
    for i in range(n_normal):
        conds += ["normal", "unknown"]
        events += [0, 1 if i < round(hit_after_normal * n_normal) else 0]
    return conds, events


# --- the hour rule's anchor ------------------------------------------------------------------


def test_high_hour_anchor_is_the_episode_dwell_and_orienting_window() -> None:
    from orion.attention.world_first import EVENT_ORIENTING_WINDOW_SEC
    from orion.metacog.self_signal_episode import (
        DEFAULT_OPEN_DWELL_SEC,
        DEFAULT_OPEN_WINDOW_SEC,
        DEFAULT_POLL_SEC,
    )

    assert sso.HIGH_HOUR_HOT_SEC == EVENT_ORIENTING_WINDOW_SEC == DEFAULT_OPEN_DWELL_SEC
    assert sso.OPEN_WINDOW_SEC == DEFAULT_OPEN_WINDOW_SEC
    assert sso.POLL_SEC == DEFAULT_POLL_SEC


# --- classify: absent is never calm ------------------------------------------------------------


def _week(value_fn, *, start: datetime, end: datetime, step_s: int = 35) -> list[tuple[datetime, float]]:
    out, t, k = [], start, 0
    while t < end:
        out.append((t, float(value_fn(k))))
        t += timedelta(seconds=step_s)
        k += 1
    return out


def test_no_rows_in_an_hour_is_unknown_not_normal() -> None:
    h = T0 + timedelta(days=8)
    hist = _week(lambda k: 0.01 + (k % 50) / 1000.0, start=T0, end=h - timedelta(hours=2))
    got = sso.classify_signal_hours(BIO, hist, [h])
    assert got[h].condition == "unknown"
    assert got[h].judged_sec == 0.0


def test_too_little_history_is_unknown() -> None:
    h = T0 + timedelta(hours=1)
    hist = _week(lambda k: 0.5, start=h, end=h + timedelta(hours=1), step_s=60)  # 60 readings < 200
    got = sso.classify_signal_hours(BIO, hist, [h])
    assert got[h].condition == "unknown"
    assert got[h].n_readings == 60 and got[h].n_judged == 0


def test_placeholder_kind_is_never_classified() -> None:
    h = T0 + timedelta(days=8)
    hist = _week(lambda k: 0.65, start=T0, end=h + timedelta(hours=1), step_s=600)
    got = sso.classify_signal_hours("node:substrate.harness_closure", hist, [h])
    assert got[h].condition == "unknown"


def test_a_sustained_top_of_week_hour_is_high_a_typical_hour_normal() -> None:
    hot_h = T0 + timedelta(days=8)
    calm_h = hot_h + timedelta(hours=1)
    hist = _week(lambda k: 0.01 + (k % 50) / 1000.0, start=T0, end=hot_h)
    hist += _week(lambda k: 0.9, start=hot_h, end=hot_h + timedelta(minutes=20))  # 20 min at the top
    hist += _week(lambda k: 0.01 + (k % 50) / 1000.0, start=hot_h + timedelta(minutes=20), end=calm_h + timedelta(hours=1))
    got = sso.classify_signal_hours(BIO, hist, [hot_h, calm_h])
    assert got[hot_h].condition == "high"
    assert got[hot_h].hot_sec > sso.HIGH_HOUR_HOT_SEC
    assert got[calm_h].condition == "normal"
    assert got[calm_h].judged_sec > 3000


def test_one_hot_reading_is_not_a_high_hour() -> None:
    h = T0 + timedelta(days=8)
    hist = _week(lambda k: 0.01 + (k % 50) / 1000.0, start=T0, end=h + timedelta(hours=1))
    hist.append((h + timedelta(minutes=30, seconds=1), 0.95))  # one spike, next reading 35 s later
    got = sso.classify_signal_hours(BIO, sorted(hist), [h])
    assert got[h].n_hot >= 1
    assert got[h].condition == "normal"


def test_scattered_top_decile_readings_are_a_normal_hour() -> None:
    """~10% of a per-tick signal's readings are 'high' by construction, spread through every
    typical hour (~360 s of hot time): that is the signal's normal, not a high hour."""
    h = T0 + timedelta(days=8)
    hist = _week(lambda k: 0.01 + (k * 7919 % 1000) / 10000.0, start=T0, end=h + timedelta(hours=1))
    got = sso.classify_signal_hours(BIO, hist, [h])
    assert got[h].hot_sec >= sso.HIGH_HOUR_HOT_SEC  # plenty of scattered hot time...
    assert got[h].condition == "normal"  # ...but never sustained


def test_trigger_hour_is_high_only_when_it_turned_on() -> None:
    h = T0 + timedelta(days=2)
    rows = [(h + timedelta(seconds=30 * k), 0.0) for k in range(120)]
    rows[10] = (rows[10][0], 0.7)
    rows[11] = (rows[11][0], 0.7)
    calm = h + timedelta(hours=1)
    rows += [(calm + timedelta(seconds=30 * k), 0.0) for k in range(120)]
    got = sso.classify_signal_hours(CAB, rows, [h, calm, calm + timedelta(hours=1)])
    assert got[h].condition == "high" and got[h].n_hot == 1  # one onset, two "on" rows
    assert got[calm].condition == "normal"
    assert got[calm + timedelta(hours=1)].condition == "unknown"


# --- outcomes ----------------------------------------------------------------------------------


def test_turn_outcomes_count_failed_and_repaired_with_exposure() -> None:
    h = T0
    rows = [(h, "failed", False), (h + timedelta(minutes=5), "completed", True), (h, None, None),
            (h + timedelta(minutes=9), "partial", None)]
    got = {o.outcome: o for o in sso.turn_outcome_hours(rows)[h]}
    assert (got["turn_failed"].exposure, got["turn_failed"].events) == (3, 1)
    assert (got["response_repair"].exposure, got["response_repair"].events) == (2, 1)


def test_gpu_pool_waits_are_not_rpc_timeouts() -> None:
    h = T0
    rows = [(h, "gpu_pool:chat#gpu_pool_wait", 1, 9), (h, "orion:exec:request:LLMGatewayService", 10, 2)]
    o = sso.rpc_outcome_hours(rows)[h]
    assert (o.exposure, o.events) == (12, 2)


# --- the interval ------------------------------------------------------------------------------


def test_katz_interval_excludes_one_only_with_a_real_gap() -> None:
    strong = sso.katz_rate_ratio(20, 30, 10, 100)  # 67% vs 10%
    assert strong.ratio == pytest.approx(6.6667, rel=1e-3)
    assert strong.low > 1.0
    weak = sso.katz_rate_ratio(4, 30, 10, 100)  # 13% vs 10%
    assert weak.low < 1.0 < weak.high


def test_katz_only_corrects_zero_cells() -> None:
    full = sso.katz_rate_ratio(30, 30, 10, 100)  # a == n1 is not a zero cell
    assert full.ratio == pytest.approx(10.0)
    assert full.low == pytest.approx(math.exp(math.log(10) - sso.Z_95 * math.sqrt(1 / 30 - 1 / 30 + 1 / 10 - 1 / 100)))


def test_katz_zero_cells_keep_a_finite_interval_and_no_fake_ratio() -> None:
    iv = sso.katz_rate_ratio(5, 30, 0, 100)
    assert iv.ratio is None and iv.low is not None and iv.high is not None
    assert sso.katz_rate_ratio(0, 30, 0, 100).low is None


# --- association --------------------------------------------------------------------------------


def test_min_n_suppresses_the_clause_even_for_a_big_ratio() -> None:
    conds, events = _pattern(n_high=10, n_normal=60, hit_after_high=0.9, hit_after_normal=0.1)
    a = sso.associate(_hours(conds, events), BIO, "turn_failed", computed_at=NOW, min_hours=24)
    assert a.n_high == 10 and a.verdict == "below_min_n" and a.clause == ""


def test_interval_excluding_one_renders_a_cited_clause() -> None:
    conds, events = _pattern(n_high=30, n_normal=60, hit_after_high=0.6, hit_after_normal=0.1)
    a = sso.associate(_hours(conds, events), BIO, "turn_failed", computed_at=NOW, min_hours=24)
    assert a.verdict == "tends_more"
    assert (a.n_high, a.events_high, a.n_normal, a.events_normal) == (30, 18, 60, 6)
    assert a.ci_low > 1.0
    assert a.clause == (
        "when this is high, my next hour had a failed turn 6.0x as often at equal activity (60% of 30 high hours "
        f"vs 10% of 60 normal) [assoc:{a.association_id}]"
    )


def test_interval_containing_one_says_nothing() -> None:
    conds, events = _pattern(n_high=30, n_normal=60, hit_after_high=0.15, hit_after_normal=0.1)
    a = sso.associate(_hours(conds, events), BIO, "turn_failed", computed_at=NOW, min_hours=24)
    assert a.verdict == "no_clear_difference" and a.clause == ""


def test_less_often_is_a_verdict_too() -> None:
    conds, events = _pattern(n_high=40, n_normal=60, hit_after_high=0.05, hit_after_normal=0.6)
    a = sso.associate(_hours(conds, events), BIO, "turn_failed", computed_at=NOW, min_hours=24)
    assert a.verdict == "tends_less" and "0.1x as often" in a.clause


@pytest.mark.parametrize("pair", sorted(sso.CIRCULAR_PAIRS))
def test_circular_pairs_never_render_however_strong(pair) -> None:
    signal, outcome = pair
    conds, events = _pattern(n_high=40, n_normal=60, hit_after_high=0.9, hit_after_normal=0.05)
    a = sso.associate(_hours(conds, events, signal=signal, outcome=outcome), signal, outcome,
                      computed_at=NOW, min_hours=24)
    assert a.verdict == "excluded_circular" and a.clause == ""
    assert a.excluded_reason == sso.CIRCULAR_PAIRS[pair]
    assert a.ci_low > 1.0  # the numbers are still recorded, for inspection


def test_circular_pairs_cite_real_code() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for reason in sso.CIRCULAR_PAIRS.values():
        path = reason.split("(")[1].split()[0]
        assert (root / path).exists(), path


def test_unknown_hours_are_not_counted_as_normal() -> None:
    conds, events = _pattern(n_high=30, n_normal=30, hit_after_high=0.5, hit_after_normal=0.1)
    rows = _hours(conds + ["unknown", "unknown"] * 50, events + [0, 0] * 50)
    a = sso.associate(rows, BIO, "turn_failed", computed_at=NOW)
    assert a.n_normal == 30  # the 50 unknown hours added nothing


def test_next_hour_without_exposure_is_skipped() -> None:
    conds, events = _pattern(n_high=30, n_normal=30, hit_after_high=0.5, hit_after_normal=0.1)
    rows = _hours(conds, events, exposure=0)
    a = sso.associate(rows, BIO, "turn_failed", computed_at=NOW)
    assert a.n_high == 0 and a.n_normal == 0 and a.verdict == "below_min_n"


def test_associate_caps_the_hours_it_folds() -> None:
    conds, events = _pattern(n_high=sso.MAX_HOURS, n_normal=sso.MAX_HOURS, hit_after_high=0.5,
                             hit_after_normal=0.1)
    rows = _hours(conds, events)
    a = sso.associate(rows, BIO, "turn_failed", computed_at=NOW)
    assert a.n_high + a.n_normal <= sso.MAX_HOURS


def test_index_by_signal_takes_the_newest_row_and_the_strongest_outcome() -> None:
    def row(outcome, verdict, low, high, at, clause="c"):
        return SelfSignalOutcomeAssociationV1(
            association_id=f"ssa-{outcome}-{at.hour}", computed_at=at, signal_id=BIO, outcome=outcome,
            verdict=verdict, ci_low=low, ci_high=high, rate_high=0.5, rate_normal=0.1, ratio=5.0,
            clause=clause if verdict.startswith("tends") else "",
        )
    old = NOW - timedelta(hours=2)
    rows = [
        row("turn_failed", "tends_more", 3.0, 9.0, old),  # superseded by the newer row below
        row("turn_failed", "no_clear_difference", 0.8, 2.0, NOW),
        row("rpc_timeout", "tends_more", 1.2, 4.0, NOW),
        row("response_repair", "tends_more", 1.8, 4.0, NOW),
    ]
    idx = sso.index_by_signal(rows)
    assert idx[BIO].outcome == "response_repair"


def test_hour_row_caps_signals() -> None:
    sigs = [SignalHourV1(signal_id=f"node:substrate.s{i:02d}", condition="normal") for i in range(80)]
    row = sso.build_hour_row(T0, signals=sigs, outcomes=[], computed_at=NOW, history_hours=10)
    assert len(row.signals) == 32


# --- review fixes: backfill floor, multiplicity, exposure, serial correlation, wording -----------


def test_hours_judged_on_a_short_history_never_count() -> None:
    conds, events = _pattern(n_high=30, n_normal=60, hit_after_high=0.6, hit_after_normal=0.1)
    short = _hours(conds, events, history_hours=40.0)
    a = sso.associate(short, BIO, "turn_failed", computed_at=NOW)
    assert (a.n_high, a.n_normal, a.verdict) == (0, 0, "below_min_n")
    assert a.min_history_hours == sso.MIN_HISTORY_HOURS == 144.0


def test_bonferroni_widens_the_interval_for_the_whole_family() -> None:
    conds, events = _pattern(n_high=30, n_normal=60, hit_after_high=0.33, hit_after_normal=0.15)
    rows = _hours(conds, events)
    one = sso.associate(rows, BIO, "turn_failed", computed_at=NOW, n_comparisons=1)
    fam = sso.associate(rows, BIO, "turn_failed", computed_at=NOW, n_comparisons=17)
    assert fam.ci_z == pytest.approx(sso.family_z(17)) and fam.ci_z > 2.9
    assert fam.ci_low < one.ci_low and fam.ci_high > one.ci_high
    assert one.verdict == "tends_more" and fam.verdict == "no_clear_difference"


def test_associate_all_counts_every_renderable_pair_as_one_family() -> None:
    conds, events = _pattern(n_high=30, n_normal=60, hit_after_high=0.6, hit_after_normal=0.1)
    out = sso.associate_all(_hours(conds, events, signal=EXE), computed_at=NOW)
    # execution x turn_failed / response_repair are circular: one renderable pair (rpc_timeout).
    assert {a.n_comparisons for a in out} == {1}


def test_busy_next_hours_alone_do_not_make_an_association() -> None:
    """High hours are followed by busy hours (20 turns), normal ones by quiet hours (1 turn); at
    equal activity the failure share is identical, so the adjusted ratio is ~1 (no clause)."""
    conds, events, exposure = [], [], []
    for i in range(40):  # high -> busy next hour, 50% have a failure
        conds += ["high", "unknown"]
        events += [0, 1 if i % 2 else 0]
        exposure += [1, 20]
    for i in range(40):  # normal -> busy next hour, same 50%
        conds += ["normal", "unknown"]
        events += [0, 1 if i % 2 else 0]
        exposure += [1, 20]
    for i in range(80):  # normal -> quiet next hour, 10%
        conds += ["normal", "unknown"]
        events += [0, 1 if i % 10 == 0 else 0]
        exposure += [1, 1]
    rows = _hours(conds, events, exposure=exposure)
    a = sso.associate(rows, BIO, "turn_failed", computed_at=NOW)
    crude = (a.events_high / a.n_high) / (a.events_normal / a.n_normal)
    assert crude > 1.6  # the raw rates alone would have said "tends to"
    assert a.ratio == pytest.approx(1.0) and a.n_strata == 2
    assert a.verdict == "no_clear_difference" and a.clause == ""


def test_clustered_hours_inflate_the_interval() -> None:
    conds = (["high"] * 6 + ["normal"] * 6) * 8
    events = ([1] * 6 + [0] * 6) * 8
    events = [0] + events[:-1]  # hour H's condition -> hour H+1's outcome, in runs
    a = sso.associate(_hours(conds, events), BIO, "turn_failed", computed_at=NOW, min_hours=1)
    assert a.se_inflation > 1.5


def test_never_is_said_plainly_not_as_zero_x() -> None:
    conds, events = _pattern(n_high=40, n_normal=60, hit_after_high=0.0, hit_after_normal=0.6)
    a = sso.associate(_hours(conds, events), BIO, "turn_failed", computed_at=NOW)
    assert a.verdict == "tends_less"
    assert "my next hour never had a failed turn (0% of 40 high hours" in a.clause
    assert "0.0x" not in a.clause


def test_one_event_on_the_poll_grid_is_not_a_high_hour() -> None:
    """The seam drops an event at age >= 300 s: one hot event must not fill 11 polls (330 s)."""
    chat = "node:substrate.chat"
    h = T0 + timedelta(days=8)
    hist = [(T0 + timedelta(minutes=15 * k), 0.0 if k % 3 else 0.1 + (k % 9) / 10.0)
            for k in range(int(8 * 24 * 4))]
    hist.append((h + timedelta(minutes=10), 1.0))  # exactly on the 30 s grid
    got = sso.classify_signal_hours(chat, sorted(hist), [h])
    assert got[h].hot_sec <= sso.HIGH_HOUR_HOT_SEC
    assert got[h].condition == "normal"


def test_a_trigger_still_on_from_the_previous_hour_is_high() -> None:
    h = T0 + timedelta(days=2)
    rows = [(h - timedelta(minutes=2), 0.0), (h - timedelta(minutes=1), 0.7)]
    rows += [(h + timedelta(seconds=30 * k), 0.7 if k < 4 else 0.0) for k in range(120)]
    got = sso.classify_signal_hours(CAB, rows, [h])
    assert got[h].condition == "high" and got[h].n_hot == 0  # on, but no new onset this hour
