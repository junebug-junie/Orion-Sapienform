"""Attention as the want competition, patch 2: event salience, event rank, self-grade modes (pure)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.physiology.appetite import (
    EVENT_HISTORY_MIN,
    ActRecord,
    build_family_appetite,
    rank_value,
)
from orion.physiology.choice import ReservationBook, build_context, decide
from orion.physiology.event_wants import (
    MIN_SOURCE_HISTORY,
    event_want,
    event_want_id,
    saliences_over_own_history,
    source_salience,
)
from orion.physiology.families import parse_families
from orion.physiology.weights import evidence_for, reliability_grades, shrunk_target
from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R
from orion.schemas.physiology import PhysiologyStateV1
from orion.schemas.physiology_want import WantV1

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
FAMS = parse_families({"version": 1, "families": {
    "reading": {"tier": 2, "deferrable_window_s": 10800, "lanes": []},
    "camera": {"tier": 2, "deferrable_window_s": 86400, "lanes": []},
}})


# --- salience ---------------------------------------------------------------------------------------

def test_a_typical_event_reads_zero_the_rest_point():
    history = [0] * 80 + [1] * 15 + [3] * 5
    assert source_salience(0, history) == 0.0          # no bigger than the usual: not "a little unusual"
    assert source_salience(1, history) == pytest.approx(0.80)
    assert source_salience(4, history) == 1.0


def test_too_little_history_is_unknown_not_calm():
    assert source_salience(5, [0] * (MIN_SOURCE_HISTORY - 1)) is None
    assert source_salience(float("nan"), [0] * 50) is None


def test_each_event_ranks_only_against_the_30_days_before_it():
    t0 = NOW - timedelta(days=60)
    # 40 old big events (outside the window of the last one), then 30 small ones, then one medium.
    events = [(t0 + timedelta(hours=k), 9.0) for k in range(40)]
    events += [(NOW - timedelta(days=10) + timedelta(hours=k), 0.0) for k in range(30)]
    events.append((NOW, 1.0))
    sal = saliences_over_own_history(events)
    assert sal[-1] == 1.0          # the 9s are > 30 days older, so they are not its history
    assert sal[0] is None          # the first event has no history at all


def test_the_event_want_carries_its_event_and_a_stable_id():
    w = event_want(source="reading", key="seed1", family="reading", producer="p", verb="look", tier=2,
                   event_ref="world_pulse_read_seed:seed1", salience=0.9, opened_at=NOW)
    assert w.want_id == event_want_id("reading", "seed1") == w.correlation_id
    assert w.event_ref == "world_pulse_read_seed:seed1" and w.salience == 0.9
    assert event_want_id("reading", "seed1") != event_want_id("camera_ask", "seed1")
    assert WantV1.model_validate_json(w.model_dump_json()) == w


# --- event rank -------------------------------------------------------------------------------------

def _acts(n=40):
    # A read every 30 min, each closing an act; ungraded (no outcomes).
    return [ActRecord(act_id=f"a{k}", family="reading", opened_at=NOW - timedelta(minutes=30 * k + 20),
                      closed_at=NOW - timedelta(minutes=30 * k)) for k in range(1, n)]


def test_an_event_ranks_against_past_events_not_hunger_history():
    acts = _acts()
    hunger_only = build_family_appetite(FAMS.get("reading"), acts, NOW)
    past = [0.05] * 30 + [0.9] * 10          # most past events were typical, a few salient
    app = build_family_appetite(FAMS.get("reading"), acts, NOW, event_intensities=past)
    old, old_basis, _ = rank_value(hunger_only, 0.9, None)
    new, new_basis, _ = rank_value(app, 0.9, None, event=True)
    assert old_basis == "ungraded" and old == 1.0           # against hunger: always the top
    assert new_basis == "event:ungraded" and new == pytest.approx((30 + 5) / 40)
    # A want without an event is ranked exactly as before.
    assert rank_value(app, 0.9, None) == rank_value(hunger_only, 0.9, None)


def test_under_the_minimum_event_history_an_event_ranks_as_before():
    app = build_family_appetite(FAMS.get("reading"), _acts(), NOW, event_intensities=[0.5] * (EVENT_HISTORY_MIN - 1))
    rank, basis, _ = rank_value(app, 0.3, None, event=True)
    assert basis == "event_cold:ungraded" and rank == rank_value(app, 0.3, None)[0]


def _state():
    return PhysiologyStateV1(tick_id="t", generated_at=NOW, constraints=[
        R(constraint="heat", unit="c", level=24.0, budget=30.0, budget_kind="soft_line",
          price={"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": ""})])


def test_decide_reads_the_markets_event_history():
    acts = _acts()
    ctx = build_context(FAMS, acts, [], _state(), NOW, event_history={"reading": [0.05] * 30 + [0.9] * 10})
    ev = WantV1(want_id="w1", family="reading", producer="p", verb="look", tier=2, opened_at=NOW,
                event_ref="world_pulse_read_seed:s", salience=0.9)
    d = decide(ev, _state(), ctx, ReservationBook(), NOW)
    assert d.value_basis == "event:ungraded" and d.value_rank == pytest.approx(0.875) and d.outcome == "funded"
    clock = WantV1(want_id="w2", family="reading", producer="p", verb="look", tier=2, opened_at=NOW)
    d2 = decide(clock, _state(), ctx, ReservationBook(), NOW)
    assert d2.value_basis == "ungraded"


def test_a_typical_event_on_a_family_that_just_acted_is_sated_without_event_history():
    acts = [ActRecord(act_id="a", family="camera", opened_at=NOW, closed_at=NOW)]
    ctx = build_context(FAMS, acts, [], _state(), NOW)
    ev = WantV1(want_id="w", family="camera", producer="p", verb="ask", tier=2, opened_at=NOW,
                event_ref="orion_ask:x", salience=0.0)
    d = decide(ev, _state(), ctx, ReservationBook(), NOW)
    assert d.value_basis == "event_cold:no_history" and d.outcome == "deferred" and d.reason == "sated"


# --- self-grade modes (source (d)) ------------------------------------------------------------------

SINCE, UNTIL = NOW - timedelta(days=30), NOW


def _closes():
    out = []
    for k in range(12):   # producer self-grades, all upheld
        out.append(ActRecord(act_id=f"p{k}", family="reading", opened_at=NOW - timedelta(days=1, hours=k),
                             closed_at=NOW - timedelta(days=1, hours=k), outcome_value=1.0,
                             outcome_source="producer", outcome_note="ok"))
    for k in range(4):    # the world's grade: the deadline passed with no artifact
        out.append(ActRecord(act_id=f"e{k}", family="reading", opened_at=NOW - timedelta(days=2, hours=k),
                             closed_at=NOW - timedelta(days=2, hours=k), outcome_value=0.0,
                             outcome_source="expired", outcome_note="expired"))
    return out


def test_all_is_todays_behaviour():
    ev = evidence_for("reading", _closes(), [], since=SINCE, until=UNTIL, reliability=True)
    assert ev.reliability_rate == pytest.approx(12 / 16) and ev.n == 16
    assert [g[:2] for g in reliability_grades("reading", _closes(), since=SINCE, until=UNTIL)][:1] == [("p0", True)]


def test_world_only_keeps_only_the_worlds_grades():
    ev = evidence_for("reading", _closes(), [], since=SINCE, until=UNTIL, reliability=True, self_grades="world_only")
    assert ev.reliability_rate == 0.0 and ev.reliability_n == 4 and ev.n == 4
    assert shrunk_target(1.0, ev) == (1.0, "prior")     # 4 acts is under MIN_EVIDENCED


def test_half_counts_a_self_grade_as_half_a_grade():
    ev = evidence_for("reading", _closes(), [], since=SINCE, until=UNTIL, reliability=True, self_grades="half")
    assert ev.reliability_rate == pytest.approx(6 / (6 + 4)) and ev.n == pytest.approx(4 + 12 * 0.5)


def test_an_unknown_mode_refuses():
    with pytest.raises(ValueError):
        reliability_grades("reading", _closes(), since=SINCE, until=UNTIL, self_grades="sometimes")
